import hashlib
import json
from pathlib import Path

import pytest

from runtime.checkpoint_identity import (
    RAW_FAST_TIER_BINDING_NAME,
    REPLACEMENT_MARKER_NAME,
    REPLACEMENT_RECEIPT_NAME,
    checkpoint_identity,
    checkpoint_release_revision,
    raw_fast_tier_binding,
    refuse_incomplete_checkpoint,
    validate_raw_fast_tier_binding,
)
from runtime.hf_checkpoint_overlay import _atomic_json
from runtime.hf_checkpoint_replace import (
    apply_plan,
    audit,
    build_plan,
    finalize,
)


def _record(path: str, payload: bytes):
    return {
        "path": path,
        "size": len(payload),
        "hash_kind": "sha256",
        "hash": hashlib.sha256(payload).hexdigest(),
    }


def _checkpoint(root: Path, shard: bytes = b"base") -> dict[str, bytes]:
    root.mkdir()
    payloads = {
        "config.json": b'{"model_type":"test"}',
        "model.safetensors.index.json": json.dumps({
            "weight_map": {"model.weight": "model.safetensors"},
        }, sort_keys=True).encode(),
        "model.safetensors": shard,
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return payloads


def _replacement_plan(root: Path, base: dict[str, bytes], candidate_shard: bytes):
    candidate = dict(base)
    candidate["model.safetensors"] = candidate_shard
    return build_plan(
        base_repo="org/base",
        base_revision="a" * 40,
        base_records={name: _record(name, data) for name, data in base.items()},
        candidate_repo="org/candidate",
        candidate_revision="b" * 40,
        candidate_records={
            name: _record(name, data) for name, data in candidate.items()},
        model_dir=root,
    )


def test_replacement_is_resumable_and_loader_marker_is_fail_closed(
    tmp_path, monkeypatch,
):
    root = tmp_path / "model"
    base = _checkpoint(root)
    plan = _replacement_plan(root, base, b"candidate")
    _atomic_json(root / REPLACEMENT_MARKER_NAME, plan)
    with pytest.raises(RuntimeError, match="replacement is incomplete"):
        refuse_incomplete_checkpoint(root)
    assert audit(root)["states"] == {
        "base": 1, "candidate": 0, "missing_new": 0, "invalid": 0,
    }

    def fake_download(*, repo, revision, record, staging):
        assert (repo, revision) == ("org/candidate", "b" * 40)
        target = staging / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"candidate")
        return target

    monkeypatch.setattr(
        "runtime.hf_checkpoint_replace._download_one", fake_download)
    result = apply_plan(root, reserve_bytes=0)
    assert result["states"]["candidate"] == 1
    # Crash/restart classification does not redownload an already committed file.
    assert apply_plan(root, reserve_bytes=0)["applied_files"] == 0
    receipt = finalize(root)
    assert receipt["candidate"]["revision"] == "b" * 40
    assert not (root / REPLACEMENT_MARKER_NAME).exists()
    assert checkpoint_release_revision(root) == "b" * 40
    assert (root / REPLACEMENT_RECEIPT_NAME).is_file()


def test_replacement_refuses_unrecognized_local_bytes(tmp_path):
    root = tmp_path / "model"
    base = _checkpoint(root, shard=b"corrupt")
    expected_base = dict(base)
    expected_base["model.safetensors"] = b"base"
    plan = _replacement_plan(root, expected_base, b"candidate")
    _atomic_json(root / REPLACEMENT_MARKER_NAME, plan)
    assert audit(root)["states"]["invalid"] == 1
    with pytest.raises(ValueError, match="unrecognized local bytes"):
        apply_plan(root, reserve_bytes=0)


def test_same_layout_replacement_requires_identical_serving_metadata(tmp_path):
    base = _checkpoint(tmp_path / "model")
    candidate = dict(base)
    candidate["config.json"] = b'{"model_type":"other"}'
    with pytest.raises(ValueError, match="identical config.json"):
        build_plan(
            base_repo="org/base", base_revision="a" * 40,
            base_records={name: _record(name, data) for name, data in base.items()},
            candidate_repo="org/candidate", candidate_revision="b" * 40,
            candidate_records={
                name: _record(name, data) for name, data in candidate.items()},
            model_dir=tmp_path / "model",
        )


def test_raw_fast_tier_binding_detects_checkpoint_or_manifest_change(tmp_path):
    root = tmp_path / "model"
    _checkpoint(root)
    receipt = {
        "candidate": {"repo": "org/candidate", "revision": "b" * 40},
    }
    (root / REPLACEMENT_RECEIPT_NAME).write_text(json.dumps(receipt))
    tier = tmp_path / "tier"
    tier.mkdir()
    manifest = b'{"model.weight":{"file":"layer.bin"}}'
    binding = raw_fast_tier_binding(root, manifest)
    (tier / RAW_FAST_TIER_BINDING_NAME).write_text(json.dumps(binding))
    assert validate_raw_fast_tier_binding(root, tier, manifest) == binding
    assert checkpoint_identity(root)["release_revision"] == "b" * 40

    with pytest.raises(ValueError, match="source identity mismatch"):
        validate_raw_fast_tier_binding(root, tier, manifest + b"\n")
    (root / "model.safetensors").touch()
    with pytest.raises(ValueError, match="source identity mismatch"):
        validate_raw_fast_tier_binding(root, tier, manifest)

