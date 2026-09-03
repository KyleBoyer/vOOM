import hashlib
import json
from pathlib import Path

import pytest

from runtime.checkpoint_identity import (
    REPLACEMENT_MARKER_NAME,
    REPLACEMENT_RECEIPT_NAME,
    checkpoint_release_revision,
    refuse_incomplete_checkpoint,
)
from runtime.hf_checkpoint_overlay import _atomic_json
from runtime.hf_checkpoint_relayout_replace import (
    apply_plan,
    audit,
    build_plan,
    finalize,
)


def _record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size": len(payload),
        "hash_kind": "sha256",
        "hash": hashlib.sha256(payload).hexdigest(),
    }


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def _fixture(root: Path):
    config = b'{"model_type":"test"}'
    base_index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
            "c": "model-00002-of-00002.safetensors",
            "d": "model-00002-of-00002.safetensors",
        },
    }
    candidate_index = {
        "weight_map": {
            "a": "model-00001-of-00003.safetensors",
            "b": "model-00002-of-00003.safetensors",
            "c": "model-00002-of-00003.safetensors",
            "d": "model-00003-of-00003.safetensors",
        },
    }
    base_payloads = {
        "config.json": config,
        "model.safetensors.index.json": _json(base_index),
        "model-00001-of-00002.safetensors": b"AAA",
        "model-00002-of-00002.safetensors": b"BBBB",
        "tokenizer.json": b"upstream-tokenizer",
    }
    candidate_payloads = {
        "config.json": config,
        "model.safetensors.index.json": _json(candidate_index),
        "model-00001-of-00003.safetensors": b"C",
        "model-00002-of-00003.safetensors": b"DDDDD",
        "model-00003-of-00003.safetensors": b"E",
        "generation_config.json": b'{"temperature":1}',
    }
    root.mkdir()
    for name, payload in base_payloads.items():
        (root / name).write_bytes(payload)
    base_records = {
        name: _record(name, payload) for name, payload in base_payloads.items()}
    candidate_records = {
        name: _record(name, payload)
        for name, payload in candidate_payloads.items()}
    plan = build_plan(
        base_repo="org/base", base_revision="a" * 40,
        base_records=base_records, base_index=base_index,
        candidate_repo="org/candidate", candidate_revision="b" * 40,
        candidate_records=candidate_records,
        candidate_index=candidate_index, model_dir=root,
    )
    return plan, base_payloads, candidate_payloads


def test_relayout_plan_is_coverage_ordered_and_preserves_support_files(tmp_path):
    plan, _, _ = _fixture(tmp_path / "model")
    assert plan["tensors"] == {"count": 4, "name_sets_equal": True}
    assert plan["shards"]["download_order"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    assert plan["shards"]["candidate_to_base"][
        "model-00002-of-00003.safetensors"] == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
    assert plan["shards"]["peak_extra_bytes"] == 11
    assert [record["path"] for record in plan["metadata"][
        "preserve_base_only"]] == ["tokenizer.json"]


def test_relayout_is_resumable_atomic_and_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "model"
    plan, _, candidate_payloads = _fixture(root)
    _atomic_json(root / REPLACEMENT_MARKER_NAME, plan)
    with pytest.raises(RuntimeError, match="replacement is incomplete"):
        refuse_incomplete_checkpoint(root)

    def fake_download(*, repo, revision, record, staging):
        assert (repo, revision) == ("org/candidate", "b" * 40)
        target = staging / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(candidate_payloads[record["path"]])
        return target

    monkeypatch.setattr(
        "runtime.hf_checkpoint_relayout_replace._download_one", fake_download)
    first = apply_plan(root, reserve_bytes=0, max_files=1)
    assert first["candidate_shards"]["candidate"] == 1
    assert first["released_base_files"] == 0
    second = apply_plan(root, reserve_bytes=0, max_files=1)
    assert second["candidate_shards"]["candidate"] == 2
    assert second["released_base_files"] == 1
    assert not (root / "model-00001-of-00002.safetensors").exists()

    result = apply_plan(root, reserve_bytes=0)
    assert result["complete"]
    assert not (root / "model-00002-of-00002.safetensors").exists()
    assert (root / "tokenizer.json").read_bytes() == b"upstream-tokenizer"
    assert json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"]["d"] == "model-00003-of-00003.safetensors"
    receipt = finalize(root)
    assert receipt["safetensor_shards"] == 3
    assert receipt["preserved_base_only_files"] == ["tokenizer.json"]
    assert not (root / REPLACEMENT_MARKER_NAME).exists()
    assert (root / REPLACEMENT_RECEIPT_NAME).is_file()
    assert checkpoint_release_revision(root) == "b" * 40


def test_relayout_detects_a_crash_created_coverage_hole(tmp_path, monkeypatch):
    root = tmp_path / "model"
    plan, _, candidate_payloads = _fixture(root)
    _atomic_json(root / REPLACEMENT_MARKER_NAME, plan)

    def fake_download(*, repo, revision, record, staging):
        target = staging / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(candidate_payloads[record["path"]])
        return target

    monkeypatch.setattr(
        "runtime.hf_checkpoint_relayout_replace._download_one", fake_download)
    apply_plan(root, reserve_bytes=0, max_files=1)
    (root / "model-00001-of-00002.safetensors").unlink()
    report = audit(root)
    assert report["coverage_holes"] == [
        "model-00001-of-00002.safetensors"]
    with pytest.raises(RuntimeError, match="uncovered deleted base shards"):
        apply_plan(root, reserve_bytes=0)


def test_relayout_rejects_tensor_or_config_changes(tmp_path):
    root = tmp_path / "model"
    plan, base_payloads, candidate_payloads = _fixture(root)
    del plan
    base_index = json.loads(base_payloads[
        "model.safetensors.index.json"])
    candidate_index = json.loads(candidate_payloads[
        "model.safetensors.index.json"])
    base_records = {
        name: _record(name, payload) for name, payload in base_payloads.items()}
    candidate_records = {
        name: _record(name, payload)
        for name, payload in candidate_payloads.items()}

    bad_index = json.loads(json.dumps(candidate_index))
    bad_index["weight_map"]["extra"] = "model-00003-of-00003.safetensors"
    with pytest.raises(ValueError, match="tensor names differ"):
        build_plan(
            base_repo="org/base", base_revision="a" * 40,
            base_records=base_records, base_index=base_index,
            candidate_repo="org/candidate", candidate_revision="b" * 40,
            candidate_records=candidate_records, candidate_index=bad_index,
            model_dir=root,
        )

    changed_config = dict(candidate_records)
    changed_config["config.json"] = _record(
        "config.json", b'{"model_type":"other"}')
    with pytest.raises(ValueError, match="identical config.json"):
        build_plan(
            base_repo="org/base", base_revision="a" * 40,
            base_records=base_records, base_index=base_index,
            candidate_repo="org/candidate", candidate_revision="b" * 40,
            candidate_records=changed_config, candidate_index=candidate_index,
            model_dir=root,
        )
