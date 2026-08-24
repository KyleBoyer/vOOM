import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

from runtime.qwen_mtp_bf16_sidecar import (
    build_sidecar,
    inspect_sidecar_inputs,
    validate_sidecar,
)


REVISION = "a" * 40
REPOSITORY = "test/released-qwen"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, *, mtp_dtype=mx.bfloat16):
    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    shard = path / shard_name
    mx.save_safetensors(str(shard), {
        "model.body.weight": mx.arange(16, dtype=mx.float32).reshape(4, 4),
        "mtp.fc.weight": mx.arange(32, dtype=mx.float32).reshape(4, 8).astype(
            mtp_dtype),
        "mtp.norm.weight": mx.arange(4, dtype=mx.float32).astype(mtp_dtype),
    })
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {
            "model.body.weight": shard_name,
            "mtp.fc.weight": shard_name,
            "mtp.norm.weight": shard_name,
        },
    }))
    tree_dir = path / ".cache" / "huggingface" / "trees"
    tree_dir.mkdir(parents=True)
    (tree_dir / f"{REVISION}.json").write_text(json.dumps({
        "format_version": 1,
        "files": {
            shard_name: {
                "size": shard.stat().st_size,
                "lfs_size": shard.stat().st_size,
                "lfs_sha256": _sha256(shard),
            },
        },
    }))


def _write_target(path: Path):
    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    shard = path / shard_name
    mx.save_safetensors(str(shard), {
        "model.body.weight": mx.ones((4, 4), dtype=mx.uint32),
        "model.body.scales": mx.ones((4, 1), dtype=mx.float32),
        "mtp.fc.weight": mx.ones((4, 1), dtype=mx.uint32),
        "mtp.fc.scales": mx.ones((4, 1), dtype=mx.float32),
        "mtp.norm.weight": mx.ones((4,), dtype=mx.bfloat16),
    })
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    }))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {
            "model.body.weight": shard_name,
            "model.body.scales": shard_name,
            "mtp.fc.weight": shard_name,
            "mtp.fc.scales": shard_name,
            "mtp.norm.weight": shard_name,
        },
    }))


def _common():
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "expected_tensors": 2,
    }


def test_bf16_sidecar_copies_only_released_mtp_and_preserves_body_map(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_source(source)
    _write_target(target)
    original_index = json.loads(
        (target / "model.safetensors.index.json").read_text())
    original_non_mtp = {
        name: shard for name, shard in original_index["weight_map"].items()
        if not name.startswith("mtp.")
    }

    report = build_sidecar(
        source, target, min_free_bytes=0, **_common())

    assert report["passed"] is True
    assert report["tensor_count"] == 2
    sidecar = mx.load(str(target / "mtp-bf16.safetensors"))
    assert set(sidecar) == {"mtp.fc.weight", "mtp.norm.weight"}
    assert all(value.dtype == mx.bfloat16 for value in sidecar.values())
    assert "model.body.weight" not in sidecar

    updated_index = json.loads(
        (target / "model.safetensors.index.json").read_text())
    assert updated_index["weight_map"] == original_non_mtp
    assert updated_index["metadata"]["mtplx_mtp_sidecar"] == (
        "mtp-bf16.safetensors")
    assert updated_index["metadata"]["mtplx_mtp_source_revision"] == REVISION
    manifest = json.loads((target / "mtp-bf16.manifest.json").read_text())
    assert manifest["source"]["revision"] == REVISION
    assert manifest["source"]["shards"][0]["actual_sha256"] == (
        manifest["source"]["shards"][0]["pinned_lfs_sha256"])
    assert manifest["proof"]["target_non_mtp_weight_map_unchanged"] is True
    assert not list(target.glob(".*.tmp-*.safetensors"))

    # A second invocation is a strict idempotent validation, not an overwrite.
    repeated = build_sidecar(
        source, target, min_free_bytes=0, **_common())
    assert repeated["sidecar_sha256"] == report["sidecar_sha256"]


def test_bf16_sidecar_output_is_reproducible_across_equivalent_targets(tmp_path):
    source = tmp_path / "source"
    target_a = tmp_path / "target-a"
    target_b = tmp_path / "target-b"
    _write_source(source)
    _write_target(target_a)
    _write_target(target_b)

    first = build_sidecar(source, target_a, min_free_bytes=0, **_common())
    second = build_sidecar(source, target_b, min_free_bytes=0, **_common())

    assert first["sidecar_sha256"] == second["sidecar_sha256"]
    assert (target_a / "mtp-bf16.safetensors").read_bytes() == (
        target_b / "mtp-bf16.safetensors").read_bytes()


def test_bf16_sidecar_refuses_source_shard_hash_mismatch_before_writing(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_source(source)
    _write_target(target)
    index_before = (target / "model.safetensors.index.json").read_bytes()
    shard = source / "model-00001-of-00001.safetensors"
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))

    with pytest.raises(ValueError, match="pinned source SHA-256 mismatch"):
        build_sidecar(source, target, min_free_bytes=0, **_common())

    assert not (target / "mtp-bf16.safetensors").exists()
    assert not (target / "mtp-bf16.manifest.json").exists()
    assert (target / "model.safetensors.index.json").read_bytes() == index_before


def test_bf16_sidecar_refuses_non_bf16_released_mtp_header(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_source(source, mtp_dtype=mx.float32)
    _write_target(target)

    with pytest.raises(ValueError, match="must be BF16"):
        inspect_sidecar_inputs(source, target, **_common())


def test_bf16_sidecar_validator_detects_index_binding_tamper(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_source(source)
    _write_target(target)
    build_sidecar(source, target, min_free_bytes=0, **_common())
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["mtplx_mtp_source_revision"] = "b" * 40
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="target index fingerprint"):
        validate_sidecar(source, target, **_common())
