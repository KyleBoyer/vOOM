import json

import numpy as np
from safetensors.numpy import save_file

from runtime.safetensors_diff_gate import compare_checkpoints, read_header


def _checkpoint(directory, writer, stable):
    directory.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file({"writer": writer, "stable": stable}, directory / shard)
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"writer": shard, "stable": shard},
    }))


def test_diff_gate_finds_only_expected_bounded_tensor(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    stable = np.arange(16, dtype=np.float32)
    _checkpoint(
        base, np.arange(8, dtype=np.float32), stable)
    _checkpoint(
        candidate, np.arange(8, dtype=np.float32) + 1, stable)
    result = compare_checkpoints(
        base, candidate, expected_names={"writer"},
        allowed_patterns=(__import__("re").compile(r"writer"),),
        chunk_bytes=7,
    )
    assert result["passed"]
    assert result["changed_tensor_count"] == 1
    assert result["changed_tensors"][0]["tensor"] == "writer"
    assert result["payload_bytes_compared"] == 96


def test_diff_gate_rejects_unexpected_change(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _checkpoint(base, np.zeros(2, np.float32), np.zeros(2, np.float32))
    _checkpoint(candidate, np.ones(2, np.float32), np.ones(2, np.float32))
    result = compare_checkpoints(
        base, candidate, expected_names={"writer"},
        allowed_patterns=(__import__("re").compile(r"writer"),),
        chunk_bytes=3,
    )
    assert not result["passed"]
    assert result["unexpected_changed_tensors"] == ["stable"]
    assert result["extra_vs_expected_tensors"] == ["stable"]


def test_diff_gate_skips_payload_for_exact_symlinked_shard(tmp_path):
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _checkpoint(
        base, np.arange(2, dtype=np.float32),
        np.arange(2, dtype=np.float32))
    candidate.mkdir()
    (candidate / "model-00001-of-00001.safetensors").symlink_to(
        base / "model-00001-of-00001.safetensors")
    (candidate / "model.safetensors.index.json").write_text(
        (base / "model.safetensors.index.json").read_text())
    result = compare_checkpoints(base, candidate, expected_names=set())
    assert result["passed"]
    assert result["linked_shards_skipped"] == 1
    assert result["payload_bytes_compared"] == 0


def test_header_rejects_truncated_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"short")
    try:
        read_header(path)
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("truncated safetensors file was accepted")
