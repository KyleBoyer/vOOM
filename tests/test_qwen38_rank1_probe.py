"""Small no-network gates for the bounded Qwen3.8 rank-one probe."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from runtime.qwen38_rank1_probe import (
    _bf16_to_f32,
    _read_local_header,
    _validate_tensor_meta,
    analyze_pair,
)


def _bf16_payload(values):
    f32 = np.asarray(values, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype("<u2").tobytes()


def test_bf16_parser_and_rank1_probe_recover_output_direction():
    base = np.array([
        [1.0, 2.0, 3.0],
        [2.0, -1.0, 0.5],
        [0.5, 1.5, -2.0],
    ], dtype=np.float32)
    direction = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    # Ideal residual-writer ablation: W' = W - rr^T W.
    ablated = base - np.outer(direction, direction @ base)
    parsed = _bf16_to_f32(_bf16_payload(base), base.shape)
    np.testing.assert_array_equal(parsed, base)
    probe = analyze_pair(
        "fixture", base, ablated,
        base_sha256="a" * 64, ablated_sha256="b" * 64)
    assert probe.edited is True
    assert probe.metrics["rank1_energy"] == pytest.approx(1.0, abs=1e-6)
    assert probe.metrics["subtracts"] is True
    assert probe.metrics["cosine_projection_form"] == pytest.approx(
        1.0, abs=1e-6)
    assert abs(float(probe.direction @ direction)) == pytest.approx(
        1.0, abs=1e-6)


def test_identical_pair_is_reported_unedited():
    matrix = np.eye(3, dtype=np.float32)
    probe = analyze_pair(
        "fixture", matrix, matrix.copy(),
        base_sha256="a" * 64, ablated_sha256="a" * 64)
    assert probe.edited is False
    assert probe.direction is None


def test_safetensors_header_contract_is_bounded(tmp_path):
    header = {
        "weight": {
            "dtype": "BF16",
            "shape": [2, 3],
            "data_offsets": [0, 12],
        },
    }
    encoded = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"x" * 12)
    actual, data_start = _read_local_header(path)
    assert data_start == 8 + len(encoded)
    assert _validate_tensor_meta("weight", actual["weight"]) == (
        (2, 3), 0, 12)
    with pytest.raises(ValueError, match="extent"):
        _validate_tensor_meta("weight", {
            "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 10],
        })
