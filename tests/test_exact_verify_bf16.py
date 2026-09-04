from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from runtime.exact_verify_bf16 import (
    exact_verify_bf16_available,
    exact_verify_bf16_matmul,
    exact_verify_bf16_rejection_reason,
)


def test_non_array_weight_representation_falls_back_cleanly():
    class PackedWeight:
        pass

    x = mx.zeros((1, 2, 8), dtype=mx.bfloat16)
    weight = PackedWeight()
    assert exact_verify_bf16_rejection_reason(
        x, weight) == "weight_representation"
    assert exact_verify_bf16_matmul(x, weight) is None


@pytest.mark.parametrize(
    ("length", "inputs", "outputs"),
    [
        (2, 512, 2560),
        (3, 2560, 512),
        (4, 2560, 2560),
        (5, 10240, 2560),
        (8, 2560, 512),
    ],
)
def test_exact_verify_bf16_matches_singleton_gemv(
    length: int,
    inputs: int,
    outputs: int,
):
    if not exact_verify_bf16_available():
        pytest.skip("Metal unavailable")
    generator = np.random.default_rng(length * 100_000 + inputs + outputs)
    x = mx.array(generator.standard_normal((1, length, inputs))).astype(
        mx.bfloat16)
    weight = mx.array(generator.standard_normal((outputs, inputs))).astype(
        mx.bfloat16)
    reference = mx.concatenate(
        [x[:, row:row + 1] @ weight.T for row in range(length)], axis=1)
    candidate = exact_verify_bf16_matmul(x, weight)
    assert candidate is not None
    mx.eval(reference, candidate)
    assert mx.array_equal(reference, candidate).item()


def test_exact_verify_bf16_fails_closed_outside_contract():
    x = mx.zeros((1, 2, 16), dtype=mx.bfloat16)
    assert exact_verify_bf16_matmul(x[:, :1], mx.zeros(
        (16, 16), dtype=mx.bfloat16)) is None
    assert exact_verify_bf16_matmul(x.astype(mx.float16), mx.zeros(
        (16, 16), dtype=mx.float16)) is None
    assert exact_verify_bf16_matmul(x, mx.zeros(
        (3, 16), dtype=mx.bfloat16)) is None
    assert exact_verify_bf16_matmul(
        mx.zeros((1, 2, 64), dtype=mx.bfloat16),
        mx.zeros((4, 64), dtype=mx.bfloat16),
    ) is None


def test_exact_verify_bf16_reports_stable_rejection_reasons():
    if not exact_verify_bf16_available():
        pytest.skip("Metal unavailable")
    x = mx.zeros((1, 2, 16), dtype=mx.bfloat16)
    assert exact_verify_bf16_rejection_reason(
        x[:, :1], mx.zeros((16, 16), dtype=mx.bfloat16)
    ) == "singleton_window"
    assert exact_verify_bf16_rejection_reason(
        x.astype(mx.float16), mx.zeros((16, 16), dtype=mx.bfloat16)
    ) == "dtype"
    assert exact_verify_bf16_rejection_reason(
        x, mx.zeros((3, 16), dtype=mx.bfloat16)
    ) == "output_geometry"
    assert exact_verify_bf16_rejection_reason(
        mx.zeros((1, 2, 64), dtype=mx.bfloat16),
        mx.zeros((4, 64), dtype=mx.bfloat16),
    ) == "skinny_output"
    assert exact_verify_bf16_rejection_reason(
        x, mx.zeros((16, 16), dtype=mx.bfloat16)
    ) is None
