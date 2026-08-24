"""Opt-in serial Metal prefill scan for Qwen DeltaNet.

This path is intentionally classified lossy: the recurrence is causal and
state-equivalent within a small numerical tolerance, but its scalar Metal
reductions do not promise the exact association used by MLX ``mx.sum``.
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from runtime.qwen35 import (
    _native_fused_gated_delta_prefill,
    _sequential_gated_delta_rule,
)


def _inputs(length: int, *, heads: int = 4, dim: int = 8):
    rng = np.random.default_rng(381_000 + length)
    q = mx.array(rng.normal(0, 0.04, (1, length, heads, dim)).astype(np.float32))
    k = mx.array(rng.normal(0, 0.04, (1, length, heads, dim)).astype(np.float32))
    v = mx.array(rng.normal(0, 0.04, (1, length, heads, dim)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (1, length, heads)).astype(np.float32))
    decay = mx.array(-rng.uniform(0.001, 0.08, (1, length, heads)).astype(np.float32))
    state = mx.array(rng.normal(
        0, 0.01, (1, heads, dim, dim)).astype(np.float32))
    return q, k, v, beta, decay, state


@pytest.mark.parametrize("length", [2, 31, 32, 33, 64, 128])
def test_native_serial_prefill_stays_numerically_close(length):
    args = _inputs(length)
    reference_out, reference_state = _sequential_gated_delta_rule(*args)
    candidate_out, candidate_state = _native_fused_gated_delta_prefill(*args)
    mx.eval(reference_out, reference_state, candidate_out, candidate_state)

    out_error = np.max(np.abs(
        np.asarray(candidate_out) - np.asarray(reference_out)))
    state_error = np.max(np.abs(
        np.asarray(candidate_state) - np.asarray(reference_state)))
    assert out_error < 2e-6
    assert state_error < 2e-6


@pytest.mark.parametrize("split", [1, 31, 32, 63])
def test_native_serial_prefill_is_split_stable(split):
    length = 64
    q, k, v, beta, decay, state = _inputs(length)
    full_out, full_state = _native_fused_gated_delta_prefill(
        q, k, v, beta, decay, state)
    if split == 1:
        left_out, left_state = _sequential_gated_delta_rule(
            q[:, :split], k[:, :split], v[:, :split],
            beta[:, :split], decay[:, :split], state)
    else:
        left_out, left_state = _native_fused_gated_delta_prefill(
            q[:, :split], k[:, :split], v[:, :split],
            beta[:, :split], decay[:, :split], state)
    if length - split == 1:
        right_out, right_state = _sequential_gated_delta_rule(
            q[:, split:], k[:, split:], v[:, split:],
            beta[:, split:], decay[:, split:], left_state)
    else:
        right_out, right_state = _native_fused_gated_delta_prefill(
            q[:, split:], k[:, split:], v[:, split:],
            beta[:, split:], decay[:, split:], left_state)
    split_out = mx.concatenate((left_out, right_out), axis=1)
    mx.eval(full_out, full_state, split_out, right_state)
    assert np.max(np.abs(
        np.asarray(full_out) - np.asarray(split_out))) < 2e-6
    assert np.max(np.abs(
        np.asarray(full_state) - np.asarray(right_state))) < 2e-6


def test_native_serial_prefill_rejects_single_position():
    with pytest.raises(ValueError, match="more than one position"):
        _native_fused_gated_delta_prefill(*_inputs(1))
