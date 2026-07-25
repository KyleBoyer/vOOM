"""Chunked-parallel DeltaNet (WY/UT-transform) vs the sequential recurrence.

The sequential loop (`_sequential_gated_delta_rule`) is the path already
oracle-verified against real HF sources (tests/test_qwen35_oracle.py), so
it IS the oracle here: the chunkwise form must reproduce its outputs and
final state on random tensors across chunk-boundary shapes -- including
L not divisible by the chunk size, L smaller than one chunk, and a
non-zero carried-in state (the mid-conversation case).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from runtime.qwen35 import (
    _chunked_gated_delta_rule,
    _sequential_gated_delta_rule,
)

B, H, K, V = 1, 4, 16, 24


def _random_inputs(L, seed, with_state, decay_range=(-2.0, -0.01)):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, L, H, K)), dtype=mx.float32)
    k = mx.array(rng.standard_normal((B, L, H, K)), dtype=mx.float32)
    # L2-normalized k mirrors the real call site (l2norm before the rule).
    k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    v = mx.array(rng.standard_normal((B, L, H, V)), dtype=mx.float32)
    beta = mx.array(rng.uniform(0.1, 0.9, (B, L, H)), dtype=mx.float32)
    decay = mx.array(rng.uniform(*decay_range, (B, L, H)), dtype=mx.float32)
    state = (
        mx.array(rng.standard_normal((B, H, K, V)) * 0.3, dtype=mx.float32)
        if with_state else mx.zeros((B, H, K, V), dtype=mx.float32))
    return q, k, v, beta, decay, state


@pytest.mark.parametrize("L,chunk", [
    (1, 64), (5, 64), (64, 64), (65, 64), (130, 64), (37, 8), (96, 32),
])
@pytest.mark.parametrize("with_state", [False, True])
def test_chunked_matches_sequential(L, chunk, with_state):
    q, k, v, beta, decay, state = _random_inputs(L, 1234 + L, with_state)
    seq_out, seq_state = _sequential_gated_delta_rule(
        q, k, v, beta, decay, state)
    chk_out, chk_state = _chunked_gated_delta_rule(
        q, k, v, beta, decay, state, chunk=chunk)
    mx.eval(seq_out, seq_state, chk_out, chk_state)
    out_diff = float(mx.max(mx.abs(seq_out - chk_out)))
    state_diff = float(mx.max(mx.abs(seq_state - chk_state)))
    assert out_diff < 1e-4, f"output diff {out_diff} (L={L}, chunk={chunk})"
    assert state_diff < 1e-4, f"state diff {state_diff} (L={L}, chunk={chunk})"
    assert bool(mx.all(mx.isfinite(chk_out)))
    assert bool(mx.all(mx.isfinite(chk_state)))


def test_large_magnitude_decay_does_not_overflow_to_nan():
    """Real bug caught live: exp(cs_j - cs_l) computed for ALL (j,l) before
    masking overflows to +inf for invalid (upper-triangular) entries when
    decay magnitude is large enough; inf * 0 = nan then poisons the whole
    chunk. The small default decay_range above never triggered this --
    real model decay values did (live prefill produced all-token-0 output).
    """
    q, k, v, beta, decay, state = _random_inputs(
        130, 999, with_state=True, decay_range=(-40.0, -0.01))
    chk_out, chk_state = _chunked_gated_delta_rule(
        q, k, v, beta, decay, state, chunk=64)
    seq_out, seq_state = _sequential_gated_delta_rule(
        q, k, v, beta, decay, state)
    mx.eval(chk_out, chk_state, seq_out, seq_state)
    assert bool(mx.all(mx.isfinite(chk_out))), "chunked output has non-finite values"
    assert bool(mx.all(mx.isfinite(chk_state))), "chunked state has non-finite values"
    out_diff = float(mx.max(mx.abs(seq_out - chk_out)))
    assert out_diff < 1e-3, f"output diff {out_diff}"
