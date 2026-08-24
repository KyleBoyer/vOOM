"""Exact gates for opt-in same-operator Qwen DeltaNet compilation."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.qwen35 import (
    _compiled_gated_delta_rule,
    _qwen35_attention_residual,
    _sequential_gated_delta_rule,
)


BATCH, HEADS, KEY_DIM, VALUE_DIM = 1, 2, 16, 24
LENGTHS = (1, 2, 31, 32, 33, 64, 127, 128)


def _inputs(length: int, seed: int = 311):
    rng = np.random.default_rng(seed + length)
    q = mx.array(rng.standard_normal(
        (BATCH, length, HEADS, KEY_DIM), dtype=np.float32))
    k = mx.array(rng.standard_normal(
        (BATCH, length, HEADS, KEY_DIM), dtype=np.float32))
    v = mx.array(rng.standard_normal(
        (BATCH, length, HEADS, VALUE_DIM), dtype=np.float32))
    beta = mx.array(rng.uniform(
        0.05, 0.95, (BATCH, length, HEADS)).astype(np.float32))
    decay = mx.array(rng.uniform(
        -3.0, -0.001, (BATCH, length, HEADS)).astype(np.float32))
    state = mx.array(rng.standard_normal(
        (BATCH, HEADS, KEY_DIM, VALUE_DIM), dtype=np.float32))
    return q, k, v, beta, decay, state


@pytest.mark.parametrize("length", LENGTHS)
def test_compiled_delta_is_array_equal_to_reference(length):
    q, k, v, beta, decay, state = _inputs(length)
    reference_out, reference_state = _sequential_gated_delta_rule(
        q, k, v, beta, decay, state)
    compiled_out, compiled_state = _compiled_gated_delta_rule(
        q, k, v, beta, decay, state)
    mx.eval(reference_out, reference_state, compiled_out, compiled_state)

    assert bool(mx.array_equal(compiled_out, reference_out))
    assert bool(mx.array_equal(compiled_state, reference_state))


@pytest.mark.parametrize("length", LENGTHS)
def test_compiled_delta_is_array_equal_across_checkpoint_split(length):
    q, k, v, beta, decay, state = _inputs(length, seed=733)
    full_out, full_state = _compiled_gated_delta_rule(
        q, k, v, beta, decay, state)

    if length == 1:
        split_out, split_state = _compiled_gated_delta_rule(
            q, k, v, beta, decay, state)
    else:
        # Put the checkpoint immediately before the final position. Across
        # this length ladder that covers both sides of the 32-position
        # materialization boundary, including 31|1 and 32|1.
        split = length - 1
        left_out, left_state = _compiled_gated_delta_rule(
            q[:, :split], k[:, :split], v[:, :split],
            beta[:, :split], decay[:, :split], state)
        mx.eval(left_state)
        right_out, split_state = _compiled_gated_delta_rule(
            q[:, split:], k[:, split:], v[:, split:],
            beta[:, split:], decay[:, split:], left_state)
        split_out = mx.concatenate([left_out, right_out], axis=1)

    mx.eval(full_out, full_state, split_out, split_state)
    assert bool(mx.array_equal(split_out, full_out))
    assert bool(mx.array_equal(split_state, full_state))


def test_compiled_delta_is_array_equal_at_segment_checkpoint():
    q, k, v, beta, decay, state = _inputs(128, seed=991)
    full_out, full_state = _compiled_gated_delta_rule(
        q, k, v, beta, decay, state)
    left_out, left_state = _compiled_gated_delta_rule(
        q[:, :32], k[:, :32], v[:, :32],
        beta[:, :32], decay[:, :32], state)
    mx.eval(left_state)
    right_out, split_state = _compiled_gated_delta_rule(
        q[:, 32:], k[:, 32:], v[:, 32:],
        beta[:, 32:], decay[:, 32:], left_state)
    split_out = mx.concatenate([left_out, right_out], axis=1)
    mx.eval(full_out, full_state, split_out, split_state)

    assert bool(mx.array_equal(split_out, full_out))
    assert bool(mx.array_equal(split_state, full_state))


def test_attention_residual_forwards_compiled_delta_opt_in(monkeypatch):
    import runtime.qwen35 as qwen35

    observed = []

    def fake_delta(h, *_args, **kwargs):
        observed.append(kwargs["compiled_delta_prefill"])
        return mx.zeros_like(h)

    monkeypatch.setattr(qwen35, "_gated_delta_net", fake_delta)
    cfg = SimpleNamespace(
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",),
    )
    x = mx.ones((1, 2, 8), dtype=mx.bfloat16)
    weights = {
        "model.layers.0.input_layernorm.weight": mx.zeros(
            (8,), dtype=mx.bfloat16),
    }
    result = _qwen35_attention_residual(
        x, weights, "model.layers.0", cfg,
        SimpleNamespace(kda_cache=None), 0, 0,
        compiled_delta_prefill=True,
    )
    mx.eval(result)

    assert observed == [True]
    assert bool(mx.array_equal(result, x))


def test_compiled_and_chunked_modes_fail_before_model_load():
    from runtime.engine import RuntimeConfig, StreamingEngine

    with pytest.raises(ValueError, match="mutually exclusive"):
        StreamingEngine(
            "unused",
            RuntimeConfig(
                qwen_compiled_delta_prefill=True,
                qwen_chunked_delta_prefill=True,
            ),
        )
