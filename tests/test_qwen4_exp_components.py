from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.config import ModelConfig
from runtime.qwen4_exp import (
    _dilated_causal_depthwise_conv1d,
    _qsa_selection_mask,
    hyper_connection_inject,
    hyper_connection_mix,
    qwen4_rms_norm,
)
from runtime.qwen4_exp_state import Qwen4ExpStateCache


def _cfg(**overrides):
    values = dict(
        model_type="qwen4_exp", hidden_size=4, intermediate_size=3,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=32, rms_norm_eps=1e-6,
        rope_theta=10_000_000.0, max_position_embeddings=128,
        tie_word_embeddings=False, attention_bias=False, head_dim=4,
        eos_token_ids=(31,), torch_dtype="bfloat16", num_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=3,
        shared_expert_intermediate_size=3,
        layer_types=("full_attention",), partial_rotary_factor=0.5,
        qwen4_hc_count=2, qwen4_hc_lowrank=3,
        qwen4_indexer_budget=4, qwen4_indexer_compress_ratio=2,
        qwen4_indexer_head_dim=4, qwen4_indexer_kv_heads=1,
        qwen4_indexer_n_heads=2, qwen4_output_gate_type="sigmoid",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_grouped_zero_centered_rmsnorm_matches_explicit_stream_formula():
    x = mx.array(np.arange(1, 17, dtype=np.float32).reshape(1, 1, 16))
    weight = mx.array(np.linspace(-0.2, 0.2, 16, dtype=np.float32))
    got = qwen4_rms_norm(x, weight, 1e-6, group_size=4)
    streams = x.reshape(1, 1, 4, 4)
    expected = streams * mx.rsqrt(
        mx.mean(streams * streams, axis=-1, keepdims=True) + 1e-6)
    expected = expected * (1 + weight.reshape(4, 4))
    mx.eval(got, expected)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected).reshape(1, 1, 16))


def test_hyper_mix_and_injection_match_released_formula():
    cfg = _cfg()
    rng = np.random.default_rng(7)
    hidden = mx.array(rng.normal(size=(1, 3, 8)).astype(np.float32))
    prefix = "hc"
    weights = {
        f"{prefix}.hc_norm.weight": mx.array(
            rng.normal(scale=.1, size=(8,)).astype(np.float32)),
        f"{prefix}.input_mix_weight_down.weight": mx.array(
            rng.normal(scale=.1, size=(3, 8)).astype(np.float32)),
        f"{prefix}.input_mix_weight_up.weight": mx.array(
            rng.normal(scale=.1, size=(8, 3)).astype(np.float32)),
        f"{prefix}.block_inject_weight.weight": mx.array(
            rng.normal(scale=.1, size=(2, 8)).astype(np.float32)),
    }
    mixed, residual, injection = hyper_connection_mix(
        hidden, weights, prefix, cfg)
    branch = mx.array(rng.normal(size=(1, 3, 4)).astype(np.float32))
    got = hyper_connection_inject(branch, residual, injection)

    normed = qwen4_rms_norm(
        hidden, weights[f"{prefix}.hc_norm.weight"], 1e-6, group_size=4)
    down = normed @ weights[f"{prefix}.input_mix_weight_down.weight"].T / 2
    gate = mx.sigmoid(
        (down * mx.sigmoid(down))
        @ weights[f"{prefix}.input_mix_weight_up.weight"].T)
    expected_mixed = mx.mean(
        gate.reshape(1, 3, 2, 4) * normed.reshape(1, 3, 2, 4), axis=-2)
    expected_injection = 2 * mx.sigmoid(
        (normed @ weights[f"{prefix}.block_inject_weight.weight"].T) / 2)
    expected = hidden + (
        branch[..., None, :] * expected_injection[..., None]).reshape(hidden.shape)
    mx.eval(mixed, got, expected_mixed, expected)
    np.testing.assert_array_equal(np.asarray(mixed), np.asarray(expected_mixed))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_hyper_mix_preserves_released_bfloat16_activation_dtype():
    cfg = _cfg()
    rng = np.random.default_rng(17)
    hidden = mx.array(
        rng.normal(size=(1, 3, 8)).astype(np.float32)).astype(mx.bfloat16)
    prefix = "hc"
    weights = {
        f"{prefix}.hc_norm.weight": mx.array(
            rng.normal(scale=.1, size=(8,)).astype(np.float32)
        ).astype(mx.bfloat16),
        f"{prefix}.input_mix_weight_down.weight": mx.array(
            rng.normal(scale=.1, size=(3, 8)).astype(np.float32)
        ).astype(mx.bfloat16),
        f"{prefix}.input_mix_weight_up.weight": mx.array(
            rng.normal(scale=.1, size=(8, 3)).astype(np.float32)
        ).astype(mx.bfloat16),
        f"{prefix}.block_inject_weight.weight": mx.array(
            rng.normal(scale=.1, size=(2, 8)).astype(np.float32)
        ).astype(mx.bfloat16),
    }
    mixed, residual, injection = hyper_connection_mix(
        hidden, weights, prefix, cfg)
    branch = mx.array(
        rng.normal(size=(1, 3, 4)).astype(np.float32)).astype(mx.bfloat16)
    output = hyper_connection_inject(branch, residual, injection)
    mx.eval(mixed, injection, output)
    assert mixed.dtype == mx.bfloat16
    assert injection.dtype == mx.bfloat16
    assert output.dtype == mx.bfloat16


def test_hyper_injection_rejects_promoted_branch_dtype():
    residual = mx.zeros((1, 2, 8), dtype=mx.bfloat16)
    branch = mx.zeros((1, 2, 4), dtype=mx.float32)
    injection = mx.ones((1, 2, 2), dtype=mx.bfloat16)
    with pytest.raises(TypeError, match="one activation dtype"):
        hyper_connection_inject(branch, residual, injection)


def test_dilated_ple_convolution_split_endpoint_is_exact():
    rng = np.random.default_rng(11)
    x = mx.array(rng.normal(size=(1, 9, 6)).astype(np.float32))
    weight = mx.array(rng.normal(size=(6, 1, 4)).astype(np.float32))
    whole, whole_state = _dilated_causal_depthwise_conv1d(
        x, weight, None, kernel_size=4, dilation=3)
    first, state = _dilated_causal_depthwise_conv1d(
        x[:, :5], weight, None, kernel_size=4, dilation=3)
    second, state = _dilated_causal_depthwise_conv1d(
        x[:, 5:], weight, state, kernel_size=4, dilation=3)
    split = mx.concatenate([first, second], axis=1)
    mx.eval(whole, split, whole_state, state)
    np.testing.assert_array_equal(np.asarray(whole), np.asarray(split))
    np.testing.assert_array_equal(np.asarray(whole_state), np.asarray(state))


def test_qsa_mask_is_causal_and_switches_to_bounded_blocks():
    cfg = _cfg()
    rng = np.random.default_rng(19)
    length = 12
    hidden = mx.array(rng.normal(size=(1, length, 4)).astype(np.float32))
    prefix = "model.layers.0"
    weights = {
        f"{prefix}.self_attn.indexer.index_qk_proj.weight": mx.array(
            rng.normal(scale=.2, size=(12, 4)).astype(np.float32)),
        f"{prefix}.self_attn.indexer.q_layernorm.weight": mx.array(
            rng.normal(scale=.05, size=(4,)).astype(np.float32)),
        f"{prefix}.self_attn.indexer.k_layernorm.weight": mx.array(
            rng.normal(scale=.05, size=(4,)).astype(np.float32)),
    }
    state = Qwen4ExpStateCache(1)
    mask = _qsa_selection_mask(
        hidden, weights, prefix, cfg, 0, 0, state)
    mx.eval(mask)
    host = np.asarray(mask)[0, 0]
    assert host.shape == (length, length)
    assert not np.any(np.triu(host, k=1))
    # At query 3 there are only two complete blocks: dense causal behavior.
    np.testing.assert_array_equal(host[3], np.arange(length) <= 3)
    # Once more than two blocks are complete, selection is bounded to the
    # released four-token budget plus the incomplete causal tail.
    assert host[10].sum() <= cfg.qwen4_indexer_budget + 1
    assert state.qsa_keys[0].shape == (1, length, 4)
    assert state.qsa_positions[0].shape == (1, length)
