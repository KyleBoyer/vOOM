"""Side-Quest 3: opt-in, lossy REAP-style expert pruning for Kimi K3.

`ModelConfig.expert_prune_masks` (a runtime routing policy, never populated
from any checkpoint's own config.json -- see its field docstring) tells
`runtime.kimi_linear._route_experts` to mask specific expert indices out of
top-k selection for a given layer, so a pruned expert is never chosen and
therefore never fetched from disk. This test verifies the masking itself
(pure routing math, no real checkpoint needed) and that leaving the field
at its None default is a byte-for-byte no-op, matching every other
checkpoint's config.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from runtime.config import ModelConfig
from runtime.kimi_linear import _route_experts


def _tiny_k3_cfg(num_experts: int, top_k: int, expert_prune_masks=None) -> ModelConfig:
    return ModelConfig(
        model_type="kimi_k3", hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        vocab_size=100, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=8, eos_token_ids=(), torch_dtype="float32",
        num_experts=num_experts, num_experts_per_tok=top_k,
        norm_topk_prob=False, routed_scaling_factor=1.0,
        expert_prune_masks=expert_prune_masks,
    )


def _random_weights(rng, hidden: int, num_experts: int) -> dict:
    return {
        "layer0.gate.weight": mx.array(
            rng.standard_normal((num_experts, hidden)).astype(np.float32)),
        "layer0.gate.e_score_correction_bias": mx.array(
            rng.standard_normal((num_experts,)).astype(np.float32)),
    }


def test_pruned_experts_are_never_selected():
    rng = np.random.default_rng(0)
    hidden, num_experts, top_k, n_tokens = 8, 12, 4, 50
    h = mx.array(rng.standard_normal((1, n_tokens, hidden)).astype(np.float32))
    w = _random_weights(rng, hidden, num_experts)

    pruned = (2, 5, 9)
    cfg = _tiny_k3_cfg(num_experts, top_k, expert_prune_masks={0: pruned})
    idx, pw = _route_experts(h, w, "layer0", cfg, layer=0)
    mx.eval(idx, pw)

    selected = set(np.array(idx).flatten().tolist())
    for e in pruned:
        assert e not in selected, f"pruned expert {e} was selected anyway"


def test_none_expert_prune_masks_is_a_byte_for_byte_noop():
    rng = np.random.default_rng(1)
    hidden, num_experts, top_k, n_tokens = 8, 12, 4, 20
    h = mx.array(rng.standard_normal((1, n_tokens, hidden)).astype(np.float32))
    w = _random_weights(rng, hidden, num_experts)

    cfg_unset = _tiny_k3_cfg(num_experts, top_k, expert_prune_masks=None)
    cfg_empty = _tiny_k3_cfg(num_experts, top_k, expert_prune_masks={})

    idx1, pw1 = _route_experts(h, w, "layer0", cfg_unset, layer=0)
    idx2, pw2 = _route_experts(h, w, "layer0", cfg_unset)  # layer omitted entirely
    idx3, pw3 = _route_experts(h, w, "layer0", cfg_empty, layer=0)
    mx.eval(idx1, pw1, idx2, pw2, idx3, pw3)

    assert np.array_equal(np.array(idx1), np.array(idx2))
    assert np.array_equal(np.array(idx1), np.array(idx3))
    assert np.allclose(np.array(pw1), np.array(pw2))
    assert np.allclose(np.array(pw1), np.array(pw3))


def test_mask_only_applies_to_the_named_layer():
    rng = np.random.default_rng(2)
    hidden, num_experts, top_k, n_tokens = 8, 12, 4, 50
    h = mx.array(rng.standard_normal((1, n_tokens, hidden)).astype(np.float32))
    w = _random_weights(rng, hidden, num_experts)

    pruned = (3,)
    cfg = _tiny_k3_cfg(num_experts, top_k, expert_prune_masks={0: pruned})
    idx_layer0, _ = _route_experts(h, w, "layer0", cfg, layer=0)
    idx_layer1, _ = _route_experts(h, w, "layer0", cfg, layer=1)
    mx.eval(idx_layer0, idx_layer1)

    assert 3 not in set(np.array(idx_layer0).flatten().tolist())
    # Layer 1 has no mask entry -- expert 3 remains eligible there (may or
    # may not actually be selected on this random data, but must not be
    # unconditionally excluded the way layer 0's is).
    unmasked_idx, _ = _route_experts(h, w, "layer0", _tiny_k3_cfg(num_experts, top_k), layer=1)
    mx.eval(unmasked_idx)
    assert np.array_equal(np.array(idx_layer1), np.array(unmasked_idx))


def test_non_k3_model_type_ignores_expert_prune_masks():
    """kimi_linear (and every other model_type) must never consult this
    field -- expert_prune_masks is a kimi_k3-only opt-in policy."""
    rng = np.random.default_rng(3)
    hidden, num_experts, top_k, n_tokens = 8, 12, 4, 50
    h = mx.array(rng.standard_normal((1, n_tokens, hidden)).astype(np.float32))
    w = _random_weights(rng, hidden, num_experts)

    cfg = ModelConfig(
        model_type="kimi_linear", hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        vocab_size=100, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=hidden, eos_token_ids=(), torch_dtype="float32",
        num_experts=num_experts, num_experts_per_tok=top_k,
        norm_topk_prob=False, routed_scaling_factor=1.0,
        expert_prune_masks={0: (2, 5, 9)},
    )
    idx, _ = _route_experts(h, w, "layer0", cfg, layer=0)
    mx.eval(idx)
    selected = set(np.array(idx).flatten().tolist())
    assert selected & {2, 5, 9}, (
        "kimi_linear must ignore expert_prune_masks entirely -- a kimi_k3-only field")
