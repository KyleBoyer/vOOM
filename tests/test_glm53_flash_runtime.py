"""Small released-operator and state gates for GLM-5.3-Flash."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _dsa_cfg():
    return SimpleNamespace(
        index_head_dim=2,
        index_n_heads=2,
        index_topk=4,
        index_kpool=2,
        index_kpool_always_select_tail=True,
    )


def _dsa_weights(prefix: str, hidden_size: int = 4):
    import mlx.core as mx

    base = f"{prefix}.self_attn.indexer"
    return {
        f"{base}.wk.weight": mx.array([
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
        ], dtype=mx.float32),
        f"{base}.k_norm.weight": mx.ones((2,), dtype=mx.float32),
        f"{base}.k_norm.bias": mx.zeros((2,), dtype=mx.float32),
        # Bare released parameters: neither has a ``.weight`` suffix.
        f"{base}.index_kpool_compress_gate": mx.array([
            [0.25, 0.5, -0.25, 0.125],
            [-0.5, 0.125, 0.25, 0.75],
        ], dtype=mx.float32),
        f"{base}.index_kpool_compress_ape": mx.array([
            [0.1, -0.2],
            [-0.3, 0.4],
        ], dtype=mx.float32),
        f"{base}.wq_b.weight": mx.array([
            [0.5, -0.25],
            [0.25, 0.75],
            [-0.5, 0.125],
            [0.375, -0.25],
        ], dtype=mx.float32),
        f"{base}.weights_proj.weight": mx.array([
            [0.25, -0.5, 0.75, 0.125],
            [-0.125, 0.5, 0.25, -0.75],
        ], dtype=mx.float32),
    }


def test_glm53_swiglu_matches_released_asymmetric_clamp():
    import mlx.core as mx

    from runtime.glm5_next import glm5_next_swiglu

    cfg = SimpleNamespace(swiglu_limit=2.0)
    prefix = "model.layers.0.mlp"
    x = mx.array([[[3.0, -4.0]]], dtype=mx.float32)
    w = {
        f"{prefix}.gate_proj.weight": mx.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32),
        f"{prefix}.up_proj.weight": mx.array(
            [[1.0, 1.0], [-1.0, 1.0]], dtype=mx.float32),
        f"{prefix}.down_proj.weight": mx.eye(2, dtype=mx.float32),
    }
    got = glm5_next_swiglu(x, w, prefix, cfg)
    gate = np.minimum(np.array([3.0, -4.0]), 2.0)
    up = np.clip(np.array([-1.0, -7.0]), -2.0, 2.0)
    expected = gate / (1.0 + np.exp(-gate)) * up
    mx.eval(got)
    np.testing.assert_allclose(np.array(got)[0, 0], expected, rtol=1e-6)


def test_glm53_pooled_dsa_is_causal_chronological_and_forkable():
    import mlx.core as mx

    from runtime.glm5_next_dsa import GLM5NextDSAState

    cfg = _dsa_cfg()
    prefix = "model.layers.3"
    weights = _dsa_weights(prefix)
    hidden = mx.arange(24, dtype=mx.float32).reshape(1, 6, 4) / 13
    q_resid = mx.arange(12, dtype=mx.float32).reshape(1, 6, 2) / 11
    state = GLM5NextDSAState(cfg)
    assert state.stats["shared_reuses"] == 0
    state.observe(3, "full", hidden, weights, prefix, offset=0)
    selected = state.update_and_select(
        3, "full", hidden, q_resid, weights, prefix, offset=0)
    mx.eval(selected)
    values = np.array(selected)

    assert values.shape == (1, 6, 5)
    for query, row in enumerate(values[0]):
        valid = row[row >= 0]
        assert np.all(valid <= query)
        assert np.all(valid[:-1] <= valid[1:])
        assert len(valid) == len(set(valid.tolist()))
    # All visible tokens are retained while the causal prefix fits the budget.
    assert np.array_equal(values[0, 3], np.array([0, 1, 2, 3, -1]))

    branch = state.fork()
    state.trim(4)
    assert state.k_idx[3].shape[1] == 4
    assert branch.k_idx[3].shape[1] == 6
    assert branch.nbytes() > state.nbytes()


def test_glm53_pooled_dsa_rejects_misaligned_append():
    import mlx.core as mx

    from runtime.glm5_next_dsa import GLM5NextDSAState

    prefix = "model.layers.3"
    state = GLM5NextDSAState(_dsa_cfg())
    weights = _dsa_weights(prefix)
    hidden = mx.ones((1, 2, 4), dtype=mx.float32)
    state.observe(3, "full", hidden, weights, prefix, offset=0)
    with pytest.raises(ValueError, match="offset"):
        state.observe(3, "full", hidden[:, :1], weights, prefix, offset=1)


def test_glm53_hybrid_kv_fork_preserves_compressed_and_dsa_state():
    import mlx.core as mx

    from runtime.glm5_next_dsa import GLM5NextDSAState
    from runtime.kimi_linear import KDAStateCache
    from runtime.kv_cache import KVCache

    kv = KVCache(4)
    kv.compressed_mla = True
    kv.keys[3] = mx.ones((1, 3, 2), dtype=mx.bfloat16)
    kv.kda_cache = KDAStateCache(4)
    kv.dsa = GLM5NextDSAState(_dsa_cfg())
    kv.dsa.k_idx[3] = mx.ones((1, 3, 5), dtype=mx.float32)
    branch = kv.fork()

    assert branch.compressed_mla
    assert branch.keys[3] is kv.keys[3]
    assert branch.dsa is not kv.dsa
    branch.trim(2)
    assert branch.keys[3].shape[1] == 2
    assert branch.dsa.k_idx[3].shape[1] == 2
    assert kv.keys[3].shape[1] == 3
    assert kv.dsa.k_idx[3].shape[1] == 3


def test_glm53_layer_stationary_moe_preserves_tile_shapes_and_expert_order():
    import mlx.core as mx

    from runtime.glm5_next import (
        glm5_next_mlp, glm5_next_mlp_layer_stationary_tiles)

    cfg = SimpleNamespace(
        mlp_layer_types=("moe",), first_k_dense_replace=0,
        num_experts_per_tok=2, norm_topk_prob=True,
        routed_scaling_factor=1.0, swiglu_limit=4.0)
    prefix = "model.layers.0"
    hidden = 4
    inter = 3
    w = {
        f"{prefix}.mlp.gate.weight": mx.array([
            [0.5, -0.25, 0.75, 0.125],
            [-0.5, 0.25, 0.125, 0.75],
            [0.25, 0.5, -0.25, 0.375],
            [-0.125, 0.5, 0.25, -0.75],
        ], dtype=mx.float32),
        f"{prefix}.mlp.gate.e_score_correction_bias": mx.array(
            [0.1, -0.2, 0.05, 0.3], dtype=mx.float32),
    }
    shared_prefix = f"{prefix}.mlp.shared_experts"
    w.update({
        f"{shared_prefix}.gate_proj.weight": mx.ones(
            (inter, hidden), dtype=mx.float32) * 0.1,
        f"{shared_prefix}.up_proj.weight": mx.ones(
            (inter, hidden), dtype=mx.float32) * -0.2,
        f"{shared_prefix}.down_proj.weight": mx.ones(
            (hidden, inter), dtype=mx.float32) * 0.3,
    })
    experts = {}
    for expert in range(4):
        base = f"{prefix}.mlp.experts.{expert}"
        scale = 0.03 * (expert + 1)
        experts[expert] = {
            f"{base}.gate_proj.weight": mx.ones(
                (inter, hidden), dtype=mx.float32) * scale,
            f"{base}.up_proj.weight": mx.eye(
                inter, hidden, dtype=mx.float32) * (scale + 0.02),
            f"{base}.down_proj.weight": mx.ones(
                (hidden, inter), dtype=mx.float32) * (scale - 0.01),
        }

    tiles = [
        mx.array([[[1.0, 0.5, -0.25, 0.75]]], dtype=mx.float32),
        mx.array([[[0.25, -0.5, 1.0, 0.125],
                   [-0.75, 0.25, 0.5, 1.0]]], dtype=mx.float32),
    ]
    reference = [
        glm5_next_mlp(
            tile, w, prefix, cfg, 0,
            lambda _layer, ids, **_kw: {i: experts[i] for i in ids})
        for tile in tiles
    ]
    requested = []

    def batches(_layer, ids, **_kwargs):
        for expert in ids:
            requested.append(expert)
            yield [expert], {expert: experts[expert]}

    candidate = glm5_next_mlp_layer_stationary_tiles(
        tiles, w, prefix, cfg, 0,
        lambda *_args, **_kwargs: {}, iter_expert_batches=batches)
    mx.eval(*reference, *candidate)

    assert requested == sorted(set(requested))
    assert len(requested) <= 4
    for expected, actual in zip(reference, candidate):
        assert expected.shape == actual.shape
        assert bool(mx.all(expected == actual))


def test_glm53_mtp_block_is_plain_residual_not_hyperconnected(monkeypatch):
    import mlx.core as mx

    import runtime.glm5_next as glm5

    cfg = SimpleNamespace(rms_norm_eps=1e-5)
    prefix = "model.layers.45"
    weights = {
        f"{prefix}.input_layernorm.weight": mx.ones(
            (4,), dtype=mx.float32),
        f"{prefix}.post_attention_layernorm.weight": mx.ones(
            (4,), dtype=mx.float32),
    }
    seen = {}

    def attention(hidden, *_args, **kwargs):
        seen["attention_override"] = kwargs.get("indexer_type_override")
        return mx.ones_like(hidden) * 0.25

    def mlp(hidden, *_args, **_kwargs):
        seen["mlp_shape"] = hidden.shape
        return mx.ones_like(hidden) * -0.125

    monkeypatch.setattr(glm5, "glm5_next_mla_attention", attention)
    monkeypatch.setattr(glm5, "glm5_next_mlp", mlp)
    x = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.bfloat16)
    got = glm5.run_glm5_next_mtp_block(
        x, weights, prefix, cfg, object(), 45, 7,
        lambda *_args, **_kwargs: {})
    mx.eval(got)

    assert seen["attention_override"] == "full"
    assert seen["mlp_shape"] == x.shape
    expected = (x + mx.array(0.125, dtype=mx.bfloat16)).astype(mx.bfloat16)
    assert bool(mx.all(got == expected))
