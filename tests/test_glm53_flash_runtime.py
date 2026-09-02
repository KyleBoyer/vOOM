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


def test_glm53_incremental_pool_cache_matches_full_rebuild_across_tail():
    import mlx.core as mx

    from runtime.glm5_next_dsa import GLM5NextDSAState

    cfg = _dsa_cfg()
    prefix = "model.layers.3"
    weights = _dsa_weights(prefix)
    hidden = mx.arange(40, dtype=mx.float32).reshape(1, 10, 4) / 17
    q_resid = mx.arange(20, dtype=mx.float32).reshape(1, 10, 2) / 19

    rebuilt = GLM5NextDSAState(cfg)
    rebuilt.observe(3, "full", hidden, weights, prefix, offset=0)
    expected = rebuilt.update_and_select(
        3, "full", hidden[:, 7:], q_resid[:, 7:], weights, prefix,
        offset=7)
    expected_pools, _indices, _valid = rebuilt._pooled_states_reference(
        rebuilt.k_idx[3], weights, prefix)

    cached = GLM5NextDSAState(cfg, incremental_pool_cache=True)
    cached.observe(3, "full", hidden[:, :7], weights, prefix, offset=0)
    cached.update_and_select(
        3, "full", hidden[:, :7], q_resid[:, :7], weights, prefix,
        offset=0)
    cached.observe(3, "full", hidden[:, 7:], weights, prefix, offset=7)
    got = cached.update_and_select(
        3, "full", hidden[:, 7:], q_resid[:, 7:], weights, prefix,
        offset=7)
    mx.eval(expected, expected_pools, got, cached.pool_keys[3])

    assert np.array_equal(np.array(got), np.array(expected))
    assert np.array_equal(
        np.array(cached.pool_keys[3]), np.array(expected_pools))
    assert cached.stats["pool_rows_reused"] == 3
    assert cached.stats["pool_rows_computed"] == 6
    assert cached.stats["packed_capacity_grows"] == 1
    assert cached.stats["packed_rows_copied"] == 0
    assert cached.stats["packed_rows_appended"] == 10
    assert cached.stats["packed_capacity_rows_peak"] == 1024
    assert cached._packed_capacity[3].shape[1] == 1024
    assert cached.stats["pool_capacity_grows"] == 1
    assert cached.stats["pool_rows_copied"] == 0
    assert cached.stats["pool_capacity_rows_peak"] == 256
    assert cached.stats["pool_metadata_rows_avoided"] == 9
    assert cached._pool_capacity[3].shape[1] == 256

    branch = cached.fork()
    branch.observe(
        3, "full", hidden[:, :1], weights, prefix, offset=10)
    assert branch.k_idx[3].shape[1] == 11
    assert cached.k_idx[3].shape[1] == 10
    assert branch.stats["packed_capacity_grows"] == 2
    assert branch.stats["packed_rows_copied"] == 10
    cached.trim(5)
    assert cached.pool_keys[3].shape[1] == 2
    assert branch.pool_keys[3].shape[1] == 5


def test_glm53_incremental_pool_keys_grow_by_capacity_not_every_tile():
    import mlx.core as mx

    from runtime.glm5_next_dsa import GLM5NextDSAState

    cfg = _dsa_cfg()
    prefix = "model.layers.3"
    weights = _dsa_weights(prefix)
    hidden = mx.arange(600 * 4, dtype=mx.float32).reshape(1, 600, 4) / 97
    q_resid = mx.arange(600 * 2, dtype=mx.float32).reshape(1, 600, 2) / 89

    rebuilt = GLM5NextDSAState(cfg)
    rebuilt.observe(3, "full", hidden, weights, prefix, offset=0)
    expected = rebuilt.update_and_select(
        3, "full", hidden[:, -1:], q_resid[:, -1:], weights, prefix,
        offset=599)

    cached = GLM5NextDSAState(cfg, incremental_pool_cache=True)
    cached.observe(3, "full", hidden[:, :512], weights, prefix, offset=0)
    cached.update_and_select(
        3, "full", hidden[:, 511:512], q_resid[:, 511:512], weights,
        prefix, offset=511)
    cached.observe(3, "full", hidden[:, 512:], weights, prefix, offset=512)
    got = cached.update_and_select(
        3, "full", hidden[:, -1:], q_resid[:, -1:], weights, prefix,
        offset=599)
    mx.eval(expected, got, cached.pool_keys[3])

    assert np.array_equal(np.array(got), np.array(expected))
    assert cached.pool_keys[3].shape[1] == 300
    assert cached._pool_capacity[3].shape[1] == 512
    assert cached.stats["pool_capacity_grows"] == 2
    assert cached.stats["pool_rows_copied"] == 256


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


def test_glm53_compressed_mla_factory_uses_exact_stepped_cache():
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.kv_cache import SteppedKVCache

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = RuntimeConfig(mla_compressed_kv=True)
    engine.cfg = SimpleNamespace(
        model_type="glm5_next",
        num_hidden_layers=6,
    )
    engine._position_free_pool = None
    engine._dsa_elided = False

    kv = engine.new_kv()

    assert isinstance(kv, SteppedKVCache)
    assert kv.compressed_mla is True
    assert kv.mla_absorbed is False
    assert kv.glm53_sparse_absorbed_mla is False
    assert kv.glm53_sparse_fused_attention is False
    assert kv.glm53_sparse_fused_kv_int8 is False
    assert kv.dsa.incremental_pool_cache is False
    assert hasattr(kv, "dsa")
    assert hasattr(kv, "kda_cache")

    engine.rc.glm53_sparse_absorbed_mla = True
    absorbed = engine.new_kv()
    assert absorbed.glm53_sparse_absorbed_mla is True

    engine.rc.glm53_sparse_absorbed_mla = False
    engine.rc.glm53_sparse_fused_attention = True
    engine.rc.glm53_sparse_fused_kv_int8 = True
    fused = engine.new_kv()
    assert fused.glm53_sparse_fused_attention is True
    assert fused.glm53_sparse_fused_kv_int8 is True

    engine.rc.glm53_incremental_dsa_pool = True
    pooled = engine.new_kv()
    assert pooled.dsa.incremental_pool_cache is True


def test_full_glm_long_context_factory_wires_exact_spill_and_tiling(tmp_path):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.kv_cache import SteppedKVCache

    spill = tmp_path / "glm-full-spill"
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = RuntimeConfig(
        mla_compressed_kv=True,
        glm_dsa_key_tile_size=128,
        glm_dsa_index_step_size=512,
        glm_dsa_selection_query_tile_size=64,
        glm_dsa_sparse_absorbed_mla=True,
        glm_dsa_mla_kv_spill_dir=str(spill),
    )
    engine.cfg = SimpleNamespace(
        model_type="glm_moe_dsa",
        num_hidden_layers=3,
    )
    engine._position_free_pool = None
    engine._dsa_elided = False

    kv = engine.new_kv()
    assert isinstance(kv, SteppedKVCache)
    assert kv.compressed_mla
    assert kv.latent_spill_enabled
    assert kv.dsa.key_tile_size == 128
    assert kv.dsa.index_step_size == 512
    assert kv.dsa.selection_query_tile_size == 64
    assert kv.mla_absorbed_prefill
    assert kv.dsa.selection_spill_dir == spill.resolve()
    kv.close_latent_spill()
    kv.dsa.close_selection_spill()


def test_glm53_stepped_latents_equal_plain_cache_across_tiles():
    import mlx.core as mx

    from runtime.kv_cache import KVCache, SteppedKVCache

    plain = KVCache(1)
    plain.compressed_mla = True
    stepped = SteppedKVCache(1)
    stepped.compressed_mla = True
    expected = actual = None
    for length in (31, 32, 33, 128, 1):
        tile = mx.arange(length * 7, dtype=mx.float32).reshape(
            1, length, 7).astype(mx.bfloat16)
        expected = plain.update_latent(0, tile)
        actual = stepped.update_latent(0, tile)
        mx.eval(expected, actual)
        assert expected.shape == actual.shape
        assert bool(mx.all(expected == actual))
    assert plain.offset == stepped.offset == 225
    assert stepped.allocated_nbytes() < plain.nbytes() * 2


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
    reference_order = list(requested)
    requested.clear()
    coalesced_stats = {}
    coalesced = glm5_next_mlp_layer_stationary_tiles(
        tiles, w, prefix, cfg, 0,
        lambda *_args, **_kwargs: {}, iter_expert_batches=batches,
        coalesce_expert_positions=True,
        coalesced_expert_max_positions=1,
        coalesced_stats=coalesced_stats)
    mx.eval(*reference, *candidate, *coalesced)

    assert reference_order == sorted(set(reference_order))
    assert requested == reference_order
    assert len(requested) <= 4
    for expected, actual in zip(reference, candidate):
        assert expected.shape == actual.shape
        assert bool(mx.all(expected == actual))
    for expected, actual in zip(reference, coalesced):
        assert expected.shape == actual.shape
        assert bool(mx.allclose(expected, actual, atol=1e-5, rtol=1e-5))
    assert coalesced_stats["max_positions"] <= 1
    assert coalesced_stats["gemm_calls"] >= len(requested)
    assert coalesced_stats["split_experts"] >= 1


def test_glm53_coalesced_expert_position_limit_must_be_positive():
    import mlx.core as mx

    from runtime.glm5_next import glm5_next_mlp_layer_stationary_tiles

    cfg = SimpleNamespace(
        mlp_layer_types=("moe",), first_k_dense_replace=0)
    with pytest.raises(ValueError, match="must be positive"):
        glm5_next_mlp_layer_stationary_tiles(
            [mx.zeros((1, 1, 4))], {}, "model.layers.0", cfg, 0,
            lambda *_args, **_kwargs: {},
            coalesce_expert_positions=True,
            coalesced_expert_max_positions=0)

@pytest.mark.parametrize("tile_size", [1, 2, 4])
def test_glm53_sparse_mla_query_tiling_is_byte_exact(tile_size):
    import mlx.core as mx

    from runtime.glm5_next import _glm5_next_sparse_mla_attention

    prefix = "model.layers.3"
    heads, queries, width = 2, 5, 7
    key_dim, value_dim, latent_dim = 3, 2, 4
    query = (mx.arange(
        heads * queries * key_dim, dtype=mx.float32).reshape(
            1, heads, queries, key_dim) / 17).astype(mx.bfloat16)
    latent = (mx.arange(
        13 * latent_dim, dtype=mx.float32).reshape(
            1, 13, latent_dim) / 23).astype(mx.bfloat16)
    selection = mx.array([[list(range(row, row + width))
                           for row in range(queries)]], dtype=mx.int32)
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 31
        ).astype(mx.bfloat16),
    }
    reference = _glm5_next_sparse_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=queries)
    candidate = _glm5_next_sparse_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=tile_size)
    mx.eval(reference, candidate)

    assert reference.shape == candidate.shape == (
        1, heads, queries, value_dim)
    assert bool(mx.all(reference == candidate))


@pytest.mark.parametrize("tile_size", [1, 2, 5])
def test_glm53_sparse_absorbed_mla_matches_eager_and_tiles(tile_size):
    import mlx.core as mx

    from runtime.glm5_next import (
        _glm5_next_sparse_absorbed_mla_attention,
        _glm5_next_sparse_mla_attention,
    )

    prefix = "model.layers.3"
    heads, queries, width = 2, 5, 7
    key_dim, value_dim, latent_dim = 3, 2, 4
    query = (mx.arange(
        heads * queries * key_dim, dtype=mx.float32).reshape(
            1, heads, queries, key_dim) / 17).astype(mx.bfloat16)
    latent = (mx.arange(
        13 * latent_dim, dtype=mx.float32).reshape(
            1, 13, latent_dim) / 23).astype(mx.bfloat16)
    selection = mx.array([[
        [-1, *range(row, row + width - 1)]
        for row in range(queries)
    ]], dtype=mx.int32)
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 31
        ).astype(mx.bfloat16),
    }
    eager = _glm5_next_sparse_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=queries)
    reference = _glm5_next_sparse_absorbed_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=queries)
    candidate = _glm5_next_sparse_absorbed_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=tile_size)
    mx.eval(eager, reference, candidate)

    assert bool(mx.all(reference == candidate))
    np.testing.assert_allclose(
        np.asarray(reference.astype(mx.float32)),
        np.asarray(eager.astype(mx.float32)), rtol=8e-3, atol=2e-2)


def test_absorbed_mla_keeps_dense_prefix_then_releases_expanded_layer():
    from types import SimpleNamespace

    from runtime.engine import (
        _glm53_expanded_prefill_cache,
        _glm53_release_expanded_prefill_layer,
    )

    ordinary = SimpleNamespace(glm53_sparse_absorbed_mla=False)
    absorbed = SimpleNamespace(glm53_sparse_absorbed_mla=True)
    assert _glm53_expanded_prefill_cache(ordinary) == {}
    assert _glm53_expanded_prefill_cache(absorbed) == {}

    ordinary._glm53_expanded_prefill = {3: object(), 7: object()}
    absorbed._glm53_expanded_prefill = {3: object(), 7: object()}
    _glm53_release_expanded_prefill_layer(ordinary, 3)
    _glm53_release_expanded_prefill_layer(absorbed, 3)
    assert set(ordinary._glm53_expanded_prefill) == {7}
    assert set(absorbed._glm53_expanded_prefill) == {7}


def test_glm53_dense_absorbed_mla_matches_row_selected_eager():
    import mlx.core as mx

    from runtime.glm5_next import (
        _glm5_next_dense_absorbed_mla_attention,
        _glm5_next_sparse_mla_attention,
    )

    prefix = "model.layers.3"
    heads, queries, key_length = 2, 5, 13
    key_dim, value_dim, latent_dim = 3, 2, 4
    offset = key_length - queries
    query = (mx.arange(
        heads * queries * key_dim, dtype=mx.float32).reshape(
            1, heads, queries, key_dim) / 17).astype(mx.bfloat16)
    latent = (mx.arange(
        key_length * latent_dim, dtype=mx.float32).reshape(
            1, key_length, latent_dim) / 23).astype(mx.bfloat16)
    key_positions = mx.arange(key_length, dtype=mx.int32)[None, None, :]
    query_positions = mx.arange(
        offset, offset + queries, dtype=mx.int32)[None, :, None]
    selection = mx.where(
        key_positions <= query_positions, key_positions, -1)
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 31
        ).astype(mx.bfloat16),
    }
    eager = _glm5_next_sparse_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=queries)
    absorbed = _glm5_next_dense_absorbed_mla_attention(
        query, latent, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim, offset=offset)
    mx.eval(eager, absorbed)

    np.testing.assert_allclose(
        np.asarray(absorbed.astype(mx.float32)),
        np.asarray(eager.astype(mx.float32)), rtol=2e-2, atol=1.3e-1)


def test_glm53_exact_prefill_kv_projects_each_row_once_byte_exact():
    import mlx.core as mx

    from runtime.glm5_next import _glm5_next_update_expanded_prefill_kv
    from runtime.layer_runner import _linear

    prefix = "model.layers.3"
    heads, key_dim, value_dim, latent_dim = 2, 3, 2, 4
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 31
        ).astype(mx.bfloat16),
    }
    tiles = [
        (mx.arange(length * latent_dim, dtype=mx.float32).reshape(
            1, length, latent_dim) + shift).astype(mx.bfloat16)
        for length, shift in ((31, 0), (32, 100), (33, 300))
    ]
    cache = {}
    keys = values = None
    for tile in tiles:
        keys, values = _glm5_next_update_expanded_prefill_kv(
            tile, weights, prefix, layer=3, cache=cache,
            heads=heads, key_dim=key_dim, value_dim=value_dim)
    latent = mx.concatenate(tiles, axis=1)
    full = _linear(
        latent, weights, f"{prefix}.self_attn.kv_b_proj").reshape(
            1, latent.shape[1], heads, key_dim + value_dim).transpose(
                0, 2, 1, 3)
    mx.eval(keys, values, full)

    assert bool(mx.all(keys == full[..., :key_dim]))
    assert bool(mx.all(values == full[..., key_dim:]))
    assert cache[3][0].shape[2] == 256


def test_glm53_int8_prefill_kv_is_bounded_and_row_scaled():
    import mlx.core as mx
    import numpy as np

    from runtime.glm5_next import (
        _glm5_next_update_expanded_prefill_kv_int8)

    prefix = "model.layers.3"
    heads, key_dim, value_dim, latent_dim = 2, 32, 32, 4
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 257
        ).astype(mx.bfloat16),
    }
    cache = {}
    keys = values = key_scales = value_scales = None
    for shift in (0, 11):
        latent = (mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
                  + shift).astype(mx.bfloat16)
        keys, values, key_scales, value_scales = (
            _glm5_next_update_expanded_prefill_kv_int8(
                latent, weights, prefix, layer=3, cache=cache,
                heads=heads, key_dim=key_dim, value_dim=value_dim))
    mx.eval(keys, values, key_scales, value_scales)

    assert keys.dtype == mx.int8
    assert values.dtype == mx.int8
    assert key_scales.dtype == mx.float32
    assert value_scales.dtype == mx.float32
    assert keys.shape == (1, heads, 6, key_dim)
    assert cache[3][0].shape[2] == 256
    assert cache[3][0].nbytes * 2 < (
        1 * heads * 256 * key_dim * 2 * 2)
    dequant = keys.astype(mx.float32) * key_scales
    mx.eval(dequant)
    assert np.isfinite(np.asarray(dequant)).all()


@pytest.mark.parametrize("tile_size", [1, 2, 4])
def test_glm53_sparse_expanded_attention_matches_reprojection(tile_size):
    import mlx.core as mx

    from runtime.glm5_next import (
        _glm5_next_sparse_expanded_attention,
        _glm5_next_sparse_mla_attention,
    )
    from runtime.layer_runner import _linear

    prefix = "model.layers.3"
    heads, queries, width = 2, 5, 7
    key_dim, value_dim, latent_dim = 3, 2, 4
    query = (mx.arange(
        heads * queries * key_dim, dtype=mx.float32).reshape(
            1, heads, queries, key_dim) / 17).astype(mx.bfloat16)
    latent = (mx.arange(
        13 * latent_dim, dtype=mx.float32).reshape(
            1, 13, latent_dim) / 23).astype(mx.bfloat16)
    selection = mx.array([[
        [-1, *range(row, row + width - 1)]
        for row in range(queries)
    ]], dtype=mx.int32)
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": (
            mx.arange(
                heads * (key_dim + value_dim) * latent_dim,
                dtype=mx.float32).reshape(
                    heads * (key_dim + value_dim), latent_dim) / 31
        ).astype(mx.bfloat16),
    }
    expanded = _linear(
        latent, weights, f"{prefix}.self_attn.kv_b_proj").reshape(
            1, latent.shape[1], heads, key_dim + value_dim).transpose(
                0, 2, 1, 3)
    reference = _glm5_next_sparse_mla_attention(
        query, latent, selection, weights, prefix,
        heads=heads, key_dim=key_dim, value_dim=value_dim,
        query_tile_size=queries)
    candidate = _glm5_next_sparse_expanded_attention(
        query, expanded[..., :key_dim], expanded[..., key_dim:], selection,
        key_dim=key_dim, query_tile_size=tile_size)
    mx.eval(reference, candidate)

    assert bool(mx.all(reference == candidate))


def test_glm53_mtp_prompt_prefill_keeps_exact_cache_latent_and_skips_dead_block():
    import mlx.core as mx

    from runtime.glm_mtp import MTPDrafter
    from runtime.kv_cache import KVCache
    from runtime.layer_runner import _linear

    hidden, latent_width, positions = 4, 3, 5
    layer = 1
    prefix = f"model.layers.{layer}"
    weights = {
        f"{prefix}.enorm.weight": mx.ones((hidden,), dtype=mx.bfloat16),
        f"{prefix}.hnorm.weight": mx.ones((hidden,), dtype=mx.bfloat16),
        f"{prefix}.eh_proj.weight": (
            mx.arange(hidden * hidden * 2, dtype=mx.float32).reshape(
                hidden, hidden * 2) / 31).astype(mx.bfloat16),
        f"{prefix}.input_layernorm.weight": mx.ones(
            (hidden,), dtype=mx.bfloat16),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": (
            mx.arange(latent_width * hidden, dtype=mx.float32).reshape(
                latent_width, hidden) / 17).astype(mx.bfloat16),
        f"{prefix}.self_attn.kv_a_layernorm.weight": mx.ones(
            (latent_width,), dtype=mx.bfloat16),
    }

    class _Store:
        @staticmethod
        def names_with_prefix(_prefix):
            return tuple(weights)

    class _Cache:
        @staticmethod
        def get(_key, _names):
            return weights

    embeddings = (
        mx.arange(20 * hidden, dtype=mx.float32).reshape(20, hidden) / 13
    ).astype(mx.bfloat16)
    cfg = SimpleNamespace(
        model_type="glm5_next",
        num_hidden_layers=layer,
        rms_norm_eps=1e-6,
        mla_latent_norm_eps=1e-6,
    )
    engine = SimpleNamespace(
        cfg=cfg,
        store=_Store(),
        cache=_Cache(),
        _norm_w=mx.ones((hidden,), dtype=mx.bfloat16),
        _embed=lambda ids: embeddings[mx.array(ids, dtype=mx.int32)][None],
    )
    tokens = [1, 2, 3, 4, 5]
    h_window = (
        mx.arange(positions * hidden, dtype=mx.float32).reshape(
            1, positions, hidden) / 19).astype(mx.bfloat16)

    drafter = MTPDrafter(engine)
    cache = KVCache(layer + 1)
    cache.compressed_mla = True
    drafter.prefill(tokens, h_window, cache)

    e = engine._embed(tokens[1:])
    e = mx.concatenate([mx.zeros_like(e[:, :1]), e[:, 1:]], axis=1)
    e = mx.fast.rms_norm(e, weights[f"{prefix}.enorm.weight"], 1e-6)
    target_hidden = mx.fast.rms_norm(h_window[:, :-1], engine._norm_w, 1e-6)
    target_hidden = mx.fast.rms_norm(
        target_hidden, weights[f"{prefix}.hnorm.weight"], 1e-6)
    x = _linear(
        mx.concatenate([e, target_hidden], axis=-1), weights,
        f"{prefix}.eh_proj")
    attention_input = mx.fast.rms_norm(
        x, weights[f"{prefix}.input_layernorm.weight"], 1e-6)
    expected = _linear(
        attention_input, weights,
        f"{prefix}.self_attn.kv_a_proj_with_mqa")
    expected = mx.fast.rms_norm(
        expected,
        weights[f"{prefix}.self_attn.kv_a_layernorm.weight"], 1e-6)
    mx.eval(expected, cache.keys[layer])

    assert cache.offset == positions - 1
    assert mx.array_equal(cache.keys[layer], expected).item()
    assert engine._glm53_mtp_state_only_prefill_tokens == positions - 1

    # A strict prompt extension must append precisely the missing token/hidden
    # pairs and reproduce the one-shot prompt boundary byte-for-byte.
    engine._glm53_mtp_state_only_prefill_tokens = 0
    split_cache = KVCache(layer + 1)
    split_cache.compressed_mla = True
    drafter.prefill(tokens[:3], h_window[:, :3], split_cache)
    drafter.prefill_extension(
        tokens[3:], h_window[:, 2:4], split_cache)
    mx.eval(split_cache.keys[layer])
    assert split_cache.offset == positions - 1
    assert mx.array_equal(split_cache.keys[layer], expected).item()
    assert engine._glm53_mtp_state_only_prefill_tokens == positions - 1


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


def test_glm53_plain_controller_round_needs_no_speculative_kda_factors():
    from runtime.speculative import _glm5_kda_rollback_factors_required

    cfg = SimpleNamespace(model_type="glm5_next")
    assert not _glm5_kda_rollback_factors_required(cfg, 1)
    assert _glm5_kda_rollback_factors_required(cfg, 2)
    assert _glm5_kda_rollback_factors_required(cfg, 6)
    assert not _glm5_kda_rollback_factors_required(
        SimpleNamespace(model_type="qwen3_5"), 2)


def test_glm53_native_mtp_prefix_cache_is_strict_and_layout_bound():
    from runtime.speculative import _native_mtp_strict_prefix_cache

    cache = (((11, 12, 13), False, False), "target", "draft", "logits", "h")
    assert _native_mtp_strict_prefix_cache(
        cache, [11, 12, 13, 14], target_stepped=False,
        draft_stepped=False) is cache
    assert _native_mtp_strict_prefix_cache(
        cache, [11, 12, 13], target_stepped=False,
        draft_stepped=False) is None
    assert _native_mtp_strict_prefix_cache(
        cache, [11, 99, 13, 14], target_stepped=False,
        draft_stepped=False) is None
    assert _native_mtp_strict_prefix_cache(
        cache, [11, 12, 13, 14], target_stepped=True,
        draft_stepped=False) is None
