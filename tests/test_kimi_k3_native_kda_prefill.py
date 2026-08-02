"""Algebraic and numerical gates for K3's fused multi-position KDA scan."""

import mlx.core as mx
import numpy as np
import pytest
from types import SimpleNamespace

from runtime.engine import (
    RuntimeConfig,
    StreamingEngine,
    _recurring_layer_transient_reserve_margin,
)
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import SteppedKVCache
from runtime.kimi_linear import (
    DiskBackedAttnResSnapshots,
    _compiled_kda_prefill_scan,
    _kimi_moe_output,
    _native_fused_kda_prefill_scan,
    attn_res_wrap_layer,
    attn_res_wrap_layer_streamed,
)


def _reference(q, k, v, gate, beta, state):
    outputs = []
    for position in range(q.shape[1]):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * mx.exp(gate[:, position])[..., None]
        predicted = mx.sum(k_t[..., None] * state, axis=-2)
        residual = v_t - predicted
        state = state + (
            beta[:, position, :, None] * k_t
        )[..., None] * residual[..., None, :]
        outputs.append(mx.sum(q_t[..., None] * state, axis=-2))
    return mx.stack(outputs, axis=1), state


@pytest.mark.parametrize("length", [2, 17, 64])
def test_native_kda_prefill_matches_serial_recurrence(length):
    rng = np.random.default_rng(151 + length)
    batch, heads, dim = 1, 3, 16
    q = mx.array(rng.standard_normal(
        (batch, length, heads, dim), dtype=np.float32))
    k = mx.array(rng.standard_normal(
        (batch, length, heads, dim), dtype=np.float32))
    v = mx.array(rng.standard_normal(
        (batch, length, heads, dim), dtype=np.float32))
    gate = mx.array(rng.uniform(
        -5.0, -0.001, (batch, length, heads, dim)).astype(np.float32))
    beta = mx.array(rng.uniform(
        0.01, 0.99, (batch, length, heads)).astype(np.float32))
    state = mx.array(rng.standard_normal(
        (batch, heads, dim, dim), dtype=np.float32) * 0.05)

    reference_out, reference_state = _reference(
        q, k, v, gate, beta, state)
    fused_out, fused_state = _native_fused_kda_prefill_scan(
        q, k, v, gate, beta, state)
    mx.eval(reference_out, reference_state, fused_out, fused_state)

    output_error = float(mx.max(mx.abs(fused_out - reference_out)))
    state_error = float(mx.max(mx.abs(fused_state - reference_state)))
    assert output_error < 2e-4
    assert state_error < 2e-4


def test_native_kda_prefill_rejects_single_position():
    value = mx.zeros((1, 1, 2, 8), dtype=mx.float32)
    beta = mx.zeros((1, 1, 2), dtype=mx.float32)
    state = mx.zeros((1, 2, 8, 8), dtype=mx.float32)
    with pytest.raises(ValueError, match="more than one position"):
        _native_fused_kda_prefill_scan(
            value, value, value, value, beta, state)


@pytest.mark.parametrize("length", [2, 33, 64])
def test_compiled_kda_prefill_is_byte_identical(length):
    rng = np.random.default_rng(311 + length)
    shape = (1, length, 2, 16)
    q = mx.array(rng.standard_normal(shape, dtype=np.float32))
    k = mx.array(rng.standard_normal(shape, dtype=np.float32))
    v = mx.array(rng.standard_normal(shape, dtype=np.float32))
    gate = mx.array(rng.uniform(-5, 0, shape).astype(np.float32))
    beta = mx.array(rng.uniform(
        0, 1, (1, length, 2)).astype(np.float32))
    state = mx.array(rng.standard_normal(
        (1, 2, 16, 16), dtype=np.float32))
    reference_out, reference_state = _reference(
        q, k, v, gate, beta, state)
    compiled_out, compiled_state = _compiled_kda_prefill_scan(
        q, k, v, gate, beta, state)
    mx.eval(reference_out, reference_state, compiled_out, compiled_state)
    assert bool(mx.array_equal(compiled_out, reference_out))
    assert bool(mx.array_equal(compiled_state, reference_state))


def test_kda_prefill_modes_are_mutually_exclusive_before_model_load():
    with pytest.raises(ValueError, match="mutually exclusive"):
        StreamingEngine(
            "unused",
            RuntimeConfig(
                kimi_k3_compiled_kda_prefill=True,
                kimi_k3_native_fused_kda_prefill=True,
            ),
        )


def test_disk_attnres_snapshot_round_trips_exact_bf16_tiles(tmp_path):
    rng = np.random.default_rng(419)
    value = mx.array(
        rng.standard_normal((11, 16), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    store = DiskBackedAttnResSnapshots(tmp_path, write_tile_rows=3)
    directory = store.directory
    store.append(value)
    loaded = store[0][2:9]
    mx.eval(loaded)
    assert bool(mx.array_equal(loaded, value[2:9]))
    stats = store.stats()
    assert stats.pop("uncached_descriptors") >= 0
    assert stats == {
        "snapshots": 1,
        "bytes_written": 11 * 16 * 2,
        "bytes_read": 7 * 16 * 2,
        "write_calls": 4,
        "read_calls": 1,
    }
    store.close()
    assert not directory.exists()


def test_disk_attnres_packs_multiple_snapshots_by_row(tmp_path):
    rng = np.random.default_rng(421)
    first = mx.array(
        rng.standard_normal((11, 16), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    second = mx.array(
        rng.standard_normal((11, 16), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    store = DiskBackedAttnResSnapshots(tmp_path, write_tile_rows=3)
    store.append(first)
    store.append(second)
    stacked = store.read_stacked(2, 9)
    mx.eval(stacked)
    expected = mx.stack([first[2:9], second[2:9]], axis=1)
    mx.eval(expected)
    assert bool(mx.array_equal(stacked, expected))
    assert store._packed_path.stat().st_size == 11 * 2 * 16 * 2
    stats = store.stats()
    assert stats.pop("uncached_descriptors") >= 0
    assert stats == {
        "snapshots": 2,
        "bytes_written": (11 * 16 * 2) + (11 * 2 * 16 * 2),
        "bytes_read": (11 * 16 * 2) + (7 * 2 * 16 * 2),
        "write_calls": 8,
        "read_calls": 5,
    }
    store.close()


def test_disk_attnres_bounds_repack_files_with_exact_packed_groups(tmp_path):
    rng = np.random.default_rng(423)
    values = [
        mx.array(
            rng.standard_normal((11, 16), dtype=np.float32),
            dtype=mx.bfloat16,
        )
        for _ in range(6)
    ]
    store = DiskBackedAttnResSnapshots(
        tmp_path, write_tile_rows=3, group_size=4)
    for value in values:
        store.append(value)
    stacked = store.read_stacked(2, 9)
    mx.eval(stacked)
    expected = mx.stack([value[2:9] for value in values], axis=1)
    mx.eval(expected)
    assert bool(mx.array_equal(stacked, expected))
    paths = sorted(store.directory.glob("snapshots-*.bf16"))
    assert [path.stat().st_size for path in paths] == [
        11 * 4 * 16 * 2,
        11 * 2 * 16 * 2,
    ]
    assert max(path.stat().st_size for path in paths) <= 11 * 4 * 16 * 2
    store.close()


def test_attnres_spill_requires_fused_tiling_before_model_load():
    with pytest.raises(ValueError, match="requires fused AttnRes"):
        StreamingEngine(
            "unused",
            RuntimeConfig(kimi_k3_attnres_spill_dir="scratch"),
        )


def test_mla_kv_spill_requires_compressed_mla_before_model_load():
    with pytest.raises(ValueError, match="requires compressed MLA"):
        StreamingEngine(
            "unused",
            RuntimeConfig(kimi_k3_mla_kv_spill_dir="scratch"),
        )


def test_streamed_disk_attnres_matches_ordinary_tiled_wrapper(tmp_path):
    rng = np.random.default_rng(431)
    cfg = SimpleNamespace(attn_res_block_size=2, rms_norm_eps=1e-6)
    reference = mx.array(
        rng.standard_normal((1, 11, 16), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    streamed = reference
    reference_residuals = []
    disk_residuals = DiskBackedAttnResSnapshots(
        tmp_path, write_tile_rows=3)

    for layer in range(3):
        prefix = f"model.layers.{layer}"
        w = {
            f"{prefix}.self_attention_res_proj.weight": mx.array(
                rng.standard_normal((1, 16), dtype=np.float32),
                dtype=mx.bfloat16,
            ),
            f"{prefix}.self_attention_res_norm.weight": mx.array(
                rng.standard_normal(16, dtype=np.float32),
                dtype=mx.bfloat16,
            ),
            f"{prefix}.input_layernorm.weight": mx.array(
                rng.standard_normal(16, dtype=np.float32),
                dtype=mx.bfloat16,
            ),
            f"{prefix}.mlp_res_proj.weight": mx.array(
                rng.standard_normal((1, 16), dtype=np.float32),
                dtype=mx.bfloat16,
            ),
            f"{prefix}.mlp_res_norm.weight": mx.array(
                rng.standard_normal(16, dtype=np.float32),
                dtype=mx.bfloat16,
            ),
            f"{prefix}.post_attention_layernorm.weight": mx.array(
                rng.standard_normal(16, dtype=np.float32),
                dtype=mx.bfloat16,
            ),
        }

        def attention(value):
            return (value.astype(mx.float32) * 0.125).astype(value.dtype)

        def mlp(value):
            return (value.astype(mx.float32) * -0.0625).astype(value.dtype)

        reference, reference_residuals = attn_res_wrap_layer(
            reference,
            reference_residuals,
            w,
            prefix,
            cfg,
            layer,
            attention,
            mlp,
            fused_tile_size=3,
        )
        streamed, disk_residuals = attn_res_wrap_layer_streamed(
            streamed,
            disk_residuals,
            w,
            prefix,
            cfg,
            layer,
            lambda value, _start, _end: attention(value),
            mlp,
            tile_size=3,
            fused_tile_size=3,
        )
        mx.eval(reference, streamed)
        assert bool(mx.array_equal(streamed, reference))

    disk_residuals.close()


def test_recurring_transient_margin_retires_after_one_observation():
    assert _recurring_layer_transient_reserve_margin(46107, 0) == 400_000_000
    assert _recurring_layer_transient_reserve_margin(46107, 1) == 0
    assert _recurring_layer_transient_reserve_margin(46107, 2) == 0
    assert _recurring_layer_transient_reserve_margin(46107, 20) == 0
    with pytest.raises(ValueError, match="non-negative"):
        _recurring_layer_transient_reserve_margin(46107, -1)


def test_kda_state_spill_round_trips_exact_state_and_conv(tmp_path):
    rng = np.random.default_rng(443)
    state = mx.array(
        rng.standard_normal((1, 3, 8, 8), dtype=np.float32))
    conv = tuple(
        mx.array(
            rng.standard_normal((1, 2, 24), dtype=np.float32),
            dtype=mx.bfloat16,
        )
        for _ in range(3)
    )
    cache = KDAStateCache(2)
    cache.enable_disk_spill(tmp_path)
    cache.set_state(1, state)
    cache.set_conv_history(1, conv)
    assert cache.spill_layer(1)
    assert cache.nbytes() == 0
    restored_state = cache.state(1)
    restored_conv = cache.conv_history(1)
    mx.eval(restored_state, *restored_conv)
    assert bool(mx.array_equal(restored_state, state))
    assert all(
        bool(mx.array_equal(actual, expected))
        for actual, expected in zip(restored_conv, conv)
    )
    stats = cache.spill_stats()
    assert stats["layers"] == 1
    assert stats["reloads"] == 1
    assert stats["bytes_written"] == stats["bytes_read"]
    assert stats["resident_bytes"] == state.nbytes + sum(
        value.nbytes for value in conv)


def test_compressed_mla_spill_round_trips_exact_capacity_and_decode(tmp_path):
    rng = np.random.default_rng(449)
    prompt = mx.array(
        rng.standard_normal((1, 263, 24), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    decode = mx.array(
        rng.standard_normal((1, 1, 24), dtype=np.float32),
        dtype=mx.bfloat16,
    )
    reference = SteppedKVCache(4)
    reference.compressed_mla = True
    candidate = SteppedKVCache(4)
    candidate.compressed_mla = True
    candidate.enable_latent_disk_spill(tmp_path)

    reference.update_latent(3, prompt)
    candidate.update_latent(3, prompt)
    assert candidate.spill_latent_layer(3)
    assert candidate.keys[3] is None
    assert candidate.offset == 263
    expected = reference.update_latent(3, decode)
    actual = candidate.update_latent(3, decode)
    mx.eval(expected, actual)
    assert bool(mx.array_equal(actual, expected))
    assert candidate.keys[3].shape == reference.keys[3].shape
    stats = candidate.latent_spill_stats()
    assert stats["layers"] == 1
    assert stats["reloads"] == 1
    assert stats["bytes_written"] == stats["bytes_read"]
    candidate.close_latent_spill()


def test_k3_latent_moe_tiled_output_matches_full_materialization():
    rng = np.random.default_rng(457)
    hidden, latent, intermediate, experts = 8, 4, 6, 3
    prefix = "model.layers.1"
    moe = f"{prefix}.block_sparse_moe"

    def bf16(shape):
        return mx.array(
            rng.standard_normal(shape, dtype=np.float32) * 0.125,
            dtype=mx.bfloat16,
        )

    h = bf16((1, 7, hidden))
    w = {
        f"{moe}.gate.weight": bf16((experts, hidden)),
        f"{moe}.gate.e_score_correction_bias": mx.zeros(
            (experts,), dtype=mx.float32),
        f"{moe}.routed_expert_down_proj.weight": bf16((latent, hidden)),
        f"{moe}.routed_expert_norm.weight": bf16((latent,)),
        f"{moe}.routed_expert_up_proj.weight": bf16((hidden, latent)),
        f"{moe}.shared_experts.gate_proj.weight": bf16(
            (intermediate, hidden)),
        f"{moe}.shared_experts.up_proj.weight": bf16(
            (intermediate, hidden)),
        f"{moe}.shared_experts.down_proj.weight": bf16(
            (hidden, intermediate)),
    }
    expert_weights = {}
    for expert in range(experts):
        expert_prefix = f"{moe}.experts.{expert}"
        expert_weights[expert] = {
            f"{expert_prefix}.w1.weight": bf16((intermediate, latent)),
            f"{expert_prefix}.w3.weight": bf16((intermediate, latent)),
            f"{expert_prefix}.w2.weight": bf16((latent, intermediate)),
        }
    cfg = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=2,
        num_experts=experts,
        expert_prune_masks={},
        expert_top_k_by_layer=(),
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_latent_hidden_size=latent,
        moe_latent_use_norm=True,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        hidden_size=hidden,
    )

    def get_experts(_layer, expert_ids, **_kwargs):
        return {expert: expert_weights[expert] for expert in expert_ids}

    reference = _kimi_moe_output(
        h, w, prefix, cfg, 1, get_experts, shared_tile_size=0)
    tiled = _kimi_moe_output(
        h, w, prefix, cfg, 1, get_experts, shared_tile_size=2)
    mx.eval(reference, tiled)
    assert bool(mx.array_equal(tiled, reference))
