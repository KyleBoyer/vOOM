"""Exactness and capacity gates for the stepped resident KV cache."""

from __future__ import annotations

import mlx.core as mx
import pytest

import runtime.kv_cache as kv_cache_module
from runtime.kv_cache import (
    KVCache,
    PositionFreeKVCache,
    PositionFreePagePool,
    SteppedKVCache,
)
from runtime.tool_capsules import ReusedRange, ToolPICPlan


def _kv(values) -> tuple[mx.array, mx.array]:
    array = mx.array(list(values), dtype=mx.float32).reshape(1, 1, -1, 1)
    return array, array + 1000


def test_short_decode_uses_exact_length_concatenation():
    cache = KVCache(1)
    keys, values = cache.update(0, *_kv([1, 2, 3]))
    mx.eval(keys, values)

    assert cache.offset == 3
    assert keys.shape[2] == values.shape[2] == 3
    assert cache.keys[0].shape[2] == 3
    assert cache.nbytes() == keys.nbytes + values.nbytes
    assert cache.allocated_nbytes() == cache.nbytes()

    original_buffer = cache.keys[0]
    keys, values = cache.update(0, *_kv([4]))
    mx.eval(keys, values)
    assert cache.keys[0] is not original_buffer
    assert keys.flatten().tolist() == [1, 2, 3, 4]
    assert values.flatten().tolist() == [1001, 1002, 1003, 1004]


def test_crossing_capacity_preserves_prefix_exactly():
    cache = SteppedKVCache(1)
    first = list(range(cache.step))
    cache.update(0, *_kv(first))
    keys, _ = cache.update(0, *_kv([cache.step]))
    mx.eval(keys)

    assert cache.offset == cache.step + 1
    assert cache.keys[0].shape[2] == cache.step * 2
    assert keys.flatten().tolist() == list(range(cache.step + 1))


def test_trim_discards_speculative_tail_and_allows_regrowth():
    cache = SteppedKVCache(1)
    cache.update(0, *_kv(range(cache.step + 4)))
    cache.trim(cache.step + 1)
    assert cache.offset == cache.step + 1
    assert cache.keys[0].shape[2] == cache.step + 1

    keys, _ = cache.update(0, *_kv([999]))
    mx.eval(keys)
    assert cache.offset == cache.step + 2
    assert keys[0, 0, -2:, 0].tolist() == [cache.step, 999]


def test_directly_loaded_exact_arrays_adopt_their_length():
    cache = SteppedKVCache(1)
    keys, values = _kv([7, 8, 9])
    cache.keys[0] = keys
    cache.values[0] = values

    assert cache.offset == 3
    combined, _ = cache.update(0, *_kv([10]))
    mx.eval(combined)
    assert combined.flatten().tolist() == [7, 8, 9, 10]


def test_compressed_mla_keeps_axis_one_semantics():
    cache = KVCache(1)
    cache.compressed_mla = True
    first = mx.arange(6).reshape(1, 2, 3)
    second = mx.arange(3).reshape(1, 1, 3) + 6
    cache.update_latent(0, first)
    combined = cache.update_latent(0, second)
    mx.eval(combined)

    assert cache.offset == 3
    assert combined.reshape(-1).tolist() == list(range(9))


def test_dense_trim_uses_one_barrier_for_every_layer(monkeypatch):
    cache = KVCache(4)
    for layer in range(4):
        keys, values = _kv(range(layer * 10, layer * 10 + 8))
        cache.update(layer, keys, values)
    mx.eval(*[array for pair in zip(cache.keys, cache.values) for array in pair])
    calls = []
    original_eval = kv_cache_module.mx.eval

    def counted_eval(*arrays):
        calls.append(len(arrays))
        return original_eval(*arrays)

    monkeypatch.setattr(kv_cache_module.mx, "eval", counted_eval)
    cache.trim(3)

    assert calls == [8]
    for layer in range(4):
        assert cache.keys[layer].shape[2] == 3
        assert cache.values[layer].shape[2] == 3
        assert cache.keys[layer].flatten().tolist() == list(
            range(layer * 10, layer * 10 + 3))


def test_compressed_and_stepped_trim_each_use_one_barrier(monkeypatch):
    compressed = KVCache(4)
    compressed.compressed_mla = True
    stepped = SteppedKVCache(4)
    for layer in range(4):
        compressed.keys[layer] = mx.arange(16).reshape(1, 8, 2) + layer * 100
        stepped.update(layer, *_kv(range(layer * 10, layer * 10 + 8)))
    mx.eval(*[array for array in compressed.keys if array is not None])
    mx.eval(*[array for pair in zip(stepped.keys, stepped.values) for array in pair])
    calls = []
    original_eval = kv_cache_module.mx.eval

    def counted_eval(*arrays):
        calls.append(len(arrays))
        return original_eval(*arrays)

    monkeypatch.setattr(kv_cache_module.mx, "eval", counted_eval)
    compressed.trim(3)
    stepped.trim(3)

    assert calls == [4, 8]
    assert all(array.shape[1] == 3 for array in compressed.keys)
    assert all(stepped._layer_length(layer) == 3 for layer in range(4))


def _position_free_values(layer: int, width: int, dim: int = 32):
    keys = (
        mx.arange(width * dim).reshape(1, 1, width, dim)
        + layer * 10_000
    ).astype(mx.float32)
    return keys, keys + 1000


def test_position_free_pic_shares_pages_and_releases_by_reference():
    pool = PositionFreePagePool(2, 1, 32, min_capacity=8)
    source = PositionFreeKVCache(pool)
    for layer in range(2):
        source.update_unrotated(layer, *_position_free_values(layer, 4))
    assert source.is_complete
    assert pool.live_pages == 4

    plan = ToolPICPlan(
        reused=(
            ReusedRange(0, 1, 0, "tool_capsule"),
            ReusedRange(2, 4, 1, "tool_capsule"),
        ),
        selected_positions=(1, 4),
        exact_prefix_tokens=0,
        capsule_tokens_reused=3,
        capsule_tokens_repaired=0,
    )
    destination = PositionFreeKVCache.from_pic_plan(source, plan, 5)
    # Four source pages plus only two newly selected pages: logical ownership is
    # 4+5 positions, but physical storage contains six unique positions.
    assert pool.live_pages == 6
    assert destination.page_ids[0] == source.page_ids[0]
    assert destination.page_ids[2:4] == source.page_ids[1:3]
    for page_id in destination.page_ids[0], *destination.page_ids[2:4]:
        assert pool.reference_count(page_id) == 2

    selected = (1, 4)
    for layer in range(2):
        keys, values = _position_free_values(layer + 10, len(selected))
        destination.write_selected(layer, selected, keys, values)
    assert destination.is_complete

    source.release()
    assert pool.live_pages == 5
    assert all(pool.reference_count(page_id) == 1
               for page_id in destination.page_ids)
    gathered, _ = destination.gather_unrotated(1)
    mx.eval(gathered)
    assert float(gathered[0, 0, 0, 0]) == 10_000
    assert float(gathered[0, 0, 2, 0]) == 10_032

    destination.release()
    assert pool.live_pages == 0
    assert pool.free_pages == 6


def test_position_free_trim_recycles_only_unreferenced_tail_pages():
    pool = PositionFreePagePool(1, 1, 32, min_capacity=4)
    source = PositionFreeKVCache(pool)
    source.update_unrotated(0, *_position_free_values(0, 4))
    shared_ids = source.page_ids[:2]
    pool.retain(shared_ids)

    source.trim(2)
    assert pool.live_pages == 2
    assert all(pool.reference_count(page_id) == 2 for page_id in shared_ids)
    source.release()
    assert all(pool.reference_count(page_id) == 1 for page_id in shared_ids)
    pool.release(shared_ids)
    assert pool.live_pages == 0


def test_position_free_live_counter_tracks_reference_transitions():
    pool = PositionFreePagePool(1, 1, 32, min_capacity=8)
    cache = PositionFreeKVCache(pool)
    cache.update_unrotated(0, *_position_free_values(0, 5))
    assert pool.live_pages == 5
    assert pool.live_pages == sum(
        reference > 0 for reference in pool._refs[:pool._next_id])

    shared = cache.page_ids[1:4]
    pool.retain(shared)
    assert pool.live_pages == 5
    pool.release(shared)
    assert pool.live_pages == 5

    cache.trim(2)
    assert pool.live_pages == 2
    cache.release()
    assert pool.live_pages == 0
    pool.close()
    assert pool.live_pages == 0


def test_position_free_failed_plan_restores_live_counter():
    pool = PositionFreePagePool(1, 1, 32, min_capacity=8)
    source = PositionFreeKVCache(pool)
    source.update_unrotated(0, *_position_free_values(0, 2))
    invalid = ToolPICPlan(
        reused=(),
        selected_positions=(0, 0),
        exact_prefix_tokens=0,
        capsule_tokens_reused=0,
        capsule_tokens_repaired=2,
    )

    with pytest.raises(ValueError, match="invalid PIC selected position"):
        PositionFreeKVCache.from_pic_plan(source, invalid, 2)
    assert pool.live_pages == 2
    assert pool.free_pages == 2


def test_position_free_rotated_view_trim_uses_one_barrier(monkeypatch):
    pool = PositionFreePagePool(2, 1, 32, min_capacity=8)
    cache = PositionFreeKVCache(pool)
    for layer in range(2):
        keys, values = _position_free_values(layer, 4)
        cache.update_unrotated(layer, keys, values)
        cache.set_rotated_view(layer, keys, values)
    mx.eval(*[
        array for array in (*pool.key_pages, *pool.value_pages)
        if array is not None
    ])
    calls = []
    original_eval = kv_cache_module.mx.eval

    def counted_eval(*arrays):
        calls.append(len(arrays))
        return original_eval(*arrays)

    monkeypatch.setattr(kv_cache_module.mx, "eval", counted_eval)
    cache.trim(2)

    assert calls == [4]
    assert cache.offset == pool.live_pages == 2
    assert cache._rotated_view.offset == 2


def test_position_free_attention_matches_dense_prefill_and_decode():
    from types import SimpleNamespace

    from runtime import layer_runner

    mx.random.seed(741)
    hidden = 64
    cfg = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
    )
    prefix = "model.layers.0"
    weights = {
        f"{prefix}.self_attn.q_proj.weight": (
            mx.random.normal((hidden, hidden)) * 0.03).astype(mx.bfloat16),
        f"{prefix}.self_attn.k_proj.weight": (
            mx.random.normal((32, hidden)) * 0.03).astype(mx.bfloat16),
        f"{prefix}.self_attn.v_proj.weight": (
            mx.random.normal((32, hidden)) * 0.03).astype(mx.bfloat16),
        f"{prefix}.self_attn.o_proj.weight": (
            mx.random.normal((hidden, hidden)) * 0.03).astype(mx.bfloat16),
    }
    dense = KVCache(1)
    shared = PositionFreeKVCache(
        PositionFreePagePool(1, 1, 32, min_capacity=16))
    prefill = (mx.random.normal((1, 7, hidden)) * 0.1).astype(mx.bfloat16)
    dense_prefill = layer_runner._attention(
        prefill, weights, prefix, cfg, dense, 0, 0)
    shared_prefill = layer_runner._attention(
        prefill, weights, prefix, cfg, shared, 0, 0)
    mx.eval(dense_prefill, shared_prefill)
    assert mx.array_equal(dense_prefill, shared_prefill)

    token = (mx.random.normal((1, 1, hidden)) * 0.1).astype(mx.bfloat16)
    dense_decode = layer_runner._attention(
        token, weights, prefix, cfg, dense, 0, 7)
    shared_decode = layer_runner._attention(
        token, weights, prefix, cfg, shared, 0, 7)
    mx.eval(dense_decode, shared_decode)
    assert mx.allclose(shared_decode, dense_decode, rtol=2e-2, atol=2e-2)
    assert shared.offset == dense.offset == 8

    # Above the hybrid threshold the gathered prefill view is retained only for
    # this request, so decode takes MLX's ordinary pre-rotated SDPA path.
    dense_view = KVCache(1)
    shared_view = PositionFreeKVCache(
        PositionFreePagePool(1, 1, 32, min_capacity=16))
    shared_view.rotated_view_min_keys = 4
    dense_prefill = layer_runner._attention(
        prefill, weights, prefix, cfg, dense_view, 0, 0)
    shared_prefill = layer_runner._attention(
        prefill, weights, prefix, cfg, shared_view, 0, 0)
    assert shared_view.rotated_view_nbytes() > 0
    dense_decode = layer_runner._attention(
        token, weights, prefix, cfg, dense_view, 0, 7)
    shared_decode = layer_runner._attention(
        token, weights, prefix, cfg, shared_view, 0, 7)
    mx.eval(dense_prefill, shared_prefill, dense_decode, shared_decode)
    assert mx.array_equal(shared_prefill, dense_prefill)
    assert mx.array_equal(shared_decode, dense_decode)
    logical_and_view = shared_view.nbytes()
    shared_view.drop_rotated_view()
    assert shared_view.nbytes() < logical_and_view

    # Static YaRN supplies explicit denominators and scales Q/K before RoPE.
    # The shared cache must store the scaled-but-unrotated key and reproduce the
    # same decode result when the custom kernel applies those denominators.
    freqs = 10_000.0 ** (
        mx.arange(16, dtype=mx.float32) / 16.0)
    dense_yarn = KVCache(1)
    shared_yarn = PositionFreeKVCache(
        PositionFreePagePool(1, 1, 32, min_capacity=16))
    layer_runner._attention(
        prefill, weights, prefix, cfg, dense_yarn, 0, 0,
        rope_freqs=freqs, rope_mscale=1.1)
    layer_runner._attention(
        prefill, weights, prefix, cfg, shared_yarn, 0, 0,
        rope_freqs=freqs, rope_mscale=1.1)
    dense_decode = layer_runner._attention(
        token, weights, prefix, cfg, dense_yarn, 0, 7,
        rope_freqs=freqs, rope_mscale=1.1)
    shared_decode = layer_runner._attention(
        token, weights, prefix, cfg, shared_yarn, 0, 7,
        rope_freqs=freqs, rope_mscale=1.1)
    mx.eval(dense_decode, shared_decode)
    assert mx.allclose(shared_decode, dense_decode, rtol=2e-2, atol=2e-2)
