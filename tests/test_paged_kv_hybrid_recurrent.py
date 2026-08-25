"""Regression gates for hybrid recurrent state on exact paged KV backends."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import attach_hybrid_recurrent_cache
from runtime.kv_paged import PagedKVCache


def test_qwen_paged_kv_gets_recurrent_companion(tmp_path):
    kv = PagedKVCache(
        num_layers=8,
        max_bytes=1,
        spill_dir=tmp_path,
        page_positions=4,
        resident_pages=0,
    )
    returned = attach_hybrid_recurrent_cache(
        kv, model_type="qwen3_5", num_hidden_layers=8)
    assert returned is kv
    assert kv.kda_cache.state(0) is None

    state = mx.ones((1, 2, 3, 3), dtype=mx.float32)
    history = (mx.zeros((1, 2, 2), dtype=mx.bfloat16),)
    kv.kda_cache.set_state(3, state)
    kv.kda_cache.set_conv_history(3, history)

    # Attention paging does not own or reset the O(1)-sized recurrent state.
    k = mx.ones((1, 2, 4, 3), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 4, 3), dtype=mx.bfloat16)
    kv.update(3, k, v)
    assert kv.kda_cache.state(3) is state
    assert kv.kda_cache.conv_history(3) is history


def test_recurrent_attachment_is_idempotent_and_architecture_scoped():
    existing = SimpleNamespace(kda_cache=object())
    marker = existing.kda_cache
    attach_hybrid_recurrent_cache(
        existing, model_type="qwen3_5_moe", num_hidden_layers=4)
    assert existing.kda_cache is marker

    ordinary = SimpleNamespace()
    attach_hybrid_recurrent_cache(
        ordinary, model_type="qwen2", num_hidden_layers=4)
    assert not hasattr(ordinary, "kda_cache")


def test_paged_kv_per_layer_speculative_rollback_is_exact(tmp_path):
    kv = PagedKVCache(
        num_layers=4,
        max_bytes=1,
        spill_dir=tmp_path,
        page_positions=4,
        resident_pages=0,
    )
    initial = mx.array(
        np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1))
    proposed = mx.array(
        np.arange(6, 9, dtype=np.float32).reshape(1, 1, 3, 1))
    for layer in (1, 3):
        kv.update(layer, initial, initial + 100)
    assert kv.offset == 6
    checkpoint = kv.layer_lengths()
    assert checkpoint == (0, 6, 0, 6)

    for layer in (1, 3):
        kv.update(layer, proposed, proposed + 100)
    assert kv.offset == 9
    assert kv.layer_lengths() == (0, 9, 0, 9)

    # Keep the anchor plus one accepted proposal from a width-three verifier.
    kv.trim_layer_lengths((0, 8, 0, 8))
    assert kv.offset == 8
    assert kv.layer_lengths() == (0, 8, 0, 8)
    expected = np.arange(8, dtype=np.float32)
    for layer in (1, 3):
        keys, values = kv.materialize_layer(layer)
        np.testing.assert_array_equal(np.array(keys).reshape(-1), expected)
        np.testing.assert_array_equal(
            np.array(values).reshape(-1), expected + 100)

    # A second rollback can reopen a spilled full page without changing bits.
    kv.trim(5)
    assert kv.offset == 5
    assert kv.layer_lengths() == (0, 5, 0, 5)
    for layer in (1, 3):
        keys, _values = kv.materialize_layer(layer)
        np.testing.assert_array_equal(
            np.array(keys).reshape(-1), np.arange(5, dtype=np.float32))

    with pytest.raises(ValueError, match="cannot grow"):
        kv.trim_layer_lengths((0, 6, 0, 5))


def test_paged_kv_global_offset_uses_longest_mixed_depth_layer(tmp_path):
    kv = PagedKVCache(
        num_layers=4,
        max_bytes=1_000_000,
        spill_dir=tmp_path,
        page_positions=4,
    )
    full = mx.zeros((1, 1, 6, 1), dtype=mx.bfloat16)
    suffix = mx.zeros((1, 1, 2, 1), dtype=mx.bfloat16)

    kv.update(1, full, full)
    kv.update(3, suffix, suffix)

    assert kv.layer_lengths() == (0, 6, 0, 2)
    assert kv.offset == 6
