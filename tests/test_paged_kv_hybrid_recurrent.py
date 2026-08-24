"""Regression gates for hybrid recurrent state on exact paged KV backends."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

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
