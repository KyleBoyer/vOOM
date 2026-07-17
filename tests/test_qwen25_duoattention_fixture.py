"""Model-free semantics for the standalone Qwen DuoAttention gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest


FIXTURE = Path(__file__).parent / "fixtures/qwen25_duoattention_kv_group_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen25_duo_gate", FIXTURE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_partition_and_gqa_query_head_mapping_fail_closed():
    pattern = MODULE.validate_full_groups(((1,), (), (0, 1)), 3, 2)
    assert pattern == ((1,), (), (0, 1))
    assert MODULE.query_heads_for_groups((1,), 6) == tuple(range(6, 12))
    assert MODULE.query_heads_for_groups((0, 1), 6) == tuple(range(12))
    with pytest.raises(ValueError, match="layers"):
        MODULE.validate_full_groups(((0,),), 2, 2)
    with pytest.raises(ValueError, match="repeats"):
        MODULE.query_heads_for_groups((0,), 0)
    with pytest.raises(ValueError, match="out-of-range"):
        MODULE.validate_full_groups(((2,), ()), 2, 2)
    with pytest.raises(ValueError, match="all-dense"):
        MODULE.validate_full_groups(((0, 1), (0, 1)), 2, 2)


def test_stream_positions_keep_unique_sink_and_recent_window():
    view, kept = MODULE.retained_stream_positions((), 0, 10, sink=2, recent=4)
    assert view == tuple(range(10))
    assert kept == (0, 1, 6, 7, 8, 9)
    view, kept = MODULE.retained_stream_positions(
        kept, 10, 3, sink=2, recent=4)
    assert view == (0, 1, 6, 7, 8, 9, 10, 11, 12)
    assert kept == (0, 1, 9, 10, 11, 12)
    with pytest.raises(ValueError, match="chronological"):
        MODULE.retained_stream_positions((0, 2, 1), 3, 1, 1, 2)


def test_duo_cache_views_bytes_clone_and_unsafe_rollback():
    pattern = ((0,),)
    cache = MODULE.DuoKVCache(pattern, num_kv_heads=2, sink=2, recent=3)
    keys = mx.arange(2 * 8 * 4).reshape(1, 2, 8, 4).astype(mx.bfloat16)
    values = keys + 100
    view = cache.update_duo(0, keys, values)
    mx.eval(view.full_keys, view.stream_keys)
    assert view.full_groups == (0,)
    assert view.stream_groups == (1,)
    assert view.stream_positions == tuple(range(8))
    assert cache.stream_positions[0] == (0, 1, 5, 6, 7)
    assert cache.offset == 8
    expected = MODULE.logical_kv_bytes(
        pattern, length=8, num_kv_heads=2, head_dim=4,
        sink=2, recent=3)
    assert cache.nbytes() == expected
    branch = cache.clone_for_branch()
    assert branch.nbytes() == cache.nbytes()
    assert branch.offset == cache.offset
    with pytest.raises(RuntimeError, match="rollback"):
        cache.trim(7)


def test_duo_cache_trim_before_any_eviction_is_supported():
    cache = MODULE.DuoKVCache(((0,),), 2, sink=8, recent=8)
    values = mx.zeros((1, 2, 4, 8), dtype=mx.bfloat16)
    cache.update_duo(0, values, values)
    cache.trim(2)
    assert cache.offset == 2
    assert cache.stream_positions[0] == (0, 1)
    assert cache.nbytes() == MODULE.logical_kv_bytes(
        ((0,),), 2, 2, 8, 8, 8)
