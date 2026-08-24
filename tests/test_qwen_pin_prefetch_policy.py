"""Pure gates for dense-Qwen pin/prefetch accounting and telemetry."""

from types import SimpleNamespace

import pytest

from runtime.engine import (
    StreamingEngine,
    _cache_io_snapshot,
    _record_cache_io_delta,
)


class SizedStore:
    def storage_bytes_unknown(self, _names):
        return False

    def storage_bytes(self, names):
        return 100 * len(names)


def _planner(*, persistent=200, depth=2):
    engine = SimpleNamespace(
        rc=SimpleNamespace(
            pin_trunk_budget_mb=0.0005,
            expert_fetch_batch=0,
            pin_trunk_expert_reserve_mb=0,
            prefetch_depth=depth,
        ),
        cfg=SimpleNamespace(num_experts_per_tok=0, num_experts=0),
        store=SizedStore(),
        cache=SimpleNamespace(max_bytes=1_000, pinned_bytes=persistent),
        _expert_fetch_page_bytes=999,
        _layer_transient_margin=100,
        _trunk_pages=lambda layer: [(f"layer.{layer}", [f"w.{layer}"])],
        planned_trunk_pin_bytes=0,
    )
    count = StreamingEngine._plan_trunk_pin_layers(engine, 8)
    return count, engine


def test_dense_planner_charges_head_bytes_and_prefetch_slots():
    count, engine = _planner(persistent=200, depth=2)
    assert count == 4
    assert engine.planned_trunk_pin_bytes == 400

    count, engine = _planner(persistent=400, depth=2)
    assert count == 2
    assert engine.planned_trunk_pin_bytes == 200

    count, _engine = _planner(persistent=200, depth=0)
    assert count == 5  # requested trunk cap, independent of cache room


def test_request_telemetry_exposes_pin_and_prefetch_effectiveness():
    cache_stats = SimpleNamespace(
        hits=1, misses=2, evictions=3, bytes_read=4,
        pinned_hits=5, prefetch_hits=6, prefetch_waits=7,
        prefetch_wait_s=0.25,
        prefetch_loads=8, prefetch_loaded_bytes=800,
        prefetch_load_s=0.5,
        prefetch_useful_pages=6, prefetch_useful_bytes=600,
        prefetch_useful_load_s=0.4,
        prefetch_wasted_pages=2, prefetch_wasted_bytes=200,
        prefetch_wasted_load_s=0.1,
    )
    engine = SimpleNamespace(
        cache=SimpleNamespace(
            stats=cache_stats, total_bytes=700, max_bytes=1_000,
            pinned_bytes=400, prefetched_bytes=200),
        store=SimpleNamespace(),
        governor=None,
        expert_hits=0,
        expert_misses=0,
        _layer_transient=11,
        _token_transient=12,
        planned_trunk_pin_layers=4,
        planned_trunk_pin_bytes=400,
        rc=SimpleNamespace(prefetch_depth=2),
    )
    before = _cache_io_snapshot(engine)
    cache_stats.pinned_hits += 3
    cache_stats.prefetch_hits += 2
    cache_stats.prefetch_waits += 1
    cache_stats.prefetch_wait_s += 0.125
    cache_stats.prefetch_loads += 4
    cache_stats.prefetch_loaded_bytes += 400
    cache_stats.prefetch_load_s += 0.25
    cache_stats.prefetch_useful_pages += 3
    cache_stats.prefetch_useful_bytes += 300
    cache_stats.prefetch_useful_load_s += 0.2
    cache_stats.prefetch_wasted_pages += 1
    cache_stats.prefetch_wasted_bytes += 100
    cache_stats.prefetch_wasted_load_s += 0.05
    stats = {}
    _record_cache_io_delta(engine, before, stats)

    assert stats["weight_cache_pinned_bytes"] == 400
    assert stats["weight_cache_pinned_hits"] == 3
    assert stats["weight_cache_prefetch_hits"] == 2
    assert stats["weight_prefetch_waits"] == 1
    assert stats["weight_prefetch_wait_s"] == 0.125
    assert stats["weight_prefetch_loads"] == 4
    assert stats["weight_prefetch_loaded_bytes"] == 400
    assert stats["weight_prefetch_useful_pages"] == 3
    assert stats["weight_prefetch_wasted_pages"] == 1
    assert stats["weight_prefetch_hidden_lower_bound_s"] == pytest.approx(0.075)
    assert stats["weight_cache_prefetched_bytes"] == 200
    assert stats["planned_trunk_pin_layers"] == 4
    assert stats["weight_prefetch_depth"] == 2


def test_parallel_tier_snapshot_reports_service_overlap():
    import threading

    from runtime.model_loader import WeightStore

    store = object.__new__(WeightStore)
    store._stage_lock = threading.Lock()
    store.parallel_tier_fetches = 0
    store.parallel_tier_fast_bytes = 0
    store.parallel_tier_archive_bytes = 0
    store.parallel_tier_wall_ns = 0
    store.parallel_tier_fast_service_ns = 0
    store.parallel_tier_archive_service_ns = 0
    store.parallel_tier_hidden_ns = 0

    store._record_parallel_tier(
        fast_bytes=400, archive_bytes=600, wall_ns=700,
        fast_service_ns=500, archive_service_ns=650,
    )

    assert store.parallel_tier_snapshot() == (
        1, 400, 600, 700, 500, 650, 450)
