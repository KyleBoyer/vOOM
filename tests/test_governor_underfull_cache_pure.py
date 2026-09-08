"""Measured Qwen admission geometry, with fake Metal and no model imports."""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tests.test_governor_reserve_pure import load_pressure, make_governor


def measured_governor():
    module, mx = load_pressure(1_642_572_264)
    gov = make_governor(module, mx, cache_max=2_200_000_000, floor=68_800_000)
    gov.cache.total_bytes = 1_617_699_456
    gov.critical = 5_600_000_000
    gov.metal_limit = 8_500_000_000
    released = [0]
    steps = []

    def evict():
        before = gov.cache.total_bytes
        gov.cache.total_bytes = min(before, gov.cache.max_bytes)
        delta = before - gov.cache.total_bytes
        released[0] += delta
        mx.active -= delta
        steps.append((gov.cache.max_bytes, delta))

    gov.cache._evict_locked = evict
    module.psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
        available=6_016_663_552 + released[0]))
    module.time = SimpleNamespace(sleep=lambda _: None)
    return gov, mx, steps, released


@pytest.mark.parametrize('reason', ['qwen-prefill-layer-page',
    'serial-verify-layer-page', 'qwen4-phase-lm-head', 'glm53-expert-page'])
def test_underfull_budget_does_not_mean_cache_is_unreclaimable(reason):
    gov, mx, steps, released = measured_governor()
    gov.reserve(213_873_879, margin=400_000_000, reason=reason)
    assert steps[0] == (1_870_000_000, 0)  # still ABOVE actual resident bytes
    assert released[0] == 266_624_456
    assert gov.reservations == 3
    assert gov.reservation_zero_release_short_circuits == 0
    assert gov.reservation_failures == 0
    assert gov.cache.max_bytes == 2_200_000_000 - released[0]
    assert mx.active + 213_873_879 + 400_000_000 <= gov.current_ceiling()
    assert gov.critical == 5_600_000_000 and gov.floor == 68_800_000
    assert gov.prefetcher.paused


@pytest.mark.parametrize('unknown', [None, 0, 'unknown', 'raises'])
def test_unknown_reclaimability_cannot_short_circuit_eviction(unknown):
    gov, _mx, _steps, released = measured_governor()
    def probe(_floor):
        if unknown == 'raises':
            raise RuntimeError('unavailable cache metadata')
        return unknown
    gov.cache.can_reclaim_to = probe
    gov.reserve(213_873_879, margin=400_000_000, reason='qwen-prefill-layer-page')
    assert released[0] == 266_624_456
    assert gov.reservation_zero_release_short_circuits == 0


def test_confirmed_pinned_only_cache_keeps_one_step_debounce_and_refuses():
    gov, mx, _steps, _released = measured_governor()
    gov.cache.can_reclaim_to = lambda floor: False
    gov.cache._evict_locked = lambda: None
    with pytest.raises(MemoryError, match='unsafe Metal reservation'):
        gov.reserve(213_873_879, margin=400_000_000, reason='qwen-prefill-layer-page')
    assert gov.reservations == 1
    assert gov.reservation_zero_release_short_circuits == 1
    assert gov.reservation_cache_released_bytes == 0
    assert gov.cache.max_bytes == 2_200_000_000
    assert mx.active == 1_642_572_264


def test_unknown_unreclaimable_cache_still_stops_at_floor_and_refuses():
    gov, _mx, _steps, _released = measured_governor()
    budgets = []
    gov.cache.can_reclaim_to = lambda floor: None
    gov.cache._evict_locked = lambda: budgets.append(gov.cache.max_bytes)
    with pytest.raises(MemoryError, match='unsafe Metal reservation'):
        gov.reserve(213_873_879, margin=400_000_000, reason='qwen-prefill-layer-page')
    assert gov.floor in budgets and min(budgets) == gov.floor
    assert 1 < gov.reservations < 40
    assert gov.reservation_zero_release_short_circuits == 0
    assert gov.reservation_cache_released_bytes == 0
    assert gov.cache.max_bytes == 2_200_000_000


def test_underfull_reclaim_never_restores_concurrent_pressure_reduction():
    gov, _mx, _steps, released = measured_governor()
    original = gov.cache._evict_locked
    def evict():
        original()
        gov.shrinks += 1
    gov.cache._evict_locked = evict
    gov.reserve(213_873_879, margin=400_000_000, reason='qwen-prefill-layer-page')
    assert released[0] == 266_624_456
    assert gov.cache.max_bytes == 1_351_075_000
    assert gov.reservation_budget_restored_bytes == 0


def test_safe_reservation_does_not_query_reclaimability():
    gov, _mx, steps, _released = measured_governor()
    gov.cache.can_reclaim_to = lambda _: pytest.fail('fast path must not probe')
    gov.reserve(1, margin=0, reason='qwen-prefill-layer-page')
    assert not steps and gov.reservation_fast_path_calls == 1


@pytest.mark.parametrize('resident', [None, -1, True, 1.0, '0'])
def test_unknown_fallback_residency_does_not_claim_empty(resident):
    gov, _mx, _steps, _released = measured_governor()
    gov.cache.total_bytes = resident
    assert gov._cache_can_reclaim_to_floor() is True


def actual_cache_probe():
    """Execute the real small cache method without importing the MLX module."""
    path = Path(__file__).resolve().parents[1] / 'runtime/weight_cache.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WeightCache')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'can_reclaim_to')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['can_reclaim_to']


@pytest.mark.parametrize('resident,floor,pins,expected', [
    (0, 10, [], False), (10, 10, [False], False), (9, 10, [False], False),
    (100, 10, [True], False), (100, 10, [False], True),
    (100, 10, [True, False], True), (100, 10, [False, True], True)])
def test_cache_probe_uses_residency_floor_and_real_pin_flags(resident, floor, pins, expected):
    class Cache:
        _lock = threading.Lock()
        _total_bytes = resident
        _pages = {str(i): SimpleNamespace(pinned=p) for i, p in enumerate(pins)}
        @property
        def total_bytes(self):
            raise AssertionError('must not acquire a nested lock through total_bytes')
    cache = Cache()
    class GuardedPages(dict):
        def values(self):
            assert cache._lock.locked()
            return super().values()
    cache._pages = GuardedPages(cache._pages)
    original = dict(cache._pages)
    assert actual_cache_probe()(cache, floor) is expected
    assert cache._total_bytes == resident and cache._pages == original
    assert not cache._lock.locked()
