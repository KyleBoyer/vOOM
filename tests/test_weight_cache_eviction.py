"""Policy, accounting, and concurrency gates for WeightCache eviction."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import random
import threading
import time

import pytest

import runtime.weight_cache as cache_module
from runtime.weight_cache import WeightCache


@dataclass(frozen=True)
class FakeTensor:
    name: str
    nbytes: int = 10


class FakeStore:
    def __init__(self, *, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.released = []
        self._lock = threading.Lock()

    def fetch(self, names):
        with self._lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        tensors = {name: FakeTensor(name) for name in names}
        return tensors, 0.0, sum(tensor.nbytes for tensor in tensors.values())

    def release_cache_pages(self, names):
        self.released.append(tuple(names))


class RecordingWarmTier:
    def __init__(self):
        self.admitted = []

    def take(self, _key):
        return None

    def admit(self, key, _tensors):
        self.admitted.append(key)


def _reference_eviction(pages, frequencies, budget):
    pages = OrderedDict(pages)
    victims = []
    while sum(page.nbytes for page in pages.values()) > budget:
        candidates = [
            (frequencies.get(key, 0), age, key)
            for age, (key, page) in enumerate(pages.items())
            if not page.pinned and page.origin != "prefetch"
        ]
        if not candidates:
            candidates = [
                (frequencies.get(key, 0), age, key)
                for age, (key, page) in enumerate(pages.items())
                if not page.pinned
            ]
        if not candidates:
            break
        victim = min(candidates)[2]
        victims.append(victim)
        del pages[victim]
    return list(pages), victims


def test_one_pass_eviction_preserves_policy_order_and_clears_once(monkeypatch):
    clears = []
    monkeypatch.setattr(cache_module, "_clear_device_cache",
                        lambda: clears.append(True))
    warm = RecordingWarmTier()
    cache = WeightCache(FakeStore(), max_bytes=1_000, warm=warm)
    cache.pin("pin", ["pin.weight"])
    cache.get("prefetch-old", ["prefetch-old.weight"], origin="prefetch")
    cache.get("demand-old", ["demand-old.weight"])
    cache.get("demand-new", ["demand-new.weight"])
    cache.get("prefetch-new", ["prefetch-new.weight"], origin="prefetch")
    cache.freq["demand-old"] = 5
    cache.freq["demand-new"] = 1

    cache.max_bytes = 25
    with cache._lock:
        cache._evict_locked()

    assert warm.admitted == ["demand-new", "demand-old", "prefetch-old"]
    assert cache.resident_keys == ["pin", "prefetch-new"]
    assert cache.total_bytes == 20
    assert cache.stats.evictions == 3
    assert clears == [True]
    assert cache.would_fit(5)
    assert not cache.would_fit(6)


def test_trim_to_releases_lru_pages_without_lowering_future_budget(monkeypatch):
    clears = []
    monkeypatch.setattr(cache_module, "_clear_device_cache",
                        lambda: clears.append(True))
    store = FakeStore()
    cache = WeightCache(store, max_bytes=100)
    cache.pin("pin", ["pin.weight"])
    cache.get("old", ["old.weight"])
    cache.get("new", ["new.weight"])

    released = cache.trim_to(20)

    assert released == 10
    assert cache.resident_keys == ["pin", "new"]
    assert cache.total_bytes == 20
    assert cache.max_bytes == 100
    assert cache.stats.evictions == 1
    assert store.released == [("old.weight",)]
    assert clears == [True]


def test_one_pass_eviction_matches_legacy_policy_across_random_edge_cases(
        monkeypatch):
    clears = []
    monkeypatch.setattr(cache_module, "_clear_device_cache",
                        lambda: clears.append(True))
    for seed in range(100):
        rng = random.Random(seed)
        warm = RecordingWarmTier()
        cache = WeightCache(FakeStore(), max_bytes=rng.randrange(0, 300), warm=warm)
        with cache._lock:
            for index in range(rng.randrange(1, 32)):
                key = f"page-{index}"
                page = cache_module.WeightPage(
                    key,
                    {key: FakeTensor(key, rng.randrange(1, 40))},
                    0,
                    pinned=rng.random() < 0.15,
                    origin="prefetch" if rng.random() < 0.3 else "demand",
                )
                page.nbytes = next(iter(page.tensors.values())).nbytes
                cache._put_page_locked(page)
                cache.freq[key] = rng.randrange(0, 12)
            expected_keys, expected_victims = _reference_eviction(
                cache._pages, cache.freq, cache.max_bytes)
            before_clears = len(clears)
            cache._evict_locked()

        assert cache.resident_keys == expected_keys
        assert warm.admitted == expected_victims
        assert cache.total_bytes == sum(
            page.nbytes for page in cache._pages.values())
        assert len(clears) - before_clears == bool(expected_victims)


def test_prefetch_hit_updates_reserved_accounting_without_changing_total(
        monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=100)
    tensors = cache.get("page", ["page.weight"], origin="prefetch")
    assert not cache.would_fit(95)
    assert cache.total_bytes == 10

    assert cache.get("page", ["page.weight"], origin="demand") is tensors
    assert cache.would_fit(95)
    assert cache.total_bytes == 10
    assert cache.stats.prefetch_hits == 1
    assert cache.stats.prefetch_loads == 1
    assert cache.stats.prefetch_loaded_bytes == 10
    assert cache.stats.prefetch_useful_pages == 1
    assert cache.stats.prefetch_useful_bytes == 10
    assert cache.stats.prefetch_wasted_pages == 0


def test_unused_prefetch_is_counted_as_waste_on_eviction(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=10)
    cache.get("unused", ["unused.weight"], origin="prefetch")

    cache.max_bytes = 0
    cache.trim_to(0)

    assert cache.stats.prefetch_loads == 1
    assert cache.stats.prefetch_useful_pages == 0
    assert cache.stats.prefetch_wasted_pages == 1
    assert cache.stats.prefetch_wasted_bytes == 10


def test_replacing_a_pinned_page_keeps_byte_counters_exact(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=100)
    cache.pin("shared", ["first"])
    cache.pin("shared", ["second", "third"])

    assert cache.total_bytes == 20
    assert cache.resident_keys == ["shared"]
    assert not cache.would_fit(81)
    assert cache.would_fit(80)


def test_pin_fails_closed_before_pinned_pages_exceed_capacity(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=10)
    cache.pin("first", ["first.weight"])

    with pytest.raises(MemoryError, match="weight-cache capacity"):
        cache.pin("second", ["second.weight"])

    assert cache.pinned_bytes == 10
    assert cache.total_bytes == 10
    assert cache.resident_keys == ["first"]


def test_pinned_hits_and_resident_class_bytes_are_exact(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=100)
    cache.pin("pinned", ["pinned.weight"])
    cache.get("prefetched", ["prefetched.weight"], origin="prefetch")

    assert cache.pinned_bytes == 10
    assert cache.prefetched_bytes == 10
    cache.get("pinned", ["pinned.weight"])
    assert cache.stats.pinned_hits == 1
    assert cache.pinned_bytes == 10


def test_demand_wait_on_inflight_prefetch_is_measured(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(delay=0.04), max_bytes=100)
    with ThreadPoolExecutor(max_workers=2) as executor:
        prefetched = executor.submit(
            cache.get, "shared", ["shared.weight"], "prefetch")
        deadline = time.monotonic() + 1.0
        while not cache.inflight("shared") and time.monotonic() < deadline:
            time.sleep(0.001)
        demanded = executor.submit(
            cache.get, "shared", ["shared.weight"], "demand")
        assert demanded.result() is prefetched.result()

    assert cache.stats.prefetch_hits == 1
    assert cache.stats.prefetch_waits == 1
    assert cache.stats.prefetch_wait_s > 0


def test_clear_releases_all_pages_and_reservations_once(monkeypatch):
    clears = []
    monkeypatch.setattr(cache_module, "_clear_device_cache",
                        lambda: clears.append(True))
    cache = WeightCache(FakeStore(), max_bytes=100)
    cache.pin("pinned", ["pinned.weight"])
    cache.get("prefetched", ["prefetched.weight"], origin="prefetch")
    cache.get("demand", ["demand.weight"])

    assert cache.total_bytes == 30
    assert not cache.would_fit(81)

    cache.clear()

    assert cache.total_bytes == 0
    assert cache.resident_keys == []
    assert cache.would_fit(100)
    assert clears == [True]
    assert cache.store.released == [
        (
            "pinned.weight",
            "prefetched.weight",
            "demand.weight",
        )
    ]
    assert cache.stats.prefetch_wasted_pages == 1


def test_clear_refuses_to_race_an_inflight_fetch(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=100)
    with cache._lock:
        cache._inflight["loading"] = threading.Event()

    try:
        cache.clear()
    except RuntimeError as exc:
        assert "fetches are in flight" in str(exc)
    else:
        raise AssertionError("clear must reject an in-flight producer")


def test_discard_retries_store_release_after_budget_evicted_page(monkeypatch):
    clears = []
    monkeypatch.setattr(cache_module, "_clear_device_cache",
                        lambda: clears.append(True))
    cache = WeightCache(FakeStore(), max_bytes=5)
    values = cache.get("layer.0", ["model.layers.0.weight"])

    # The page is pass-through because it exceeds the budget, but the caller
    # still owns the values. The true consumer boundary must retry the store
    # release even though the cache entry is already absent.
    assert values
    assert not cache.contains("layer.0")
    assert cache.discard(
        "layer.0", ["model.layers.0.weight"]
    ) is False
    assert cache.store.released[-1] == ("model.layers.0.weight",)
    assert len(clears) == 2


def test_concurrent_same_key_fetch_keeps_single_page_and_exact_accounting(
        monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    store = FakeStore(delay=0.02)
    cache = WeightCache(store, max_bytes=100)

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(
            lambda _index: cache.get("shared", ["shared.weight"]),
            range(24),
        ))

    assert store.calls == 1
    assert all(result is results[0] for result in results)
    assert cache.total_bytes == 10
    assert cache.resident_keys == ["shared"]
    assert cache.stats.misses == 1
    assert cache.stats.hits == 23
