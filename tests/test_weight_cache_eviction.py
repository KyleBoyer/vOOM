"""Policy, accounting, and concurrency gates for WeightCache eviction."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
import random
import threading
import time

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
        self._lock = threading.Lock()

    def fetch(self, names):
        with self._lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        tensors = {name: FakeTensor(name) for name in names}
        return tensors, 0.0, sum(tensor.nbytes for tensor in tensors.values())


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


def test_replacing_a_pinned_page_keeps_byte_counters_exact(monkeypatch):
    monkeypatch.setattr(cache_module, "_clear_device_cache", lambda: None)
    cache = WeightCache(FakeStore(), max_bytes=100)
    cache.pin("shared", ["first"])
    cache.pin("shared", ["second", "third"])

    assert cache.total_bytes == 20
    assert cache.resident_keys == ["shared"]
    assert not cache.would_fit(81)
    assert cache.would_fit(80)


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
