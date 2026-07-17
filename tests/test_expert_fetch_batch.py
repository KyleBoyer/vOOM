"""F74: WeightCache.get_many() must not materialize its whole missing-expert
union in a single _fetch()/mx.eval() call. Real-GLM incident (2026-07-14):
with 256 routed experts/layer and 8 active/token, even a small prefill chunk's
expert union can approach the full 256 on a cold-cache layer (coupon-collector
effect), and the old code fetched the ENTIRE missing set in one store.fetch()
call before any eviction ran. expert_fetch_batch sub-batches that call so no
single fetch ever exceeds the configured batch size, independent of how large
the total missing union is.
"""
import sys
import gc
import weakref
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runtime.weight_cache as weight_cache_module
from runtime.weight_cache import WeightCache
from runtime.expert_batching import consume_expert_batches


@dataclass(frozen=True)
class FakeTensor:
    """Minimal tensor protocol used by WeightCache: only nbytes is required."""

    name: str
    nbytes: int = 4096


class FakeStore:
    """Records the size of every fetch() call so tests can assert on it."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def fetch(self, names: list[str]):
        self.calls.append(list(names))
        tensors = {n: FakeTensor(n) for n in names}
        nbytes = sum(t.nbytes for t in tensors.values())
        return tensors, 0.0, nbytes


class FailOnceStore(FakeStore):
    """Raises on its first fetch, then behaves normally for retry coverage."""

    def __init__(self):
        super().__init__()
        self.fail = True

    def fetch(self, names: list[str]):
        self.calls.append(list(names))
        if self.fail:
            self.fail = False
            raise OSError("injected first-batch read failure")
        tensors = {n: FakeTensor(n) for n in names}
        nbytes = sum(t.nbytes for t in tensors.values())
        return tensors, 0.0, nbytes


def make_items(n: int) -> list[tuple[str, list[str]]]:
    return [(f"expert.{i}", [f"expert.{i}.w"]) for i in range(n)]


def test_zero_batch_size_preserves_single_fetch_call():
    store = FakeStore()
    cache = WeightCache(store, max_bytes=10_000_000, max_fetch_batch=0)
    items = make_items(5)
    result = cache.get_many(items)
    assert len(store.calls) == 1
    assert len(store.calls[0]) == 5
    assert set(result.keys()) == {k for k, _ in items}


def test_sub_batching_bounds_each_fetch_call_size():
    store = FakeStore()
    cache = WeightCache(store, max_bytes=10_000_000, max_fetch_batch=2)
    items = make_items(5)
    result = cache.get_many(items)
    assert len(store.calls) == 3  # ceil(5/2)
    assert all(len(call) <= 2 for call in store.calls)
    assert sum(len(call) for call in store.calls) == 5
    assert set(result.keys()) == {k for k, _ in items}


def test_sub_batching_produces_identical_tensors_to_unbounded():
    items = make_items(7)

    store_a = FakeStore()
    cache_a = WeightCache(store_a, max_bytes=10_000_000, max_fetch_batch=0)
    result_a = cache_a.get_many(items)

    store_b = FakeStore()
    cache_b = WeightCache(store_b, max_bytes=10_000_000, max_fetch_batch=3)
    result_b = cache_b.get_many(items)

    for key, _ in items:
        tensors_a = result_a[key]
        tensors_b = result_b[key]
        for name in tensors_a:
            assert tensors_a[name] == tensors_b[name]


def test_sub_batching_runs_cache_eviction_between_fetches():
    # Each expert's tensor is 4KB and the cache budget fits ~2.5 pages.  This
    # proves the LRU's *accounted cache* remains bounded between fetches.  It
    # deliberately does not claim a process-residency bound: get_many()'s
    # returned result still owns every page until the consumer releases it.
    original_clear = weight_cache_module._clear_device_cache
    weight_cache_module._clear_device_cache = lambda: None
    try:
        store = FakeStore()
        cache = WeightCache(store, max_bytes=10_000, max_fetch_batch=2)
        items = make_items(10)
        cache.get_many(items)
        assert cache.total_bytes <= 10_000
        assert len(store.calls) == 5  # ceil(10/2)
        assert all(len(call) <= 2 for call in store.calls)
    finally:
        weight_cache_module._clear_device_cache = original_clear


def test_failed_batch_releases_every_owned_inflight_key():
    """A first-batch failure must not strand Events for later registered batches.

    get_many() registers the whole missing union before fetching batch 0.  Before
    this regression fix, an exception in batch 0 released only that batch; keys
    in batches 1..N remained permanently inflight and a retry deadlocked.
    """
    store = FailOnceStore()
    cache = WeightCache(store, max_bytes=10_000_000, max_fetch_batch=2)
    items = make_items(7)

    try:
        cache.get_many(items)
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("injected fetch failure did not propagate")

    assert not any(cache.inflight(key) for key, _ in items), \
        "failed get_many left one or more keys permanently inflight"
    result = cache.get_many(items)
    assert set(result) == {key for key, _ in items}


def test_consumer_releases_batch_before_requesting_next_one():
    """Prove the Python-level F74-v2 lifetime boundary without MLX.

    A normal ``for ids, pages in generator`` retains the previous loop target
    while requesting the next item. The producer asserts that its prior payload
    has already become collectable before it begins the next simulated fetch.
    """
    class Payload:
        pass

    seen = []

    def batches():
        previous = None
        for i in range(4):
            if previous is not None:
                gc.collect()
                assert previous() is None, f"batch {i - 1} survived into fetch {i}"
            payload = Payload()
            previous = weakref.ref(payload)
            yield [i], {i: payload}
            del payload

    consume_expert_batches(batches(), lambda ids, pages: seen.append(ids[0]))
    assert seen == [0, 1, 2, 3]


def test_consumer_exception_closes_producer_and_releases_payload():
    """A middle-batch compute failure must not leave the producer suspended."""
    class Payload:
        pass

    state = {"closed": False, "payload": None}

    def batches():
        try:
            for i in range(3):
                payload = Payload()
                state["payload"] = weakref.ref(payload)
                yield [i], {i: payload}
                del payload
        finally:
            state["closed"] = True

    try:
        consume_expert_batches(
            batches(),
            lambda ids, pages: (_ for _ in ()).throw(RuntimeError("injected compute")),
        )
    except RuntimeError as exc:
        assert "injected compute" in str(exc)
    else:
        raise AssertionError("injected compute failure did not propagate")

    gc.collect()
    assert state["closed"], "batch producer was not closed during unwind"
    assert state["payload"]() is None, "failed batch payload survived unwind"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for test in tests:
        try:
            test()
            print(f"  {test.__name__}: PASS")
        except Exception as exc:
            print(f"  {test.__name__}: FAIL ({type(exc).__name__}: {exc})")
            failed.append(test.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
