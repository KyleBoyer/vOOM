import threading

from runtime.prefetcher import Prefetcher


class BlockingCache:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.second_loaded = threading.Event()
        self.loaded = []

    def contains(self, _key):
        return False

    def inflight(self, _key):
        return False

    def would_fit(self, _nbytes):
        return True

    def get(self, key, _names, origin="demand"):
        assert origin == "prefetch"
        self.started.set()
        self.release.wait(timeout=5)
        self.loaded.append(key)
        if key == "second":
            self.second_loaded.set()


def test_idle_only_speculation_does_not_build_a_backlog():
    cache = BlockingCache()
    prefetcher = Prefetcher(cache, page_size_hint=100, workers=1)
    try:
        assert prefetcher.schedule("known-work", ["a"])
        assert cache.started.wait(timeout=2)
        assert not prefetcher.schedule(
            "speculative", ["b"], only_if_idle=True)
        assert prefetcher.skipped_busy == 1
        cache.release.set()
    finally:
        cache.release.set()
        prefetcher.close()
    assert cache.loaded == ["known-work"]


def test_aggressive_mode_can_queue_for_explicit_ab():
    cache = BlockingCache()
    prefetcher = Prefetcher(cache, page_size_hint=100, workers=1)
    try:
        assert prefetcher.schedule("first", ["a"])
        assert cache.started.wait(timeout=2)
        assert prefetcher.schedule("second", ["b"], only_if_idle=False)
        cache.release.set()
        assert cache.second_loaded.wait(timeout=2)
    finally:
        cache.release.set()
        prefetcher.close()
    assert cache.loaded == ["first", "second"]
