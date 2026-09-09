import threading
import time

import pytest

from runtime.prefetcher import Prefetcher


class BlockingCache:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.second_loaded = threading.Event()
        self.loaded = []
        self.max_fit = 1_000_000

    def contains(self, _key):
        return False

    def inflight(self, _key):
        return False

    def would_fit(self, nbytes):
        return nbytes <= self.max_fit

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


def test_worker_rechecks_page_hint_after_live_budget_shrink():
    cache = BlockingCache()
    cache.max_fit = 100
    prefetcher = Prefetcher(cache, page_size_hint=10, workers=1)
    try:
        assert prefetcher.schedule(
            "first", ["a"], page_size_hint=100)
        assert cache.started.wait(timeout=2)
        assert prefetcher.schedule(
            "second", ["b"], page_size_hint=100)
        cache.max_fit = 50
        cache.release.set()
        deadline = time.monotonic() + 2
        while prefetcher.skipped_budget == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        cache.release.set()
        prefetcher.close()

    assert cache.loaded == ["first"]
    assert prefetcher.skipped_budget == 1


def test_per_page_hint_rejects_oversized_page_before_enqueue():
    cache = BlockingCache()
    cache.max_fit = 150
    prefetcher = Prefetcher(cache, page_size_hint=10, workers=1)
    try:
        assert not prefetcher.schedule(
            "oversized", ["a"], page_size_hint=200)
    finally:
        prefetcher.close()
    assert prefetcher.skipped_budget == 1


class BeforeAdmissionLock:
    """Park one scheduler after its fast check, before the actual lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scheduler_ident = None
        self.arrived = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        if threading.get_ident() == self.scheduler_ident:
            self.arrived.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test scheduler was not released")
        self.lock.acquire()
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.mark.parametrize("only_if_idle", [False, True])
def test_pause_rejects_scheduler_that_already_passed_fast_check(only_if_idle):
    cache = BlockingCache()
    cache.release.set()
    prefetcher = Prefetcher(cache, page_size_hint=100)
    gate = BeforeAdmissionLock()
    prefetcher._lock = gate
    accepted = []

    def schedule():
        gate.scheduler_ident = threading.get_ident()
        accepted.append(prefetcher.schedule(
            "late", ["a"], only_if_idle=only_if_idle))

    scheduler = threading.Thread(target=schedule)
    try:
        scheduler.start()
        assert gate.arrived.wait(timeout=2)
        prefetcher.pause_and_wait_idle(timeout_s=1)
        assert prefetcher.paused and not prefetcher._scheduled
        # The pause has returned. This producer must NOT repopulate the cache.
        gate.release.set()
        scheduler.join(timeout=2)
        assert not scheduler.is_alive()
        assert accepted == [False]
        assert prefetcher.scheduled_count == 0
    finally:
        gate.release.set()
        scheduler.join(timeout=2)
        prefetcher.close()
    assert cache.loaded == [] and not prefetcher._scheduled


def test_pause_waits_for_accepted_work_then_explicit_resume_allows_new_hint():
    cache = BlockingCache()
    prefetcher = Prefetcher(cache, page_size_hint=100)
    finished = threading.Event()

    def pause():
        prefetcher.pause_and_wait_idle(timeout_s=2)
        finished.set()

    pauser = threading.Thread(target=pause)
    try:
        assert prefetcher.schedule("first", ["a"])
        assert cache.started.wait(timeout=2)
        pauser.start()
        deadline = time.monotonic() + 2
        while not prefetcher.paused and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prefetcher.paused
        assert not finished.is_set()
        assert not prefetcher.schedule("late", ["b"])
        cache.release.set()
        assert finished.wait(timeout=2)
        assert cache.loaded == ["first"] and not prefetcher._scheduled
        prefetcher.paused = False
        assert prefetcher.schedule("second", ["b"])
        assert cache.second_loaded.wait(timeout=2)
    finally:
        cache.release.set()
        if pauser.ident is not None:
            pauser.join(timeout=3)
        prefetcher.close()
    assert cache.loaded == ["first", "second"]


def test_pause_timeout_keeps_new_hints_disabled():
    cache = BlockingCache()
    prefetcher = Prefetcher(cache, page_size_hint=100)
    try:
        assert prefetcher.schedule("first", ["a"])
        assert cache.started.wait(timeout=2)
        with pytest.raises(TimeoutError, match="did not become idle"):
            prefetcher.pause_and_wait_idle(timeout_s=0)
        assert prefetcher.paused
        assert not prefetcher.schedule("late", ["b"])
        cache.release.set()
        prefetcher.pause_and_wait_idle(timeout_s=2)
        assert cache.loaded == ["first"] and not prefetcher._scheduled
    finally:
        cache.release.set()
        prefetcher.close()
