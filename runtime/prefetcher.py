"""Sequential prefetcher: a single background worker that materializes upcoming
weight pages into the WeightCache while the main thread computes.

One worker is deliberate — the underlying disk is serial, and MLX eval from a second
thread is safe (verified empirically). The prefetcher never violates the cache
budget: it skips scheduling when the page would not fit without evicting, so demand
traffic always wins over speculation.
"""

from __future__ import annotations

import queue
import threading

from .weight_cache import WeightCache


class Prefetcher:
    def __init__(self, cache: WeightCache, page_size_hint: int = 0, workers: int = 1):
        """workers=1 for raw safetensors (the disk is serial). For packed stores,
        workers=2 overlaps one thread's zstd decode with the other's disk read."""
        self.cache = cache
        self.page_size_hint = page_size_hint  # bytes; used for budget check before size is known
        self._q: "queue.Queue[tuple[str, list[str]] | None]" = queue.Queue()
        self._scheduled: set[str] = set()
        self._lock = threading.Lock()
        self.scheduled_count = 0
        self.skipped_budget = 0
        self.paused = False  # set by the memory-pressure governor
        self._closing = threading.Event()
        self._workers = [threading.Thread(target=self._run, daemon=True) for _ in range(max(1, workers))]
        for w in self._workers:
            w.start()

    def schedule(self, key: str, names: list[str]):
        if self.paused or self._closing.is_set():
            return
        with self._lock:
            if key in self._scheduled:
                return
            if self.cache.contains(key) or self.cache.inflight(key):
                return
            if not self.cache.would_fit(self.page_size_hint):
                self.skipped_budget += 1
                return
            self._scheduled.add(key)
            self.scheduled_count += 1
        self._q.put((key, names))

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            key, names = item
            try:
                if not self._closing.is_set():
                    self.cache.get(key, names, origin="prefetch")
            except Exception:
                pass  # demand path will retry and surface the real error
            finally:
                with self._lock:
                    self._scheduled.discard(key)

    def close(self):
        self._closing.set()
        # Cancel work that has not started. Sentinels queued behind a deep backlog
        # let the old engine keep touching disk/MLX after a model swap.
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                key, _ = item
                with self._lock:
                    self._scheduled.discard(key)
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join(timeout=15)
        alive = [w.name for w in self._workers if w.is_alive()]
        if alive:
            # Fail closed: EngineManager must not construct a replacement while
            # an old worker may still finish a lazy MLX/NAS transaction.
            raise RuntimeError(f"prefetch workers did not stop during close: {alive}")

    def summary(self) -> str:
        return f"prefetch: {self.scheduled_count} scheduled, {self.skipped_budget} skipped (budget)"
