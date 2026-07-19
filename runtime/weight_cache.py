"""WeightCache: budgeted, LRU, pin-aware cache of materialized weight pages.

A WeightPage is a group of tensors that live and die together (one transformer
block, or a pinned group like embeddings). Pages are inserted on demand-miss or by
the prefetch thread; unpinned pages are evicted LRU-first when the byte budget is
exceeded. If the budget cannot hold even the newest page (tiny budgets), the cache
degrades to pass-through: the caller's reference keeps the tensors alive during
compute and the page is dropped immediately.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

    from .model_loader import WeightStore


@dataclass
class WeightPage:
    key: str
    tensors: dict[str, mx.array]
    nbytes: int
    pinned: bool = False
    origin: str = "demand"  # "demand" | "prefetch" | "pin"
    hits: int = 0  # re-uses after admission; scan-resistant eviction protects hits>0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0  # hits on pages inserted by the prefetch thread
    disk_s: float = 0.0
    bytes_read: int = 0

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0.0
        return (
            f"cache: {self.hits} hits / {self.misses} misses ({rate:.0f}% hit rate, "
            f"{self.prefetch_hits} via prefetch), {self.evictions} evictions, "
            f"store-accounted {self.bytes_read / 1e6:.0f}MB in {self.disk_s:.2f}s"
        )


def _tensor_bytes(t) -> int:
    return t.nbytes  # mx.array and QTensor both expose nbytes


def _clear_device_cache() -> None:
    """Keep MLX off the pure cache-coordination import path."""
    import mlx.core as mx

    mx.clear_cache()


class WeightCache:
    def __init__(self, store: WeightStore, max_bytes: int, transform=None, warm=None,
                 max_fetch_batch: int = 0):
        self.store = store
        self.max_bytes = max_bytes
        self.transform = transform  # e.g. QuantPolicy.transform: (name, arr) -> arr|QTensor
        self.warm = warm  # F04: compressed-RAM tier consulted before disk on miss
        # F74: cap how many missing pages get_many() materializes in one _fetch()/
        # mx.eval() call. 0 = old behavior (fetch the whole missing set at once).
        # Needed for architectures with large expert counts (GLM: 256 routed/layer):
        # a coupon-collector effect means even a SMALL chunk's expert union can
        # approach the full 256 on a cold-cache layer, so bounding chunk size alone
        # (F68) does not bound this call's peak transient allocation. Sub-batching
        # here bounds it to max_fetch_batch pages regardless of union size, and lets
        # _evict_locked() run between sub-batches so earlier pages in the SAME call
        # can be reclaimed before the next sub-batch is fetched.
        self.max_fetch_batch = max_fetch_batch
        self._pages: "OrderedDict[str, WeightPage]" = OrderedDict()
        self._total_bytes = 0
        self._reserved_bytes = 0
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self.stats = CacheStats()
        # F03: cumulative access frequency per key. Simulated on real MoE traces,
        # LFU-with-admission reaches 74% of the Belady bound at tight budgets
        # where LRU-family policies score ~0% (reuse distance = one full sweep).
        self.freq: dict[str, int] = {}

    # ---- public API -----------------------------------------------------

    def pin(self, key: str, names: list[str]) -> dict[str, mx.array]:
        tensors, secs, nbytes = self._fetch(names)
        with self._lock:
            self.stats.disk_s += secs
            self.stats.bytes_read += nbytes
            resident = sum(_tensor_bytes(t) for t in tensors.values())
            self._put_page_locked(
                WeightPage(key, tensors, resident, pinned=True, origin="pin"))
        return tensors

    def _fetch(self, names: list[str], *, apply_transform: bool = True):
        tensors, secs, nbytes = self.store.fetch(names)
        if self.transform and apply_transform:
            tensors = {n: self.transform(n, a) for n, a in tensors.items()}
        return tensors, secs, nbytes

    def get(self, key: str, names: list[str], origin: str = "demand", *,
            apply_transform: bool = True) -> dict[str, mx.array]:
        """Return a page, optionally bypassing the cache's lossy transform.

        Callers that bypass transforms must use a representation-specific key;
        a page hit does not re-transform an already admitted representation.
        """
        while True:
            with self._lock:
                if origin == "demand":
                    self.freq[key] = self.freq.get(key, 0) + 1
                page = self._pages.get(key)
                if page is not None:
                    self._pages.move_to_end(key)
                    if origin == "demand":
                        self.stats.hits += 1
                        page.hits += 1
                        if page.origin == "prefetch":
                            self.stats.prefetch_hits += 1
                            self._mark_demand_locked(page)  # count each prefetch once
                    return page.tensors
                inflight = self._inflight.get(key)
                if inflight is None:
                    self._inflight[key] = threading.Event()
                    break  # this thread loads
            # another thread is loading this page; wait and re-check
            inflight.wait()

        try:
            tensors = self.warm.take(key) if self.warm is not None else None
            if tensors is not None:
                secs, nbytes = 0.0, 0
            else:
                tensors, secs, nbytes = self._fetch(
                    names, apply_transform=apply_transform)
            with self._lock:
                if origin == "demand":
                    self.stats.misses += 1
                self.stats.disk_s += secs
                self.stats.bytes_read += nbytes
                resident = sum(_tensor_bytes(t) for t in tensors.values())
                self._put_page_locked(
                    WeightPage(key, tensors, resident, origin=origin))
                self._evict_locked()
            return tensors
        finally:
            with self._lock:
                self._inflight.pop(key).set()

    def get_many(self, items: list[tuple[str, list[str]]], origin: str = "demand") -> dict[str, dict]:
        """Batch get: all missing pages are fetched in ONE store.fetch call (grouped
        by shard → far fewer random reads than per-page fetches). Used for MoE
        experts, where 8 pages become needed at the same instant after routing."""
        result: dict[str, dict] = {}
        missing: list[tuple[str, list[str]]] = []
        # Keys whose inflight Events were created by THIS call.  If any warm-tier
        # lookup or fetch batch raises, every not-yet-processed key must be
        # released: otherwise later demand calls wait forever on an Event that no
        # loader can ever set.  Keep ownership explicit so the failure cleanup
        # cannot pop a replacement Event created by a waiter after an earlier
        # batch was released.
        owned_inflight: set[str] = set()
        with self._lock:
            for key, names in items:
                self.freq[key] = self.freq.get(key, 0) + 1
                page = self._pages.get(key)
                if page is not None:
                    self._pages.move_to_end(key)
                    self.stats.hits += 1
                    page.hits += 1
                    if page.origin == "prefetch":
                        self.stats.prefetch_hits += 1
                        self._mark_demand_locked(page)
                    result[key] = page.tensors
                elif key in self._inflight:
                    pass  # rare: prefetch racing — resolved via get() below
                else:
                    missing.append((key, names))
                    self._inflight[key] = threading.Event()
                    owned_inflight.add(key)

        def release_owned(keys):
            """Publish completion for keys still owned by this call.

            Remove local ownership while holding the cache lock, before waking a
            waiter.  A woken waiter may immediately install a new Event for the
            same key; the outer failure cleanup must never remove that new Event.
            """
            with self._lock:
                for key in keys:
                    if key not in owned_inflight:
                        continue
                    event = self._inflight.pop(key, None)
                    owned_inflight.discard(key)
                    if event is not None:
                        event.set()

        try:
            if missing and self.warm is not None:
                still = []
                for key, names in missing:
                    t = self.warm.take(key)
                    if t is not None:
                        with self._lock:
                            resident = sum(_tensor_bytes(x) for x in t.values())
                            self._put_page_locked(
                                WeightPage(key, t, resident, origin="demand"))
                            self._evict_locked()
                        result[key] = t
                        release_owned((key,))
                    else:
                        still.append((key, names))
                missing = still
            batch_size = self.max_fetch_batch if self.max_fetch_batch > 0 else len(missing)
            for start in range(0, len(missing), max(batch_size, 1)):
                batch = missing[start:start + batch_size]
                try:
                    all_names = [n for _, names in batch for n in names]
                    tensors, secs, nbytes = self._fetch(all_names)
                    with self._lock:
                        self.stats.misses += len(batch)
                        self.stats.disk_s += secs
                        self.stats.bytes_read += nbytes
                        for key, names in batch:
                            page_tensors = {n: tensors[n] for n in names}
                            resident = sum(_tensor_bytes(t) for t in page_tensors.values())
                            self._put_page_locked(
                                WeightPage(key, page_tensors, resident, origin=origin))
                            result[key] = page_tensors
                        # Evicting between sub-batches limits cache residency.  The
                        # MoE consumer must separately avoid retaining every
                        # returned page; see F74's compute-batch follow-up.
                        self._evict_locked()
                finally:
                    release_owned(key for key, _ in batch)
            for key, names in items:
                if key not in result:  # was inflight on another thread
                    result[key] = self.get(key, names, origin=origin)
            return result
        finally:
            # Includes batches that were registered up front but never reached
            # after an earlier batch failed.
            release_owned(tuple(owned_inflight))

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._pages

    def inflight(self, key: str) -> bool:
        with self._lock:
            return key in self._inflight

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def resident_keys(self) -> list[str]:
        with self._lock:
            return list(self._pages)

    def would_fit(self, nbytes: int) -> bool:
        """True if a page of this size can be admitted by evicting only *consumed*
        pages — i.e. pinned pages plus not-yet-used prefetched pages plus the new
        page stay under budget. Keeps the prefetcher from thrashing its own work."""
        with self._lock:
            return self._reserved_bytes + nbytes <= self.max_bytes

    def prepare_for(self, incoming_bytes: int) -> None:
        """Evict before a known-size demand fetch instead of after allocation.

        Ordinary ``get()`` cannot know a page's materialized size until the
        store returns it, so its historical budget enforcement necessarily
        happens after fetch. Callers with a conservative size estimate can use
        this method to keep ``old cache + incoming page`` within the residency
        budget and avoid delegating that overlap to macOS swap/compression.
        Pinned pages remain non-evictable; the governor separately decides
        whether the resulting allocation is safe.
        """
        incoming_bytes = max(0, int(incoming_bytes))
        with self._lock:
            target = max(0, self.max_bytes - incoming_bytes)
            self._evict_to_locked(target)

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _reserved(page: WeightPage) -> bool:
        return page.pinned or page.origin == "prefetch"

    def _put_page_locked(self, page: WeightPage) -> None:
        previous = self._pages.get(page.key)
        if previous is not None:
            self._total_bytes -= previous.nbytes
            if self._reserved(previous):
                self._reserved_bytes -= previous.nbytes
        self._pages[page.key] = page
        self._total_bytes += page.nbytes
        if self._reserved(page):
            self._reserved_bytes += page.nbytes

    def _remove_page_locked(self, key: str) -> WeightPage:
        page = self._pages.pop(key)
        self._total_bytes -= page.nbytes
        if self._reserved(page):
            self._reserved_bytes -= page.nbytes
        return page

    def _mark_demand_locked(self, page: WeightPage) -> None:
        was_reserved = self._reserved(page)
        page.origin = "demand"
        if was_reserved and not self._reserved(page):
            self._reserved_bytes -= page.nbytes

    def _evict_locked(self):
        self._evict_to_locked(self.max_bytes)

    def _evict_to_locked(self, target_bytes: int):
        """Evict by the exact historical policy with one selection/clear cycle.

        Unconsumed prefetch pages remain protected until every ordinary unpinned
        page is gone. Within each class, lowest frequency wins and OrderedDict
        position supplies the age tie-break, exactly matching the former repeated
        ``min`` loop.
        """
        target_bytes = max(0, int(target_bytes))
        if self._total_bytes <= target_bytes:
            return
        ordinary = []
        prefetched = []
        for age, (key, page) in enumerate(self._pages.items()):
            if page.pinned:
                continue
            candidate = (self.freq.get(key, 0), age, key)
            (prefetched if page.origin == "prefetch" else ordinary).append(
                candidate)
        victims = sorted(ordinary) + sorted(prefetched)
        evicted = False
        try:
            for _frequency, _age, key in victims:
                if self._total_bytes <= target_bytes:
                    break
                page = self._pages[key]
                if self.warm is not None:
                    self.warm.admit(key, page.tensors)
                self._remove_page_locked(key)
                self.stats.evictions += 1
                evicted = True
        finally:
            if evicted:
                # Device cleanup is lazy/import-isolated so coordination and
                # failure handling remain testable without importing MLX.
                _clear_device_cache()
