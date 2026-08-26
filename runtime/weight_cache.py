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
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .cache_policy import CacheEntry, rank_victims

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
    fetch_s: float = 0.0  # store wall attributed to this admission
    store_bytes: int = 0  # store-accounted bytes attributed to this admission


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0  # hits on pages inserted by the prefetch thread
    pinned_hits: int = 0  # demand hits served by permanent pin groups
    prefetch_waits: int = 0  # demand calls that waited on an in-flight prefetch
    prefetch_wait_s: float = 0.0
    # FreeToken-style overlap evidence. A prefetch is useful only when demand
    # consumes the admitted page before eviction; merely completing a worker
    # read is not a latency win. The load/wait split gives a conservative lower
    # bound on I/O hidden behind compute without adding synchronization.
    prefetch_loads: int = 0
    prefetch_loaded_bytes: int = 0
    prefetch_load_s: float = 0.0
    prefetch_useful_pages: int = 0
    prefetch_useful_bytes: int = 0
    prefetch_useful_load_s: float = 0.0
    prefetch_wasted_pages: int = 0
    prefetch_wasted_bytes: int = 0
    prefetch_wasted_load_s: float = 0.0
    disk_s: float = 0.0
    bytes_read: int = 0
    # F197: pages a producer returned without every requested tensor. Any
    # nonzero value means a page was served short and the arithmetic that
    # consumed it is unsound -- the cache raises rather than admitting one, so
    # this counts detections, not survivals. It exists because a short page is
    # otherwise silent: an expert whose weights never arrive contributes
    # nothing to the routed sum and the output is merely *wrong*, not an error.
    incomplete_pages: int = 0

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0.0
        short = (f", {self.incomplete_pages} INCOMPLETE PAGES"
                 if self.incomplete_pages else "")
        return (
            f"cache: {self.hits} hits / {self.misses} misses ({rate:.0f}% hit rate, "
            f"{self.prefetch_hits} via prefetch, {self.pinned_hits} pinned), "
            f"{self.prefetch_waits} prefetch waits/{self.prefetch_wait_s:.3f}s, "
            f"{self.evictions} evictions, "
            f"store-accounted {self.bytes_read / 1e6:.0f}MB in {self.disk_s:.2f}s"
            f"{short}"
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
        self._inflight_origin: dict[str, str] = {}
        # Exact byte allowances for explicitly suspended pin groups. A later
        # zero-copy restoration may reclaim only the same key and byte count;
        # this preserves a startup-admitted permanent pin even if the live
        # governor has since lowered the ordinary demand-cache ceiling.
        self._suspended_pin_bytes: dict[str, int] = {}
        self.stats = CacheStats()
        # F03: cumulative access frequency per key. Simulated on real MoE traces,
        # LFU-with-admission reaches 74% of the Belady bound at tight budgets
        # where LRU-family policies score ~0% (reuse distance = one full sweep).
        self.freq: dict[str, int] = {}

    # ---- public API -----------------------------------------------------

    def pin(self, key: str, names: list[str]) -> dict[str, mx.array]:
        tensors, secs, nbytes = self._fetch(names)
        capacity_error = None
        with self._lock:
            self.stats.disk_s += secs
            self.stats.bytes_read += nbytes
            resident = sum(_tensor_bytes(t) for t in tensors.values())
            previous = self._pages.get(key)
            pinned_before = sum(
                page.nbytes for page in self._pages.values() if page.pinned)
            if previous is not None and previous.pinned:
                pinned_before -= previous.nbytes
            projected_pinned = pinned_before + resident
            if projected_pinned > self.max_bytes:
                capacity_error = MemoryError(
                    f"pin {key!r} would require {projected_pinned} resident "
                    f"pinned bytes, exceeding the {self.max_bytes}-byte "
                    "weight-cache capacity")
            else:
                self._put_page_locked(WeightPage(
                    key, tensors, resident, pinned=True, origin="pin"))
                self._evict_locked()
        if capacity_error is not None:
            # A store may retain source-file mappings independently of the
            # returned tensor dictionary. Undo both layers of residency before
            # refusing startup so a failed experimental pin does not poison the
            # following configuration's memory baseline.
            release_names = tuple(tensors)
            del tensors
            _clear_device_cache()
            release = getattr(self.store, "release_cache_pages", None)
            if release is not None and release_names:
                release(release_names)
            raise capacity_error
        return tensors

    def _fetch(self, names: list[str], *, apply_transform: bool = True):
        tensors, secs, nbytes = self.store.fetch(names)
        self._require_complete(names, tensors, "store fetch")
        if self.transform and apply_transform:
            tensors = {n: self.transform(n, a) for n, a in tensors.items()}
            self._require_complete(names, tensors, "cache transform")
        return tensors, secs, nbytes

    def _require_complete(self, names, tensors, source: str) -> None:
        """Refuse a page that is missing any tensor the caller asked for.

        A short page cannot be detected downstream: a routed expert whose
        weights never arrived simply contributes zero to the MoE sum, so the
        request completes and returns a plausible, wrong answer. Failing here
        converts that into an error at the exact point the bytes went missing.
        """
        missing = [name for name in names if name not in tensors]
        if not missing:
            return
        with self._lock:
            self.stats.incomplete_pages += 1
        raise KeyError(
            f"{source} returned an incomplete page: {len(missing)} of "
            f"{len(names)} tensors missing, first {missing[:3]}")

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
                        if page.pinned:
                            self.stats.pinned_hits += 1
                        if page.origin == "prefetch":
                            self.stats.prefetch_hits += 1
                            self._mark_demand_locked(page)  # count each prefetch once
                    return page.tensors
                inflight = self._inflight.get(key)
                if inflight is None:
                    self._inflight[key] = threading.Event()
                    self._inflight_origin[key] = origin
                    break  # this thread loads
                waits_for_prefetch = (
                    origin == "demand"
                    and self._inflight_origin.get(key) == "prefetch")
            # another thread is loading this page; wait and re-check
            wait_started = time.perf_counter() if waits_for_prefetch else 0.0
            inflight.wait()
            if waits_for_prefetch:
                waited = time.perf_counter() - wait_started
                with self._lock:
                    self.stats.prefetch_waits += 1
                    self.stats.prefetch_wait_s += waited

        try:
            tensors = self.warm.take(key) if self.warm is not None else None
            if tensors is not None:
                self._require_complete(names, tensors, "warm tier")
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
                page = WeightPage(
                    key, tensors, resident, origin=origin,
                    fetch_s=secs, store_bytes=nbytes,
                )
                self._put_page_locked(page)
                if origin == "prefetch":
                    self.stats.prefetch_loads += 1
                    self.stats.prefetch_loaded_bytes += nbytes
                    self.stats.prefetch_load_s += secs
                self._evict_locked()
            return tensors
        finally:
            with self._lock:
                self._inflight_origin.pop(key, None)
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
                    if page.pinned:
                        self.stats.pinned_hits += 1
                    if page.origin == "prefetch":
                        self.stats.prefetch_hits += 1
                        self._mark_demand_locked(page)
                    result[key] = page.tensors
                elif key in self._inflight:
                    pass  # rare: prefetch racing — resolved via get() below
                else:
                    missing.append((key, names))
                    self._inflight[key] = threading.Event()
                    self._inflight_origin[key] = origin
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
                    self._inflight_origin.pop(key, None)
                    owned_inflight.discard(key)
                    if event is not None:
                        event.set()

        try:
            if missing and self.warm is not None:
                still = []
                for key, names in missing:
                    t = self.warm.take(key)
                    if t is not None:
                        self._require_complete(names, t, "warm tier")
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
    def pinned_bytes(self) -> int:
        with self._lock:
            return sum(page.nbytes for page in self._pages.values()
                       if page.pinned)

    @property
    def prefetched_bytes(self) -> int:
        with self._lock:
            return sum(page.nbytes for page in self._pages.values()
                       if page.origin == "prefetch")

    @property
    def resident_keys(self) -> list[str]:
        with self._lock:
            return list(self._pages)

    def clear(self) -> None:
        """Release all resident pages after producers have been joined.

        Engine shutdown previously relied on cyclic-GC timing to destroy the
        cache object. Large decoded sidecar pages made back-to-back cold-engine
        gates retain several GiB until a later collection, even though every
        worker had already stopped. Explicit close-time release is safe once
        prefetch/expert producers are joined and changes no live-engine policy.
        """
        released_names: list[str] = []
        with self._lock:
            if self._inflight:
                raise RuntimeError(
                    "cannot clear WeightCache while fetches are in flight"
                )
            for page in self._pages.values():
                self._record_unused_prefetch_locked(page)
                released_names.extend(page.tensors)
            self._pages.clear()
            self._total_bytes = 0
            self._reserved_bytes = 0
            self._inflight_origin.clear()
            self._suspended_pin_bytes.clear()
        _clear_device_cache()
        release = getattr(self.store, "release_cache_pages", None)
        if release is not None and released_names:
            release(tuple(released_names))

    def discard(self, key: str, names: list[str] | tuple[str, ...] = ()) -> bool:
        """Release one consumed unpinned page at its caller-known lifetime end.

        Budget eviction can occur while the consumer still owns the returned
        tensor mapping, so it is not a sufficient file-mapping lifetime signal.
        Layer-stationary sidecar paths call this only after their synchronized
        compute and local weight reference are gone. The optional ``names`` let
        the store retry source-file invalidation even if ordinary pressure
        already removed the cache entry earlier.
        """
        released_names = list(names)
        removed = False
        with self._lock:
            page = self._pages.get(key)
            if page is not None:
                if page.pinned:
                    return False
                if not released_names:
                    released_names.extend(page.tensors)
                self._record_unused_prefetch_locked(page)
                self._remove_page_locked(key)
                self.stats.evictions += 1
                removed = True
        # The explicit consumer boundary warrants a clear even when ordinary
        # budget pressure removed the page first: evaluated MLX graph/cache
        # state can outlive the WeightPage object itself.
        _clear_device_cache()
        release = getattr(self.store, "release_cache_pages", None)
        if release is not None and released_names:
            release(tuple(released_names))
        return removed

    def release_pinned(
        self, key: str, names: list[str] | tuple[str, ...] = (),
    ) -> int:
        """Release one explicitly pinned page at a proven consumer boundary.

        This is deliberately separate from :meth:`discard`, which refuses
        pinned pages.  A caller must first drop every tensor reference it owns;
        the cache then removes the complete pin group, clears evaluated device
        state, and invalidates any source-file mappings.  The returned byte
        count is zero when ``key`` is absent or no longer pinned.

        The operation exists for phase-scoped weights such as an untied Qwen
        LM head: it is useful for proposal projection and target verification,
        but dead throughout the much larger streamed target-trunk sweep.
        """
        released_names = list(names)
        released_bytes = 0
        removed_page = None
        with self._lock:
            page = self._pages.get(key)
            if page is None or not page.pinned:
                return 0
            if not released_names:
                released_names.extend(page.tensors)
            removed_page = self._remove_page_locked(key)
            released_bytes = removed_page.nbytes
            self._suspended_pin_bytes[key] = released_bytes
            self.stats.evictions += 1
        # Do not retain a local page/tensor mapping across the explicit device
        # clear. The caller has already dropped its own tensor reference.
        del page, removed_page
        _clear_device_cache()
        release = getattr(self.store, "release_cache_pages", None)
        if release is not None and released_names:
            release(tuple(released_names))
        return released_bytes

    def register_suspended_pin(self, key: str, nbytes: int) -> None:
        """Register an exact dormant pin lease without materializing tensors.

        This is the startup counterpart to :meth:`release_pinned`: a caller
        with an exact metadata-derived resident size may defer a phase-scoped
        pin until its first real use.  The lease never widens ``max_bytes`` and
        may only be consumed by a page with exactly the registered byte count.
        Collisions and inconsistent repeated declarations fail closed.
        """
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("suspended pin bytes must be positive")
        with self._lock:
            if key in self._pages:
                raise RuntimeError(
                    f"cannot suspend pin {key!r}: cache key already exists")
            previous = self._suspended_pin_bytes.get(key)
            if previous is not None and previous != nbytes:
                raise RuntimeError(
                    f"cannot change suspended pin {key!r} from {previous} "
                    f"to {nbytes} bytes")
            pinned = sum(
                page.nbytes for page in self._pages.values() if page.pinned)
            if pinned + nbytes > self.max_bytes:
                raise MemoryError(
                    f"suspended pin {key!r} would require {pinned + nbytes} "
                    f"resident pinned bytes, exceeding the "
                    f"{self.max_bytes}-byte weight-cache capacity")
            self._suspended_pin_bytes[key] = nbytes

    def promote_to_pin(
        self, source_key: str, target_key: str, *,
        tensors: dict[str, mx.array] | None = None,
    ) -> dict[str, mx.array] | None:
        """Atomically reclassify a resident demand page as a permanent pin.

        No fetch or tensor copy occurs.  Promotion fails softly when the source
        page was pass-through/evicted or when the current cache budget cannot
        contain it alongside existing pins.  In that case the demand page is
        left unchanged and the caller may continue with ordinary demand loads.
        A target-key collision is a lifecycle bug and fails closed.
        """
        with self._lock:
            page = self._pages.get(source_key)
            if page is not None and page.pinned:
                return None
            if target_key != source_key and target_key in self._pages:
                raise RuntimeError(
                    f"cannot promote {source_key!r}: target cache key "
                    f"{target_key!r} already exists")
            if page is None:
                if tensors is None:
                    return None
                resident = sum(_tensor_bytes(value) for value in tensors.values())
                page = WeightPage(
                    source_key, tensors, resident, origin="demand")
            allowance = self._suspended_pin_bytes.get(target_key)
            pinned_before = sum(
                value.nbytes for value in self._pages.values()
                if value.pinned
            )
            leased_restore = allowance == page.nbytes
            if not leased_restore and pinned_before + page.nbytes > self.max_bytes:
                return None
            if source_key in self._pages:
                self._remove_page_locked(source_key)
            page.key = target_key
            page.pinned = True
            page.origin = "pin"
            self._put_page_locked(page)
            if leased_restore:
                self._suspended_pin_bytes.pop(target_key, None)
            # A restored startup pin may legitimately exceed a cache ceiling
            # that the governor lowered while it was suspended. Reclaim that
            # exact pin first, then shed every evictable demand/prefetch page;
            # this reproduces the pre-suspension invariant without ratcheting
            # the configured budget upward.
            self._evict_locked()
            return page.tensors

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

    def trim_to(self, target_bytes: int) -> int:
        """Evict consumed pages to a temporary residency target.

        Unlike changing ``max_bytes``, this does not lower the admission budget
        for the next request: it is a post-request pressure valve that sheds
        cold LRU pages now while allowing later demand to refill the configured
        cache normally. Pinned pages remain protected by the ordinary eviction
        policy. Returns the cache-accounted bytes actually released.
        """
        target = max(0, int(target_bytes))
        with self._lock:
            before = self._total_bytes
            self._evict_to_locked(target)
            return before - self._total_bytes

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
        if page.origin == "prefetch":
            self.stats.prefetch_useful_pages += 1
            self.stats.prefetch_useful_bytes += page.store_bytes
            self.stats.prefetch_useful_load_s += page.fetch_s
        page.origin = "demand"
        if was_reserved and not self._reserved(page):
            self._reserved_bytes -= page.nbytes

    def _record_unused_prefetch_locked(self, page: WeightPage) -> None:
        if page.origin != "prefetch":
            return
        self.stats.prefetch_wasted_pages += 1
        self.stats.prefetch_wasted_bytes += page.store_bytes
        self.stats.prefetch_wasted_load_s += page.fetch_s

    def _evict_locked(self):
        self._evict_to_locked(self.max_bytes)

    def _evict_to_locked(self, target_bytes: int):
        """Evict by the exact historical policy with one selection/clear cycle.

        Unconsumed prefetch pages remain protected until every ordinary unpinned
        page is gone. Within each class, lowest frequency wins and OrderedDict
        position supplies the age tie-break, exactly matching the former repeated
        ``min`` loop.

        The ordering itself lives in ``runtime.cache_policy`` so the offline
        capacity simulator replays this policy rather than a second, drifting
        transcription of it.
        """
        target_bytes = max(0, int(target_bytes))
        if self._total_bytes <= target_bytes:
            return
        victims = rank_victims(
            (
                CacheEntry(key, page.pinned, page.origin == "prefetch")
                for key, page in self._pages.items()
            ),
            self.freq,
        )
        evicted = False
        released_names: list[str] = []
        try:
            for key in victims:
                if self._total_bytes <= target_bytes:
                    break
                page = self._pages[key]
                if self.warm is not None:
                    self.warm.admit(key, page.tensors)
                else:
                    released_names.extend(page.tensors)
                self._record_unused_prefetch_locked(page)
                self._remove_page_locked(key)
                del page
                self.stats.evictions += 1
                evicted = True
        finally:
            if evicted:
                # Device cleanup is lazy/import-isolated so coordination and
                # failure handling remain testable without importing MLX.
                _clear_device_cache()
                release = getattr(self.store, "release_cache_pages", None)
                if release is not None and released_names:
                    release(tuple(released_names))
