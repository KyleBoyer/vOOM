"""F16: memory-pressure governor.

The fixed 55%-of-RAM budget clamp cannot see external consumers (an 8 GB VM, the
GLM downloader's buffers, other apps) — four macOS "out of application memory"
incidents in two days came from exactly that blind spot. This governor samples
system availability + our own Metal telemetry on a background thread and, with
hysteresis and dwell, sheds load in escalating order:

  WARN     pause prefetch scheduling, clear the MLX buffer cache
  CRITICAL also shrink the WeightCache budget stepwise (evicting immediately)
  GREEN    (sustained) restore the budget gradually toward the configured value.

The allocation ceiling is sampled, not a machine-wide constant: it is the
smaller of MLX's device-recommended working set and the current process's Metal
footprint plus system-available RAM after preserving the critical reserve.

Signals: psutil available RAM (external view) and mx.get_active_memory (our own
footprint). DispatchSourceMemoryPressure would be event-driven; polling at 2 s is
sufficient for how fast our allocations move and keeps this dependency-free.
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import psutil

_DEFAULT_CRITICAL_AVAILABLE = int(1.2e9)


def _device_recommended_limit() -> int:
    """Return a hardware-derived upper bound, never a machine-specific guess.

    Current MLX exposes ``max_recommended_working_set_size``.  Older builds did
    not, so fall back to the device's unified-memory size and finally macOS's
    physical-memory total.  The live system-availability sample remains the
    tighter bound in normal operation; this value only prevents treating
    reclaimable system RAM as an unlimited Metal working set.
    """
    try:
        info = mx.device_info()
    except (AttributeError, TypeError, ValueError):
        info = {}
    for key in ("max_recommended_working_set_size", "memory_size"):
        try:
            value = int(info.get(key, 0))
        except (AttributeError, TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return int(psutil.virtual_memory().total)


class MemoryGovernor:
    def __init__(
        self,
        cache,
        prefetcher=None,
        poll_s: float = 2.0,
        warn_available: int = int(2.0e9),
        critical_available: int = _DEFAULT_CRITICAL_AVAILABLE,
        green_available: int = int(3.5e9),
        shrink_step: float = 0.15,
        floor_bytes: int = int(1.5e9),
        metal_limit: int | None = None,
    ):
        self.cache = cache
        self.prefetcher = prefetcher
        self.poll_s = poll_s
        self.warn = warn_available
        self.critical = critical_available
        self.green = green_available
        self.shrink_step = shrink_step
        self.floor = floor_bytes
        self.metal_limit = int(
            _device_recommended_limit() if metal_limit is None else metal_limit
        )
        if self.metal_limit <= 0:
            raise ValueError("metal_limit must be positive")
        self.configured_max = cache.max_bytes
        self.shrinks = 0
        self.restores = 0
        self.reservations = 0  # F42: synchronous pre-allocation sheds
        self.reservation_failures = 0
        self.paused_prefetch = False
        self._green_streak = 0
        self._stop = threading.Event()
        self._peak_lock = threading.Lock()
        self._request_peak_metal_bytes = mx.get_active_memory()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset_request_peak(self, initial: int | None = None) -> None:
        """Start a request-scoped peak observed by the governor thread."""
        value = mx.get_active_memory() if initial is None else int(initial)
        with self._peak_lock:
            self._request_peak_metal_bytes = value

    def request_peak(self) -> int:
        with self._peak_lock:
            return self._request_peak_metal_bytes

    def _metal_ceiling(self, active: int, available: int) -> int:
        """Live unified-memory allocation ceiling for the current sample."""
        system_headroom = max(0, int(available) - self.critical)
        return min(self.metal_limit, int(active) + system_headroom)

    def current_ceiling(self) -> int:
        """Sample the safe Metal boundary for admission/scheduling now."""
        return self._metal_ceiling(
            mx.get_active_memory(), psutil.virtual_memory().available
        )

    # ---- actions ------------------------------------------------------

    def _set_cache_max(self, nbytes: int):
        with self.cache._lock:
            self.cache.max_bytes = nbytes
            self.cache._evict_locked()

    def _pause_prefetch(self, paused: bool):
        if self.prefetcher is not None:
            self.prefetcher.paused = paused
        self.paused_prefetch = paused

    def reserve(self, incoming_bytes: int, margin: int = int(0.4e9)):
        """F42: synchronous pre-allocation reservation. The 2 s poll reacts AFTER
        a transient crosses the ceiling; callers that know a big allocation is
        coming (an expert batch fetch) declare it here first. If the projected
        footprint would cross the live device/system ceiling, shed now: MLX
        scratch first, then shrink the weight cache below the projected
        overshoot. Green restores the budget later exactly as with poll-driven
        shrinks."""
        # cache_memory is deliberately NOT counted: MLX reuses those buffers as
        # the very scratch being declared, so counting both double-books it.
        active = mx.get_active_memory()
        available = psutil.virtual_memory().available
        ceiling = self._metal_ceiling(active, available)
        projected = active + incoming_bytes + margin
        if projected <= ceiling:
            return
        # Stop admitting speculative work while we try to make room. A running
        # fetch cannot be cancelled safely, but no new one should race this
        # reservation and consume the space being reclaimed.
        self._pause_prefetch(True)
        mx.clear_cache()
        active = mx.get_active_memory()
        available = psutil.virtual_memory().available
        ceiling = self._metal_ceiling(active, available)
        projected = active + incoming_bytes + margin
        if projected <= ceiling:
            return
        overshoot = projected - ceiling
        new_max = max(self.floor, self.cache.max_bytes - overshoot)
        if new_max < self.cache.max_bytes:
            self._set_cache_max(new_max)
            self.reservations += 1
            print(f"[governor] RESERVE {incoming_bytes / 1e9:.2f}GB incoming -> "
                  f"cache budget {new_max / 1e9:.1f}GB", flush=True)
        # Eviction may be insufficient because pinned/current tensors and the
        # operation's own scratch are not reclaimable cache pages. The old path
        # continued anyway after reaching the cache floor—the exact fail-open
        # behavior that let known transients run past the sampled safe ceiling.
        # Re-measure after reclamation and reject before the allocation.
        mx.clear_cache()
        active = mx.get_active_memory()
        available = psutil.virtual_memory().available
        ceiling = self._metal_ceiling(active, available)
        projected = active + incoming_bytes + margin
        if projected > ceiling:
            self.reservation_failures += 1
            raise MemoryError(
                "unsafe Metal reservation refused before allocation: "
                f"active={active / 1e9:.2f}GB incoming={incoming_bytes / 1e9:.2f}GB "
                f"margin={margin / 1e9:.2f}GB projected={projected / 1e9:.2f}GB "
                f"available={available / 1e9:.2f}GB "
                f"ceiling={ceiling / 1e9:.2f}GB"
            )

    def fit_cache_to_live_headroom(self, margin: int = int(0.4e9)) -> int:
        """Cap future cache residency to the headroom sampled right now.

        Unlike :meth:`reserve`, this is for a *negotiable* cache target rather
        than a fixed imminent allocation.  Persistent tensors already counted
        by ``active`` and ``cache.total_bytes`` stay resident; only the
        additional cache allowance is fitted below the live Metal ceiling.
        GREEN restoration can grow it back toward ``configured_max`` later.
        """
        active = mx.get_active_memory()
        available = psutil.virtual_memory().available
        ceiling = self._metal_ceiling(active, available)
        additional = max(0, ceiling - active - margin)
        safe_max = max(
            self.floor,
            int(self.cache.total_bytes) + additional,
        )
        new_max = min(self.cache.max_bytes, safe_max)
        if new_max < self.cache.max_bytes:
            self._pause_prefetch(True)
            self._set_cache_max(new_max)
            self.reservations += 1
            print(
                f"[governor] FIT live headroom -> cache budget "
                f"{new_max / 1e9:.1f}GB (configured "
                f"{self.configured_max / 1e9:.1f}GB)",
                flush=True,
            )
        return self.cache.max_bytes

    # ---- loop ----------------------------------------------------------

    def _run(self):
        while not self._stop.wait(self.poll_s):
            try:
                avail = psutil.virtual_memory().available
                metal = mx.get_active_memory()
                ceiling = self._metal_ceiling(metal, avail)
                with self._peak_lock:
                    self._request_peak_metal_bytes = max(
                        self._request_peak_metal_bytes, metal)

                if avail < self.critical or metal > ceiling:
                    new_max = max(self.floor, int(self.cache.max_bytes * (1 - self.shrink_step)))
                    if new_max < self.cache.max_bytes:
                        self._set_cache_max(new_max)
                        self.shrinks += 1
                        print(f"[governor] CRITICAL avail={avail / 1e9:.1f}GB "
                              f"metal={metal / 1e9:.1f}GB ceiling={ceiling / 1e9:.1f}GB "
                              f"-> cache budget {new_max / 1e9:.1f}GB", flush=True)
                    self._pause_prefetch(True)
                    mx.clear_cache()
                    self._green_streak = 0
                elif avail < self.warn:
                    self._pause_prefetch(True)
                    mx.clear_cache()
                    self._green_streak = 0
                elif avail > self.green:
                    self._green_streak += 1
                    if self._green_streak >= 3:  # dwell: ~6 s of sustained green
                        if self.paused_prefetch:
                            self._pause_prefetch(False)
                        if self.cache.max_bytes < self.configured_max:
                            new_max = min(self.configured_max,
                                          int(self.cache.max_bytes * (1 + self.shrink_step / 2)))
                            self._set_cache_max(new_max)
                            self.restores += 1
                else:
                    self._green_streak = 0
            except Exception:
                pass  # governor must never take the runtime down

    def close(self):
        self._stop.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            # The governor calls process-global MLX cache operations. Never let
            # an old engine's thread survive into a replacement engine.
            raise RuntimeError("memory-governor thread did not stop during close")

    def summary(self) -> str:
        return (f"governor: {self.shrinks} shrinks, {self.restores} restores, "
                f"{self.reservations} reservations, "
                f"{self.reservation_failures} refused, "
                f"budget now {self.cache.max_bytes / 1e9:.1f}GB "
                f"(configured {self.configured_max / 1e9:.1f}GB)")
