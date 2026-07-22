"""Real, on-demand local disk throughput measurement.

2026-07-20: audited the whole tree for hardcoded disk-speed constants after
a stale "315 MB/s" figure (measured years ago on a USB-attached SSD, now
replaced by a PCIe NVMe volume measuring ~3.0 GB/s) turned up as a default
in two offline planning tools. Neither was wired into the live server path
(runtime/server.py, engine.py, pressure.py's governor, weight_cache.py,
model_loader.py, formats/packed.py, formats/packed2.py all confirmed free of
any hardcoded throughput assumption -- placement/eviction/reservation
decisions there are driven by live psutil/mx.get_active_memory samples, not
an assumed disk speed), but a stale default in a *planning* tool still
produces a misleading estimate. Rather than swap one hardcoded number for
another that will just as quietly go stale the next time storage changes,
this measures the real, current device on demand.

Pure Python, no MLX -- safe to import from runtime/expert_plan.py, which
deliberately stays weight/MLX-free.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskProfile:
    """Measured device profile, with cached and device reads kept distinct."""

    path: str
    sequential_mb_per_s: float
    scattered_mb_per_s: dict[int, float]
    uncached_requested: bool
    uncached_applied: bool


def _set_uncached(fd: int, enabled: bool) -> bool:
    """Disable new cache population on Darwin without making it a dependency.

    Apple documents ``F_NOCACHE`` as disabling data caching for a descriptor.
    It does not promise to evict pages that were cached before this descriptor
    was opened, so repeated runs over a small file can still be cache hits.
    Other platforms retain the old best-effort random-offset behavior and
    report that the request was not applied.
    """
    if not enabled or sys.platform != "darwin":
        return False
    try:
        import fcntl

        command = getattr(fcntl, "F_NOCACHE", None)
        if command is None:
            return False
        fcntl.fcntl(fd, command, 1)
        return True
    except (ImportError, OSError):
        return False


def _aligned_random_offset(
        rng: random.Random, size: int, length: int, alignment: int = 4096,
) -> int:
    upper = max(0, size - length)
    return rng.randint(0, upper // alignment) * alignment if upper else 0


def _largest_file(root: Path, min_bytes: int) -> Path | None:
    if root.is_file():
        return root if root.stat().st_size >= min_bytes else None
    best, best_size = None, 0
    for path in root.rglob("*"):
        if path.is_file():
            size = path.stat().st_size
            if size > best_size:
                best, best_size = path, size
    return best if best_size >= min_bytes else None


def measure_sequential_mb_per_s(
    root: str | Path, sample_bytes: int = 200_000_000, chunk_bytes: int = 4_000_000,
    min_file_bytes: int = 10_000_000, *, uncached: bool = False,
) -> float:
    """Real sequential-read throughput (MB/s) of the largest file under
    `root` (or `root` itself, if it is already a file).

    A random start offset means repeated calls do not always warm the same
    page-cache region, but this is still a single point measurement, not
    the fuller chunk-size-vs-bandwidth curve `experiments/f66_storage_trace.py`
    can produce when that level of detail is worth the extra wall time.
    """
    path = _largest_file(Path(root), min_file_bytes)
    if path is None:
        raise ValueError(
            f"no file >={min_file_bytes / 1e6:.0f}MB found under {root} "
            "to measure real disk throughput")
    size = path.stat().st_size
    sample_bytes = min(sample_bytes, size)
    start = _aligned_random_offset(
        random.SystemRandom(), size, sample_bytes)
    fd = os.open(path, os.O_RDONLY)
    try:
        _set_uncached(fd, uncached)
        read = 0
        t0 = time.perf_counter()
        while read < sample_bytes:
            chunk = os.pread(fd, min(chunk_bytes, sample_bytes - read), start + read)
            if not chunk:
                break
            read += len(chunk)
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    if dt <= 0 or read <= 0:
        raise ValueError(f"disk throughput measurement read 0 bytes from {path}")
    return read / 1e6 / dt


def measure_scattered_mb_per_s(
    root: str | Path,
    chunk_sizes: tuple[int, ...] = (16_384, 65_536, 262_144, 1_048_576),
    target_bytes: int = 16_000_000,
    min_file_bytes: int = 10_000_000,
    *,
    uncached: bool = False,
    seed: int | None = None,
) -> tuple[dict[int, float], bool]:
    """Measure aligned random ``pread`` throughput by request granularity."""
    path = _largest_file(Path(root), min_file_bytes)
    if path is None:
        raise ValueError(
            f"no file >={min_file_bytes / 1e6:.0f}MB found under {root} "
            "to measure real disk throughput")
    size = path.stat().st_size
    rng = random.Random(time.time_ns() if seed is None else seed)
    output: dict[int, float] = {}
    applied = False
    fd = os.open(path, os.O_RDONLY)
    try:
        applied = _set_uncached(fd, uncached)
        for raw_chunk in chunk_sizes:
            chunk = min(int(raw_chunk), size)
            if chunk <= 0:
                raise ValueError("chunk sizes must be positive")
            reads = max(8, int(target_bytes) // chunk)
            offsets = [
                _aligned_random_offset(rng, size, chunk)
                for _ in range(reads)
            ]
            started = time.perf_counter()
            read_bytes = sum(len(os.pread(fd, chunk, offset))
                             for offset in offsets)
            elapsed = time.perf_counter() - started
            if elapsed <= 0 or read_bytes <= 0:
                raise ValueError(
                    f"scattered throughput measurement read 0 bytes from {path}")
            output[int(raw_chunk)] = read_bytes / 1e6 / elapsed
    finally:
        os.close(fd)
    return output, applied


def measure_disk_profile(
    root: str | Path,
    *,
    sample_bytes: int = 200_000_000,
    scattered_target_bytes: int = 16_000_000,
    min_file_bytes: int = 10_000_000,
    uncached: bool = True,
) -> DiskProfile:
    """Measure one tier and report whether Darwin ``F_NOCACHE`` was applied.

    For a credible device number, use a file/range larger than the page cache
    and treat the first run as authoritative. ``uncached_applied`` means new
    caching was disabled; it is deliberately not named ``cold`` because the OS
    may satisfy reads from pages populated before this call.
    """
    scattered, applied = measure_scattered_mb_per_s(
        root, target_bytes=scattered_target_bytes,
        min_file_bytes=min_file_bytes, uncached=uncached)
    sequential = measure_sequential_mb_per_s(
        root, sample_bytes=sample_bytes, min_file_bytes=min_file_bytes,
        uncached=uncached)
    return DiskProfile(
        path=str(Path(root)),
        sequential_mb_per_s=sequential,
        scattered_mb_per_s=scattered,
        uncached_requested=bool(uncached),
        uncached_applied=applied,
    )
