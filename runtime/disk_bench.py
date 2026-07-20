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
import time
from pathlib import Path


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
    min_file_bytes: int = 10_000_000,
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
    start = random.randint(0, max(0, size - sample_bytes))
    fd = os.open(path, os.O_RDONLY)
    try:
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
