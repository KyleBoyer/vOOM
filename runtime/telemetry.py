"""Memory and timing telemetry. Tracks both process RSS (psutil) and MLX Metal
allocator stats, since mx.get_active_memory() only sees Metal allocations.
"""

from __future__ import annotations

import time

import mlx.core as mx
import psutil

_PROC = psutil.Process()


def mem() -> dict[str, float]:
    return {
        "rss_mb": _PROC.memory_info().rss / 1e6,
        "mlx_active_mb": mx.get_active_memory() / 1e6,
        "mlx_peak_mb": mx.get_peak_memory() / 1e6,
        "mlx_cache_mb": mx.get_cache_memory() / 1e6,
    }


def fmt_mem(m: dict[str, float] | None = None) -> str:
    m = m or mem()
    return (
        f"rss={m['rss_mb']:.0f}MB metal_active={m['mlx_active_mb']:.0f}MB "
        f"metal_peak={m['mlx_peak_mb']:.0f}MB metal_cache={m['mlx_cache_mb']:.0f}MB"
    )


class Timer:
    """Accumulates named durations across a run."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float):
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> str:
        lines = []
        for name, total in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            lines.append(f"  {name}: total={total:.2f}s n={n} avg={total / n * 1000:.1f}ms")
        return "\n".join(lines)


class stopwatch:
    """Context manager that evals `arrays` before stopping the clock (MLX is lazy)."""

    def __init__(self, timer: Timer, name: str):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.add(self.name, time.perf_counter() - self.t0)
        return False
