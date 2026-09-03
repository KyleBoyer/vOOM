"""Memory and timing telemetry. Tracks both process RSS (psutil) and MLX Metal
allocator stats, since mx.get_active_memory() only sees Metal allocations.
"""

from __future__ import annotations

import time
from collections import defaultdict

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


class RequestProfiler:
    """Bounded, request-local execution attribution.

    ``layers`` records the synchronization boundaries the streamed runtime
    already has: layer-page wait, materialized layer compute, expert-page
    fetches, cache events, and store-accounted bytes. ``ops`` additionally
    asks supported hybrid blocks to materialize attention, router, and MLP
    boundaries separately. Those extra barriers preserve arithmetic and token
    IDs, but can perturb wall time, so the result labels them explicitly.

    All expert/substep measurements are nested inside ``compute_s``. Consumers
    must not sum them with the layer total.
    """

    LEVELS = ("", "layers", "ops")
    _CACHE_FIELDS = (
        "hits", "misses", "evictions", "bytes_read", "disk_s",
        "parallel_tier_fetches", "parallel_tier_fast_bytes",
        "parallel_tier_archive_bytes", "parallel_tier_wall_ns",
        "parallel_tier_fast_service_ns",
        "parallel_tier_archive_service_ns", "parallel_tier_hidden_ns",
        "ct_mxfp4_transform_ns", "ct_mxfp4_transform_calls",
        "ct_mxfp4_input_bytes", "ct_mxfp4_resident_bytes",
        "glm53_fp8_transform_ns", "glm53_fp8_transform_calls",
        "glm53_fp8_native_calls", "glm53_fp8_input_bytes",
        "glm53_fp8_resident_bytes",
        "glm53_fp8_prefetch_transform_ns",
        "glm53_fp8_prefetch_transform_calls",
        "glm53_fp8_prefetch_native_calls",
        "k3_scale_sidecar_read_bytes", "k3_scale_sidecar_output_bytes",
        "k3_scale_sidecar_decode_ns", "k3_scale_sidecar_decode_calls",
        "bf16_nf12_read_bytes", "bf16_nf12_output_bytes",
        "bf16_nf12_decode_ns", "bf16_nf12_decode_calls",
    )

    def __init__(self, level: str):
        level = str(level or "").strip().lower()
        if level not in self.LEVELS:
            raise ValueError(
                f"execution_profile must be one of {self.LEVELS}, got {level!r}")
        self.level = level
        self.enabled = bool(level)
        self.sync_substeps = level == "ops"
        self.phase = "unattributed"
        self.started = time.perf_counter()
        self._phases = defaultdict(lambda: {
            "sweeps": 0,
            "positions": 0,
            "paths": defaultdict(int),
        })
        self._layers = defaultdict(lambda: {
            "calls": 0,
            "positions": 0,
            "weight_wait_s": 0.0,
            "compute_s": 0.0,
            "expert_fetch_s": 0.0,
            "expert_fetch_calls": 0,
            "expert_pages": 0,
            "expert_misses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "store_bytes_read": 0,
            "store_disk_s": 0.0,
            "parallel_tier_fetches": 0,
            "parallel_tier_fast_bytes": 0,
            "parallel_tier_archive_bytes": 0,
            "parallel_tier_wall_ns": 0,
            "parallel_tier_fast_service_ns": 0,
            "parallel_tier_archive_service_ns": 0,
            "parallel_tier_hidden_ns": 0,
            "ct_mxfp4_transform_ns": 0,
            "ct_mxfp4_transform_calls": 0,
            "ct_mxfp4_input_bytes": 0,
            "ct_mxfp4_resident_bytes": 0,
            "glm53_fp8_transform_ns": 0,
            "glm53_fp8_transform_calls": 0,
            "glm53_fp8_native_calls": 0,
            "glm53_fp8_input_bytes": 0,
            "glm53_fp8_resident_bytes": 0,
            "glm53_fp8_prefetch_transform_ns": 0,
            "glm53_fp8_prefetch_transform_calls": 0,
            "glm53_fp8_prefetch_native_calls": 0,
            "k3_scale_sidecar_read_bytes": 0,
            "k3_scale_sidecar_output_bytes": 0,
            "k3_scale_sidecar_decode_ns": 0,
            "k3_scale_sidecar_decode_calls": 0,
            "bf16_nf12_read_bytes": 0,
            "bf16_nf12_output_bytes": 0,
            "bf16_nf12_decode_ns": 0,
            "bf16_nf12_decode_calls": 0,
            "substeps": defaultdict(lambda: {
                "calls": 0, "positions": 0, "wall_s": 0.0,
            }),
        })
        self._notes: set[str] = set()

    def set_phase(self, phase: str) -> None:
        if self.enabled:
            self.phase = str(phase or "unattributed")

    def begin_sweep(
        self, positions: int, *, path: str = "streamed",
        phase: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        if phase is not None:
            self.set_phase(phase)
        bucket = self._phases[self.phase]
        bucket["sweeps"] += 1
        bucket["positions"] += max(0, int(positions))
        bucket["paths"][str(path)] += 1

    @classmethod
    def cache_snapshot(cls, cache) -> tuple:
        stats = cache.stats
        stage_snapshot = getattr(getattr(cache, "store", None),
                                 "stage_snapshot", None)
        store_stages = (
            stage_snapshot() if callable(stage_snapshot) else (0, 0, 0, 0))
        glm53_fp8_snapshot = getattr(
            getattr(cache, "store", None), "glm53_fp8_snapshot", None
        )
        glm53_fp8_stages = (
            glm53_fp8_snapshot()
            if callable(glm53_fp8_snapshot)
            else (0, 0, 0, 0, 0, 0, 0, 0)
        )
        scale_snapshot = getattr(
            getattr(cache, "store", None),
            "k3_scale_sidecar_snapshot",
            None,
        )
        scale_stages = (
            scale_snapshot()
            if callable(scale_snapshot)
            else (0, 0, 0, 0)
        )
        nf12_snapshot = getattr(
            getattr(cache, "store", None), "bf16_nf12_snapshot", None
        )
        nf12_stages = (
            nf12_snapshot()
            if callable(nf12_snapshot)
            else (0, 0, 0, 0)
        )
        parallel_snapshot = getattr(
            getattr(cache, "store", None), "parallel_tier_snapshot", None
        )
        parallel_stages = (
            parallel_snapshot()
            if callable(parallel_snapshot)
            else (0, 0, 0, 0, 0, 0, 0)
        )
        return (
            int(stats.hits), int(stats.misses), int(stats.evictions),
            int(stats.bytes_read), float(stats.disk_s),
            *parallel_stages,
            *store_stages,
            *glm53_fp8_stages,
            *scale_stages,
            *nf12_stages,
        )

    @staticmethod
    def _cache_delta(before: tuple, after: tuple) -> tuple:
        return tuple(max(0, end - start)
                     for start, end in zip(before, after, strict=True))

    def _layer(self, layer: int):
        return self._layers[(self.phase, int(layer))]

    def record_layer(
        self, layer: int, *, positions: int, weight_wait_s: float,
        compute_s: float, cache_before: tuple, cache_after: tuple,
        layer_type: str,
    ) -> None:
        if not self.enabled:
            return
        bucket = self._layer(layer)
        bucket["calls"] += 1
        bucket["positions"] += max(0, int(positions))
        bucket["weight_wait_s"] += max(0.0, float(weight_wait_s))
        bucket["compute_s"] += max(0.0, float(compute_s))
        bucket["layer_type"] = str(layer_type)
        (hits, misses, evictions, bytes_read, disk_s,
         parallel_fetches, parallel_fast_bytes, parallel_archive_bytes,
         parallel_wall_ns, parallel_fast_service_ns,
         parallel_archive_service_ns, parallel_hidden_ns,
         ct_transform_ns, ct_transform_calls, ct_input_bytes,
         ct_resident_bytes, glm53_fp8_transform_ns,
         glm53_fp8_transform_calls, glm53_fp8_native_calls,
         glm53_fp8_input_bytes, glm53_fp8_resident_bytes,
         glm53_fp8_prefetch_transform_ns,
         glm53_fp8_prefetch_transform_calls,
         glm53_fp8_prefetch_native_calls,
         scale_read_bytes, scale_output_bytes,
         scale_decode_ns, scale_decode_calls,
         nf12_read_bytes, nf12_output_bytes,
         nf12_decode_ns, nf12_decode_calls) = self._cache_delta(
             cache_before, cache_after
         )
        bucket["cache_hits"] += int(hits)
        bucket["cache_misses"] += int(misses)
        bucket["cache_evictions"] += int(evictions)
        bucket["store_bytes_read"] += int(bytes_read)
        bucket["store_disk_s"] += float(disk_s)
        bucket["parallel_tier_fetches"] += int(parallel_fetches)
        bucket["parallel_tier_fast_bytes"] += int(parallel_fast_bytes)
        bucket["parallel_tier_archive_bytes"] += int(parallel_archive_bytes)
        bucket["parallel_tier_wall_ns"] += int(parallel_wall_ns)
        bucket["parallel_tier_fast_service_ns"] += int(
            parallel_fast_service_ns)
        bucket["parallel_tier_archive_service_ns"] += int(
            parallel_archive_service_ns)
        bucket["parallel_tier_hidden_ns"] += int(parallel_hidden_ns)
        bucket["ct_mxfp4_transform_ns"] += int(ct_transform_ns)
        bucket["ct_mxfp4_transform_calls"] += int(ct_transform_calls)
        bucket["ct_mxfp4_input_bytes"] += int(ct_input_bytes)
        bucket["ct_mxfp4_resident_bytes"] += int(ct_resident_bytes)
        bucket["glm53_fp8_transform_ns"] += int(glm53_fp8_transform_ns)
        bucket["glm53_fp8_transform_calls"] += int(
            glm53_fp8_transform_calls)
        bucket["glm53_fp8_native_calls"] += int(glm53_fp8_native_calls)
        bucket["glm53_fp8_input_bytes"] += int(glm53_fp8_input_bytes)
        bucket["glm53_fp8_resident_bytes"] += int(glm53_fp8_resident_bytes)
        bucket["glm53_fp8_prefetch_transform_ns"] += int(
            glm53_fp8_prefetch_transform_ns)
        bucket["glm53_fp8_prefetch_transform_calls"] += int(
            glm53_fp8_prefetch_transform_calls)
        bucket["glm53_fp8_prefetch_native_calls"] += int(
            glm53_fp8_prefetch_native_calls)
        bucket["k3_scale_sidecar_read_bytes"] += int(scale_read_bytes)
        bucket["k3_scale_sidecar_output_bytes"] += int(scale_output_bytes)
        bucket["k3_scale_sidecar_decode_ns"] += int(scale_decode_ns)
        bucket["k3_scale_sidecar_decode_calls"] += int(scale_decode_calls)
        bucket["bf16_nf12_read_bytes"] += int(nf12_read_bytes)
        bucket["bf16_nf12_output_bytes"] += int(nf12_output_bytes)
        bucket["bf16_nf12_decode_ns"] += int(nf12_decode_ns)
        bucket["bf16_nf12_decode_calls"] += int(nf12_decode_calls)

    def record_stack(
        self, *, positions: int, path: str, wall_s: float,
    ) -> None:
        """Record a fused/resident stack whose layers have no sync boundaries."""
        if not self.enabled:
            return
        bucket = self._phases[self.phase]
        bucket["stack_calls"] = int(bucket.get("stack_calls", 0)) + 1
        bucket["stack_positions"] = int(
            bucket.get("stack_positions", 0)) + max(0, int(positions))
        bucket["stack_wall_s"] = float(
            bucket.get("stack_wall_s", 0.0)) + max(0.0, float(wall_s))
        self._notes.add(
            f"{path} timing has no per-layer boundaries")

    def record_expert_fetch(
        self, layer: int, *, pages: int, misses: int, wall_s: float,
    ) -> None:
        if not self.enabled:
            return
        bucket = self._layer(layer)
        bucket["expert_fetch_calls"] += 1
        bucket["expert_pages"] += max(0, int(pages))
        bucket["expert_misses"] += max(0, int(misses))
        bucket["expert_fetch_s"] += max(0.0, float(wall_s))

    def start_substep(self) -> float | None:
        if not self.sync_substeps:
            return None
        return time.perf_counter()

    def finish_substep(
        self, name: str, layer: int, started: float | None, *arrays,
        positions: int = 0,
    ) -> bool:
        """Materialize and record one supported op boundary.

        Returns True when this method performed the synchronization, letting a
        caller retain its ordinary ``mx.eval`` only for the non-profiled case.
        """
        if started is None:
            return False
        if arrays:
            mx.eval(*arrays)
        self.record_substep(
            name, layer, time.perf_counter() - started,
            positions=positions)
        return True

    def record_substep(
        self, name: str, layer: int, wall_s: float, *, positions: int = 0,
    ) -> None:
        if not self.enabled:
            return
        step = self._layer(layer)["substeps"][str(name)]
        step["calls"] += 1
        step["positions"] += max(0, int(positions))
        step["wall_s"] += max(0.0, float(wall_s))

    def note(self, value: str) -> None:
        if self.enabled and value:
            self._notes.add(str(value))

    @staticmethod
    def _round(value: float) -> float:
        return round(float(value), 6)

    def result(self, request_wall_s: float | None = None) -> dict | None:
        if not self.enabled:
            return None
        request_wall_s = (
            time.perf_counter() - self.started
            if request_wall_s is None else max(0.0, float(request_wall_s)))
        phase_values = {}
        for phase, raw in sorted(self._phases.items()):
            phase_values[phase] = {
                "sweeps": int(raw["sweeps"]),
                "positions": int(raw["positions"]),
                "paths": dict(sorted(raw["paths"].items())),
            }
            for key in ("stack_calls", "stack_positions"):
                if key in raw:
                    phase_values[phase][key] = int(raw[key])
            if "stack_wall_s" in raw:
                phase_values[phase]["stack_wall_s"] = self._round(
                    raw["stack_wall_s"])

        layers = []
        for (phase, layer), raw in sorted(self._layers.items()):
            substeps = {
                name: {
                    "calls": int(value["calls"]),
                    "positions": int(value["positions"]),
                    "wall_s": self._round(value["wall_s"]),
                }
                for name, value in sorted(raw["substeps"].items())
            }
            item = {
                "phase": phase,
                "layer": layer,
                "layer_type": raw.get("layer_type", "unknown"),
                "calls": int(raw["calls"]),
                "positions": int(raw["positions"]),
                "weight_wait_s": self._round(raw["weight_wait_s"]),
                "compute_s": self._round(raw["compute_s"]),
                "total_s": self._round(
                    raw["weight_wait_s"] + raw["compute_s"]),
                "expert_fetch_s": self._round(raw["expert_fetch_s"]),
                "expert_fetch_calls": int(raw["expert_fetch_calls"]),
                "expert_pages": int(raw["expert_pages"]),
                "expert_misses": int(raw["expert_misses"]),
                "cache_hits": int(raw["cache_hits"]),
                "cache_misses": int(raw["cache_misses"]),
                "cache_evictions": int(raw["cache_evictions"]),
                "store_bytes_read": int(raw["store_bytes_read"]),
                "store_disk_s": self._round(raw["store_disk_s"]),
                "parallel_tier_fetches": int(
                    raw["parallel_tier_fetches"]),
                "parallel_tier_fast_bytes": int(
                    raw["parallel_tier_fast_bytes"]),
                "parallel_tier_archive_bytes": int(
                    raw["parallel_tier_archive_bytes"]),
                "parallel_tier_wall_s": self._round(
                    raw["parallel_tier_wall_ns"] / 1_000_000_000),
                "parallel_tier_fast_service_s": self._round(
                    raw["parallel_tier_fast_service_ns"] / 1_000_000_000),
                "parallel_tier_archive_service_s": self._round(
                    raw["parallel_tier_archive_service_ns"] / 1_000_000_000),
                "parallel_tier_hidden_s": self._round(
                    raw["parallel_tier_hidden_ns"] / 1_000_000_000),
                "ct_mxfp4_transform_s": self._round(
                    raw["ct_mxfp4_transform_ns"] / 1_000_000_000),
                "ct_mxfp4_transform_calls": int(
                    raw["ct_mxfp4_transform_calls"]),
                "ct_mxfp4_input_bytes": int(raw["ct_mxfp4_input_bytes"]),
                "ct_mxfp4_resident_bytes": int(
                    raw["ct_mxfp4_resident_bytes"]),
                "glm53_fp8_transform_s": self._round(
                    raw["glm53_fp8_transform_ns"] / 1_000_000_000),
                "glm53_fp8_transform_calls": int(
                    raw["glm53_fp8_transform_calls"]),
                "glm53_fp8_native_calls": int(
                    raw["glm53_fp8_native_calls"]),
                "glm53_fp8_input_bytes": int(
                    raw["glm53_fp8_input_bytes"]),
                "glm53_fp8_resident_bytes": int(
                    raw["glm53_fp8_resident_bytes"]),
                "glm53_fp8_prefetch_transform_s": self._round(
                    raw["glm53_fp8_prefetch_transform_ns"] / 1_000_000_000),
                "glm53_fp8_prefetch_transform_calls": int(
                    raw["glm53_fp8_prefetch_transform_calls"]),
                "glm53_fp8_prefetch_native_calls": int(
                    raw["glm53_fp8_prefetch_native_calls"]),
                "k3_scale_sidecar_read_bytes": int(
                    raw["k3_scale_sidecar_read_bytes"]),
                "k3_scale_sidecar_output_bytes": int(
                    raw["k3_scale_sidecar_output_bytes"]),
                "k3_scale_sidecar_decode_s": self._round(
                    raw["k3_scale_sidecar_decode_ns"] / 1_000_000_000),
                "k3_scale_sidecar_decode_calls": int(
                    raw["k3_scale_sidecar_decode_calls"]),
                "bf16_nf12_read_bytes": int(
                    raw["bf16_nf12_read_bytes"]),
                "bf16_nf12_output_bytes": int(
                    raw["bf16_nf12_output_bytes"]),
                "bf16_nf12_decode_s": self._round(
                    raw["bf16_nf12_decode_ns"] / 1_000_000_000),
                "bf16_nf12_decode_calls": int(
                    raw["bf16_nf12_decode_calls"]),
            }
            if substeps:
                item["substeps"] = substeps
            layers.append(item)

        layer_total = sum(item["total_s"] for item in layers)
        hotspots = sorted(
            ({
                "phase": item["phase"],
                "layer": item["layer"],
                "layer_type": item["layer_type"],
                "total_s": item["total_s"],
                "weight_wait_s": item["weight_wait_s"],
                "compute_s": item["compute_s"],
                "store_bytes_read": item["store_bytes_read"],
            } for item in layers),
            key=lambda item: (-item["total_s"], item["phase"], item["layer"]),
        )[:12]
        return {
            "schema_version": 1,
            "level": self.level,
            "request_wall_s": self._round(request_wall_s),
            "layer_accounted_s": self._round(layer_total),
            "phases": phase_values,
            "layers": layers,
            "hotspots": hotspots,
            "semantics": {
                "weight_wait_s": (
                    "wall around WeightCache demand lookup/materialization"),
                "compute_s": (
                    "wall through the runtime's materialization boundary; "
                    "includes nested router, expert fetch, and expert compute"),
                "expert_fetch_s": (
                    "nested wall around expert WeightCache fetch; do not add "
                    "to compute_s"),
                "store_disk_s": (
                    "nested store-accounted fetch/decode time; parallel work "
                    "may overlap"),
                "parallel_tier_service_s": (
                    "per-physical-tier nested service time; fast and archive "
                    "service overlap each other, parallel_tier_wall_s is the "
                    "critical interval, and parallel_tier_hidden_s is the "
                    "overlap lower bound"),
                "ct_mxfp4_transform_s": (
                    "nested inside store_disk_s/weight wait; eager dense "
                    "dequantization or native packed-view materialization, "
                    "depending on the active representation"),
                "glm53_fp8_transform_s": (
                    "nested inside store_disk_s/weight wait; released E4M3 "
                    "plus FP32 block-scale reconstruction to exact BF16"),
                "glm53_fp8_prefetch_transform_s": (
                    "subset of glm53_fp8_transform_s executed by "
                    "vmodel-expert-batch background workers; do not add it "
                    "to the total"),
                "k3_scale_sidecar_decode_s": (
                    "nested inside store_disk_s/weight wait; exact fused "
                    "E8M0 scale reconstruction, excluding sidecar file reads"),
                "bf16_nf12_decode_s": (
                    "nested inside store_disk_s/weight wait; exact fixed-width "
                    "BF16 bit reconstruction from a mapped sidecar"),
                "substeps": (
                    "nested inside compute_s; ops level adds synchronization "
                    "and is diagnostic, not an uninstrumented speed result"),
            },
            "notes": sorted(self._notes),
        }
