#!/usr/bin/env python3
"""Fail-closed host-memory admission without importing MLX.

macOS does not eagerly page stale anonymous data back in merely because the
applications that caused pressure have closed. On a host whose entire swap is
2 GiB, the historical requirement "swap free >= 2 GB" therefore stays red
after any nonzero swap use even when many gigabytes of unified memory are
available and swap-out activity has stopped.

This gate preserves the clean-swap rule and adds one conservative alternative:
stable stale swap is admissible only when system-available memory is high,
swap usage does not grow, and swap-out churn remains below a material bound
during a sampling window. Runtime allocation safety remains the responsibility
of MemoryGovernor and the <=8.5 GB true-Metal gate; this module never imports
MLX or authorizes an allocation by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class PressureSnapshot:
    monotonic_s: float
    system_available_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int
    swap_free_bytes: int
    swap_in_bytes: int
    swap_out_bytes: int
    root_free_bytes: int
    workspace_free_bytes: int


def capture(workspace: Path) -> PressureSnapshot:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return PressureSnapshot(
        monotonic_s=time.monotonic(),
        system_available_bytes=int(virtual.available),
        swap_total_bytes=int(swap.total),
        swap_used_bytes=int(swap.used),
        swap_free_bytes=int(swap.free),
        swap_in_bytes=int(swap.sin),
        swap_out_bytes=int(swap.sout),
        root_free_bytes=int(shutil.disk_usage("/").free),
        workspace_free_bytes=int(shutil.disk_usage(workspace).free),
    )


def evaluate(
    start: PressureSnapshot,
    end: PressureSnapshot,
    *,
    min_clean_swap_free_bytes: int,
    min_stable_available_bytes: int,
    min_root_free_bytes: int,
    max_swap_growth_bytes: int,
    max_swap_out_growth_bytes: int,
    intermediate: tuple[PressureSnapshot, ...] = (),
) -> dict:
    """Evaluate endpoints plus any explicitly sampled intermediate observations."""
    points = (start, *intermediate, end)
    swap_growth = max(p.swap_used_bytes for p in points) - start.swap_used_bytes
    swap_out_growth = max(p.swap_out_bytes for p in points) - start.swap_out_bytes
    available_min = min(p.system_available_bytes for p in points)
    root_ok = min(p.root_free_bytes for p in points) >= min_root_free_bytes
    clean_swap = min(p.swap_free_bytes for p in points) >= min_clean_swap_free_bytes
    stable_stale_swap = (
        available_min >= min_stable_available_bytes
        and swap_growth <= max_swap_growth_bytes
        and swap_out_growth <= max_swap_out_growth_bytes
    )
    passed = root_ok and (clean_swap or stable_stale_swap)
    reasons = []
    if not root_ok:
        reasons.append("root_free_below_minimum")
    if not clean_swap and not stable_stale_swap:
        if available_min < min_stable_available_bytes:
            reasons.append("system_available_below_stable_swap_minimum")
        if swap_growth > max_swap_growth_bytes:
            reasons.append("swap_usage_growing")
        if swap_out_growth > max_swap_out_growth_bytes:
            reasons.append("swap_outs_active")
    return {
        "verdict": "PASS" if passed else "DEFERRED_PRECONDITION",
        "passed": passed,
        "admission_path": (
            "clean_swap" if clean_swap else
            "stable_stale_swap" if stable_stale_swap else
            "none"
        ),
        "root_ok": root_ok,
        "clean_swap": clean_swap,
        "stable_stale_swap": stable_stale_swap,
        "swap_growth_bytes": swap_growth,
        "swap_out_growth_bytes": swap_out_growth,
        "reasons": reasons,
    }


def sample_pressure_window(seconds, *, sample, activity=None,
                           clock=time.monotonic, sleep=time.sleep):
    """Bounded two-second observations, not continuous or atomic coverage.

    No MLX, pressure responses, or process controls. Errors/caps cannot certify
    a stable window, even when the endpoints happen to be healthy.
    """
    if (type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 <= seconds <= 3600):
        raise ValueError("pressure window must be finite and in 0..3600 seconds")
    result = dict(complete=False, reason=None, snapshots=[], activity=[])
    started = clock()
    previous = started
    for _ in range(1802):
        try:
            point = sample()
            if (not isinstance(point, PressureSnapshot)
                    or type(point.monotonic_s) not in (int, float)
                    or not math.isfinite(point.monotonic_s)
                    or point.monotonic_s < previous
                    or any(type(value) is not int or value < 0 for key, value
                           in asdict(point).items() if key != "monotonic_s")):
                result['reason'] = 'invalid-pressure-sample'
                return result
            result['snapshots'].append(point)
            if activity is not None:
                result['activity'].append(activity())
            now = clock()
            if not math.isfinite(now) or now < point.monotonic_s:
                result['reason'] = 'invalid-pressure-clock'
                return result
            previous = now
            remaining = seconds - (now - started)
            if remaining <= 0:
                result['complete'] = True
                return result
            sleep(min(2.0, remaining))
        except Exception as error:
            result['reason'] = 'pressure-sampling-error'
            result['error_type'] = type(error).__name__
            return result
    result['reason'] = 'pressure-sample-limit'
    return result


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--min-clean-swap-free-gb", type=float, default=2.0)
    parser.add_argument("--min-stable-available-gb", type=float, default=6.0)
    parser.add_argument("--min-root-free-gb", type=float, default=5.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    parser.add_argument("--max-swap-out-growth-mb", type=float, default=16.0)
    parser.add_argument("--require-no-transcoders", action="store_true",
        help="Also defer if ffmpeg/HandBrakeCLI is observed anywhere in the sample window; read-only, not a general host-idle proof.")
    parser.add_argument("--sample-memory-window", action="store_true",
        help="Also sample pressure every two seconds and reject observed interior pressure; endpoints-only behavior is otherwise preserved.")
    args = parser.parse_args()
    if args.sample_seconds < 0:
        parser.error("sample-seconds must be nonnegative")
    if (args.require_no_transcoders or args.sample_memory_window) and (
            not math.isfinite(args.sample_seconds) or args.sample_seconds > 3600):
        parser.error("sampling window must be finite and at most 3600 seconds")
    for name, value in vars(args).items():
        if name.endswith(("_gb", "_mb")) and value < 0:
            parser.error(f"{name.replace('_', '-')} must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    start = capture(args.workspace)
    host_activity = None
    pressure_window = None
    if getattr(args, 'sample_memory_window', False):
        from .host_activity_witness import sample_known_transcoders, summarize_known_transcoders
        pressure_window = sample_pressure_window(args.sample_seconds,
            sample=lambda: capture(args.workspace),
            activity=sample_known_transcoders if args.require_no_transcoders else None)
        if args.require_no_transcoders:
            host_activity = summarize_known_transcoders(pressure_window['activity'])
    elif args.require_no_transcoders:
        from .host_activity_witness import sample_transcoder_window
        host_activity = sample_transcoder_window(args.sample_seconds)
    elif args.sample_seconds:
        time.sleep(args.sample_seconds)
    end = capture(args.workspace)
    decision = evaluate(
        start,
        end,
        min_clean_swap_free_bytes=int(args.min_clean_swap_free_gb * 1e9),
        min_stable_available_bytes=int(args.min_stable_available_gb * 1e9),
        min_root_free_bytes=int(args.min_root_free_gb * 1e9),
        max_swap_growth_bytes=int(args.max_swap_growth_mb * 1e6),
        max_swap_out_growth_bytes=int(args.max_swap_out_growth_mb * 1e6),
        intermediate=tuple(pressure_window['snapshots']) if pressure_window else (),
    )
    if pressure_window is not None and not pressure_window['complete']:
        decision['passed'] = False
        decision['verdict'] = 'DEFERRED_PRECONDITION'
        decision['admission_path'] = 'none'
        decision['reasons'].append('pressure_window_unavailable')
    if host_activity is not None and not host_activity['passed']:
        decision['passed'] = False
        decision['verdict'] = 'DEFERRED_PRECONDITION'
        decision['admission_path'] = 'none'
        decision['reasons'].append('known_transcoders_active' if host_activity['transcoders']
                                   else 'transcoder_inventory_unavailable')
    result = {
        "schema": "voom.memory-preflight.v1",
        "sample_seconds": end.monotonic_s - start.monotonic_s,
        "thresholds": {
            "min_clean_swap_free_bytes": int(args.min_clean_swap_free_gb * 1e9),
            "min_stable_available_bytes": int(args.min_stable_available_gb * 1e9),
            "min_root_free_bytes": int(args.min_root_free_gb * 1e9),
            "max_swap_growth_bytes": int(args.max_swap_growth_mb * 1e6),
            "max_swap_out_growth_bytes": int(args.max_swap_out_growth_mb * 1e6),
        },
        "start": asdict(start),
        "end": asdict(end),
        **decision,
    }
    if host_activity is not None:
        result['known_transcoders'] = host_activity
    if pressure_window is not None:
        points = (start, *pressure_window['snapshots'], end)
        result['pressure_window'] = dict(
            schema='voom.preflight-pressure-window.v1',
            scope='two-second sampled observations, not continuous or atomic coverage',
            complete=pressure_window['complete'], reason=pressure_window['reason'],
            error_type=pressure_window.get('error_type'),
            interval_seconds=2.0,
            sample_count=len(points), samples=[asdict(p) for p in points],
            minimum_available_bytes=min(p.system_available_bytes for p in points),
            minimum_root_free_bytes=min(p.root_free_bytes for p in points),
            peak_swap_used_bytes=max(p.swap_used_bytes for p in points),
            maximum_swap_out_bytes=max(p.swap_out_bytes for p in points))
    _atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
