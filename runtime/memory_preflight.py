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
) -> dict:
    """Return a structured admission decision for two pressure samples."""
    swap_growth = max(0, end.swap_used_bytes - start.swap_used_bytes)
    swap_out_growth = max(0, end.swap_out_bytes - start.swap_out_bytes)
    root_ok = min(start.root_free_bytes, end.root_free_bytes) >= min_root_free_bytes
    clean_swap = min(start.swap_free_bytes, end.swap_free_bytes) >= (
        min_clean_swap_free_bytes)
    stable_stale_swap = (
        min(start.system_available_bytes, end.system_available_bytes)
        >= min_stable_available_bytes
        and swap_growth <= max_swap_growth_bytes
        and swap_out_growth <= max_swap_out_growth_bytes
    )
    passed = root_ok and (clean_swap or stable_stale_swap)
    reasons = []
    if not root_ok:
        reasons.append("root_free_below_minimum")
    if not clean_swap and not stable_stale_swap:
        if min(start.system_available_bytes, end.system_available_bytes) < (
                min_stable_available_bytes):
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
    args = parser.parse_args()
    if args.sample_seconds < 0:
        parser.error("sample-seconds must be nonnegative")
    for name, value in vars(args).items():
        if name.endswith(("_gb", "_mb")) and value < 0:
            parser.error(f"{name.replace('_', '-')} must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    start = capture(args.workspace)
    if args.sample_seconds:
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
    )
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
    _atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
