#!/usr/bin/env python3
"""Bounded FreeToken CPU/GPU split admission probe for Apple unified memory.

FreeToken's q* policy sends cache-missed MoE experts either through PCIe to a
discrete GPU or directly through host CPU GEMMs. Huihui Qwen3.8-27B is dense
and this M4 shares one DRAM fabric between CPU and GPU, so the transferable
question is narrower: can splitting independent output rows between MLX CPU
and Metal beat a full Metal BF16 projection while preserving its exact bytes?

The default shape is one released Qwen gate projection (1x5120 by
17408x5120). The job uses synthetic BF16 arrays, no checkpoint weights, and
refuses a failed/stale vOOM memory preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _fresh_preflight(path: Path, max_age_seconds: float) -> dict:
    value = json.loads(path.read_text())
    if not value.get("passed"):
        raise ValueError("memory preflight did not pass")
    ended = float(value.get("end", {}).get("monotonic_s", -1))
    age = time.monotonic() - ended
    if age < 0 or age > max_age_seconds:
        raise ValueError(
            f"memory preflight is stale ({age:.1f}s; maximum "
            f"{max_age_seconds:.1f}s)")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run_partition(x, weight, reference, cpu_fraction: float, repeats: int):
    rows = int(weight.shape[0])
    cpu_rows = min(rows, max(0, round(rows * cpu_fraction)))
    timings = []
    exact = True
    max_abs = 0.0
    for _ in range(repeats):
        mx.clear_cache()
        started = time.perf_counter()
        cpu_out = (
            mx.matmul(x, weight[:cpu_rows].T, stream=mx.cpu)
            if cpu_rows else None)
        gpu_out = (
            mx.matmul(x, weight[cpu_rows:].T, stream=mx.gpu)
            if cpu_rows < rows else None)
        outputs = [value for value in (cpu_out, gpu_out) if value is not None]
        mx.eval(*outputs)
        timings.append(time.perf_counter() - started)
        if cpu_out is not None:
            equal = bool(mx.array_equal(
                cpu_out, reference[:, :cpu_rows]).item())
            exact = exact and equal
            error = mx.max(mx.abs(
                cpu_out.astype(mx.float32)
                - reference[:, :cpu_rows].astype(mx.float32)))
            mx.eval(error)
            max_abs = max(max_abs, float(error.item()))
        if gpu_out is not None:
            equal = bool(mx.array_equal(
                gpu_out, reference[:, cpu_rows:]).item())
            exact = exact and equal
            error = mx.max(mx.abs(
                gpu_out.astype(mx.float32)
                - reference[:, cpu_rows:].astype(mx.float32)))
            mx.eval(error)
            max_abs = max(max_abs, float(error.item()))
    return {
        "cpu_fraction": cpu_fraction,
        "cpu_rows": cpu_rows,
        "gpu_rows": rows - cpu_rows,
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "timings_s": timings,
        "array_equal_to_full_metal": exact,
        "max_abs_error": max_abs,
    }


def run(args: argparse.Namespace) -> dict:
    preflight = _fresh_preflight(args.preflight, args.max_preflight_age)
    if args.hidden <= 0 or args.output_rows <= 0 or args.repeats <= 0:
        raise ValueError("hidden, output, and repeats must be positive")
    fractions = tuple(float(value) for value in args.cpu_fractions.split(","))
    if not fractions or any(value < 0 or value > 1 for value in fractions):
        raise ValueError("CPU fractions must be comma-separated values in [0,1]")

    mx.random.seed(args.seed)
    x = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, args.hidden), dtype=mx.bfloat16)
    weight = mx.random.uniform(
        low=-1.0, high=1.0,
        shape=(args.output_rows, args.hidden), dtype=mx.bfloat16)
    mx.eval(x, weight)
    mx.reset_peak_memory()

    # Full-Metal reference and warmup. Partitioned rows are compared to this
    # released-runtime operation, not merely to a tolerance.
    reference = mx.matmul(x, weight.T, stream=mx.gpu)
    mx.eval(reference)
    rows = [
        _run_partition(x, weight, reference, fraction, args.repeats)
        for fraction in fractions
    ]
    baseline = next((row for row in rows if row["cpu_fraction"] == 0), None)
    if baseline is None:
        raise ValueError("CPU fractions must include 0 for the Metal baseline")
    for row in rows:
        row["speedup_vs_full_metal"] = (
            baseline["median_s"] / row["median_s"])
    exact_rows = [row for row in rows if row["array_equal_to_full_metal"]]
    best_exact = max(exact_rows, key=lambda row: row["speedup_vs_full_metal"])
    useful_cpu = [
        row for row in exact_rows
        if row["cpu_fraction"] > 0 and row["speedup_vs_full_metal"] >= 1.05
    ]
    result = {
        "schema": "voom.freetoken-cpu-probe.v1",
        "preflight": preflight,
        "shape": {
            "batch": 1, "hidden": args.hidden, "output": args.output_rows},
        "dtype": "bfloat16",
        "repeats": args.repeats,
        "rows": rows,
        "peak_metal_bytes": int(mx.get_peak_memory()),
        "decision": {
            "verdict": "REOPEN" if useful_cpu else "STOP",
            "best_exact_cpu_fraction": best_exact["cpu_fraction"],
            "best_exact_speedup": best_exact["speedup_vs_full_metal"],
            "threshold": 1.05,
            "reason": (
                "At least one nonzero CPU split is byte-identical and 5% faster."
                if useful_cpu else
                "No nonzero CPU split is both byte-identical and 5% faster than Metal."
            ),
            "applicability": (
                "FreeToken q* targets cache-missed MoE experts on discrete GPUs; "
                "this dense model has no expert-miss branch and Apple CPU/Metal "
                "share unified-memory bandwidth."
            ),
        },
    }
    mx.clear_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-preflight-age", type=float, default=300.0)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--output-rows", type=int, default=17408)
    parser.add_argument("--cpu-fractions", default="0,0.125,0.25,0.5,1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    result = run(args)
    if args.output:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
