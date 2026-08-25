#!/usr/bin/env python3
"""Bounded real-geometry A/B for DFlash2 grouped dynamic convolution."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.dflash2 import (
    fused_grouped_dynamic_convolve_projected,
    grouped_dynamic_convolve,
    project_out_direction,
)


def _measure(call, repetitions: int) -> list[float]:
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(call())
        values.append(time.perf_counter() - started)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.repetitions < 20 or args.warmup < 1:
        parser.error("use at least 20 repetitions and one warmup")

    batch, length, hidden_size = 1, 5, 5120
    kernel_size, group_size = 2, 16
    groups = hidden_size // group_size
    mx.random.seed(20260824)
    hidden = mx.random.normal((batch, length, hidden_size)).astype(mx.bfloat16)
    dynamic = mx.random.normal(
        (batch, length, kernel_size, groups)).astype(mx.bfloat16)
    base = mx.random.normal((kernel_size, hidden_size)).astype(mx.bfloat16)
    direction = mx.random.normal((hidden_size,)).astype(mx.float32)
    direction = direction / mx.sqrt(mx.sum(direction * direction))
    mx.eval(hidden, dynamic, base, direction)

    reference_call = lambda: grouped_dynamic_convolve(
        hidden, dynamic, base, group_size)
    fused_call = lambda: grouped_dynamic_convolve(
        hidden, dynamic, base, group_size, fused=True)
    projected_reference_call = lambda: project_out_direction(
        reference_call(), direction, 1.3)
    projected_composed_fused_call = lambda: project_out_direction(
        fused_call(), direction, 1.3)
    projected_fused_call = lambda: fused_grouped_dynamic_convolve_projected(
        hidden, dynamic, base, direction, group_size, 1.3)
    for _ in range(args.warmup):
        mx.eval(
            reference_call(), fused_call(),
            projected_reference_call(), projected_composed_fused_call(),
            projected_fused_call())

    mx.reset_peak_memory()
    # ABBA ordering reduces first-arm thermal/cache bias while preserving each
    # individual dispatch timing rather than batching the graph evaluations.
    reference = []
    fused = []
    half = args.repetitions // 2
    reference.extend(_measure(reference_call, half))
    fused.extend(_measure(fused_call, half))
    fused.extend(_measure(fused_call, args.repetitions - half))
    reference.extend(_measure(reference_call, args.repetitions - half))
    projected_reference = _measure(projected_reference_call, args.repetitions)
    projected_composed_fused = _measure(
        projected_composed_fused_call, args.repetitions)
    projected_fused = _measure(projected_fused_call, args.repetitions)

    expected = reference_call()
    actual = fused_call()
    mx.eval(expected, actual)
    difference = np.abs(
        np.asarray(expected.astype(mx.float32))
        - np.asarray(actual.astype(mx.float32)))
    projected_expected = projected_reference_call()
    projected_actual = projected_fused_call()
    mx.eval(projected_expected, projected_actual)
    projected_difference = np.abs(
        np.asarray(projected_expected.astype(mx.float32))
        - np.asarray(projected_actual.astype(mx.float32)))
    reference_median = statistics.median(reference)
    fused_median = statistics.median(fused)
    projected_reference_median = statistics.median(projected_reference)
    projected_composed_fused_median = statistics.median(
        projected_composed_fused)
    projected_fused_median = statistics.median(projected_fused)
    report = {
        "schema": "voom.dflash2-dynamic-conv-bench.v1",
        "geometry": {
            "batch": batch,
            "length": length,
            "hidden_size": hidden_size,
            "kernel_size": kernel_size,
            "group_size": group_size,
            "groups": groups,
            "dtype": "bfloat16",
            "calls_per_five_layer_proposal_block": 20,
        },
        "warmup": args.warmup,
        "repetitions_per_arm": args.repetitions,
        "reference_median_ms": reference_median * 1000,
        "fused_median_ms": fused_median * 1000,
        "isolated_speedup": reference_median / fused_median,
        "estimated_reference_ms_per_proposal_block": reference_median * 20_000,
        "estimated_fused_ms_per_proposal_block": fused_median * 20_000,
        "projected_reference_median_ms": projected_reference_median * 1000,
        "projected_composed_fused_median_ms": (
            projected_composed_fused_median * 1000),
        "projected_fused_median_ms": projected_fused_median * 1000,
        "projected_isolated_speedup": (
            projected_reference_median / projected_fused_median),
        "projected_speedup_vs_composed_fused": (
            projected_composed_fused_median / projected_fused_median),
        "estimated_projected_reference_ms_per_proposal_block": (
            projected_reference_median * 20_000),
        "estimated_projected_fused_ms_per_proposal_block": (
            projected_fused_median * 20_000),
        "estimated_projected_composed_fused_ms_per_proposal_block": (
            projected_composed_fused_median * 20_000),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "exact_elements": int(np.count_nonzero(difference == 0)),
        "elements": int(difference.size),
        "projected_max_abs_difference": float(projected_difference.max()),
        "projected_mean_abs_difference": float(projected_difference.mean()),
        "projected_exact_elements": int(np.count_nonzero(
            projected_difference == 0)),
        "peak_metal_bytes": int(mx.get_peak_memory()),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
