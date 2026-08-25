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

from runtime.dflash2 import grouped_dynamic_convolve


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
    mx.eval(hidden, dynamic, base)

    reference_call = lambda: grouped_dynamic_convolve(
        hidden, dynamic, base, group_size)
    fused_call = lambda: grouped_dynamic_convolve(
        hidden, dynamic, base, group_size, fused=True)
    for _ in range(args.warmup):
        mx.eval(reference_call(), fused_call())

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

    expected = reference_call()
    actual = fused_call()
    mx.eval(expected, actual)
    difference = np.abs(
        np.asarray(expected.astype(mx.float32))
        - np.asarray(actual.astype(mx.float32)))
    reference_median = statistics.median(reference)
    fused_median = statistics.median(fused)
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
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "exact_elements": int(np.count_nonzero(difference == 0)),
        "elements": int(difference.size),
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
