#!/usr/bin/env python3
"""Benchmark the byte-exact SIMD-striped Qwen DeltaNet prefill candidate.

Run only after ``runtime.memory_preflight`` passes.  The result contains
synthetic timings and byte-equality witnesses, never model or prompt data.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.qwen35 import (
    _compiled_gated_delta_rule,
    _native_striped_gated_delta_prefill,
)


def _inputs(length: int, heads: int, dim: int):
    rng = np.random.default_rng(381_053)
    shape = (1, length, heads, dim)
    q = mx.array(rng.normal(0, 0.04, shape).astype(np.float32))
    k = mx.array(rng.normal(0, 0.04, shape).astype(np.float32))
    v = mx.array(rng.normal(0, 0.04, shape).astype(np.float32))
    beta = mx.array(
        rng.uniform(0.1, 0.9, (1, length, heads)).astype(np.float32))
    decay = mx.array(
        -rng.uniform(0.001, 0.08, (1, length, heads)).astype(np.float32))
    state = mx.array(
        rng.normal(0, 0.01, (1, heads, dim, dim)).astype(np.float32))
    mx.eval(q, k, v, beta, decay, state)
    return q, k, v, beta, decay, state


def _run(fn, args, repeats: int) -> tuple[list[float], tuple[mx.array, mx.array]]:
    # First invocation includes shape-specialized graph/kernel compilation and
    # is intentionally excluded from steady-state timings.
    result = fn(*args)
    mx.eval(*result)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn(*args)
        mx.eval(*result)
        samples.append(time.perf_counter() - started)
    return samples, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if min(args.length, args.heads, args.dim, args.repeats) <= 0:
        parser.error("length, heads, dim, and repeats must be positive")
    if args.length <= 1:
        parser.error("length must exceed one")

    inputs = _inputs(args.length, args.heads, args.dim)
    compiled_samples, compiled = _run(
        _compiled_gated_delta_rule, inputs, args.repeats)
    striped_samples, striped = _run(
        _native_striped_gated_delta_prefill, inputs, args.repeats)
    compiled_host = tuple(np.asarray(value) for value in compiled)
    striped_host = tuple(np.asarray(value) for value in striped)
    matched = {
        "output": bool(np.array_equal(compiled_host[0], striped_host[0])),
        "state": bool(np.array_equal(compiled_host[1], striped_host[1])),
    }
    compiled_median = statistics.median(compiled_samples)
    striped_median = statistics.median(striped_samples)
    report = {
        "schema": "voom.qwen-striped-delta-prefill-bench.v1",
        "geometry": {
            "batch": 1,
            "length": args.length,
            "heads": args.heads,
            "key_dim": args.dim,
            "value_dim": args.dim,
            "dtype": "float32",
        },
        "compiled_reference_seconds": compiled_samples,
        "striped_fused_seconds": striped_samples,
        "compiled_reference_median_seconds": compiled_median,
        "striped_fused_median_seconds": striped_median,
        "speedup": compiled_median / striped_median,
        "matched": matched,
        "verdict": "PASS" if all(matched.values()) else "STOP",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    return 0 if all(matched.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
