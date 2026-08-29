"""Weights-free timing gate for the explicit GLM-5.3 sparse Metal kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from runtime.glm5_next import _glm5_next_sparse_expanded_attention
from runtime.glm5_next_sparse_fused import (
    glm5_next_sparse_fused_attention,
)


def _timed(call, repetitions: int) -> tuple[list[float], int, mx.array]:
    samples = []
    peak = 0
    output = None
    for _ in range(repetitions):
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = call()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        peak = max(peak, int(mx.get_peak_memory()))
    return samples, peak, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, default=8192)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.keys < args.topk or min(
            args.keys, args.queries, args.topk, args.repetitions) <= 0:
        raise SystemExit("invalid sparse benchmark geometry")

    batch, heads, key_dim, value_dim = 1, 64, 192, 128
    mx.random.seed(53)
    query = mx.random.normal(
        (batch, heads, args.queries, key_dim)).astype(mx.bfloat16)
    keys = mx.random.normal(
        (batch, heads, args.keys, key_dim)).astype(mx.bfloat16)
    values = mx.random.normal(
        (batch, heads, args.keys, value_dim)).astype(mx.bfloat16)
    selection = mx.stack([
        mx.stack([
            (mx.arange(args.topk, dtype=mx.int32) + row * 17) % args.keys
            for row in range(args.queries)
        ])
    ])
    mx.eval(query, keys, values, selection)

    def reference():
        return _glm5_next_sparse_expanded_attention(
            query, keys, values, selection,
            key_dim=key_dim, query_tile_size=4)

    def fused():
        return glm5_next_sparse_fused_attention(
            query, keys, values, selection, key_dim=key_dim)

    # Compile/warm each path outside the recorded medians.
    mx.eval(reference(), fused())
    reference_samples, reference_peak, expected = _timed(
        reference, args.repetitions)
    fused_samples, fused_peak, candidate = _timed(
        fused, args.repetitions)
    expected32 = expected.astype(mx.float32)
    candidate32 = candidate.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(expected32 - candidate32)).item())
    cosine = float((
        mx.sum(expected32 * candidate32)
        / mx.sqrt(mx.sum(expected32 ** 2) * mx.sum(candidate32 ** 2))
    ).item())
    reference_median = statistics.median(reference_samples)
    fused_median = statistics.median(fused_samples)
    document = {
        "schema": "voom.glm53-sparse-fused-bench.v1",
        "geometry": {
            "batch": batch,
            "heads": heads,
            "queries": args.queries,
            "keys": args.keys,
            "topk": args.topk,
            "key_dim": key_dim,
            "value_dim": value_dim,
        },
        "reference": {
            "samples_s": reference_samples,
            "median_s": reference_median,
            "peak_metal_bytes": reference_peak,
        },
        "fused": {
            "samples_s": fused_samples,
            "median_s": fused_median,
            "peak_metal_bytes": fused_peak,
        },
        "speedup": reference_median / fused_median,
        "peak_reduction_bytes": reference_peak - fused_peak,
        "byte_identical": bool(mx.array_equal(expected, candidate).item()),
        "max_abs": max_abs,
        "cosine": cosine,
        "classification": "lossy-floating-reassociation",
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
