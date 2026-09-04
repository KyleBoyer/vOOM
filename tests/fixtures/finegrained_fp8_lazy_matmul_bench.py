#!/usr/bin/env python3
"""Measure whether deferring exact FP8 reconstruction into BF16 GEMM helps.

The candidate preserves the released FP8 -> FP32 scale -> BF16 boundary.  It
only removes the host synchronization between reconstruction and the ordinary
MLX matmul.  Results contain hashes/timings but no checkpoint tensor values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.model_loader import WeightStore
from runtime.quant import (
    FineGrainedFP8Tensor,
    dequantize_finegrained_fp8,
    dequantize_finegrained_fp8_metal,
)


def _digest(value: mx.array) -> str:
    host = np.asarray(value.view(mx.uint16))
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def _measure(call, *, warmups: int, samples: int) -> dict:
    for _ in range(warmups):
        mx.eval(call())
    values = []
    for _ in range(samples):
        started = time.perf_counter()
        mx.eval(call())
        values.append(time.perf_counter() - started)
    return {
        "median_s": statistics.median(values),
        "minimum_s": min(values),
        "samples_s": values,
    }


def _projection(
    weight: FineGrainedFP8Tensor,
    x: mx.array,
    decoder,
    *,
    synchronize_reconstruction: bool,
) -> mx.array:
    dense = decoder(
        weight.packed,
        weight.weight_scale_inv,
        block_shape=weight.block_shape,
    )
    if synchronize_reconstruction:
        mx.eval(dense)
    return x @ dense.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rows", default="2,4,8,16,32")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (not rows or any(value <= 1 for value in rows)
            or args.warmups < 0 or args.samples < 3):
        parser.error("rows must be >1, with at least three samples")

    store = WeightStore(args.model)
    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    report = {
        "schema": "voom.finegrained-fp8-lazy-matmul-bench.v1",
        "model": args.model.name,
        "layer": args.layer,
        "expert": args.expert,
        "rows": list(rows),
        "projections": [],
    }
    try:
        for projection_index, projection in enumerate(
                ("gate_proj", "up_proj", "down_proj")):
            name = f"{prefix}.{projection}.weight"
            fetched, _seconds, physical_bytes = store.fetch([name])
            weight = fetched[name]
            if not isinstance(weight, FineGrainedFP8Tensor):
                raise TypeError(f"{name} was not retained as fine-grained FP8")
            output_rows, input_cols = map(int, weight.packed.shape)
            generator = np.random.default_rng(53_900 + projection_index)
            row_results = []
            for row_count in rows:
                x = mx.array(generator.standard_normal(
                    (row_count, input_cols)).astype(np.float32)).astype(
                        mx.bfloat16)
                variants = {}
                for decoder_name, decoder in (
                    ("eager_decoder", dequantize_finegrained_fp8),
                    ("native_decoder", dequantize_finegrained_fp8_metal),
                ):
                    for sync in (True, False):
                        key = f"{decoder_name}_{'sync' if sync else 'lazy'}"
                        call = lambda d=decoder, s=sync: _projection(
                            weight, x, d, synchronize_reconstruction=s)
                        output = call()
                        mx.eval(output)
                        variants[key] = {
                            "sha256": _digest(output),
                            **_measure(
                                call,
                                warmups=args.warmups,
                                samples=args.samples,
                            ),
                        }
                reference = variants["eager_decoder_sync"]["sha256"]
                for value in variants.values():
                    value["byte_exact"] = value["sha256"] == reference
                row_results.append({
                    "rows": row_count,
                    "input_cols": input_cols,
                    "output_rows": output_rows,
                    "variants": variants,
                    "eager_lazy_speedup": (
                        variants["eager_decoder_sync"]["median_s"]
                        / variants["eager_decoder_lazy"]["median_s"]),
                    "native_lazy_speedup": (
                        variants["native_decoder_sync"]["median_s"]
                        / variants["native_decoder_lazy"]["median_s"]),
                })
            report["projections"].append({
                "name": projection,
                "physical_bytes": int(physical_bytes),
                "rows": row_results,
            })
        report["all_byte_exact"] = all(
            value["byte_exact"]
            for projection in report["projections"]
            for row in projection["rows"]
            for value in row["variants"].values()
        )
        eager_speedups = [
            row["eager_lazy_speedup"]
            for projection in report["projections"]
            for row in projection["rows"]
        ]
        native_speedups = [
            row["native_lazy_speedup"]
            for projection in report["projections"]
            for row in projection["rows"]
        ]
        report["eager_lazy_geomean_speedup"] = float(
            np.exp(np.mean(np.log(eager_speedups))))
        report["native_lazy_geomean_speedup"] = float(
            np.exp(np.mean(np.log(native_speedups))))
        report["verdict"] = "PASS" if report["all_byte_exact"] else "STOP"
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(encoded)
        print(encoded, end="")
        return 0 if report["all_byte_exact"] else 1
    finally:
        store.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
