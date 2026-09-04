#!/usr/bin/env python3
"""Exact output-tile sweep for the released fine-grained FP8 singleton QMV.

Run only after ``runtime.memory_preflight`` passes. The result stores hashes,
timings, and tensor geometry; it never persists checkpoint tensor payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
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
    _FINEGRAINED_FP8_METAL_HEADER,
    _FINEGRAINED_FP8_QMV_SOURCE,
    dequantize_finegrained_fp8,
)


def _digest(value: mx.array) -> str:
    host = np.asarray(
        value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def _kernel(rows: int, cols: int, block_rows: int, block_cols: int,
            outputs_per_simd: int):
    scale_cols = (cols + block_cols - 1) // block_cols
    source = (_FINEGRAINED_FP8_QMV_SOURCE
              .replace("OUTPUTS_PER_SIMD_VALUE", str(outputs_per_simd))
              .replace("SCALE_COLS", str(scale_cols))
              .replace("BLOCK_ROWS", str(block_rows))
              .replace("BLOCK_COLS", str(block_cols))
              .replace("ROWS", str(rows))
              .replace("COLS", str(cols)))
    return mx.fast.metal_kernel(
        name=(f"voom_finegrained_fp8_qmv_tile{outputs_per_simd}_"
              f"r{rows}_c{cols}"),
        input_names=["x", "packed", "weight_scale_inv"],
        output_names=["out"],
        header=_FINEGRAINED_FP8_METAL_HEADER,
        source=source,
        ensure_row_contiguous=False,
    )


def _run_kernel(kernel, x: mx.array, weight: FineGrainedFP8Tensor,
                outputs_per_simd: int) -> mx.array:
    rows = int(weight.packed.shape[0])
    return kernel(
        inputs=[x, weight.packed, weight.weight_scale_inv],
        template=[("T", mx.bfloat16)],
        grid=(((rows + outputs_per_simd - 1) // outputs_per_simd) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.bfloat16],
    )[0]


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
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tiles", default="1,2,4,8,16")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=81)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    tiles = tuple(int(value) for value in args.tiles.split(","))
    if (not tiles or any(value <= 0 for value in tiles)
            or args.warmups < 0 or args.samples <= 0):
        parser.error("tiles/samples must be positive and warmups non-negative")

    store = WeightStore(args.model)
    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    report = {
        "schema": "voom.finegrained-fp8-qmv-tile-bench.v1",
        "model": args.model.name,
        "layer": args.layer,
        "expert": args.expert,
        "tiles": list(tiles),
        "projections": [],
    }
    try:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.{projection}.weight"
            fetched, _seconds, physical_bytes = store.fetch([name])
            weight = fetched[name]
            if not isinstance(weight, FineGrainedFP8Tensor):
                raise TypeError(f"{name} was not retained as fine-grained FP8")
            rows, cols = map(int, weight.packed.shape)
            block_rows, block_cols = map(int, weight.block_shape)
            # Values exercise every input column without depending on random
            # generator implementation details.
            x = (mx.arange(cols, dtype=mx.float32) / max(1, cols)
                 - 0.5).astype(mx.bfloat16)
            dense = dequantize_finegrained_fp8(
                weight.packed, weight.weight_scale_inv,
                block_shape=weight.block_shape)
            reference = x @ dense.T
            mx.eval(reference)
            reference_digest = _digest(reference)
            candidates = []
            for tile in tiles:
                kernel = _kernel(
                    rows, cols, block_rows, block_cols, tile)
                call = lambda k=kernel, t=tile: _run_kernel(k, x, weight, t)
                value = call()
                mx.eval(value)
                digest = _digest(value)
                candidates.append({
                    "outputs_per_simd": tile,
                    "byte_exact": digest == reference_digest,
                    "sha256": digest,
                    **_measure(call, warmups=args.warmups, samples=args.samples),
                })
            report["projections"].append({
                "name": projection,
                "shape": [rows, cols],
                "physical_bytes": int(physical_bytes),
                "reference_sha256": reference_digest,
                "candidates": candidates,
            })
        exact = all(
            candidate["byte_exact"]
            for projection in report["projections"]
            for candidate in projection["candidates"])
        report["verdict"] = "PASS" if exact else "STOP"
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(encoded)
        print(encoded, end="")
        return 0 if exact else 1
    finally:
        store.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
