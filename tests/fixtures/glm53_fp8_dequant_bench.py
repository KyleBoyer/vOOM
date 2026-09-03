#!/usr/bin/env python3
"""Real-checkpoint gate for GLM-5.3's native fine-grained FP8 decoder.

This deliberately benchmarks the representation transform in isolation: the
released uint8 E4M3 payload and its released FP32 128x128 multipliers remain
resident while each decoder builds and evaluates a fresh BF16 output.  Kernel
compilation is warmed outside the samples.  Byte equality, not tolerance, is
the admission gate for any end-to-end profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from runtime.model_loader import WeightStore
from runtime.quant import (
    dequantize_finegrained_fp8,
    dequantize_finegrained_fp8_metal,
)


DEFAULT_NAMES = (
    "model.layers.3.mlp.experts.0.gate_proj.weight",
    "model.layers.3.mlp.shared_experts.gate_proj.weight",
    "model.layers.3.self_attn.q_b_proj.weight",
)


def _hash_bf16(value: mx.array) -> str:
    return hashlib.sha256(
        np.asarray(value.view(mx.uint16)).tobytes()
    ).hexdigest()


def _load_physical(store: WeightStore, name: str) -> mx.array:
    shard = store.weight_map[name]
    values = store._load_shard(store.dir / shard)
    value = values[store._real_name.get(name, name)]
    mx.eval(value)
    return value


def _sample(function, packed, scale, block_shape, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = function(packed, scale, block_shape=block_shape)
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        del output
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/GLM-5.3"))
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("repeats must be positive")

    store = WeightStore(args.model)
    block = store.quantization.get("weight_block_size")
    if not isinstance(block, (list, tuple)) or len(block) != 2:
        raise SystemExit("checkpoint has no two-dimensional FP8 block shape")
    block_shape = (int(block[0]), int(block[1]))
    rows = []
    for name in args.names:
        aux = store._glm53_fp8_aux.get(name)
        if aux is None:
            raise SystemExit(f"{name!r} is not a released GLM FP8 pair")
        packed = _load_physical(store, aux.weight)
        scale = _load_physical(store, aux.scale)

        # Warm compilation and both MLX graphs outside the samples.
        eager = dequantize_finegrained_fp8(
            packed, scale, block_shape=block_shape)
        native = dequantize_finegrained_fp8_metal(
            packed, scale, block_shape=block_shape)
        mx.eval(eager, native)
        eager_hash = _hash_bf16(eager)
        native_hash = _hash_bf16(native)
        if eager_hash != native_hash:
            raise RuntimeError(f"native FP8 decoder changed BF16 bytes for {name}")
        del eager, native

        eager_samples = _sample(
            dequantize_finegrained_fp8, packed, scale, block_shape,
            args.repeats)
        native_samples = _sample(
            dequantize_finegrained_fp8_metal, packed, scale, block_shape,
            args.repeats)
        eager_median = statistics.median(eager_samples)
        native_median = statistics.median(native_samples)
        raw_bytes = int(packed.nbytes + scale.nbytes)
        rows.append({
            "name": name,
            "shape": list(map(int, packed.shape)),
            "scale_shape": list(map(int, scale.shape)),
            "raw_bytes": raw_bytes,
            "bf16_bytes": int(packed.size * 2),
            "output_sha256": eager_hash,
            "byte_exact": True,
            "eager_samples_s": eager_samples,
            "eager_median_s": eager_median,
            "native_samples_s": native_samples,
            "native_median_s": native_median,
            "native_speedup": eager_median / native_median,
            "eager_raw_gbps": raw_bytes / eager_median / 1e9,
            "native_raw_gbps": raw_bytes / native_median / 1e9,
        })
        del packed, scale
        mx.clear_cache()

    result = {
        "schema": "voom.glm53-fp8-dequant-bench.v1",
        "model": str(args.model),
        "block_shape": list(block_shape),
        "repeats": args.repeats,
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
