#!/usr/bin/env python3
"""Released-weight timing/equality gate for full GLM dense MLP tiles.

The first three GLM-5.3 blocks are dense.  Long-context inference evaluates
their MLP rows in bounded position tiles so the full intermediate never has to
be resident.  This fixture finds the widest useful exact tile on the current
machine using the released layer-0 weights.
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

from runtime.config import ModelConfig
from runtime.glm import _glm_mlp_residual
from runtime.model_loader import WeightStore


def _hash_bf16(value: mx.array) -> str:
    raw = np.asarray(value.view(mx.uint16))
    return hashlib.sha256(raw.tobytes()).hexdigest()


def _run_tiled(
    x: mx.array,
    weights: dict[str, mx.array],
    cfg: ModelConfig,
    prefix: str,
    width: int,
) -> tuple[mx.array, float, int]:
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = mx.concatenate([
        _glm_mlp_residual(
            x[:, start:start + width], weights, prefix, cfg, 0,
            lambda *_args, **_kwargs: {},
        )
        for start in range(0, int(x.shape[1]), width)
    ], axis=1)
    mx.eval(output)
    elapsed = time.perf_counter() - started
    return output, elapsed, int(mx.get_peak_memory())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/GLM-5.3"))
    parser.add_argument("--positions", type=int, default=4096)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=(256, 512, 1024, 2048))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.positions <= 0 or args.repeats <= 0:
        raise SystemExit("positions and repeats must be positive")
    if any(width <= 0 for width in args.widths):
        raise SystemExit("tile widths must be positive")

    cfg = ModelConfig.from_dir(args.model)
    prefix = "model.layers.0"
    names = [
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]
    store = WeightStore(args.model)
    weights, fetch_s, physical_bytes = store.fetch(names)
    mx.eval(list(weights.values()))

    row = (
        (mx.arange(cfg.hidden_size, dtype=mx.float32) % 257.0) - 128.0
    ) / 256.0
    x = mx.broadcast_to(
        row.astype(mx.bfloat16), (1, args.positions, cfg.hidden_size))
    mx.eval(x)

    rows = []
    reference_hash = None
    for width in args.widths:
        samples = []
        peaks = []
        output_hash = None
        for _ in range(args.repeats):
            output, elapsed, peak = _run_tiled(
                x, weights, cfg, prefix, width)
            current_hash = _hash_bf16(output)
            output_hash = current_hash if output_hash is None else output_hash
            if current_hash != output_hash:
                raise RuntimeError(
                    f"non-deterministic output for tile width {width}")
            samples.append(elapsed)
            peaks.append(peak)
            del output
        if reference_hash is None:
            reference_hash = output_hash
        exact = output_hash == reference_hash
        rows.append({
            "tile_width": width,
            "samples_s": samples,
            "median_s": statistics.median(samples),
            "peak_bytes": max(peaks),
            "output_sha256": output_hash,
            "reference_exact": exact,
        })
        if not exact:
            raise RuntimeError(
                f"tile width {width} changed released BF16 output bytes")

    result = {
        "model": str(args.model),
        "positions": args.positions,
        "hidden_size": cfg.hidden_size,
        "fetch_s": fetch_s,
        "physical_bytes": physical_bytes,
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
