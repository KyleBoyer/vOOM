"""Weights-free GLM-5.3 incremental DSA pool timing/equality gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from runtime.glm5_next_dsa import GLM5NextDSAState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=8192)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if (args.positions <= args.topk or args.tile <= 0
            or args.topk <= 0 or args.topk % 4):
        raise SystemExit("invalid DSA pool benchmark geometry")

    dim, pool = 128, 4
    cfg = SimpleNamespace(
        index_head_dim=dim,
        index_n_heads=32,
        index_topk=args.topk,
        index_kpool=pool,
        index_kpool_always_select_tail=True,
    )
    # Deterministic BF16 packed [key, gate, valid] rows at released geometry.
    row = mx.arange(args.positions, dtype=mx.float32)[:, None]
    column = mx.arange(dim, dtype=mx.float32)[None, :]
    keys = mx.sin((row + 1) * (column + 1) / 8192).astype(mx.bfloat16)
    gates = mx.cos((row + 3) * (column + 1) / 4096).astype(mx.bfloat16)
    valid = mx.ones((args.positions, 1), dtype=mx.bfloat16)
    packed = mx.concatenate([keys, gates, valid], axis=-1)[None]
    prefix = "model.layers.3"
    weights = {
        f"{prefix}.self_attn.indexer.index_kpool_compress_ape": (
            mx.arange(pool * dim, dtype=mx.float32).reshape(pool, dim)
            / 1024).astype(mx.bfloat16),
    }
    mx.eval(packed, *weights.values())
    ends = list(range(args.topk + args.tile, args.positions + 1, args.tile))
    if not ends or ends[-1] != args.positions:
        ends.append(args.positions)

    rebuilt = GLM5NextDSAState(cfg)
    mx.reset_peak_memory()
    started = time.perf_counter()
    expected = None
    for end in ends:
        expected, _indices, _pool_valid = rebuilt._pooled_states_reference(
            packed[:, :end, :], weights, prefix)
        mx.eval(expected)
    rebuilt_s = time.perf_counter() - started
    rebuilt_peak = int(mx.get_peak_memory())

    cached = GLM5NextDSAState(cfg, incremental_pool_cache=True)
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    actual = None
    for end in ends:
        actual, _indices, _pool_valid = cached._pooled_states(
            packed[:, :end, :], weights, prefix, layer=3)
        mx.eval(actual)
    cached_s = time.perf_counter() - started
    cached_peak = int(mx.get_peak_memory())
    assert expected is not None and actual is not None
    identical = bool(mx.array_equal(expected, actual).item())

    document = {
        "schema": "voom.glm53-dsa-pool-cache-bench.v1",
        "geometry": {
            "positions": args.positions,
            "tile": args.tile,
            "topk": args.topk,
            "pool": pool,
            "dim": dim,
            "updates": len(ends),
        },
        "full_rebuild_s": rebuilt_s,
        "incremental_s": cached_s,
        "speedup": rebuilt_s / cached_s,
        "full_rebuild_peak_metal_bytes": rebuilt_peak,
        "incremental_peak_metal_bytes": cached_peak,
        "peak_reduction_bytes": rebuilt_peak - cached_peak,
        "byte_identical_final_pool_keys": identical,
        "pool_rows_computed": cached.stats["pool_rows_computed"],
        "pool_rows_reused": cached.stats["pool_rows_reused"],
        "classification": (
            "candidate-lossless" if identical else "rejected-nonidentical"),
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    if not identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
