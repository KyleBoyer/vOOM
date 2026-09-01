"""Weights-free full-GLM selected absorbed-MLA tile-width gate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import mlx.core as mx

from runtime.glm import _mla_selected_absorbed_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--tiles", default="4,8,16,32")
    parser.add_argument("--selected", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    tiles = tuple(int(value) for value in args.tiles.split(","))
    if (min((args.positions, args.selected, args.repetitions, *tiles)) <= 0
            or any(args.positions % tile for tile in tiles)):
        parser.error("positions must be divisible by every positive tile")

    batch, heads = 1, 64
    nope, rope, value, latent, hidden = 192, 64, 256, 512, 6144
    prefix = "model.layers.3"
    cfg = SimpleNamespace(v_head_dim=value, mla_use_output_gate=False)
    mx.random.seed(53)
    q_nope = mx.random.normal(
        (batch, heads, args.positions, nope)).astype(mx.bfloat16)
    q_rope = mx.random.normal(
        (batch, heads, args.positions, rope)).astype(mx.bfloat16)
    selected = mx.random.normal(
        (batch, args.positions, args.selected, latent + rope)
    ).astype(mx.bfloat16)
    valid = mx.ones(
        (batch, args.positions, args.selected), dtype=mx.bool_)
    h = mx.random.normal(
        (batch, args.positions, hidden)).astype(mx.bfloat16)
    weights = {
        f"{prefix}.self_attn.kv_b_proj.weight": mx.random.normal(
            (heads * (nope + value), latent)).astype(mx.bfloat16),
        f"{prefix}.self_attn.o_proj.weight": mx.random.normal(
            (hidden, heads * value)).astype(mx.bfloat16),
    }
    mx.eval(q_nope, q_rope, selected, valid, h, *weights.values())

    def run(tile: int) -> mx.array:
        outputs = []
        for start in range(0, args.positions, tile):
            end = start + tile
            outputs.append(_mla_selected_absorbed_attention(
                q_nope[:, :, start:end],
                q_rope[:, :, start:end],
                selected[:, start:end],
                valid[:, start:end],
                weights,
                prefix,
                cfg,
                h[:, start:end],
            ))
        return mx.concatenate(outputs, axis=1)

    baseline = run(tiles[0])
    mx.eval(baseline)
    rows = []
    all_identical = True
    for tile in tiles:
        candidate = run(tile)
        mx.eval(candidate)
        identical = bool(mx.array_equal(baseline, candidate).item())
        all_identical = all_identical and identical
        max_abs = float(mx.max(mx.abs(
            baseline.astype(mx.float32)
            - candidate.astype(mx.float32))).item())
        samples = []
        peak = 0
        for _ in range(args.repetitions):
            mx.clear_cache()
            mx.reset_peak_memory()
            started = time.perf_counter()
            value_out = run(tile)
            mx.eval(value_out)
            samples.append(time.perf_counter() - started)
            peak = max(peak, int(mx.get_peak_memory()))
        rows.append({
            "tile": tile,
            "samples_s": samples,
            "median_s": statistics.median(samples),
            "peak_metal_bytes": peak,
            "byte_identical_to_first_tile": identical,
            "max_abs_to_first_tile": max_abs,
        })
    print(json.dumps({
        "schema": "voom.glm-full-absorbed-tile-bench.v1",
        "geometry": {
            "positions": args.positions,
            "selected": args.selected,
            "heads": heads,
            "nope_dim": nope,
            "rope_dim": rope,
            "value_dim": value,
            "latent_dim": latent,
            "hidden": hidden,
            "repetitions": args.repetitions,
        },
        "rows": rows,
        "all_byte_identical": all_identical,
        "classification": (
            "lossless-schedule" if all_identical
            else "greedy-gated-floating-reassociation"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
