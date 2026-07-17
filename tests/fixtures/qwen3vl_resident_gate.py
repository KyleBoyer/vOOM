#!/usr/bin/env python3
"""Exact Qwen3-VL resident decode ABBA gate on a deterministic image."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.qwen3vl import generate_vl  # noqa: E402


PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image and transcribe its text.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _image() -> Image.Image:
    image = Image.new("RGB", (256, 256), (35, 90, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 70, 206, 186), fill=(240, 240, 240))
    draw.text((85, 120), "CODE 731", fill=(0, 0, 0))
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    if args.pairs <= 0:
        parser.error("--pairs must be positive")

    engine = StreamingEngine(args.model.expanduser(), RuntimeConfig(
        max_weight_cache_mb=6000,
        pin_embeddings=True,
        pin_lm_head=True,
        resident_fast_decode=True,
        resident_fast_prefill_limit=2048,
        vision_max_patches=4096,
        governor=True,
    ))
    rows = []
    exact = True
    try:
        warm = generate_vl(
            engine, PROMPT, [_image()], max_tokens=args.max_tokens)
        for pair in range(args.pairs):
            order = (False, True) if pair % 2 == 0 else (True, False)
            pair_results = []
            for resident in order:
                engine.rc.resident_fast_decode = resident
                result = generate_vl(
                    engine, PROMPT, [_image()], max_tokens=args.max_tokens)
                pair_results.append(result)
                rows.append({
                    "resident": resident,
                    "tokens": result["tokens"],
                    "decode_s": result["decode_s"],
                    "total_s": result["total_s"],
                    "pipelined_steps": result[
                        "resident_pipelined_decode_steps"],
                    "prompt_cache_exact_hit": result[
                        "vision_prompt_cache_exact_hit"],
                    "peak_metal_bytes": result["true_peak_metal_bytes"],
                })
            exact &= (
                pair_results[0]["tokens"]
                == pair_results[1]["tokens"]
                == warm["tokens"]
            )
    finally:
        engine.close()
        mx.clear_cache()

    ordinary = [row for row in rows if not row["resident"]]
    resident = [row for row in rows if row["resident"]]
    ordinary_decode = statistics.median(row["decode_s"] for row in ordinary)
    resident_decode = statistics.median(row["decode_s"] for row in resident)
    ordinary_total = statistics.median(row["total_s"] for row in ordinary)
    resident_total = statistics.median(row["total_s"] for row in resident)
    exercised = all(
        row["pipelined_steps"] == max(0, len(row["tokens"]) - 1)
        for row in resident)
    report = {
        "exact": exact,
        "resident_path_exercised": exercised,
        "pairs": args.pairs,
        "tokens_compared_per_run": len(warm["tokens"]),
        "ordinary_decode_median_s": ordinary_decode,
        "resident_decode_median_s": resident_decode,
        "decode_speedup": ordinary_decode / resident_decode,
        "ordinary_total_median_s": ordinary_total,
        "resident_total_median_s": resident_total,
        "total_speedup": ordinary_total / resident_total,
        "peak_metal_bytes": max(row["peak_metal_bytes"] for row in rows),
    }
    print(json.dumps(report, indent=2))
    return 0 if exact and exercised and report["total_speedup"] >= 1.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())
