#!/usr/bin/env python3
"""Real-checkpoint gate for governor-admitted exact Qwen2 residency.

Example:
  python tests/fixtures/qwen_lossless_resident_gate.py \
    --target ~/models/Qwen2.5-7B-Instruct

The streamed control and resident candidate use the same released weights and
greedy sampler. The gate requires byte-identical token IDs and proof telemetry
that the resident decode loop actually ran.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig  # noqa: E402
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.server import _dense_lossless_resident_bytes  # noqa: E402


PROMPTS = [
    "Give four concise reasons deterministic decoding helps reproducible tests.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Explain the latency versus throughput tradeoff in local model inference.",
]


def _close(engine):
    if engine is not None:
        engine.close()
    mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    target = args.target.expanduser()
    prompts = PROMPTS[:1] if args.quick else PROMPTS

    streamed = StreamingEngine(target, RuntimeConfig(
        max_weight_cache_mb=6000,
        prefetch_depth=4,
        prefetch_workers=2,
        pin_lm_head=True,
        embed_rows=True,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
    ))
    try:
        controls = [streamed.generate(
            prompt, max_tokens=args.max_tokens, stop=[])
            for prompt in prompts]
    finally:
        _close(streamed)
        streamed = None

    required = _dense_lossless_resident_bytes(ModelConfig.from_dir(target))
    resident = StreamingEngine(target, RuntimeConfig(
        max_weight_cache_mb=math.ceil(required * 1.07 / 1_000_000),
        prefetch_depth=4,
        prefetch_workers=2,
        pin_lm_head=True,
        embed_rows=False,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
        resident_fast_decode=True,
        resident_fast_prefill_limit=2048,
        stepped_kv_threshold=2048,
    ))
    try:
        fitted = (resident.governor.fit_cache_to_live_headroom()
                  if resident.governor is not None else resident.cache.max_bytes)
        if fitted < required:
            print(json.dumps({
                "admitted": False,
                "required_bytes": required,
                "fitted_cache_bytes": fitted,
            }, indent=2))
            return 2
        candidates = [resident.generate(
            prompt, max_tokens=args.max_tokens, stop=[])
            for prompt in prompts]
    finally:
        _close(resident)

    mismatches = [
        index for index, (control, candidate) in enumerate(
            zip(controls, candidates))
        if control["tokens"] != candidate["tokens"]
    ]
    exercised = all(
        result["path_stats"]["resident_fast_decode_sweeps"] > 0
        for result in candidates
        if len(result["tokens"]) > 1
    )
    control_s = sum(result["total_s"] for result in controls)
    resident_s = sum(result["total_s"] for result in candidates)
    report = {
        "admitted": True,
        "exact": not mismatches,
        "resident_path_exercised": exercised,
        "mismatched_prompt_indices": mismatches,
        "prompts": len(prompts),
        "tokens_compared": sum(len(result["tokens"]) for result in controls),
        "streamed_total_s": round(control_s, 4),
        "resident_total_s": round(resident_s, 4),
        "speedup": round(control_s / resident_s, 3) if resident_s else None,
        "required_bytes": required,
        "fitted_cache_bytes": fitted,
        "resident_peak_metal_bytes": max(
            result["true_peak_metal_bytes"] for result in candidates),
    }
    print(json.dumps(report, indent=2))
    return 0 if not mismatches and exercised else 1


if __name__ == "__main__":
    raise SystemExit(main())
