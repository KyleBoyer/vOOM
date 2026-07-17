#!/usr/bin/env python3
"""Real-checkpoint exact-token and throughput gate for dense Qwen speculation.

Example:
  python tests/fixtures/qwen_speculative_gate.py \
    --target ~/models/Qwen2.5-7B-Instruct \
    --draft ~/models/Qwen2.5-1.5B-Instruct-mlx-mxfp4 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.speculative import SpeculativeEngine  # noqa: E402


PROMPTS = [
    "Give four concise reasons deterministic decoding helps reproducible tests.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Explain the latency versus throughput tradeoff in local model inference.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    prompts = PROMPTS[:1] if args.quick else PROMPTS

    target = StreamingEngine(args.target.expanduser(), RuntimeConfig(
        max_weight_cache_mb=6000,
        prefetch_depth=2,
        pin_lm_head=True,
        embed_rows=True,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
    ))
    draft = None
    try:
        baselines = []
        for prompt in prompts:
            baselines.append(target.generate(
                prompt, max_tokens=args.max_tokens, stop=[]))

        draft = StreamingEngine(args.draft.expanduser(), RuntimeConfig(
            max_weight_cache_mb=1200,
            pin_embeddings=True,
            pin_lm_head=True,
            resident_fast_decode=True,
            resident_fast_prefill_limit=2048,
            stepped_kv_threshold=512,
            fused_swiglu=True,
        ))
        engine = SpeculativeEngine(target, draft, k=args.k, max_prompt_tokens=2048)
        speculative = [engine.generate(
            prompt, max_tokens=args.max_tokens, stop=[])
            for prompt in prompts]

        mismatches = [
            index for index, (plain, spec) in enumerate(zip(baselines, speculative))
            if plain["tokens"] != spec["tokens"]
        ]
        plain_s = sum(result["total_s"] for result in baselines)
        spec_s = sum(result["total_s"] for result in speculative)
        path_stats = [result["path_stats"] for result in speculative]
        report = {
            "exact": not mismatches,
            "mismatched_prompt_indices": mismatches,
            "prompts": len(prompts),
            "tokens_compared": sum(len(result["tokens"]) for result in baselines),
            "target_total_s": round(plain_s, 4),
            "speculative_total_s": round(spec_s, 4),
            "speedup": round(plain_s / spec_s, 3) if spec_s else None,
            "accepted": sum(stats["speculative_accepted"] for stats in path_stats),
            "proposed": sum(stats["speculative_proposed"] for stats in path_stats),
            "target_sweeps": sum(
                stats["speculative_target_sweeps"] for stats in path_stats),
            "peak_metal_bytes": max(
                result["true_peak_metal_bytes"] for result in speculative),
        }
        print(json.dumps(report, indent=2))
        return 0 if not mismatches else 1
    finally:
        if draft is not None:
            draft.close()
        target.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
