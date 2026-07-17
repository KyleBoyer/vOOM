#!/usr/bin/env python3
"""Real Qwen3/DSpark exact-token, memory, and throughput gate.

Example:
  ~/.hf-pull/bin/python tests/fixtures/qwen3_dspark_gate.py \
    --target ~/models/Qwen3-4B \
    --draft ~/models/dspark_qwen3_4b_block7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.dspark import DSparkSpeculativeEngine  # noqa: E402
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402


PROMPTS = [
    "Give four concise reasons deterministic decoding helps reproducible tests.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Explain the latency versus throughput tradeoff in local model inference.",
]


def _prompts(target: Path, quick: bool) -> list[str]:
    if quick:
        return PROMPTS[:1]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(target), local_files_only=True)
    chat = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": "Solve 17 * 23, then explain in one sentence."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return PROMPTS + [chat]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--target-cache-mb", type=int, default=6000)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    target_path = args.target.expanduser().resolve()
    draft_path = args.draft.expanduser().resolve()
    prompts = _prompts(target_path, args.quick)

    target = StreamingEngine(target_path, RuntimeConfig(
        # Force the streamed-target regime in which DSpark is useful. The
        # server separately prefers ordinary full-target residency whenever
        # the live governor admits the complete ~8.45GB target footprint.
        max_weight_cache_mb=args.target_cache_mb,
        prefetch_workers=2,
        prefetch_depth=4,
        pin_lm_head=True,
        embed_rows=True,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
    ))
    engine = None
    try:
        plain = [target.generate(prompt, max_tokens=args.max_tokens, stop=[])
                 for prompt in prompts]
        engine = DSparkSpeculativeEngine(
            target, draft_path,
            max_draft_tokens=args.cap,
            max_prompt_tokens=2048,
            confidence_threshold=args.confidence_threshold,
        )
        speculative = [
            engine.generate(prompt, max_tokens=args.max_tokens, stop=[])
            for prompt in prompts
        ]
        mismatches = [
            index for index, (baseline, changed) in
            enumerate(zip(plain, speculative))
            if baseline["tokens"] != changed["tokens"]
        ]
        plain_s = sum(row["total_s"] for row in plain)
        dspark_s = sum(row["total_s"] for row in speculative)
        stats = [row["stats"] for row in speculative]
        report = {
            "exact": not mismatches,
            "mismatched_prompt_indices": mismatches,
            "prompts": len(prompts),
            "tokens_compared": sum(len(row["tokens"]) for row in plain),
            "cap": args.cap,
            "target_cache_mb": args.target_cache_mb,
            "confidence_threshold": args.confidence_threshold,
            "target_total_s": round(plain_s, 4),
            "dspark_total_s": round(dspark_s, 4),
            "speedup": round(plain_s / dspark_s, 3) if dspark_s else None,
            "accepted": sum(row.accepted for row in stats),
            "proposed": sum(row.proposed for row in stats),
            "target_decode_sweeps": sum(
                max(0, row.target_sweeps - 1) for row in stats),
            "peak_metal_gb": round(max(
                row["true_peak_metal_bytes"] for row in speculative) / 1e9, 3),
        }
        print(json.dumps(report, indent=2))
        return 0 if not mismatches and report["speedup"] >= 1.01 else 1
    finally:
        if engine is not None:
            engine.close()
        else:
            target.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
