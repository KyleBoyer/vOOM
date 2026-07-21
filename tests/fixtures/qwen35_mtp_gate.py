#!/usr/bin/env python3
"""F94: real-checkpoint byte-identical + throughput gate for Qwen native MTP
speculative decoding (QwenMTPSpeculativeEngine).

Byte-identical here means the emitted TOKEN ID sequence, not raw logit
values -- batching two positions into one forward call (the MTP accept
path) versus the plain engine's always-one-at-a-time path can differ by
~1e-7 in raw float32 logits from ordinary reduction-order non-associativity
(directly measured in tests/test_qwen35_mtp_rollback.py); this is exactly
why speculative.py's own verification is an argmax/token-id comparison, not
a logit-value comparison, and why this gate checks the same thing.

Example:
  .venv/bin/python tests/fixtures/qwen35_mtp_gate.py \
    --target ~/models/Qwen3.6-35B-A3B-mlx-expert-mxfp4 --max-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.qwen35_mtp import QwenMTPSpeculativeEngine  # noqa: E402


PROMPTS = [
    "Give four concise reasons deterministic decoding helps reproducible tests.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Explain the latency versus throughput tradeoff in local model inference.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    prompts = PROMPTS[:1] if args.quick else PROMPTS

    plain = StreamingEngine(args.target.expanduser(), RuntimeConfig(
        max_weight_cache_mb=7000,
        prefetch_depth=2,
        pin_lm_head=False,
        stream_lm_head=True,
        embed_rows=True,
        prompt_kv_dir="",
    ))
    ok = True
    try:
        baselines = []
        for prompt in prompts:
            t0 = time.perf_counter()
            result = plain.generate(prompt, max_tokens=args.max_tokens, stop=[])
            baselines.append(result)
            print(f"[plain] {result['tok_per_s']:.2f} tok/s "
                  f"({time.perf_counter() - t0:.1f}s wall) tokens={result['tokens']}",
                  flush=True)

        mtp = QwenMTPSpeculativeEngine(plain)
        for prompt, baseline in zip(prompts, baselines):
            t0 = time.perf_counter()
            result = mtp.generate(prompt, max_tokens=args.max_tokens, stop=[])
            wall = time.perf_counter() - t0
            stats = result["path_stats"]
            used = stats.get("qwen_mtp_used", 0)
            if result["tokens"] != baseline["tokens"]:
                ok = False
                print(f"[MISMATCH] prompt={prompt!r}\n"
                      f"  plain: {baseline['tokens']}\n"
                      f"  mtp:   {result['tokens']}", flush=True)
                continue
            acc = ""
            if used:
                proposed = stats.get("qwen_mtp_proposed", 0)
                accepted = stats.get("qwen_mtp_accepted", 0)
                acc = (f" accept={accepted}/{proposed}"
                       f"({100 * accepted / proposed:.0f}%)" if proposed else "")
            print(f"[mtp]   {result['tok_per_s']:.2f} tok/s ({wall:.1f}s wall) "
                  f"used={used}{acc} tokens match plain: OK", flush=True)
    finally:
        plain.close()

    if ok:
        print("\nALL PROMPTS: MTP speculative output byte-identical (token ids) to plain decode.")
        return 0
    print("\nFAILURE: at least one prompt's MTP output diverged from plain decode.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
