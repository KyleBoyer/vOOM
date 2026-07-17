#!/usr/bin/env python3
"""Real-checkpoint gate for governor-sized exact OLMoE weight caching.

The candidate deliberately keeps the released per-expert execution path. Only
the cache allowance changes, so the gate requires identical greedy IDs and zero
candidate evictions.
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

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.server import _checkpoint_payload_bytes  # noqa: E402


PROMPT = "Explain deterministic validation with two concrete examples."


def _run(model: Path, cache_mb: int, max_tokens: int):
    engine = StreamingEngine(model, RuntimeConfig(
        max_weight_cache_mb=cache_mb,
        prefetch_depth=2,
        pin_lm_head=True,
        embed_rows=True,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
    ))
    try:
        fitted = (engine.governor.fit_cache_to_live_headroom()
                  if engine.governor is not None else engine.cache.max_bytes)
        result = engine.generate(PROMPT, max_tokens=max_tokens, stop=[])
        stats = {
            "fitted_cache_bytes": fitted,
            "cache_resident_bytes": engine.cache.total_bytes,
            "cache_evictions": engine.cache.stats.evictions,
            "bytes_read": engine.cache.stats.bytes_read,
        }
        return result, stats
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    model = args.model.expanduser()
    payload = _checkpoint_payload_bytes(model)
    if payload <= 0:
        parser.error("could not determine checkpoint payload size")

    control, control_stats = _run(model, 6000, args.max_tokens)
    mx.clear_cache()
    candidate_mb = math.ceil(payload * 1.07 / 1_000_000)
    candidate, candidate_stats = _run(model, candidate_mb, args.max_tokens)
    mx.clear_cache()

    exact = control["tokens"] == candidate["tokens"]
    admitted = candidate_stats["fitted_cache_bytes"] >= payload
    eviction_free = candidate_stats["cache_evictions"] == 0
    report = {
        "exact": exact,
        "admitted": admitted,
        "candidate_eviction_free": eviction_free,
        "tokens_compared": len(control["tokens"]),
        "streamed_total_s": round(control["total_s"], 4),
        "candidate_total_s": round(candidate["total_s"], 4),
        "speedup": round(
            control["total_s"] / candidate["total_s"], 3),
        "streamed_peak_metal_bytes": control["true_peak_metal_bytes"],
        "candidate_peak_metal_bytes": candidate["true_peak_metal_bytes"],
        "payload_bytes": payload,
        "streamed_cache": control_stats,
        "candidate_cache": candidate_stats,
    }
    print(json.dumps(report, indent=2))
    return 0 if exact and admitted and eviction_free else 1


if __name__ == "__main__":
    raise SystemExit(main())
