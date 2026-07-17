#!/usr/bin/env python3
"""Real-Qwen gate for restart/fork/regenerate/eviction prompt-KV reuse.

This complements ``tests/test_hot_prompt_kv.py``'s tiny deterministic fixture
with the on-disk quantized Qwen checkpoint used by the lossy side quest.  It
proves four properties in one run:

* an identical prompt resumes after a fresh engine instance (restart);
* regenerating the same prompt is an exact endpoint hit;
* an edited suffix reuses only a chunk-aligned common prefix and remains
  token-identical to a cold engine (fork);
* a task evicted from a one-slot in-memory LRU is recovered from the persisted
  segment DAG (the multi-agent/fleet case).

Example:
  ~/.hf-pull/bin/python tests/fixtures/qwen_hot_kv_live_gate.py \
    --model ~/models/Qwen2.5-1.5B-Instruct-mlx-mxfp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.server import PreparedPrompt  # noqa: E402


def _config(persist_dir: Path | None, chunk_size: int) -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=1200,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        prompt_kv_dir="",
        prefill_chunk_size=chunk_size,
        resident_fast_decode=True,
        resident_fast_prefill_limit=512,
        stepped_kv_threshold=512,
        fused_swiglu=True,
        hot_prompt_kv=persist_dir is not None,
        hot_prompt_kv_chunk_size=chunk_size,
        hot_prompt_kv_slots=1,
        hot_prompt_kv_min_tokens=0,
        hot_prompt_kv_persist_dir=str(persist_dir or ""),
        hot_prompt_kv_persist_max_checkpoints=16,
    )


def _repeated_prompt_ids(
        engine: StreamingEngine, paragraph: str, minimum: int) -> list[int]:
    text = paragraph
    while len(engine.tokenizer.encode(text).ids) < minimum:
        text += paragraph
    return list(engine.tokenizer.encode(text).ids)


def _shared_prompt_ids(engine: StreamingEngine, minimum: int) -> list[int]:
    return _repeated_prompt_ids(engine, (
        "You are one worker in a deterministic software-engineering agent fleet. "
        "Preserve tool-call order, report concrete evidence, and never invent test results. "
        "The shared tool catalog and system policy in this paragraph are intentionally stable.\n"
    ), minimum)


def _prepared(label: str, ids: list[int]) -> PreparedPrompt:
    # PreparedPrompt proves the exact token sequence under test and avoids a
    # second tokenizer pass changing the intended branch boundary.
    return PreparedPrompt(label, ids)


def _run(engine: StreamingEngine, label: str, ids: list[int], max_tokens: int):
    started = time.perf_counter()
    result = engine.generate(
        _prepared(label, ids), max_tokens=max_tokens, stop=[])
    result["gate_wall_s"] = time.perf_counter() - started
    return result


def _disk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--persist-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    if not (model / "config.json").exists():
        parser.error(f"model is not a complete local checkpoint: {model}")
    if args.prompt_tokens < args.chunk_size * 2:
        parser.error("--prompt-tokens must cover at least two cache chunks")
    if args.chunk_size <= 0 or args.max_tokens <= 1:
        parser.error("--chunk-size must be positive and --max-tokens must exceed one")

    persist_dir = (args.persist_dir.expanduser().resolve()
                   if args.persist_dir else
                   model.parent / ".voom-qwen-hot-kv-live-gate")
    shutil.rmtree(persist_dir, ignore_errors=True)
    persist_dir.mkdir(parents=True)

    cold = StreamingEngine(model, _config(None, args.chunk_size))
    try:
        shared = _shared_prompt_ids(cold, args.prompt_tokens)
        suffix_a = list(cold.tokenizer.encode(
            "Agent A: inspect the cache implementation and give one finding.").ids)
        suffix_b = list(cold.tokenizer.encode(
            "Agent B: inspect the cache implementation and give one finding.").ids)
        unrelated_prefix = _repeated_prompt_ids(cold, (
            "This independent networking task has a disjoint policy and vocabulary. "
            "Analyze packet loss, retransmission timers, MTU discovery, and DNS latency. "
            "Do not consult any software-engineering agent instructions.\n"
        ), args.prompt_tokens + args.chunk_size)
        unrelated_suffix = list(cold.tokenizer.encode(
            "Independent task: summarize an unrelated networking incident.").ids)

        prompt_a = shared + suffix_a
        prompt_b = shared + suffix_b
        unrelated = unrelated_prefix + unrelated_suffix
        cold_a = _run(cold, "cold-a", prompt_a, args.max_tokens)
        cold_b = _run(cold, "cold-b", prompt_b, args.max_tokens)
    finally:
        cold.close()
        mx.clear_cache()

    first = StreamingEngine(model, _config(persist_dir, args.chunk_size))
    try:
        seeded = _run(first, "seed-a", prompt_a, args.max_tokens)
        seed_leaf = first._hot_prompt_slots[-1].segment_chain[-1]
    finally:
        first.close()
        mx.clear_cache()

    # A genuinely fresh engine instance simulates the serving process's model
    # reconstruction after restart. Startup restores the newest slot from disk.
    restarted_engine = StreamingEngine(model, _config(persist_dir, args.chunk_size))
    try:
        restarted = _run(
            restarted_engine, "restart-a", prompt_a, args.max_tokens)
        restart_stats = restarted["path_stats"]

        regenerated = _run(
            restarted_engine, "regenerate-a", prompt_a, args.max_tokens - 1)
        regenerate_stats = regenerated["path_stats"]
        regenerate_leaf = restarted_engine._hot_prompt_slots[-1].segment_chain[-1]

        forked = _run(
            restarted_engine, "fork-b", prompt_b, args.max_tokens)
        fork_stats = forked["path_stats"]

        # One slot means this unrelated, equally expensive task evicts B. A's
        # older checkpoint must remain discoverable through disk metadata.
        _run(restarted_engine, "unrelated", unrelated, 3)
        evicted = _run(
            restarted_engine, "evicted-a", prompt_a, args.max_tokens)
        evicted_stats = evicted["path_stats"]
    finally:
        restarted_engine.close()
        mx.clear_cache()

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(seeded["tokens"] == cold_a["tokens"],
            "seeded prompt differs from cold target tokens")
    require(restarted["tokens"] == cold_a["tokens"],
            "restart hit differs from cold target tokens")
    require(restarted["tokens"][: args.max_tokens - 1] == regenerated["tokens"],
            "regenerate output is not the deterministic prefix of the cold output")
    require(forked["tokens"] == cold_b["tokens"],
            "forked suffix differs from cold target tokens")
    require(evicted["tokens"] == cold_a["tokens"],
            "disk-recovered task differs from cold target tokens")

    require(restart_stats.get("prompt_cache_source") == "memory",
            "fresh engine did not preload the persisted checkpoint")
    require(restart_stats.get("prompt_cache_exact_hit") == 1,
            "restart was not an exact prompt endpoint hit")
    require(regenerate_stats.get("prompt_cache_source") == "memory",
            "regenerate did not reuse the in-memory prompt endpoint")
    require(regenerate_stats.get("prompt_cache_exact_hit") == 1,
            "regenerate was not an exact prompt endpoint hit")
    require(fork_stats.get("prompt_cache_source") in ("memory", "hot_disk"),
            "edited suffix did not reuse a persisted/in-memory prefix")
    require(0 < int(fork_stats.get("prompt_cache_prefix_tokens", 0)) < len(prompt_b),
            "fork did not reuse a strict proper prefix")
    require(int(fork_stats.get("prompt_cache_prefix_tokens", 0)) % args.chunk_size == 0,
            "fork reuse was not chunk aligned")
    require(evicted_stats.get("prompt_cache_source") == "hot_disk",
            "evicted task was not recovered from the on-disk segment DAG")
    require(evicted_stats.get("hot_prompt_kv_disk_hit") == 1,
            "evicted task did not report a hot-disk hit")
    require(evicted_stats.get("prompt_cache_exact_hit") == 1,
            "evicted task recovery was not an exact prompt endpoint hit")
    require(seed_leaf != regenerate_leaf,
            "different-length regenerations unexpectedly collapsed to one leaf")

    report = {
        "gate": "qwen-hot-kv-live-v1",
        "passed": not failures,
        "failures": failures,
        "model": str(model),
        "prompt_tokens": {"a": len(prompt_a), "b": len(prompt_b)},
        "chunk_size": args.chunk_size,
        "tokens_compared": (
            len(cold_a["tokens"]) + len(restarted["tokens"])
            + len(forked["tokens"]) + len(evicted["tokens"])),
        "restart": {
            "source": restart_stats.get("prompt_cache_source"),
            "exact": restart_stats.get("prompt_cache_exact_hit"),
            "cold_s": round(cold_a["total_s"], 6),
            "hit_s": round(restarted["total_s"], 6),
            "speedup": round(cold_a["total_s"] / restarted["total_s"], 3)
            if restarted["total_s"] else None,
        },
        "regenerate": {
            "source": regenerate_stats.get("prompt_cache_source"),
            "exact": regenerate_stats.get("prompt_cache_exact_hit"),
            "prefix_tokens": regenerate_stats.get("prompt_cache_prefix_tokens"),
        },
        "fork": {
            "source": fork_stats.get("prompt_cache_source"),
            "prefix_tokens": fork_stats.get("prompt_cache_prefix_tokens"),
            "cold_s": round(cold_b["total_s"], 6),
            "reused_s": round(forked["total_s"], 6),
            "speedup": round(cold_b["total_s"] / forked["total_s"], 3)
            if forked["total_s"] else None,
        },
        "eviction": {
            "source": evicted_stats.get("prompt_cache_source"),
            "exact": evicted_stats.get("prompt_cache_exact_hit"),
            "cold_s": round(cold_a["total_s"], 6),
            "recovered_s": round(evicted["total_s"], 6),
            "speedup": round(cold_a["total_s"] / evicted["total_s"], 3)
            if evicted["total_s"] else None,
        },
        "persisted": {
            "checkpoints": len(list(persist_dir.glob("*.ckpt.json"))),
            "segments": len(list(persist_dir.glob("*.seg.json"))),
            "bytes": _disk_bytes(persist_dir),
        },
        "peak_metal_gb": round(max(
            row["true_peak_metal_bytes"] for row in
            (cold_a, cold_b, seeded, restarted, regenerated, forked, evicted)
        ) / 1e9, 3),
    }
    print(json.dumps(report, indent=2))

    if not args.keep:
        shutil.rmtree(persist_dir, ignore_errors=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
