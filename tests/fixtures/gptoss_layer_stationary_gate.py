#!/usr/bin/env python3
"""Released GPT-OSS A/B for bounded layer-stationary prefill.

Run the candidate first to give it the conservative cold-storage arm; the
chunk-major control follows with any filesystem-cache advantage.  Both arms
use the same fixed tile, bounded expert batches, cache budget, released
checkpoint, prompt IDs, sampler, and output length.  Only loop nesting differs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams
from runtime.server import PreparedPrompt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _pressure() -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
        "metal_active_bytes": int(mx.get_active_memory()),
        "metal_peak_bytes": int(mx.get_peak_memory()),
    }


def _config(candidate: bool, tile: int, cache_mb: int,
            expert_batch: int) -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=cache_mb,
        min_weight_cache_mb=1500,
        mlx_cache_limit_mb=256,
        prefetch_depth=2,
        prefetch_workers=2,
        pin_lm_head=True,
        pin_first_layers=36,
        embed_rows=True,
        prompt_kv_dir="",
        prefill_chunk_size=tile,
        adaptive_chunk_size=False,
        layer_stationary_prefill=candidate,
        expert_fetch_batch=expert_batch,
        decode_expert_fetch_batch=4,
        final_dead_token_elim=True,
    )


def _run(model: Path, token_ids: list[int], *, candidate: bool,
         tile: int, cache_mb: int, expert_batch: int,
         max_tokens: int) -> dict:
    config = _config(candidate, tile, cache_mb, expert_batch)
    before = _pressure()
    started = time.perf_counter()
    engine = StreamingEngine(str(model), config)
    initialized = time.perf_counter()
    try:
        prompt = PreparedPrompt("", token_ids)
        result = engine.generate(
            prompt, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
        path = result.get("path_stats") or {}
        selected_path = {
            key: path.get(key) for key in (
                "prompt_cache_source", "prompt_cache_tokens",
                "prefill_chunks", "layer_stationary_gptoss",
                "layer_stationary_endpoint_fused",
                "weight_store_bytes_read", "weight_archive_bytes",
                "weight_fast_tier_bytes", "expert_cache_hits",
                "expert_cache_misses", "expert_compute_batches",
                "max_experts_per_compute_batch", "rope_profile",
                "weight_integrity_mode")
        }
        return {
            "candidate": candidate,
            "runtime_config": dataclasses.asdict(config),
            "prompt_tokens": len(token_ids),
            "prompt_sha256": hashlib.sha256(
                bytes().join(int(token).to_bytes(4, "little", signed=False)
                             for token in token_ids)).hexdigest(),
            "output_ids": list(result.get("tokens") or ()),
            "output_text_sha256": hashlib.sha256(
                str(result.get("text", "")).encode()).hexdigest(),
            "termination_reason": result.get("termination_reason"),
            "true_peak_metal_bytes": int(result["true_peak_metal_bytes"]),
            "prefill_seconds": float(result.get("prefill_s", 0.0)),
            "decode_seconds": float(result.get("decode_s", 0.0)),
            "engine_seconds": float(result.get("total_s", 0.0)),
            "initialization_seconds": initialized - started,
            "wall_seconds": time.perf_counter() - started,
            "path_stats": selected_path,
            "pressure_before": before,
            "pressure_after": _pressure(),
        }
    finally:
        engine.close()
        mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/gpt-oss-120b"))
    parser.add_argument("--prompt-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--cache-mb", type=int, default=4500)
    parser.add_argument("--expert-batch", type=int, default=8)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if (args.prompt_tokens <= args.tile or args.max_tokens <= 0
            or args.tile <= 0 or args.cache_mb <= 0
            or args.expert_batch <= 0):
        parser.error("invalid positive bounds or prompt does not span tiles")

    # Initialize only the tokenizer to build one deterministic ID sequence;
    # the candidate engine is still the first process to sweep model weights.
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    seed_text = (
        "A released-model scheduling oracle compares identical causal "
        "tokens across storage plans. " * (args.prompt_tokens // 8 + 16))
    token_ids = list(tokenizer.encode(seed_text).ids)
    if len(token_ids) < args.prompt_tokens:
        raise RuntimeError("failed to construct enough prompt tokens")
    token_ids = token_ids[:args.prompt_tokens]

    candidate = _run(
        args.model, token_ids, candidate=True, tile=args.tile,
        cache_mb=args.cache_mb, expert_batch=args.expert_batch,
        max_tokens=args.max_tokens)
    control = _run(
        args.model, token_ids, candidate=False, tile=args.tile,
        cache_mb=args.cache_mb, expert_batch=args.expert_batch,
        max_tokens=args.max_tokens)
    token_match = candidate["output_ids"] == control["output_ids"]
    termination_match = (
        candidate["termination_reason"] == control["termination_reason"])
    peak = max(int(arm["true_peak_metal_bytes"])
               for arm in (candidate, control))
    result = {
        "schema": "voom.gptoss-layer-stationary-gate.v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip()),
        "model": str(args.model),
        "model_fingerprint": {
            name: _sha256(args.model / name) for name in (
                "config.json", "weights.vpack2.index.json", "tokenizer.json")
        },
        "order": ["candidate-cold-first", "control-second"],
        "candidate": candidate,
        "control": control,
        "token_match": token_match,
        "termination_match": termination_match,
        "peak_metal_bytes": peak,
        "peak_limit_bytes": 8_500_000_000,
        "passed": bool(
            token_match and termination_match
            and candidate["path_stats"].get("layer_stationary_endpoint_fused") == 1
            and peak <= 8_500_000_000),
    }
    _atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
