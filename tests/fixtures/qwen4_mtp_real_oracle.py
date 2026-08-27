#!/usr/bin/env python3
"""Real Qwen4 plain-vs-Lightning-MTP greedy/state oracle.

This is a correctness gate, not a clean timing A/B: plain runs first and MTP
second in one process, so filesystem cache state is not balanced.  It persists
only hashes, token IDs, timing/read counters, and pressure telemetry.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen4_flash_next_real_oracle import _state_digest
from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen4_mtp import Qwen4MTPSpeculativeEngine
from runtime.sampler import SamplingParams


def _pressure() -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
    }


def _config(
    cache_mb: int, *, pin_head: bool = False, phase_head: bool = False,
) -> RuntimeConfig:
    if pin_head and phase_head:
        raise ValueError("pin_head and phase_head are mutually exclusive")
    return RuntimeConfig(
        max_weight_cache_mb=cache_mb,
        min_weight_cache_mb=64,
        mlx_cache_limit_mb=128,
        metal_limit_mb=8500,
        prefill_chunk_size=128,
        expert_fetch_batch=16,
        decode_expert_fetch_batch=16,
        prefetch_depth=0,
        qwen4_ple_read_workers=8,
        pin_embeddings=False,
        pin_lm_head=pin_head or phase_head,
        stream_lm_head=not (pin_head or phase_head),
        qwen4_phase_lm_head=phase_head,
        embed_rows=True,
        layer_stationary_prefill=True,
        qwen_compiled_delta_prefill=True,
        expert_route_overlap_telemetry=True,
        governor=True,
    )


def _run(
    model: Path, prompt: str, max_tokens: int, depth: int | None,
    cache_mb: int, *, pin_head: bool = False, phase_head: bool = False,
) -> dict:
    before = _pressure()
    started = time.perf_counter()
    target = StreamingEngine(str(model), _config(
        cache_mb, pin_head=pin_head, phase_head=phase_head))
    initialized = time.perf_counter()
    engine = (
        target if depth is None
        else Qwen4MTPSpeculativeEngine(target, depth=depth)
    )
    try:
        result = engine.generate(
            prompt,
            max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0),
        )
        completed = time.perf_counter()
        kv = target.last_kv
        state_sha, arrays, payload, components = _state_digest(kv)
        decode_routes = [
            row for row in target.expert_route_overlap_trace
            if int(row.get("positions", 0)) == 1
        ]
        decode_current = sum(
            int(row.get("cross_call_current_experts", 0))
            for row in decode_routes)
        decode_reused = sum(
            int(row.get("cross_call_intersection_experts", 0))
            for row in decode_routes)
        return {
            "mode": "plain" if depth is None else f"mtp-depth{depth}",
            "pin_lm_head": int(pin_head),
            "phase_lm_head": int(phase_head),
            "tokens": list(result["tokens"]),
            "text_sha256": hashlib.sha256(
                result["text"].encode()).hexdigest(),
            "state_sha256": state_sha,
            "state_component_sha256": components,
            "state_arrays": arrays,
            "state_payload_bytes": payload,
            "startup_seconds": round(initialized - started, 6),
            "generation_seconds": round(completed - initialized, 6),
            "wall_seconds": round(completed - started, 6),
            "prefill_seconds": float(result.get("prefill_s", 0.0)),
            "decode_seconds": float(result.get("decode_s", 0.0)),
            "prompt_tokens": int(result.get("prompt_tokens", 0)),
            "kv_positions": int(result.get("kv_positions", kv.offset)),
            "peak_metal_bytes": int(result["true_peak_metal_bytes"]),
            "path_stats": {
                key: value
                for key, value in result.get("path_stats", {}).items()
                if key.startswith("qwen4_mtp_")
                or key.startswith("qwen4_serial_verify_")
                or key.startswith("qwen4_phase_lm_head")
                or key in {
                    "weight_store_bytes_read", "weight_archive_bytes",
                    "weight_fast_tier_bytes", "cache_source",
                }
            },
            "decode_route_overlap": {
                "calls": len(decode_routes),
                "current_experts": decode_current,
                "reused_experts": decode_reused,
                "reuse_fraction": (
                    decode_reused / decode_current
                    if decode_current else 0.0),
            },
            "pressure_before": before,
            "pressure_after": _pressure(),
        }
    finally:
        engine.close()
        del engine, target
        gc.collect()
        mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=96)
    parser.add_argument("--mtp-cache-mb", type=int)
    parser.add_argument("--mtp-pin-head", action="store_true")
    parser.add_argument("--mtp-phase-head", action="store_true")
    parser.add_argument("--plain-only", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("max-tokens must be at least two")
    if args.cache_mb < 64:
        parser.error("cache-mb must be at least 64")
    mtp_cache_mb = int(args.mtp_cache_mb or args.cache_mb)
    if mtp_cache_mb < 64:
        parser.error("mtp-cache-mb must be at least 64")
    if args.mtp_pin_head and mtp_cache_mb < 1400:
        parser.error("mtp-pin-head requires mtp-cache-mb at least 1400")
    if args.mtp_pin_head and args.mtp_phase_head:
        parser.error("mtp-pin-head and mtp-phase-head are mutually exclusive")

    initial = _pressure()
    plain = _run(
        args.model, args.prompt, args.max_tokens, None, args.cache_mb)
    if args.plain_only:
        pressure_after = _pressure()
        swap_out_growth = max(
            0,
            pressure_after["swap_out_bytes"] - initial["swap_out_bytes"],
        )
        report = {
            "schema": "voom.qwen4-decode-overlap-probe.v1",
            "model": args.model.name,
            "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "max_tokens": args.max_tokens,
            "weight_cache_mb": args.cache_mb,
            "plain": plain,
            "swap_out_growth_bytes": swap_out_growth,
            "passed": swap_out_growth <= 16_000_000,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.result.write_text(encoded)
        print(encoded, end="")
        return 0 if report["passed"] else 1
    mtp = _run(
        args.model, args.prompt, args.max_tokens, args.depth, mtp_cache_mb,
        pin_head=args.mtp_pin_head, phase_head=args.mtp_phase_head)
    tokens_equal = plain["tokens"] == mtp["tokens"]
    text_equal = plain["text_sha256"] == mtp["text_sha256"]
    state_equal = plain["state_sha256"] == mtp["state_sha256"]
    components_equal = (
        plain["state_component_sha256"] == mtp["state_component_sha256"])
    pressure_after = _pressure()
    report = {
        "schema": "voom.qwen4-mtp-real-oracle.v1",
        "model": args.model.name,
        "order": ["plain", f"mtp-depth{args.depth}"],
        "order_is_timing_balanced": False,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "weight_cache_mb": args.cache_mb,
        "mtp_weight_cache_mb": mtp_cache_mb,
        "plain": plain,
        "mtp": mtp,
        "tokens_equal": tokens_equal,
        "text_equal": text_equal,
        "state_equal": state_equal,
        "state_components_equal": components_equal,
        "initial_pressure": initial,
        "pressure_after": pressure_after,
        "swap_growth_bytes": max(
            0, pressure_after["swap_used_bytes"] - initial["swap_used_bytes"]),
        "swap_out_growth_bytes": max(
            0, pressure_after["swap_out_bytes"] - initial["swap_out_bytes"]),
        "passed": bool(
            tokens_equal and text_equal and state_equal and components_equal),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.result.write_text(encoded)
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
