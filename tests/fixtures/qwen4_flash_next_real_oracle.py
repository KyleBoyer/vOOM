#!/usr/bin/env python3
"""Real-checkpoint Qwen3.8-Flash-Next layer-stationary state oracle.

Run only after ``runtime.memory_preflight`` passes. The artifact contains
hashes/timing/pressure telemetry, never prompt text or tensor payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams


def _pressure() -> dict[str, int]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
    }


def _state_digest(kv) -> tuple[str, int, int, dict[str, str]]:
    digest = hashlib.sha256()
    components = {
        name: hashlib.sha256() for name in ("kv", "kda", "qsa", "ple")
    }
    arrays = 0
    payload = 0

    def add(label: str, value) -> None:
        nonlocal arrays, payload
        if value is None:
            return
        dtype = str(value.dtype)
        # NumPy's PEP-3118 bridge on this Python/MLX build cannot expose
        # bfloat16 directly. Viewing the same two bytes as uint16 is an exact
        # bit-preserving representation for the digest.
        host = np.asarray(
            value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)
        raw = host.tobytes(order="C")
        digest.update(label.encode())
        digest.update(str(host.shape).encode())
        digest.update(dtype.encode())
        digest.update(raw)
        component = label.split(".", 1)[0]
        if component in components:
            components[component].update(label.encode())
            components[component].update(str(host.shape).encode())
            components[component].update(dtype.encode())
            components[component].update(raw)
        arrays += 1
        payload += len(raw)

    digest.update(f"offset:{int(kv.offset)}".encode())
    for layer, (key, value) in enumerate(zip(kv.keys, kv.values)):
        add(f"kv.{layer}.key", key)
        add(f"kv.{layer}.value", value)
    recurrent = getattr(kv, "kda_cache", None)
    if recurrent is not None:
        for layer in range(len(kv.keys)):
            add(f"kda.{layer}.state", recurrent.state(layer))
            history = recurrent.conv_history(layer)
            for index, value in enumerate(history or ()):
                add(f"kda.{layer}.conv.{index}", value)
    qwen4 = getattr(kv, "qwen4_cache", None)
    if qwen4 is not None:
        for layer in range(len(kv.keys)):
            add(f"qsa.{layer}.key", qwen4.qsa_keys[layer])
            add(f"qsa.{layer}.pos", qwen4.qsa_positions[layer])
            add(f"ple.{layer}.conv", qwen4.ple_conv[layer])
            digest.update(
                f"ple.{layer}.context:{qwen4.ple_context[layer]}".encode())
            digest.update(
                f"ple.{layer}.length:{qwen4.ple_lengths[layer]}".encode())
            components["ple"].update(
                f"context:{layer}:{qwen4.ple_context[layer]}".encode())
            components["ple"].update(
                f"length:{layer}:{qwen4.ple_lengths[layer]}".encode())
    return (
        digest.hexdigest(), arrays, payload,
        {name: value.hexdigest() for name, value in components.items()},
    )


def _run(
    model: Path, prompt: str, chunk: int, layer_stationary: bool,
    compiled_delta: bool = False,
    ple_read_workers: int = 1,
    global_expert_rows: bool = False,
    sparse_expert_batch_rows: bool = False,
    fast_tier_dir: str = "",
    parallel_storage_reads: bool = False,
    native_fused_delta: bool = False,
) -> dict:
    before = _pressure()
    rc = RuntimeConfig(
        max_weight_cache_mb=300,
        min_weight_cache_mb=64,
        mlx_cache_limit_mb=128,
        prefill_chunk_size=chunk,
        expert_fetch_batch=16,
        decode_expert_fetch_batch=16,
        pin_lm_head=False,
        stream_lm_head=True,
        pin_embeddings=False,
        embed_rows=True,
        prompt_kv_dir="",
        hot_prompt_kv=False,
        layer_stationary_prefill=layer_stationary,
        qwen_compiled_delta_prefill=compiled_delta,
        qwen_native_fused_delta_prefill=native_fused_delta,
        qwen4_ple_read_workers=ple_read_workers,
        qwen4_global_expert_rows=global_expert_rows,
        qwen4_sparse_expert_batch_rows=sparse_expert_batch_rows,
        fast_dirs=((fast_tier_dir,) if fast_tier_dir else ()),
        parallel_storage_reads=parallel_storage_reads,
        governor=True,
    )
    started = time.perf_counter()
    engine = StreamingEngine(str(model), rc)
    try:
        initialized = time.perf_counter()
        result = engine.generate(
            prompt, max_tokens=1,
            sampling=SamplingParams(temperature=0.0))
        completed = time.perf_counter()
        (state_sha, state_arrays, state_bytes,
         state_components) = _state_digest(engine.last_kv)
        stats = {
            **asdict(engine.cache.stats),
            "resident_bytes": int(engine.cache.total_bytes),
            "max_bytes": int(engine.cache.max_bytes),
        }
        return {
            "layer_stationary": layer_stationary,
            "compiled_delta": compiled_delta,
            "native_fused_delta": native_fused_delta,
            "ple_read_workers": ple_read_workers,
            "global_expert_rows": global_expert_rows,
            "sparse_expert_batch_rows": sparse_expert_batch_rows,
            "fast_tier_dir": fast_tier_dir,
            "parallel_storage_reads": parallel_storage_reads,
            "startup_seconds": round(initialized - started, 6),
            "generation_seconds": round(completed - initialized, 6),
            "wall_seconds": round(completed - started, 6),
            "prompt_tokens": int(result.get("prompt_tokens", 0)),
            "tokens": list(result["tokens"]),
            "text_sha256": hashlib.sha256(
                result["text"].encode("utf-8")).hexdigest(),
            "state_sha256": state_sha,
            "state_arrays": state_arrays,
            "state_bytes": state_bytes,
            "state_component_sha256": state_components,
            "peak_metal_bytes": int(result["true_peak_metal_bytes"]),
            "released_weight_read_bytes": int(
                result["path_stats"].get("weight_store_bytes_read", 0)),
            "fast_tier_read_bytes": int(
                result["path_stats"].get("weight_fast_tier_bytes", 0)),
            "archive_read_bytes": int(
                result["path_stats"].get("weight_archive_bytes", 0)),
            "parallel_tier": {
                key: result["path_stats"].get(f"parallel_tier_{key}", 0)
                for key in (
                    "fetches", "fast_bytes", "archive_bytes", "wall_s",
                    "fast_service_s", "archive_service_s", "hidden_s")
            },
            "expert_read_bytes": int(
                result["path_stats"].get("qwen4_fused_expert_bytes", 0)),
            "ple_read_bytes": int(
                result["path_stats"].get("qwen4_ple_bytes_read", 0)),
            "host_spool_peak_bytes": int(
                result["path_stats"].get(
                    "qwen4_host_spool_peak_host_bytes", 0)),
            "host_spool_copy_seconds": float(
                result["path_stats"].get(
                    "qwen4_host_spool_copy_seconds", 0.0)),
            "host_spool_h2d_bytes": int(
                result["path_stats"].get(
                    "qwen4_host_spool_h2d_bytes", 0)),
            "host_spool_d2h_bytes": int(
                result["path_stats"].get(
                    "qwen4_host_spool_d2h_bytes", 0)),
            "host_spool_phase_seconds": {
                phase: float(result["path_stats"].get(
                    f"qwen4_host_spool_{phase}_seconds", 0.0))
                for phase in (
                    "ple", "attention", "route_and_spool",
                    "experts", "output")
            },
            "cache": stats,
            "pressure_before": before,
            "pressure_after": _pressure(),
        }
    finally:
        engine.close()
        mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--chunk", type=int, default=3)
    parser.add_argument("--candidate-compiled", action="store_true")
    parser.add_argument(
        "--candidate-native-fused-delta", action="store_true")
    parser.add_argument("--candidate-ple-read-workers", type=int, default=1)
    parser.add_argument("--candidate-global-expert-rows", action="store_true")
    parser.add_argument(
        "--candidate-sparse-expert-batch-rows", action="store_true")
    parser.add_argument("--compare-layer-stationary", action="store_true")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--candidate-fast-tier-dir", default="")
    parser.add_argument(
        "--candidate-parallel-storage-reads", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.chunk <= 0:
        parser.error("chunk must be positive")
    if not 1 <= args.candidate_ple_read_workers <= 16:
        parser.error("candidate-ple-read-workers must be in [1, 16]")
    if args.prompt_repeat <= 0:
        parser.error("prompt-repeat must be positive")
    prompt = "Say hello in one word. " * args.prompt_repeat
    baseline = _run(
        args.model, prompt, args.chunk, args.compare_layer_stationary,
        args.candidate_compiled if args.compare_layer_stationary else False,
        (args.candidate_ple_read_workers
         if args.compare_layer_stationary else 1),
        False, False, "", False, False)
    candidate = _run(
        args.model, prompt, args.chunk, True,
        (args.candidate_compiled
         and not args.candidate_native_fused_delta),
        args.candidate_ple_read_workers, args.candidate_global_expert_rows,
        args.candidate_sparse_expert_batch_rows,
        args.candidate_fast_tier_dir,
        args.candidate_parallel_storage_reads,
        args.candidate_native_fused_delta)
    matched = {
        "tokens": candidate["tokens"] == baseline["tokens"],
        "text": candidate["text_sha256"] == baseline["text_sha256"],
        "state": candidate["state_sha256"] == baseline["state_sha256"],
    }
    matched["state_components"] = {
        name: candidate["state_component_sha256"][name]
        == baseline["state_component_sha256"][name]
        for name in baseline["state_component_sha256"]
    }
    swap_growth = max(
        baseline["pressure_after"]["swap_out_bytes"]
        - baseline["pressure_before"]["swap_out_bytes"],
        candidate["pressure_after"]["swap_out_bytes"]
        - candidate["pressure_before"]["swap_out_bytes"],
    )
    passed = (
        matched["tokens"] and matched["text"] and matched["state"]
        and max(baseline["peak_metal_bytes"], candidate["peak_metal_bytes"])
        <= 8_500_000_000
        and swap_growth <= 16_000_000
    )
    document = {
        "schema": "voom.qwen4-flash-next-layer-stationary-oracle.v1",
        "model_revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
        "chunk": args.chunk,
        "prompt_repeat": args.prompt_repeat,
        "compare_layer_stationary": args.compare_layer_stationary,
        "baseline": baseline,
        "candidate": candidate,
        "matched": matched,
        "max_swap_out_growth_bytes": swap_growth,
        "verdict": "PASS" if passed else "STOP",
    }
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
