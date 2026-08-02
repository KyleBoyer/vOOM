#!/usr/bin/env python3
"""Time full-schema Kimi K3 prefill and one decode sweep on a pinned capture.

The request is rendered with the ordinary lossless server prompt path: all
messages and tools remain in capture order, with only protocol normalization
needed to convert a Responses request into the checkpoint prompt. The model is
overridden to local Kimi K3, and ``max_output_tokens`` is set explicitly so a
timing probe cannot free-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CAPTURE_SHA256 = (
    "8ac18b8e8bc190180b4cc0e02c2453d313ec850642cc5d5f63b32e5537b90e85"
)
CAPTURE_BYTES = 178_616
CAPTURE_TOOLS = 134


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=True
    ).strip()


def _phase_stats(path_stats: dict, phase: str) -> dict:
    prefix = f"{phase}_"
    keys = (
        "weight_store_bytes_read",
        "weight_store_disk_ns",
        "expert_cache_hits",
        "expert_cache_misses",
        "ct_mxfp4_input_bytes",
        "ct_mxfp4_resident_bytes",
        "ct_mxfp4_transform_calls",
        "ct_mxfp4_transform_ns",
        "k3_scale_sidecar_read_bytes",
        "k3_scale_sidecar_output_bytes",
        "k3_scale_sidecar_decode_calls",
        "k3_scale_sidecar_decode_ns",
    )
    return {
        key: path_stats.get(prefix + key, 0)
        for key in keys
    }


def _render_capture(engine, model_dir: Path, request: dict, max_tokens: int):
    from runtime.server import _prepare_chat_prompt
    from runtime.toolcalls import (
        merge_leading_system_messages,
        normalize_messages,
        responses_input_to_messages,
    )

    messages = responses_input_to_messages(
        request.get("input", ""), request.get("instructions")
    )
    messages, image_sources = normalize_messages(messages)
    if image_sources:
        raise ValueError("timing fixture unexpectedly contains images")
    messages = merge_leading_system_messages(messages)

    raw_tools = request.get("tools") or []
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {},
            },
        }
        if tool.get("type") == "function"
        else tool
        for tool in raw_tools
    ]
    # The capture has no explicit reasoning field, so the Responses default is
    # low with reasoning_requested=False. K3 has no native chat template and
    # therefore receives no additional reasoning directive.
    return _prepare_chat_prompt(
        engine,
        model_dir,
        messages,
        "low",
        tools,
        raw_tools,
        "lossless",
        max_tokens,
        enable_thinking=None,
        reasoning_requested=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models" / "Kimi-K3"
    )
    parser.add_argument(
        "--scale-sidecar-dir",
        type=Path,
        default=ROOT / "logs" / "f139_k3_scale_sidecar_full",
    )
    parser.add_argument(
        "--expert-top-k",
        type=int,
        default=0,
        help="0 preserves released top-16; a positive value is explicitly lossy.",
    )
    parser.add_argument("--expert-fetch-batch", type=int, default=32)
    parser.add_argument("--cache-mb", type=int, default=3000)
    parser.add_argument(
        "--mlx-cache-mb",
        type=int,
        default=1024,
        help=(
            "MLX command-buffer cache cap; lower values trade cache reuse "
            "for bounded long-context Metal residency."
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=("layers", "ops"),
        default="layers",
        help=(
            "Telemetry detail. 'ops' synchronizes attention/router/MLP "
            "substeps and is diagnostic rather than a clean speed result."
        ),
    )
    parser.add_argument(
        "--prefill-tile-width",
        type=int,
        default=1,
        help=(
            "Layer-stationary attention tile width. The previous captured-"
            "request baseline used 1; larger values amortize attention "
            "dispatch while preserving the prompt and model computation."
        ),
    )
    parser.add_argument(
        "--prefill-tile-policy",
        choices=("fixed", "prompt-length"),
        default="fixed",
        help=(
            "Use the production K3 prompt-length schedule/retry contract when "
            "requested. The initial long-context width remains "
            "--prefill-tile-width."
        ),
    )
    parser.add_argument(
        "--compressed-mla",
        action="store_true",
        help="Retain exact K3 MLA latents in the stepped cache.",
    )
    parser.add_argument(
        "--absorbed-mla",
        action="store_true",
        help=(
            "Use weight-absorbed latent-space MLA for prefill/decode; "
            "requires --compressed-mla."
        ),
    )
    parser.add_argument(
        "--mla-key-tile-size",
        type=int,
        default=2048,
        help="Online-softmax key width for absorbed MLA; 0 is untiled.",
    )
    parser.add_argument(
        "--fused-attnres-tile-size",
        type=int,
        default=0,
        help="Position width for the fused Metal AttnRes path; 0 disables.",
    )
    parser.add_argument(
        "--attnres-spill-dir",
        type=Path,
        default=None,
        help=(
            "External-volume scratch root for exact BF16 AttnRes snapshots; "
            "requires --fused-attnres-tile-size."
        ),
    )
    parser.add_argument(
        "--kda-spill-dir",
        type=Path,
        default=None,
        help=(
            "External-volume scratch root for exact FP32/BF16 KDA state; "
            "completed prefill layers reload lazily for decode."
        ),
    )
    parser.add_argument(
        "--mla-kv-spill-dir",
        type=Path,
        default=None,
        help=(
            "External-volume scratch root for exact compressed MLA latents; "
            "completed prefill layers reload lazily for decode."
        ),
    )
    parser.add_argument(
        "--dense-mlp-tile-size",
        type=int,
        default=0,
        help="Position width for K3's dense MLP; 0 disables.",
    )
    parser.add_argument(
        "--native-fused-kda-prefill",
        action="store_true",
        help=(
            "Use the opt-in algebraically exact Metal serial KDA scan; its "
            "FP32 reduction schedule can differ from the MLX reference."
        ),
    )
    parser.add_argument(
        "--compiled-kda-prefill",
        action="store_true",
        help="Compile byte-identical 32-position MLX KDA scan segments.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional sampling seed for paired A/B only; the captured "
            "request itself has no seed."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="0 uses all released layers; otherwise run this prefix depth.",
    )
    parser.add_argument(
        "--prompt-token-limit",
        type=int,
        default=0,
        help=(
            "0 uses the complete rendered prompt; otherwise benchmark this "
            "many leading tokens as a scaling calibration."
        ),
    )
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.native_fused_kda_prefill and args.compiled_kda_prefill:
        parser.error(
            "--native-fused-kda-prefill and --compiled-kda-prefill are "
            "mutually exclusive")
    if args.attnres_spill_dir and not args.fused_attnres_tile_size:
        parser.error(
            "--attnres-spill-dir requires --fused-attnres-tile-size")
    if args.result.exists():
        raise SystemExit(f"refusing existing result: {args.result}")
    if args.max_output_tokens < 2 and not args.render_only:
        parser.error("--max-output-tokens must be at least 2 to time decode")
    if not 1 <= args.expert_fetch_batch <= 64:
        parser.error("--expert-fetch-batch must be in [1, 64]")
    if args.cache_mb < 150:
        parser.error("--cache-mb must be at least 150")
    if args.mlx_cache_mb < 64:
        parser.error("--mlx-cache-mb must be at least 64")
    if args.prefill_tile_width < 1:
        parser.error("--prefill-tile-width must be positive")
    if args.absorbed_mla and not args.compressed_mla:
        parser.error("--absorbed-mla requires --compressed-mla")
    if args.mla_kv_spill_dir and not args.compressed_mla:
        parser.error("--mla-kv-spill-dir requires --compressed-mla")
    if args.mla_key_tile_size < 0:
        parser.error("--mla-key-tile-size must be non-negative")
    if args.fused_attnres_tile_size < 0:
        parser.error("--fused-attnres-tile-size must be non-negative")
    if args.dense_mlp_tile_size < 0:
        parser.error("--dense-mlp-tile-size must be non-negative")
    if args.layers < 0:
        parser.error("--layers must be non-negative")
    if args.prompt_token_limit < 0:
        parser.error("--prompt-token-limit must be non-negative")

    raw = args.capture.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (digest, len(raw)) != (CAPTURE_SHA256, CAPTURE_BYTES):
        raise SystemExit(
            f"capture identity mismatch: {digest}/{len(raw)}"
        )
    request = json.loads(raw)
    if len(request.get("tools") or []) != CAPTURE_TOOLS:
        raise SystemExit("capture tool-count mismatch")

    import mlx.core as mx

    from runtime.config import ModelConfig
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    checkpoint = ModelConfig.from_dir(args.model_dir)
    released_top_k = int(checkpoint.num_experts_per_tok)
    if args.expert_top_k and not 1 <= args.expert_top_k <= released_top_k:
        parser.error(
            f"--expert-top-k must be in [1, {released_top_k}]"
        )
    schedule = (
        (args.expert_top_k,) * checkpoint.num_hidden_layers
        if args.expert_top_k
        else ()
    )
    rc = RuntimeConfig(
        prefill_chunk_size=args.prefill_tile_width,
        min_weight_cache_mb=150,
        max_weight_cache_mb=args.cache_mb,
        mlx_cache_limit_mb=args.mlx_cache_mb,
        embed_rows=True,
        stream_lm_head=True,
        expert_fetch_batch=args.expert_fetch_batch,
        decode_expert_fetch_batch=args.expert_fetch_batch,
        expert_batch_prefetch=True,
        expert_top_k_by_layer=schedule,
        prefetch_depth=1,
        prefetch_workers=1,
        native_ct_mxfp4=True,
        kimi_k3_scale_sidecar_dir=str(
            args.scale_sidecar_dir.expanduser().resolve()
        ),
        layer_stationary_prefill=True,
        execution_profile=args.execution_profile,
        kimi_k3_native_fused_kda_prefill=(
            args.native_fused_kda_prefill
        ),
        kimi_k3_prefill_tile_policy=args.prefill_tile_policy,
        kimi_k3_compiled_kda_prefill=args.compiled_kda_prefill,
        kimi_k3_compressed_mla=args.compressed_mla,
        kimi_k3_absorbed_mla=args.absorbed_mla,
        kimi_k3_mla_key_tile_size=args.mla_key_tile_size,
        kimi_k3_fused_attnres_tile_size=(
            args.fused_attnres_tile_size
        ),
        kimi_k3_attnres_spill_dir=(
            str(args.attnres_spill_dir.expanduser().resolve())
            if args.attnres_spill_dir else ""
        ),
        kimi_k3_kda_spill_dir=(
            str(args.kda_spill_dir.expanduser().resolve())
            if args.kda_spill_dir else ""
        ),
        kimi_k3_mla_kv_spill_dir=(
            str(args.mla_kv_spill_dir.expanduser().resolve())
            if args.mla_kv_spill_dir else ""
        ),
        kimi_k3_dense_mlp_tile_size=args.dense_mlp_tile_size,
    )
    engine = StreamingEngine(str(args.model_dir), rc)
    try:
        released_layers = int(engine.cfg.num_hidden_layers)
        if args.layers:
            if args.layers > released_layers:
                parser.error(
                    f"--layers {args.layers} exceeds {released_layers}"
                )
            engine.cfg.num_hidden_layers = args.layers
            if engine.cfg.expert_top_k_by_layer:
                engine.cfg.expert_top_k_by_layer = (
                    engine.cfg.expert_top_k_by_layer[:args.layers]
                )
                rc.expert_top_k_by_layer = (
                    rc.expert_top_k_by_layer[:args.layers]
                )
        render_started = time.perf_counter()
        (
            prompt,
            full_prompt_tokens,
            selected_tools,
            _response_tools,
            prompt_metadata,
        ) = _render_capture(
            engine, args.model_dir, request, args.max_output_tokens
        )
        prompt_tokens = full_prompt_tokens
        if args.prompt_token_limit:
            if args.prompt_token_limit > full_prompt_tokens:
                parser.error(
                    "--prompt-token-limit exceeds the rendered prompt: "
                    f"{args.prompt_token_limit} > {full_prompt_tokens}"
                )
            from runtime.server import PreparedPrompt

            prompt = PreparedPrompt(
                str(prompt),
                prompt.token_ids[:args.prompt_token_limit],
            )
            prompt_tokens = args.prompt_token_limit
        render_s = time.perf_counter() - render_started
        prompt_token_ids = list(prompt.token_ids)
        prompt_sha256 = hashlib.sha256(
            json.dumps(
                prompt_token_ids, separators=(",", ":")
            ).encode()
        ).hexdigest()
        base = {
            "schema": "voom.kimi-k3-captured-request-timing.v1",
            "capture": {
                "sha256": digest,
                "bytes": len(raw),
                "messages": len(request.get("input") or []),
                "tools": len(request.get("tools") or []),
                "temperature": request.get("temperature"),
                "stream": request.get("stream"),
                "store": request.get("store"),
                "tool_choice": request.get("tool_choice"),
            },
            "request_mutations": {
                "model": str(args.model_dir.resolve()),
                "max_output_tokens": args.max_output_tokens,
                "prompt_token_limit": (
                    args.prompt_token_limit or "complete"
                ),
                "messages": "preserved",
                "tools": "preserved",
                "temperature": "preserved",
                "seed": (
                    args.seed if args.seed is not None else "preserved"
                ),
                "stream": (
                    "engine timing excludes HTTP/SSE; model prompt is unchanged"
                ),
            },
            "prompt": {
                "tokens": prompt_tokens,
                "full_capture_tokens": full_prompt_tokens,
                "token_sha256": prompt_sha256,
                "render_seconds": render_s,
                "selected_tools": len(selected_tools),
                "schema_profile": prompt_metadata.get("schema_profile"),
                "tool_order_profile": prompt_metadata.get(
                    "tool_order_profile"
                ),
            },
            "profile": {
                # Native KDA preserves the released recurrence algebra but
                # changes FP32 reduction order.  Expert pruning and a reduced
                # top-k are independently lossy.  Keep the artifact honest
                # even when any one of those switches is used alone.
                "lossy": bool(
                    args.expert_top_k
                    or args.native_fused_kda_prefill
                    or engine.cfg.expert_prune_masks
                ),
                "released_expert_top_k": released_top_k,
                "effective_expert_top_k": (
                    args.expert_top_k or released_top_k
                ),
                "released_layers": released_layers,
                "effective_layers": int(engine.cfg.num_hidden_layers),
                "runtime_config": asdict(rc),
                "expert_pruning": (
                    {
                        "enabled": True,
                        "layers": len(engine.cfg.expert_prune_masks),
                        "pruned_per_layer_min": min(
                            len(values)
                            for values in engine.cfg.expert_prune_masks.values()
                        ),
                        "pruned_per_layer_max": max(
                            len(values)
                            for values in engine.cfg.expert_prune_masks.values()
                        ),
                        "manifest_sha256": hashlib.sha256(
                            Path(os.environ[
                                "VMODEL_KIMI_K3_EXPERT_PRUNE_MANIFEST"
                            ]).read_bytes()
                        ).hexdigest(),
                    }
                    if engine.cfg.expert_prune_masks
                    else {"enabled": False}
                ),
            },
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--porcelain"),
            "render_only": args.render_only,
        }
        if args.render_only:
            base["verdict"] = "PASS"
            _atomic_json(args.result.resolve(), base)
            print(json.dumps(base, indent=2, sort_keys=True), flush=True)
            return 0

        stage_before = engine.store.stage_snapshot()
        scale_before = engine.store.k3_scale_sidecar_snapshot()
        sampling = SamplingParams(
            temperature=float(request.get("temperature", 0.0)),
            seed=args.seed,
        )
        generation_started = time.perf_counter()

        def progress(event: dict) -> None:
            if event.get("phase") == "prefill_layer":
                print(
                    "[capture] prefill layer "
                    f"{event.get('completed_layers')}/"
                    f"{event.get('total_layers')} "
                    f"tokens={event.get('total_tokens')} "
                    f"elapsed={time.perf_counter() - generation_started:.3f}s",
                    flush=True,
                )

        # Match runtime.server._engine_generate for the concrete streaming
        # engine. In prompt-length mode this preserves exact arithmetic while
        # allowing the production fail-slow 128 -> 32 -> 8 -> 1 replay ladder
        # when the live governor refuses an unsampled prefill allocation.
        result = engine.generate_with_memory_retry(
            prompt,
            max_tokens=args.max_output_tokens,
            sampling=sampling,
            on_progress=progress,
        )
        stage_after = engine.store.stage_snapshot()
        scale_after = engine.store.k3_scale_sidecar_snapshot()
        path_stats = result["path_stats"]
        output_bytes = engine.tokenizer.decode(result["tokens"]).encode()
        base.update({
            "sampling": asdict(sampling),
            "timing": {
                "prefill_seconds": result["prefill_s"],
                "decode_seconds": result["decode_s"],
                "total_seconds": result["total_s"],
                "first_token_seconds": result["first_token_s"],
                "timed_decode_tokens": max(0, len(result["tokens"]) - 1),
                "decode_seconds_per_token": (
                    result["decode_s"] / (len(result["tokens"]) - 1)
                    if len(result["tokens"]) > 1
                    else None
                ),
            },
            "output": {
                "tokens": result["tokens"],
                "text_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "text_bytes": len(output_bytes),
                "termination_reason": result["termination_reason"],
            },
            "resources": {
                "true_peak_metal_bytes": result[
                    "true_peak_metal_bytes"
                ],
                "kv_bytes": result["kv_bytes"],
                "kv_positions": result["kv_positions"],
                "attnres_spill": getattr(
                    engine, "_last_k3_attnres_spill_stats", None
                ),
                "kda_spill": getattr(
                    engine, "_last_k3_kda_spill_stats", None
                ),
                "mla_kv_spill": getattr(
                    engine, "_last_k3_mla_kv_spill_stats", None
                ),
            },
            "io": {
                "prefill": _phase_stats(path_stats, "prefill"),
                "decode": _phase_stats(path_stats, "decode"),
                "total_weight_store_bytes_read": path_stats.get(
                    "weight_store_bytes_read", 0
                ),
                "total_expert_misses": path_stats.get(
                    "expert_cache_misses", 0
                ),
                "expert_compute_batches": path_stats.get(
                    "expert_compute_batches", 0
                ),
                "expert_prefetch_wait_seconds": path_stats.get(
                    "expert_batch_prefetch_wait_s", 0.0
                ),
                "expert_prefetch_hidden_seconds": path_stats.get(
                    "expert_batch_prefetch_hidden_s", 0.0
                ),
            },
            "cache": {
                "source": path_stats.get("prompt_cache_source", "cold"),
                "prefix_tokens": int(path_stats.get(
                    "prompt_cache_prefix_tokens", 0
                ) or 0),
                "exact_hit": bool(path_stats.get(
                    "prompt_cache_exact_hit", 0
                )),
                "lookup_seconds": float(path_stats.get(
                    "prompt_cache_lookup_s", 0.0
                ) or 0.0),
            },
            "memory_retry": {
                "count": int(path_stats.get(
                    "memory_prefill_retries", 0
                ) or 0),
                "chunks": list(path_stats.get(
                    "memory_prefill_retry_chunks", []
                ) or []),
                "failed_seconds": float(path_stats.get(
                    "memory_prefill_retry_seconds", 0.0
                ) or 0.0),
                "effective_prefill_tile_width": int(path_stats.get(
                    "kimi_k3_prefill_tile_width", rc.prefill_chunk_size
                ) or 0),
            },
            "native_mxfp4_delta": {
                "transform_ns": stage_after[0] - stage_before[0],
                "transform_calls": stage_after[1] - stage_before[1],
                "input_bytes": stage_after[2] - stage_before[2],
                "resident_bytes": stage_after[3] - stage_before[3],
            },
            "scale_sidecar_delta": {
                "read_bytes": scale_after[0] - scale_before[0],
                "output_bytes": scale_after[1] - scale_before[1],
                "decode_ns": scale_after[2] - scale_before[2],
                "decode_calls": scale_after[3] - scale_before[3],
            },
            "execution_profile": result.get("execution_profile"),
            "verdict": "PASS",
        })
        _atomic_json(args.result.resolve(), base)
        print(json.dumps({
            "verdict": "PASS",
            "lossy": base["profile"]["lossy"],
            "effective_expert_top_k": base["profile"][
                "effective_expert_top_k"
            ],
            "prompt_tokens": prompt_tokens,
            **base["timing"],
            **base["resources"],
            "output_tokens": result["tokens"],
        }, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        engine.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
