#!/usr/bin/env python3
"""Gate an unseen K3 harness request from an exact static-prefix endpoint.

The seed contains only the request's tools plus leading system/developer
messages.  No user/history item from the target is admitted to the seed.  The
target capture remains byte-for-byte untouched; this script refuses the run
unless the seed and target dynamic-message hashes differ and the independently
rendered static prefixes are token-identical.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.kimi_k3_captured_request_timing import (
    _phase_stats,
    _render_capture,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _json_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _token_hash(values) -> str:
    return hashlib.sha256(json.dumps(
        list(values), separators=(",", ":")
    ).encode()).hexdigest()


def _longest_common_prefix(left, right) -> tuple[int, ...]:
    length = min(len(left), len(right))
    index = 0
    while index < length and left[index] == right[index]:
        index += 1
    return tuple(left[:index])


def _fallback_static_stem(prompt: str) -> str:
    """Remove only the generic renderer's synthetic generation marker."""
    marker = "assistant:"
    if not prompt.endswith(marker):
        raise ValueError(
            "K3 shared-prefix gate expected the generic fallback renderer"
        )
    return prompt[:-len(marker)]


def _tokenizer_safe_static_prefix(engine, static_stem: str) -> tuple[int, ...]:
    """Find the exact token boundary before any dynamic message content.

    BPE tokenization is not generally prefix-stable at an arbitrary character
    boundary.  Two deliberately different synthetic user continuations expose
    the maximal stable token prefix while containing no served-request data.
    The caller still proves these IDs are a literal prefix of both real
    captures before allowing model execution.
    """
    left_encoding = engine.tokenizer.encode(
        static_stem + "user: A"
    )
    right_encoding = engine.tokenizer.encode(
        static_stem + "user: Z"
    )
    left = tuple(getattr(left_encoding, "ids", left_encoding))
    right = tuple(getattr(right_encoding, "ids", right_encoding))
    prefix = _longest_common_prefix(left, right)
    if not prefix:
        raise ValueError("synthetic continuations have no stable token prefix")
    return prefix


def _read_capture(path: Path, expected_sha256: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"capture identity mismatch for {path}: {digest} != "
            f"{expected_sha256}"
        )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"capture {path} is not a JSON object")
    return raw, value


def _leading_static_request(request: dict) -> tuple[dict, list, list]:
    items = request.get("input")
    if not isinstance(items, list):
        raise ValueError("shared-prefix gate requires list-valued Responses input")
    leading = []
    for item in items:
        if not isinstance(item, dict):
            break
        role = str(item.get("role", item.get("type", "")))
        if role not in ("system", "developer"):
            break
        leading.append(item)
    dynamic = items[len(leading):]
    if not leading:
        raise ValueError("capture has no leading system/developer preamble")
    if not dynamic:
        raise ValueError("capture has no dynamic user/history suffix")
    static = dict(request)
    static["input"] = leading
    return static, leading, dynamic


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def _sampling_for(request: dict, seed: int | None):
    from runtime.sampler import SamplingParams

    return SamplingParams(
        temperature=float(request.get("temperature", 0.0) or 0.0),
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-capture", required=True, type=Path)
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--target-capture", required=True, type=Path)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models" / "Kimi-K3"
    )
    parser.add_argument(
        "--scale-sidecar-dir", type=Path,
        default=ROOT / "logs" / "f139_k3_scale_sidecar_full",
    )
    parser.add_argument("--expert-prune-manifest", type=Path)
    parser.add_argument("--expert-top-k", type=int, default=0)
    parser.add_argument("--expert-fetch-batch", type=int, default=16)
    parser.add_argument("--cache-mb", type=int, default=150)
    parser.add_argument("--mlx-cache-mb", type=int, default=64)
    parser.add_argument("--prefill-tile-width", type=int, default=128)
    parser.add_argument("--mla-key-tile-size", type=int, default=1024)
    parser.add_argument("--fused-attnres-tile-size", type=int, default=128)
    parser.add_argument("--dense-mlp-tile-size", type=int, default=256)
    parser.add_argument("--attnres-spill-dir", required=True, type=Path)
    parser.add_argument("--kda-spill-dir", required=True, type=Path)
    parser.add_argument("--mla-kv-spill-dir", required=True, type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--max-target-first-token-seconds", type=float, default=0.0,
        help="Positive values make the target latency part of the PASS gate.",
    )
    args = parser.parse_args()

    if args.result.exists():
        raise SystemExit(f"refusing existing result: {args.result}")
    if args.seed_capture.resolve() == args.target_capture.resolve():
        parser.error("seed and target captures must be different files")
    if args.seed_sha256 == args.target_sha256:
        parser.error("seed and target capture hashes must differ")
    if args.max_output_tokens < 2 and not args.render_only:
        parser.error("--max-output-tokens must be at least 2")
    if not 1 <= args.expert_fetch_batch <= 64:
        parser.error("--expert-fetch-batch must be in [1, 64]")
    if args.cache_mb < 150 or args.mlx_cache_mb < 64:
        parser.error("cache limits are below the gated minimum")
    if min(
        args.prefill_tile_width,
        args.mla_key_tile_size,
        args.fused_attnres_tile_size,
        args.dense_mlp_tile_size,
    ) <= 0:
        parser.error("all tile sizes must be positive")
    if args.layers < 0:
        parser.error("--layers must be non-negative")
    if args.max_target_first_token_seconds < 0:
        parser.error("latency threshold must be non-negative")

    seed_raw, seed_request = _read_capture(
        args.seed_capture, args.seed_sha256)
    target_raw, target_request = _read_capture(
        args.target_capture, args.target_sha256)
    seed_static, seed_leading, seed_dynamic = _leading_static_request(
        seed_request)
    target_static, target_leading, target_dynamic = _leading_static_request(
        target_request)
    seed_dynamic_hash = _json_hash(seed_dynamic)
    target_dynamic_hash = _json_hash(target_dynamic)
    if seed_dynamic_hash == target_dynamic_hash:
        raise SystemExit(
            "seed and target dynamic message/history hashes are identical"
        )
    if _json_hash(seed_request.get("tools") or []) != _json_hash(
        target_request.get("tools") or []
    ):
        raise SystemExit("seed and target tool catalogs differ")
    if _json_hash(seed_leading) != _json_hash(target_leading):
        raise SystemExit("seed and target leading system/developer messages differ")

    if args.expert_prune_manifest is not None:
        manifest = args.expert_prune_manifest.expanduser().resolve()
        if not manifest.is_file():
            parser.error(f"expert prune manifest is missing: {manifest}")
        os.environ["VMODEL_KIMI_K3_EXPERT_PRUNE_MANIFEST"] = str(manifest)

    import mlx.core as mx

    from runtime.config import ModelConfig
    from runtime.engine import RuntimeConfig, StreamingEngine

    checkpoint = ModelConfig.from_dir(args.model_dir)
    released_top_k = int(checkpoint.num_experts_per_tok)
    if args.expert_top_k and not 1 <= args.expert_top_k <= released_top_k:
        parser.error(
            f"--expert-top-k must be in [1, {released_top_k}]"
        )
    top_k_schedule = (
        (args.expert_top_k,) * int(checkpoint.num_hidden_layers)
        if args.expert_top_k else ()
    )
    rc = RuntimeConfig(
        prefill_chunk_size=args.prefill_tile_width,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=args.prefill_tile_width,
        hot_prompt_kv_min_tokens=1,
        hot_prompt_kv_slots=1,
        min_weight_cache_mb=150,
        max_weight_cache_mb=args.cache_mb,
        mlx_cache_limit_mb=args.mlx_cache_mb,
        embed_rows=True,
        stream_lm_head=True,
        expert_fetch_batch=args.expert_fetch_batch,
        decode_expert_fetch_batch=args.expert_fetch_batch,
        expert_batch_prefetch=True,
        expert_top_k_by_layer=top_k_schedule,
        prefetch_depth=1,
        prefetch_workers=1,
        native_ct_mxfp4=True,
        kimi_k3_scale_sidecar_dir=str(
            args.scale_sidecar_dir.expanduser().resolve()
        ),
        layer_stationary_prefill=True,
        execution_profile="layers",
        kimi_k3_native_fused_kda_prefill=True,
        kimi_k3_prefill_tile_policy="fixed",
        kimi_k3_compressed_mla=True,
        kimi_k3_absorbed_mla=True,
        kimi_k3_mla_key_tile_size=args.mla_key_tile_size,
        kimi_k3_fused_attnres_tile_size=args.fused_attnres_tile_size,
        kimi_k3_attnres_spill_dir=str(
            args.attnres_spill_dir.expanduser().resolve()
        ),
        kimi_k3_kda_spill_dir=str(
            args.kda_spill_dir.expanduser().resolve()
        ),
        kimi_k3_mla_kv_spill_dir=str(
            args.mla_kv_spill_dir.expanduser().resolve()
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
        seed_prompt, seed_tokens, *_ = _render_capture(
            engine, args.model_dir, seed_request, args.max_output_tokens
        )
        target_prompt, target_tokens, *_ = _render_capture(
            engine, args.model_dir, target_request, args.max_output_tokens
        )
        static_rendered_prompt, _static_rendered_tokens, *_ = _render_capture(
            engine, args.model_dir, seed_static, 1
        )
        (
            target_static_rendered_prompt,
            _target_static_rendered_tokens,
            *_,
        ) = _render_capture(engine, args.model_dir, target_static, 1)
        render_seconds = time.perf_counter() - render_started

        static_stem = _fallback_static_stem(str(static_rendered_prompt))
        target_static_stem = _fallback_static_stem(
            str(target_static_rendered_prompt)
        )
        if static_stem != target_static_stem:
            raise SystemExit("independently rendered static prompt text differs")
        static_ids = _tokenizer_safe_static_prefix(engine, static_stem)
        seed_ids = tuple(seed_prompt.token_ids)
        target_ids = tuple(target_prompt.token_ids)
        if (
            not static_ids
            or len(static_ids) >= min(len(seed_ids), len(target_ids))
        ):
            raise SystemExit(
                "static prefix is empty or consumes a complete request"
            )
        if seed_ids[:len(static_ids)] != static_ids:
            raise SystemExit("static seed is not an exact seed-request prefix")
        if target_ids[:len(static_ids)] != static_ids:
            raise SystemExit("static seed is not an exact target-request prefix")
        if seed_ids[len(static_ids):] == target_ids[len(static_ids):]:
            raise SystemExit("rendered dynamic suffixes are identical")

        from runtime.server import PreparedPrompt

        # Engine.generate consumes PreparedPrompt.token_ids directly.  The
        # display string deliberately contains only static data; its token IDs
        # are the proven tokenizer-safe prefix above rather than a retokenized
        # guess at an arbitrary character boundary.
        static_prompt = PreparedPrompt(static_stem, static_ids)

        base = {
            "schema": "voom.kimi-k3-shared-prefix-gate.v1",
            "seed_capture": {
                "path": str(args.seed_capture.resolve()),
                "sha256": args.seed_sha256,
                "bytes": len(seed_raw),
                "messages": len(seed_request.get("input") or []),
                "tools": len(seed_request.get("tools") or []),
                "dynamic_messages_sha256": seed_dynamic_hash,
                "prompt_tokens": seed_tokens,
                "prompt_token_sha256": _token_hash(seed_ids),
            },
            "target_capture": {
                "path": str(args.target_capture.resolve()),
                "sha256": args.target_sha256,
                "bytes": len(target_raw),
                "messages": len(target_request.get("input") or []),
                "tools": len(target_request.get("tools") or []),
                "temperature": target_request.get("temperature"),
                "stream": target_request.get("stream"),
                "tool_choice": target_request.get("tool_choice"),
                "dynamic_messages_sha256": target_dynamic_hash,
                "prompt_tokens": target_tokens,
                "prompt_token_sha256": _token_hash(target_ids),
            },
            "shared_prefix": {
                "derivation": (
                    "tools-plus-leading-system-developer-plus-user-role-"
                    "boundary"
                ),
                "target_dynamic_content_used_in_seed": False,
                "tools_sha256": _json_hash(seed_request.get("tools") or []),
                "leading_messages_sha256": _json_hash(seed_leading),
                "token_sha256": _token_hash(static_ids),
                "tokens": len(static_ids),
                "seed_suffix_tokens": len(seed_ids) - len(static_ids),
                "target_suffix_tokens": len(target_ids) - len(static_ids),
                "independent_target_static_text_match": True,
                "synthetic_continuation_boundary": True,
            },
            "request_mutations": {
                "seed": "static-prefix warmup, not a served request",
                "target_capture": "byte-for-byte preserved",
                "target_messages": "preserved",
                "target_tools": "preserved",
                "target_temperature": "preserved",
                "target_stream": (
                    "engine timing excludes HTTP/SSE; model prompt is unchanged"
                ),
                "target_model": str(args.model_dir.resolve()),
                "target_max_output_tokens": args.max_output_tokens,
                "sampling_seed": (
                    args.seed if args.seed is not None else "preserved"
                ),
            },
            "profile": {
                "lossy": bool(
                    args.expert_top_k
                    or args.expert_prune_manifest
                    or rc.kimi_k3_native_fused_kda_prefill
                ),
                "released_layers": released_layers,
                "effective_layers": int(engine.cfg.num_hidden_layers),
                "released_expert_top_k": released_top_k,
                "effective_expert_top_k": (
                    args.expert_top_k or released_top_k
                ),
                "runtime_config": asdict(rc),
                "expert_prune_manifest_sha256": (
                    hashlib.sha256(
                        args.expert_prune_manifest.read_bytes()
                    ).hexdigest()
                    if args.expert_prune_manifest else None
                ),
            },
            "render_seconds": render_seconds,
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--porcelain"),
            "render_only": args.render_only,
        }
        if args.render_only:
            base["verdict"] = "PASS"
            _atomic_json(args.result.resolve(), base)
            print(json.dumps(base, indent=2, sort_keys=True), flush=True)
            return 0

        seed_started = time.perf_counter()
        seed_result = engine.generate(
            static_prompt,
            max_tokens=1,
            sampling=_sampling_for(seed_request, args.seed),
        )
        seed_wall = time.perf_counter() - seed_started
        slots = list(engine._hot_prompt_slots)
        if len(slots) != 1:
            raise RuntimeError(f"expected one seeded hot slot, got {len(slots)}")
        slot = slots[0]
        if tuple(slot.tokens) != static_ids or slot.kv.offset != len(static_ids):
            raise RuntimeError(
                "seeded endpoint does not exactly match the static prefix"
            )

        target_started = time.perf_counter()
        target_result = engine.generate(
            target_prompt,
            max_tokens=args.max_output_tokens,
            sampling=_sampling_for(target_request, args.seed),
        )
        target_wall = time.perf_counter() - target_started
        target_stats = target_result["path_stats"]
        cache_pass = (
            target_stats.get("prompt_cache_source") == "memory"
            and int(target_stats.get("prompt_cache_prefix_tokens", 0))
            == len(static_ids)
            and int(target_stats.get("hot_prompt_lcp_tokens", 0))
            == len(static_ids)
            and not int(target_stats.get("prompt_cache_exact_hit", 0))
        )
        latency_pass = (
            not args.max_target_first_token_seconds
            or target_result["first_token_s"]
            <= args.max_target_first_token_seconds
        )
        peak_pass = target_result["true_peak_metal_bytes"] <= 8_500_000_000
        verdict = "PASS" if cache_pass and latency_pass and peak_pass else "FAIL"
        output_bytes = engine.tokenizer.decode(target_result["tokens"]).encode()
        base.update({
            "seed": {
                "wall_seconds": seed_wall,
                "prefill_seconds": seed_result["prefill_s"],
                "first_token_seconds": seed_result["first_token_s"],
                "output_tokens": seed_result["tokens"],
                "kv_positions": seed_result["kv_positions"],
                "cache_source": seed_result["path_stats"].get(
                    "prompt_cache_source"
                ),
                "true_peak_metal_bytes": seed_result[
                    "true_peak_metal_bytes"
                ],
            },
            "target": {
                "wall_seconds": target_wall,
                "prefill_seconds": target_result["prefill_s"],
                "first_token_seconds": target_result["first_token_s"],
                "decode_seconds": target_result["decode_s"],
                "total_seconds": target_result["total_s"],
                "output_tokens": target_result["tokens"],
                "output_text_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "kv_positions": target_result["kv_positions"],
                "true_peak_metal_bytes": target_result[
                    "true_peak_metal_bytes"
                ],
                "cache": {
                    "source": target_stats.get("prompt_cache_source"),
                    "prefix_tokens": target_stats.get(
                        "prompt_cache_prefix_tokens", 0
                    ),
                    "lcp_tokens": target_stats.get("hot_prompt_lcp_tokens", 0),
                    "exact_hit": target_stats.get(
                        "prompt_cache_exact_hit", 0
                    ),
                    "lookup_seconds": target_stats.get(
                        "prompt_cache_lookup_s", 0.0
                    ),
                },
                "io": {
                    "prefill": _phase_stats(target_stats, "prefill"),
                    "decode": _phase_stats(target_stats, "decode"),
                    "total_weight_store_bytes_read": target_stats.get(
                        "weight_store_bytes_read", 0
                    ),
                },
            },
            "gates": {
                "cache_extension_exact": cache_pass,
                "first_token_threshold_seconds": (
                    args.max_target_first_token_seconds or None
                ),
                "first_token_threshold_pass": latency_pass,
                "peak_metal_at_most_8_5gb": peak_pass,
            },
            "verdict": verdict,
        })
        _atomic_json(args.result.resolve(), base)
        print(json.dumps({
            "verdict": verdict,
            "shared_prefix_tokens": len(static_ids),
            "target_suffix_tokens": len(target_ids) - len(static_ids),
            "seed_wall_seconds": seed_wall,
            "target_first_token_seconds": target_result["first_token_s"],
            "target_decode_seconds": target_result["decode_s"],
            "target_total_seconds": target_result["total_s"],
            "target_cache": base["target"]["cache"],
            "target_output_tokens": target_result["tokens"],
            "true_peak_metal_bytes": target_result["true_peak_metal_bytes"],
        }, indent=2, sort_keys=True), flush=True)
        return 0 if verdict == "PASS" else 1
    finally:
        engine.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
