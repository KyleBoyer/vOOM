#!/usr/bin/env python3
"""Real Huihui Qwen3.8/DFlash2 target-authoritative state and timing gate.

The plain and speculative arms use fresh ``StreamingEngine`` instances under
the named fast profile.  The gate hashes every retained full-attention K/V
tensor, every DeltaNet recurrent state and convolution history, and the final
retained hidden row.  Matching tokens alone is not sufficient for promotion.

Examples::

  .venv/bin/python tests/fixtures/qwen38_dflash2_gate.py \
    --mode compare --max-tokens 16 --force-reject \
    --result logs/qwen38_dflash2_reject16.json

  .venv/bin/python tests/fixtures/qwen38_dflash2_gate.py \
    --mode spec --max-tokens 64 \
    --result logs/qwen38_dflash2_spec64.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.dflash2_adapter import DFlash2SpeculativeEngine  # noqa: E402
from runtime.profiles import apply_runtime_profiles  # noqa: E402
from runtime.server import EngineManager  # noqa: E402


DEFAULT_TARGET = (
    ROOT / "models" / "Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4")
DEFAULT_DRAFT = ROOT / "models" / "Qwen3.8-27B-DFlash2-mlx-affine4-g64"
DEFAULT_PROMPT = (
    "Explain in six concise bullet points how speculative decoding can speed "
    "a target model while keeping the target distribution authoritative."
)


def _array_payload(value: mx.array) -> bytes:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        return np.asarray(value.view(mx.uint16)).tobytes()
    return np.asarray(value).tobytes()


def _state_digest(engine) -> dict[str, object]:
    kv = engine.last_kv
    if kv is None:
        raise RuntimeError("target did not retain a generation endpoint")
    digest = hashlib.sha256()
    component_digests = {
        name: hashlib.sha256()
        for name in ("attention_kv", "recurrent_state", "conv_history", "hidden")
    }
    tensor_sha256: dict[str, str] = {}
    tensors = 0
    payload_bytes = 0

    def add(component: str, name: str, value) -> None:
        nonlocal tensors, payload_bytes
        if value is None:
            return
        payload = _array_payload(value)
        header = (
            f"{name}|{tuple(map(int, value.shape))}|{value.dtype}|"
            f"{len(payload)}\n"
        ).encode()
        digest.update(header)
        digest.update(payload)
        component_digests[component].update(header)
        component_digests[component].update(payload)
        tensor_sha256[name] = hashlib.sha256(payload).hexdigest()
        tensors += 1
        payload_bytes += len(payload)

    for layer, value in enumerate(kv.keys):
        add("attention_kv", f"kv.{layer}.k", value)
    for layer, value in enumerate(kv.values):
        add("attention_kv", f"kv.{layer}.v", value)

    recurrent = getattr(kv, "kda_cache", None)
    if recurrent is None:
        raise RuntimeError("Qwen endpoint omitted its recurrent companion")
    for layer in range(len(recurrent._state)):
        add("recurrent_state", f"kda.{layer}.state", recurrent.state(layer))
        history = recurrent.conv_history(layer)
        if history is not None:
            for index, value in enumerate(history):
                add("conv_history", f"kda.{layer}.conv.{index}", value)
    add("hidden", "hidden.last", engine._h_last)
    return {
        "sha256": digest.hexdigest(),
        "tensors": tensors,
        "payload_bytes": payload_bytes,
        "kv_offset": int(kv.offset),
        "layer_lengths": list(map(int, kv.layer_lengths())),
        "component_sha256": {
            name: value.hexdigest()
            for name, value in component_digests.items()
        },
        "tensor_sha256": tensor_sha256,
    }


def _configure_profile(profile: str) -> None:
    # These are explicit experimental-arm choices and therefore override the
    # profile defaults before application.  They keep each process isolated:
    # no native MTP, no durable/in-memory prompt endpoint reused across arms.
    os.environ["VMODEL_QWEN_MTP_SPECULATIVE"] = "0"
    os.environ["VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST"] = "0"
    os.environ["VMODEL_QWEN35_HOT_KV_PERSIST_DIR"] = ""
    apply_runtime_profiles((profile,), activate=True)


def _run_arm(
    *,
    mode: str,
    target_path: Path,
    draft_path: Path,
    prompt: str,
    max_tokens: int,
    cap: int,
    force_reject: bool,
    proposal_policy: str,
    fused_dynamic_conv: bool,
    ablation_direction: Path | None,
    ablation_strength: float,
    native_mtp_fallback: bool,
    fallback_min_rounds: int,
    fallback_min_accepted_per_round: float,
    tree_budget: int,
    load_margin_mb: int,
) -> dict[str, object]:
    manager = EngineManager()
    wrapper = None
    swap_before = int(psutil.swap_memory().sout)
    mx.reset_peak_memory()
    started = time.perf_counter()
    try:
        target = manager.get(target_path, "fast")
        if mode == "spec":
            wrapper = DFlash2SpeculativeEngine(
                target,
                draft_path,
                max_draft_tokens=cap,
                max_prompt_tokens=262_144,
                prompt_cache_min_tokens=0,
                release_between_sweeps=True,
                drafter_load_margin_bytes=load_margin_mb * 1_000_000,
                proposal_policy=proposal_policy,
                fused_dynamic_conv=fused_dynamic_conv,
                ablation_direction_dir=ablation_direction,
                ablation_strength=ablation_strength,
                native_mtp_fallback=native_mtp_fallback,
                fallback_min_dflash_rounds=fallback_min_rounds,
                fallback_min_accepted_per_round=(
                    fallback_min_accepted_per_round),
                tree_budget=tree_budget,
            )
            if force_reject:
                mask_token = int(wrapper.decoder._cfg.mask_token_id)

                def propose_reject(
                    _pending, _offset, _ctx_caches, width, _sampling, _history,
                ):
                    return [mask_token] * int(width), None

                wrapper.decoder._propose = propose_reject
            result = wrapper.generate(
                prompt, max_tokens=max_tokens, stop=[])
            served = wrapper
        else:
            generate = getattr(
                target, "generate_with_memory_retry", target.generate)
            result = generate(prompt, max_tokens=max_tokens, stop=[])
            served = target
        wall = time.perf_counter() - started
        endpoint = _state_digest(target)
        stats = dict(result.get("path_stats", {}))
        return {
            "mode": mode,
            "force_reject": bool(force_reject),
            "proposal_policy": proposal_policy if mode == "spec" else None,
            "fused_dynamic_conv": (
                bool(fused_dynamic_conv) if mode == "spec" else None),
            "ablation_direction": (
                str(ablation_direction) if mode == "spec" and ablation_direction
                else None),
            "ablation_strength": (
                ablation_strength if mode == "spec" and ablation_direction
                else None),
            "native_mtp_fallback": (
                bool(native_mtp_fallback) if mode == "spec" else None),
            "fallback_min_rounds": (
                fallback_min_rounds if mode == "spec" else None),
            "fallback_min_accepted_per_round": (
                fallback_min_accepted_per_round if mode == "spec" else None),
            "tree_budget": tree_budget if mode == "spec" else 0,
            "load_margin_mb": load_margin_mb if mode == "spec" else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_tokens": max_tokens,
            "cap": cap if mode == "spec" else 0,
            "tokens": list(map(int, result["tokens"])),
            "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
            "wall_seconds": round(wall, 4),
            "prefill_seconds": round(float(result.get("prefill_s", 0.0)), 4),
            "decode_seconds": round(float(result.get("decode_s", 0.0)), 4),
            "true_peak_metal_bytes": int(mx.get_peak_memory()),
            "swap_out_growth_bytes": max(
                0, int(psutil.swap_memory().sout) - swap_before),
            "state": endpoint,
            "speculative_kind": getattr(served, "_speculative_kind", None),
            "path_stats": stats,
        }
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            manager.close()
        gc.collect()
        mx.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain", "spec", "compare"),
                        default="compare")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--profile", default="huihui-qwen38-27b-fast-agent")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--cap", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--force-reject", action="store_true")
    parser.add_argument("--proposal-policy", choices=("selector", "unary"),
                        default="selector")
    parser.add_argument("--fused-dynamic-conv", action="store_true")
    parser.add_argument("--ablation-direction", type=Path)
    parser.add_argument("--ablation-strength", type=float, default=1.0)
    parser.add_argument("--native-mtp-fallback", action="store_true")
    parser.add_argument("--fallback-min-rounds", type=int, default=4)
    parser.add_argument(
        "--fallback-min-accepted-per-round", type=float, default=1.0)
    parser.add_argument("--tree-budget", type=int, choices=range(0, 9),
                        default=0)
    parser.add_argument("--load-margin-mb", type=int, default=400)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("max-tokens must be at least 2")
    if args.fallback_min_rounds <= 0:
        parser.error("fallback-min-rounds must be positive")
    if not 0 <= args.fallback_min_accepted_per_round <= 4:
        parser.error("fallback-min-accepted-per-round must be in [0, 4]")
    if args.force_reject and args.tree_budget:
        parser.error("force-reject and tree-budget are mutually exclusive")
    if args.load_margin_mb < 0:
        parser.error("load-margin-mb must be non-negative")
    target = args.target.expanduser().resolve()
    draft = args.draft.expanduser().resolve()
    if not (target / "config.json").is_file():
        parser.error(f"target is not a complete checkpoint: {target}")
    if not (draft / "model.safetensors").is_file():
        parser.error(f"draft is not a complete sidecar: {draft}")

    _configure_profile(args.profile)
    arms = []
    if args.mode in ("plain", "compare"):
        arms.append(_run_arm(
            mode="plain", target_path=target, draft_path=draft,
            prompt=args.prompt, max_tokens=args.max_tokens, cap=args.cap,
            force_reject=False, proposal_policy=args.proposal_policy,
            fused_dynamic_conv=False, ablation_direction=None,
            ablation_strength=1.0, native_mtp_fallback=False,
            fallback_min_rounds=args.fallback_min_rounds,
            fallback_min_accepted_per_round=(
                args.fallback_min_accepted_per_round),
            tree_budget=0,
            load_margin_mb=args.load_margin_mb))
    if args.mode in ("spec", "compare"):
        arms.append(_run_arm(
            mode="spec", target_path=target, draft_path=draft,
            prompt=args.prompt, max_tokens=args.max_tokens, cap=args.cap,
            force_reject=args.force_reject,
            proposal_policy=args.proposal_policy,
            fused_dynamic_conv=args.fused_dynamic_conv,
            ablation_direction=(
                args.ablation_direction.expanduser().resolve()
                if args.ablation_direction else None),
            ablation_strength=args.ablation_strength,
            native_mtp_fallback=args.native_mtp_fallback,
            fallback_min_rounds=args.fallback_min_rounds,
            fallback_min_accepted_per_round=(
                args.fallback_min_accepted_per_round),
            tree_budget=args.tree_budget,
            load_margin_mb=args.load_margin_mb))

    report: dict[str, object] = {
        "schema": "voom.qwen38-dflash2-gate.v1",
        "profile": args.profile,
        "target": str(target),
        "draft": str(draft),
        "arms": arms,
    }
    exit_code = 0
    if len(arms) == 2:
        plain, spec = arms
        token_exact = plain["tokens"] == spec["tokens"]
        plain_state = plain["state"]
        spec_state = spec["state"]
        component_exact = {
            name: value == spec_state["component_sha256"].get(name)
            for name, value in plain_state["component_sha256"].items()
        }
        endpoint_layout_exact = (
            plain_state["kv_offset"] == spec_state["kv_offset"]
            and plain_state["layer_lengths"] == spec_state["layer_lengths"]
        )
        # `_h_last` is an auxiliary draft/constraint projection row, not the
        # target's persistent recurrent state. Serial multi-position verify
        # may reassociate its final dense row while still retaining byte-exact
        # full-attention KV, DeltaNet matrices, and convolution histories.
        # Report it independently; never let it hide a recurrent mismatch.
        released_state_exact = bool(
            endpoint_layout_exact
            and component_exact["attention_kv"]
            and component_exact["recurrent_state"]
            and component_exact["conv_history"]
        )
        state_exact = bool(released_state_exact and component_exact["hidden"])
        report.update({
            "token_exact": token_exact,
            "state_exact": state_exact,
            "released_state_exact": released_state_exact,
            "component_exact": component_exact,
            "passed": bool(token_exact and released_state_exact),
            "speedup": round(
                float(plain["wall_seconds"]) / float(spec["wall_seconds"]),
                4,
            ),
        })
        exit_code = 0 if report["passed"] else 1
    else:
        report["passed"] = True
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(rendered, flush=True)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
