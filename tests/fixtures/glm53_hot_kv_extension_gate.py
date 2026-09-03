#!/usr/bin/env python3
"""Real-weight exact GLM-5.3 post-generation prefix-extension gate.

The hot arm first generates a bounded prefix response, then extends the exact
fed-back token endpoint with new input.  A fresh control engine computes that
complete sequence cold.  Greedy tokens/text must match and cache telemetry must
prove reuse of the complete available endpoint.  Tensor hashes are also
reported; ``--require-state-exact`` makes them a hard gate, though cold
layer-stationary and incremental continuation GEMM shapes may legitimately
produce different BF16 bits while retaining identical target tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.profiles import apply_runtime_profiles  # noqa: E402
from runtime.server import EngineManager  # noqa: E402
from tests.fixtures.qwen38_dflash2_gate import _state_digest  # noqa: E402


def _repeat_to_length(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise ValueError("tokenizer returned an empty seed")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def _generate(engine, token_ids: list[int], max_tokens: int) -> dict:
    generate = getattr(engine, "generate_with_memory_retry", engine.generate)
    started = time.perf_counter()
    result = generate(
        SimpleNamespace(token_ids=tuple(token_ids)),
        max_tokens=max_tokens,
        stop=[],
    )
    result["client_wall_s"] = time.perf_counter() - started
    return result


def _summary(result: dict) -> dict:
    stats = result["path_stats"]
    return {
        "tokens": list(map(int, result["tokens"])),
        "text": result["text"],
        "prompt_tokens": int(result["prompt_tokens"]),
        "prefill_s": float(result["prefill_s"]),
        "decode_s": float(result["decode_s"]),
        "total_s": float(result["total_s"]),
        "client_wall_s": float(result["client_wall_s"]),
        "true_peak_metal_bytes": int(result["true_peak_metal_bytes"]),
        "cache_source": stats.get("prompt_cache_source"),
        "cache_exact_hit": int(stats.get("prompt_cache_exact_hit", 0)),
        "cache_prefix_tokens": int(stats.get("prompt_cache_prefix_tokens", 0)),
        "cache_lcp_tokens": int(stats.get("hot_prompt_lcp_tokens", 0)),
        "cache_reusable_prefix_tokens": int(
            stats.get("hot_prompt_reusable_prefix_tokens", 0)),
        "prefill_weight_store_bytes_read": int(
            stats.get("prefill_weight_store_bytes_read", 0)),
        "weight_store_bytes_read": int(stats.get("weight_store_bytes_read", 0)),
        "expert_prefetch_wait_s": float(
            stats.get("expert_batch_prefetch_wait_s", 0.0)),
        "expert_prefetch_hidden_s": float(
            stats.get("expert_batch_prefetch_hidden_s", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--control-profile",
        default="glm53-flash-lossless-expert-prefetch-batch8-workers2")
    parser.add_argument(
        "--hot-profile",
        default="glm53-flash-lossless-expert-prefetch-batch8-workers2-hot-kv")
    parser.add_argument("--prefix-tokens", type=int, default=48)
    parser.add_argument("--extension-tokens", type=int, default=12)
    parser.add_argument("--seed-output-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--require-state-exact", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or args.extension_tokens <= 0:
        parser.error("prefix and extension token counts must be positive")
    if args.seed_output_tokens <= 0 or args.max_tokens <= 0:
        parser.error("output token counts must be positive")
    if args.result.exists():
        parser.error(f"result already exists: {args.result}")

    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"model is not a complete checkpoint: {model}")

    # This fixture selects several profiles in one process. Profile settings
    # are defaults by design, so a key written by the control selection would
    # otherwise look like an explicit operator override when the hot arm is
    # selected later. Clear only keys introduced by the preceding selection;
    # unrelated caller environment remains authoritative.
    managed_profile_keys: set[str] = set()

    def activate_profile(name: str) -> None:
        for key in managed_profile_keys:
            os.environ.pop(key, None)
        application = apply_runtime_profiles((name,), activate=True)
        if application is None:
            raise RuntimeError(f"profile {name!r} did not activate")
        managed_profile_keys.clear()
        managed_profile_keys.update(application.setting_keys)

    # Keep native GLM MTP out of this isolation gate. Qwen4 profiles may use
    # their target-verified MTP wrapper; the cache comparison still changes
    # only endpoint retention between the two otherwise identical arms.
    os.environ["VMODEL_GLM53_MTP"] = "0"
    os.environ.pop("VMODEL_GLM53_HOT_PROMPT_KV", None)
    activate_profile(args.control_profile)

    manager = EngineManager()
    try:
        tokenizer_engine = manager.get(model, "fast")
        prefix_seed = list(tokenizer_engine.tokenizer.encode(
            "A careful systems note compares exact schedules, measurements, "
            "and independent validation. ").ids)
        extension_seed = list(tokenizer_engine.tokenizer.encode(
            "\nA distinct follow-up asks for one verified conclusion.").ids)
        prefix = _repeat_to_length(prefix_seed, args.prefix_tokens)
        extension = _repeat_to_length(extension_seed, args.extension_tokens)
    finally:
        manager.close()

    # Produce the exact assistant-token endpoint that the later continuation
    # must contain.  This is done under the hot profile because that profile is
    # responsible for retaining the post-generation target state.
    activate_profile(args.hot_profile)
    hot_manager = EngineManager()
    swap_before = int(psutil.swap_memory().sout)
    try:
        hot_engine = hot_manager.get(model, "fast")
        seed = _generate(hot_engine, prefix, args.seed_output_tokens)
        if not seed["tokens"]:
            raise RuntimeError("seed request emitted no endpoint token")
        complete_input = prefix + list(map(int, seed["tokens"])) + extension
        hot = _generate(hot_engine, complete_input, args.max_tokens)
        hot_state = _state_digest(hot_engine)
    finally:
        hot_manager.close()

    # Start from an empty target state and recompute the identical complete
    # token sequence.  Remove the opt-in before constructing the control.
    activate_profile(args.control_profile)
    control_manager = EngineManager()
    try:
        control_engine = control_manager.get(model, "fast")
        control = _generate(control_engine, complete_input, args.max_tokens)
        control_state = _state_digest(control_engine)
    finally:
        control_manager.close()

    # Autoregressive KV ends before the final emitted token: that token was
    # selected from ``logits`` but is not fed until a later decode/continuation.
    expected_prefix = len(prefix) + len(seed["tokens"]) - 1
    component_exact = {
        name: digest == hot_state["component_sha256"].get(name)
        for name, digest in control_state["component_sha256"].items()
    }
    tensor_exact = control_state["tensor_sha256"] == hot_state["tensor_sha256"]
    hot_stats = hot["path_stats"]
    failures = []
    if control["tokens"] != hot["tokens"]:
        failures.append("output tokens differ")
    if control["text"] != hot["text"]:
        failures.append("output text differs")
    state_warnings = []
    if control_state["sha256"] != hot_state["sha256"] or not tensor_exact:
        state_warnings.append("retained target state differs")
    if not all(component_exact.values()):
        state_warnings.append("one or more target-state components differ")
    if args.require_state_exact:
        failures.extend(state_warnings)
    if hot_stats.get("prompt_cache_source") not in (
            "memory", "hot-prompt-extension"):
        failures.append("hot request did not use an exact extension endpoint")
    if int(hot_stats.get("prompt_cache_prefix_tokens", 0)) != expected_prefix:
        failures.append("hot request reused the wrong prefix length")
    if int(hot_stats.get("prompt_state_approximate", 0)):
        failures.append("hot request reported approximate prompt state")

    document = {
        "schema": "voom.glm53-hot-kv-extension-gate.v1",
        "model": str(model),
        "control_profile": args.control_profile,
        "hot_profile": args.hot_profile,
        "prefix_tokens": len(prefix),
        "seed_output_tokens": len(seed["tokens"]),
        "extension_tokens": len(extension),
        "complete_input_tokens": len(complete_input),
        "expected_reused_prefix_tokens": expected_prefix,
        "seed": _summary(seed),
        "hot": _summary(hot),
        "control": _summary(control),
        "token_exact": control["tokens"] == hot["tokens"],
        "text_exact": control["text"] == hot["text"],
        "state_exact": control_state["sha256"] == hot_state["sha256"],
        "tensor_exact": tensor_exact,
        "component_exact": component_exact,
        "control_state_sha256": control_state["sha256"],
        "hot_state_sha256": hot_state["sha256"],
        "require_state_exact": args.require_state_exact,
        "state_warnings": state_warnings,
        "swap_out_growth_bytes": max(
            0, int(psutil.swap_memory().sout) - swap_before),
        "failures": failures,
        "passed": not failures,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
