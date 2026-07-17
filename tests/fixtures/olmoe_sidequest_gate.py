#!/usr/bin/env python3
"""Real-checkpoint speed/token gate for the resident OLMoE side quest.

This is intentionally outside the ordinary pytest suite. It requires a complete
expert-quantized OLMoE artifact and compares the candidate-reranked head against
the same engine's exact BF16 head on several chat prompts.

    python tests/fixtures/olmoe_sidequest_gate.py \
      /path/to/OLMoE-1B-7B-0924-Instruct-mlx-expert-mxfp4

    # Explicit lossy schedule: top-7 on layers 0-8, released top-8 elsewhere.
    python tests/fixtures/olmoe_sidequest_gate.py MODEL \
      --expert-top-k 7 --expert-top-k-layers 0-8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.quant import RerankedQHead
from runtime.server import _chat_prompt


PROMPTS = (
    "Explain deterministic validation with two concrete examples.",
    "Return only JSON: what is 91 minus 37?",
    "Write a Python binary search function with type hints.",
    "List the first twenty Fibonacci numbers separated by commas.",
    "Compare mutexes and semaphores in concise technical prose.",
)

EXTENDED_HEAD_PROMPTS = (
    "Return only JSON with keys result and explanation: compute 137 * 29.",
    "Write a Python merge sort with type hints and explain stability.",
    "Explain memory ordering and acquire-release semantics concisely.",
    "List the first 40 prime numbers separated only by commas.",
    "Compare TCP flow control and congestion control in a table.",
    "Give a rigorous counterexample to the claim that correlation implies causation.",
    "Write Rust code for a bounded ring buffer and mention its invariants.",
    "Explain Unicode normalization forms NFC and NFD with examples.",
    'Return only JSON: {"result": value} for (918 - 273) / 5.',
    "Describe three failure modes in speculative decoding implementations.",
    "Prove by induction that the sum of the first n odd integers is n squared.",
    "Generate a concise SQL query using a window function for top-3 per group.",
)

FIBONACCI_20 = (
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
    55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
)


def _task_quality(case: int, text: str, max_tokens: int) -> bool | None:
    """Cheap deterministic checks for prompts with machine-checkable answers."""
    if case == 1:
        try:
            value, _end = json.JSONDecoder().raw_decode(text.lstrip())
        except (TypeError, ValueError):
            return False
        return isinstance(value, dict) and value.get("result") == 54
    if case == 3 and max_tokens >= 64:
        numbers = tuple(int(value) for value in re.findall(r"\b\d+\b", text))
        return numbers == FIBONACCI_20
    return None


def _parse_layer_selection(spec: str, num_layers: int) -> tuple[int, ...]:
    """Parse zero-based layer indices, inclusive ranges, or named subsets."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    first_third = num_layers // 3
    second_third = 2 * num_layers // 3
    named = {
        "all": range(num_layers),
        "early": range(0, first_third),
        "middle": range(first_third, second_third),
        "late": range(second_third, num_layers),
        "even": range(0, num_layers, 2),
        "odd": range(1, num_layers, 2),
    }
    selected: set[int] = set()
    for raw_term in spec.split(","):
        term = raw_term.strip().lower()
        if not term:
            raise ValueError("layer selection contains an empty term")
        if term in named:
            selected.update(named[term])
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", term)
        if match is None:
            raise ValueError(
                f"invalid layer term {raw_term!r}; expected an index, inclusive "
                "range, or all/early/middle/late/even/odd")
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if start > stop:
            raise ValueError(f"descending layer range is not allowed: {term!r}")
        if stop >= num_layers:
            raise ValueError(
                f"layer {stop} is outside the zero-based range [0, {num_layers})")
        selected.update(range(start, stop + 1))
    if not selected:
        raise ValueError("layer selection must not be empty")
    return tuple(sorted(selected))


def _make_expert_top_k_schedule(
    num_layers: int,
    released_top_k: int,
    selected_top_k: int,
    layer_spec: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build a complete, fail-closed top-k schedule and selected-layer list."""
    if not 1 <= selected_top_k <= released_top_k:
        raise ValueError(
            f"scheduled top-k must be within [1, {released_top_k}], "
            f"got {selected_top_k}")
    selected = _parse_layer_selection(layer_spec, num_layers)
    schedule = [released_top_k] * num_layers
    for layer in selected:
        schedule[layer] = selected_top_k
    return tuple(schedule), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument(
        "--extended-head-recall", action="store_true",
        help="append a diverse 12-prompt/256-token BF16-head recall corpus")
    parser.add_argument("--rerank-candidates", type=int, default=32)
    parser.add_argument("--kv-step", type=int, default=256)
    parser.add_argument(
        "--attention-profile",
        choices=(
            "bf16", "mxfp8", "mxfp4", "nvfp4",
            "mxfp8-v-o-nvfp4", "mxfp8-v-nvfp4",
            "mxfp8-o-nvfp4",
            "affine6-g64",
            "mxfp8-v-o-affine6", "mxfp8-q-k-affine6",
            "mxfp8-q-affine6", "mxfp8-k-affine6",
        ),
        default="mxfp8",
    )
    parser.add_argument(
        "--head-profile", choices=(
            "auto", "mxfp4", "nvfp4", "mxfp8",
            "affine2-g64", "affine3-g64", "affine4-g64", "affine6-g64",
        ),
        default="auto")
    parser.add_argument("--expert-top-k", type=int, default=0)
    parser.add_argument(
        "--expert-top-k-layers", metavar="SPEC", default="",
        help=(
            "apply --expert-top-k only to zero-based indices/inclusive ranges "
            "(for example 0-3,8,10-12) or named subsets "
            "all/early/middle/late/even/odd; omitted means a global override"
        ),
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.kv_step <= 0:
        parser.error("--kv-step must be positive")
    if args.expert_top_k_layers and not args.expert_top_k:
        parser.error("--expert-top-k-layers requires --expert-top-k")

    from runtime.kv_cache import SteppedKVCache

    SteppedKVCache.step = args.kv_step

    config = json.loads((args.model / "config.json").read_text())
    quantization = config.get("quantization", {})
    if config.get("model_type") != "olmoe" or not quantization:
        parser.error("model must be a quantized OLMoE checkpoint")

    selected_top_k_layers: tuple[int, ...] = ()
    expert_top_k_by_layer: tuple[int, ...] = ()
    if args.expert_top_k:
        num_experts = int(config.get(
            "num_experts", config.get("num_local_experts", 0)))
        if not 1 <= args.expert_top_k <= num_experts:
            parser.error("--expert-top-k must be within the model's expert count")
        if args.expert_top_k_layers:
            try:
                expert_top_k_by_layer, selected_top_k_layers = (
                    _make_expert_top_k_schedule(
                        int(config["num_hidden_layers"]),
                        int(config["num_experts_per_tok"]),
                        args.expert_top_k,
                        args.expert_top_k_layers,
                    )
                )
            except ValueError as error:
                parser.error(str(error))

    expert_mxfp8 = (
        quantization.get("mode") == "mxfp8"
        and int(quantization.get("bits", 0)) == 8
    )
    head_profile = (
        ("mxfp4" if expert_mxfp8 else "affine2-g64")
        if args.head_profile == "auto" else args.head_profile
    )
    rc = RuntimeConfig(
        max_weight_cache_mb=(
            9000 if quantization.get("mode") == "mxfp8" else 7000),
        pin_lm_head=True,
        quant_bits=4,
        quant_group_size=32,
        quant_mode="mxfp4",
        quant_min_dim=0,
        quant_attention=False,
        quant_router=False,
        quant_lm_head=False,
        resident_moe_decode=True,
        expert_top_k_by_layer=expert_top_k_by_layer,
        resident_attention_mode=(
            "" if (expert_mxfp8 or args.attention_profile == "bf16"
                   or args.attention_profile.startswith("mxfp8-"))
            else ("affine" if args.attention_profile.startswith("affine")
                  else args.attention_profile)
        ),
        resident_attention_bits=(
            8 if args.attention_profile == "mxfp8" else
            6 if args.attention_profile == "affine6-g64" else 4),
        resident_attention_group_size=(
            16 if args.attention_profile == "nvfp4" else
            64 if args.attention_profile.startswith("affine") else 32),
        stepped_kv_threshold=1,
        fused_swiglu=True,
        rerank_lm_head=True,
        rerank_lm_head_candidates=args.rerank_candidates,
        rerank_lm_head_mode=(
            "affine" if head_profile.startswith("affine")
            else head_profile),
        rerank_lm_head_bits=(
            int(head_profile.removeprefix("affine").split("-", 1)[0])
            if head_profile.startswith("affine")
            else 8 if head_profile == "mxfp8" else 4),
        rerank_lm_head_group_size=(
            64 if head_profile.startswith("affine")
            else 16 if head_profile == "nvfp4" else 32),
        prefill_chunk_size=2048,
        prefill_last_token_separate=True,
        governor=True,
    )
    engine = StreamingEngine(args.model, rc)
    if args.expert_top_k and not args.expert_top_k_layers:
        engine.cfg.num_experts_per_tok = args.expert_top_k
    if args.attention_profile.startswith("mxfp8-"):
        from olmoe_attention_quality_gate import _candidate_layers

        engine._resident_moe_layers = _candidate_layers(
            engine, args.attention_profile)
    reranked = engine._lm_head_w
    if not isinstance(reranked, RerankedQHead):
        engine.close()
        raise RuntimeError("engine did not activate the reranked LM head")

    exact_tokens = 0
    base_decode_s = 0.0
    fast_decode_s = 0.0
    peak = 0
    mismatches = []
    quality_failures = []
    prompts = list(PROMPTS)
    if args.extended_head_recall:
        prompts.extend(EXTENDED_HEAD_PROMPTS)
    try:
        for index, question in enumerate(prompts):
            case_max_tokens = (
                max(args.max_tokens, 256)
                if index >= len(PROMPTS) else args.max_tokens)
            prompt = _chat_prompt(
                engine, args.model,
                [{"role": "user", "content": question}], "medium")
            engine._lm_head_w = reranked.exact
            baseline = engine.generate(prompt, max_tokens=case_max_tokens)
            engine._lm_head_w = reranked
            fast = engine.generate(prompt, max_tokens=case_max_tokens)
            same = fast["tokens"] == baseline["tokens"]
            if not same:
                mismatches.append(index)
            quality = (_task_quality(index, fast["text"], case_max_tokens)
                       if index < len(PROMPTS) else None)
            if quality is False:
                quality_failures.append(index)
            exact_tokens += len(baseline["tokens"])
            base_decode_s += baseline["decode_s"]
            fast_decode_s += fast["decode_s"]
            peak = max(peak, fast["true_peak_metal_bytes"])
            row = {
                "case": index,
                "tokens": len(baseline["tokens"]),
                "exact": same,
                "baseline_tps": baseline["tok_per_s"],
                "reranked_tps": fast["tok_per_s"],
                "quality": quality,
            }
            if args.show_text:
                row["text"] = fast["text"]
            print(json.dumps(row), flush=True)
    finally:
        engine._lm_head_w = reranked
        engine.close()
        mx.clear_cache()

    # decode_s covers tokens after the first prompt prediction.
    measured_tokens = max(0, exact_tokens - len(prompts))
    summary = {
        "model": str(args.model),
        "rerank_candidates": args.rerank_candidates,
        "kv_step": args.kv_step,
        "attention_profile": args.attention_profile,
        "head_profile": head_profile,
        "expert_top_k": engine.cfg.num_experts_per_tok,
        "expert_top_k_layers": list(selected_top_k_layers),
        "expert_top_k_by_layer": list(engine.cfg.expert_top_k_by_layer),
        "cases": len(prompts),
        "matching_tokens": exact_tokens if not mismatches else None,
        "mismatches": mismatches,
        "quality_failures": quality_failures,
        "baseline_tps": measured_tokens / base_decode_s,
        "reranked_tps": measured_tokens / fast_decode_s,
        "speedup": base_decode_s / fast_decode_s,
        "peak_gb": peak / 1e9,
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if mismatches or quality_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
