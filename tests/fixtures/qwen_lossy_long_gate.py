#!/usr/bin/env python3
"""Real Qwen2.5 lossy-long 32K/48K/64K retrieval and tool-loop gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.server import PreparedPrompt  # noqa: E402
from runtime.structured import GrammarConstraint  # noqa: E402
from runtime.toolcalls import parse_tool_calls, tools_preamble  # noqa: E402


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_secret",
            "description": "Submit the exact secret code retrieved from the archive.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_receipt",
            "description": "Confirm the receipt returned by submit_secret.",
            "parameters": {
                "type": "object",
                "properties": {"receipt": {"type": "string"}},
                "required": ["receipt"],
                "additionalProperties": False,
            },
        },
    },
]


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=7000,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=2,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        resident_fast_decode=True,
        resident_fast_prefill_limit=512,
        fused_swiglu=True,
        stepped_kv_threshold=512,
        prompt_kv_dir="",
        prefill_chunk_size=4096,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=4096,
        hot_prompt_kv_slots=1,
        hot_prompt_kv_min_tokens=0,
        qwen_yarn_factor=2.0,
        governor=True,
    )


def _encode(engine, text: str) -> list[int]:
    return list(engine.tokenizer.encode(text).ids)


def _filler(engine, minimum_tokens: int) -> list[int]:
    """Produce varied benign archive text once, with no competing code fields."""
    lines: list[str] = []
    token_ids: list[int] = []
    start = 0
    while len(token_ids) < minimum_tokens:
        lines.clear()
        for index in range(start, start + 1000):
            lines.append(
                f"Archive entry {index:06d}: cedar lantern, quiet harbor, "
                "silver compass, and ordinary inventory notes.\n"
            )
        token_ids.extend(_encode(engine, "".join(lines)))
        start += 1000
    return token_ids


def _prompt_ids(engine, filler: list[int], target: int,
                lookup_id: str, code: str) -> list[int]:
    prefix = _encode(engine, (
        "<|im_start|>system\n"
        "You are a precise archival retrieval agent. Read the target record, "
        "ignore ordinary inventory prose, and use the requested tool.\n"
        + tools_preamble(TOOLS)
        + "<|im_end|>\n<|im_start|>user\n"
    ))
    needle = _encode(engine, (
        f"\nTARGET RECORD: lookup_id={lookup_id}; secret_code={code}. "
        f"For lookup_id {lookup_id}, the exact secret_code is {code}.\n"
    ))
    suffix = _encode(engine, (
        f"\nQuestion: Find lookup_id {lookup_id}. Call submit_secret exactly once "
        "and put its secret_code in the code argument.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ))
    filler_count = target - len(prefix) - len(needle) - len(suffix)
    if filler_count <= 0 or filler_count > len(filler):
        raise ValueError(f"cannot construct {target}-token prompt")
    before = min(filler_count // 8, 8192)
    tokens = (
        prefix + filler[:before] + needle
        + filler[before:filler_count] + suffix
    )
    if len(tokens) != target:
        raise AssertionError((len(tokens), target))
    return tokens


def _call(result: dict) -> tuple[str | None, dict]:
    _content, calls = parse_tool_calls(result["text"], "qwen2")
    if len(calls) != 1:
        return None, {}
    function = calls[0]["function"]
    try:
        arguments = json.loads(function["arguments"])
    except (TypeError, ValueError):
        arguments = {}
    return function["name"], arguments


def _tool_suffix(engine, receipt: str) -> list[int]:
    return _encode(engine, (
        "<|im_end|>\n<|im_start|>user\n<tool_response>"
        f"submit_secret accepted the code and returned receipt {receipt}."
        "</tool_response>\n"
        "Call confirm_receipt exactly once with that receipt."
        "<|im_end|>\n<|im_start|>assistant\n"
    ))


def _final_suffix(engine) -> list[int]:
    return _encode(engine, (
        "<|im_end|>\n<|im_start|>user\n<tool_response>"
        "The receipt is confirmed.</tool_response>\n"
        "Reply with only DONE.<|im_end|>\n<|im_start|>assistant\n"
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4")
    parser.add_argument(
        "--targets", type=int, nargs="+", default=[32_000, 48_000, 64_000])
    parser.add_argument("--max-tool-tokens", type=int, default=48)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"not a complete local checkpoint: {model}")
    if sorted(args.targets) != args.targets or min(args.targets) < 1024:
        parser.error("targets must be increasing and at least 1024")

    engine = StreamingEngine(model, _config())
    rows = []
    failures: list[str] = []
    try:
        if engine.rope_profile != "experimental-qwen-yarn-2x":
            failures.append(f"unexpected rope profile: {engine.rope_profile}")
        if max(args.targets) + args.max_tool_tokens + 512 > \
                engine.effective_max_position_embeddings:
            parser.error("largest target leaves insufficient follow-up headroom")
        filler = _filler(engine, max(args.targets))

        for index, target in enumerate(args.targets):
            lookup_id = f"case-{target}"
            code = ("AMBER32" if target < 40_000 else
                    "COBALT48" if target < 56_000 else "VIOLET64")
            receipt = f"R{target}OK"
            prompt_ids = _prompt_ids(
                engine, filler, target, lookup_id, code)
            print(f"[long-gate] starting {target} tokens", file=sys.stderr, flush=True)
            started = time.perf_counter()
            first = engine.generate(
                PreparedPrompt(f"long-{target}", prompt_ids),
                max_tokens=args.max_tool_tokens,
                stop=[],
                constraint=GrammarConstraint.tools(
                    engine, TOOLS, required=True,
                    specific_name="submit_secret", allow_parallel=False),
            )
            first_wall = time.perf_counter() - started
            first_name, first_args = _call(first)

            followup_ids = (
                prompt_ids + first["tokens"] + _tool_suffix(engine, receipt))
            second = engine.generate(
                PreparedPrompt(f"loop-{target}", followup_ids),
                max_tokens=args.max_tool_tokens,
                stop=[],
                constraint=GrammarConstraint.tools(
                    engine, TOOLS, required=True,
                    specific_name="confirm_receipt", allow_parallel=False),
            )
            second_name, second_args = _call(second)

            row = {
                "target_prompt_tokens": target,
                "first_prompt_tokens": first["prompt_tokens"],
                "followup_prompt_tokens": second["prompt_tokens"],
                "expected_code": code,
                "first_text": first["text"],
                "first_call": {"name": first_name, "arguments": first_args},
                "first_total_s": first["total_s"],
                "first_wall_s": first_wall,
                "first_prefill_s": first["prefill_s"],
                "first_decode_s": first["decode_s"],
                "first_peak_metal_bytes": first["true_peak_metal_bytes"],
                "first_kv_bytes": first["kv_bytes"],
                "first_termination": first["termination_reason"],
                "followup_text": second["text"],
                "followup_call": {"name": second_name, "arguments": second_args},
                "followup_total_s": second["total_s"],
                "followup_cache_source": second["path_stats"].get(
                    "prompt_cache_source"),
                "followup_cache_prefix_tokens": second["path_stats"].get(
                    "prompt_cache_prefix_tokens"),
                "followup_peak_metal_bytes": second["true_peak_metal_bytes"],
            }

            if first_name != "submit_secret" or first_args.get("code") != code:
                failures.append(
                    f"{target}: retrieval/tool call was {first_name} {first_args}")
            if first["termination_reason"] != "grammar":
                failures.append(
                    f"{target}: first call did not terminate on grammar")
            if (second_name != "confirm_receipt"
                    or second_args.get("receipt") != receipt):
                failures.append(
                    f"{target}: follow-up tool call was {second_name} {second_args}")
            if second["path_stats"].get("prompt_cache_source") != "memory":
                failures.append(f"{target}: tool-loop follow-up missed hot KV")
            # The retained endpoint contains the prompt plus every generated
            # token except the final un-fed token. A strict extension should
            # reuse that entire endpoint, not fall back to a 4K boundary.
            if int(second["path_stats"].get(
                    "prompt_cache_prefix_tokens", 0)) <= target:
                failures.append(f"{target}: tool-loop did not reuse the full endpoint")

            if index == len(args.targets) - 1:
                final_ids = (
                    followup_ids + second["tokens"] + _final_suffix(engine))
                final = engine.generate(
                    PreparedPrompt("long-final", final_ids),
                    max_tokens=16, stop=[])
                row["final_text"] = final["text"]
                row["final_termination"] = final["termination_reason"]
                row["final_total_s"] = final["total_s"]
                row["final_cache_prefix_tokens"] = final["path_stats"].get(
                    "prompt_cache_prefix_tokens")
                if final["text"].strip().rstrip(".!?").upper() != "DONE":
                    failures.append(
                        f"{target}: post-tool final response was {final['text']!r}")
                if final["termination_reason"] != "eos":
                    failures.append(
                        f"{target}: post-tool response did not terminate at EOS")

            rows.append(row)
            print(
                f"[long-gate] {target}: code={first_args.get('code')!r} "
                f"prefill={first['prefill_s']:.3f}s "
                f"peak={first['true_peak_metal_bytes'] / 1e9:.3f}GB",
                file=sys.stderr, flush=True,
            )
            # One long conversation at a time: retain it for its immediate tool
            # loop, then release before allocating the next larger context.
            engine._hot_prompt_slots.clear()
            engine.release_request_state()
            mx.clear_cache()
    finally:
        engine.close()
        mx.clear_cache()

    report = {
        "gate": "qwen-lossy-long-v1",
        "passed": not failures,
        "failures": failures,
        "model": str(model),
        "rope_profile": engine.rope_profile,
        "effective_max_positions": engine.effective_max_position_embeddings,
        "rows": rows,
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="", flush=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
