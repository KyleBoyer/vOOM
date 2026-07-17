#!/usr/bin/env python3
"""Real Qwen3-VL edited-tool-catalog PIC quality and latency gate.

Example:
  ~/.hf-pull/bin/python tests/fixtures/qwen3vl_tool_pic_gate.py \
    --model ~/hf_cache/modelscope/models/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.qwen3vl import generate_vl  # noqa: E402
from runtime.server import _prepare_chat_prompt  # noqa: E402
from runtime.toolcalls import VISION_SPAN, parse_tool_calls  # noqa: E402


def _runtime(*, pic: bool, repair: int) -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=7000,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        prompt_kv_dir="",
        prefill_chunk_size=512,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        quant_attention=False,
        quant_lm_head=False,
        quantize_tied_lm_head=False,
        resident_fast_decode=True,
        resident_fast_prefill_limit=512,
        fused_swiglu=True,
        vision_max_patches=256,
        tool_pic=pic,
        tool_pic_repair_tokens=repair,
        tool_pic_min_savings=128,
        governor=True,
    )


def _catalog(count: int, *, edited: bool) -> list[dict]:
    names = [f"tool_{index:03d}" for index in range(count)]
    if edited:
        names.append(f"tool_{count // 2:03d}x_new")
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": f"Inspect an image and repository with {name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    } for name in names]


def _prompt(engine, model: Path, count: int, *, edited: bool,
            target: str, path: str, max_tokens: int):
    tools = _catalog(count, edited=edited)
    messages = [{
        "role": "user",
        "content": (
            VISION_SPAN
            + f"\nCall {target} with path {path}. Return only the tool call."),
    }]
    return _prepare_chat_prompt(
        engine, model, messages, "none", tools, tools, "fast", max_tokens)[0]


def _parsed(text: str, model_type: str):
    _content, calls = parse_tool_calls(text, model_type)
    if not calls:
        return None, {}
    function = calls[0]["function"]
    try:
        arguments = json.loads(function["arguments"])
    except (TypeError, json.JSONDecodeError):
        arguments = {}
    return function["name"], arguments


def _run(model: Path, args, cases, *, pic: bool):
    engine = StreamingEngine(model, _runtime(
        pic=pic, repair=args.repair_tokens))
    image = Image.new("RGB", (224, 224), (30, 180, 70))
    rows = []
    try:
        for target, path in cases:
            engine._vision_prompt_cache = None
            source = _prompt(
                engine, model, args.tools, edited=False,
                target=target, path=path, max_tokens=args.max_tokens)
            generate_vl(engine, source, [image], max_tokens=1, stop=[])
            edited = _prompt(
                engine, model, args.tools, edited=True,
                target=target, path=path, max_tokens=args.max_tokens)
            result = generate_vl(
                engine, edited, [image], max_tokens=args.max_tokens, stop=[])
            name, arguments = _parsed(result["text"], engine.cfg.model_type)
            path_stats = result["path_stats"]
            rows.append({
                "target": target,
                "path": path,
                "name": name,
                "argument_path": arguments.get("path"),
                "tokens": result["tokens"],
                "prefill_s": result["prefill_s"],
                "pic": result["vision_tool_pic"],
                "selected": result["vision_tool_pic_selected_tokens"],
                "reused": result["vision_tool_pic_reused_tokens"],
                "path_pic": path_stats["tool_pic"],
                "path_selected": path_stats["tool_pic_selected_tokens"],
                "path_reused": path_stats["tool_pic_reused_tokens"],
                "path_source": path_stats["prompt_cache_source"],
                "path_prefix": path_stats["prompt_cache_prefix_tokens"],
                "path_approximate": path_stats["prompt_state_approximate"],
                "peak_bytes": result["true_peak_metal_bytes"],
            })
    finally:
        engine.close()
        mx.clear_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=(Path.home()
                 / "hf_cache/modelscope/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--tools", type=int, default=24)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repair-tokens", type=int, default=4)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").exists():
        parser.error(f"model is not a local checkpoint: {model}")
    if args.tools < 8 or args.cases <= 0 or args.max_tokens <= 1:
        parser.error("tools>=8, cases>0, and max-tokens>1 are required")

    probes = (5, 13, 21, 7, 17)
    indices = [min(args.tools - 1, value)
               for value in probes[:args.cases]]
    cases = [(f"tool_{index:03d}", f"src/image_{index}.png")
             for index in indices]
    candidate = _run(model, args, cases, pic=True)
    control = _run(model, args, cases, pic=False)
    rows = []
    for pic, exact in zip(candidate, control):
        speedup = exact["prefill_s"] / pic["prefill_s"]
        rows.append({
            "target": pic["target"],
            "semantic_ok": (
                pic["name"] == pic["target"]
                and pic["argument_path"] == pic["path"]),
            "exact_control_ok": (
                exact["name"] == exact["target"]
                and exact["argument_path"] == exact["path"]),
            "same_ids": pic["tokens"] == exact["tokens"],
            "pic_exercised": pic["pic"],
            "telemetry_ok": (
                pic["path_pic"] == 1
                and pic["path_selected"] == pic["selected"]
                and pic["path_reused"] == pic["reused"]
                and pic["path_source"] == "vision_tool_pic"
                and pic["path_prefix"] > 0
                and pic["path_approximate"] == 1),
            "selected_tokens": pic["selected"],
            "reused_tokens": pic["reused"],
            "pic_prefill_s": round(pic["prefill_s"], 6),
            "exact_prefill_s": round(exact["prefill_s"], 6),
            "speedup": round(speedup, 3),
        })
    passed = (
        all(row["semantic_ok"] and row["exact_control_ok"]
            and row["same_ids"] and row["pic_exercised"]
            and row["telemetry_ok"] for row in rows)
        and statistics.median(row["speedup"] for row in rows) >= 1.01)
    report = {
        "gate": "qwen3vl-tool-pic-v1",
        "passed": passed,
        "model": str(model),
        "tools": args.tools,
        "repair_tokens": args.repair_tokens,
        "cases": rows,
        "median_speedup": round(statistics.median(
            row["speedup"] for row in rows), 3),
        "peak_metal_gb": round(max(
            row["peak_bytes"] for row in candidate) / 1e9, 3),
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
