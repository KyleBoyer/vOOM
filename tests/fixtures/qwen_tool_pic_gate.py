#!/usr/bin/env python3
"""Real-model quality/latency gate for edited-catalog tool KV capsules.

Example:
  ~/.hf-pull/bin/python tests/fixtures/qwen_tool_pic_gate.py \
    --model ~/models/Qwen2.5-7B-Instruct-mlx-mxfp4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig  # noqa: E402
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.server import _prepare_chat_prompt  # noqa: E402
from runtime.toolcalls import parse_tool_calls  # noqa: E402


def _runtime(model: Path, *, pic: bool, cache_mb: int, repair: int,
             chunk: int, shared_pages: bool = False) -> RuntimeConfig:
    cfg = ModelConfig.from_dir(model)
    olmoe = cfg.model_type == "olmoe"
    return RuntimeConfig(
        max_weight_cache_mb=cache_mb,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        prompt_kv_dir="",
        prefill_chunk_size=chunk,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        quant_attention=not olmoe,
        quant_router=not olmoe,
        quant_lm_head=not olmoe,
        quantize_tied_lm_head=cfg.tie_word_embeddings,
        resident_fast_decode=not olmoe,
        resident_fast_prefill_limit=512,
        resident_moe_decode=olmoe,
        resident_attention_mode="mxfp8" if olmoe else "",
        resident_attention_bits=8,
        resident_attention_group_size=32,
        rerank_lm_head=olmoe and not cfg.tie_word_embeddings,
        rerank_lm_head_mode="affine",
        rerank_lm_head_bits=2,
        rerank_lm_head_group_size=64,
        stepped_kv_threshold=512,
        fused_swiglu=True,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=chunk,
        hot_prompt_kv_slots=1,
        hot_prompt_kv_min_tokens=0,
        tool_pic=pic,
        tool_pic_shared_pages=shared_pages,
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
            "description": f"Perform {name} on a repository and return status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "note": {"type": "string"},
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
            f"Call {target} with path {path}. Return only the tool call."),
    }]
    return _prepare_chat_prompt(
        engine, model, messages, "none", tools, tools, "fast", max_tokens)[0]


def _parsed(text: str, model_type: str):
    _content, calls = parse_tool_calls(text, model_type)
    if calls:
        function = calls[0]["function"]
    else:
        # The 1.5B checkpoint sometimes emits the requested bare JSON object
        # instead of the learned XML wrapper. It is still an unambiguous call
        # for this gate and is accepted only when the whole output is valid JSON.
        candidate = text.strip()
        if candidate.startswith("<tool_call>") and candidate.endswith("</tool_call>"):
            candidate = candidate[len("<tool_call>"):-len("</tool_call>")].strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            return None, {}
        if not isinstance(value, dict):
            return None, {}
        function = {
            "name": value.get("name"),
            "arguments": json.dumps(value.get("arguments", {})),
        }
    try:
        arguments = json.loads(function["arguments"])
    except (TypeError, json.JSONDecodeError):
        arguments = {}
    return function["name"], arguments


def _run_profile(model: Path, args, cases, *, pic: bool,
                 shared_pages: bool = False):
    engine = StreamingEngine(
        model, _runtime(
            model, pic=pic, cache_mb=args.cache_mb,
            repair=args.repair_tokens, chunk=args.chunk_size,
            shared_pages=shared_pages))
    rows = []
    try:
        for target, path in cases:
            # A clean exact source generation for each case isolates one catalog
            # edit while preserving resident weights and compiled kernels.
            engine.release_request_state()
            source = _prompt(
                engine, model, args.tools, edited=False,
                target=target, path=path, max_tokens=args.max_tokens)
            source_result = engine.generate(source, max_tokens=1, stop=[])
            edited = _prompt(
                engine, model, args.tools, edited=True,
                target=target, path=path, max_tokens=args.max_tokens)
            result = engine.generate(
                edited, max_tokens=args.max_tokens, stop=[])
            name, arguments = _parsed(result["text"], engine.cfg.model_type)
            rows.append({
                "target": target,
                "path": path,
                "name": name,
                "argument_path": arguments.get("path"),
                "tokens": result["tokens"],
                "prefill_s": result["prefill_s"],
                "decode_s": result["decode_s"],
                "total_s": result["total_s"],
                "source_prefill_s": source_result["prefill_s"],
                "source_total_s": source_result["total_s"],
                "source": result["path_stats"]["prompt_cache_source"],
                "pic": result["path_stats"]["tool_pic"],
                "selected": result["path_stats"]["tool_pic_selected_tokens"],
                "reused": result["path_stats"]["tool_pic_reused_tokens"],
                "peak_bytes": result["true_peak_metal_bytes"],
                "pool_live_bytes": (
                    engine._position_free_pool.live_nbytes()
                    if engine._position_free_pool is not None else 0),
                "pool_allocated_bytes": (
                    engine._position_free_pool.allocated_nbytes()
                    if engine._position_free_pool is not None else 0),
            })
    finally:
        engine.close()
        mx.clear_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-7B-Instruct-mlx-mxfp4")
    parser.add_argument("--tools", type=int, default=60)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repair-tokens", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=7000)
    parser.add_argument(
        "--shared-pages", action="store_true",
        help="exercise the experimental unrotated shared-page PIC cache")
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").exists():
        parser.error(f"model is not a local checkpoint: {model}")
    if args.tools < 8 or args.cases <= 0 or args.max_tokens <= 1:
        parser.error("tools>=8, cases>0, and max-tokens>1 are required")

    probes = (5, 19, 31, 47, 58)
    indices = [min(args.tools - 1, value) for value in probes[:args.cases]]
    while len(indices) < args.cases:
        indices.append((len(indices) * 7) % args.tools)
    cases = [(f"tool_{index:03d}", f"src/case_{index}.py")
             for index in indices]

    candidate = _run_profile(
        model, args, cases, pic=True, shared_pages=args.shared_pages)
    private_pic = (
        _run_profile(model, args, cases, pic=True, shared_pages=False)
        if args.shared_pages else candidate)
    control = _run_profile(model, args, cases, pic=False)
    rows = []
    for pic, private, exact in zip(candidate, private_pic, control):
        speedup = exact["prefill_s"] / pic["prefill_s"]
        shared_speedup = private["prefill_s"] / pic["prefill_s"]
        shared_total_speedup = private["total_s"] / pic["total_s"]
        cold_source_ratio = pic["source_total_s"] / private["source_total_s"]
        rows.append({
            "target": pic["target"],
            "name": pic["name"],
            "argument_path": pic["argument_path"],
            "semantic_ok": (
                pic["name"] == pic["target"]
                and pic["argument_path"] == pic["path"]),
            "private_pic_semantic_ok": (
                private["name"] == private["target"]
                and private["argument_path"] == private["path"]),
            "exact_control_ok": (
                exact["name"] == exact["target"]
                and exact["argument_path"] == exact["path"]),
            "same_ids": pic["tokens"] == exact["tokens"],
            "same_private_pic_ids": pic["tokens"] == private["tokens"],
            "pic_exercised": pic["pic"] == 1,
            "selected_tokens": pic["selected"],
            "reused_tokens": pic["reused"],
            "pic_prefill_s": round(pic["prefill_s"], 6),
            "private_pic_prefill_s": round(private["prefill_s"], 6),
            "exact_prefix_prefill_s": round(exact["prefill_s"], 6),
            "pic_decode_s": round(pic["decode_s"], 6),
            "private_pic_decode_s": round(private["decode_s"], 6),
            "pic_total_s": round(pic["total_s"], 6),
            "private_pic_total_s": round(private["total_s"], 6),
            "shared_cold_source_s": round(pic["source_total_s"], 6),
            "private_cold_source_s": round(private["source_total_s"], 6),
            "speedup": round(speedup, 3),
            "shared_page_speedup": round(shared_speedup, 3),
            "shared_page_total_speedup": round(shared_total_speedup, 3),
            "shared_cold_source_ratio": round(cold_source_ratio, 3),
            "shared_peak_gb": round(pic["peak_bytes"] / 1e9, 3),
            "private_peak_gb": round(private["peak_bytes"] / 1e9, 3),
        })
    base_pic_quality_passed = all(
        row["semantic_ok"] and row["exact_control_ok"]
        and row["same_ids"] and row["pic_exercised"]
        for row in rows)
    base_pic_quality_passed = (
        base_pic_quality_passed
        and statistics.median(row["speedup"] for row in rows) >= 1.01)
    shared_page_passed = all(
        row["semantic_ok"] and row["private_pic_semantic_ok"]
        and row["exact_control_ok"] and row["same_private_pic_ids"]
        and row["pic_exercised"]
        for row in rows)
    if args.shared_pages:
        shared_page_passed = (
            shared_page_passed
            and statistics.median(
                row["shared_page_speedup"] for row in rows) >= 1.01
            and statistics.median(
                row["shared_page_total_speedup"] for row in rows) >= 1.01
        )
    passed = shared_page_passed if args.shared_pages else base_pic_quality_passed
    report = {
        "gate": "qwen-tool-pic-v1",
        "passed": passed,
        "model": str(model),
        "tools": args.tools,
        "repair_tokens": args.repair_tokens,
        "shared_pages": args.shared_pages,
        "base_pic_quality_passed": base_pic_quality_passed,
        "shared_page_passed": shared_page_passed if args.shared_pages else None,
        "cases": rows,
        "same_ids": sum(row["same_ids"] for row in rows),
        "same_private_pic_ids": sum(
            row["same_private_pic_ids"] for row in rows),
        "median_speedup": round(statistics.median(
            row["speedup"] for row in rows), 3),
        "median_shared_page_speedup": round(statistics.median(
            row["shared_page_speedup"] for row in rows), 3),
        "median_shared_page_total_speedup": round(statistics.median(
            row["shared_page_total_speedup"] for row in rows), 3),
        "median_shared_cold_source_ratio": round(statistics.median(
            row["shared_cold_source_ratio"] for row in rows), 3),
        "peak_metal_gb": round(max(
            row["peak_bytes"] for row in candidate) / 1e9, 3),
        "pool_live_gb": round(max(
            row["pool_live_bytes"] for row in candidate) / 1e9, 3),
        "pool_allocated_gb": round(max(
            row["pool_allocated_bytes"] for row in candidate) / 1e9, 3),
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
