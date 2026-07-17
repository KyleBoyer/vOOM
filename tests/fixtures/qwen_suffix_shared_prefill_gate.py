#!/usr/bin/env python3
"""Six-case real-Qwen gate for shared-prefill linear SuffixDecoding.

The candidate and control both use the production chunked hot-KV/tool-PIC
prefill. Only the candidate enables the engine-local suffix verifier. Resident,
PositionFree, paged, and stepped decode are deliberately disabled because the
initial suffix implementation fails closed for those unproved cache contracts.

This script loads an MLX model. Invoke it only while holding the repository-wide
``/tmp/voom-mlx-benchmark.lock`` atomic directory lock.
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

from runtime.engine import StreamingEngine  # noqa: E402
from tests.fixtures.qwen_tool_pic_gate import (  # noqa: E402
    _parsed, _prompt, _runtime)


def _profile_runtime(model: Path, args, *, suffix: bool):
    rc = _runtime(
        model,
        pic=True,
        cache_mb=args.cache_mb,
        repair=args.repair_tokens,
        chunk=args.chunk_size,
        shared_pages=False,
    )
    # The shared-prefill suffix contract is intentionally narrower than the
    # general tool-PIC fixture's automatic fast profile.
    rc.resident_fast_decode = False
    rc.resident_fast_prefill_limit = 0
    rc.resident_moe_decode = False
    rc.stepped_kv_threshold = 0
    rc.max_kv_mb = 0
    rc.suffix_decoding = suffix
    rc.suffix_decoding_k = args.k
    rc.suffix_decoding_factor = args.factor
    rc.suffix_decoding_max_depth = args.max_depth
    rc.suffix_decoding_min_probability = args.min_probability
    return rc


def _cases(tool_count: int, count: int):
    indices = [((index * 4) + 1) % tool_count for index in range(count)]
    training = [
        (f"tool_{index:03d}", f"src/training_case_{index}.py")
        for index in indices
    ]
    held_out = [
        (f"tool_{index:03d}", f"src/evaluation_case_{index}.py")
        for index in indices
    ]
    return training, held_out


def _run_profile(model: Path, args, training, held_out, *, suffix: bool):
    engine = StreamingEngine(
        model, _profile_runtime(model, args, suffix=suffix))
    training_rows = []
    rows = []
    try:
        # Warm the exact production path before measured training/evaluation.
        warm = _prompt(
            engine, model, args.tools, edited=False,
            target=training[0][0], path="src/warmup.py",
            max_tokens=args.max_tokens)
        engine.release_request_state()
        engine.generate(warm, max_tokens=2, stop=[])

        # These six completed target outputs are the candidate's bounded global
        # suffix corpus. release_request_state() intentionally does not clear it.
        for target, path in training:
            engine.release_request_state()
            prompt = _prompt(
                engine, model, args.tools, edited=False,
                target=target, path=path, max_tokens=args.max_tokens)
            result = engine.generate(
                prompt, max_tokens=args.max_tokens, stop=[])
            training_rows.append({
                "tokens": result["tokens"],
                "total_s": result["total_s"],
                "suffix_used": result["path_stats"]["suffix_decoding_used"],
            })

        for target, path in held_out:
            # A one-token exact source leaves the retained KV exactly at the
            # unedited prompt endpoint, matching the production tool-PIC gate.
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
            stats = result["path_stats"]
            name, arguments = _parsed(result["text"], engine.cfg.model_type)
            rows.append({
                "target": target,
                "path": path,
                "name": name,
                "argument_path": arguments.get("path"),
                "tokens": result["tokens"],
                "prompt_tokens": result["prompt_tokens"],
                "kv_positions": result["kv_positions"],
                "prefill_s": result["prefill_s"],
                "decode_s": result["decode_s"],
                "first_token_s": result["first_token_s"],
                "total_s": result["total_s"],
                "source_total_s": source_result["total_s"],
                "peak_bytes": result["true_peak_metal_bytes"],
                "prompt_source": stats["prompt_cache_source"],
                "tool_pic": stats["tool_pic"],
                "prompt_state_approximate": stats["prompt_state_approximate"],
                "suffix_prompt_approximate": (
                    stats["suffix_decoding_prompt_approximate"]),
                "suffix_used": stats["suffix_decoding_used"],
                "suffix_fallback": stats["suffix_decoding_fallback_reason"],
                "proposed": stats["suffix_decoding_proposed"],
                "accepted": stats["suffix_decoding_accepted"],
                "target_sweeps": stats["suffix_decoding_target_sweeps"],
                "suffix_cpu_s": stats["suffix_decoding_cpu_s"],
                "cache_update_cpu_s": (
                    stats["suffix_decoding_cache_update_cpu_s"]),
                "cache_requests": stats.get(
                    "suffix_decoding_cache_requests", 0),
                "cache_tokens": stats.get(
                    "suffix_decoding_cache_tokens", 0),
                "cache_nodes": stats.get(
                    "suffix_decoding_cache_nodes", 0),
                "cache_bytes": stats.get(
                    "suffix_decoding_cache_bytes", 0),
            })
    finally:
        engine.close()
        mx.clear_cache()
    return training_rows, rows


def _ratio(numerator: float, denominator: float):
    return round(numerator / denominator, 4) if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4")
    parser.add_argument("--tools", type=int, default=24)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repair-tokens", type=int, default=4)
    parser.add_argument("--cache-mb", type=int, default=1_800)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--factor", type=float, default=4.0)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--min-probability", type=float, default=0.1)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").exists():
        parser.error(f"model is not a local checkpoint: {model}")
    if (args.tools < 8 or args.cases != 6 or args.max_tokens <= 1
            or args.chunk_size <= 0 or args.k <= 0 or args.factor < 0
            or args.max_depth <= 1
            or not 0 <= args.min_probability <= 1):
        parser.error(
            "tools>=8, exactly six cases, max-tokens>1, chunk/k>0, and valid "
            "suffix bounds are required")

    training, held_out = _cases(args.tools, args.cases)
    # Candidate first is conservative for timing: any process-global compile or
    # filesystem warmth benefits the later control rather than the candidate.
    candidate_training, candidate = _run_profile(
        model, args, training, held_out, suffix=True)
    control_training, control = _run_profile(
        model, args, training, held_out, suffix=False)

    rows = []
    for suffix, plain in zip(candidate, control):
        same_ids = suffix["tokens"] == plain["tokens"]
        semantic_ok = (
            suffix["name"] == suffix["target"]
            and suffix["argument_path"] == suffix["path"]
        )
        rows.append({
            "target": suffix["target"],
            "semantic_ok": semantic_ok,
            "same_ids": same_ids,
            "output_tokens": len(suffix["tokens"]),
            "pic_exercised": suffix["tool_pic"] == 1,
            "pic_approximate_preserved": (
                suffix["prompt_state_approximate"] == 1
                and suffix["suffix_prompt_approximate"] == 1),
            "suffix_used": suffix["suffix_used"] == 1,
            "suffix_fallback": suffix["suffix_fallback"],
            "kv_endpoint_ok": suffix["kv_positions"] == (
                suffix["prompt_tokens"] + len(suffix["tokens"]) - 1),
            "proposed": suffix["proposed"],
            "accepted": suffix["accepted"],
            "target_sweeps": suffix["target_sweeps"],
            "suffix_cpu_s": round(suffix["suffix_cpu_s"], 6),
            "cache_update_cpu_s": round(
                suffix["cache_update_cpu_s"], 6),
            "suffix_prefill_s": round(suffix["prefill_s"], 6),
            "control_prefill_s": round(plain["prefill_s"], 6),
            "suffix_decode_s": round(suffix["decode_s"], 6),
            "control_decode_s": round(plain["decode_s"], 6),
            "suffix_total_s": round(suffix["total_s"], 6),
            "control_total_s": round(plain["total_s"], 6),
            "suffix_workflow_s": round(
                suffix["source_total_s"] + suffix["total_s"], 6),
            "control_workflow_s": round(
                plain["source_total_s"] + plain["total_s"], 6),
            "suffix_peak_metal_gb": round(
                suffix["peak_bytes"] / 1e9, 3),
            "control_peak_metal_gb": round(
                plain["peak_bytes"] / 1e9, 3),
        })

    quality_passed = all(
        row["same_ids"]
        and row["semantic_ok"]
        and row["pic_exercised"]
        and row["pic_approximate_preserved"]
        and row["suffix_used"]
        and row["kv_endpoint_ok"]
        for row in rows
    )
    proposed = sum(row["proposed"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    candidate_decode = sum(row["suffix_decode_s"] for row in rows)
    control_decode = sum(row["control_decode_s"] for row in rows)
    candidate_total = sum(row["suffix_total_s"] for row in rows)
    control_total = sum(row["control_total_s"] for row in rows)
    candidate_workflow = sum(row["suffix_workflow_s"] for row in rows)
    control_workflow = sum(row["control_workflow_s"] for row in rows)
    decode_speedup = _ratio(control_decode, candidate_decode)
    total_speedup = _ratio(control_total, candidate_total)
    workflow_speedup = _ratio(control_workflow, candidate_workflow)
    performance_passed = bool(
        decode_speedup is not None and decode_speedup > 1.0
        and total_speedup is not None and total_speedup > 1.0)
    report = {
        "gate": "qwen-suffix-shared-prefill-v1",
        "passed": quality_passed and performance_passed,
        "quality_passed": quality_passed,
        "performance_passed": performance_passed,
        "model": str(model),
        "profile_order": ["suffix", "control"],
        "configuration": {
            "tools": args.tools,
            "training_cases": len(training),
            "held_out_cases": len(held_out),
            "max_tokens": args.max_tokens,
            "prefill_chunk_size": args.chunk_size,
            "tool_pic": True,
            "tool_pic_shared_pages": False,
            "resident_fast_decode": False,
            "stepped_kv_threshold": 0,
            "k": args.k,
            "factor": args.factor,
            "max_depth": args.max_depth,
            "min_probability": args.min_probability,
        },
        "training_output_tokens": sum(
            len(row["tokens"]) for row in candidate_training),
        "control_training_output_tokens": sum(
            len(row["tokens"]) for row in control_training),
        "exact_id_cases": sum(row["same_ids"] for row in rows),
        "semantic_cases": sum(row["semantic_ok"] for row in rows),
        "proposed": proposed,
        "accepted": accepted,
        "proposal_acceptance": _ratio(accepted, proposed),
        "target_sweeps": sum(row["target_sweeps"] for row in rows),
        "ordinary_decode_steps": sum(row["output_tokens"] - 1 for row in rows),
        "target_sweep_speedup_upper_bound": _ratio(
            sum(row["output_tokens"] - 1 for row in rows),
            sum(row["target_sweeps"] for row in rows)),
        "decode_speedup": decode_speedup,
        "edited_total_speedup": total_speedup,
        "source_plus_edited_workflow_speedup": workflow_speedup,
        "suffix_cpu_s": round(sum(
            row["suffix_cpu_s"] for row in rows), 6),
        "cache_update_cpu_s": round(sum(
            row["cache_update_cpu_s"] for row in rows), 6),
        "suffix_peak_metal_gb": round(max(
            row["peak_bytes"] for row in candidate) / 1e9, 3),
        "control_peak_metal_gb": round(max(
            row["peak_bytes"] for row in control) / 1e9, 3),
        "cache_requests": candidate[-1]["cache_requests"],
        "cache_tokens": candidate[-1]["cache_tokens"],
        "cache_nodes": candidate[-1]["cache_nodes"],
        "cache_accounted_bytes": candidate[-1]["cache_bytes"],
        "cases": rows,
        "guardrails": {
            "new_model_downloaded": False,
            "suffix_default_enabled": False,
            "single_tenant_engine_required": True,
            "pic_approximation_relabelled_exact": False,
            "measured_speedups_are_theoretical_upper_bounds": False,
        },
        "median_case_decode_speedup": round(statistics.median(
            row["control_decode_s"] / row["suffix_decode_s"]
            for row in rows), 4),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
