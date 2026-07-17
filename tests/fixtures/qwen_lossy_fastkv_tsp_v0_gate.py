#!/usr/bin/env python3
"""Real-model gate for the fixture-only ``lossy-fastkv-tsp-qwen-v0`` proof.

Exit status:
  0  the calibrated profile passed every configured quality/performance gate
  1  the run completed, but one or more falsifiable gates failed
  2  checkpoint/runtime architecture was not admitted (fail closed)
  3  operational failure (reported as JSON when possible)

This script loads an MLX model. Callers must hold the repository-wide
``/tmp/voom-mlx-benchmark.lock`` for the entire process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FIXTURES))

from lossy_fastkv_tsp_qwen_v0 import (  # noqa: E402
    PROFILE_FAMILY,
    TSPProfile,
    UnsupportedFastKVTSP,
    common_prefix_length,
    generate_greedy,
    run_prefill,
    score_teacher_tokens,
    validate_admission,
)
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.structured import GrammarConstraint  # noqa: E402
from runtime.toolcalls import parse_tool_calls, tools_preamble  # noqa: E402


TOOLS = [{
    "type": "function",
    "function": {
        "name": "open_archive",
        "description": "Open exactly one archive path from the requested lookup record.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}]

# Preregistered in the research report. This is a release/promotion boundary,
# not a post-hoc tuning knob: diagnostics may report larger deltas, but they do
# not pass by widening this value after observation.
PREREGISTERED_MAX_NLL_DELTA = 0.02


def preregistered_nll_gate(delta: float) -> bool:
    return bool(np.isfinite(delta) and delta <= PREREGISTERED_MAX_NLL_DELTA)


def _runtime(cache_mb: int) -> RuntimeConfig:
    """No KV reuse, persistence, PIC, paging, checkpoints, or speculation."""
    return RuntimeConfig(
        max_weight_cache_mb=cache_mb,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        max_kv_mb=0,
        stepped_kv_threshold=0,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        resident_fast_decode=False,
        resident_fast_prefill_limit=0,
        resident_moe_decode=False,
        fused_swiglu=False,
        prompt_kv_dir="",
        hot_prompt_kv=False,
        tool_pic=False,
        tool_pic_shared_pages=False,
        hot_prompt_kv_persist_dir="",
        prefill_chunk_size=0,
        prefill_checkpoint_every=0,
        adaptive_chunk_size=False,
        mla_compressed_kv=False,
        governor=False,
        final_dead_token_elim=True,
    )


def _encode(engine, text: str) -> list[int]:
    return list(engine.tokenizer.encode(text).ids)


def _filler(engine, minimum_tokens: int) -> list[int]:
    result: list[int] = []
    start = 0
    while len(result) < minimum_tokens:
        lines = []
        for index in range(start, start + 512):
            lines.append(
                f"Inventory note {index:06d}: cedar lantern, quiet harbor, "
                "silver compass, ordinary paper ledger, no action required.\n")
        result.extend(_encode(engine, "".join(lines)))
        start += 512
    return result


def _prompt(
    engine, filler: list[int], target: int, *, depth: float,
) -> tuple[list[int], str, str]:
    lookup = f"needle-{target}-cobalt"
    path = f"/vault/cobalt-{target}/record.json"
    prefix = _encode(engine, (
        "<|im_start|>system\n"
        "You are a precise archive operator. Find the requested lookup record "
        "in the long ledger and call the available tool with its exact path.\n"
        + tools_preamble(TOOLS)
        + "<|im_end|>\n<|im_start|>user\n"
        "The ledger begins below. Ignore ordinary inventory notes.\n"
    ))
    needle = _encode(engine, (
        f"\nTARGET LOOKUP RECORD: lookup_id={lookup}; exact_path={path}. "
        f"For lookup_id {lookup}, the only valid path is {path}.\n"
    ))
    suffix = _encode(engine, (
        f"\nThe ledger ends here. Find lookup_id {lookup}. Call open_archive "
        "exactly once with the exact_path from that record. Return only the "
        "tool call.<|im_end|>\n<|im_start|>assistant\n"
    ))
    filler_count = target - len(prefix) - len(needle) - len(suffix)
    if filler_count <= 0 or filler_count > len(filler):
        raise ValueError(f"cannot construct an exact {target}-token prompt")
    before = int(round(filler_count * depth))
    before = max(0, min(before, filler_count))
    tokens = (
        prefix + filler[:before] + needle
        + filler[before:filler_count] + suffix)
    if len(tokens) != target:
        raise AssertionError((len(tokens), target))
    return tokens, lookup, path


def _constraint(engine):
    return GrammarConstraint.tools(
        engine, TOOLS, required=True, specific_name="open_archive",
        allow_parallel=False)


def _parsed_tool(text: str) -> tuple[str | None, dict]:
    _content, calls = parse_tool_calls(
        text, "qwen2", allowed_names={"open_archive"},
        argument_schemas={
            "open_archive": TOOLS[0]["function"]["parameters"],
        })
    if calls:
        function = calls[0]["function"]
        try:
            return function["name"], json.loads(function["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None, {}
    # Keep the 1.5B checkpoint's occasional bare-JSON behavior visible without
    # accepting a substring or malformed wrapper as a successful tool call.
    candidate = text.strip()
    if candidate.startswith("<tool_call>") and candidate.endswith("</tool_call>"):
        candidate = candidate[len("<tool_call>"):-len("</tool_call>")].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(value, dict) or value.get("name") != "open_archive":
        return None, {}
    arguments = value.get("arguments")
    return "open_archive", arguments if isinstance(arguments, dict) else {}


def _release() -> None:
    gc.collect()
    mx.clear_cache()
    mx.synchronize()


def _warm_weights(engine) -> dict:
    started = time.perf_counter()
    engine._eval_weight(engine._embed_weight())
    engine._eval_weight(engine._norm_w)
    for layer in range(engine.cfg.num_hidden_layers):
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer))
        for value in weights.values():
            engine._eval_weight(value)
    engine._eval_weight(engine._lm_head_weight())
    mx.synchronize()
    return {
        "wall_s": time.perf_counter() - started,
        "active_bytes": int(mx.get_active_memory()),
        "cache_evictions": int(engine.cache.stats.evictions),
    }


def _digest_positions(values: tuple[int, ...]) -> str:
    array = np.asarray(values, dtype=np.int32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _control_summary(prefill, generated, expected_path: str) -> dict:
    name, arguments = _parsed_tool(generated.text)
    semantic = name == "open_archive" and arguments.get("path") == expected_path
    result = prefill.metrics()
    result.update({
        "tokens": generated.tokens,
        "text": generated.text,
        "decode_wall_s": generated.wall_s,
        "total_wall_s": prefill.wall_s + generated.wall_s,
        "mean_nll": statistics.fmean(generated.nll),
        "constraint_completed": generated.completed_constraint,
        "tool_name": name,
        "tool_arguments": arguments,
        "semantic_ok": semantic,
    })
    return result


def _candidate_summary(
    engine, prefill, generated, teacher_nll: list[float],
    control: dict, expected_path: str, profile: TSPProfile, target: int,
    *, min_speedup: float,
) -> dict:
    name, arguments = _parsed_tool(generated.text)
    semantic = name == "open_archive" and arguments.get("path") == expected_path
    candidate_mean_nll = statistics.fmean(teacher_nll)
    nll_delta = candidate_mean_nll - control["mean_nll"]
    speedup = control["wall_s"] / prefill.wall_s
    expected_selected = max(
        profile.recent_window, int(np.ceil(profile.retention * target)))
    counts = [item["count"] for item in prefill.state.position_geometry()]
    expected_counts = (
        [target] * (profile.tsp_layer + 1)
        + [expected_selected] * (
            engine.cfg.num_hidden_layers - profile.tsp_layer - 1))
    metadata = prefill.state.metadata()
    geometry_ok = (
        counts == expected_counts
        and len(prefill.selected_positions) == expected_selected
        and prefill.selected_positions[-1] == target - 1)
    isolation_contract_ok = (
        metadata["approximate"] is True
        and metadata["exact"] is False
        and metadata["persistent"] is False
        and metadata["reusable"] is False)
    task_ok = (
        control["semantic_ok"] and control["constraint_completed"]
        and semantic and generated.completed_constraint)
    id_ok = generated.tokens == control["tokens"]
    nll_ok = preregistered_nll_gate(nll_delta)
    quality_ok = (
        task_ok and id_ok and nll_ok and geometry_ok
        and isolation_contract_ok)
    performance_ok = speedup >= min_speedup
    promotion_ok = quality_ok and performance_ok
    result = prefill.metrics()
    result.update({
        "tokens": generated.tokens,
        "text": generated.text,
        "decode_wall_s": generated.wall_s,
        "total_wall_s": prefill.wall_s + generated.wall_s,
        "teacher_mean_nll": candidate_mean_nll,
        "teacher_nll_delta": nll_delta,
        "same_ids": id_ok,
        "same_id_count": sum(
            left == right for left, right in zip(
                generated.tokens, control["tokens"])),
        "common_id_prefix": common_prefix_length(
            generated.tokens, control["tokens"]),
        "constraint_completed": generated.completed_constraint,
        "tool_name": name,
        "tool_arguments": arguments,
        "semantic_ok": semantic,
        "selected_positions_sha256": _digest_positions(
            prefill.selected_positions),
        "selected_positions_first": list(prefill.selected_positions[:16]),
        "selected_positions_last": list(prefill.selected_positions[-16:]),
        "expected_selected_tokens": expected_selected,
        "position_geometry_ok": geometry_ok,
        "isolation_contract_ok": isolation_contract_ok,
        "task_passed": task_ok,
        "id_passed": id_ok,
        "nll_gate_passed": nll_ok,
        "preregistered_max_nll_delta": PREREGISTERED_MAX_NLL_DELTA,
        "prefill_speedup": speedup,
        "cache_byte_reduction": (
            1.0 - prefill.state.cache_bytes / control["cache_bytes"]),
        "quality_passed": quality_ok,
        "performance_passed": performance_ok,
        "promotion_passed": promotion_ok,
        "passed": promotion_ok,
    })
    return result


def _run_candidate(
    engine, tokens: list[int], expected_path: str, profile: TSPProfile,
    control: dict, *, max_tokens: int, min_speedup: float,
) -> dict:
    print(
        f"[fastkv-tsp] candidate {profile.name} at {len(tokens)} tokens",
        file=sys.stderr, flush=True)
    prefill = run_prefill(engine, tokens, profile)
    generated = generate_greedy(
        engine, prefill, max_tokens=max_tokens,
        constraint=_constraint(engine))
    teacher_nll = score_teacher_tokens(engine, prefill, control["tokens"])
    summary = _candidate_summary(
        engine, prefill, generated, teacher_nll, control, expected_path,
        profile, len(tokens), min_speedup=min_speedup)
    del generated, teacher_nll, prefill
    _release()
    return summary


def _run_target(
    engine, tokens: list[int], expected_path: str,
    profiles: list[TSPProfile], *, max_tokens: int,
    min_speedup: float, diagnose: bool,
) -> tuple[dict, TSPProfile]:
    target = len(tokens)
    dense_profile = TSPProfile(
        tsp_layer=profiles[0].tsp_layer, retention=1.0,
        recent_window=profiles[0].recent_window,
        pool_width=profiles[0].pool_width,
        query_chunk=profiles[0].query_chunk)
    print(
        f"[fastkv-tsp] dense control at {target} tokens",
        file=sys.stderr, flush=True)
    dense_prefill = run_prefill(engine, tokens, dense_profile)
    dense_generated = generate_greedy(
        engine, dense_prefill, max_tokens=max_tokens,
        constraint=_constraint(engine))
    control = _control_summary(dense_prefill, dense_generated, expected_path)
    # Copy all scalar/list diagnostics, then release both full dense states before
    # allocating the candidate's ragged state.
    del dense_generated, dense_prefill
    _release()

    candidate_rows = []
    candidates_to_run = profiles if diagnose else profiles[:1]
    for index, profile in enumerate(candidates_to_run):
        row = _run_candidate(
            engine, tokens, expected_path, profile, control,
            max_tokens=max_tokens,
            min_speedup=min_speedup)
        candidate_rows.append(row)
        # The first quality-passing configuration ends diagnosis. A speed-only
        # failure cannot be repaired by retaining more tokens or pruning later.
        if row["quality_passed"]:
            break
        # If the dense control itself cannot retrieve the needle, candidate
        # retention diagnostics would not isolate TSP quality.
        if not control["semantic_ok"]:
            break
        if index + 1 < len(candidates_to_run):
            print(
                f"[fastkv-tsp] quality failed; targeted diagnostic -> "
                f"{candidates_to_run[index + 1].name}",
                file=sys.stderr, flush=True)

    passing = [
        profile for profile, row in zip(candidates_to_run, candidate_rows)
        if row["quality_passed"]]
    if passing:
        chosen = passing[0]
    else:
        # Decisive dead-end reporting: retain the least-bad measured profile,
        # prioritizing tool semantics and then minimum absolute NLL drift.
        best_index = min(
            range(len(candidate_rows)),
            key=lambda index: (
                not candidate_rows[index]["semantic_ok"],
                abs(candidate_rows[index]["teacher_nll_delta"]),
            ))
        chosen = candidates_to_run[best_index]
    return {
        "target_prompt_tokens": target,
        "prompt_sha256": hashlib.sha256(
            np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest(),
        "control": control,
        "candidates": candidate_rows,
        "chosen_profile": chosen.name,
    }, chosen


def _find_candidate(row: dict, profile_name: str) -> dict | None:
    return next((
        candidate for candidate in row["candidates"]
        if candidate["profile"] == profile_name), None)


def _write_report(report: dict, path: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4")
    parser.add_argument("--targets", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--tsp-layer", type=int, default=13)
    parser.add_argument("--later-tsp-layer", type=int, default=17)
    parser.add_argument("--initial-retention", type=float, default=0.20)
    parser.add_argument("--diagnostic-retention", type=float, default=0.30)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--pool-width", type=int, default=7)
    parser.add_argument("--query-chunk", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--needle-depth", type=float, default=0.25)
    parser.add_argument("--min-prefill-speedup", type=float, default=1.05)
    parser.add_argument("--cache-mb", type=int, default=7000)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"not a complete local checkpoint: {model}")
    if sorted(set(args.targets)) != args.targets or min(args.targets) < 256:
        parser.error("targets must be unique, increasing, and at least 256")
    if not 0 <= args.needle_depth <= 1:
        parser.error("needle-depth must be within [0, 1]")

    initial = TSPProfile(
        args.tsp_layer, args.initial_retention, args.recent_window,
        args.pool_width, args.query_chunk)
    diagnostic = TSPProfile(
        args.tsp_layer, args.diagnostic_retention, args.recent_window,
        args.pool_width, args.query_chunk)
    later = TSPProfile(
        args.later_tsp_layer, args.diagnostic_retention, args.recent_window,
        args.pool_width, args.query_chunk)
    diagnostic_profiles = [initial, diagnostic, later]

    engine = None
    report = {
        "gate": PROFILE_FAMILY,
        "schema_version": 1,
        "approximate": True,
        "quarantined": True,
        "runtime_default_enabled": False,
        "exit_semantics": {
            "0": "all preregistered promotion gates passed; fixture remains quarantined",
            "1": "completed but an overall promotion gate failed",
            "2": "unsupported or unadmitted architecture/configuration",
            "3": "operational failure",
        },
        "model": str(model),
        "targets": [],
        "passed": False,
    }
    try:
        engine = StreamingEngine(model, _runtime(args.cache_mb))
        validate_admission(engine, initial, max(args.targets))
        report["architecture"] = {
            "model_type": engine.cfg.model_type,
            "layers": engine.cfg.num_hidden_layers,
            "hidden_size": engine.cfg.hidden_size,
            "attention_heads": engine.cfg.num_attention_heads,
            "kv_heads": engine.cfg.num_key_value_heads,
            "head_dim": engine.cfg.head_dim,
            "max_positions": engine.effective_max_position_embeddings,
            "admitted": True,
        }
        report["settings"] = {
            "initial_profile": initial.name,
            "diagnostic_profile": diagnostic.name,
            "later_diagnostic_profile": later.name,
            "diagnostics": not args.no_diagnostics,
            "diagnostic_policy": (
                "20% first; 30% same layer only on quality failure; "
                "30% later layer only if 30% same-layer still fails"),
            "query_chunk": args.query_chunk,
            "recent_window": args.recent_window,
            "pool_width": args.pool_width,
            "max_tokens": args.max_tokens,
            "needle_depth": args.needle_depth,
            "preregistered_max_teacher_nll_delta": (
                PREREGISTERED_MAX_NLL_DELTA),
            "min_prefill_speedup": args.min_prefill_speedup,
            "benchmark_order": "dense control then targeted candidate(s)",
        }
        report["weight_warmup"] = _warm_weights(engine)
        filler = _filler(engine, max(args.targets))
        first_tokens, _lookup, _path = _prompt(
            engine, filler, args.targets[0], depth=args.needle_depth)
        warm_profile = TSPProfile(
            initial.tsp_layer, 1.0, initial.recent_window,
            initial.pool_width, initial.query_chunk)
        warm_tokens = first_tokens[-max(256, initial.recent_window + 1):]
        # Warm the exact stateless kernels without retaining any state.
        warm = run_prefill(engine, warm_tokens, warm_profile)
        report["kernel_warmup"] = {
            "tokens": len(warm_tokens),
            "wall_s": warm.wall_s,
            "peak_bytes": warm.peak_bytes,
        }
        del warm
        _release()

        chosen = initial
        for target_index, target in enumerate(args.targets):
            tokens, lookup, expected_path = _prompt(
                engine, filler, target, depth=args.needle_depth)
            profiles = diagnostic_profiles if target_index == 0 else [chosen]
            row, measured_choice = _run_target(
                engine, tokens, expected_path, profiles,
                max_tokens=args.max_tokens,
                min_speedup=args.min_prefill_speedup,
                diagnose=(target_index == 0 and not args.no_diagnostics))
            row["lookup_id"] = lookup
            row["expected_path"] = expected_path
            report["targets"].append(row)
            if target_index == 0:
                chosen = measured_choice

        report["selected_profile"] = chosen.name
        selected_rows = [
            _find_candidate(row, chosen.name) for row in report["targets"]]
        report["failures"] = []
        for row, candidate in zip(report["targets"], selected_rows):
            target = row["target_prompt_tokens"]
            if candidate is None:
                report["failures"].append(
                    f"{target}: selected profile was not measured")
                continue
            if not candidate["task_passed"]:
                report["failures"].append(
                    f"{target}: selected profile failed task/tool retrieval")
            if not candidate["id_passed"]:
                report["failures"].append(
                    f"{target}: selected profile failed greedy-ID equality")
            if not candidate["nll_gate_passed"]:
                report["failures"].append(
                    f"{target}: NLL delta {candidate['teacher_nll_delta']:.6f} "
                    f"exceeded preregistered {PREREGISTERED_MAX_NLL_DELTA:.6f}")
            if not candidate["position_geometry_ok"]:
                report["failures"].append(
                    f"{target}: selected profile failed position geometry")
            if not candidate["isolation_contract_ok"]:
                report["failures"].append(
                    f"{target}: selected profile failed state isolation")
            if not candidate["performance_passed"]:
                report["failures"].append(
                    f"{target}: selected profile failed prefill speedup")
        report["promotion_passed"] = not report["failures"]
        report["promotion_decision"] = (
            "pass-but-remain-quarantined" if report["promotion_passed"]
            else "fail-keep-quarantined")
        report["passed"] = report["promotion_passed"]
        _write_report(report, args.result_json)
        return 0 if report["passed"] else 1
    except UnsupportedFastKVTSP as error:
        report["architecture"] = {"admitted": False, "reason": str(error)}
        report["failure_kind"] = "unsupported"
        report["failures"] = [str(error)]
        _write_report(report, args.result_json)
        return 2
    except Exception as error:  # pragma: no cover - exercised by real hardware
        report["failure_kind"] = "operational"
        report["failures"] = [f"{type(error).__name__}: {error}"]
        report["traceback"] = traceback.format_exc()
        _write_report(report, args.result_json)
        return 3
    finally:
        if engine is not None:
            engine.close()
        mx.clear_cache()


if __name__ == "__main__":
    raise SystemExit(main())
