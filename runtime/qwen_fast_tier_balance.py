#!/usr/bin/env python3
"""Plan exact Qwen4 expert placement from a request-weighted trace corpus.

This is a read-only, MLX-free planner.  It reuses the builder's ranking and
selection semantics, evaluates every useful hot-prefix length, and reports
both corpus and leave-one-request-out route coverage.  When traces contain
measured two-device counters it also projects the aggregate critical service
time after holding non-routed traffic constant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from formats.qwen4_fast_tier import (
    BINDING_NAME,
    MANIFEST_NAME,
    _select_trace_balanced,
    _trace_rankings,
)


PAIR_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
SCHEMA = "voom.qwen4-fast-tier-balance-plan.v1"


def _atomic_json(path: Path, value: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _selected_pairs(manifest: dict[str, dict]) -> tuple[set[tuple[int, int]], dict]:
    pairs: dict[tuple[int, int], int] = {}
    tensors: dict[tuple[int, int], int] = {}
    for name, entry in manifest.items():
        match = PAIR_RE.match(name)
        if match is None:
            continue
        pair = tuple(map(int, match.groups()))
        pairs[pair] = pairs.get(pair, 0) + int(entry["nbytes"])
        tensors[pair] = tensors.get(pair, 0) + 1
    if not pairs:
        raise ValueError("fast-tier manifest contains no Qwen expert groups")
    group_sizes = set(pairs.values())
    tensor_counts = set(tensors.values())
    if len(group_sizes) != 1 or len(tensor_counts) != 1:
        raise ValueError("fast-tier expert groups are not uniform")
    return set(pairs), {
        "selected_experts": len(pairs),
        "storage_expert_bytes": next(iter(group_sizes)),
        "tensors_per_expert": next(iter(tensor_counts)),
    }


def _route_stats(selected: set[tuple[int, int]], document: dict) -> dict:
    occurrences = hits = 0
    per_layer_occurrences: dict[int, int] = {}
    per_layer_hits: dict[int, int] = {}
    for sweep in document["sweeps"]:
        for route in sweep["routes"]:
            layer = int(route["layer"])
            experts = [int(expert) for expert in route["experts"]]
            occurrences += len(experts)
            layer_hits = sum((layer, expert) in selected for expert in experts)
            hits += layer_hits
            per_layer_occurrences[layer] = (
                per_layer_occurrences.get(layer, 0) + len(experts))
            per_layer_hits[layer] = per_layer_hits.get(layer, 0) + layer_hits
    if occurrences <= 0:
        raise ValueError("expert trace has no routed occurrences")
    return {
        "route_occurrences": occurrences,
        "route_hits": hits,
        "route_hit_fraction": hits / occurrences,
        "per_layer_route_hit_fraction": {
            str(layer): per_layer_hits.get(layer, 0) / count
            for layer, count in sorted(per_layer_occurrences.items())
        },
    }


def _candidate_pairs(
    rankings: dict[int, list[int]], *, layers: int, experts: int,
    storage_expert_bytes: int, selected_experts: int, hot: int,
) -> set[tuple[int, int]]:
    groups = [
        {"layer": layer, "expert": expert, "nbytes": storage_expert_bytes}
        for layer in range(layers) for expert in range(experts)
    ]
    selected = _select_trace_balanced(
        groups, selected_experts * storage_expert_bytes, layers, experts,
        rankings, hot)
    if len(selected) != selected_experts:
        raise ValueError("planner could not reproduce active expert capacity")
    return {(item["layer"], item["expert"]) for item in selected}


def _service_rates(documents: list[dict]) -> tuple[float, float] | None:
    fast_bytes = fast_ns = archive_bytes = archive_ns = 0
    for document in documents:
        io = document.get("baseline_io") or {}
        if min(
            int(io.get("parallel_tier_fast_bytes", 0)),
            int(io.get("parallel_tier_fast_service_ns", 0)),
            int(io.get("parallel_tier_archive_bytes", 0)),
            int(io.get("parallel_tier_archive_service_ns", 0)),
        ) <= 0:
            continue
        fast_bytes += int(io["parallel_tier_fast_bytes"])
        fast_ns += int(io["parallel_tier_fast_service_ns"])
        archive_bytes += int(io["parallel_tier_archive_bytes"])
        archive_ns += int(io["parallel_tier_archive_service_ns"])
    if min(fast_bytes, fast_ns, archive_bytes, archive_ns) <= 0:
        return None
    return fast_bytes * 1e9 / fast_ns, archive_bytes * 1e9 / archive_ns


def plan(
    fast_dir: Path,
    trace_paths: list[Path],
    *,
    evaluated_selected_experts: int = 0,
) -> dict:
    fast_dir = fast_dir.expanduser().resolve()
    trace_paths = [path.expanduser().resolve() for path in trace_paths]
    if len(trace_paths) < 2:
        raise ValueError("balance planning requires at least two request traces")
    binding_raw = (fast_dir / BINDING_NAME).read_bytes()
    manifest_raw = (fast_dir / MANIFEST_NAME).read_bytes()
    binding = json.loads(binding_raw)
    manifest = json.loads(manifest_raw)
    if binding.get("fast_manifest_sha256") != hashlib.sha256(
            manifest_raw).hexdigest():
        raise ValueError("active fast-tier binding does not authenticate manifest")
    current, capacity = _selected_pairs(manifest)
    layers = int(binding["layers"])
    experts = int(binding["experts"])
    evaluated_selected_experts = int(evaluated_selected_experts)
    if evaluated_selected_experts == 0:
        evaluated_selected_experts = capacity["selected_experts"]
    if not 1 <= evaluated_selected_experts <= layers * experts:
        raise ValueError(
            "evaluated selected experts must be in [1, layers * experts]")
    model_name = str(binding["target_model"])
    if capacity["storage_expert_bytes"] * capacity["selected_experts"] != int(
            binding["selected_bytes"]):
        raise ValueError("active binding and manifest selected bytes differ")

    documents = [json.loads(path.read_text()) for path in trace_paths]
    trace_page_sizes = {
        int(document["expert_page_bytes"]) for document in documents
    }
    if len(trace_page_sizes) != 1:
        raise ValueError("trace corpus has inconsistent expert page sizes")
    trace_expert_page_bytes = next(iter(trace_page_sizes))
    rankings, ranking_metadata = _trace_rankings(
        trace_paths, model_name=model_name, layers=layers, experts=experts)
    rates = _service_rates(documents)

    current_stats = [_route_stats(current, document) for document in documents]
    fixed: list[tuple[int, int] | None] = []
    incompatible_calibration_traces = []
    for path, document, stats in zip(
            trace_paths, documents, current_stats, strict=True):
        io = document.get("baseline_io") or {}
        if min(
            int(io.get("parallel_tier_fast_bytes", 0)),
            int(io.get("parallel_tier_archive_bytes", 0)),
        ) <= 0:
            fixed.append(None)
            continue
        hit_bytes = stats["route_hits"] * trace_expert_page_bytes
        miss_bytes = (
            stats["route_occurrences"] - stats["route_hits"]
        ) * trace_expert_page_bytes
        fixed_fast = int(io["parallel_tier_fast_bytes"]) - hit_bytes
        fixed_archive = int(io["parallel_tier_archive_bytes"]) - miss_bytes
        if min(fixed_fast, fixed_archive) < 0:
            # A route trace remains valid for placement scoring after the
            # active tier changes, but its measured fast/archive byte split
            # belongs to the older placement. Refuse only that timing
            # calibration rather than inventing negative fixed traffic.
            fixed.append(None)
            incompatible_calibration_traces.append(path.name)
            continue
        fixed.append((fixed_fast, fixed_archive))

    maximum_hot = min(
        experts,
        (evaluated_selected_experts + layers - 1) // layers,
    )
    ladder = []
    for hot in [0, *range(1, maximum_hot + 1)]:
        selected = _candidate_pairs(
            rankings, layers=layers, experts=experts,
            storage_expert_bytes=capacity["storage_expert_bytes"],
            selected_experts=evaluated_selected_experts, hot=hot)
        request_rows = []
        calibrated_seconds = []
        for path, document, fixed_io in zip(
                trace_paths, documents, fixed, strict=True):
            stats = _route_stats(selected, document)
            row = {"trace": path.name, **stats}
            if rates is not None and fixed_io is not None:
                hit_bytes = stats["route_hits"] * trace_expert_page_bytes
                miss_bytes = (
                    stats["route_occurrences"] - stats["route_hits"]
                ) * trace_expert_page_bytes
                predicted_fast = fixed_io[0] + hit_bytes
                predicted_archive = fixed_io[1] + miss_bytes
                fast_seconds = predicted_fast / rates[0]
                archive_seconds = predicted_archive / rates[1]
                critical_seconds = max(fast_seconds, archive_seconds)
                row["calibrated_parallel_service"] = {
                    "fast_bytes": predicted_fast,
                    "archive_bytes": predicted_archive,
                    "fast_seconds": fast_seconds,
                    "archive_seconds": archive_seconds,
                    "critical_seconds": critical_seconds,
                }
                calibrated_seconds.append(critical_seconds)
            request_rows.append(row)
        ladder.append({
            "trace_hot_experts_per_layer": hot,
            "selection_mode": "all-ranked-hot" if hot == 0 else "hot-then-cold",
            "mean_route_hit_fraction": sum(
                row["route_hit_fraction"] for row in request_rows
            ) / len(request_rows),
            "mean_calibrated_critical_seconds": (
                sum(calibrated_seconds) / len(calibrated_seconds)
                if calibrated_seconds else None
            ),
            "requests": request_rows,
        })

    def objective(row: dict) -> tuple[float, float, int]:
        critical = row["mean_calibrated_critical_seconds"]
        if critical is None:
            critical = float("inf")
        # Prefer greater request-equal coverage when measured service ties.
        # ``0`` is the builder's canonical spelling for all-ranked-hot.
        return critical, -row["mean_route_hit_fraction"], row[
            "trace_hot_experts_per_layer"]

    recommended = min(ladder, key=objective)

    leave_one_out = []
    if len(trace_paths) >= 3:
        recommended_hot = recommended["trace_hot_experts_per_layer"]
        for held_index, held_path in enumerate(trace_paths):
            training = [
                path for index, path in enumerate(trace_paths)
                if index != held_index
            ]
            held_rankings, _metadata = _trace_rankings(
                training, model_name=model_name, layers=layers,
                experts=experts)
            selected = _candidate_pairs(
                held_rankings, layers=layers, experts=experts,
                storage_expert_bytes=capacity["storage_expert_bytes"],
                selected_experts=evaluated_selected_experts,
                hot=recommended_hot)
            leave_one_out.append({
                "held_out_trace": held_path.name,
                "training_traces": [path.name for path in training],
                **_route_stats(selected, documents[held_index]),
            })

    return {
        "schema": SCHEMA,
        "fast_dir": str(fast_dir),
        "binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "model": model_name,
        "layers": layers,
        "experts": experts,
        **capacity,
        "evaluated_selected_experts": evaluated_selected_experts,
        "evaluated_selected_bytes": (
            evaluated_selected_experts * capacity["storage_expert_bytes"]),
        "trace_expert_page_bytes": trace_expert_page_bytes,
        "measured_service_bytes_per_second": (
            {"fast": rates[0], "archive": rates[1]} if rates else None
        ),
        "incompatible_calibration_traces": incompatible_calibration_traces,
        "ranking": ranking_metadata,
        "current": {
            "trace_hot_experts_per_layer": binding.get(
                "trace_hot_experts_per_layer"),
            "requests": [
                {"trace": path.name, **stats}
                for path, stats in zip(trace_paths, current_stats, strict=True)
            ],
        },
        "recommended": recommended,
        "leave_one_out": leave_one_out,
        "ladder": ladder,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fast_dir", type=Path)
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument(
        "--evaluated-selected-experts", type=int, default=0,
        help=("score a different complete-expert capacity while calibrating "
              "fixed I/O from the active tier; zero keeps active capacity"),
    )
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = plan(
        args.fast_dir, args.trace,
        evaluated_selected_experts=args.evaluated_selected_experts)
    if args.result:
        _atomic_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
