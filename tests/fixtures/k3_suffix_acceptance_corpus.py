#!/usr/bin/env python3
"""Offline, multi-domain acceptance audit for K3 suffix-history drafting.

This loads only K3's released tokenizer, never the 1.5 TB target weights. The
fixed public traces are reference continuations, not claims about what K3
would generate. Results measure the production proposer's structural
acceptance on heterogeneous text and exact-history replay; target-verifier
latency must be measured separately on released K3 weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.suffix_decoding import SuffixDecodingCache
from runtime.tiktoken_convert import build_kimi_fast_tokenizer


CASES = (
    {
        "name": "workspace_tool_json",
        "domain": "software-tools",
        "prompt": "Inspect the application configuration and report the selected file.",
        "training": (
            '{"name":"read_file","arguments":{"path":"src/config.py"}}'
        ),
        "evaluation": (
            '{"name":"workspace_search","arguments":{"query":"cache policy"}}'
        ),
    },
    {
        "name": "calendar_tool_json",
        "domain": "calendar-tools",
        "prompt": "Find the next design review after Tuesday.",
        "training": (
            '{"name":"calendar_lookup","arguments":{"date":"2026-08-04"}}'
        ),
        "evaluation": (
            '{"name":"calendar_lookup","arguments":{"date":"2026-08-11"}}'
        ),
    },
    {
        "name": "geography_bullets",
        "domain": "geography",
        "prompt": "Give three concise country and capital facts.",
        "training": (
            "- France: Paris\n- Japan: Tokyo\n- Kenya: Nairobi\n"
        ),
        "evaluation": (
            "- Canada: Ottawa\n- Peru: Lima\n- Thailand: Bangkok\n"
        ),
    },
    {
        "name": "chemistry_table",
        "domain": "chemistry",
        "prompt": "Make a compact markdown table of element symbols.",
        "training": (
            "| Element | Symbol |\n|---|---|\n| Gold | Au |\n| Iron | Fe |\n"
        ),
        "evaluation": (
            "| Element | Symbol |\n|---|---|\n| Silver | Ag |\n| Neon | Ne |\n"
        ),
    },
    {
        "name": "python_code",
        "domain": "programming",
        "prompt": "Write a small function that keeps unique values in order.",
        "training": (
            "def clamp(value, low, high):\n"
            "    return max(low, min(value, high))\n"
        ),
        "evaluation": (
            "def unique(items):\n"
            "    return list(dict.fromkeys(items))\n"
        ),
    },
    {
        "name": "debug_steps",
        "domain": "systems",
        "prompt": "List a short, safe debugging sequence for an intermittent service.",
        "training": (
            "1. Reproduce the failure.\n"
            "2. Capture logs and metrics.\n"
            "3. Change one variable and retest.\n"
        ),
        "evaluation": (
            "1. Check power and cables.\n"
            "2. Record the exact error.\n"
            "3. Test one component at a time.\n"
        ),
    },
    {
        "name": "math_explanation",
        "domain": "mathematics",
        "prompt": "Explain a basic identity in two sentences.",
        "training": (
            "The derivative of x squared is 2x because the power rule lowers "
            "the exponent by one. This holds for every real x."
        ),
        "evaluation": (
            "The sum of the first n odd numbers is n squared. Pairing the "
            "successive L-shaped borders forms an n by n square."
        ),
    },
    {
        "name": "history_prose",
        "domain": "history",
        "prompt": "Give a neutral two-sentence historical summary.",
        "training": (
            "The printing press reduced the cost of reproducing texts. Its "
            "spread changed education, scholarship, and public debate."
        ),
        "evaluation": (
            "Railways reduced travel time between growing cities. Their "
            "expansion changed trade, migration, and industrial planning."
        ),
    },
    {
        "name": "biology_summary",
        "domain": "biology",
        "prompt": "Summarize a biological process without jargon.",
        "training": (
            "Photosynthesis stores light energy in sugars and releases oxygen. "
            "Plants use those sugars to support growth and metabolism."
        ),
        "evaluation": (
            "Cellular respiration releases usable energy from nutrients. "
            "Cells capture part of that energy in molecules called ATP."
        ),
    },
    {
        "name": "recipe_steps",
        "domain": "cooking",
        "prompt": "Provide three numbered preparation steps.",
        "training": (
            "1. Rinse the rice.\n2. Add water and bring it to a simmer.\n"
            "3. Cover, cook, and rest before serving.\n"
        ),
        "evaluation": (
            "1. Chop the vegetables.\n2. Heat the pan and add oil.\n"
            "3. Cook until tender, then season.\n"
        ),
    },
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _new_cache(
    *,
    k: int = 2,
    factor: float = 2.0,
    min_probability: float = 0.75,
) -> SuffixDecodingCache:
    return SuffixDecodingCache(
        identity="k3-public-structural-corpus-v1",
        max_depth=8,
        max_spec_tokens=k,
        factor=factor,
        min_probability=min_probability,
        max_cached_requests=64,
        max_cached_tokens=16_384,
        max_nodes=400_000,
        max_bytes=128_000_000,
        max_local_tokens=8_192,
    )


def _simulate(cache, prompt: list[int], target: list[int]) -> dict:
    if not target:
        raise ValueError("reference continuation must not be empty")
    state = cache.begin_request(prompt)
    state.append_committed(target[:1])
    cursor = 1
    proposed = 0
    accepted = 0
    sweeps = 0
    proposed_rounds = 0
    global_rounds = 0
    local_rounds = 0
    while cursor < len(target):
        remaining = len(target) - cursor
        proposal = cache.propose(
            state, max_tokens=max(0, remaining - 1)
        )
        draft = list(proposal.tokens)
        proposed += len(draft)
        proposed_rounds += int(bool(draft))
        global_rounds += int(bool(draft) and proposal.source == "global")
        local_rounds += int(bool(draft) and proposal.source == "local")

        matched = 0
        while (
            matched < len(draft)
            and cursor + matched < len(target)
            and draft[matched] == target[cursor + matched]
        ):
            matched += 1
        accepted += matched
        commit_count = min(remaining, matched + 1)
        state.append_committed(target[cursor : cursor + commit_count])
        cursor += commit_count
        sweeps += 1

    baseline_sweeps = max(0, len(target) - 1)
    return {
        "target_tokens": len(target),
        "baseline_sweeps": baseline_sweeps,
        "target_sweeps": sweeps,
        "proposed": proposed,
        "accepted": accepted,
        "proposal_rounds": proposed_rounds,
        "global_rounds": global_rounds,
        "local_rounds": local_rounds,
        "proposal_acceptance": (
            accepted / proposed if proposed else None
        ),
        "sweep_speedup_upper_bound": (
            baseline_sweeps / sweeps if sweeps else 1.0
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "target_tokens",
        "baseline_sweeps",
        "target_sweeps",
        "proposed",
        "accepted",
        "proposal_rounds",
        "global_rounds",
        "local_rounds",
    )
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in keys
    }
    totals["proposal_acceptance"] = (
        totals["accepted"] / totals["proposed"]
        if totals["proposed"] else None
    )
    totals["sweep_speedup_upper_bound"] = (
        totals["baseline_sweeps"] / totals["target_sweeps"]
        if totals["target_sweeps"] else 1.0
    )
    return totals


def _heldout_rows(
    encoded: list[dict], *,
    k: int = 2,
    factor: float = 2.0,
    min_probability: float = 0.75,
) -> list[dict]:
    cache = _new_cache(
        k=k,
        factor=factor,
        min_probability=min_probability,
    )
    for case in encoded:
        if not cache.add_output(case["training_tokens"]):
            raise RuntimeError("training trace exceeded suffix-cache bounds")
    rows = []
    for case in encoded:
        rows.append({
            "name": case["name"],
            "domain": case["domain"],
            **_simulate(
                cache,
                case["prompt_tokens"],
                case["evaluation_tokens"],
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "Kimi-K3",
    )
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = build_kimi_fast_tokenizer(args.model_dir)
    encoded = []
    for case in CASES:
        encoded.append({
            **case,
            "prompt_tokens": list(tokenizer.encode(case["prompt"]).ids),
            "training_tokens": list(
                tokenizer.encode(case["training"]).ids
            ),
            "evaluation_tokens": list(
                tokenizer.encode(case["evaluation"]).ids
            ),
        })

    heldout_rows = _heldout_rows(encoded)

    replay_rows = []
    for case in encoded:
        replay_cache = _new_cache()
        if not replay_cache.add_output(case["evaluation_tokens"]):
            raise RuntimeError("replay trace exceeded suffix-cache bounds")
        row = _simulate(
            replay_cache,
            case["prompt_tokens"],
            case["evaluation_tokens"],
        )
        replay_rows.append({
            "name": case["name"],
            "domain": case["domain"],
            **row,
        })

    parameter_grid = []
    for k in (1, 2):
        for factor in (1.0, 2.0, 4.0):
            for min_probability in (0.5, 0.75, 0.9):
                aggregate = _aggregate(_heldout_rows(
                    encoded,
                    k=k,
                    factor=factor,
                    min_probability=min_probability,
                ))
                parameter_grid.append({
                    "k": k,
                    "factor": factor,
                    "min_probability": min_probability,
                    **aggregate,
                })
    parameter_grid.sort(key=lambda row: (
        -row["sweep_speedup_upper_bound"],
        -(row["proposal_acceptance"] or 0.0),
        row["proposed"],
    ))

    report = {
        "schema": "voom.k3-suffix-acceptance-corpus.v1",
        "model_dir": str(args.model_dir.resolve()),
        "cases": len(encoded),
        "configuration": {
            "max_depth": 8,
            "k": 2,
            "factor": 2.0,
            "min_probability": 0.75,
        },
        "paired_heldout": {
            "aggregate": _aggregate(heldout_rows),
            "cases": heldout_rows,
        },
        "exact_history_replay": {
            "aggregate": _aggregate(replay_rows),
            "cases": replay_rows,
        },
        "heldout_parameter_grid": parameter_grid,
        "guardrails": {
            "loads_target_weights": False,
            "reference_traces_are_k3_generations": False,
            "sweep_speedup_is_wall_speedup": False,
            "target_verification_required_for_commit": True,
            "default_on_evidence": False,
        },
    }
    _atomic_json(args.result.resolve(), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
