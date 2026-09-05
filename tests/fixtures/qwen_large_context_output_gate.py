#!/usr/bin/env python3
"""Large-context semantic retrieval plus sustained-output HTTP gate.

The synthetic prompt is deterministic but intentionally not persisted.  Two
unique canaries occur only in their distant records, never in the final suffix;
the response must recover both before continuing a bounded sequence. An explicit
legacy copy-output mode reproduces older timing bodies but is not retrieval
evidence. The result artifact contains
only hashes, counts, boolean quality witnesses, runtime telemetry, and pressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from tokenizers import Tokenizer


CANARY_A = "LANTERN-7329-COBALT"
CANARY_B = "HARBOR-1846-AMBER"


@dataclass(frozen=True)
class Pressure:
    available_bytes: int
    swap_used_bytes: int
    swap_out_bytes: int


def _pressure() -> Pressure:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return Pressure(
        available_bytes=int(memory.available),
        swap_used_bytes=int(swap.used),
        swap_out_bytes=int(swap.sout),
    )


def _response_text(response: dict) -> str:
    parts: list[str] = []
    top = response.get("output_text")
    if isinstance(top, str):
        parts.append(top)
    for item in response.get("output") or ():
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or ():
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _build_user_text(
        tokenizer: Tokenizer, target_tokens: int, *,
        legacy_copy_output_diagnostic: bool = False) -> tuple[str, int]:
    prefix = (
        "Read this synthetic archive carefully. Two TARGET RECORD lines contain "
        "the only authoritative retrieval values. Ordinary entries are filler.\n"
    )
    needle_a = f"\nTARGET RECORD A: retrieval_code={CANARY_A}.\n"
    needle_b = f"\nTARGET RECORD B: retrieval_code={CANARY_B}.\n"
    answer_instruction = (
        "\nEnd of archive. Begin the answer with exactly: "
        f"A={CANARY_A} B={CANARY_B}\n"
        if legacy_copy_output_diagnostic else
        "\nEnd of archive. Retrieve the retrieval_code values from TARGET "
        "RECORD A and TARGET RECORD B. Begin the answer with A= followed by "
        "record A's code, a single space, and B= followed by record B's code.\n"
    )
    suffix = (
        answer_instruction
        +
        "Then write VALIDATION followed by consecutive three-digit integers "
        "starting at 001, separated by single spaces, and continue until the "
        "output limit. Do not discuss the archive and do not invent a code."
    )
    fixed = tokenizer.encode(prefix + needle_a + needle_b + suffix).ids
    if target_tokens <= len(fixed) + 256:
        raise ValueError("target token count leaves insufficient filler")
    filler_unit = tokenizer.encode(
        " Archive entry: cedar compass, quiet inlet, silver ledger, ordinary "
        "inventory, no retrieval code."
    ).ids
    filler_count = target_tokens - len(fixed)
    filler = (filler_unit * ((filler_count // len(filler_unit)) + 1))[
        :filler_count]
    first = int(len(filler) * 0.13)
    second = int(len(filler) * 0.73)
    token_ids = (
        tokenizer.encode(prefix).ids
        + filler[:first]
        + tokenizer.encode(needle_a).ids
        + filler[first:second]
        + tokenizer.encode(needle_b).ids
        + filler[second:]
        + tokenizer.encode(suffix).ids
    )
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    actual = len(tokenizer.encode(text).ids)
    return text, actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8077/v1/responses")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--target-user-tokens", type=int, default=30_000)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--min-output-tokens", type=int, default=96)
    parser.add_argument("--min-consecutive-integers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-available-gb", type=float, default=5.3)
    parser.add_argument("--max-swap-growth-mb", type=float, default=64.0)
    parser.add_argument(
        "--legacy-copy-output-diagnostic", action="store_true",
        help="reproduce the old answer-revealing timing body; NOT retrieval proof")
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    if args.result_json.exists():
        parser.error("result-json already exists")
    if not args.tokenizer.is_file():
        parser.error("tokenizer file does not exist")
    if not 0 <= args.temperature:
        parser.error("temperature must be non-negative")
    if not 1 <= args.min_output_tokens <= args.max_output_tokens:
        parser.error("output-token bounds are inconsistent")
    if args.min_consecutive_integers < 0:
        parser.error("min-consecutive-integers must be non-negative")
    if args.target_user_tokens < 1024 or args.timeout <= 0:
        parser.error("target-user-tokens and timeout must be positive")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    user_text, local_user_tokens = _build_user_text(
        tokenizer, args.target_user_tokens,
        legacy_copy_output_diagnostic=args.legacy_copy_output_diagnostic)
    request_value = {
        "model": args.model,
        "input": [
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": "You are a precise long-context retrieval assistant.",
                }],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        ],
        "tools": [],
        "tool_choice": "none",
        "temperature": args.temperature,
        "seed": args.seed,
        "max_output_tokens": args.max_output_tokens,
        "stream": False,
    }
    private_request = json.dumps(
        request_value, ensure_ascii=False, separators=(",", ":"),
    ).encode()
    before = _pressure()
    started = time.perf_counter()
    error = None
    response_value: dict = {}
    try:
        request = urllib.request.Request(
            args.url, data=private_request,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_value = json.loads(response.read())
    except urllib.error.HTTPError as caught:
        error = f"HTTP {caught.code}: {caught.read()[:1000]!r}"
    except Exception as caught:  # artifact must survive timeout/connection failure
        error = f"{type(caught).__name__}: {caught}"
    wall = time.perf_counter() - started
    after = _pressure()

    output_text = _response_text(response_value)
    folded = output_text.upper()
    usage = response_value.get("usage") or {}
    timing = response_value.get("vmodel_timing") or {}
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    prefix = f"A={CANARY_A} B={CANARY_B}"
    normalized = re.sub(r"\s+", " ", output_text).strip().upper()
    sequence_values = [int(value) for value in re.findall(
        r"(?<!\d)(\d{3})(?!\d)", output_text)]
    consecutive_prefix = 0
    for expected, value in enumerate(sequence_values, 1):
        if value != expected:
            break
        consecutive_prefix += 1

    failures: list[str] = []
    if error is not None:
        failures.append(error)
    if input_tokens < int(args.target_user_tokens * 0.95):
        failures.append(
            f"server input tokens {input_tokens} are below the large-context floor")
    if output_tokens < args.min_output_tokens:
        failures.append(
            f"output tokens {output_tokens} are below {args.min_output_tokens}")
    if CANARY_A not in folded or CANARY_B not in folded:
        failures.append("one or both retrieval canaries are absent")
    if not normalized.startswith(prefix):
        failures.append("response does not begin with the exact canary pair")
    if "VALIDATION" not in folded:
        failures.append("response omitted the sustained-output marker")
    if consecutive_prefix < args.min_consecutive_integers:
        failures.append(
            f"only {consecutive_prefix} consecutive validation integers")
    peak_bytes = int(timing.get("true_peak_metal_bytes", 0) or 0)
    if peak_bytes >= int(args.max_peak_metal_gb * 1e9):
        failures.append("true peak Metal exceeded the configured ceiling")
    if after.available_bytes < int(args.min_available_gb * 1e9):
        failures.append("available memory fell below the configured floor")
    swap_growth = max(
        after.swap_used_bytes - before.swap_used_bytes,
        after.swap_out_bytes - before.swap_out_bytes,
    )
    if swap_growth > int(args.max_swap_growth_mb * 1e6):
        failures.append("swap growth exceeded the configured ceiling")

    report = {
        "schema": "voom.qwen-large-context-output-gate.v2",
        "request": {
            "model": args.model,
            "target_user_tokens": args.target_user_tokens,
            "local_user_tokens": local_user_tokens,
            "server_input_tokens": input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "min_output_tokens": args.min_output_tokens,
            "min_consecutive_integers": args.min_consecutive_integers,
            "temperature": args.temperature,
            "seed": args.seed,
            "request_sha256": hashlib.sha256(private_request).hexdigest(),
            "canary_depths": [0.13, 0.73],
            "task": ("copy-output-diagnostic" if args.legacy_copy_output_diagnostic
                     else "retrieval-and-sustained-output"),
            "answers_in_suffix": args.legacy_copy_output_diagnostic,
        },
        "result": {
            "wall_seconds": round(wall, 4),
            "response_status": response_value.get("status"),
            "output_tokens": output_tokens,
            "output_bytes": len(output_text.encode()),
            "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            "canary_a_found": CANARY_A in folded,
            "canary_b_found": CANARY_B in folded,
            "exact_prefix": normalized.startswith(prefix),
            "validation_marker_found": "VALIDATION" in folded,
            "consecutive_validation_integers": consecutive_prefix,
            "timing": timing,
            "pressure_before": asdict(before),
            "pressure_after": asdict(after),
            "swap_growth_bytes": swap_growth,
        },
        "failures": failures,
        "passed": not failures,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
