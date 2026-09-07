#!/usr/bin/env python3
"""Large-context semantic retrieval plus sustained-output HTTP gate.

The synthetic prompt is deterministic but intentionally not persisted.  Two
unique canaries occur only in their distant records, never in the final suffix;
the response must recover both before continuing a bounded sequence. An explicit
legacy copy-output mode reproduces older timing bodies but is not retrieval
evidence. Explicit --completed-retrieval uses seeded, domain-varied records and
requires a naturally completed exact JSON answer instead of an endless sequence.
These are synthetic no-tool cases, not captured-harness/Plex or DSA proof.
The result artifact contains hashes, counts, boolean quality witnesses, runtime
telemetry, and pressure. Completed mode also saves the parsed terminal response
to a separate immutable private receipt so it can be independently rescored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.fixtures.completed_retrieval_corpus import DOMAINS, build_case, score_response
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private


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
    for item in response.get("output") or ():
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or ():
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "".join(parts)
    top = response.get("output_text")
    return top if isinstance(top, str) else ""


def _response_integrity_failures(response: dict, *, require_completed=False) -> list[str]:
    """Legacy diagnostics allow an explicit cap; completed mode never does."""
    failures = []
    if response.get("error"):
        failures.append("response contains a protocol error")
    status = response.get("status")
    details = response.get("incomplete_details") or {}
    if require_completed and (status != 'completed' or details):
        failures.append('completed retrieval requires natural completion')
    elif status != "completed" and not (
            status == "incomplete" and isinstance(details, dict)
            and details.get("reason") == "max_output_tokens"):
        failures.append("response neither completed nor reached its output cap")
    output = response.get('output') or []
    if not isinstance(output, list) or any(not isinstance(item, dict) or item.get("type") not in (
            "message", "reasoning") for item in output):
        failures.append("no-tool diagnostic returned an unexpected output item")
    timing = response.get("vmodel_timing") or {}
    peak = timing.get("true_peak_metal_bytes") if isinstance(timing, dict) else None
    if (isinstance(peak, bool) or not isinstance(peak, (int, float))
            or not 0 < peak < float("inf")):
        failures.append("true peak Metal telemetry is missing or invalid")
    return failures


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


def _validate_task_options(parser, args):
    if args.min_output_tokens is None:
        args.min_output_tokens = 1 if args.completed_retrieval else 96
    if args.min_consecutive_integers is None:
        args.min_consecutive_integers = 0 if args.completed_retrieval else 8
    if args.max_swap_growth_mb is None:
        args.max_swap_growth_mb = 16.0 if args.completed_retrieval else 64.0
    if args.completed_retrieval:
        if args.legacy_copy_output_diagnostic:
            parser.error('completed retrieval cannot expose answers in the suffix')
        if args.fixture_seed is None or args.retrieval_domain is None:
            parser.error('completed retrieval requires fixture-seed and retrieval-domain')
        if not 0 <= args.fixture_seed < 2**63:
            parser.error('fixture-seed must be in 0..2**63-1')
        if args.max_output_tokens < 256 or args.min_output_tokens != 1 or args.min_consecutive_integers != 0:
            parser.error('completed retrieval requires max-output-tokens>=256, min-output-tokens=1, min-consecutive-integers=0')
        if not re.fullmatch('[0-9a-f]{64}', args.expected_profile_digest or ''):
            parser.error('completed retrieval requires expected-profile-digest')
        if (not math.isfinite(args.max_peak_metal_gb) or not 0 < args.max_peak_metal_gb <= 8.5
                or not math.isfinite(args.min_available_gb) or args.min_available_gb < 5.3
                or not math.isfinite(args.max_swap_growth_mb) or not 0 <= args.max_swap_growth_mb <= 16):
            parser.error('completed retrieval cannot weaken Metal/available/swap acceptance limits')
    elif args.fixture_seed is not None or args.retrieval_domain is not None or args.expected_profile_digest is not None:
        parser.error('fixture-seed/retrieval-domain/expected-profile-digest require completed-retrieval')


def _count_or_zero(value):
    return value if type(value) is int and value >= 0 else 0


def _finite_json(value):
    """Keep malformed non-finite telemetry from destroying a failed receipt."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value


def _reject_nonfinite_constant(value):
    raise ValueError('non-finite JSON constant in response')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8077/v1/responses")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--target-user-tokens", type=int, default=30_000)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--min-output-tokens", type=int)
    parser.add_argument("--min-consecutive-integers", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=(
        "none", "minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-available-gb", type=float, default=5.3)
    parser.add_argument("--max-swap-growth-mb", type=float)
    task_mode = parser.add_mutually_exclusive_group()
    task_mode.add_argument(
        "--legacy-copy-output-diagnostic", action="store_true",
        help="reproduce the old answer-revealing timing body; NOT retrieval proof")
    task_mode.add_argument('--completed-retrieval', action='store_true',
        help='require a complete exact answer to seeded synthetic distant-record lookup; not sustained-output/captured/Plex proof')
    parser.add_argument('--fixture-seed', type=int)
    parser.add_argument('--retrieval-domain', choices=tuple(DOMAINS))
    parser.add_argument('--expected-profile-digest')
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    _validate_task_options(parser, args)
    if args.result_json.exists():
        parser.error("result-json already exists")
    response_path = args.result_json.with_name(args.result_json.stem + '.response.json') if args.completed_retrieval else None
    if response_path and response_path.exists():
        parser.error('response receipt already exists')
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
    completed_case = None
    if args.completed_retrieval:
        completed_case = build_case(tokenizer, args.target_user_tokens,
            fixture_seed=args.fixture_seed, domain=args.retrieval_domain)
        user_text, local_user_tokens = completed_case.user_text, completed_case.local_user_tokens
    else:
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
    if args.reasoning_effort is not None:
        request_value["reasoning"] = {"effort": args.reasoning_effort}
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
            response_value = json.loads(response.read(), parse_constant=_reject_nonfinite_constant)
            if not isinstance(response_value, dict):
                response_value = {}
                raise TypeError("response is not a JSON object")
    except urllib.error.HTTPError as caught:
        error = f"HTTP {caught.code}: {caught.read()[:1000]!r}"
    except Exception as caught:  # artifact must survive timeout/connection failure
        error = f"{type(caught).__name__}: {caught}"
    wall = time.perf_counter() - started
    after = _pressure()
    response_sha256 = None
    if response_path and response_value:
        _atomic_write_private(response_path, response_value)
        response_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()

    completion = score_response(response_value, completed_case.expected,
        max_output_tokens=args.max_output_tokens) if completed_case else None
    output_text = '' if completed_case else _response_text(response_value)
    folded = output_text.upper()
    usage = response_value.get('usage')
    usage = usage if isinstance(usage, dict) else {}
    timing = response_value.get('vmodel_timing')
    timing = timing if isinstance(timing, dict) else {}
    output_tokens = _count_or_zero(usage.get('output_tokens'))
    input_tokens = _count_or_zero(usage.get('input_tokens'))
    prefix = f"A={CANARY_A} B={CANARY_B}"
    normalized = re.sub(r"\s+", " ", output_text).strip().upper()
    sequence_values = [int(value) for value in re.findall(
        r"(?<!\d)(\d{3})(?!\d)", output_text)]
    consecutive_prefix = 0
    for expected, value in enumerate(sequence_values, 1):
        if value != expected:
            break
        consecutive_prefix += 1

    failures = _response_integrity_failures(response_value, require_completed=bool(completed_case))
    if error is not None:
        failures.append(error)
    if input_tokens < int(args.target_user_tokens * 0.95):
        failures.append(
            f"server input tokens {input_tokens} are below the large-context floor")
    if output_tokens < args.min_output_tokens:
        failures.append(
            f"output tokens {output_tokens} are below {args.min_output_tokens}")
    if completed_case:
        failures.extend('completed retrieval: ' + name for name, ok in completion['checks'].items() if not ok)
        if (response_value.get('vmodel_backend') != 'voom'
                or response_value.get('vmodel_checkpoint') != args.model
                or response_value.get('vmodel_runtime_profile_digest') != args.expected_profile_digest
                or response_value.get('vmodel_runtime_profile_overrides')):
            failures.append('completed retrieval backend/profile identity mismatch')
        witness = timing.get('generation_witness') or {}
        if (not isinstance(witness, dict) or witness.get('available') is not True
                or type(witness.get('generated_token_count')) is not int
                or witness.get('generated_token_count') != output_tokens
                or not all(isinstance(witness.get(key), str) and re.fullmatch('[0-9a-f]{64}', witness[key])
                    for key in ('generated_token_ids_sha256', 'prepared_prompt_token_ids_sha256'))):
            failures.append('completed retrieval generation witness missing or inconsistent')
        if timing.get('memory_prefill_retries') != 0:
            failures.append('completed retrieval prefill retry telemetry missing or nonzero')
    else:
        if CANARY_A not in folded or CANARY_B not in folded:
            failures.append("one or both retrieval canaries are absent")
        if not normalized.startswith(prefix):
            failures.append("response does not begin with the exact canary pair")
        if "VALIDATION" not in folded:
            failures.append("response omitted the sustained-output marker")
        if consecutive_prefix < args.min_consecutive_integers:
            failures.append(
                f"only {consecutive_prefix} consecutive validation integers")
    peak = timing.get('true_peak_metal_bytes')
    peak_bytes = int(peak) if type(peak) in (int, float) and math.isfinite(peak) and peak > 0 else 0
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
        "schema": "voom.qwen-large-context-output-gate.v4" if completed_case else "voom.qwen-large-context-output-gate.v3",
        "request": {
            "model": args.model,
            "target_user_tokens": args.target_user_tokens,
            "local_user_tokens": local_user_tokens,
            "server_input_tokens": input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "min_output_tokens": args.min_output_tokens,
            "min_consecutive_integers": args.min_consecutive_integers,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "tool_count": 0,
            "stream": False,
            "canonical_bytes": len(private_request),
            "completed_answer_benchmark": bool(completed_case),
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
            "incomplete_details": response_value.get("incomplete_details"),
            "checkpoint": response_value.get("vmodel_checkpoint"),
            "backend": response_value.get("vmodel_backend"),
            "runtime_profiles": response_value.get("vmodel_runtime_profiles"),
            "runtime_profile_digest": response_value.get("vmodel_runtime_profile_digest"),
            "runtime_effective_digest": response_value.get("vmodel_runtime_effective_digest"),
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
    if completed_case:
        report['request'].update(fixture=completed_case.metadata,
            task='completed-synthetic-retrieval', canary_depths=None,
            expected_profile_digest=args.expected_profile_digest)
        report['result'].update(completion=completion,
            response_path=str(response_path) if response_sha256 else None,
            response_sha256=response_sha256,
            output_bytes=completion['output_bytes'], output_sha256=completion['output_sha256'],
            canary_a_found=None, canary_b_found=None, exact_prefix=None,
            validation_marker_found=None, consecutive_validation_integers=None)
    report = _finite_json(report)
    _atomic_write_private(args.result_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
