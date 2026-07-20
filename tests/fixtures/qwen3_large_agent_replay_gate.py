#!/usr/bin/env python3
"""Replay a private real-harness Qwen3 request without exposing its payload.

The request body remains under ``logs/captured_requests`` (gitignored).  This
tracked gate pins each known capture's identity (see KNOWN_CAPTURES below),
submits exact copies to a running vOOM server, and records only hashes,
counts, timing/cache telemetry, progress events, and host pressure.  Response
text and tool schemas are never printed or persisted.

Example (qwen35_large_agent_v1, the original 132-tool capture):

  .venv/bin/python tests/fixtures/qwen3_large_agent_replay_gate.py \
    logs/captured_requests/1784492063459_cfb3f558.json \
    --repeats 2 --max-output-tokens 16 --stream \
    --expected-selected-tools 32 --expected-max-input-tokens 12000

Example (qwen36_gateway_thrash_v1, the 134-tool lossy-Qwen3.6-35B-A3B
prefill/gateway-thrashing baseline from 2026-07-20):

  .venv/bin/python tests/fixtures/qwen3_large_agent_replay_gate.py \
    logs/captured_requests/1784574315421_94161f5f.json \
    --repeats 1 --max-output-tokens 16 --stream

Replay each user boundary from a saved Kai conversation while retaining the
captured tool catalog and harness system prefix:

  .venv/bin/python tests/fixtures/qwen3_large_agent_replay_gate.py \
    logs/captured_requests/1784492063459_cfb3f558.json \
    --kai-conversation /path/to/conversation.json --max-output-tokens 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


# 2026-07-20: was a single (sha256, bytes, tools) constant -- generalized to
# a small registry so more than one real captured request can be pinned as
# a durable baseline over time without displacing an existing one. Each
# entry's identity is checked against the file's actual bytes; the payload
# itself is still never committed (logs/captured_requests/ stays
# gitignored -- only these hashes/counts, never the request body, live in
# git).
KNOWN_CAPTURES = {
    "qwen35_large_agent_v1": {
        "sha256": "e921a49c770cfa1625bf946616aa1cb9f4f63f1bbfe9eddb66db45fd092a034d",
        "bytes": 157_866,
        "tools": 132,
    },
    # Live-confirmed prefill/gateway-thrashing baseline (2026-07-20): a
    # 134-tool, ~29,829-token lossy-Qwen3.6-35B-A3B request that originally
    # crashed outright (duplicate leading system messages, then repeated
    # MemoryError), used to validate the tool-gateway reduction,
    # expert-fetch batching, and memory-adaptive trunk-pin/chunk-size work
    # from that session.
    "qwen36_gateway_thrash_v1": {
        "sha256": "8ac18b8e8bc190180b4cc0e02c2453d313ec850642cc5d5f63b32e5537b90e85",
        "bytes": 178_616,
        "tools": 134,
    },
}

DEFERRED_ACTION_TURNS = [
    {"role": "user", "content": "Tell me a joke about Node.js."},
    {"role": "assistant", "content": "Why did the Node.js developer get stuck in the ocean? Because he tried to run a script on a boat."},
    {"role": "user", "content": "What folder are we in?"},
    {"role": "assistant", "content": "/Volumes/Workspace NVME/git/kai-plugin-plex"},
    {"role": "user", "content": "What's the largest top-level directory?"},
    {"role": "assistant", "content": "The largest top-level directory is git."},
    {"role": "user", "content": "Check for real."},
    {"role": "assistant", "content": "I'll run a shell command to inspect the directories and their sizes."},
    {"role": "user", "content": "do it"},
]

TOOL_RESULT_TURNS = [
    {"role": "user", "content": "What folder are we in?"},
    {
        "type": "function_call", "call_id": "call_fixture_1",
        "name": "mastra_workspace_list_files", "arguments": "{}",
    },
    {
        "type": "function_call_output", "call_id": "call_fixture_1",
        "output": "/workspace\n  src\n  tests",
    },
]


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


def _safe_selection(value) -> dict:
    value = value if isinstance(value, dict) else {}
    keys = (
        "requested", "selected", "lossy_shortlist", "shortlist_soft_limit",
        "pinned", "tool_retrieval_profile", "hidden_tool_gateway", "gateway_phase",
        "gateway_decision_prompt_tokens", "gateway_decision_output_tokens",
        "gateway_search_rounds", "gateway_query_sha256",
        "gateway_search_result_cap",
        "gateway_enable_rounds", "gateway_catalog_action",
        "gateway_activated_tools", "gateway_activation_profile",
        "gateway_activation_previous_tools",
        "gateway_activation_top_tool_reused",
        "gateway_requested_results", "gateway_decision_branch",
        "gateway_direct_streaming", "gateway_late_search_suppressed",
        "gateway_late_catalog_action_suppressed",
        "gateway_search_forced", "gateway_force_reason",
        "gateway_execution_choice_required", "gateway_real_tool_required",
        "gateway_abstention_available", "gateway_execution_outcome",
        "tool_embedding_profile", "tool_embedding_status",
        "tool_embedding_catalog_id", "tool_embedding_tool_cache_hits",
        "tool_embedding_tool_cache_misses", "tool_embedding_query_cache_hit",
        "tool_embedding_semantic_weight", "tool_embedding_seconds",
        "tool_embedding_score_min", "tool_embedding_score_max",
        "tool_embedding_fallback",
        "resident_kv_bytes_per_token", "resident_kv_projected_bytes",
        "resident_kv_limit_bytes", "resident_kv_paged",
    )
    return {key: value[key] for key in keys if key in value}


def _summary(response: dict, *, wall_s: float, events: list[str],
             progress: list[dict], deltas: list[str]) -> dict:
    output = response.get("output") or []
    stable_output = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            stable_output.append({
                "type": "function_call", "name": item.get("name"),
                "arguments": item.get("arguments"),
            })
        else:
            stable_output.append({
                "type": item.get("type"),
                "content": item.get("content"),
            })
    private_output = json.dumps(
        stable_output, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    delta_sizes = [len(delta.encode("utf-8")) for delta in deltas]
    final_output_text = response.get("output_text", "")
    return {
        "http_status": 200,
        "response_status": response.get("status"),
        "error": response.get("error"),
        "wall_seconds": round(wall_s, 4),
        "usage": response.get("usage"),
        "cache_phases": response.get("vmodel_cache_phases"),
        "timing": response.get("vmodel_timing"),
        "max_output_tokens": response.get("vmodel_max_output_tokens"),
        "output_budget_source": response.get("vmodel_output_budget_source"),
        "tool_selection": _safe_selection(response.get("vmodel_tool_selection")),
        "output_types": [item.get("type") for item in output if isinstance(item, dict)],
        "function_call_names": [
            item.get("name") for item in output
            if isinstance(item, dict) and item.get("type") == "function_call"],
        "output_sha256": hashlib.sha256(private_output).hexdigest(),
        "output_bytes": len(private_output),
        "sse_event_types": events,
        "output_text_delta_events": len(delta_sizes),
        "output_text_delta_bytes": sum(delta_sizes),
        "output_text_delta_max_bytes": max(delta_sizes, default=0),
        "streamed_text_matches_final": (
            isinstance(final_output_text, str)
            and "".join(deltas) == final_output_text),
        "virtual_search_marker_exposed": (
            b"vmodel_search_tools" in private_output),
        "hidden_gateway_marker_exposed": any(
            marker in private_output
            for marker in (b"vmodel_search_tools", b"vmodel_no_suitable_tool")),
        "prefill_progress": progress,
    }


def _post(url: str, payload: bytes, timeout: float, stream: bool) -> dict:
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
        method="POST")
    started = time.perf_counter()
    events: list[str] = []
    progress: list[dict] = []
    deltas: list[str] = []
    response_value = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not stream:
                response_value = json.loads(response.read())
            else:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    event_type = str(event.get("type", ""))
                    events.append(event_type)
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if isinstance(delta, str):
                            deltas.append(delta)
                    if event_type.startswith("response.vmodel.") \
                            and event_type.endswith("_progress"):
                        progress.append({
                            key: event.get(key)
                            for key in (
                                "phase", "completed", "total", "fraction",
                                "cache_source")
                        })
                    if event_type in (
                            "response.completed", "response.incomplete",
                            "response.failed"):
                        response_value = event.get("response")
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            detail = json.loads(body)
        except ValueError:
            detail = {"error": body.decode("utf-8", errors="replace")[:1000]}
        return {
            "http_status": error.code,
            "response_status": None,
            "error": detail.get("error", detail),
            "wall_seconds": round(time.perf_counter() - started, 4),
            "sse_event_types": events,
            "output_text_delta_events": len(deltas),
            "output_text_delta_bytes": sum(
                len(delta.encode("utf-8")) for delta in deltas),
            "output_text_delta_max_bytes": max(
                (len(delta.encode("utf-8")) for delta in deltas), default=0),
            "prefill_progress": progress,
        }
    if not isinstance(response_value, dict):
        return {
            "http_status": 599,
            "response_status": None,
            "error": "server stream ended without a final response object",
            "wall_seconds": round(time.perf_counter() - started, 4),
            "sse_event_types": events,
            "output_text_delta_events": len(deltas),
            "output_text_delta_bytes": sum(
                len(delta.encode("utf-8")) for delta in deltas),
            "output_text_delta_max_bytes": max(
                (len(delta.encode("utf-8")) for delta in deltas), default=0),
            "prefill_progress": progress,
        }
    return _summary(
        response_value, wall_s=time.perf_counter() - started,
        events=events, progress=progress, deltas=deltas)


def _write(path: Path | None, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)


def _kai_text(parts) -> str:
    if not isinstance(parts, list):
        return str(parts or "")
    return "\n".join(
        str(part.get("text", "")) for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
        and part.get("text")
    )


def _kai_input_items(messages: list[dict]) -> list[dict]:
    """Convert persisted Kai turns to the Responses input item sequence.

    Tool results remain local/private. The caller hashes request identity and
    records aggregate telemetry only; neither this helper nor the report emits
    result bodies.
    """
    items: list[dict] = []
    for message in messages:
        role = message.get("role")
        parts = message.get("content")
        if role == "user":
            text = _kai_text(parts)
            items.append({
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
            continue
        if role != "assistant" or not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool-call":
                call_id = str(part.get("toolCallId", ""))
                name = str(part.get("toolName", ""))
                arguments = part.get("argsText")
                if not isinstance(arguments, str):
                    arguments = json.dumps(
                        part.get("args") or {}, ensure_ascii=False,
                        separators=(",", ":"))
                items.append({
                    "type": "function_call", "call_id": call_id,
                    "name": name, "arguments": arguments,
                })
                items.append({
                    "type": "function_call_output", "call_id": call_id,
                    "output": json.dumps(
                        part.get("result"), ensure_ascii=False,
                        separators=(",", ":"), default=str),
                })
            elif part.get("type") == "text" and part.get("text"):
                items.append({
                    "role": "assistant",
                    "content": [{
                        "type": "output_text", "text": str(part["text"]),
                    }],
                })
    return items


def _kai_request_snapshots(base_request: dict, path: Path) -> list[tuple[str, dict]]:
    conversation = json.loads(path.read_text())
    messages = conversation.get("messages")
    if not isinstance(messages, list):
        raise SystemExit("Kai conversation has no messages list")
    base_inputs = base_request.get("input")
    if (not isinstance(base_inputs, list) or not base_inputs
            or base_inputs[-1].get("role") != "user"):
        raise SystemExit("capture has no replaceable conversation")
    prefix = base_inputs[:-1]
    snapshots = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        value = dict(base_request)
        value["input"] = [*prefix, *_kai_input_items(messages[:index + 1])]
        snapshots.append((f"user_{len(snapshots) + 1}", value))
    if not snapshots:
        raise SystemExit("Kai conversation has no user turns")
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8077/v1/responses")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--omit-max-output-tokens", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--expected-selected-tools", type=int)
    parser.add_argument("--expected-max-input-tokens", type=int)
    parser.add_argument("--expected-embedding-status")
    parser.add_argument("--expected-min-text-deltas", type=int)
    parser.add_argument("--expected-output-type", choices=("message", "function_call"))
    parser.add_argument(
        "--expected-gateway-phase", choices=("direct", "search", "enable"))
    parser.add_argument("--expected-execution-outcome")
    parser.add_argument("--expected-min-output-tokens", type=int)
    parser.add_argument(
        "--expected-response-status", choices=("completed", "incomplete"))
    parser.add_argument(
        "--expected-output-budget-source", choices=("request", "eos_safety_ceiling"))
    parser.add_argument(
        "--replacement-user-text",
        help="replace only the final user turn after capture identity validation")
    parser.add_argument(
        "--scenario", choices=("deferred-action", "tool-result-answer"),
        help="replace conversation turns with a tracked regression scenario")
    parser.add_argument(
        "--kai-conversation", type=Path,
        help="replay every user boundary from this local Kai conversation")
    parser.add_argument("--min-available-gb", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.max_output_tokens <= 0 or args.timeout <= 0:
        parser.error("repeats, max-output-tokens, and timeout must be positive")
    if (args.expected_min_output_tokens is not None
            and args.expected_min_output_tokens <= 0):
        parser.error("expected-min-output-tokens must be positive")
    if args.expected_min_text_deltas is not None:
        if args.expected_min_text_deltas <= 0:
            parser.error("expected-min-text-deltas must be positive")
        if not args.stream:
            parser.error("expected-min-text-deltas requires --stream")
    mutations = sum(value is not None for value in (
        args.replacement_user_text, args.scenario, args.kai_conversation))
    if mutations > 1:
        parser.error(
            "replacement-user-text, scenario, and kai-conversation are "
            "mutually exclusive")

    raw = args.capture.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    known_label = next(
        (label for label, identity in KNOWN_CAPTURES.items()
         if (identity["sha256"], identity["bytes"]) == (digest, len(raw))),
        None)
    if known_label is None:
        expected = ", ".join(
            f"{label}={identity['sha256']}/{identity['bytes']}"
            for label, identity in KNOWN_CAPTURES.items())
        raise SystemExit(
            f"capture identity mismatch: got {digest}/{len(raw)}, "
            f"expected one of: {expected}")
    capture_tools = KNOWN_CAPTURES[known_label]["tools"]
    request_value = json.loads(raw)
    if len(request_value.get("tools") or []) != capture_tools:
        raise SystemExit("capture tool count mismatch")
    replacement_sha256 = None
    if args.scenario is not None:
        inputs = request_value.get("input")
        if (not isinstance(inputs, list) or not inputs
                or inputs[-1].get("role") != "user"):
            raise SystemExit("capture has no replaceable conversation")
        scenario_turns = (
            DEFERRED_ACTION_TURNS
            if args.scenario == "deferred-action" else TOOL_RESULT_TURNS)
        request_value["input"] = [*inputs[:-1], *scenario_turns]
        replacement_sha256 = hashlib.sha256(json.dumps(
            scenario_turns, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    elif args.replacement_user_text is not None:
        inputs = request_value.get("input")
        if (not isinstance(inputs, list) or not inputs
                or inputs[-1].get("role") != "user"):
            raise SystemExit("capture has no replaceable final user turn")
        inputs[-1]["content"] = [{
            "type": "input_text", "text": args.replacement_user_text,
        }]
        replacement_sha256 = hashlib.sha256(
            args.replacement_user_text.encode("utf-8")).hexdigest()
    if args.kai_conversation is not None:
        request_values = _kai_request_snapshots(
            request_value, args.kai_conversation)
    else:
        request_values = [
            (f"repeat_{index + 1}", dict(request_value))
            for index in range(args.repeats)]
    payloads = []
    for label, value in request_values:
        value["stream"] = args.stream
        if args.omit_max_output_tokens:
            value.pop("max_output_tokens", None)
            value.pop("max_tokens", None)
        else:
            value["max_output_tokens"] = args.max_output_tokens
        if args.stream:
            value["vmodel_progress_events"] = True
        payloads.append((label, json.dumps(
            value, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")))

    initial = _pressure()
    rows = []
    failures = []
    for index, (label, payload) in enumerate(payloads):
        before = _pressure()
        row = _post(args.url, payload, args.timeout, args.stream)
        after = _pressure()
        row["repeat"] = index + 1
        row["request_label"] = label
        row["pressure_before"] = asdict(before)
        row["pressure_after"] = asdict(after)
        rows.append(row)
        if row.get("http_status") != 200 or row.get("error"):
            failures.append(f"repeat {index + 1}: request failed")
        usage = row.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        selected = (row.get("tool_selection") or {}).get("selected")
        embedding_status = (row.get("tool_selection") or {}).get(
            "tool_embedding_status")
        if (args.expected_selected_tools is not None
                and selected != args.expected_selected_tools):
            failures.append(
                f"repeat {index + 1}: selected {selected}, "
                f"expected {args.expected_selected_tools}")
        if (args.expected_max_input_tokens is not None
                and input_tokens > args.expected_max_input_tokens):
            failures.append(
                f"repeat {index + 1}: input_tokens {input_tokens} exceeds "
                f"{args.expected_max_input_tokens}")
        if (args.expected_min_output_tokens is not None
                and output_tokens < args.expected_min_output_tokens):
            failures.append(
                f"repeat {index + 1}: output_tokens {output_tokens} is below "
                f"{args.expected_min_output_tokens}")
        if (args.expected_response_status is not None
                and row.get("response_status") != args.expected_response_status):
            failures.append(
                f"repeat {index + 1}: response status "
                f"{row.get('response_status')!r}, expected "
                f"{args.expected_response_status!r}")
        if (args.expected_output_budget_source is not None
                and row.get("output_budget_source")
                != args.expected_output_budget_source):
            failures.append(
                f"repeat {index + 1}: output budget source "
                f"{row.get('output_budget_source')!r}, expected "
                f"{args.expected_output_budget_source!r}")
        if (args.expected_embedding_status is not None
                and embedding_status != args.expected_embedding_status):
            failures.append(
                f"repeat {index + 1}: embedding status {embedding_status!r}, "
                f"expected {args.expected_embedding_status!r}")
        delta_events = int(row.get("output_text_delta_events", 0) or 0)
        if (args.expected_min_text_deltas is not None
                and delta_events < args.expected_min_text_deltas):
            failures.append(
                f"repeat {index + 1}: received {delta_events} text deltas, "
                f"expected at least {args.expected_min_text_deltas}")
        if args.stream and row.get("streamed_text_matches_final") is not True:
            failures.append(
                f"repeat {index + 1}: streamed text does not match final output_text")
        if row.get("virtual_search_marker_exposed") is True:
            failures.append(
                f"repeat {index + 1}: virtual search marker reached public output")
        if row.get("hidden_gateway_marker_exposed") is True:
            failures.append(
                f"repeat {index + 1}: hidden gateway marker reached public output")
        output_types = row.get("output_types") or []
        if (args.expected_output_type is not None
                and args.expected_output_type not in output_types):
            failures.append(
                f"repeat {index + 1}: output types {output_types!r} do not "
                f"include {args.expected_output_type!r}")
        selection = row.get("tool_selection") or {}
        if (args.expected_gateway_phase is not None
                and selection.get("gateway_phase") != args.expected_gateway_phase):
            failures.append(
                f"repeat {index + 1}: gateway phase "
                f"{selection.get('gateway_phase')!r}, expected "
                f"{args.expected_gateway_phase!r}")
        if (args.expected_execution_outcome is not None
                and selection.get("gateway_execution_outcome")
                != args.expected_execution_outcome):
            failures.append(
                f"repeat {index + 1}: execution outcome "
                f"{selection.get('gateway_execution_outcome')!r}, expected "
                f"{args.expected_execution_outcome!r}")
        if after.available_bytes < int(args.min_available_gb * 1e9):
            failures.append(
                f"repeat {index + 1}: available memory fell below safety floor")
        if after.swap_used_bytes - initial.swap_used_bytes > int(
                args.max_swap_growth_mb * 1e6):
            failures.append(f"repeat {index + 1}: swap usage grew beyond limit")
        if after.swap_out_bytes - initial.swap_out_bytes > int(
                args.max_swap_growth_mb * 1e6):
            failures.append(f"repeat {index + 1}: swap-outs grew beyond limit")

    report = {
        "gate": "qwen3-large-agent-private-replay-v1",
        "capture": {
            "label": known_label, "sha256": digest, "bytes": len(raw),
            "tools": capture_tools,
        },
        "request": {
            "stream": args.stream,
            "max_output_tokens": (
                None if args.omit_max_output_tokens else args.max_output_tokens),
            "max_output_tokens_omitted": args.omit_max_output_tokens,
            "repeats": args.repeats,
            "scenario": args.scenario,
            "kai_conversation_sha256": (
                hashlib.sha256(args.kai_conversation.read_bytes()).hexdigest()
                if args.kai_conversation is not None else None),
            "request_snapshots": len(payloads),
            "replacement_user_sha256": replacement_sha256,
        },
        "initial_pressure": asdict(initial),
        "runs": rows,
        "failures": failures,
        "passed": not failures,
    }
    _write(args.result_json, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
