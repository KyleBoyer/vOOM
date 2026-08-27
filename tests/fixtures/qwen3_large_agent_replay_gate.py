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

DEVELOPER_ACTION_INPUT = [
    {
        "role": "system",
        "content": "You are a tool-using assistant. Never guess live workspace state.",
    },
    {
        "role": "developer",
        "content": (
            "When the user asks to inspect the workspace, call the most "
            "relevant available workspace tool."),
    },
    {
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "List the files in the current workspace root now.",
        }],
    },
]

SHORT_DIRECT_NO_TOOLS_INPUT = [
    {
        "role": "system",
        "content": "Answer directly and concisely.",
    },
    {
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "Tell me a one-sentence joke about databases.",
        }],
    },
]


def _tool_name(tool: dict) -> str:
    function = tool.get("function", tool) if isinstance(tool, dict) else {}
    return str(function.get("name", "")) if isinstance(function, dict) else ""


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
        "gateway_query_context_profile",
        "gateway_execution_choice_required", "gateway_real_tool_required",
        "gateway_abstention_available", "gateway_abstention_policy_reason",
        "gateway_execution_outcome",
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
             progress: list[dict], deltas: list[str],
             expected_function_arguments: dict | None = None,
             expected_positive_function_arguments: tuple[str, ...] = (),
             expected_nonempty_function_arguments: tuple[str, ...] = (),
             expected_output_text_terms: tuple[str, ...] = (),
             expected_output_text_any_terms: tuple[str, ...] = (),
             score_plex_profile: bool = False) -> dict:
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
    parsed_function_arguments = []
    parsed_function_calls = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except (TypeError, ValueError):
            continue
        if isinstance(arguments, dict):
            parsed_function_arguments.append(arguments)
            parsed_function_calls.append({
                "name": str(item.get("name") or ""),
                "arguments": arguments,
            })
    argument_match = (
        None if expected_function_arguments is None else any(
            all(arguments.get(key) == value
                for key, value in expected_function_arguments.items())
            for arguments in parsed_function_arguments)
    )
    positive_argument_match = (
        None if not expected_positive_function_arguments else any(
            all(
                isinstance(arguments.get(key), (int, float))
                and not isinstance(arguments.get(key), bool)
                and arguments[key] > 0
                for key in expected_positive_function_arguments
            )
            for arguments in parsed_function_arguments)
    )
    nonempty_argument_match = (
        None if not expected_nonempty_function_arguments else any(
            all(
                arguments.get(key) is not None
                and not isinstance(arguments.get(key), bool)
                and (
                    arguments[key] > 0
                    if isinstance(arguments[key], (int, float))
                    else bool(arguments[key])
                )
                for key in expected_nonempty_function_arguments
            )
            for arguments in parsed_function_arguments)
    )
    # Responses API servers are not required to synthesize the SDK's
    # top-level ``output_text`` convenience property. Build the semantic
    # witness from both that field and canonical message content, while still
    # persisting only the boolean below.
    private_text_parts = (
        [final_output_text]
        if isinstance(final_output_text, str) and final_output_text else []
    )
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                private_text_parts.append(text)
    folded_output_text = "\n".join(private_text_parts).casefold()
    visible_output_text = (
        final_output_text
        if isinstance(final_output_text, str) and final_output_text
        else "\n".join(private_text_parts)
    )
    plex_profile = None
    if score_plex_profile:
        # Import only when explicitly requested. The scorer returns static
        # rubric booleans/points and never the private response text or tool
        # arguments, preserving this fixture's artifact privacy contract.
        from plex_agent_profile import score_profile

        plex_profile = score_profile(
            parsed_function_calls,
            visible_output_text,
            str(response.get("vmodel_reasoning") or ""),
        )
    output_text_terms_match = (
        None if not expected_output_text_terms else all(
            term.casefold() in folded_output_text
            for term in expected_output_text_terms
        )
    )
    output_text_term_matches = (
        None if not expected_output_text_terms else [
            term.casefold() in folded_output_text
            for term in expected_output_text_terms
        ]
    )
    output_text_any_term_matches = (
        None if not expected_output_text_any_terms else [
            term.casefold() in folded_output_text
            for term in expected_output_text_any_terms
        ]
    )
    output_text_any_term_match = (
        None if output_text_any_term_matches is None
        else any(output_text_any_term_matches)
    )
    return {
        "http_status": 200,
        "response_status": response.get("status"),
        "error": response.get("error"),
        "wall_seconds": round(wall_s, 4),
        "usage": response.get("usage"),
        "cache_phases": response.get("vmodel_cache_phases"),
        "timing": response.get("vmodel_timing"),
        "execution_profile": response.get("vmodel_execution_profile"),
        "backend": response.get("vmodel_backend"),
        "backend_admission": response.get("vmodel_backend_admission"),
        "checkpoint": response.get("vmodel_checkpoint"),
        "weight_profile": response.get("vmodel_weight_profile"),
        "runtime_profiles": response.get("vmodel_runtime_profiles"),
        "runtime_profile_groups": response.get("vmodel_runtime_profile_groups"),
        "runtime_profile_digest": response.get("vmodel_runtime_profile_digest"),
        "runtime_effective_digest": response.get(
            "vmodel_runtime_effective_digest"),
        "runtime_profile_overrides": response.get(
            "vmodel_runtime_profile_overrides"),
        "max_output_tokens": response.get("vmodel_max_output_tokens"),
        "output_budget_source": response.get("vmodel_output_budget_source"),
        "tool_selection": _safe_selection(response.get("vmodel_tool_selection")),
        "output_types": [item.get("type") for item in output if isinstance(item, dict)],
        "function_call_names": [
            item.get("name") for item in output
            if isinstance(item, dict) and item.get("type") == "function_call"],
        # Boolean-only semantic witness: expected values are operator-supplied
        # and the private response arguments remain absent from the artifact.
        "function_call_arguments_match": argument_match,
        "function_call_positive_arguments_match": positive_argument_match,
        "function_call_nonempty_arguments_match": nonempty_argument_match,
        # Boolean only: terms are operator-supplied and neither the private
        # response text nor matched spans are written to the artifact.
        "output_text_terms_match": output_text_terms_match,
        "output_text_term_matches": output_text_term_matches,
        "output_text_any_term_match": output_text_any_term_match,
        "output_text_any_term_matches": output_text_any_term_matches,
        "plex_profile": plex_profile,
        "plex_profile_scope": (
            "single_response_whole_visible_output"
            if score_plex_profile else None),
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


def _post(
        url: str, payload: bytes, timeout: float, stream: bool,
        expected_function_arguments: dict | None = None,
        expected_positive_function_arguments: tuple[str, ...] = (),
        expected_nonempty_function_arguments: tuple[str, ...] = (),
        expected_output_text_terms: tuple[str, ...] = (),
        expected_output_text_any_terms: tuple[str, ...] = (),
        score_plex_profile: bool = False) -> dict:
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
                                "cache_source", "diagnostic", "subphase",
                                "layer", "completed_tokens",
                                "active_metal_bytes", "peak_metal_bytes",
                                "host_spool_bytes", "metal_limit_bytes")
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
    except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
        return {
            "http_status": 599,
            "response_status": None,
            "error": f"{type(error).__name__}: {error}",
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
        events=events, progress=progress, deltas=deltas,
        expected_function_arguments=expected_function_arguments,
        expected_positive_function_arguments=(
            expected_positive_function_arguments),
        expected_nonempty_function_arguments=(
            expected_nonempty_function_arguments),
        expected_output_text_terms=expected_output_text_terms,
        expected_output_text_any_terms=expected_output_text_any_terms,
        score_plex_profile=score_plex_profile)


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
    parser.add_argument(
        "--model",
        help=(
            "override only the request model after capture identity validation; "
            "the private prompt and tool catalog remain byte-for-byte unchanged"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument(
        "--temperature", type=float,
        help="override sampling temperature after capture identity validation")
    parser.add_argument(
        "--seed", type=int,
        help="override sampling seed after capture identity validation")
    parser.add_argument("--omit-max-output-tokens", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--preserve-stream", action="store_true",
        help="preserve the captured stream field instead of overriding it")
    parser.add_argument("--expected-selected-tools", type=int)
    parser.add_argument("--expected-max-input-tokens", type=int)
    parser.add_argument("--expected-embedding-status")
    parser.add_argument("--expected-min-text-deltas", type=int)
    parser.add_argument("--expected-output-type", choices=("message", "function_call"))
    parser.add_argument(
        "--expected-gateway-phase", choices=("direct", "search", "enable"))
    parser.add_argument("--expected-execution-outcome")
    parser.add_argument("--expected-min-output-tokens", type=int)
    parser.add_argument("--expected-function-call-name")
    parser.add_argument(
        "--expected-function-arguments-json",
        help=(
            "require at least one function call to contain this JSON-object "
            "subset; only a boolean match witness is persisted"))
    parser.add_argument(
        "--expected-positive-function-argument", action="append", default=[],
        help=(
            "require these argument names to have positive numeric values in "
            "one function call; only a boolean match witness is persisted"))
    parser.add_argument(
        "--expected-nonempty-function-argument", action="append", default=[],
        help=(
            "require these argument names to have nonempty values in one "
            "function call; only a boolean match witness is persisted"))
    parser.add_argument(
        "--expected-output-text-term", action="append", default=[],
        help=(
            "require every case-insensitive term in final output_text; only "
            "a boolean match witness is persisted"))
    parser.add_argument(
        "--expected-output-text-any-term", action="append", default=[],
        help=(
            "require at least one case-insensitive term in final output_text; "
            "only boolean match witnesses are persisted"))
    parser.add_argument(
        "--score-plex-profile", action="store_true",
        help=(
            "score each single response with the whole-visible Plex rubric; "
            "only the static rubric result is persisted, never response text "))
    parser.add_argument("--expected-min-plex-score", type=float)
    parser.add_argument(
        "--expected-plex-pass", action="store_true",
        help="require the whole-visible single-response Plex rubric to pass")
    parser.add_argument("--expected-backend")
    parser.add_argument(
        "--expected-runtime-profile", action="append", default=[])
    parser.add_argument(
        "--expected-runtime-profile-group", action="append", default=[])
    parser.add_argument(
        "--expected-no-runtime-profile-overrides", action="store_true")
    parser.add_argument("--expected-max-first-wall-seconds", type=float)
    parser.add_argument("--expected-max-repeat-wall-seconds", type=float)
    parser.add_argument("--expected-first-cache-source")
    parser.add_argument("--expected-repeat-cache-source")
    parser.add_argument("--expected-min-repeat-cached-tokens", type=int)
    parser.add_argument("--expected-max-peak-metal-gb", type=float)
    parser.add_argument(
        "--expected-gateway-real-tool-required",
        choices=("true", "false"))
    parser.add_argument(
        "--expected-response-status", choices=("completed", "incomplete"))
    parser.add_argument(
        "--expected-output-budget-source", choices=("request", "eos_safety_ceiling"))
    parser.add_argument(
        "--replacement-user-text",
        help="replace only the final user turn after capture identity validation")
    parser.add_argument(
        "--append-final-user-text",
        help=(
            "append text to the captured final user turn after identity "
            "validation; useful for a small cached-prefix extension gate"))
    parser.add_argument(
        "--scenario", choices=(
            "deferred-action", "tool-result-answer",
            "developer-action", "short-direct-no-tools"),
        help="replace conversation turns with a tracked regression scenario")
    parser.add_argument(
        "--scenario-user-text",
        help=(
            "replace the final user text inside a selected tracked scenario; "
            "the scenario's system/developer/tool shape remains unchanged"))
    parser.add_argument(
        "--kai-conversation", type=Path,
        help="replay every user boundary from this local Kai conversation")
    parser.add_argument("--min-available-gb", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.max_output_tokens <= 0 or args.timeout <= 0:
        parser.error("repeats, max-output-tokens, and timeout must be positive")
    if args.temperature is not None and args.temperature < 0:
        parser.error("temperature must be non-negative")
    expected_function_arguments = None
    if args.expected_function_arguments_json is not None:
        try:
            expected_function_arguments = json.loads(
                args.expected_function_arguments_json)
        except ValueError as error:
            parser.error(
                f"expected-function-arguments-json is invalid: {error}")
        if not isinstance(expected_function_arguments, dict):
            parser.error(
                "expected-function-arguments-json must decode to an object")
    if args.stream and args.preserve_stream:
        parser.error("stream and preserve-stream are mutually exclusive")
    if (args.expected_min_output_tokens is not None
            and args.expected_min_output_tokens <= 0):
        parser.error("expected-min-output-tokens must be positive")
    if args.expected_min_text_deltas is not None:
        if args.expected_min_text_deltas <= 0:
            parser.error("expected-min-text-deltas must be positive")
        if not args.stream:
            parser.error("expected-min-text-deltas requires --stream")
    for name, value in (
        ("expected-max-first-wall-seconds",
         args.expected_max_first_wall_seconds),
        ("expected-max-repeat-wall-seconds",
         args.expected_max_repeat_wall_seconds),
        ("expected-max-peak-metal-gb", args.expected_max_peak_metal_gb),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    if (args.expected_min_repeat_cached_tokens is not None
            and args.expected_min_repeat_cached_tokens <= 0):
        parser.error("expected-min-repeat-cached-tokens must be positive")
    if args.expected_min_plex_score is not None:
        if not 0 <= args.expected_min_plex_score <= 100:
            parser.error("expected-min-plex-score must be in [0, 100]")
        if not args.score_plex_profile:
            parser.error("expected-min-plex-score requires --score-plex-profile")
    if args.expected_plex_pass and not args.score_plex_profile:
        parser.error("expected-plex-pass requires --score-plex-profile")
    mutations = sum(value is not None for value in (
        args.replacement_user_text, args.append_final_user_text,
        args.scenario, args.kai_conversation))
    if mutations > 1:
        parser.error(
            "replacement-user-text, append-final-user-text, scenario, and "
            "kai-conversation are mutually exclusive")
    if args.scenario_user_text is not None and args.scenario is None:
        parser.error("scenario-user-text requires --scenario")
    if (args.scenario_user_text is not None
            and args.scenario not in (
                "developer-action", "short-direct-no-tools")):
        parser.error(
            "scenario-user-text is supported only for scenarios whose final "
            "item is a user message")

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
    if args.model is not None:
        if not args.model.strip():
            parser.error("model must not be empty")
        request_value["model"] = args.model
    replacement_sha256 = None
    final_user_append_sha256 = None
    if args.scenario is not None:
        inputs = request_value.get("input")
        if (not isinstance(inputs, list) or not inputs
                or inputs[-1].get("role") != "user"):
            raise SystemExit("capture has no replaceable conversation")
        if args.scenario == "deferred-action":
            scenario_turns = DEFERRED_ACTION_TURNS
            request_value["input"] = [*inputs[:-1], *scenario_turns]
        elif args.scenario == "tool-result-answer":
            scenario_turns = TOOL_RESULT_TURNS
            request_value["input"] = [*inputs[:-1], *scenario_turns]
        elif args.scenario == "developer-action":
            scenario_turns = json.loads(json.dumps(DEVELOPER_ACTION_INPUT))
            if args.scenario_user_text is not None:
                scenario_turns[-1]["content"] = [{
                    "type": "input_text", "text": args.scenario_user_text,
                }]
            request_value["input"] = scenario_turns
            retained = {
                "mastra_workspace_list_files",
                "mastra_workspace_read_file",
            }
            request_value["tools"] = [
                tool for tool in request_value.get("tools", ())
                if _tool_name(tool) in retained]
            if len(request_value["tools"]) != len(retained):
                raise SystemExit(
                    "developer-action scenario is missing workspace tools")
        else:
            scenario_turns = json.loads(json.dumps(SHORT_DIRECT_NO_TOOLS_INPUT))
            if args.scenario_user_text is not None:
                scenario_turns[-1]["content"] = [{
                    "type": "input_text", "text": args.scenario_user_text,
                }]
            request_value["input"] = scenario_turns
            request_value["tools"] = []
            request_value["tool_choice"] = "none"
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
    elif args.append_final_user_text is not None:
        inputs = request_value.get("input")
        if (not isinstance(inputs, list) or not inputs
                or inputs[-1].get("role") != "user"):
            raise SystemExit("capture has no appendable final user turn")
        content = inputs[-1].get("content")
        if not isinstance(content, list):
            raise SystemExit("capture final user content is not a parts list")
        text_part = next(
            (part for part in reversed(content)
             if isinstance(part, dict)
             and part.get("type") == "input_text"
             and isinstance(part.get("text"), str)),
            None,
        )
        if text_part is None:
            raise SystemExit("capture final user turn has no input_text part")
        text_part["text"] += args.append_final_user_text
        final_user_append_sha256 = hashlib.sha256(
            args.append_final_user_text.encode("utf-8")).hexdigest()
    if args.kai_conversation is not None:
        request_values = _kai_request_snapshots(
            request_value, args.kai_conversation)
    else:
        request_values = [
            (f"repeat_{index + 1}", dict(request_value))
            for index in range(args.repeats)]
    request_stream = (
        bool(request_value.get("stream"))
        if args.preserve_stream else bool(args.stream))
    payloads = []
    for label, value in request_values:
        if not args.preserve_stream:
            value["stream"] = request_stream
        if args.temperature is not None:
            value["temperature"] = args.temperature
        if args.seed is not None:
            value["seed"] = args.seed
        if args.omit_max_output_tokens:
            value.pop("max_output_tokens", None)
            value.pop("max_tokens", None)
        else:
            value["max_output_tokens"] = args.max_output_tokens
        if request_stream and not args.preserve_stream:
            value["vmodel_progress_events"] = True
        payloads.append((label, json.dumps(
            value, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")))

    initial = _pressure()
    rows = []
    failures = []
    for index, (label, payload) in enumerate(payloads):
        before = _pressure()
        row = _post(
            args.url, payload, args.timeout, request_stream,
            expected_function_arguments=expected_function_arguments,
            expected_positive_function_arguments=tuple(
                args.expected_positive_function_argument),
            expected_nonempty_function_arguments=tuple(
                args.expected_nonempty_function_argument),
            expected_output_text_terms=tuple(
                args.expected_output_text_term),
            expected_output_text_any_terms=tuple(
                args.expected_output_text_any_term),
            score_plex_profile=args.score_plex_profile)
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
        if request_stream and row.get("streamed_text_matches_final") is not True:
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
        function_names = row.get("function_call_names") or []
        if (args.expected_function_call_name is not None
                and args.expected_function_call_name not in function_names):
            failures.append(
                f"repeat {index + 1}: function calls {function_names!r} do not "
                f"include {args.expected_function_call_name!r}")
        if (
            expected_function_arguments is not None
            and row.get("function_call_arguments_match") is not True
        ):
            failures.append(
                f"repeat {index + 1}: function arguments did not contain "
                "the expected subset")
        if (
            args.expected_positive_function_argument
            and row.get("function_call_positive_arguments_match") is not True
        ):
            failures.append(
                f"repeat {index + 1}: function arguments did not contain "
                "the expected positive numeric fields")
        if (
            args.expected_nonempty_function_argument
            and row.get("function_call_nonempty_arguments_match") is not True
        ):
            failures.append(
                f"repeat {index + 1}: function arguments did not contain "
                "the expected nonempty fields")
        if (
            args.expected_output_text_term
            and row.get("output_text_terms_match") is not True
        ):
            failures.append(
                f"repeat {index + 1}: output text did not contain every "
                "expected term")
        if (
            args.expected_output_text_any_term
            and row.get("output_text_any_term_match") is not True
        ):
            failures.append(
                f"repeat {index + 1}: output text did not contain any "
                "accepted mechanism term")
        plex_profile = row.get("plex_profile") or {}
        if (args.expected_min_plex_score is not None
                and float(plex_profile.get("score", -1))
                < args.expected_min_plex_score):
            failures.append(
                f"repeat {index + 1}: Plex score "
                f"{plex_profile.get('score')!r} is below "
                f"{args.expected_min_plex_score}")
        if args.expected_plex_pass and plex_profile.get("passed") is not True:
            failures.append(
                f"repeat {index + 1}: whole-visible Plex rubric did not pass")
        if (args.expected_backend is not None
                and row.get("backend") != args.expected_backend):
            failures.append(
                f"repeat {index + 1}: backend {row.get('backend')!r}, "
                f"expected {args.expected_backend!r}")
        if (args.expected_runtime_profile
                and row.get("runtime_profiles")
                != args.expected_runtime_profile):
            failures.append(
                f"repeat {index + 1}: runtime profiles "
                f"{row.get('runtime_profiles')!r}, expected "
                f"{args.expected_runtime_profile!r}")
        if (args.expected_runtime_profile_group
                and row.get("runtime_profile_groups")
                != args.expected_runtime_profile_group):
            failures.append(
                f"repeat {index + 1}: runtime profile groups "
                f"{row.get('runtime_profile_groups')!r}, expected "
                f"{args.expected_runtime_profile_group!r}")
        if (args.expected_no_runtime_profile_overrides
                and row.get("runtime_profile_overrides") not in (None, [])):
            failures.append(
                f"repeat {index + 1}: unexpected runtime profile overrides "
                f"{row.get('runtime_profile_overrides')!r}")
        for digest_name in (
            "runtime_profile_digest", "runtime_effective_digest",
        ):
            if args.expected_runtime_profile:
                digest = row.get(digest_name)
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    failures.append(
                        f"repeat {index + 1}: invalid {digest_name} "
                        f"{digest!r}")
        wall_s = float(row.get("wall_seconds", float("inf")))
        if (index == 0
                and args.expected_max_first_wall_seconds is not None
                and wall_s >= args.expected_max_first_wall_seconds):
            failures.append(
                f"repeat 1: wall_seconds {wall_s} is not below "
                f"{args.expected_max_first_wall_seconds}")
        if (index > 0
                and args.expected_max_repeat_wall_seconds is not None
                and wall_s >= args.expected_max_repeat_wall_seconds):
            failures.append(
                f"repeat {index + 1}: wall_seconds {wall_s} is not below "
                f"{args.expected_max_repeat_wall_seconds}")
        timing = row.get("timing") or {}
        if (index == 0 and args.expected_first_cache_source is not None
                and timing.get("cache_source")
                != args.expected_first_cache_source):
            failures.append(
                f"repeat 1: cache source {timing.get('cache_source')!r}, "
                f"expected {args.expected_first_cache_source!r}")
        if (index > 0 and args.expected_repeat_cache_source is not None
                and timing.get("cache_source")
                != args.expected_repeat_cache_source):
            failures.append(
                f"repeat {index + 1}: cache source "
                f"{timing.get('cache_source')!r}, expected "
                f"{args.expected_repeat_cache_source!r}")
        cached_tokens = int(
            ((usage.get("input_tokens_details") or {}).get(
                "cached_tokens", 0)) or 0)
        if (index > 0
                and args.expected_min_repeat_cached_tokens is not None
                and cached_tokens < args.expected_min_repeat_cached_tokens):
            failures.append(
                f"repeat {index + 1}: cached_tokens {cached_tokens} is below "
                f"{args.expected_min_repeat_cached_tokens}")
        peak_bytes = int(timing.get("true_peak_metal_bytes", 0) or 0)
        if (args.expected_max_peak_metal_gb is not None
                and peak_bytes >= int(
                    args.expected_max_peak_metal_gb * 1_000_000_000)):
            failures.append(
                f"repeat {index + 1}: true peak Metal "
                f"{peak_bytes / 1e9:.4f}GB is not below "
                f"{args.expected_max_peak_metal_gb}GB")
        selection = row.get("tool_selection") or {}
        if args.expected_gateway_real_tool_required is not None:
            expected_required = (
                args.expected_gateway_real_tool_required == "true")
            if selection.get("gateway_real_tool_required") is not expected_required:
                failures.append(
                    f"repeat {index + 1}: gateway_real_tool_required "
                    f"{selection.get('gateway_real_tool_required')!r}, "
                    f"expected {expected_required!r}")
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
            # Recompute from immutable input bytes here.  The expectation loop
            # above also validates runtime profile/effective digests; it must
            # never be able to overwrite the capture identity persisted in the
            # proof artifact merely by reusing a local variable name.
            "label": known_label,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "tools": capture_tools,
        },
        "request": {
            "model_override": args.model,
            "temperature_override": args.temperature,
            "seed_override": args.seed,
            "stream": request_stream,
            "stream_preserved": args.preserve_stream,
            "max_output_tokens": (
                None if args.omit_max_output_tokens else args.max_output_tokens),
            "max_output_tokens_omitted": args.omit_max_output_tokens,
            "repeats": args.repeats,
            "scenario": args.scenario,
            "scenario_user_sha256": (
                hashlib.sha256(args.scenario_user_text.encode("utf-8")).hexdigest()
                if args.scenario_user_text is not None else None),
            "kai_conversation_sha256": (
                hashlib.sha256(args.kai_conversation.read_bytes()).hexdigest()
                if args.kai_conversation is not None else None),
            "request_snapshots": len(payloads),
            "replacement_user_sha256": replacement_sha256,
            "final_user_append_sha256": final_user_append_sha256,
            "effective_input_items": len(request_value.get("input") or ()),
            "effective_tool_count": len(request_value.get("tools") or ()),
            "effective_developer_messages": sum(
                item.get("role") == "developer"
                for item in (request_value.get("input") or ())
                if isinstance(item, dict)),
            "effective_system_chars": sum(
                len(str(item.get("content", "")))
                for item in (request_value.get("input") or ())
                if isinstance(item, dict) and item.get("role") == "system"),
            "score_plex_profile": args.score_plex_profile,
        },
        "expectations": {
            "max_first_wall_seconds": args.expected_max_first_wall_seconds,
            "max_repeat_wall_seconds": args.expected_max_repeat_wall_seconds,
            "first_cache_source": args.expected_first_cache_source,
            "repeat_cache_source": args.expected_repeat_cache_source,
            "min_repeat_cached_tokens": (
                args.expected_min_repeat_cached_tokens),
            "max_peak_metal_gb": args.expected_max_peak_metal_gb,
            "function_call_name": args.expected_function_call_name,
            "backend": args.expected_backend,
            "runtime_profiles": args.expected_runtime_profile,
            "runtime_profile_groups": args.expected_runtime_profile_group,
            "no_runtime_profile_overrides": (
                args.expected_no_runtime_profile_overrides),
            "gateway_real_tool_required": (
                args.expected_gateway_real_tool_required),
            "min_plex_score": args.expected_min_plex_score,
            "plex_pass": args.expected_plex_pass,
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
