#!/usr/bin/env python3
"""Exercise one Qwen serving configuration across heterogeneous request shapes.

This is the anti-overfit companion to the real Plex temperature matrix.  It
uses two identity-pinned real captures plus tracked synthetic mutations that
vary subject, tool count, system length, developer-role presence, tool-result
history, and streaming mode.  Every mutation is named in the resulting
artifact; none of these rows is reported as an unmodified-capture benchmark.
The corpus gates correctness, exact-hit/miss behavior, and a loose 60-second
runaway bound; only the separate Plex matrix enforces the requested <30-second
warm SLA because arbitrary direct answers may legitimately emit far more
tokens than the captured tool call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from qwen_agent_temperature_matrix_gate import (
    _atomic_json,
    _port_is_free,
    _stop_server,
    _wait_for_release,
    _wait_ready,
)


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "tests/fixtures/qwen3_large_agent_replay_gate.py"


def _case_command(
    *, capture: Path, model: str, port: int, result: Path,
    temperature: float, repeats: int, output_type: str,
    first_cache_source: str, stream_mode: str,
    scenario: str | None = None, function_name: str | None = None,
    min_cached_tokens: int = 20, require_real_tool: bool | None = None,
    max_warm_seconds: float = 60.0, gateway_phase: str | None = None,
    max_output_tokens: int | None = None,
    response_status: str | None = "completed",
) -> list[str]:
    command = [
        sys.executable, str(REPLAY), str(capture),
        "--url", f"http://127.0.0.1:{port}/v1/responses",
        "--model", model,
        "--repeats", str(repeats),
        "--temperature", format(temperature, "g"),
        "--timeout", "180",
        "--expected-output-type", output_type,
        "--expected-backend", "mlx-lm",
        "--expected-max-first-wall-seconds", "60",
        "--expected-max-repeat-wall-seconds", str(max_warm_seconds),
        "--expected-first-cache-source", first_cache_source,
        "--expected-max-peak-metal-gb", "8.5",
        "--min-available-gb", "3.2",
        "--max-swap-growth-mb", "16",
        "--result-json", str(result),
    ]
    if max_output_tokens is None:
        command.append("--omit-max-output-tokens")
    else:
        command.extend([
            "--max-output-tokens", str(max_output_tokens),
        ])
    if response_status is not None:
        command.extend(["--expected-response-status", response_status])
    if repeats > 1:
        command.extend([
            "--expected-repeat-cache-source", "hot-prompt-exact",
            "--expected-min-repeat-cached-tokens", str(min_cached_tokens),
        ])
    if stream_mode == "preserved":
        command.append("--preserve-stream")
    elif stream_mode == "forced":
        command.append("--stream")
    elif stream_mode != "disabled":
        raise ValueError(f"unknown stream mode: {stream_mode}")
    if scenario is not None:
        command.extend(["--scenario", scenario])
    if function_name is not None:
        command.extend(["--expected-function-call-name", function_name])
    if require_real_tool is not None:
        command.extend([
            "--expected-gateway-real-tool-required",
            "true" if require_real_tool else "false",
        ])
    if gateway_phase is not None:
        command.extend(["--expected-gateway-phase", gateway_phase])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plex-capture", required=True, type=Path)
    parser.add_argument("--direct-capture", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8130)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")
    if not _port_is_free(args.port):
        raise SystemExit(f"port {args.port} is already in use")

    server_overrides = {
        "VMODEL_RESIDENT_BACKEND": "mlx-lm",
        "VMODEL_MLX_LM_PROMPT_CACHE": "1",
        "VMODEL_FAST_TOOL_GATEWAY": "1",
        "VMODEL_FAST_TOOL_GATEWAY_HOST_ROUTE": "1",
        "VMODEL_FAST_TOOL_GATEWAY_ABSTAIN": "0",
        "VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT": "full",
        "VMODEL_FAST_TOOL_GATEWAY_QWEN_MOE_TOP_K": "released",
        "VMODEL_GRAMMAR_JUMP_FORWARD_LOSSY": "0",
    }
    cases = [
        {
            "name": "real-direct-long-system-132-tools-stream",
            "capture": args.direct_capture,
            "temperature": 0.0,
            "repeats": 2,
            "output_type": "message",
            "first_cache_source": "cold",
            "stream_mode": "preserved",
            "gateway_phase": "direct",
            "min_cached_tokens": 1_000,
            "declared_shape": {
                "capture": "identity-pinned-real",
                "conversation": "unchanged",
                "tools": 132,
                "system": "long",
                "developer": False,
                "stream": True,
                "subject": "direct-humor",
            },
        },
        {
            "name": "same-real-direct-nonstream",
            "capture": args.direct_capture,
            "temperature": 0.0,
            "repeats": 1,
            "output_type": "message",
            # Wire streaming does not affect prompt/model state, so this must
            # reuse the exact state left by the preceding streamed request.
            "first_cache_source": "hot-prompt-exact",
            "stream_mode": "disabled",
            "gateway_phase": "direct",
            "declared_shape": {
                "capture": "identity-pinned-real",
                "conversation": "unchanged",
                "tools": 132,
                "system": "long",
                "developer": False,
                "stream": False,
                "subject": "direct-humor",
            },
        },
        {
            "name": "short-system-developer-two-workspace-tools",
            "capture": args.plex_capture,
            "temperature": 0.7,
            "repeats": 2,
            "output_type": "function_call",
            "first_cache_source": "cold",
            "stream_mode": "disabled",
            "scenario": "developer-action",
            "function_name": "mastra_workspace_list_files",
            "min_cached_tokens": 100,
            "require_real_tool": True,
            "declared_shape": {
                "capture": "tracked-synthetic-mutation",
                "conversation": "developer-action",
                "tools": 2,
                "system": "short",
                "developer": True,
                "stream": False,
                "subject": "workspace-inspection",
            },
        },
        {
            "name": "short-system-no-tools-stream",
            "capture": args.plex_capture,
            "temperature": 1.0,
            "repeats": 2,
            "output_type": "message",
            "first_cache_source": "cold",
            "stream_mode": "forced",
            "scenario": "short-direct-no-tools",
            "min_cached_tokens": 20,
            "declared_shape": {
                "capture": "tracked-synthetic-mutation",
                "conversation": "short-direct-no-tools",
                "tools": 0,
                "system": "short",
                "developer": False,
                "stream": True,
                "subject": "direct-humor",
            },
        },
        {
            "name": "long-system-tool-result-answer-134-tools",
            "capture": args.plex_capture,
            "temperature": 0.0,
            "repeats": 2,
            "output_type": "message",
            "first_cache_source": "cold",
            "stream_mode": "disabled",
            "scenario": "tool-result-answer",
            "gateway_phase": "direct",
            # This row is already a declared synthetic conversation mutation
            # and gates role/history behavior, not latency. Bound generation so
            # a verbose direct answer cannot monopolize the single Metal slot.
            "max_output_tokens": 128,
            "response_status": None,
            "min_cached_tokens": 1_000,
            "declared_shape": {
                "capture": "tracked-synthetic-mutation",
                "conversation": "tool-result-answer",
                "tools": 134,
                "system": "long",
                "developer": False,
                "stream": False,
                "subject": "workspace-tool-result",
                "max_output_tokens": (
                    "128 synthetic runaway bound; not a latency result"),
            },
        },
        {
            "name": "long-system-confirmed-action-134-tools",
            "capture": args.plex_capture,
            "temperature": 0.3,
            "repeats": 2,
            "output_type": "function_call",
            "first_cache_source": "cold",
            "stream_mode": "disabled",
            "scenario": "deferred-action",
            "function_name": "mastra_workspace_execute_command",
            "min_cached_tokens": 1_000,
            "require_real_tool": True,
            "declared_shape": {
                "capture": "tracked-synthetic-mutation",
                "conversation": "deferred-action",
                "tools": 134,
                "system": "long",
                "developer": False,
                "stream": False,
                "subject": "confirmed-workspace-action",
            },
        },
    ]
    report = {
        "schema": "voom.qwen-agent-generality-corpus.v1",
        "model_override": args.model,
        "server_environment_overrides": server_overrides,
        "cases": [],
        "failures": [],
        "passed": False,
    }

    _wait_for_release(6.0)
    environment = os.environ.copy()
    environment.update(server_overrides)
    server = subprocess.Popen(
        [sys.executable, "-m", "runtime.server", "--port", str(args.port)],
        cwd=ROOT,
        env=environment,
    )
    try:
        _wait_ready(server, args.port, 30.0)
        for index, case in enumerate(cases):
            child_result = args.result_json.with_name(
                f"{args.result_json.stem}.case{index + 1}.json")
            if child_result.exists():
                failure = f"stale case artifact exists: {child_result}"
                report["failures"].append(f"{case['name']}: {failure}")
                report["cases"].append({
                    "name": case["name"], "failure": failure,
                    "declared_shape": case["declared_shape"],
                })
                continue
            print(f"[corpus] case={case['name']}", flush=True)
            command_args = {
                key: value for key, value in case.items()
                if key not in ("name", "declared_shape")
            }
            command = _case_command(
                model=args.model, port=args.port, result=child_result,
                **command_args)
            code = subprocess.run(command, cwd=ROOT).returncode
            child = (
                json.loads(child_result.read_text())
                if child_result.is_file() else None)
            failure = None
            if code != 0:
                failure = f"replay gate exited {code}"
            elif not isinstance(child, dict) or child.get("passed") is not True:
                failure = "replay artifact missing or failed"
            report["cases"].append({
                "name": case["name"],
                "declared_shape": case["declared_shape"],
                "fixture_result_path": str(child_result),
                "fixture_exit_code": code,
                "result": child,
                "failure": failure,
            })
            if failure:
                report["failures"].append(f"{case['name']}: {failure}")
            _atomic_json(args.result_json, report)
    finally:
        _stop_server(server)
        _wait_for_release(6.0)

    report["passed"] = (
        not report["failures"]
        and len(report["cases"]) == len(cases)
        and all(case.get("failure") is None for case in report["cases"]))
    _atomic_json(args.result_json, report)
    print(
        f"[corpus] {'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}",
        flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
