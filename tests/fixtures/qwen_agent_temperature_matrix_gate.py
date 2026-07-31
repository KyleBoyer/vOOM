#!/usr/bin/env python3
"""Run a fresh-server cold/warm Qwen replay at each requested temperature.

The private request stays in ``logs/captured_requests``.  This orchestrator
starts a new server for every temperature so row one cannot inherit model or
prompt state from another temperature, delegates request identity and semantic
checks to ``qwen3_large_agent_replay_gate.py``, and atomically publishes one
matrix artifact.  It intentionally applies no prompt-, schema-, or
conversation-specific cache boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "tests/fixtures/qwen3_large_agent_replay_gate.py"
TEMPERATURES = (0.0, 0.3, 0.5, 0.7, 1.0)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _temperature_label(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_ready(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before readiness with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.25)
    raise TimeoutError(f"server readiness timed out: {last_error}")


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_release(min_available_gb: float, timeout: float = 30.0) -> None:
    floor = int(min_available_gb * 1_000_000_000)
    deadline = time.monotonic() + timeout
    while (psutil.virtual_memory().available < floor
           and time.monotonic() < deadline):
        time.sleep(0.25)
    available = int(psutil.virtual_memory().available)
    if available < floor:
        raise MemoryError(
            "system memory did not recover after the prior server: "
            f"available={available / 1e9:.3f}GB < {min_available_gb:.3f}GB")


def _parse_temperatures(values: list[str] | None) -> tuple[float, ...]:
    if not values:
        return TEMPERATURES
    parsed = tuple(float(value) for value in values)
    if any(value < 0 for value in parsed):
        raise ValueError("temperatures must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise ValueError("temperatures must be unique")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-function", required=True)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--temperature", action="append")
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--server-ready-timeout", type=float, default=30.0)
    parser.add_argument("--max-cold-seconds", type=float, default=60.0)
    parser.add_argument("--max-warm-seconds", type=float, default=30.0)
    parser.add_argument("--min-cached-tokens", type=int, default=5_000)
    parser.add_argument("--min-available-gb", type=float, default=3.2)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    args = parser.parse_args()
    try:
        temperatures = _parse_temperatures(args.temperature)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.port <= 65_535:
        parser.error("port must be in [1, 65535]")
    if not _port_is_free(args.port):
        raise SystemExit(f"port {args.port} is already in use")
    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")
    if min(
        args.request_timeout, args.server_ready_timeout,
        args.max_cold_seconds, args.max_warm_seconds,
        args.min_available_gb, args.max_swap_growth_mb,
    ) <= 0 or args.min_cached_tokens <= 0:
        parser.error("timeouts, thresholds, and safety limits must be positive")

    server_overrides = {
        "VMODEL_RESIDENT_BACKEND": "mlx-lm",
        "VMODEL_MLX_LM_PROMPT_CACHE": "1",
        "VMODEL_MLX_LM_LOGIT_CHAIN": "1",
        "VMODEL_MLX_LM_NATIVE_MTP": "0",
        "VMODEL_FAST_TOOL_GATEWAY": "1",
        "VMODEL_FAST_TOOL_GATEWAY_HOST_ROUTE": "1",
        "VMODEL_FAST_TOOL_GATEWAY_ABSTAIN": "0",
        "VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT": "full",
        "VMODEL_FAST_TOOL_GATEWAY_QWEN_MOE_TOP_K": "released",
        "VMODEL_GRAMMAR_JUMP_FORWARD_LOSSY": "0",
    }
    report = {
        "schema": "voom.qwen-agent-temperature-matrix.v1",
        "capture_path": str(args.capture),
        "model_override": args.model,
        "temperatures": list(temperatures),
        "server_environment_overrides": server_overrides,
        "request_mutations": {
            "model": args.model,
            "temperature_per_row": list(temperatures),
            "stream": "preserved",
            "max_output_tokens": "preserved-absent",
            "prompt": "unchanged",
            "tools": "unchanged",
            "messages": "unchanged",
        },
        "thresholds": {
            "cold_wall_seconds_strictly_below": args.max_cold_seconds,
            "warm_wall_seconds_strictly_below": args.max_warm_seconds,
            "repeat_cache_source": "hot-prompt-exact",
            "repeat_cached_tokens_at_least": args.min_cached_tokens,
            "true_peak_metal_gb_strictly_below": 8.5,
        },
        "runs": [],
        "failures": [],
        "passed": False,
    }

    for temperature in temperatures:
        label = _temperature_label(temperature)
        child_result = args.result_json.with_name(
            f"{args.result_json.stem}.temp{label}.json")
        preflight_result = args.result_json.with_name(
            f"{args.result_json.stem}.temp{label}.preflight.json")
        if child_result.exists():
            report["failures"].append(
                f"temperature {temperature:g}: stale result exists: "
                f"{child_result}")
            continue
        if preflight_result.exists():
            report["failures"].append(
                f"temperature {temperature:g}: stale preflight exists: "
                f"{preflight_result}")
            continue
        _wait_for_release(6.0)
        preflight_code = subprocess.run([
            sys.executable, "-m", "runtime.memory_preflight",
            "--result", str(preflight_result),
            "--sample-seconds", "30",
        ], cwd=ROOT).returncode
        preflight_report = (
            json.loads(preflight_result.read_text())
            if preflight_result.is_file() else None)
        if (preflight_code != 0
                or not isinstance(preflight_report, dict)
                or preflight_report.get("passed") is not True):
            failure = (
                f"temperature {temperature:g}: memory preflight failed")
            report["runs"].append({
                "temperature": temperature,
                "memory_preflight_exit_code": preflight_code,
                "memory_preflight_path": str(preflight_result),
                "memory_preflight": preflight_report,
                "fixture_exit_code": None,
                "fixture_result_path": str(child_result),
                "result": None,
                "failure": failure,
            })
            report["failures"].append(failure)
            _atomic_json(args.result_json, report)
            continue
        server_env = os.environ.copy()
        server_env.update(server_overrides)
        print(
            f"[matrix] starting fresh server temperature={temperature:g}",
            flush=True)
        server = subprocess.Popen(
            [sys.executable, "-m", "runtime.server",
             "--port", str(args.port)],
            cwd=ROOT,
            env=server_env,
        )
        fixture_code = None
        fixture_report = None
        failure = None
        try:
            _wait_ready(server, args.port, args.server_ready_timeout)
            command = [
                sys.executable, str(REPLAY), str(args.capture),
                "--url", f"http://127.0.0.1:{args.port}/v1/responses",
                "--model", args.model,
                "--repeats", "2",
                "--omit-max-output-tokens",
                "--temperature", format(temperature, "g"),
                "--preserve-stream",
                "--timeout", str(args.request_timeout),
                "--expected-output-type", "function_call",
                "--expected-function-call-name", args.expected_function,
                "--expected-response-status", "completed",
                "--expected-backend", "mlx-lm",
                "--expected-max-first-wall-seconds",
                str(args.max_cold_seconds),
                "--expected-max-repeat-wall-seconds",
                str(args.max_warm_seconds),
                "--expected-repeat-cache-source", "hot-prompt-exact",
                "--expected-min-repeat-cached-tokens",
                str(args.min_cached_tokens),
                "--expected-max-peak-metal-gb", "8.5",
                "--expected-gateway-real-tool-required", "true",
                "--min-available-gb", str(args.min_available_gb),
                "--max-swap-growth-mb", str(args.max_swap_growth_mb),
                "--result-json", str(child_result),
            ]
            fixture_code = subprocess.run(command, cwd=ROOT).returncode
            if child_result.is_file():
                fixture_report = json.loads(child_result.read_text())
            if fixture_code != 0:
                failure = (
                    f"temperature {temperature:g}: replay gate exited "
                    f"{fixture_code}")
            elif not isinstance(fixture_report, dict):
                failure = (
                    f"temperature {temperature:g}: missing replay artifact")
            elif fixture_report.get("passed") is not True:
                failure = (
                    f"temperature {temperature:g}: replay artifact failed")
        except Exception as error:
            failure = (
                f"temperature {temperature:g}: "
                f"{type(error).__name__}: {error}")
        finally:
            _stop_server(server)
            _wait_for_release(6.0)
        report["runs"].append({
            "temperature": temperature,
            "memory_preflight_exit_code": preflight_code,
            "memory_preflight_path": str(preflight_result),
            "memory_preflight": preflight_report,
            "fixture_exit_code": fixture_code,
            "fixture_result_path": str(child_result),
            "result": fixture_report,
            "failure": failure,
        })
        if failure:
            report["failures"].append(failure)
        _atomic_json(args.result_json, report)

    report["passed"] = (
        not report["failures"]
        and len(report["runs"]) == len(temperatures)
        and all(row.get("failure") is None for row in report["runs"]))
    _atomic_json(args.result_json, report)
    print(
        f"[matrix] {'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}",
        flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
