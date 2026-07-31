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
sys.path.insert(0, str(ROOT))

from runtime.profiles import (discover_runtime_profiles,
                              resolve_runtime_profiles,
                              runtime_profile_dirs)

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
    parser.add_argument(
        "--runtime-profile", action="append", default=[],
        help="server runtime profile; repeat to layer groups in order")
    parser.add_argument(
        "--profile-settings-only", action="store_true",
        help="do not duplicate profile settings through environment overrides")
    parser.add_argument("--expected-function", required=True)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument(
        "--backend", choices=("auto", "voom", "mlx-lm"), default="mlx-lm")
    parser.add_argument("--lossy-suffix-prefill")
    parser.add_argument("--qwen-moe-expert-top-k", default="released")
    parser.add_argument(
        "--grammar-jump-forward-lossy", choices=("0", "1"), default="0")
    parser.add_argument("--qwen35-weight-cache-mb", type=int)
    parser.add_argument(
        "--qwen-mtp-speculative", choices=("auto", "0", "1"), default="0")
    parser.add_argument("--qwen-mtp-min-output-tokens", type=int, default=32)
    parser.add_argument("--expected-function-arguments-json")
    parser.add_argument(
        "--expected-positive-function-argument", action="append", default=[])
    parser.add_argument("--temperature", action="append")
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--server-ready-timeout", type=float, default=30.0)
    parser.add_argument("--max-cold-seconds", type=float, default=60.0)
    parser.add_argument("--max-warm-seconds", type=float, default=30.0)
    parser.add_argument("--min-cached-tokens", type=int, default=5_000)
    parser.add_argument("--min-available-gb", type=float, default=3.2)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    parser.add_argument("--system-reserve-mb", type=int, default=1_500)
    parser.add_argument("--persistent-prompt-cache-dir", type=Path)
    parser.add_argument(
        "--persistent-prompt-cache-max-mb", type=int, default=1_000)
    parser.add_argument("--expected-first-cache-source")
    parser.add_argument("--expected-first-output-sha256")
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
    if args.profile_settings_only and not args.runtime_profile:
        parser.error("--profile-settings-only requires --runtime-profile")
    if min(
        args.request_timeout, args.server_ready_timeout,
        args.max_cold_seconds, args.max_warm_seconds,
        args.min_available_gb, args.max_swap_growth_mb,
        args.system_reserve_mb, args.persistent_prompt_cache_max_mb,
        args.qwen_mtp_min_output_tokens,
    ) <= 0 or args.min_cached_tokens <= 0:
        parser.error("timeouts, thresholds, and safety limits must be positive")

    server_overrides = {
        "VMODEL_RESIDENT_BACKEND": args.backend,
        "VMODEL_MLX_LM_PROMPT_CACHE": "1",
        "VMODEL_MLX_LM_LOGIT_CHAIN": "1",
        "VMODEL_MLX_LM_NATIVE_MTP": "0",
        "VMODEL_MLX_LM_SYSTEM_RESERVE_MB": str(args.system_reserve_mb),
        "VMODEL_FAST_TOOL_GATEWAY": "1",
        "VMODEL_FAST_TOOL_GATEWAY_HOST_ROUTE": "1",
        "VMODEL_FAST_TOOL_GATEWAY_ABSTAIN": "0",
        "VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT": "full",
        "VMODEL_FAST_TOOL_GATEWAY_QWEN_MOE_TOP_K": "released",
        "VMODEL_GRAMMAR_JUMP_FORWARD_LOSSY":
            args.grammar_jump_forward_lossy,
    }
    if args.lossy_suffix_prefill:
        server_overrides["VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL"] = (
            args.lossy_suffix_prefill)
    if args.backend == "voom":
        server_overrides["VMODEL_QWEN35_POSTGEN_MIN_AVAILABLE_MB"] = str(
            int(args.min_available_gb * 1_000))
    server_overrides["VMODEL_QWEN_MOE_EXPERT_TOP_K"] = (
        args.qwen_moe_expert_top_k)
    if args.qwen35_weight_cache_mb is not None:
        server_overrides["VMODEL_QWEN35_WEIGHT_CACHE_MB"] = str(
            args.qwen35_weight_cache_mb)
    server_overrides["VMODEL_QWEN_MTP_SPECULATIVE"] = (
        args.qwen_mtp_speculative)
    server_overrides["VMODEL_QWEN_MTP_MIN_OUTPUT_TOKENS"] = str(
        args.qwen_mtp_min_output_tokens)
    expected_backend = (
        "mlx-lm" if args.backend == "mlx-lm" else "voom")
    expected_repeat_cache_source = (
        "hot-prompt-exact" if expected_backend == "mlx-lm" else "memory")
    if args.persistent_prompt_cache_dir is not None:
        server_overrides.update({
            "VMODEL_MLX_LM_PERSISTENT_PROMPT_CACHE_DIR": str(
                args.persistent_prompt_cache_dir),
            "VMODEL_MLX_LM_PERSISTENT_PROMPT_CACHE_MAX_MB": str(
                args.persistent_prompt_cache_max_mb),
        })
    if args.profile_settings_only:
        server_overrides = {}
    profile_groups: tuple[str, ...] = ()
    profile_settings: dict[str, str] = {}
    if args.runtime_profile:
        catalog = discover_runtime_profiles(runtime_profile_dirs())
        profile_groups, profile_settings = resolve_runtime_profiles(
            tuple(args.runtime_profile), catalog)
    report = {
        "schema": "voom.qwen-agent-temperature-matrix.v1",
        "capture_path": str(args.capture),
        "model_override": args.model,
        "temperatures": list(temperatures),
        "server_environment_overrides": server_overrides,
        "runtime_profiles": list(args.runtime_profile),
        "runtime_profile_groups": list(profile_groups),
        "profile_settings_only": args.profile_settings_only,
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
            "first_cache_source": args.expected_first_cache_source,
            "first_output_sha256": args.expected_first_output_sha256,
            "repeat_cache_source": expected_repeat_cache_source,
            "repeat_cached_tokens_at_least": args.min_cached_tokens,
            "true_peak_metal_gb_strictly_below": 8.5,
            "in_run_available_gb_at_least": args.min_available_gb,
            "max_swap_growth_mb": args.max_swap_growth_mb,
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
        if args.profile_settings_only:
            for setting_name in profile_settings:
                server_env.pop(setting_name, None)
        server_env.update(server_overrides)
        print(
            f"[matrix] starting fresh server temperature={temperature:g}",
            flush=True)
        server_command = [
            sys.executable, "-m", "runtime.server", "--port", str(args.port),
        ]
        for profile_name in args.runtime_profile:
            server_command.extend(("--profile", profile_name))
        server = subprocess.Popen(
            server_command,
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
                "--expected-backend", expected_backend,
                "--expected-max-first-wall-seconds",
                str(args.max_cold_seconds),
                "--expected-max-repeat-wall-seconds",
                str(args.max_warm_seconds),
                "--expected-repeat-cache-source",
                expected_repeat_cache_source,
                "--expected-min-repeat-cached-tokens",
                str(args.min_cached_tokens),
                "--expected-max-peak-metal-gb", "8.5",
                "--expected-gateway-real-tool-required", "true",
                "--min-available-gb", str(args.min_available_gb),
                "--max-swap-growth-mb", str(args.max_swap_growth_mb),
                "--result-json", str(child_result),
            ]
            for profile_name in args.runtime_profile:
                command.extend(("--expected-runtime-profile", profile_name))
            for profile_name in profile_groups:
                command.extend((
                    "--expected-runtime-profile-group", profile_name))
            if args.profile_settings_only:
                command.append("--expected-no-runtime-profile-overrides")
            if args.expected_first_cache_source is not None:
                command.extend([
                    "--expected-first-cache-source",
                    args.expected_first_cache_source,
                ])
            if args.expected_function_arguments_json is not None:
                command.extend([
                    "--expected-function-arguments-json",
                    args.expected_function_arguments_json,
                ])
            for argument_name in args.expected_positive_function_argument:
                command.extend([
                    "--expected-positive-function-argument", argument_name,
                ])
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
            if failure is None and args.expected_first_cache_source == (
                    "persistent-prompt-exact"):
                first = (fixture_report.get("runs") or [{}])[0]
                timing = first.get("timing") or {}
                if (
                    timing.get("resident_persistent_prompt_cache_hit") != 1
                    or timing.get("resident_persistent_prompt_cache_error") != 0
                    or float(timing.get(
                        "resident_persistent_prompt_cache_load_s", 0.0)) <= 0
                    or int(timing.get(
                        "resident_persistent_prompt_cache_bytes", 0)) <= 0
                ):
                    failure = (
                        f"temperature {temperature:g}: first request lacked "
                        "a clean checksummed persistent-cache hit witness")
            if (failure is None
                    and args.expected_first_output_sha256 is not None):
                first = (fixture_report.get("runs") or [{}])[0]
                if first.get("output_sha256") != (
                        args.expected_first_output_sha256):
                    failure = (
                        f"temperature {temperature:g}: first output SHA "
                        f"{first.get('output_sha256')!r}, expected "
                        f"{args.expected_first_output_sha256!r}")
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
