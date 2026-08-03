#!/usr/bin/env python3
"""Gate K3 startup prewarm and restart restore through the real HTTP server.

The captured Kai requests are replayed byte-for-byte except for two declared
model-harness substitutions: ``model`` selects ``lossy-Kimi-K3`` and
``max_output_tokens`` is bounded for the timing gate. Messages, tools,
temperature, streaming mode, and every other captured field are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _read_capture(path: Path, expected: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"capture {path} SHA-256 {actual} != {expected}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"capture {path} is not a JSON object")
    return raw, value


def _preflight(path: Path) -> dict:
    command = [
        sys.executable, "-m", "runtime.memory_preflight",
        "--workspace", str(ROOT), "--sample-seconds", "30",
        "--min-root-free-gb", "10", "--result", str(path),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    report = json.loads(path.read_text())
    if completed.returncode:
        raise RuntimeError(
            f"memory preflight deferred server start: {report.get('reasons')}")
    return report


def _wait_ready(process: subprocess.Popen, port: int, timeout: float) -> float:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"server exited before readiness with code {code}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return time.perf_counter() - started
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"server did not become ready within {timeout}s")


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def _post_responses(port: int, capture: dict, *, model: str,
                    max_output_tokens: int, timeout: float) -> dict:
    request = dict(capture)
    original_model = request.get("model")
    original_max = request.get("max_output_tokens", "omitted")
    request["model"] = model
    request["max_output_tokens"] = max_output_tokens
    payload = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=timeout)
    started = time.perf_counter()
    connection.request(
        "POST", "/v1/responses", body=payload,
        headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    body = response.read()
    wall = time.perf_counter() - started
    connection.close()
    completed = None
    for line in body.splitlines():
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[len(b"data: "):])
        if event.get("type") in ("response.completed", "response.incomplete"):
            completed = event.get("response")
        if event.get("type") == "response.failed":
            raise RuntimeError(f"HTTP generation failed: {event.get('response')}")
    if response.status != 200 or not isinstance(completed, dict):
        diagnostic = body[:4096].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP replay returned {response.status} without completion "
            f"({len(body)} bytes): {diagnostic}")
    return {
        "http_status": response.status,
        "wire_bytes": len(body),
        "wall_seconds": wall,
        "request_mutations": {
            "model": {"from": original_model, "to": model},
            "max_output_tokens": {
                "from": original_max, "to": max_output_tokens},
            "messages": "preserved",
            "tools": "preserved",
            "temperature": "preserved",
            "stream": "preserved",
        },
        "response": completed,
    }


def _direct_phase(replay: dict) -> dict:
    phases = replay["response"].get("vmodel_cache_phases") or []
    if len(phases) != 1:
        raise RuntimeError(f"expected one direct inference phase, got {phases}")
    return phases[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--profile", default="kimi-k3-this-mac-fast-tier")
    parser.add_argument("--prewarm-prefixes", required=True, type=Path)
    parser.add_argument("--first-capture", required=True, type=Path)
    parser.add_argument("--first-sha256", required=True)
    parser.add_argument("--restart-capture", required=True, type=Path)
    parser.add_argument("--restart-sha256", required=True)
    parser.add_argument("--model", default="lossy-Kimi-K3")
    parser.add_argument("--max-output-tokens", type=int, default=2)
    parser.add_argument("--port", type=int, default=18077)
    parser.add_argument("--startup-timeout", type=float, default=10800)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--max-restart-startup-seconds", type=float, default=600)
    parser.add_argument("--max-first-token-seconds", type=float, default=180)
    parser.add_argument("--require-cold-first", action="store_true")
    args = parser.parse_args()
    if args.result.exists():
        parser.error(f"refusing existing result: {args.result}")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.first_capture.resolve() == args.restart_capture.resolve():
        parser.error("HTTP replay captures must be distinct")

    first_raw, first_capture = _read_capture(
        args.first_capture, args.first_sha256)
    restart_raw, restart_capture = _read_capture(
        args.restart_capture, args.restart_sha256)
    run_root = args.result.resolve().with_suffix("")
    run_root.mkdir(parents=True, exist_ok=True)
    runs = []
    captures = (
        ("first", first_capture, first_raw),
        ("restart", restart_capture, restart_raw),
    )
    for index, (label, capture, raw) in enumerate(captures, start=1):
        preflight_path = run_root / f"{index}-{label}-preflight.json"
        preflight = _preflight(preflight_path)
        prewarm_result = run_root / f"{index}-{label}-prewarm.json"
        log_path = run_root / f"{index}-{label}-server.log"
        environment = dict(os.environ)
        environment["VMODEL_PREWARM_PREFIXES"] = str(
            args.prewarm_prefixes.resolve())
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "runtime.server",
                    "--port", str(args.port),
                    "--profile", args.profile,
                    "--prewarm-result", str(prewarm_result),
                ],
                cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                startup_wall = _wait_ready(
                    process, args.port, args.startup_timeout)
                replay = _post_responses(
                    args.port, capture, model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.request_timeout)
            finally:
                _stop_server(process)
        prewarm = json.loads(prewarm_result.read_text())
        runs.append({
            "label": label,
            "capture_sha256": hashlib.sha256(raw).hexdigest(),
            "capture_bytes": len(raw),
            "preflight": preflight,
            "startup_wall_seconds": startup_wall,
            "prewarm": prewarm,
            "http": replay,
            "server_log": str(log_path),
        })

    first_prefix = runs[0]["prewarm"]["prefixes"][0]
    restart_prefix = runs[1]["prewarm"]["prefixes"][0]
    first_phase = _direct_phase(runs[0]["http"])
    restart_phase = _direct_phase(runs[1]["http"])
    prefix_tokens = int(first_prefix["prefix"]["tokens"])
    gates = {
        "first_http_reused_startup_prefix": (
            first_phase.get("cache_source") == "memory"
            and int(first_phase.get("cached_tokens", 0)) == prefix_tokens),
        # K3 defers large hybrid-state loading until its first exact lookup so
        # engine construction does not transiently own two copies.  Both eager
        # preload and an exact hot_disk lookup prove cross-process durability.
        "restart_loaded_durable_prefix": bool(
            restart_prefix.get("restored_before_prewarm")
            or restart_prefix.get("cache_source") == "hot_disk"),
        "restart_http_reused_prefix": (
            restart_phase.get("cache_source") == "memory"
            and int(restart_phase.get("cached_tokens", 0)) == prefix_tokens),
        "restart_startup_within_limit": (
            runs[1]["startup_wall_seconds"]
            <= args.max_restart_startup_seconds),
        "first_tokens_within_limit": all(
            float(run["http"]["response"]["vmodel_timing"][
                "first_token_seconds"]) <= args.max_first_token_seconds
            for run in runs),
        "metal_at_most_8_5gb": all(
            int(run["http"]["response"]["vmodel_timing"].get(
                "true_peak_metal_bytes", 0)) <= 8_500_000_000
            for run in runs),
        "distinct_captures": args.first_sha256 != args.restart_sha256,
    }
    if args.require_cold_first:
        gates["first_start_was_cold"] = (
            first_prefix.get("cache_source") == "cold"
            and not first_prefix.get("restored_before_prewarm"))
    verdict = "PASS" if all(gates.values()) else "FAIL"
    output = {
        "schema": "voom.kimi-k3-http-replay-restart-gate.v1",
        "profile": args.profile,
        "model_substitution": args.model,
        "max_output_tokens_substitution": args.max_output_tokens,
        "prewarm_prefixes": str(args.prewarm_prefixes.resolve()),
        "runs": runs,
        "gates": gates,
        "verdict": verdict,
    }
    _atomic_json(args.result.resolve(), output)
    print(json.dumps({
        "verdict": verdict,
        "prefix_tokens": prefix_tokens,
        "first_startup_seconds": runs[0]["startup_wall_seconds"],
        "restart_startup_seconds": runs[1]["startup_wall_seconds"],
        "first_http_seconds": runs[0]["http"]["wall_seconds"],
        "restart_http_seconds": runs[1]["http"]["wall_seconds"],
        "first_phase": first_phase,
        "restart_phase": restart_phase,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
