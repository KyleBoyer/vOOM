#!/usr/bin/env python3
"""Start a profile-only server and prove one real HTTP request uses it.

The child removes every setting supplied by the selected profile from its
inherited environment before server launch. This prevents an existing shell
variable from making a profile smoke test pass without exercising the saved
value. The emitted artifact contains output hashes/metrics and profile
identity, but never profile setting values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.profiles import (discover_runtime_profiles,
                              resolve_runtime_profiles,
                              runtime_profile_dirs)


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


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise TypeError("HTTP response is not a JSON object")
        return payload


def _post_json(url: str, value: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": f"HTTP {error.code}"}
        return error.code, payload


def _wait_ready(
    process: subprocess.Popen, port: int, timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before readiness with code {process.returncode}")
        try:
            return _get_json(url, 1.0)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError, TypeError) as error:
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


def _output_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output") or ():
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or ():
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _profile_failures(
    payload: dict,
    *,
    selected: tuple[str, ...],
    groups: tuple[str, ...],
    label: str,
) -> list[str]:
    failures = []
    if payload.get("vmodel_runtime_profiles") != list(selected):
        failures.append(
            f"{label}: selected profiles {payload.get('vmodel_runtime_profiles')!r}, "
            f"expected {list(selected)!r}")
    if payload.get("vmodel_runtime_profile_groups") != list(groups):
        failures.append(
            f"{label}: profile groups "
            f"{payload.get('vmodel_runtime_profile_groups')!r}, "
            f"expected {list(groups)!r}")
    if payload.get("vmodel_runtime_profile_overrides") not in (None, []):
        failures.append(
            f"{label}: unexpected profile overrides "
            f"{payload.get('vmodel_runtime_profile_overrides')!r}")
    for key in (
        "vmodel_runtime_profile_digest", "vmodel_runtime_effective_digest",
    ):
        digest = payload.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            failures.append(f"{label}: invalid {key} {digest!r}")
    if (payload.get("vmodel_runtime_profile_digest")
            != payload.get("vmodel_runtime_effective_digest")):
        failures.append(
            f"{label}: configured/effective profile digests differ without overrides")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8132)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--server-ready-timeout", type=float, default=60.0)
    parser.add_argument("--expected-output-text")
    parser.add_argument("--expected-backend")
    parser.add_argument("--expected-checkpoint")
    parser.add_argument(
        "--expected-weight-profile-contains", action="append", default=[])
    parser.add_argument("--expected-k3-schedule")
    parser.add_argument("--expected-k3-prefill-tile", type=int)
    parser.add_argument("--expected-k3-dense-tile", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=180.0)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-available-gb", type=float, default=1.2)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65_535 or args.temperature < 0
            or min(
                args.max_output_tokens, args.request_timeout,
                args.server_ready_timeout, args.max_wall_seconds,
                args.max_peak_metal_gb, args.min_available_gb,
                args.max_swap_growth_mb,
            ) <= 0):
        parser.error("port/ranges are invalid or thresholds are not positive")
    if not _port_is_free(args.port):
        raise SystemExit(f"port {args.port} is already in use")
    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")

    catalog = discover_runtime_profiles(runtime_profile_dirs())
    selected = tuple(args.profile)
    groups, settings = resolve_runtime_profiles(selected, catalog)
    server_env = os.environ.copy()
    for setting_name in settings:
        server_env.pop(setting_name, None)
    # CLI selection must win over an inherited selection too; remove it so the
    # telemetry proof has only the requested names.
    server_env.pop("VMODEL_PROFILE", None)
    command = [
        sys.executable, "-m", "runtime.server", "--port", str(args.port),
    ]
    for profile_name in selected:
        command.extend(("--profile", profile_name))

    process = subprocess.Popen(command, cwd=ROOT, env=server_env)
    failures: list[str] = []
    registry: dict = {}
    response: dict = {}
    status = None
    wall_seconds = None
    samples: list[Pressure] = []
    error_text = None
    try:
        registry = _wait_ready(process, args.port, args.server_ready_timeout)
        failures.extend(_profile_failures(
            registry, selected=selected, groups=groups, label="registry"))
        private_values = [
            value for value in settings.values()
            if "/" in value or "\\" in value
        ]
        registry_encoded = json.dumps(registry, sort_keys=True)
        if any(value in registry_encoded for value in private_values):
            failures.append("registry disclosed a path-valued profile setting")

        request = {
            "model": args.model,
            "input": args.prompt,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "stream": False,
        }
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _post_json,
                f"http://127.0.0.1:{args.port}/v1/responses",
                request,
                args.request_timeout,
            )
            while not future.done():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited during request with code {process.returncode}")
                samples.append(_pressure())
                time.sleep(0.25)
            status, response = future.result()
        wall_seconds = time.monotonic() - started
        samples.append(_pressure())
        failures.extend(_profile_failures(
            response, selected=selected, groups=groups, label="response"))
        if status != 200:
            failures.append(f"HTTP status {status}, expected 200")
        if response.get("error"):
            failures.append(f"response error: {response.get('error')!r}")
        if args.expected_backend is not None:
            if response.get("vmodel_backend") != args.expected_backend:
                failures.append(
                    f"backend {response.get('vmodel_backend')!r}, expected "
                    f"{args.expected_backend!r}")
        if args.expected_checkpoint is not None:
            if response.get("vmodel_checkpoint") != args.expected_checkpoint:
                failures.append(
                    f"checkpoint {response.get('vmodel_checkpoint')!r}, expected "
                    f"{args.expected_checkpoint!r}")
        weight_profile = str(response.get("vmodel_weight_profile", ""))
        for fragment in args.expected_weight_profile_contains:
            if fragment not in weight_profile:
                failures.append(
                    f"weight profile {weight_profile!r} lacks {fragment!r}")
        text = _output_text(response)
        if (args.expected_output_text is not None
                and text != args.expected_output_text):
            failures.append(
                f"output text {text!r}, expected {args.expected_output_text!r}")
        if wall_seconds >= args.max_wall_seconds:
            failures.append(
                f"wall {wall_seconds:.4f}s is not below {args.max_wall_seconds}s")
        timing = response.get("vmodel_timing") or {}
        if args.expected_k3_schedule is not None:
            actual = timing.get("kimi_k3_prefill_schedule")
            if actual != args.expected_k3_schedule:
                failures.append(
                    f"K3 schedule {actual!r}, expected "
                    f"{args.expected_k3_schedule!r}")
        if args.expected_k3_prefill_tile is not None:
            actual = timing.get("kimi_k3_prefill_tile_width")
            if actual != args.expected_k3_prefill_tile:
                failures.append(
                    f"K3 prefill tile {actual!r}, expected "
                    f"{args.expected_k3_prefill_tile}")
        if args.expected_k3_dense_tile is not None:
            actual = timing.get("kimi_k3_dense_mlp_tile_size")
            if actual != args.expected_k3_dense_tile:
                failures.append(
                    f"K3 dense tile {actual!r}, expected "
                    f"{args.expected_k3_dense_tile}")
        peak_bytes = int(timing.get("true_peak_metal_bytes", 0) or 0)
        if peak_bytes <= 0:
            failures.append("response lacks a positive true Metal peak")
        elif peak_bytes >= int(args.max_peak_metal_gb * 1e9):
            failures.append(
                f"true Metal peak {peak_bytes / 1e9:.4f}GB is not below "
                f"{args.max_peak_metal_gb}GB")
        response_encoded = json.dumps(response, sort_keys=True)
        if any(value in response_encoded for value in private_values):
            failures.append("response disclosed a path-valued profile setting")
        available_min = min(
            (sample.available_bytes for sample in samples), default=0)
        if available_min < int(args.min_available_gb * 1e9):
            failures.append(
                f"available-memory minimum {available_min / 1e9:.3f}GB is below "
                f"{args.min_available_gb}GB")
        if samples:
            swap_growth = samples[-1].swap_used_bytes - samples[0].swap_used_bytes
            swap_out_growth = samples[-1].swap_out_bytes - samples[0].swap_out_bytes
            if swap_growth > int(args.max_swap_growth_mb * 1e6):
                failures.append(
                    f"swap usage grew {swap_growth / 1e6:.3f}MB")
            if swap_out_growth > int(args.max_swap_growth_mb * 1e6):
                failures.append(
                    f"swap-outs grew {swap_out_growth / 1e6:.3f}MB")
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        failures.append(error_text)
    finally:
        _stop_server(process)

    text = _output_text(response)
    timing = response.get("vmodel_timing") or {}
    report = {
        "schema": "voom.runtime-profile-http-gate.v1",
        "profiles": list(selected),
        "profile_groups": list(groups),
        "profile_settings_source": "saved-profile-only",
        "profile_setting_count": len(settings),
        "profile_digest": response.get("vmodel_runtime_profile_digest"),
        "effective_digest": response.get("vmodel_runtime_effective_digest"),
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "prompt_bytes": len(args.prompt.encode()),
        "http_status": status,
        "response_status": response.get("status"),
        "checkpoint": response.get("vmodel_checkpoint"),
        "weight_profile": response.get("vmodel_weight_profile"),
        "backend": response.get("vmodel_backend"),
        "usage": response.get("usage"),
        "timing": timing,
        "wall_seconds": round(wall_seconds, 4) if wall_seconds is not None else None,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_bytes": len(text.encode()),
        "output_matches_expected": (
            None if args.expected_output_text is None
            else text == args.expected_output_text),
        "pressure_samples": len(samples),
        "available_min_bytes": min(
            (sample.available_bytes for sample in samples), default=None),
        "swap_used_growth_bytes": (
            samples[-1].swap_used_bytes - samples[0].swap_used_bytes
            if samples else None),
        "swap_out_growth_bytes": (
            samples[-1].swap_out_bytes - samples[0].swap_out_bytes
            if samples else None),
        "pressure_first": asdict(samples[0]) if samples else None,
        "pressure_last": asdict(samples[-1]) if samples else None,
        "server_exit_code": process.returncode,
        "exception": error_text,
        "failures": failures,
        "passed": not failures,
    }
    _atomic_json(args.result_json, report)
    print(
        f"[profile-gate] {'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
