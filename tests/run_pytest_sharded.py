#!/usr/bin/env python3
"""Run pytest modules sequentially in fresh, memory-bounded processes.

MLX keeps allocator state for the lifetime of a Python process.  A monolithic
pytest invocation can therefore retain device allocations across otherwise
independent modules and make a healthy test suite drive macOS into swap.  This
runner preserves serial execution while giving every module a fresh allocator.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def _test_modules(repo: Path, requested: list[str]) -> list[Path]:
    if requested:
        modules = [Path(item).resolve() for item in requested]
    else:
        modules = sorted((repo / "tests").glob("test_*.py"))
    return [path for path in modules if path.is_file()]


def _wait_for_headroom(min_available_bytes: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while psutil.virtual_memory().available < min_available_bytes:
        if time.monotonic() >= deadline:
            return False
        time.sleep(2.0)
    return True


def _stop_process_group(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _run_shard(
    command: list[str],
    repo: Path,
    env: dict[str, str],
    min_available_bytes: int,
    max_swap_growth_bytes: int,
) -> tuple[int, str | None]:
    """Run one module and terminate its whole tree if RAM crosses the floor."""
    initial_swap_used = int(psutil.swap_memory().used)
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        start_new_session=True,
    )
    try:
        while proc.poll() is None:
            available = psutil.virtual_memory().available
            if available < min_available_bytes:
                _stop_process_group(proc)
                return 2, f"available memory fell to {available / 1e9:.2f} GB"
            swap_growth = max(
                0, int(psutil.swap_memory().used) - initial_swap_used)
            if swap_growth > max_swap_growth_bytes:
                _stop_process_group(proc)
                return 2, f"swap occupancy grew by {swap_growth / 1e6:.1f} MB"
            time.sleep(1.0)
    except BaseException:
        _stop_process_group(proc)
        raise
    return int(proc.returncode or 0), None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="test modules; default: tests/test_*.py")
    parser.add_argument("--min-available-gb", type=float, default=4.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=16.0)
    parser.add_argument("--headroom-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--pytest-arg", action="append", default=[])
    args = parser.parse_args()
    if args.min_available_gb < 0 or args.max_swap_growth_mb < 0:
        parser.error("memory thresholds must be nonnegative")
    if args.headroom_timeout_seconds < 0:
        parser.error("headroom timeout must be nonnegative")

    repo = Path(__file__).resolve().parents[1]
    modules = _test_modules(repo, args.paths)
    if not modules:
        parser.error("no test modules found")
    min_available = int(args.min_available_gb * 1_000_000_000)
    max_swap_growth = int(args.max_swap_growth_mb * 1_000_000)
    failures: list[tuple[Path, int]] = []

    for index, module in enumerate(modules, 1):
        if not _wait_for_headroom(min_available, args.headroom_timeout_seconds):
            available = psutil.virtual_memory().available
            print(
                f"[shard {index}/{len(modules)}] REFUSED: only "
                f"{available / 1e9:.2f} GB available before {module.name}",
                flush=True,
            )
            return 2
        print(f"[shard {index}/{len(modules)}] {module.name}", flush=True)
        env = os.environ.copy()
        returncode, refusal = _run_shard(
            [sys.executable, "-m", "pytest", "-q", str(module), *args.pytest_arg],
            repo,
            env,
            min_available,
            max_swap_growth,
        )
        if refusal is not None:
            print(
                f"[shard {index}/{len(modules)}] TERMINATED: {refusal}",
                flush=True,
            )
            return 2
        if returncode:
            failures.append((module, returncode))

    if failures:
        print("failed shards:", flush=True)
        for module, returncode in failures:
            print(f"  {module}: pytest exit {returncode}", flush=True)
        return 1
    print(f"all {len(modules)} pytest shards passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
