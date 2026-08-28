"""Bounded weights-free benchmark for Qwen4 exact-verifier BF16 GEMV."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.exact_verify_bf16 import exact_verify_bf16_matmul


SHAPES = (
    # name, verifier rows, input width, output width
    ("expert_gate_up", 4, 2560, 640),
    ("expert_down", 4, 640, 2560),
    ("router", 4, 2560, 512),
    ("hc_mix_up", 4, 320, 10240),
)


def _check_preflight(path: Path, max_age_s: float = 300.0) -> dict:
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise RuntimeError("memory preflight did not pass")
    end = report.get("end") or {}
    age = time.monotonic() - float(end.get("monotonic_s", -1))
    if age < 0 or age > max_age_s:
        raise RuntimeError(f"memory preflight is stale ({age:.1f}s)")
    return report


def _timed(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = fn()
        mx.eval(value)
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 3:
        raise ValueError("benchmark requires >=1 warmup and >=3 repeats")
    preflight = _check_preflight(args.preflight)

    rows = []
    for shape_index, (name, length, inputs, outputs) in enumerate(SHAPES):
        generator = np.random.default_rng(31_337 + shape_index)
        x = mx.array(generator.standard_normal((1, length, inputs))).astype(
            mx.bfloat16)
        weight = mx.array(generator.standard_normal((outputs, inputs))).astype(
            mx.bfloat16)

        def singleton():
            return mx.concatenate(
                [x[:, row:row + 1] @ weight.T for row in range(length)],
                axis=1,
            )

        def candidate():
            value = exact_verify_bf16_matmul(x, weight)
            if value is None:
                raise RuntimeError(f"kernel rejected benchmark shape {name}")
            return value

        reference = singleton()
        exact = candidate()
        mx.eval(reference, exact)
        equal = bool(mx.array_equal(reference, exact).item())
        if not equal:
            raise RuntimeError(f"byte equality failed for {name}")
        _timed(singleton, args.warmups)
        _timed(candidate, args.warmups)
        singleton_s = _timed(singleton, args.repeats)
        candidate_s = _timed(candidate, args.repeats)
        singleton_median = statistics.median(singleton_s)
        candidate_median = statistics.median(candidate_s)
        rows.append({
            "name": name,
            "length": length,
            "inputs": inputs,
            "outputs": outputs,
            "array_equal": equal,
            "singleton_median_s": singleton_median,
            "candidate_median_s": candidate_median,
            "speedup": singleton_median / candidate_median,
            "singleton_samples_s": singleton_s,
            "candidate_samples_s": candidate_s,
        })
        del x, weight, reference, exact
        mx.clear_cache()

    result = {
        "schema": "voom.qwen4-exact-verify-bf16-gemv-bench.v1",
        "preflight": str(args.preflight),
        "preflight_available_bytes": int(
            preflight["end"]["system_available_bytes"]),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "rows": rows,
        "all_array_equal": all(row["array_equal"] for row in rows),
        "all_faster": all(row["speedup"] > 1.0 for row in rows),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
