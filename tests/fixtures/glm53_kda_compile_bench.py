#!/usr/bin/env python3
"""Exact released-geometry timing gate for GLM-5.3 compiled KDA prefill."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np

from runtime.kimi_linear import _compiled_kda_prefill_scan


def _reference(q, k, v, gate, beta, initial_state):
    state = initial_state
    outputs = []
    for position in range(int(q.shape[1])):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * mx.exp(gate[:, position])[..., None]
        predicted = mx.sum(k_t[..., None] * state, axis=-2)
        residual = v_t - predicted
        state = state + (
            beta[:, position, :, None] * k_t
        )[..., None] * residual[..., None, :]
        outputs.append(mx.sum(q_t[..., None] * state, axis=-2))
        if (position + 1) % 32 == 0:
            mx.eval(state)
    return mx.stack(outputs, axis=1), state


def _sha256(value) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _measure(fn, args, repeats: int) -> dict:
    samples = []
    peaks = []
    output_sha = state_sha = None
    for _ in range(repeats):
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output, state = fn(*args)
        mx.eval(output, state)
        samples.append(time.perf_counter() - started)
        peaks.append(int(mx.get_peak_memory()))
        output_sha = _sha256(output)
        state_sha = _sha256(state)
    return {
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "peak_bytes": max(peaks),
        "output_sha256": output_sha,
        "state_sha256": state_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--segments", default="16,32,64,128",
        help="comma-separated compiled segment widths to compare",
    )
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    if args.length <= 1 or args.repeats <= 0:
        parser.error("length must exceed one and repeats must be positive")
    try:
        segments = tuple(dict.fromkeys(
            int(item.strip()) for item in args.segments.split(",")
            if item.strip()))
    except ValueError as error:
        parser.error(f"segments must be comma-separated integers: {error}")
    if not segments or any(segment <= 0 for segment in segments):
        parser.error("segments must contain positive integers")
    if 32 not in segments:
        parser.error("segments must include the released baseline width 32")
    if args.result.exists():
        parser.error("result already exists")

    rng = np.random.default_rng(53128)
    shape = (1, args.length, 64, 128)
    q = mx.array(rng.standard_normal(shape, dtype=np.float32))
    k = mx.array(rng.standard_normal(shape, dtype=np.float32))
    v = mx.array(rng.standard_normal(shape, dtype=np.float32))
    gate = mx.array(rng.uniform(-5, 0, shape).astype(np.float32))
    beta = mx.array(rng.uniform(
        0, 1, (1, args.length, 64)).astype(np.float32))
    state = mx.array(rng.standard_normal(
        (1, 64, 128, 128), dtype=np.float32))
    values = (q, k, v, gate, beta, state)

    reference = _measure(_reference, values, args.repeats)
    compiled = {}
    for segment in segments:
        candidate = functools.partial(
            _compiled_kda_prefill_scan, segment=segment)
        # Compile once outside the measured steady-state samples. A real model
        # amortizes this graph across 34 KDA layers and hundreds of same-width
        # tiles; the result still reports only steady-state recurrence time.
        warm_output, warm_state = candidate(*values)
        mx.eval(warm_output, warm_state)
        measurement = _measure(candidate, values, args.repeats)
        measurement["byte_identical"] = (
            reference["output_sha256"] == measurement["output_sha256"]
            and reference["state_sha256"] == measurement["state_sha256"]
        )
        measurement["speedup_vs_reference"] = (
            reference["median_s"] / measurement["median_s"]
            if measurement["median_s"] else None)
        compiled[str(segment)] = measurement
    exact_segments = [
        segment for segment in segments
        if compiled[str(segment)]["byte_identical"]
    ]
    winner = min(
        exact_segments,
        key=lambda segment: compiled[str(segment)]["median_s"],
        default=None,
    )
    baseline_exact = compiled["32"]["byte_identical"]
    result = {
        "schema": "voom.glm53-kda-compile-segment-sweep.v2",
        "geometry": {
            "batch": 1,
            "length": args.length,
            "heads": 64,
            "head_dim": 128,
            "dtype": "float32",
            "reference_state_eval_segment": 32,
            "candidate_segments": list(segments),
        },
        "reference": reference,
        "compiled": compiled,
        "byte_identical_segments": exact_segments,
        "fastest_byte_identical_segment": winner,
        "baseline_segment_32_byte_identical": baseline_exact,
        "passed": baseline_exact,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if baseline_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
