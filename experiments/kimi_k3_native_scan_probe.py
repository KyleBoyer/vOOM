#!/usr/bin/env python3
"""Benchmark vOOM's fused KDA scan at K3's released attention shape."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.kimi_linear import (
    _compiled_kda_prefill_scan,
    _native_fused_kda_prefill_scan,
)


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


def _reference(q, k, v, gate, beta, state):
    outputs = []
    for position in range(q.shape[1]):
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


def _measure(call, warmups: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        output, state = call()
        mx.eval(output, state)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output, state = call()
        mx.eval(output, state)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compiled-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=191)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(f"refusing existing result: {args.result_json}")
    if min(
        args.length, args.heads, args.dim, args.repeats,
        args.compiled_chunk,
    ) <= 0:
        parser.error("shape and repeats must be positive")

    rng = np.random.default_rng(args.seed)
    shape = (1, args.length, args.heads, args.dim)
    q_np = rng.standard_normal(shape, dtype=np.float32)
    k_np = rng.standard_normal(shape, dtype=np.float32)
    q_np /= np.sqrt(np.sum(q_np * q_np, axis=-1, keepdims=True) + 1e-6)
    q_np *= args.dim ** -0.5
    k_np /= np.sqrt(np.sum(k_np * k_np, axis=-1, keepdims=True) + 1e-6)
    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(rng.standard_normal(shape, dtype=np.float32))
    gate = mx.array(rng.uniform(-5.0, -0.001, shape).astype(np.float32))
    beta = mx.array(rng.uniform(
        0.01, 0.99, (1, args.length, args.heads)).astype(np.float32))
    initial = mx.array(rng.standard_normal(
        (1, args.heads, args.dim, args.dim), dtype=np.float32) * 0.02)
    mx.eval(q, k, v, gate, beta, initial)

    reference_call = lambda: _reference(q, k, v, gate, beta, initial)
    compiled_call = lambda: _compiled_kda_prefill_scan(
        q, k, v, gate, beta, initial, segment=args.compiled_chunk)
    fused_call = lambda: _native_fused_kda_prefill_scan(
        q, k, v, gate, beta, initial)
    reference_out, reference_state = reference_call()
    compiled_out, compiled_state = compiled_call()
    fused_out, fused_state = fused_call()
    mx.eval(
        reference_out, reference_state,
        compiled_out, compiled_state,
        fused_out, fused_state,
    )
    compiled_output_error = float(mx.max(mx.abs(
        compiled_out - reference_out)))
    compiled_state_error = float(mx.max(mx.abs(
        compiled_state - reference_state)))
    output_error = float(mx.max(mx.abs(fused_out - reference_out)))
    state_error = float(mx.max(mx.abs(fused_state - reference_state)))

    reference_median, reference_samples = _measure(
        reference_call, args.warmups, args.repeats)
    compiled_median, compiled_samples = _measure(
        compiled_call, args.warmups, args.repeats)
    fused_median, fused_samples = _measure(
        fused_call, args.warmups, args.repeats)
    finite = all(np.isfinite(value) for value in (
        reference_median, compiled_median, fused_median,
        compiled_output_error, compiled_state_error,
        output_error, state_error))
    passed = finite and output_error < 2e-4 and state_error < 2e-4
    result = {
        "schema": "voom.kimi-k3-native-scan-probe.v1",
        "shape": {
            "batch": 1,
            "length": args.length,
            "heads": args.heads,
            "key_dim": args.dim,
            "value_dim": args.dim,
        },
        "dtype": "float32",
        "reference": {
            "median_seconds": reference_median,
            "samples_seconds": reference_samples,
        },
        "compiled_reference": {
            "chunk": args.compiled_chunk,
            "median_seconds": compiled_median,
            "samples_seconds": compiled_samples,
            "speedup": reference_median / compiled_median,
            "max_output_abs_error": compiled_output_error,
            "max_state_abs_error": compiled_state_error,
            "byte_identical_output": bool(mx.array_equal(
                compiled_out, reference_out)),
            "byte_identical_state": bool(mx.array_equal(
                compiled_state, reference_state)),
        },
        "fused": {
            "median_seconds": fused_median,
            "samples_seconds": fused_samples,
        },
        "speedup": reference_median / fused_median,
        "max_output_abs_error": output_error,
        "max_state_abs_error": state_error,
        "true_peak_metal_bytes": int(mx.get_peak_memory()),
        "verdict": "PASS" if passed else "FAIL",
    }
    _atomic_json(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
