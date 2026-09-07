#!/usr/bin/env python3
"""Bounded real-MLX host-owner lifetime diagnostic; fresh preflight required.

Exercises the actual spool/packed-reader conversion helpers with synthetic
buffers, never a checkpoint. Weak references prove Python/NumPy owner lifetime,
not native allocator reclamation or whole-model pressure causation. Native
self-task/Metal observations overlap and must not be added. No serving changes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def buffer_shape(owner_mib: int) -> tuple[int, int]:
    if type(owner_mib) is not int or not 1 <= owner_mib <= 32:
        raise ValueError("owner_mib must be an integer in 1..32")
    # Every row includes all 65536 possible FP16/BF16 bit patterns.
    return owner_mib * 8, 65536


class TrackedBuffer(bytearray):
    """Weak-referenceable synthetic stand-in for a direct-read byte buffer."""


def run_probe(owner_mib: int) -> dict:
    shape = buffer_shape(owner_mib)
    import mlx.core as mx
    import numpy as np
    import psutil

    from formats.packed import to_mx
    from runtime.engine import _restore_lossless_16bit_host_spool
    from runtime.process_memory_witness import sample_self_memory

    def snapshot():
        return {
            "process": sample_self_memory(),
            "metal_active_bytes": int(mx.get_active_memory()),
            "metal_cache_bytes": int(mx.get_cache_memory()),
            "metal_peak_bytes": int(mx.get_peak_memory()),
            "system_available_bytes": int(psutil.virtual_memory().available),
            "system_swap_used_bytes": int(psutil.swap_memory().used),
            "system_swap_out_bytes": int(psutil.swap_memory().sout),
            "atomic": False,
        }

    mx.random.seed(947)
    expected_rng = mx.random.uniform(shape=(32,))
    mx.eval(expected_rng)
    mx.random.seed(947)
    before = snapshot()
    rows = []
    peak = before["metal_peak_bytes"]
    for helper, representation, selection in (
        ("restore_spool", "BF16", "last_row"),
        ("restore_spool", "F16", "last_row"),
        ("restore_spool", "BF16", "strided_tail"),
        ("restore_spool", "F16", "strided_tail"),
        ("to_mx", "BF16", "last_row"),
        ("to_mx", "F16", "last_row"),
        ("to_mx", "F8_E4M3", "last_row"),
        ("to_mx", "F32", "last_row"),
    ):
        baseline = snapshot()
        owner_bytes = owner_mib * 1024 * 1024
        if helper == "restore_spool":
            owner = np.empty(shape, dtype=np.uint16)
            owner[:] = np.arange(65536, dtype=np.uint16)
            view = (owner[-1:] if selection == "last_row"
                    else owner[-4::2, ::2])
            target_shape = tuple(view.shape)
            logical_bytes = int(view.nbytes)
            expected_sha = hashlib.sha256(view.tobytes(order="C")).hexdigest()
        else:
            owner = TrackedBuffer(owner_bytes)
            fill = np.frombuffer(owner, dtype=np.uint16).reshape(shape)
            fill[:] = np.arange(65536, dtype=np.uint16)
            del fill
            view = memoryview(owner)[-131072:]
            itemsize = {"BF16": 2, "F16": 2, "F8_E4M3": 1, "F32": 4}[representation]
            target_shape = (1, len(view) // itemsize)
            logical_bytes = len(view)
            expected_sha = hashlib.sha256(view).hexdigest()
        ref = weakref.ref(owner)
        del owner
        # Positive control: dropping only the owner's name MUST leave its
        # backing alive while the view is present. No ref() result is stored.
        positive_control = ref() is not None
        allocated = snapshot()
        started = time.perf_counter()
        if helper == "restore_spool":
            result = _restore_lossless_16bit_host_spool(
                view, dtype=mx.bfloat16 if representation == "BF16" else mx.float16)
        else:
            result = to_mx({"dtype": representation, "shape": target_shape}, view)
        construct_seconds = time.perf_counter() - started
        del view
        owner_alive_before_explicit_eval = ref() is not None
        started = time.perf_counter()
        mx.eval(result)
        eval_seconds = time.perf_counter() - started
        owner_alive_after_eval = ref() is not None
        evaluated = snapshot()
        actual_sha = hashlib.sha256(
            np.asarray(result.view(mx.uint8)).tobytes(order="C")).hexdigest()
        # GC is a separate diagnostic stage, never hidden in construction or
        # claimed as a serving optimization. Keep the result alive through it.
        gc.collect()
        owner_alive_after_gc = ref() is not None
        checked = snapshot()
        observed_peak = max(s["metal_peak_bytes"] for s in (
            baseline, allocated, evaluated, checked))
        peak = max(peak, observed_peak)
        rows.append({
            "helper": helper, "representation": representation,
            "selection": selection, "owner_bytes": owner_bytes,
            "result_logical_bytes": logical_bytes, "shape": list(target_shape),
            "weakref_positive_control_passed": positive_control,
            "owner_alive_before_explicit_eval": owner_alive_before_explicit_eval,
            "owner_alive_after_eval": owner_alive_after_eval,
            "owner_alive_after_gc_with_result_live": owner_alive_after_gc,
            "construct_seconds": construct_seconds,
            "explicit_eval_seconds": eval_seconds,
            "expected_bits_sha256": expected_sha, "actual_bits_sha256": actual_sha,
            "bits_equal": actual_sha == expected_sha,
            "snapshots": {"baseline": baseline, "owner_allocated": allocated,
                          "result_evaluated_owner_dropped": evaluated,
                          "after_gc_result_still_live": checked},
        })
        del result, ref
        # Isolate cases; cleanup cost is NOT in the helper timings.
        mx.clear_cache()

    actual_rng = mx.random.uniform(shape=(32,))
    mx.eval(actual_rng)
    rng_equal = actual_rng.tolist() == expected_rng.tolist()
    after = snapshot()
    failures = []
    if not rng_equal:
        failures.append("MLX RNG changed")
    if not all(row["bits_equal"] and row["weakref_positive_control_passed"] for row in rows):
        failures.append("raw bits or lifetime positive control failed")
    samples = [before, after] + [s for row in rows for s in row["snapshots"].values()]
    if not all(s["process"]["available"] for s in samples):
        failures.append("native self-memory reader unavailable")
    if min(s["system_available_bytes"] for s in samples) < 5_300_000_000:
        failures.append("available memory below 5.3 GB")
    if max(s["system_swap_used_bytes"] for s in samples) - before["system_swap_used_bytes"] > 16_000_000:
        failures.append("swap-used growth above 16 MB")
    if max(s["system_swap_out_bytes"] for s in samples) - before["system_swap_out_bytes"] > 16_000_000:
        failures.append("swap-out growth above 16 MB")
    if peak > 8_500_000_000:
        failures.append("Metal peak above 8.5 GB")
    return {
        "schema": "voom.host-buffer-lifetime-probe.v1", "passed": not failures,
        "scope": "Synthetic bounded host-owner/bit diagnostic; no checkpoint or serving change",
        "whole_model_attribution_proven": False, "speed_win_claimed": False,
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "numpy")},
        "before": before, "after": after, "cases": rows,
        "mlx_rng_equal": rng_equal, "true_peak_metal_bytes": peak,
        "retention_observed_after_eval": any(row["owner_alive_after_eval"] for row in rows),
        "retention_observed_after_gc": any(row["owner_alive_after_gc_with_result_live"] for row in rows),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--owner-mib", type=int, default=16)
    args = parser.parse_args()
    buffer_shape(args.owner_mib)
    if args.result.exists():
        parser.error("refusing existing result")
    from qwen4_hot_boundary_http_probe import _atomic_write_private
    started = time.perf_counter()
    try:
        result = run_probe(args.owner_mib)
    except Exception as error:
        result = {"passed": False, "failures": [type(error).__name__ + ": " + str(error)]}
    result["wall_seconds"] = time.perf_counter() - started
    _atomic_write_private(args.result, result)
    print(json.dumps({k: result.get(k) for k in (
        "passed", "wall_seconds", "retention_observed_after_eval",
        "retention_observed_after_gc", "true_peak_metal_bytes", "failures")}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
