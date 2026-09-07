#!/usr/bin/env python3
"""Bounded synthetic Metal backing-lifetime diagnostic; never a serving path.

Run only after a fresh passing 30-second memory preflight, under run_gate.
No model, disk tensor reads, global wiring changes, or undocumented Metal API.
Snapshots are non-atomic overlapping views, not unique physical RAM or I/O.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def weight_shape(mib):
    if type(mib) is not int or not 16 <= mib <= 256:
        raise ValueError("weight MiB must be an integer in [16, 256]")
    return (mib * 128, 4096)  # FP16/BF16: exactly mib * 2**20 bytes.


def snapshot(mx):
    import psutil
    from runtime.process_memory_witness import sample_self_memory
    from runtime.process_region_witness import sample_self_regions
    started = time.perf_counter()
    swap = psutil.swap_memory()
    result = {
        "process": sample_self_memory(), "regions": sample_self_regions(),
        "metal_active_bytes": int(mx.get_active_memory()),
        "metal_cache_bytes": int(mx.get_cache_memory()),
        "metal_peak_bytes": int(mx.get_peak_memory()),
        "system_available_bytes": int(psutil.virtual_memory().available),
        "system_swap_used_bytes": int(swap.used),
        "system_swap_out_bytes": int(swap.sout),
        "root_free_bytes": int(psutil.disk_usage('/').free),
        "atomic": False,
    }
    result['observation_seconds'] = time.perf_counter() - started
    return result


def pressure_failures(samples):
    """Existing hard acceptance thresholds; no synthetic-probe waiver."""
    if not samples:
        return ['no pressure observations']
    failures = []
    if not all(s['process']['available'] and s['regions']['coverage_complete']
               for s in samples):
        failures.append('native observation unavailable/incomplete')
    if min(s['system_available_bytes'] for s in samples) < 5_300_000_000:
        failures.append('available memory below 5.3 GB')
    if max(s['system_swap_used_bytes'] for s in samples) - samples[0]['system_swap_used_bytes'] > 16_000_000:
        failures.append('swap-used growth above 16 MB')
    if max(s['system_swap_out_bytes'] for s in samples) - samples[0]['system_swap_out_bytes'] > 16_000_000:
        failures.append('swap-out growth above 16 MB')
    if max(s['metal_peak_bytes'] for s in samples) > 8_500_000_000:
        failures.append('Metal peak above 8.5 GB')
    if min(s['root_free_bytes'] for s in samples) < 10_000_000_000:
        failures.append('root free below 10 GB')
    return failures


def drop_and_observe(holder, *, observe, clear_cache, marker, sleep=time.sleep,
                     clock=time.perf_counter):
    """Drop only fixture-owned weights. Idle controls precede the marker.

    This is NOT a trial of the previously rejected serving synchronization
    policy. A marker submits real, tiny arithmetic on the allocation's own
    stream; no claim of synchronizing all streams. Cache clearance is a
    separately timed diagnostic stage, not hidden in a model latency number.
    """
    if len(holder) != 1:
        raise ValueError('exactly one fixture-owned weight is required')
    stages, durations = {}, {}
    started = clock()
    holder.clear()
    durations['drop_last_reference'] = clock() - started
    stages['after_drop'] = observe()
    started = clock()
    clear_cache()
    durations['clear_allocator_cache'] = clock() - started
    stages['after_clear_cache'] = observe()
    for label, delay in (('idle_100ms', .1), ('idle_1s', 1.0)):
        started = clock()
        sleep(delay)
        durations[label] = clock() - started
        stages[label] = observe()
    started = clock()
    marker()
    durations['same_stream_marker'] = clock() - started
    stages['after_same_stream_marker'] = observe()
    return stages, durations


def run_probe(mib=256):
    shape = weight_shape(mib)
    import mlx.core as mx
    import numpy as np

    def bits(a):
        return hashlib.sha256(np.asarray(a.view(mx.uint8)).tobytes()).hexdigest()

    # Long-lived sentinels are independent of the released synthetic weight.
    raw = mx.array(np.arange(65536, dtype=np.uint16))
    sentinels = [raw.view(mx.float16), raw.view(mx.bfloat16),
                 mx.arange(1024, dtype=mx.float32).reshape(32, 32)]
    mx.eval(*sentinels)
    expected_bits = [bits(a) for a in sentinels]
    mx.random.seed(1639)
    expected_rng = mx.random.uniform(shape=(32,))
    mx.eval(expected_rng)
    expected_rng_bits = bits(expected_rng)
    del expected_rng
    mx.random.seed(1639)
    old_limit = mx.set_cache_limit(0)
    rows, all_samples = [], []
    failures = []
    try:
        mx.clear_cache()
        before = snapshot(mx)
        all_samples.append(before)
        for source in ('gpu_full', 'numpy_copy'):
            for dtype_name in ('float16', 'bfloat16'):
                for stream_name in ('default', 'isolated'):
                    # Abort before another allocation after any hard gate fails.
                    baseline = snapshot(mx)
                    all_samples.append(baseline)
                    failures = pressure_failures(all_samples)
                    if failures:
                        break
                    stream = (mx.default_stream(mx.gpu) if stream_name == 'default'
                              else mx.new_stream(mx.gpu))
                    dtype = getattr(mx, dtype_name)
                    source_ref = None
                    marker_input = mx.array([3.0, 7.0], dtype=mx.float32)
                    mx.eval(marker_input)
                    started = time.perf_counter()
                    with mx.stream(stream):
                        if source == 'gpu_full':
                            # full alone may remain a broadcast scalar view.
                            weight = mx.contiguous(mx.full(shape, 1.0, dtype=dtype))
                        else:
                            # Construct released-format bits directly; no cast
                            # through a second large MLX tensor or float32 owner.
                            one_bits = 0x3C00 if dtype_name == 'float16' else 0x3F80
                            host = np.full(shape, one_bits, dtype=np.uint16)
                            source_ref = weakref.ref(host)
                            weight = mx.array(host).view(dtype)
                            del host
                        vector = mx.ones((shape[1], 1), dtype=dtype)
                        output = weight @ vector
                        mx.eval(weight, output)
                    construction_eval_seconds = time.perf_counter() - started
                    host_owner_alive = source_ref() is not None if source_ref else None
                    # All products/sums are exact: 4096 * 1. Check every row.
                    output_ok = bool(np.all(np.asarray(output.astype(mx.float32)) == 4096))
                    del vector, output
                    allocated = snapshot(mx)
                    holder = [weight]
                    del weight
                    retained = snapshot(mx)  # positive control: holder owns weight.

                    def marker():
                        with mx.stream(stream):
                            value = marker_input + 1.0
                            mx.eval(value)
                        if value.tolist() != [4.0, 8.0]:
                            raise AssertionError('marker arithmetic changed')

                    stages, durations = drop_and_observe(
                        holder, observe=lambda: snapshot(mx),
                        clear_cache=mx.clear_cache, marker=marker)
                    stages = {'before': baseline, 'allocated': allocated,
                              'retained_positive_control': retained, **stages}
                    all_samples.extend(stages.values())
                    positive_control = (
                        retained['metal_active_bytes'] - baseline['metal_active_bytes']
                        >= mib * 1024 * 1024)
                    row = {
                        'source': source, 'dtype': dtype_name, 'stream': stream_name,
                        'shape': list(shape), 'weight_bytes': mib * 1024 * 1024,
                        'cache_limit_bytes': 0,
                        'construction_and_eval_seconds': construction_eval_seconds,
                        'all_output_rows_equal_4096': output_ok,
                        'host_owner_alive_after_eval': host_owner_alive,
                        'retained_owner_positive_control': positive_control,
                        'stages': stages, 'stage_seconds': durations,
                    }
                    rows.append(row)
                    if not output_ok or not positive_control or host_owner_alive:
                        failures.append('value or ownership positive control failed')
                    del holder, marker_input, stream, marker
                    if failures or pressure_failures(all_samples):
                        break
                if failures or pressure_failures(all_samples):
                    break
            if failures or pressure_failures(all_samples):
                break
        actual_bits = [bits(a) for a in sentinels]
        actual_rng = mx.random.uniform(shape=(32,))
        mx.eval(actual_rng)
        rng_equal = bits(actual_rng) == expected_rng_bits
        after = snapshot(mx)
        all_samples.append(after)
    finally:
        mx.clear_cache()
        mx.set_cache_limit(old_limit)
    failures.extend(pressure_failures(all_samples))
    if len(rows) != 8:
        failures.append('not all eight cases completed')
    if actual_bits != expected_bits or not rng_equal:
        failures.append('sentinel raw bits or global RNG changed')
    return {
        'schema': 'voom.metal-resource-lifetime-probe.v1',
        'scope': 'Synthetic 8-case allocation lifetime diagnostic, no checkpoint',
        'passed': not failures, 'failures': sorted(set(failures)),
        'speed_win_claimed': False, 'whole_model_attribution_proven': False,
        'cache_limit_restored': True, 'serving_policy_changed': False,
        'versions': {n: importlib.metadata.version(n) for n in ('mlx', 'numpy')},
        'before': before, 'after': after, 'cases': rows,
        'sentinel_bits_equal': actual_bits == expected_bits, 'mlx_rng_equal': rng_equal,
        'expected_sentinel_sha256': expected_bits, 'actual_sentinel_sha256': actual_bits,
        'true_peak_metal_bytes': max(s['metal_peak_bytes'] for s in all_samples),
        'observed_min_available_bytes': min(s['system_available_bytes'] for s in all_samples),
        'max_swap_used_growth_bytes': max(s['system_swap_used_bytes'] for s in all_samples) - before['system_swap_used_bytes'],
        'actual_swap_out_growth_bytes': max(s['system_swap_out_bytes'] for s in all_samples) - before['system_swap_out_bytes'],
        'mapping_sums_are_unique_physical_bytes': False,
        'sampled_not_uninterrupted_pressure': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--weight-mib', type=int, default=256)
    args = parser.parse_args()
    weight_shape(args.weight_mib)
    if args.result.exists():
        parser.error('refusing existing result')
    from qwen4_hot_boundary_http_probe import _atomic_write_private
    started = time.perf_counter()
    try:
        result = run_probe(args.weight_mib)
    except Exception as error:
        result = {'passed': False, 'failures': [type(error).__name__ + ': ' + str(error)]}
    result['wall_seconds'] = time.perf_counter() - started
    _atomic_write_private(args.result, result)
    print(json.dumps({k: result.get(k) for k in (
        'passed', 'wall_seconds', 'true_peak_metal_bytes', 'observed_min_available_bytes',
        'max_swap_used_growth_bytes', 'actual_swap_out_growth_bytes', 'failures')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
