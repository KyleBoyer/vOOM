#!/usr/bin/env python3
"""Diagnostic-only Darwin host-cache relief with live MLX/host bit guards.

Requires a fresh 30-second preflight and run_gate; no production activation.
Four synthetic cases keep GPU weights alive during a 1s idle control and one
native relief call. A separate live-host negative control checks object safety.
The API's byte goal is best effort, NOT a hard work, latency, or release limit.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_goal(goal_bytes):
    if type(goal_bytes) is not int or not 1 <= goal_bytes <= 256 * 1024 * 1024:
        raise ValueError('goal must be a positive integer at most 256 MiB')
    return goal_bytes


def pressure_relief(goal_bytes, *, fn=None):
    """One documented self-process operation; never free a named live pointer."""
    validate_goal(goal_bytes)
    if fn is None:
        if platform.system() != 'Darwin':
            raise RuntimeError('Darwin allocator required')
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        fn = lib.malloc_zone_pressure_relief
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fn.restype = ctypes.c_size_t
    started = time.perf_counter()
    released = fn(None, goal_bytes)
    elapsed = time.perf_counter() - started
    if type(released) is not int or released < 0:
        raise RuntimeError('invalid native released-byte return')
    return {'api': 'malloc_zone_pressure_relief', 'zone': 'all_current_process_zones',
            'goal_bytes': goal_bytes, 'reported_released_bytes': released,
            'call_seconds': elapsed, 'goal_is_hard_cap': False,
            'physical_footprint_delta_is_api_return': False}


def abi_source():
    fmt = json.dumps('{"size_t_bytes":%zu,"pointer_bytes":%zu}')
    return ('#include <malloc/malloc.h>\n#include <stdio.h>\n'
            '_Static_assert(__builtin_types_compatible_p(__typeof__(&malloc_zone_pressure_relief),\n'
            ' size_t (*)(malloc_zone_t *, size_t)), "malloc relief signature mismatch");\n'
            'int main(void) { printf(' + fmt + ', sizeof(size_t), sizeof(void *));\n'
            ' return 0; }\n')


def verify_sdk():
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    header = Path(sdk) / 'usr/include/malloc/malloc.h'
    source = abi_source()
    with tempfile.TemporaryDirectory(prefix='malloc-relief-abi-', dir=ROOT/'logs') as directory:
        binary = Path(directory)/'oracle'
        subprocess.run(['xcrun', 'clang', '-x', 'c', '-', '-o', str(binary)],
                       input=source, text=True, capture_output=True, check=True)
        sizes = json.loads(subprocess.check_output([str(binary)], text=True))
    assert sizes == {'size_t_bytes': ctypes.sizeof(ctypes.c_size_t),
                     'pointer_bytes': ctypes.sizeof(ctypes.c_void_p)}
    return {'header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
            'oracle_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'signature_compile_assertion_passed': True, 'sizes': sizes}


def idle_then_relieve(goal_bytes, *, observe, relieve=pressure_relief,
                      sleep=time.sleep):
    """Idle comes first, with live GPU owners throughout; no clearance/GC/sync."""
    before = observe()
    started = time.perf_counter()
    sleep(1.0)
    idle_seconds = time.perf_counter() - started
    after_idle = observe()
    receipt = relieve(goal_bytes)
    after_relief = observe()
    return {'before_idle': before, 'after_idle': after_idle,
            'after_relief': after_relief}, {'idle_seconds': idle_seconds, **receipt}


def run_probe():
    from metal_resource_lifetime_probe import snapshot, pressure_failures, weight_shape
    import mlx.core as mx
    import numpy as np

    def bits(a):
        # SHA reads all live bytes, not sampled values. No tobytes() host copy.
        return hashlib.sha256(np.asarray(a.view(mx.uint8))).hexdigest()

    raw = mx.array(np.arange(65536, dtype=np.uint16))
    sentinels = [raw.view(mx.float16), raw.view(mx.bfloat16),
                 mx.arange(1024, dtype=mx.float32).reshape(32, 32)]
    mx.eval(*sentinels)
    expected_sentinel = [bits(a) for a in sentinels]
    mx.random.seed(8227)
    rng = mx.random.uniform(shape=(32,))
    mx.eval(rng)
    expected_rng = bits(rng)
    del rng
    mx.random.seed(8227)
    old_limit = mx.set_cache_limit(0)
    samples, cases, failures = [], [], []
    try:
        mx.clear_cache()
        before = snapshot(mx)
        samples.append(before)
        for mib, dtype_name in ((64, 'float16'), (64, 'bfloat16'),
                               (256, 'bfloat16'), (256, 'float16')):
            baseline = snapshot(mx)
            samples.append(baseline)
            failures.extend(pressure_failures(samples))
            if failures:
                break
            shape = weight_shape(mib)
            dtype = getattr(mx, dtype_name)
            one_bits = 0x3C00 if dtype_name == 'float16' else 0x3F80
            host = np.full(shape, one_bits, dtype=np.uint16)
            host_ref = weakref.ref(host)
            expected_host = hashlib.sha256(host).hexdigest()
            weight = mx.array(host).view(dtype)
            vector = mx.ones((shape[1], 1), dtype=dtype)
            output = weight @ vector
            mx.eval(weight, output)
            expected_weight = bits(weight)
            expected_output = bits(output)
            rows_ok = bool(np.all(np.asarray(output.astype(mx.float32)) == 4096))
            allocated = snapshot(mx)
            # Live-host negative control: release cached pages, never this object.
            live_receipt = pressure_relief(mib * 1024 * 1024)
            live_host_unchanged = hashlib.sha256(host).hexdigest() == expected_host
            live_weight_unchanged = bits(weight) == expected_weight
            live_guarded = snapshot(mx)
            del host
            host_dead = host_ref() is None
            stages, receipt = idle_then_relieve(mib * 1024 * 1024,
                                                observe=lambda: snapshot(mx))
            # Keep the complete weight, vector and output alive until after relief.
            weight_equal = bits(weight) == expected_weight
            with mx.stream(mx.default_stream(mx.gpu)):
                repeated = weight @ vector
                mx.eval(repeated)
            output_equal = bits(repeated) == expected_output
            sentinel_equal = [bits(a) for a in sentinels] == expected_sentinel
            checked = snapshot(mx)
            stages = {'before': baseline, 'host_and_weight_live': allocated,
                      'after_live_host_control': live_guarded, **stages,
                      'after_bit_and_matvec_checks': checked}
            samples.extend(stages.values())
            checks = {'all_output_rows_equal_4096': rows_ok,
                      'live_host_bits_equal': live_host_unchanged,
                      'live_weight_bits_equal': live_weight_unchanged,
                      'host_owner_dead_before_idle': host_dead,
                      'full_weight_bits_equal_after_relief': weight_equal,
                      'matvec_bits_equal_after_relief': output_equal,
                      'sentinel_bits_equal_after_relief': sentinel_equal}
            if not all(checks.values()):
                failures.append('live host/weight/state bit or ownership failure')
            cases.append({'mib': mib, 'dtype': dtype_name, 'shape': list(shape),
                          'weight_bytes': mib * 1024 * 1024, 'checks': checks,
                          'stages': stages, 'live_host_negative_control': live_receipt,
                          'dropped_host_relief': receipt,
                          'weight_sha256': expected_weight, 'output_sha256': expected_output,
                          'production_policy_changed': False})
            del weight, vector, output, repeated, host_ref
            mx.clear_cache()  # isolate cases, OUTSIDE all relief/control timings.
            if failures or pressure_failures(samples):
                break
        rng = mx.random.uniform(shape=(32,))
        mx.eval(rng)
        rng_equal = bits(rng) == expected_rng
        after = snapshot(mx)
        samples.append(after)
    finally:
        mx.clear_cache()
        mx.set_cache_limit(old_limit)
    failures.extend(pressure_failures(samples))
    if len(cases) != 4 or not rng_equal:
        failures.append('four-case completion or RNG failure')
    return {'schema': 'voom.malloc-pressure-relief-probe.v1',
            'scope': 'Four synthetic host-cache tests with idle and live-host controls',
            'passed': not failures, 'failures': sorted(set(failures)),
            'cases': cases, 'before': before, 'after': after,
            'mlx_rng_equal': rng_equal, 'expected_sentinel_sha256': expected_sentinel,
            'versions': {n: importlib.metadata.version(n) for n in ('mlx', 'numpy')},
            'true_peak_metal_bytes': max(s['metal_peak_bytes'] for s in samples),
            'observed_min_available_bytes': min(s['system_available_bytes'] for s in samples),
            'max_swap_used_growth_bytes': max(s['system_swap_used_bytes'] for s in samples) - before['system_swap_used_bytes'],
            'actual_swap_out_growth_bytes': max(s['system_swap_out_bytes'] for s in samples) - before['system_swap_out_bytes'],
            'sampled_not_uninterrupted_pressure': True, 'cache_limit_restored': True,
            'full_model_equivalence_proven': False, 'speed_win_claimed': False,
            'no_serving_or_model_change': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        parser.error('refusing existing result')
    from qwen4_hot_boundary_http_probe import _atomic_write_private
    started = time.perf_counter()
    try:
        abi = verify_sdk()
        result = run_probe()
        result['sdk_oracle'] = abi
    except Exception as error:
        result = {'passed': False, 'failures': [type(error).__name__ + ': ' + str(error)]}
    result['wall_seconds'] = time.perf_counter() - started
    _atomic_write_private(args.result, result)
    print(json.dumps({k: result.get(k) for k in ('passed', 'wall_seconds',
        'true_peak_metal_bytes', 'observed_min_available_bytes', 'failures')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
