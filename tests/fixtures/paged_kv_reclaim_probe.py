#!/usr/bin/env python3
"""Supervised geometry/ownership probe, NOT full-model inference or a speed score.

Build actual Huihui KV geometry with deterministic BF16 activations. Compare
best-effort logical reclamation to real Metal active memory and an explicit
alias-retention negative control. Keep the 256MB cache budget and all tokens.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.fixtures.runtime_profile_http_gate import _pressure
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private


def probe(mx, root, *, hold_aliases):
    from runtime.kv_paged import PagedKVCache
    import numpy as np

    def tokens(layer, start, width):
        # Position-major indexing gives identical bits for any append width.
        raw = mx.arange(start*1024, (start+width)*1024, dtype=mx.uint32)
        keys = ((raw.reshape(1, width, 4, 256).transpose(0, 2, 1, 3)
                 + layer*17) % 251).astype(mx.bfloat16)
        return keys, (keys + 3).astype(mx.bfloat16)

    def digest(keys, values):
        return hashlib.sha256(np.array(keys.view(mx.uint16)).tobytes()
            + np.array(values.view(mx.uint16)).tobytes()).hexdigest()

    kv = PagedKVCache(64, 256_000_000, root, page_positions=256, resident_pages=1)
    layers = range(3, 64, 4)
    length = 5046
    memory_samples = []
    started = time.perf_counter()
    try:
        for layer in layers:
            for start in range(0, length, 128):
                keys, values = tokens(layer, start, min(128, length-start))
                kv.append_for_online_attention(layer, keys, values)
            del keys, values
            memory_samples.append(asdict(_pressure()))
        lengths, endpoint = kv.layer_lengths(), kv.offset
        protected = tuple((p.k, p.v, p.path) for p in kv._pages[27])
        tails = tuple(kv._tail_k), tuple(kv._tail_v)
        # Deliberately retain eligible page owners in the negative control.
        aliases = [(p.k, p.v) for pages in kv._pages for p in pages if p.resident] if hold_aliases else []
        gc.collect()
        mx.clear_cache()
        before = dict(metal_active_bytes=mx.get_active_memory(), logical_bytes=kv.nbytes(),
                      spills=kv.stats.spills, pressure=asdict(_pressure()))
        requested = 137_720_144  # observed serial-admission deficit; geometry probe only
        t0 = time.perf_counter()
        released = kv.reclaim_closed_pages(requested, protected_layer=27)
        reclaim_s = time.perf_counter() - t0
        after = dict(metal_active_bytes=mx.get_active_memory(), logical_bytes=kv.nbytes(),
                     spills=kv.stats.spills, pressure=asdict(_pressure()))
        assert before["logical_bytes"] - after["logical_bytes"] == released >= requested
        assert kv.max_bytes == 256_000_000 and kv.layer_lengths() == lengths and kv.offset == endpoint
        assert all((p.k is k and p.v is v and p.path == path)
                   for p, (k, v, path) in zip(kv._pages[27], protected))
        assert all(a is b for a, b in zip(kv._tail_k, tails[0]))
        assert all(a is b for a, b in zip(kv._tail_v, tails[1]))
        physical = before["metal_active_bytes"] - after["metal_active_bytes"]
        expected_hashes, observed_hashes = [], []
        for layer in layers:
            keys, values = tokens(layer, 0, length)
            expected_hashes.append(digest(keys, values))
            del keys, values
            keys, values = kv.materialize_layer(layer)
            observed_hashes.append(digest(keys, values))
            del keys, values
        assert observed_hashes == expected_hashes
        # This is a required positive physical proof and explicit alias control,
        # not an assumption that logical bytes always become allocation credit.
        physical_check = physical < requested if hold_aliases else physical >= requested
        return dict(hold_aliases=hold_aliases, passed=physical_check,
            before=before, after=after, requested_bytes=requested,
            logical_reclaimed_bytes=released, physical_active_released_bytes=physical,
            reclaim_seconds=reclaim_s, wall_seconds=time.perf_counter()-started,
            layer_lengths=lengths, full_layer_bf16_sha256=observed_hashes,
            exact_full_history=True, protected_layer_unchanged=True,
            tails_unchanged=True, budget_unchanged=True,
            geometry=dict(layers=64, attention_layers=16, kv_heads=4, head_dim=256,
                          context=length, append_width=128, page_positions=256),
            pressure_samples=memory_samples, stats=asdict(kv.stats))
    finally:
        kv.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--pytest-root', type=Path, required=True)
    parser.add_argument('--include-serial-recovery-tests', action='store_true',
                        help='Also test the real serial admission hook under a controlled device ceiling.')
    args = parser.parse_args()
    assert not args.result.exists() and not args.pytest_root.exists()
    pre = json.loads(args.preflight.read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30 and pre['known_transcoders']['passed']
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    started = time.perf_counter()
    before = asdict(_pressure())
    import mlx.core as mx
    import pytest
    mx.reset_peak_memory()
    tests = ['tests/test_paged_kv_reclaim_mlx.py', 'tests/test_paged_kv_hybrid_recurrent.py',
             'tests/test_qwen_paged_hybrid_prefix_persist.py', 'tests/test_qwen35_paged_online_attention.py']
    if args.include_serial_recovery_tests:
        tests += ['tests/test_qwen_kv_reclaim_mlx.py', 'tests/test_qwen_mtp_scalar_rollback.py']
    rc = int(pytest.main(['-q', '--basetemp='+str(args.pytest_root), *tests]))
    rows, error = [], None
    try:
        if rc == 0:
            for hold_aliases in (False, True):
                gc.collect()
                mx.clear_cache()
                with tempfile.TemporaryDirectory(prefix='kv-reclaim-proof-', dir=ROOT/'logs') as tmp:
                    rows.append(probe(mx, Path(tmp), hold_aliases=hold_aliases))
    except Exception as exc:
        error = type(exc).__name__
    after = asdict(_pressure())
    peak = mx.get_peak_memory()
    samples = [before, after, *[s for r in rows for s in [r['before']['pressure'], r['after']['pressure'], *r['pressure_samples']]]]
    pressure_ok = (min(s['available_bytes'] for s in samples) >= 5_300_000_000
        and max(s['swap_used_bytes']-before['swap_used_bytes'] for s in samples) <= 16_000_000
        and max(s['swap_out_bytes']-before['swap_out_bytes'] for s in samples) <= 16_000_000)
    passed = rc == 0 and error is None and len(rows) == 2 and all(r['passed'] for r in rows) and pressure_ok and 0 < peak <= 8_500_000_000
    doc = dict(schema='voom.paged-kv-reclamation-probe.v1', passed=passed,
        scope=__doc__, model_executed=False, serving_integration_enabled=False,
        serial_admission_hook_tested=args.include_serial_recovery_tests,
        test_exit_code=rc, tests=tests, rows=rows, error=error,
        pressure_before=before, pressure_after=after, sampled_pressure_passed=pressure_ok,
        observed_peak_metal_bytes=peak, preflight_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        wall_seconds=time.perf_counter()-started)
    _atomic_write_private(args.result, doc)
    print(json.dumps(doc), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
