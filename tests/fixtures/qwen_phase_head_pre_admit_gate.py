#!/usr/bin/env python3
"""Small real-Metal lifetime gate, not a model/harness or throughput benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(preflight, result_path):
    pre = json.loads(preflight.read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30
    assert pre['known_transcoders']['passed']
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    assert pre['end']['root_free_bytes'] >= 10_000_000_000
    assert not result_path.exists()
    # No MLX import or array exists before the mandatory fresh gate above.
    import mlx.core as mx
    import numpy as np
    import psutil
    from runtime.engine import StreamingEngine
    from runtime.pressure import MemoryGovernor
    from runtime.weight_cache import WeightCache
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private

    unit = 1024**2
    before = dict(available=psutil.virtual_memory().available, swap=psutil.swap_memory().used)
    started = time.perf_counter()
    cases = []
    for enabled in (False, True, True):
        case_start = time.perf_counter()
        class Store:
            head_fetch_active = None
            head_fetches = 0
            def fetch(self, names):
                name, = names
                size = 4*unit if name == 'lm_head.weight' else unit if name == 'norm' else 3*unit
                shape = (1024,2048) if name == 'lm_head.weight' else (size//2,)
                value = mx.full(shape, 1.5, dtype=mx.bfloat16)
                mx.eval(value)
                if name == 'lm_head.weight':
                    self.head_fetch_active = int(mx.get_active_memory())
                    self.head_fetches += 1
                return {name:value},0.0,size
        store = Store()
        cache = WeightCache(store,10*unit)
        cache.pin('persistent',['norm'])
        cache.register_suspended_pin('qwen35:lm_head:persistent',4*unit)
        for index in range(3):
            cache.get('body'+str(index),['body'+str(index)])
        e = object.__new__(StreamingEngine)
        e.cfg = SimpleNamespace(tie_word_embeddings=False)
        e.rc = SimpleNamespace(qwen35_phase_head_pre_admit=enabled,
            qwen35_serial_verify_suspend_lm_head=True)
        e.cache = cache
        e.prefetcher = None
        e.governor = MemoryGovernor(cache,critical_available=5_600_000_000,
            floor_bytes=unit,metal_limit=8_500_000_000)
        e._lm_head_w = e._streamed_lm_head = None
        e._qwen4_lm_head_pin_suspended = False
        e._qwen35_lm_head_pin_suspended = True
        e._qwen35_lm_head_suspend_request_active = True
        e._qwen35_phase_head_admission_stats = {}
        for key in ('calls','successes','refusals','s'):
            setattr(e,'_qwen35_serial_verify_head_restore_'+key,0)
        active_before = int(mx.get_active_memory())
        head = e._lm_head_weight()
        assert head is e._lm_head_w and store.head_fetches == 1
        assert cache.max_bytes == 10*unit and cache.pinned_bytes == 5*unit
        x = (mx.arange(2048,dtype=mx.float32)/4096).astype(mx.bfloat16)[None,:]
        logits = x @ head.T
        mx.eval(logits)
        host = np.array(logits.astype(mx.float32),copy=True)
        assert host.shape == (1,1024) and np.isfinite(host).all()
        cases.append(dict(enabled=enabled,wall_seconds=time.perf_counter()-case_start,
            active_before_head=active_before, head_fetch_active=store.head_fetch_active,
            head_fetches=store.head_fetches,cache_budget=cache.max_bytes,
            logits_sha256=hashlib.sha256(host.tobytes()).hexdigest(),
            logits_shape=list(host.shape),head_shape=list(head.shape),head_dtype=str(head.dtype),
            admission=dict(e._qwen35_phase_head_admission_stats),
            governor_reservation_calls=e.governor.reservation_calls,
            governor_reason_counts=e.governor.reservation_reason_counts))
        e._lm_head_w = None
        del head,x,logits,host
        cache.clear()
        del e,cache,store
        mx.clear_cache()
    assert len({case['logits_sha256'] for case in cases}) == 1
    for case in cases[1:]:
        assert case['head_fetch_active'] <= cases[0]['head_fetch_active'] - 5*unit
        assert case['admission']['logical_trimmed_bytes'] == 6*unit
        assert case['admission']['reservation_refusals'] == 0
        assert case['governor_reservation_calls'] == 1
        assert case['governor_reason_counts'] == {'qwen35-phase-lm-head':1}
    peak = int(mx.get_peak_memory())  # never reset the process-wide high-water
    assert 0 < peak <= 8_500_000_000
    result = dict(schema='voom.qwen35-head-pre-admit-metal-gate.v1',passed=True,
        scope='Tiny synthetic BF16 head; actual engine/cache/governor/Metal ordering. Not Huihui weights, full recurrent/KV/model tokens, harness, quality or speed proof.',
        preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),cases=cases,
        whole_process_metal_peak=peak,wall_seconds=time.perf_counter()-started,
        before=before,after=dict(available=psutil.virtual_memory().available,
            swap=psutil.swap_memory().used))
    _atomic_write_private(result_path,result)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    run(args.preflight,args.result)
