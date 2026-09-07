#!/usr/bin/env python3
"""Synthetic, bit-exact SDPA component gate. NOT a model/harness/Plex result.

Run under run_gate after a fresh 30-second memory/transcoder preflight. No model
weights are read, no serving path is changed. Timed arms include output eval,
exclude input creation/hashing/snapshots; whole child wall includes everything.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.host_activity_witness import sample_known_transcoders, summarize_known_transcoders
from runtime.process_memory_witness import sample_self_memory
from runtime.qwen4_sdpa_tiling import query_tiled_sdpa, query_tiles

CASES = ((1, 1, 257, 77123), (1, 17, 127, 60391),
         (2, 257, 2049, 44987), (1, 1031, 4099, 31729), (1, 1024, 32768, 90583))
ORDER = (0, 128, 256, 512, 512, 256, 128, 0)
MAX_SAMPLES = 4096


def pressure_sample():
    import psutil
    swap = psutil.swap_memory()
    return dict(monotonic_s=time.monotonic(), native=sample_self_memory(),
        available_bytes=psutil.virtual_memory().available, swap_used_bytes=swap.used,
        swap_out_bytes=swap.sout, root_free_bytes=psutil.disk_usage('/').free)


def pressure_failures(samples, peak):
    if not samples or any(not s['native'].get('available') for s in samples):
        return ['native pressure unavailable']
    checks = {
        'available memory below5.3GB': min(s['available_bytes'] for s in samples) >= 5_300_000_000,
        'net swap growth above16MB': max(s['swap_used_bytes'] for s in samples)-samples[0]['swap_used_bytes'] <= 16_000_000,
        'actual swap-out above16MB': max(s['swap_out_bytes'] for s in samples)-samples[0]['swap_out_bytes'] <= 16_000_000,
        'root free below10GB': min(s['root_free_bytes'] for s in samples) >= 10_000_000_000,
        'Metal peak above8.5GB': peak <= 8_500_000_000,
    }
    return [name for name, passed in checks.items() if not passed]


def run_probe(document):
    import mlx.core as mx
    import numpy as np
    samples, activity = [], [sample_known_transcoders()]
    stop = threading.Event()
    poll_errors = []
    def poll():
        try:
            while not stop.is_set():
                if len(samples) >= MAX_SAMPLES:
                    poll_errors.append('pressure sample cap'); return
                samples.append(pressure_sample())
                if stop.wait(.05): return
        except Exception:
            poll_errors.append('pressure sample error')
    samples.append(pressure_sample())
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    old_cache = mx.set_cache_limit(0)
    peak = 0
    def bits(value):
        return np.array(value.view(mx.uint16), copy=True)
    def sha(value):
        return hashlib.sha256(value.tobytes(order='C')).hexdigest()
    try:
        for dtype_name in ('bfloat16', 'float16'):
            for batch, length, key_length, seed in CASES:
                failures = pressure_failures(samples, peak)
                if failures: document['failures'].extend(failures); return
                rng = np.random.default_rng(seed)
                dtype = getattr(mx, dtype_name)
                def random_array(n, heads):
                    # Transposed layout matches the attention API layout, not
                    # a flattened/repeated KV approximation.
                    return mx.array(rng.standard_normal((batch,n,heads,256), dtype=np.float32)).astype(dtype).transpose(0,2,1,3)
                q, k, v = random_array(length,24), random_array(key_length,2), random_array(key_length,2)
                positions = mx.arange(key_length-length,key_length)[:,None]
                key_positions = mx.arange(key_length)[None,:]
                selected = key_positions <= positions
                if key_length > 2048:
                    selected = selected & (((key_positions//4 + positions*13 + seed)%16 == 0)
                                            | (key_positions == positions))
                mask = None if length==1 else mx.where(selected,0.,-mx.inf).astype(dtype)[None,None]
                mx.eval(q,k,v,*([] if mask is None else [mask]))
                inputs = (q,k,v) + (() if mask is None else (mask,))
                before_hashes = [sha(bits(a)) for a in inputs]
                peak = max(peak, int(mx.get_peak_memory()))
                case = dict(dtype=dtype_name,batch=batch,query_tokens=length,key_tokens=key_length,
                    seed=seed,query_heads=24,kv_heads=2,head_dim=256,
                    mask='none' if mask is None else 'synthetic additive causal/block mask',
                    input_sha256=before_hashes,arms=[])
                document['cases'].append(case)
                reference = mx.fast.scaled_dot_product_attention(q,k,v,scale=.0625,mask=mask)
                mx.eval(reference)
                reference_bits = bits(reference);case['reference_sha256']=sha(reference_bits)
                peak = max(peak,int(mx.get_peak_memory()));del reference
                for tile in ORDER:
                    activity.append(sample_known_transcoders())
                    failures = pressure_failures(samples,peak)
                    if failures or not summarize_known_transcoders(activity)['passed']:
                        document['failures'].extend(failures or ['known transcoder isolation']);return
                    mx.clear_cache();mx.reset_peak_memory()
                    started=time.perf_counter()
                    output=query_tiled_sdpa(mx,q,k,v,scale=.0625,mask=mask,tile_queries=tile)
                    mx.eval(output)
                    seconds=time.perf_counter()-started
                    arm_peak=int(mx.get_peak_memory());peak=max(peak,arm_peak)
                    actual=bits(output)
                    mismatches=int(np.count_nonzero(actual!=reference_bits))
                    case['arms'].append(dict(tile_queries=tile,seconds=seconds,
                        output_sha256=sha(actual),bit_mismatches=mismatches,exact=mismatches==0,
                        output_values=int(actual.size),query_chunks=len(query_tiles(length,tile)),
                        metal_peak_bytes=arm_peak,after=pressure_sample()))
                    samples.append(case['arms'][-1]['after'])
                    del output,actual
                case['input_sha256_after']=[sha(bits(a)) for a in inputs]
                case['inputs_unchanged']=case['input_sha256_after']==before_hashes
                del q,k,v,mask,inputs,reference_bits,selected,positions,key_positions
                mx.clear_cache()
    finally:
        peak=max(peak,int(mx.get_peak_memory()))
        stop.set();thread.join(timeout=2)
        if thread.is_alive():poll_errors.append('pressure observer did not stop')
        samples.append(pressure_sample());activity.append(sample_known_transcoders())
        mx.set_cache_limit(old_cache)
        document.update(pressure_samples=samples,known_transcoders=summarize_known_transcoders(activity),
            true_peak_metal_bytes=peak,pressure_failures=pressure_failures(samples,peak),
            observation_failures=poll_errors,cache_limit_restored=True)
        document['failures'].extend(document['pressure_failures']+poll_errors)
        if not document['known_transcoders']['passed']:document['failures'].append('known transcoder isolation')
    complete=len(document['cases'])==2*len(CASES) and all(len(c['arms'])==len(ORDER) and c.get('inputs_unchanged') for c in document['cases'])
    document['coverage_complete']=complete
    document['bit_exact_tiles']=[tile for tile in (128,256,512) if complete and all(a['exact'] for c in document['cases'] for a in c['arms'] if a['tile_queries']==tile)]
    if not complete:document['failures'].append('incomplete cases or mutated inputs')
    if complete and any(not a['exact'] for c in document['cases'] for a in c['arms']):document['failures'].append('bit mismatch')
    document['passed']=not document['failures']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args()
    if args.result.exists():parser.error('refusing existing result')
    pre=json.loads(args.preflight.read_text())
    if not (pre.get('passed') and pre['sample_seconds']>=30 and pre['known_transcoders']['passed']
            and 0<=time.monotonic()-pre['end']['monotonic_s']<120):
        parser.error('fresh30s memory/transcoder preflight required')
    document=dict(schema='voom.qwen4-sdpa-tiling-component.v1',passed=False,failures=[],cases=[],
        scope='Synthetic SDPA only; no model weights, tokens, full state, captured traffic or Plex proof. No serving integration.',
        preflight=pre,preflight_sha256=hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
        versions={name:importlib.metadata.version(name) for name in ('mlx','numpy')},
        timing_scope='Per-arm SDPA construction+eval only; excludes input creation, hashes, snapshots and cache clearing. Whole wall includes everything.',
        physical_model_io_bytes=0,periodic_pressure_seconds=.05,sampled_not_continuous_pressure=True,
        native_and_metal_views_overlap=True)
    started=time.perf_counter()
    try:run_probe(document)
    except BaseException as error:document['failures'].append(type(error).__name__);document['passed']=False
    document['wall_seconds']=time.perf_counter()-started
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
    _atomic_write_private(args.result,document)
    print(json.dumps({k:document.get(k) for k in ('passed','failures','bit_exact_tiles','coverage_complete','wall_seconds','true_peak_metal_bytes')}),flush=True)
    return 0 if document['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
