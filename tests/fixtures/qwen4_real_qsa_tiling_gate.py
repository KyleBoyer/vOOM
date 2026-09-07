#!/usr/bin/env python3
"""Actual installed Qwen QSA weights/state; synthetic hidden inputs, not a model answer.

Only attention layers3/47, with real projections/index ranking/RoPE/KV updates.
No embeddings, PLE/DeltaNet/MoE/head or full-model intelligence proof. Run under
run_gate with fresh30s preflight; never concurrently with another MLX/model job.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.fixtures.qwen4_sdpa_tiling_gate import pressure_sample,pressure_failures
from runtime.host_activity_witness import sample_known_transcoders,summarize_known_transcoders

# These cases cross index-budget and original1024-row boundaries independently
# of captured prompt content; final31 positions match the problematic geometry.
CASES=((3,3079,61783),(47,32799,94217))


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run_probe(document,model):
    import mlx.core as mx
    import numpy as np
    from runtime.config import ModelConfig
    from runtime.model_loader import WeightStore
    from runtime.kv_cache import KVCache
    from runtime.qwen4_exp_state import Qwen4ExpStateCache
    from runtime.qwen4_exp import qwen4_attention_branch
    from runtime.qwen4_sdpa_tiling import query_tiles
    samples=[pressure_sample()];activity=[sample_known_transcoders()];stop=threading.Event();errors=[]
    def poll():
        try:
            while not stop.is_set():
                if len(samples)>=12000:errors.append('sample cap');return
                samples.append(pressure_sample())
                if stop.wait(.05):return
        except Exception:errors.append('sample unavailable')
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    old_limit=mx.set_cache_limit(0);peak=0;store=None
    def digest(value):
        raw=np.asarray(value.view(mx.uint8)).tobytes(order='C')
        return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def state(kv,layer):
        arrays=[kv.keys[layer],kv.values[layer],kv.qwen4_cache.qsa_keys[layer],kv.qwen4_cache.qsa_positions[layer]]
        mx.eval(*arrays)
        return {name:digest(a) for name,a in zip(('key','value','qsa_key','qsa_positions'),arrays)}
    try:
        cfg=ModelConfig.from_dir(model)
        assert cfg.model_type=='qwen4_exp' and cfg.head_dim==256 and cfg.num_attention_heads==24 and cfg.num_key_value_heads==2
        store=WeightStore(model)
        for layer,total,seed in CASES:
            failures=pressure_failures(samples,peak)
            if failures:document['failures'].extend(failures);return
            assert cfg.layer_types[layer]=='full_attention'
            prefix=f'model.layers.{layer}'
            names=store.names_with_prefix(prefix+'.self_attn.')
            assert names and all('.mlp.' not in n for n in names)
            started=time.perf_counter();weights,fetch_seconds,read_bytes=store.fetch(names)
            assert all(isinstance(value,mx.array) for value in weights.values())
            mx.eval(*weights.values());load_seconds=time.perf_counter()-started
            peak=max(peak,int(mx.get_peak_memory()))
            weight_hashes={name:digest(value) for name,value in weights.items()}
            case=dict(layer=layer,total_tokens=total,seed=seed,activation_dtype='bfloat16',
                loaded_weight_hashes=weight_hashes,store_accounted_bytes=read_bytes,
                weight_fetch_seconds=fetch_seconds,weight_fetch_eval_seconds=load_seconds,arms=[])
            document['cases'].append(case)
            # Reverse the arm order on the long case; neither arm changes the
            # seed, hidden values, released weights or1024-row outer tiles.
            for tile in ((0,256) if layer==3 else (256,0)):
                kv=KVCache(cfg.num_hidden_layers);kv.qwen4_cache=Qwen4ExpStateCache(cfg.num_hidden_layers)
                rng=np.random.default_rng(seed);stats={};arm=dict(tile_queries=tile,steps=[],sdpa_stats=stats)
                case['arms'].append(arm)
                peak=max(peak,int(mx.get_peak_memory()));mx.clear_cache();mx.reset_peak_memory()
                for start in range(0,total,1024):
                    end=min(start+1024,total)
                    hidden=mx.array(rng.standard_normal((1,end-start,cfg.hidden_size),dtype=np.float32)).astype(mx.bfloat16)
                    mx.eval(hidden);hidden_before=digest(hidden)
                    activity.append(sample_known_transcoders())
                    failures=pressure_failures(samples,peak)
                    if failures or not summarize_known_transcoders(activity)['passed']:
                        document['failures'].extend(failures or ['known transcoder isolation']);return
                    started=time.perf_counter()
                    out=qwen4_attention_branch(hidden,weights,prefix,cfg,kv,layer,start,
                        sdpa_query_tile=tile,sdpa_stats=stats)
                    mx.eval(out);seconds=time.perf_counter()-started
                    peak=max(peak,int(mx.get_peak_memory()))
                    arm['steps'].append(dict(start=start,end=end,seconds=seconds,output=digest(out),
                        state=state(kv,layer),input=hidden_before,input_unchanged=digest(hidden)==hidden_before))
                    del hidden,out;mx.clear_cache()
                arm['prefill_seconds']=sum(row['seconds'] for row in arm['steps'])
                arm['prefill_peak_metal_bytes']=int(mx.get_peak_memory())
                arm['continuation']=[]
                for pos in range(total,total+3):
                    hidden=mx.array(rng.standard_normal((1,1,cfg.hidden_size),dtype=np.float32)).astype(mx.bfloat16)
                    mx.eval(hidden);started=time.perf_counter()
                    # Decode remains the original unconfigured operation.
                    out=qwen4_attention_branch(hidden,weights,prefix,cfg,kv,layer,pos)
                    mx.eval(out);seconds=time.perf_counter()-started
                    arm['continuation'].append(dict(position=pos,seconds=seconds,output=digest(out),state=state(kv,layer)))
                    peak=max(peak,int(mx.get_peak_memory()));del hidden,out
                if tile:
                    expected_chunks=sum(len(query_tiles(min(1024,total-start),tile)) for start in range(0,total,1024))
                    assert stats['query_tiles']==expected_chunks and stats['calls']==len(arm['steps']) and stats['split_calls']>0
                del kv;mx.clear_cache()
            reference=next(a for a in case['arms'] if a['tile_queries']==0);candidate=next(a for a in case['arms'] if a['tile_queries']==256)
            case['exact_steps']=all(all(a[k]==b[k] for k in ('start','end','output','state','input')) and a['input_unchanged'] and b['input_unchanged'] for a,b in zip(reference['steps'],candidate['steps'])) and len(reference['steps'])==len(candidate['steps'])
            case['exact_continuation']=all(a['output']==b['output'] and a['state']==b['state'] for a,b in zip(reference['continuation'],candidate['continuation']))
            case['weights_unchanged']={name:digest(value) for name,value in weights.items()}==weight_hashes
            if not all(case[k] for k in ('exact_steps','exact_continuation','weights_unchanged')):document['failures'].append('actual QSA output/state mismatch')
            del weights;mx.clear_cache()
    finally:
        peak=max(peak,int(mx.get_peak_memory()))
        if store is not None:store.close()
        stop.set();thread.join(timeout=2)
        if thread.is_alive():errors.append('observer did not stop')
        samples.append(pressure_sample());activity.append(sample_known_transcoders());mx.set_cache_limit(old_limit)
        document.update(pressure_samples=samples,true_peak_metal_bytes=peak,
            known_transcoders=summarize_known_transcoders(activity),cache_limit_restored=True)
        document['failures'].extend(pressure_failures(samples,peak)+errors)
        if not document['known_transcoders']['passed']:document['failures'].append('known transcoder isolation')
    if len(document['cases'])!=len(CASES):document['failures'].append('incomplete cases')
    document['passed']=not document['failures']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);args=parser.parse_args()
    if args.result.exists():parser.error('refusing existing result')
    pre=json.loads(args.preflight.read_text())
    if not(pre['passed'] and pre['sample_seconds']>=30 and pre['known_transcoders']['passed'] and 0<=time.monotonic()-pre['end']['monotonic_s']<120):parser.error('fresh30s preflight required')
    doc=dict(schema='voom.qwen4-real-qsa-tiling.v1',passed=False,failures=[],cases=[],preflight=pre,
        preflight_sha256=file_sha(args.preflight),model=str(args.model),
        model_metadata_sha256={p.name:file_sha(p) for p in (args.model/'config.json',args.model/'model.safetensors.index.json',args.model/'voom.checkpoint.receipt.json') if p.exists()},
        scope='Actual installed attention weights, synthetic hidden inputs. QSA outputs/KV/index state and3 singleton continuations only. No generated tokens, PLE/DeltaNet/MoE/head/full-model state/harness/Plex proof.',
        timing_scope='Per-step attention construction+eval, excludes input setup/hashing/observations/clearing. Weight load timed separately; whole wall includes all.',
        store_bytes_are_physical_io=False,sampled_pressure_seconds=.05)
    started=time.perf_counter()
    try:run_probe(doc,args.model)
    except BaseException as error:doc['passed']=False;doc['failures'].append(type(error).__name__)
    doc['wall_seconds']=time.perf_counter()-started
    from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
    _atomic_write_private(args.result,doc)
    print(json.dumps({k:doc.get(k) for k in ('passed','failures','wall_seconds','true_peak_metal_bytes')}),flush=True)
    return 0 if doc['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
