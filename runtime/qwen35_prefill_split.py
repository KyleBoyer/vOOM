"""Opt-in dense Qwen prefill with attention/MLP weight lifetimes separated.

Each phase retains all prompt positions and the original per-tile operators.
Only independent MLP work moves after the attention tile sequence. No expert,
vision, boundary-fork or prefetch path is admitted by this initial experiment.
"""
from __future__ import annotations

import json
import sys
import time


def partition(names, layer):
    prefix=f'model.layers.{layer}.'
    attention=[]; mlp=[]
    if not names or len(set(names)) != len(names):
        raise ValueError('split prefill requires unique nonempty layer names')
    for name in names:
        if not name.startswith(prefix):
            raise ValueError('split prefill layer prefix mismatch')
        suffix=name[len(prefix):]
        if suffix.startswith(('self_attn.','linear_attn.')) or suffix=='input_layernorm.weight':
            attention.append(name)
        elif suffix.startswith('mlp.') or suffix=='post_attention_layernorm.weight':
            mlp.append(name)
        else:
            raise ValueError('unsupported split prefill tensor category')
    if not attention or not mlp:
        raise ValueError('split prefill requires both attention and MLP')
    return attention,mlp


def validate(rc,cfg,*,positions3=None,boundary_fork_at=None,boundary_fork_kv=None):
    if (cfg.model_type!='qwen3_5' or cfg.num_experts or cfg.hidden_size!=5120
            or cfg.vocab_size!=248320 or not rc.governor
            or not rc.qwen35_mxfp4_head_rows or rc.prefetch_depth
            or not rc.layer_stationary_prefill):
        raise ValueError('split prefill requires native-head dense Huihui, governor and no prefetch')
    if positions3 is not None or boundary_fork_at is not None or boundary_fork_kv is not None:
        raise ValueError('split prefill does not yet support vision or boundary forks')


def sweep(engine,x,kv,offset,tile_width,on_progress=None,*,layer_start=0,layer_end=None,
          profile_path='layer_stationary_qwen35',positions3=None,
          boundary_fork_at=None,boundary_fork_kv=None):
    # Validate the complete metadata plan BEFORE any recurrent/KV mutation.
    validate(engine.rc,engine.cfg,positions3=positions3,boundary_fork_at=boundary_fork_at,
             boundary_fork_kv=boundary_fork_kv)
    total=int(x.shape[1]); n=engine.cfg.num_hidden_layers
    layer_end=n if layer_end is None else int(layer_end)
    if tile_width<=0 or total<=0 or not 0<=layer_start<layer_end<=n:
        raise ValueError('invalid split prefill range or tile')
    plans=[]
    for layer in range(layer_start,layer_end):
        phases=[]
        for phase,names in zip(('attention','mlp'),partition(engine._layer_names(layer),layer)):
            incoming=engine._layer_fetch_bytes_estimate(layer,names)
            if type(incoming) is not int or incoming<=0:
                raise ValueError('split prefill requires complete physical page estimates')
            phases.append((phase,names,incoming))
        plans.append((layer,phases))

    import mlx.core as mx
    from .qwen35 import _qwen35_attention_residual, _qwen35_mlp_residual
    from .engine import (_layer_transient_for_positions, _remaining_layer_transient_reserve,
                         _recurring_layer_transient_reserve_margin, _resident_adjusted_transient)

    (engine._layer_transient,engine._layer_transient_margin)=_layer_transient_for_positions(
        total,getattr(engine,'_prefill_layer_transient_by_positions',{}).get(total,0),
        getattr(engine,'_decode_layer_transient',0))
    profiler=engine._request_profiler
    tap=engine._dspark_tap_collector
    stats=engine._qwen35_split_prefill_stats
    stats['sweeps']=stats.get('sweeps',0)+1
    if profiler is not None:
        profiler.begin_sweep(total,path=profile_path+'_split_weights')
    for layer,phases in plans:
        engine._select_layer_transient(total,layer)
        cache_before=profiler.cache_snapshot(engine.cache) if profiler is not None else None
        weight_s=compute_s=0.0; transient=0
        for phase,names,incoming in phases:
            key=f'qwen35-prefill-split:{layer}:{phase}'
            w=None
            stage='admission'; tile_start=None
            try:
                started=time.perf_counter()
                # No prefetch is allowed to repopulate a retired phase.
                # Trim only unpinned weight pages, never KV or model state.
                engine.cache.trim_to(engine.cache.pinned_bytes)
                engine.cache.prepare_for(incoming)
                engine.governor.reserve(incoming,reason='qwen-prefill-layer-page')
                stage='fetch'
                w=engine.cache.get(key,names)
                weight_s+=time.perf_counter()-started
                engine._note_true_peak()
                active_before=mx.get_active_memory()
                mx.reset_peak_memory()
                tiles=[]; xt=yt=None
                started=time.perf_counter()
                stage='compute'
                for pos in range(0,total,tile_width):
                    tile_start=pos
                    end=min(pos+tile_width,total)
                    if engine._layer_transient:
                        signature=engine._transient_layer_signature(layer)
                        observations=getattr(engine,'_layer_transient_observation_counts',{}).get((total,signature),0)
                        engine.governor.reserve(
                            _remaining_layer_transient_reserve(engine._layer_transient,pos*int(x.nbytes)//total),
                            margin=_recurring_layer_transient_reserve_margin(total,observations),
                            reason='qwen-prefill-transient')
                    xt=x[:,pos:end,:]
                    if phase=='attention':
                        yt=_qwen35_attention_residual(
                            xt,w,f'model.layers.{layer}',engine.cfg,kv,layer,offset+pos,
                            mlp_last_only=False,positions3=None,
                            zmlx_fused_decode=engine.rc.zmlx_fused_deltanet_decode,
                            native_fused_decode=engine.rc.native_fused_deltanet_decode,
                            chunked_delta_prefill=engine.rc.qwen_chunked_delta_prefill,
                            compiled_delta_prefill=engine.rc.qwen_compiled_delta_prefill,
                            native_fused_delta_prefill=engine.rc.qwen_native_fused_delta_prefill)
                    else:
                        yt=_qwen35_mlp_residual(xt,w,f'model.layers.{layer}',engine.cfg,layer,
                            engine._get_experts,iter_expert_batches=engine._iter_expert_batches,profile=profiler)
                    mx.eval(yt)
                    tiles.append(yt)
                x=tiles[0] if len(tiles)==1 else mx.concatenate(tiles,axis=1)
                mx.eval(x)
                tiles.clear(); xt=yt=None
                elapsed=time.perf_counter()-started
                compute_s+=elapsed
                if profiler is not None and profiler.sync_substeps:
                    profiler.record_substep(phase,layer,elapsed,positions=total)
                transient=max(transient,_resident_adjusted_transient(
                    active_before,mx.get_active_memory(),mx.get_peak_memory()))
                engine._note_true_peak()
                stats[phase+'_phases']=stats.get(phase+'_phases',0)+1
                stats['maximum_declared_phase_page_bytes']=max(stats.get('maximum_declared_phase_page_bytes',0),incoming)
            except Exception as exc:
                # Bounded scalar context survives a failed sweep/retry. The
                # original exception and safety decision remain authoritative;
                # do not retain tensors, prompts, or traceback objects here.
                stats['phase_failures']=stats.get('phase_failures',0)+1
                failure=dict(sweep=stats['sweeps'],layer=layer,phase=phase,
                    stage=stage,tile_start=tile_start,tile_width=tile_width,
                    positions=total,declared_page_bytes=incoming,
                    error_type=type(exc).__name__)
                stats['last_phase_failure']=failure
                # Failed HTTP requests may not publish final path_stats.
                try:
                    print('[qwen35-split-prefill-failure] '+json.dumps(failure,sort_keys=True),
                          file=sys.stderr,flush=True)
                except Exception:
                    pass  # A diagnostic sink must never mask the refusal.
                raise
            finally:
                # The evaluated phase outputs survive, not its weight owner.
                # discard also clears allocator cache after dropping residency.
                w=None
                engine.cache.discard(key,names)
        if tap is not None and layer in tap.tap_layers:
            tap.observe(layer,x,position_start=offset,positions=None)
        engine.timer.add('weights_wait',weight_s)
        engine.timer.add('layer_compute',compute_s)
        if profiler is not None:
            profiler.record_layer(layer,positions=total,weight_wait_s=weight_s,
                compute_s=compute_s,cache_before=cache_before,
                cache_after=profiler.cache_snapshot(engine.cache),layer_type=engine._profile_layer_type(layer))
        if on_progress is not None:
            on_progress(dict(phase='prefill_layer',completed_layers=layer+1,total_layers=n,
                             total_tokens=total,cache_source='cold'))
        engine._record_layer_transient(total,layer,transient)
        stats['completed_layers']=stats.get('completed_layers',0)+1
    engine._restore_aggregate_layer_transient(total)
    return x
