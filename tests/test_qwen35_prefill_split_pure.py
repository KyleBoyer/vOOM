"""Split scheduler topology/ownership/guards with fake operations, no real MLX."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType

import numpy as np
import pytest

from runtime import qwen35_prefill_split as split
from runtime.profiles import apply_runtime_profiles
from runtime.server import _cache_phase_telemetry, _vision_protocol_timing


def names(i):
    p=f'model.layers.{i}.'
    return [p+'input_layernorm.weight',p+'linear_attn.in_proj.weight',
            p+'post_attention_layernorm.weight',p+'mlp.up_proj.weight']


def config():
    return NS(governor=True,qwen35_mxfp4_head_rows=8192,prefetch_depth=0,
        layer_stationary_prefill=True,zmlx_fused_deltanet_decode=False,
        native_fused_deltanet_decode=False,qwen_chunked_delta_prefill=True,
        qwen_compiled_delta_prefill=False,qwen_native_fused_delta_prefill=False), NS(
        model_type='qwen3_5',num_experts=0,hidden_size=5120,vocab_size=248320,num_hidden_layers=2)


def test_partition_disjoint_complete_and_preserves_name_order():
    a,m=split.partition(names(2),2)
    assert a==names(2)[:2] and m==names(2)[2:]


@pytest.mark.parametrize('bad',[[],names(0)+names(0),names(1),
    ['model.layers.0.unknown.weight'],names(0)[:2]])
def test_unknown_or_partial_layout_rejects(bad):
    with pytest.raises(ValueError):split.partition(bad,0)


@pytest.mark.parametrize('key,value',[('prefetch_depth',1),('governor',False),
    ('qwen35_mxfp4_head_rows',0),('layer_stationary_prefill',False)])
def test_runtime_guard_rejects_unsafe_or_unqualified_config(key,value):
    rc,cfg=config();setattr(rc,key,value)
    with pytest.raises(ValueError):split.validate(rc,cfg)


@pytest.mark.parametrize('key,value',[('model_type','qwen4_exp'),('num_experts',1),
    ('hidden_size',4096),('vocab_size',100)])
def test_only_qualified_geometry(key,value):
    rc,cfg=config();setattr(cfg,key,value)
    with pytest.raises(ValueError):split.validate(rc,cfg)


@pytest.mark.parametrize('key',['positions3','boundary_fork_at','boundary_fork_kv'])
def test_unqualified_boundary_fails_before_work(key):
    rc,cfg=config()
    with pytest.raises(ValueError):split.validate(rc,cfg,**{key:object()})


def fake_engine(monkeypatch,fail=None):
    events=[]; rc,cfg=config(); loaded=set()
    mx=ModuleType('mlx.core'); mx.get_active_memory=lambda:10; mx.get_peak_memory=lambda:20
    mx.reset_peak_memory=lambda:None; mx.eval=lambda x:None
    mx.concatenate=np.concatenate
    package=ModuleType('mlx');package.core=mx
    monkeypatch.setitem(sys.modules,'mlx',package);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    q=ModuleType('runtime.qwen35')
    def attn(x,w,p,cfg,kv,layer,offset,**kwargs):
        assert all('.mlp.' not in name for name in w)
        events.append(('attention',layer,offset,int(x.shape[1])))
        kv[layer]=kv.get(layer,0)+int(x.sum())
        if fail=='compute':raise RuntimeError('compute')
        return x+layer+1
    def mlp(x,w,p,cfg,layer,*args,**kwargs):
        assert all('.linear_attn.' not in name for name in w)
        events.append(('mlp',layer,int(x.shape[1])))
        return x*2
    q._qwen35_attention_residual=attn;q._qwen35_mlp_residual=mlp
    monkeypatch.setitem(sys.modules,'runtime.qwen35',q)
    core=ModuleType('runtime.engine')
    core._layer_transient_for_positions=lambda *a:(1,400_000_000)
    core._remaining_layer_transient_reserve=lambda *a:1
    core._recurring_layer_transient_reserve_margin=lambda *a:400_000_000
    core._resident_adjusted_transient=lambda *a:10
    monkeypatch.setitem(sys.modules,'runtime.engine',core)
    class Cache:
        pinned_bytes=0
        def trim_to(self,n):assert not loaded;events.append(('trim',n))
        def prepare_for(self,n):events.append(('prepare',n))
        def get(self,key,ns):
            assert not loaded;loaded.add(key);events.append(('get',key))
            return {n:1 for n in ns}
        def discard(self,key,ns):loaded.discard(key);events.append(('discard',key))
    def reserve(n,**kw):
        events.append(('reserve',n,kw))
        if fail=='reserve':raise MemoryError('admission')
    e=NS(rc=rc,cfg=cfg,_layer_names=names,_layer_fetch_bytes_estimate=lambda i,ns:100,
         governor=NS(reserve=reserve),cache=Cache(),_request_profiler=None,_dspark_tap_collector=None,
         _qwen35_split_prefill_stats={},_select_layer_transient=lambda *a:None,
         _note_true_peak=lambda:None,_get_experts=None,_iter_expert_batches=None,
         _transient_layer_signature=lambda i:'dense',timer=NS(add=lambda *a:None),
         _record_layer_transient=lambda *a:None,_restore_aggregate_layer_transient=lambda *a:None)
    return e,events,loaded


@pytest.mark.parametrize('tile',[1,2,8])
def test_complete_phase_order_same_tiled_math_and_kv(monkeypatch,tile):
    e,events,loaded=fake_engine(monkeypatch);x=np.arange(5).reshape(1,5,1);kv={}
    out=split.sweep(e,x,kv,3,tile)
    assert np.array_equal(out,((x+1)*2+2)*2)
    assert kv=={0:int(x.sum()),1:int(((x+1)*2).sum())}
    assert not loaded
    assert [x[0] for x in events if x[0] in ('get','discard')]==['get','discard']*4
    s=e._qwen35_split_prefill_stats
    assert s==dict(sweeps=1,attention_phases=2,mlp_phases=2,completed_layers=2,
                   maximum_declared_phase_page_bytes=100)


@pytest.mark.parametrize('failure',['reserve','compute'])
def test_failure_discards_current_phase_without_retry(monkeypatch,failure):
    e,events,loaded=fake_engine(monkeypatch,failure)
    with pytest.raises((RuntimeError,MemoryError)):
        split.sweep(e,np.ones((1,3,1)),{},0,2)
    assert not loaded
    assert sum(ev[0]=='get' for ev in events)<=1
    assert events[-1][0]=='discard'


def test_complete_layout_plan_checked_before_state_or_payload(monkeypatch):
    e,events,loaded=fake_engine(monkeypatch)
    e._layer_names=lambda i:names(i) if i==0 else ['bad']
    kv={}
    with pytest.raises(ValueError):split.sweep(e,np.ones((1,3,1)),kv,0,2)
    assert not events and not kv and not loaded


def test_profile_keys_and_structured_protocol_witness():
    before={};after={};base=['huihui-qwen38-27b-direct-head-rows-audit','qwen35-streaming-cache256']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['qwen35-prefill-split-weights'],environ=after)
    assert {k:v for k,v in after.items() if before.get(k)!=v}=={'VMODEL_QWEN35_PREFILL_SPLIT_WEIGHTS':'1'}
    witness={'completed_layers':64}
    for result in (_cache_phase_telemetry('generation',{'path_stats':{'qwen35_split_prefill_weights':witness}}),
                   _vision_protocol_timing({'path_stats':{'qwen35_split_prefill_weights':witness}})):
        assert result['qwen35_split_prefill_weights']==witness
    source=Path('runtime/server.py').read_text();tree=ast.parse(source)
    keys=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets)
          and isinstance(n.value,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='qwen35_mxfp4_head_rows' for x in n.value.elts)]
    assert len(keys)==2
    assert all(sum(isinstance(x,ast.Name) and x.id=='qwen35_split_prefill_request' for x in n.value.elts)==1 for n in keys)
    assert "'qwen35-split-prefill-v1' if self.rc.qwen35_prefill_split_weights else ''" in Path('runtime/engine.py').read_text()
