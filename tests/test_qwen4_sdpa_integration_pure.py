"""Real source hooks and serving/persistent identity without MLX."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import qwen4_sdpa_tiling as helper
from runtime.profiles import apply_runtime_profiles

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('value',[0,128,256,512])
def test_strict_setting_and_off_preserves_old_persistent_identity(value):
    assert helper.validate_prefill_query_tile(value)==value
    assert helper.parse_prefill_query_tile(str(value))==value
    assert helper.prefill_query_tile_identity(value)==(f'qwen4prefillsdpa{value}' if value else '')
    assert helper.parse_prefill_query_tile(None)==0


@pytest.mark.parametrize('value',[True,False,None,64,1,-1,128.0,'256'])
def test_direct_invalid_setting_is_rejected(value):
    with pytest.raises(ValueError):helper.validate_prefill_query_tile(value)


@pytest.mark.parametrize('value',['','auto','0256','256.0',' 256',256,True])
def test_environment_invalid_setting_is_rejected(value):
    with pytest.raises(ValueError):helper.parse_prefill_query_tile(value)


@pytest.mark.parametrize('tile',[0,256])
def test_actual_qsa_dispatch_keeps_zero_on_original_call(monkeypatch,tile):
    tree=ast.parse((ROOT/'runtime/qwen4_exp.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qsa_attention')
    branch=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='sdpa_query_tile')
    events=[];sentinel=object();q,k,v,mask=object(),object(),object(),object();stats={}
    monkeypatch.setattr(helper,'query_tiled_sdpa',lambda *a,**kw:events.append(('tiled',a,kw)) or sentinel)
    metal=SimpleNamespace(fast=SimpleNamespace(scaled_dot_product_attention=lambda *a,**kw:events.append(('original',a,kw)) or sentinel))
    ns=dict(__package__='runtime',mx=metal,query=q,keys=k,values=v,mask=mask,dim=256,sdpa_query_tile=tile,sdpa_stats=stats)
    exec(compile(ast.Module(body=[copy.deepcopy(branch)],type_ignores=[]),'<actual-qsa-dispatch>','exec'),ns)
    assert ns['attended'] is sentinel
    assert len(events)==1 and events[0][0]==('tiled' if tile else 'original')
    assert events[0][2]==(dict(scale=.0625,mask=mask,tile_queries=256,stats=stats) if tile else dict(scale=.0625,mask=mask))


def test_profile_and_server_engine_identity_include_only_explicit_setting(monkeypatch):
    env={};apply_runtime_profiles(['qwen4-prefill-sdpa-query256'],environ=env)
    assert env=={'VMODEL_QWEN4_PREFILL_SDPA_QUERY_TILE':'256'}
    tree=ast.parse((ROOT/'runtime/server.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qwen4_request_identity' for t in n.targets))
    expression=compile(ast.Expression(assignment.value),'<actual-engine-identity>','eval')
    monkeypatch.delenv('VMODEL_QWEN4_PREFILL_SDPA_QUERY_TILE',raising=False)
    old=eval(expression,{'os':os});monkeypatch.setenv('VMODEL_QWEN4_PREFILL_SDPA_QUERY_TILE','256')
    new=eval(expression,{'os':os});assert len(old)==len(new)==41 and old[:-1]==new[:-1]
    assert (old[-1],new[-1])==('0','256')


def test_only_host_spooled_prefill_passes_the_candidate_not_decode_or_mtp():
    tree=ast.parse((ROOT/'runtime/engine.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(k.arg=='sdpa_query_tile' for k in n.keywords)]
    assert len(calls)==1 and ast.unparse(calls[0].func)=='qwen4_attention_branch'
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamingEngine')
    sweep=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_layer_stationary_qwen4_sweep')
    assert calls[0] in list(ast.walk(sweep))
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    text=ast.unparse(init)
    assert text.index('validate_prefill_query_tile(')<text.index('WeightStore(')
    fp=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_kv_fingerprint')
    assert 'prefill_query_tile_identity(self.rc.qwen4_prefill_sdpa_query_tile)' in ast.unparse(fp)


def test_actual_helper_reports_calls_and_evaluated_query_ranges():
    from tests.test_qwen4_sdpa_tiling_pure import Array
    events=[];q=Array((1,24,1031,256),'q',events);k=Array((1,2,32768,256),'k',events)
    mx=SimpleNamespace(bfloat16='bf16',float16='fp16',fast=SimpleNamespace(scaled_dot_product_attention=lambda q,*a,**kw:q),
        eval=lambda a:None,concatenate=lambda a,axis:None)
    stats={};helper.query_tiled_sdpa(mx,q,k,k,scale=.0625,mask=None,tile_queries=256,stats=stats)
    assert stats==dict(calls=1,split_calls=1,query_tiles=4,max_query_rows=263,max_key_positions=32768)


def test_real_weight_gate_crosses_qsa_budget_and_preserves_continuation_scope():
    from tests.fixtures.qwen4_real_qsa_tiling_gate import CASES
    assert CASES==((3,3079,61783),(47,32799,94217))
    assert all(total>2048 and total%1024 for _,total,_ in CASES)
    assert len({seed for _,_,seed in CASES})==len(CASES)
