"""CPU allocator hook tests; native reclamation and GC are mocked."""
import ast
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from runtime import host_allocator_relief as relief
from runtime.profiles import apply_runtime_profiles


def record():
    sample=lambda n:dict(monotonic_ns=n,system_available_bytes=7_000_000_000,
        process=dict(available=True,pid=7,monotonic_start_ns=n+1,monotonic_end_ns=n+2))
    return dict(schema=relief.SCHEMA,available=True,error_type=None,pid=7,
        started_ns=1,ended_ns=20,gc_collected=3,allocator_reported_released_bytes=42,
        before=sample(2),after_gc=sample(6),after_relief=sample(10))


def test_disabled_has_no_reclaim_or_output(monkeypatch,capsys):
    monkeypatch.setattr(relief,'reclaim',lambda:pytest.fail('disabled action'))
    for env in ({},{relief.FLAG:'0'}):
        assert relief.run_if_enabled(env) is None
    assert capsys.readouterr().out == ''


def test_enabled_emits_exact_scalar_receipt(monkeypatch,capsys):
    expected=record(); monkeypatch.setattr(relief,'reclaim',lambda:expected)
    assert relief.run_if_enabled({relief.FLAG:'1'}) is expected
    assert relief.summarize(capsys.readouterr().out,expected_count=1)['passed']


def test_binding_uses_documented_current_process_abi(monkeypatch):
    fn=NS(); calls=[]
    monkeypatch.setattr(relief.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(relief.ctypes,'CDLL',lambda path:(calls.append(path) or NS(malloc_zone_pressure_relief=fn)))
    relief._bindings.cache_clear()
    try:
        assert relief._bindings()[1] is fn
        assert fn.argtypes == [ctypes.c_void_p,ctypes.c_size_t]
        assert fn.restype is ctypes.c_size_t
        assert calls == ['/usr/lib/libSystem.B.dylib']
    finally:
        relief._bindings.cache_clear()


def test_reclaim_order_and_null_zone_zero_goal(monkeypatch):
    events=[]
    monkeypatch.setattr(relief,'_bindings',lambda:(None,lambda zone,goal:(events.append(('relief',zone,goal)) or 42)))
    monkeypatch.setattr(relief.gc,'collect',lambda:(events.append('gc') or 3))
    monkeypatch.setattr(relief,'_sample',lambda:(events.append('sample') or {'process':{'available':True}}))
    result=relief.reclaim()
    assert events == ['sample','gc','sample',('relief',None,0),'sample']
    assert result['available'] and result['gc_collected']==3 and result['allocator_reported_released_bytes']==42


def test_unavailable_binding_cannot_claim_success_or_collect(monkeypatch):
    def fail(): raise RuntimeError('missing')
    monkeypatch.setattr(relief,'_bindings',fail)
    monkeypatch.setattr(relief.gc,'collect',lambda:pytest.fail('binding failure must precede GC'))
    assert relief.reclaim()['available'] is False


@pytest.mark.parametrize('key,value',[('available',False),('error_type','RuntimeError'),
    ('allocator_reported_released_bytes',None),('gc_collected',True),('ended_ns',3)])
def test_invalid_coverage_rejects(key,value):
    r=record();r[key]=value
    assert not relief.summarize(relief.PREFIX+json.dumps(r),expected_count=1)['passed']


def test_missing_extra_malformed_or_misaligned_coverage_rejects():
    line=relief.PREFIX+json.dumps(record())
    for text in ('',line+'\n'+line,relief.PREFIX+'bad'):
        assert not relief.summarize(text,expected_count=1)['passed']
    r=record();r['after_gc']['process']['pid']=8
    assert not relief.summarize(relief.PREFIX+json.dumps(r),expected_count=1)['passed']
    r=record();r['after_gc']['monotonic_ns']=1
    assert not relief.summarize(relief.PREFIX+json.dumps(r),expected_count=1)['passed']


def test_zero_reported_reclamation_is_valid_coverage_not_a_win():
    r=record();r['allocator_reported_released_bytes']=0
    summary=relief.summarize(relief.PREFIX+json.dumps(r),expected_count=1)
    assert summary['passed'] and summary['allocator_reported_released_bytes']==0


def test_profile_only_adds_explicit_hook_and_hook_runs_inside_inference_lock():
    before={};after={};base=['huihui-qwen38-27b-direct-head-rows-audit']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['host-allocator-relief'],environ=after)
    assert {k:v for k,v in after.items() if before.get(k)!=v} == {relief.FLAG:'1'}
    tree=ast.parse(Path('runtime/server.py').read_text())
    handler=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Handler')
    method=next(n for n in handler.body if isinstance(n,ast.FunctionDef) and n.name=='do_POST')
    source=ast.unparse(method)
    assert source.index('INFER_LOCK.acquire(')<source.index('run_if_enabled()')<source.index('self._do_post_locked()')<source.index('INFER_LOCK.release()')
    assert 'mlx' not in Path('runtime/host_allocator_relief.py').read_text().lower().split('from __future__')[1]
