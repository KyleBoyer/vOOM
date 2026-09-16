import copy
import ctypes
import json
from types import SimpleNamespace as NS

import pytest

from runtime import system_swap as swap
from tests.fixtures.captured_transition_tracking_gate import native_pressure_summary


def record(actual=0,paging=100):
    raw=swap.VMStatistics64Rev1();raw.swapins=2;raw.swapouts=actual
    raw.pageins=999;raw.pageouts=paging
    return swap.decode(raw,returned_count=38,kernel_code=0,page_size=16384)


def test_rev1_layout_matches_separately_compiled_installed_sdk():
    assert ctypes.sizeof(swap.VMStatistics64Rev1)==152 and swap.REV1_COUNT==38
    assert swap.VMStatistics64Rev1.swapins.offset==112
    assert swap.VMStatistics64Rev1.swapouts.offset==120


def test_paging_is_not_substituted_for_actual_swap():
    r=record(3,1000)
    assert r['swap_out_bytes']==3*16384 and r['page_out_bytes']==1000*16384
    summary=swap.summarize_native([record(3,1000),record(3,5000)])
    assert summary['available'] and summary['swap_out_growth_bytes']==0
    assert summary['page_out_growth_bytes']==4000*16384


@pytest.mark.parametrize('change',[dict(returned_count=37),dict(kernel_code=5),
    dict(page_size=0),dict(page_size=True),dict(page_size=3000)])
def test_short_or_invalid_native_data_is_never_zero_swap(change):
    with pytest.raises((ValueError,RuntimeError)):
        swap.decode(swap.VMStatistics64Rev1(),**dict(
            dict(returned_count=38,kernel_code=0,page_size=16384),**change))


@pytest.mark.parametrize('fault',['missing','wrong_source','bool','unaligned','reset','page_size'])
def test_incomplete_or_mixed_series_fails(fault):
    records=[record(3),record(4)]
    if fault=='missing':records[1]={}
    if fault=='wrong_source':records[1]['source']='psutil pageouts'
    if fault=='bool':records[1]['swap_out_bytes']=True
    if fault=='unaligned':records[1]['swap_out_bytes']+=1
    if fault=='reset':records[1]=record(2)
    if fault=='page_size':records[1]['page_size_bytes']=4096
    assert not swap.summarize_native(records)['available']


def test_wrapper_uses_native_and_retains_psutil_totals(monkeypatch):
    monkeypatch.setattr(swap.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(swap.psutil,'swap_memory',lambda:NS(total=10,used=2,free=8,sin=999,sout=888))
    monkeypatch.setattr(swap,'native_counters',lambda:record(3))
    r=swap.swap_memory()
    assert (r.total,r.used,r.free,r.sin,r.sout)==(10,2,8,32768,49152)
    def unavailable(): raise RuntimeError('native missing')
    monkeypatch.setattr(swap,'native_counters',unavailable)
    with pytest.raises(RuntimeError):swap.swap_memory()
    assert swap.sample_native_counters()['available'] is False


def test_actual_pressure_gate_keeps_16mb_threshold_and_legacy_diagnostic():
    def log(actual_end):
        return '\n'.join('[process-memory] '+json.dumps(dict(system_available_bytes=6_000_000_000,
            system_swap_used_bytes=100,system_swap_out_bytes=page*16384,
            process=dict(available=True,physical_footprint_bytes=100,internal_compressed_ledger_bytes=0),
            actual_swap_counters=record(actual,page))) for actual,page in [(3,100),(actual_end,5000)])
    good=native_pressure_summary(log(3),require_actual_swap_counters=True)
    assert good['passed'] and good['actual_swap_out_growth_bytes']==0
    assert good['legacy_psutil_out_growth_bytes']==4900*16384
    bad=native_pressure_summary(log(1003),require_actual_swap_counters=True)
    assert not bad['passed'] and bad['actual_swap_out_growth_bytes']==16_384_000
    row=json.loads(log(3).splitlines()[0].split('] ',1)[1]);row.pop('actual_swap_counters')
    assert not native_pressure_summary('[process-memory] '+json.dumps(row),
        require_actual_swap_counters=True)['passed']


def test_http_requires_native_source_and_monotonic_bytes():
    a=dict(swap_counter_source=swap.SOURCE,swap_out_bytes=100,page_out_bytes=1000)
    assert swap.http_identity(a,a)
    assert not swap.http_identity(a,{**a,'swap_counter_source':'legacy'})
    assert not swap.http_identity(a,{**a,'swap_out_bytes':0})


def test_preflight_and_http_sampler_publish_counter_identity(monkeypatch,tmp_path):
    from runtime import memory_preflight
    from tests.fixtures import runtime_profile_http_gate as gate
    value=NS(total=10,used=2,free=8,sin=16384,sout=32768,source=swap.SOURCE,
             page_in_bytes=999999,page_out_bytes=888888)
    monkeypatch.setattr(memory_preflight,'swap_memory',lambda:value)
    monkeypatch.setattr(gate,'swap_memory',lambda:value)
    p=memory_preflight.capture(tmp_path);h=gate._pressure()
    assert p.swap_out_bytes==h.swap_out_bytes==32768
    assert p.swap_counter_source==h.swap_counter_source==swap.SOURCE
    assert p.page_out_bytes==h.page_out_bytes==888888


@pytest.mark.parametrize('flag',['0','1'])
def test_periodic_observer_keeps_governor_counter_and_adds_native_only_when_enabled(monkeypatch,capsys,flag):
    from runtime import process_memory_witness as witness
    monkeypatch.setenv(swap.FLAG,flag)
    monkeypatch.setattr(witness,'sample_self_memory',lambda:dict(available=True))
    calls=[]
    def sample():
        calls.append(1)
        return record(3)
    monkeypatch.setattr(swap,'sample_native_counters',sample)
    witness.GovernorProcessMemoryObserver().record(governor_monotonic_s=1,
        system_available_bytes=6000,system_swap_used_bytes=100,system_swap_out_bytes=999,
        metal_active_bytes=100,cache_budget_bytes_after_response=200,swap_pressure_response=False)
    row=json.loads(capsys.readouterr().out.removeprefix('[process-memory] '))
    assert row['system_swap_out_bytes']==999
    assert len(calls)==int(flag)
    assert ('actual_swap_counters' in row)==(flag=='1')
    if flag=='1':assert row['actual_swap_counters']['swap_out_bytes']==49152


@pytest.mark.parametrize('fault',[None,'source','flag','interior'])
def test_qualification_requires_native_preflight_and_observer(fault):
    from tests.fixtures.huihui_memory_policy import validate
    config=dict(require_actual_swap_counters=True)
    env={swap.FLAG:'1'}
    point=dict(swap_counter_source=swap.SOURCE)
    pre=dict(start=point.copy(),end=point.copy(),pressure_window=dict(samples=[point.copy()]))
    if fault=='source':pre['start']['swap_counter_source']='legacy'
    if fault=='interior':pre['pressure_window']['samples'][0]['swap_counter_source']='legacy'
    if fault=='flag':env[swap.FLAG]='0'
    if fault is None:assert validate(config,env,pre)==5_300_000_000
    else:
        with pytest.raises(AssertionError):validate(config,env,pre)
