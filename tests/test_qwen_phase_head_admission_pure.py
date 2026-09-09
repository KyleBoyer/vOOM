"""Pre-fetch head lifetime ordering with real cache coordination, no MLX."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import qwen_phase_head_admission as admission
from runtime import weight_cache
from runtime.profiles import apply_runtime_profiles
from runtime.server import _cache_phase_telemetry, _vision_protocol_timing

ROOT = Path(__file__).resolve().parents[1]


def getter():
    tree = ast.parse((ROOT/'runtime/engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamingEngine')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_lm_head_weight')
    namespace = {'__name__':'runtime.engine', '__package__':'runtime'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
        '<actual-head-dispatch>','exec'),namespace)
    return namespace['_lm_head_weight']


def engine(monkeypatch, *, enabled=True, refuse=False, promotion=True, paused=False):
    events=[]
    monkeypatch.setattr(weight_cache, '_clear_device_cache', lambda: events.append(('clear',)))
    class Store:
        peak=0
        def fetch(self,names):
            assert len(names)==1
            size=4 if names==['lm_head.weight'] else 1 if names==['norm'] else 3
            self.peak=max(self.peak,cache.total_bytes+size)
            events.append(('fetch',names[0],cache.total_bytes))
            return {names[0]:SimpleNamespace(nbytes=size, exact_values=(1,2,3))},0.0,size
    store=Store(); cache=weight_cache.WeightCache(store,10)
    cache.pin('persistent',['norm'])
    cache.register_suspended_pin('qwen35:lm_head:persistent',4)
    for i in range(3): cache.get('body'+str(i),['body'+str(i)])
    prefetch=SimpleNamespace(paused=paused)
    def pause():
        events.append(('pause',)); prefetch.paused=True
    prefetch.pause_and_wait_idle=pause
    def reserve(size,**kwargs):
        assert prefetch.paused
        events.append(('reserve',size,kwargs,cache.total_bytes))
        if refuse: raise MemoryError('ordinary reservation refused')
    e=SimpleNamespace(rc=SimpleNamespace(qwen35_phase_head_pre_admit=enabled,
        qwen35_serial_verify_suspend_lm_head=True), cfg=SimpleNamespace(tie_word_embeddings=False),
        cache=cache, governor=SimpleNamespace(reserve=reserve), prefetcher=prefetch,
        _lm_head_w=None, _streamed_lm_head=None, _qwen35_lm_head_pin_suspended=True,
        _qwen35_lm_head_suspend_request_active=True, _qwen4_lm_head_pin_suspended=False,
        _qwen35_phase_head_admission_stats={})
    def restore(head):
        if enabled: assert prefetch.paused
        events.append(('promote',))
        if not promotion: return False
        page=cache.promote_to_pin('lm_head','qwen35:lm_head:persistent',
            tensors={'lm_head.weight':head})
        if page is not None:
            e._lm_head_w=page['lm_head.weight']; e._qwen35_lm_head_pin_suspended=False
        return page is not None
    e._restore_qwen35_serial_verify_lm_head=restore
    events.clear(); store.peak=0
    return e,events,store


def test_actual_dispatch_pretrims_before_fetch_and_restores_exact_same_head(monkeypatch,capsys):
    e,events,store=engine(monkeypatch)
    head=getter()(e)
    assert head is e._lm_head_w and head.exact_values==(1,2,3)
    assert store.peak==8  # evict two whole 3-byte pages from10, then load4
    assert e.cache.max_bytes==10 and e.cache.pinned_bytes==5
    assert [x[0] for x in events]==['pause','clear','reserve','fetch','promote']
    assert events[2]==('reserve',4,{'reason':'qwen35-phase-lm-head'},4)
    assert not e.prefetcher.paused and not e._qwen35_lm_head_pin_suspended
    row=json.loads(capsys.readouterr().out.split('] ',1)[1])
    assert row['outcome']=='loaded' and row['logical_trimmed_bytes']==6
    assert row['physical_release']=='unmeasured'
    stats=e._qwen35_phase_head_admission_stats
    assert stats['calls']==stats['fetches']==stats['pin_restores']==stats['reservation_calls']==1
    assert stats['logical_trimmed_bytes']==6 and stats['reservation_refusals']==0


def test_disabled_path_retains_old_post_fetch_eviction_and_no_governor_call(monkeypatch,capsys):
    e,events,store=engine(monkeypatch,enabled=False)
    assert getter()(e) is e._lm_head_w
    assert store.peak==14 and e.cache.max_bytes==10
    assert not any(x[0] in ('pause','reserve') for x in events)
    assert e._qwen35_phase_head_admission_stats=={} and capsys.readouterr().out==''


def test_reservation_refusal_is_before_fetch_even_after_logical_trim(monkeypatch,capsys):
    e,events,store=engine(monkeypatch,refuse=True)
    with pytest.raises(MemoryError,match='ordinary reservation refused'): getter()(e)
    assert not any(x[0] in ('fetch','promote') for x in events)
    assert e._lm_head_w is None and e._qwen35_lm_head_pin_suspended
    assert e.cache.suspended_pin_bytes('qwen35:lm_head:persistent')==4
    assert not e.prefetcher.paused and e.cache.max_bytes==10
    assert e._qwen35_phase_head_admission_stats['reservation_refusals']==1
    row=json.loads(capsys.readouterr().out.split('] ',1)[1])
    assert row['outcome']=='reservation-refused' and row['fetched'] is False


def test_owned_head_and_inactive_request_never_enter_new_path(monkeypatch):
    e,events,_=engine(monkeypatch)
    e._lm_head_w=object(); assert getter()(e) is e._lm_head_w and events==[]
    e._lm_head_w=None; e._qwen35_lm_head_suspend_request_active=False
    # Match the legacy dispatch, whose promote callback is not paused.
    e._restore_qwen35_serial_verify_lm_head=lambda head: False
    assert getter()(e).exact_values==(1,2,3)
    assert not any(x[0]=='reserve' for x in events)


@pytest.mark.parametrize('stage',['pause','trim','fetch','promote'])
def test_each_failure_restores_prefetch_state_and_preserves_original_error(monkeypatch,stage,capsys):
    e,events,_=engine(monkeypatch,paused=True)
    def fail(*a,**kw): raise OSError('original failure')
    target,key={'pause':(e.prefetcher,'pause_and_wait_idle'), 'trim':(e.cache,'trim_to'),
        'fetch':(e.cache,'get'), 'promote':(e,'_restore_qwen35_serial_verify_lm_head')}[stage]
    monkeypatch.setattr(target,key,fail)
    with pytest.raises(OSError,match='original failure'): getter()(e)
    assert e.prefetcher.paused is True and e._qwen35_phase_head_admission_stats['errors']==1
    assert json.loads(capsys.readouterr().out.split('] ',1)[1])['outcome']=='error'


def test_promotion_refusal_preserves_pass_through_without_refetch(monkeypatch):
    e,events,_=engine(monkeypatch,promotion=False)
    assert getter()(e).exact_values==(1,2,3)
    assert len([x for x in events if x[0]=='fetch'])==1
    assert e._lm_head_w is None and e._qwen35_lm_head_pin_suspended
    assert e._qwen35_phase_head_admission_stats['fetches']==1
    assert e._qwen35_phase_head_admission_stats['pin_restores']==0


@pytest.mark.parametrize('value',[0,-1,True,None,'4'])
def test_invalid_exact_lease_never_fetches_or_calls_governor(monkeypatch,value):
    e,events,_=engine(monkeypatch)
    monkeypatch.setattr(e.cache,'suspended_pin_bytes',lambda key:value)
    with pytest.raises(ValueError,match='exact positive lease'): getter()(e)
    assert events==[]


@pytest.mark.parametrize('enabled,parent',[(1,True),('1',True),(None,True),(True,False)])
def test_policy_fails_closed(enabled,parent):
    with pytest.raises(ValueError): admission.validate_policy(enabled,parent)


def test_missing_governor_never_fetches_and_logging_failure_does_not_change_output(monkeypatch):
    e,events,_=engine(monkeypatch); e.governor=None
    with pytest.raises(ValueError,match='live governor'): getter()(e)
    assert events==[]
    e,events,_=engine(monkeypatch)
    def fail(*a,**kw): raise OSError('sink unavailable')
    monkeypatch.setattr('builtins.print',fail)
    assert getter()(e).exact_values==(1,2,3)


def test_new_governor_pause_is_not_undone_on_success_or_refusal(monkeypatch):
    for refuse in (False,True):
        e,events,_=engine(monkeypatch)
        def reserve(*a,**kw):
            e.governor.paused_prefetch=True
            if refuse: raise MemoryError('pressure')
        e.governor.reserve=reserve
        if refuse:
            with pytest.raises(MemoryError): getter()(e)
        else:
            getter()(e)
        assert e.prefetcher.paused and e.governor.paused_prefetch


def test_profile_is_only_new_flag_and_phase_and_api_metrics_preserve_missing():
    env={}; apply_runtime_profiles(['qwen35-phase-head-pre-admit'],environ=env)
    assert env=={'VMODEL_QWEN35_PHASE_HEAD_PRE_ADMIT':'1'}
    stats={'qwen35_phase_head_pre_admit_enabled':1,
           'qwen35_phase_head_admission':{'calls':3,'reservation_refusals':1}}
    for observed in (_cache_phase_telemetry('gateway_decision',{'path_stats':stats}),
                     _vision_protocol_timing({'path_stats':stats})):
        assert all(observed[k]==v for k,v in stats.items())
    assert 'qwen35_phase_head_admission' not in _cache_phase_telemetry('gateway_decision',{})


def test_engine_server_mtp_identity_and_reset_wiring():
    source=(ROOT/'runtime/server.py').read_text()
    tree=ast.parse(source)
    keys=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets)
        and isinstance(n.value,ast.Tuple) and any(isinstance(x,ast.Name)
            and x.id=='qwen35_suspend_lm_head_request' for x in n.value.elts)]
    assert len(keys)==2 and all(any(isinstance(x,ast.Name) and x.id=='qwen35_head_pre_admit_request'
        for x in n.value.elts) for n in keys)
    assert 'rc.qwen35_phase_head_pre_admit = (qwen35_head_pre_admit_request == "1")' in source
    engine_source=(ROOT/'runtime/engine.py').read_text()
    assert engine_source.count('self._qwen35_phase_head_admission_stats = {}')==2
    assert 'qwen35_phase_head_pre_admit: bool = False' in engine_source
    assert 'qwen35_phase_head_pre_admit=run.get("qwen35_phase_head_pre_admit", False)' in engine_source
    mtp=(ROOT/'runtime/qwen35_mtp.py').read_text()
    assert 'tgt._qwen35_phase_head_admission_stats)' in mtp
    assert 'or reason == "qwen35-phase-lm-head"' in (ROOT/'runtime/pressure.py').read_text()


@pytest.mark.parametrize('settles',[False,True])
def test_real_governor_new_reason_restores_empty_budget_without_invented_release(settles):
    from tests.test_governor_reserve_pure import load_pressure, make_governor
    module,metal=load_pressure(4_995_000_000)
    gov=make_governor(module,metal,cache_max=2_200_000_000,floor=100_000_000)
    gov.metal_limit=5_000_000_000; gov.cache.total_bytes=0
    module.psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=6_000_000_000))
    module.time=SimpleNamespace(sleep=lambda _:None)
    if settles:
        metal.clear_release_after=4; metal.clear_release_bytes=700_000_000
        gov.reserve(675_430_400,margin=0,reason='qwen35-phase-lm-head')
        assert gov.reservation_failures==0
        assert gov.reservation_budget_restored_bytes==gov.reservation_budget_reduced_bytes
    else:
        with pytest.raises(MemoryError,match='qwen35-phase-lm-head'):
            gov.reserve(675_430_400,margin=0,reason='qwen35-phase-lm-head')
        assert gov.reservation_failures==1
    assert gov.cache.max_bytes==2_200_000_000 and gov.cache.total_bytes==0
    assert gov.reservation_cache_released_bytes==0 and gov.reservation_zero_release_short_circuits==1
    assert gov.prefetcher.paused and gov.reservation_reason_counts=={'qwen35-phase-lm-head':1}


def test_real_governor_new_reason_keeps_ordinary_400mb_margin():
    from tests.test_governor_reserve_pure import load_pressure, make_governor
    module,metal=load_pressure(400_000_000)
    gov=make_governor(module,metal,cache_max=256_000_000,floor=64_000_000)
    module.psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=5_700_000_000))
    gov.reserve(675_430_400,reason='qwen35-phase-lm-head')
    assert gov.reservation_calls==gov.reservation_fast_path_calls==1
    assert gov.reservations==metal.clears==0 and gov.cache.max_bytes==256_000_000
