"""Bounded second-spill policy, evidence, and configuration; no real MLX."""

import ast
import copy
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from runtime import qwen_kv_reclaim as recovery
from tests.test_qwen_kv_reclaim_pure import FakePaged, run
from tests.test_qwen_kv_reclaim_witness_pure import BUDGET, KEY, trace, response, log
from tests.fixtures.qwen_kv_reclaim_witness import valid_trace, phase_checks, log_coverage, validate_config


@pytest.fixture
def controlled(monkeypatch):
    def build(*, scale=1, active=(100, 40, 42, 22), ceilings=(150, 145, 137, 137),
              releases=(60, 20), refuse=False, enabled=True, error_pass=None):
        calls = []
        module = ModuleType('runtime.kv_paged')
        module.PagedKVCache = FakePaged
        monkeypatch.setitem(sys.modules, 'runtime.kv_paged', module)
        samples = iter(zip(active, ceilings))
        current = [None]
        def memory():
            current[0] = next(samples)
            return current[0][0] * scale
        metal = SimpleNamespace(get_active_memory=memory)
        monkeypatch.setattr(recovery.psutil, 'virtual_memory',
            lambda: SimpleNamespace(available=1000 * scale))
        def reserve(n, **kw):
            calls.append(('reserve', n, kw))
            if refuse:
                raise MemoryError('still unsafe')
        gov = SimpleNamespace(_metal_ceiling=lambda a, v: current[0][1] * scale, reserve=reserve)
        target = SimpleNamespace(rc=SimpleNamespace(qwen35_serial_kv_reclaim=True,
                qwen35_serial_kv_reclaim_topup=enabled),
            cfg=SimpleNamespace(model_type='qwen3_5'), governor=gov,
            _layer_transient=90 * scale, _layer_transient_margin=20 * scale)
        kv = FakePaged(calls)
        kv.resident = 150 * scale
        amounts = iter(releases)
        def spill(requested, *, protected_layer):
            calls.append(('spill', requested, protected_layer))
            amount = next(amounts) * scale if requested else 0
            kv.resident -= amount
            kv.stats.spills += int(amount > 0)
            if error_pass == len([c for c in calls if c[0] == 'spill']):
                raise OSError('partial spill failed')
            return amount
        kv.reclaim_closed_pages = spill
        return target, kv, metal, calls
    return build


@pytest.mark.parametrize('scale', [1, 1000, 1_000_000])
def test_fresh_deficit_not_previous_size_or_fixed_padding(controlled, scale, capsys):
    target, kv, metal, calls = controlled(scale=scale)
    assert run(target, kv, metal)
    assert calls == [('spill', 60*scale, 3), ('spill', 15*scale, 3),
        ('reserve', 90*scale, dict(margin=20*scale, reason='serial-verify-transient'))]
    stats = target._qwen35_serial_kv_reclaim_stats
    row = stats['records'][0]
    assert row['schema'].endswith('.v2') and row['topup_enabled'] is True
    assert row['topup_check']['deficit_bytes'] == 15*scale
    assert row['reclaim_passes'][0]['after_reclaim']['deficit_bytes'] == 5*scale
    assert row['logical_reclaimed_bytes'] == 80*scale
    # A non-atomic between-pass active increase means these sums DIFFER.
    assert row['metal_active_released_bytes'] == 78*scale
    assert sum(p['metal_active_released_bytes'] for p in row['reclaim_passes']) == 80*scale
    assert kv.max_bytes == BUDGET and kv.nbytes() == 70*scale
    assert valid_trace(stats, budget_bytes=BUDGET)
    doc = response(stats)
    for part in [*doc['vmodel_cache_phases'], doc['vmodel_timing']]:
        part[KEY+'_topup_enabled'] = 1
    assert log_coverage([doc], capsys.readouterr().out,
        budget_bytes=BUDGET, topup_required=True)['passed']


@pytest.mark.parametrize('enabled', [False, None, 1, '1', 'auto'])
def test_only_explicit_bool_enables_second_pass(controlled, enabled):
    target, kv, metal, calls = controlled(enabled=enabled, refuse=True)
    assert run(target, kv, metal) is False
    assert len(calls) == 2 and calls[-1][0] == 'reserve'
    assert target._qwen35_serial_kv_reclaim_stats['records'][0]['schema'].endswith('.v1')


@pytest.mark.parametrize('active,releases,ceilings,expected_calls', [
    ((100, 100), (60,), (150, 150), 2),  # retained aliases
    ((100, 110), (60,), (150, 150), 2),  # active increased
    ((100, 100), (0,), (150, 150), 1),   # exhausted eligible pages
    ((100, 40), (60,), (150, 150), 2),   # no remaining deficit
    ((100, 40, 40), (60,), (150, 145, 150), 2),  # recovered at fresh check
])
def test_no_progress_or_no_fresh_deficit_never_spills_again(
        controlled, active, releases, ceilings, expected_calls):
    target, kv, metal, calls = controlled(active=active, releases=releases,
        ceilings=ceilings, refuse=True)
    assert run(target, kv, metal) is False
    assert len(calls) == expected_calls and sum(c[0]=='spill' for c in calls) == 1
    row = target._qwen35_serial_kv_reclaim_stats['records'][0]
    assert len(row['reclaim_passes']) == 1
    assert row['reservation_retried'] is (expected_calls == 2)


@pytest.mark.parametrize('releases,active', [((60, 0), (100, 40, 42, 42)),
                                          ((60, 20), (100, 40, 42, 22))])
def test_two_spills_is_hard_limit_and_reserve_still_owns_failure(controlled, releases, active):
    target, kv, metal, calls = controlled(releases=releases, active=active,
        ceilings=(150, 145, 137, 100), refuse=True)
    assert run(target, kv, metal) is False
    row = target._qwen35_serial_kv_reclaim_stats['records'][0]
    assert len(calls) == 3 and len(row['reclaim_passes']) == 2
    assert row['outcome'] == 'refused' and row['reservation_retried'] is True
    assert row['after_reclaim']['deficit_bytes'] > 0
    assert not valid_trace(target._qwen35_serial_kv_reclaim_stats, budget_bytes=BUDGET)


@pytest.mark.parametrize('error_pass', [1, 2])
def test_partial_write_errors_propagate_and_do_not_admit(controlled, error_pass):
    target, kv, metal, calls = controlled(error_pass=error_pass)
    with pytest.raises(OSError, match='partial spill failed'):
        run(target, kv, metal)
    row = target._qwen35_serial_kv_reclaim_stats['records'][0]
    assert all(c[0] == 'spill' for c in calls) and len(calls) == error_pass
    assert row['logical_reclaimed_bytes'] == (60 if error_pass == 1 else 80)
    assert row['spill_pages'] == error_pass
    assert row['outcome'] == 'error' and not row['reservation_retried']
    if error_pass == 2:
        assert row['reclaim_passes'][-1]['logical_reclaimed_bytes'] == 20
    assert not valid_trace(target._qwen35_serial_kv_reclaim_stats, budget_bytes=BUDGET)


def test_invalid_fresh_snapshot_stops_before_second_spill(controlled):
    target, kv, metal, calls = controlled(active=(100, 40, -1))
    with pytest.raises(ValueError, match='invalid serial KV'):
        run(target, kv, metal)
    assert calls == [('spill', 60, 3)]


@pytest.fixture
def valid_topup(controlled):
    target, kv, metal, _ = controlled()
    assert run(target, kv, metal)
    return target._qwen35_serial_kv_reclaim_stats['records'][0]


@pytest.mark.parametrize('mutation', ['missing_pass', 'three_passes', 'missing_check',
    'bool_enabled', 'stale_size', 'stale_check', 'first_before', 'last_after',
    'second_logical_before', 'physical_sum', 'bad_seconds', 'missing_spill',
    'float_count', 'wrong_schema', 'first_no_progress', 'fresh_zero_deficit'])
def test_v2_rejects_forged_or_incomplete_evidence(valid_topup, mutation):
    row = copy.deepcopy(valid_topup)
    parts = row['reclaim_passes']
    if mutation == 'missing_pass': parts.pop()
    elif mutation == 'three_passes': parts.append(copy.deepcopy(parts[-1]))
    elif mutation == 'missing_check': row.pop('topup_check')
    elif mutation == 'bool_enabled': row['topup_enabled'] = 1
    elif mutation == 'stale_size': parts[-1]['requested_bytes'] = 5
    elif mutation == 'stale_check': row['topup_check'] = parts[0]['after_reclaim']
    elif mutation == 'first_before': parts[0]['before'] = parts[0]['after_reclaim']
    elif mutation == 'last_after': row['after_reclaim'] = parts[0]['after_reclaim']
    elif mutation == 'second_logical_before': parts[-1]['logical_before_bytes'] += 1
    elif mutation == 'physical_sum': row['metal_active_released_bytes'] = 80
    elif mutation == 'bad_seconds': parts[-1]['reclaim_seconds'] = float('nan')
    elif mutation == 'missing_spill': parts[-1]['spill_pages'] = 0
    elif mutation == 'float_count': parts[-1]['logical_reclaimed_bytes'] = 20.0
    elif mutation == 'wrong_schema': row['schema'] = 'voom.qwen35-serial-kv-reclaim.v1'
    elif mutation == 'first_no_progress': parts[0]['metal_active_released_bytes'] = 0
    elif mutation == 'fresh_zero_deficit': row['topup_check']['deficit_bytes'] = 0
    assert not valid_trace(trace([row]), budget_bytes=BUDGET)


@pytest.mark.parametrize('where,value', [('hidden', None), ('hidden', True),
    ('hidden', 0), ('final', None), ('final', True), ('final', 0)])
def test_all_phases_and_final_require_topup_flag_even_when_inactive(where, value):
    doc = response({}, {})
    for part in [*doc['vmodel_cache_phases'], doc['vmodel_timing']]:
        part[KEY+'_topup_enabled'] = 1
    assert all(phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET, topup_required=True).values())
    part = doc['vmodel_cache_phases'][0] if where == 'hidden' else doc['vmodel_timing']
    part[KEY+'_topup_enabled'] = value
    assert not all(phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET, topup_required=True).values())


def test_topup_config_cannot_accept_legacy_success_trace():
    doc = response(trace())
    for part in [*doc['vmodel_cache_phases'], doc['vmodel_timing']]:
        part[KEY+'_topup_enabled'] = 1
    assert not all(phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET, topup_required=True).values())


@pytest.mark.parametrize('active,ceilings,releases', [
    ((100, 40), (150, 150), (60,)),
    ((100, 40, 40), (150, 145, 150), (60,)),
    ((100, 40, 42, 42), (150, 145, 137, 137), (60, 0)),
])
def test_inactive_or_exhausted_topup_can_only_pass_after_ordinary_reserve(
        controlled, active, ceilings, releases):
    target, kv, metal, calls = controlled(active=active, ceilings=ceilings, releases=releases)
    assert run(target, kv, metal)
    assert calls[-1][0] == 'reserve'
    assert valid_trace(target._qwen35_serial_kv_reclaim_stats, budget_bytes=BUDGET)


@pytest.mark.parametrize('driver', ['full', 'short'])
def test_both_actual_response_gates_enforce_topup_flag(driver):
    from tests.fixtures.huihui_captured_action_gate import acceptance
    from tests.test_huihui_captured_action_gate_pure import row
    from tests.fixtures.captured_transition_tracking_gate import row_checks
    from tests.test_captured_transition_tracking_gate_pure import valid_row
    doc = response({})
    config = dict(profiles=['audit'], profile_digest='digest',
        require_serial_kv_reclaim=True, require_serial_kv_reclaim_topup=True,
        serial_kv_budget_bytes=BUDGET)
    def check():
        if driver == 'full':
            value = row()
            value['timing'].update(doc['vmodel_timing'])
            return acceptance(value, doc, config, initial_action=False)
        value = valid_row()
        value['timing'].update(doc['vmodel_timing'])
        return row_checks(value, doc, dict(kind='short_title', topic='node'), config)
    assert not check()['all_phase_serial_kv_reclaim_witness']
    for part in [*doc['vmodel_cache_phases'], doc['vmodel_timing']]:
        part[KEY+'_topup_enabled'] = 1
    assert check()['all_phase_serial_kv_reclaim_witness']


@pytest.mark.parametrize('config,env,valid', [
    ({}, {}, True),
    ({'require_serial_kv_reclaim': True}, {'VMODEL_QWEN35_SERIAL_KV_RECLAIM':'1'}, True),
    ({'require_serial_kv_reclaim': True, 'require_serial_kv_reclaim_topup': True},
     {'VMODEL_QWEN35_SERIAL_KV_RECLAIM':'1', 'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'1'}, True),
    ({}, {'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'1'}, False),
    ({'require_serial_kv_reclaim_topup': True}, {'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'1'}, False),
    ({'require_serial_kv_reclaim': True, 'require_serial_kv_reclaim_topup': True},
     {'VMODEL_QWEN35_SERIAL_KV_RECLAIM':'1'}, False),
    ({'require_serial_kv_reclaim_topup': 1}, {'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'1'}, False),
    ({}, {'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'auto'}, False),
])
def test_gates_require_matching_strict_flags(config, env, valid):
    if valid:
        validate_config(config, env)
    else:
        with pytest.raises(AssertionError):
            validate_config(config, env)


def test_profile_delta_default_off_and_protocol_exports():
    from runtime.profiles import discover_runtime_profiles, resolve_runtime_profiles
    from runtime.server import _cache_phase_telemetry, _vision_protocol_timing
    root = Path(__file__).resolve().parents[1]
    catalog = discover_runtime_profiles((root/'profiles',))
    base = ('huihui-qwen38-27b-full-workflow-lifetime-audit', 'qwen35-serial-kv-reclaim')
    _, before = resolve_runtime_profiles(base, catalog)
    _, after = resolve_runtime_profiles((*base, 'qwen35-serial-kv-reclaim-topup'), catalog)
    assert after == {**before, 'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP':'1'}
    tree = ast.parse((root/'runtime/engine.py').read_text())
    rc = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='RuntimeConfig')
    field = next(n for n in rc.body if isinstance(n, ast.AnnAssign)
        and n.target.id=='qwen35_serial_kv_reclaim_topup')
    assert field.value.value is False
    for value in (0, 1):
        stats = {KEY+'_topup_enabled': value}
        result = dict(path_stats=stats)
        assert _cache_phase_telemetry('gateway_decision', result)[KEY+'_topup_enabled'] == value
        assert _vision_protocol_timing(result)[KEY+'_topup_enabled'] == value


@pytest.mark.parametrize('value,parent,valid', [('0','0',True), ('1','1',True),
    ('1','0',False), ('1','true',False), ('auto','1',False), ('true','1',False), (' 1','1',False)])
def test_actual_server_topup_parser_requires_explicit_parent(monkeypatch, value, parent, valid):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'runtime/server.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='EngineManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
        and any(isinstance(c, ast.Name) and c.id=='qwen35_kv_topup_request' for c in ast.walk(n)))
    i = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id=='qwen35_kv_topup_request' for t in n.targets))
    namespace = dict(os=os, RequestValidationError=ValueError)
    monkeypatch.setenv('VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP', value)
    monkeypatch.setenv('VMODEL_QWEN35_SERIAL_KV_RECLAIM', parent)
    module = ast.fix_missing_locations(ast.Module(body=method.body[i:i+3], type_ignores=[]))
    if valid:
        exec(compile(module, str(root/'runtime/server.py'), 'exec'), namespace)
        assert namespace['qwen35_kv_topup_request'] == value
    else:
        with pytest.raises(ValueError):
            exec(compile(module, str(root/'runtime/server.py'), 'exec'), namespace)
    keys = [n for n in ast.walk(method) if isinstance(n, ast.Tuple)
        and any(isinstance(e, ast.Name) and e.id=='qwen35_batched_mlp_request' for e in n.elts)]
    assert len(keys) == 2 and all(any(isinstance(e, ast.Name)
        and e.id=='qwen35_kv_topup_request' for e in n.elts) for n in keys)
