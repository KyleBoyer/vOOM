"""Actual serving control provenance without constructing a GPU constraint."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import gateway_constraint_witness as witness
from runtime.server import _emit_gateway_generation_start, _hidden_gateway_decision_choice

RID = 'resp_' + 'a'*24


def controls(**changes):
    values = dict(tool_choice='specific:vmodel_search_tools', activation_names=[],
        force_reason='tool-result-pagination', allow_parallel=False,
        constraint=SimpleNamespace(profile='required_tool', stop_on_complete=True,
            completed=False, matcher=object(), tensor=object()))
    return {**values, **changes}


def record(**changes):
    return witness.start_record(RID, 'gateway_decision', **controls(**changes))


def test_snapshot_is_bounded_detached_and_never_leaks_names_reason_or_objects():
    values = controls(tool_choice='specific:PRIVATE_TOOL', activation_names=['PRIVATE_ACTIVE'],
        force_reason='PRIVATE_REASON')
    constraint = values['constraint']; state = vars(constraint).copy()
    result = witness.start_record(RID, 'gateway_decision', **values)
    assert result['available'] and result['tool_choice_kind'] == 'specific'
    assert result['constraint_profile'] == 'required_tool'
    assert vars(constraint) == state
    encoded = json.dumps(result)
    assert len(encoded) < 4096 and 'PRIVATE' not in encoded and 'matcher' not in encoded
    values['activation_names'].append('LATER')
    constraint.completed = True
    assert result['prior_activation_count'] == 1 and result['constraint_completed'] is False


def test_observed_match_does_not_claim_a_full_contract_or_token_equivalence():
    left = record(); right = copy.deepcopy(left); right['request_id'] = 'resp_'+'b'*24
    result = witness.compare_observed_controls(left, right)
    assert result['available'] and result['observed_controls_match']
    assert result['differing_fields'] == [] and not result['full_contract_equivalence_proven']
    # Catalog/schema, sampler, prompt/state, weights and kernel identity are not
    # captured here. A matching partial observation cannot stand in for them.
    assert 'not_full_generation_contract' in result['scope']


def test_cold_vs_activated_replay_exposes_different_actual_choice_controls():
    reason = 'tool-result-pagination'
    cold = record(tool_choice=_hidden_gateway_decision_choice('auto', reason, False))
    warm = record(tool_choice=_hidden_gateway_decision_choice('auto', reason, True),
        activation_names=['private_prior_tool'])
    compared = witness.compare_observed_controls(cold, warm)
    assert compared['available'] and compared['observed_controls_match'] is False
    assert set(compared['differing_fields']) == {
        'tool_choice_sha256', 'prior_activation_count', 'prior_activation_names_sha256'}


@pytest.mark.parametrize('changes', [dict(tool_choice='none', constraint=None),
    dict(tool_choice='auto', constraint=None),
    dict(tool_choice='required'),
    dict(tool_choice='auto', constraint=SimpleNamespace(profile='auto_tool_schema',
        stop_on_complete=False, completed=False)),
    dict(tool_choice='none', constraint=SimpleNamespace(profile='json_schema',
        stop_on_complete=True, completed=False))])
def test_known_constraint_modes_and_no_constraint_have_truthful_scalar_records(changes):
    result = record(**changes)
    assert result['available']
    assert witness.compare_observed_controls(result, result)['observed_controls_match']
    if changes.get('constraint', 'missing') is None:
        assert result['constraint_profile'] == 'none'
        assert result['constraint_completed'] is None


@pytest.mark.parametrize('changes', [dict(tool_choice=True), dict(tool_choice='specific:'),
    dict(tool_choice='PRIVATE_INVALID'), dict(tool_choice='specific:'+'x'*513),
    dict(activation_names='private'), dict(activation_names=[None]),
    dict(activation_names=['x']*65), dict(activation_names=['x'*513]),
    dict(force_reason=[]), dict(allow_parallel=1),
    dict(constraint=object()), dict(constraint=SimpleNamespace(profile='private')),
    dict(constraint=SimpleNamespace(profile='required_tool', stop_on_complete=1, completed=False))])
def test_invalid_controls_fail_closed_without_content(changes):
    result = record(**changes)
    assert result['available'] is False and 'PRIVATE' not in json.dumps(result)
    assert witness.compare_observed_controls(result, record())['observed_controls_match'] is None


@pytest.mark.parametrize('rid,phase', [(None,'gateway_decision'), ('PRIVATE','gateway_decision'),
    (RID,'private_phase'), (RID,False)])
def test_invalid_request_or_phase_is_not_a_valid_observation(rid, phase):
    assert not witness.start_record(rid, phase, **controls())['available']


@pytest.mark.parametrize('key,value', [('available',1), ('scope','different'),
    ('tool_choice_sha256','bad'), ('prior_activation_count',True),
    ('constraint_completed',0), ('constraint_profile','private'),
    ('allow_parallel_requested',1), ('force_reason_sha256',False),
    ('prior_activation_names_sha256','f'*64), ('phase',['gateway_decision'])])
def test_comparator_rejects_missing_malformed_and_inconsistent_observations(key, value):
    observed = record(); observed[key] = value
    compared = witness.compare_observed_controls(record(), observed)
    assert not compared['available'] and compared['observed_controls_match'] is None
    assert not compared['full_contract_equivalence_proven']


def test_comparator_rejects_missing_and_extra_fields_and_inconsistent_choice_digest():
    for value in (None, {}, {**record(), 'private': 'secret'},
                  {**record(), 'tool_choice_kind': 'none'}):
        assert not witness.compare_observed_controls(record(), value)['available']


def test_activation_order_and_names_are_compared_not_just_count():
    a = record(activation_names=['alpha', 'beta'])
    b = record(activation_names=['beta', 'alpha'])
    c = record(activation_names=['alpha', 'gamma'])
    for other in (b,c):
        assert witness.compare_observed_controls(a, other)['differing_fields'] == [
            'prior_activation_names_sha256']


def test_disabled_default_does_not_inspect_constraint(monkeypatch, capsys):
    monkeypatch.delenv('VMODEL_GENERATION_WITNESS', raising=False)
    monkeypatch.setattr(witness, 'start_record', lambda *a, **kw: pytest.fail('must stay disabled'))
    _emit_gateway_generation_start(RID, 'gateway_decision', **controls())
    assert capsys.readouterr().out == ''


def test_broken_sink_and_broken_observer_cannot_change_serving(monkeypatch):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    def broken(*a, **kw):
        raise OSError('PRIVATE')
    monkeypatch.setattr('builtins.print', broken)
    assert _emit_gateway_generation_start(RID, 'gateway_decision', **controls()) is None
    monkeypatch.setattr(witness, 'emit_start', broken)
    assert _emit_gateway_generation_start(RID, 'gateway_decision', **controls()) is None


def test_oversized_sink_record_is_unavailable_not_a_partial_match(monkeypatch, capsys):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    monkeypatch.setattr(witness, 'start_record', lambda *a, **kw: dict(private='PRIVATE'*1000))
    _emit_gateway_generation_start(RID, 'gateway_decision', **controls())
    line = capsys.readouterr().out.strip()
    observed = json.loads(line[len(witness.PREFIX):])
    assert len(line) < 4096 and 'PRIVATE' not in line
    assert observed['reason'] == 'record-limit'
    assert witness.compare_observed_controls(record(), observed)['observed_controls_match'] is None


def test_actual_server_callsite_controls_are_observed_before_failed_generation(monkeypatch, capsys):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    tree = ast.parse((Path(__file__).resolve().parents[1]/'runtime/server.py').read_text())
    seen = []
    for node in ast.walk(tree):
        for _, items in ast.iter_fields(node):
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                        and isinstance(item.value.func, ast.Name)
                        and item.value.func.id == '_emit_gateway_generation_start'):
                    continue
                following = items[index+1]
                assert isinstance(following, ast.Assign)
                assert ast.unparse(following.value.func) == '_engine_generate'
                phase = item.value.args[1].value; seen.append(phase)
                keywords = {kw.arg: ast.unparse(kw.value) for kw in item.value.keywords}
                generation = {kw.arg: ast.unparse(kw.value) for kw in following.value.keywords}
                assert keywords['constraint'] == generation['constraint']
                assert keywords['activation_names'] == 'gateway_activated_names'
                assert keywords['allow_parallel'] == 'False'
                assert keywords['tool_choice'] == (
                    'gateway_decision_choice' if phase == 'gateway_decision' else "'required'")
                constraint = controls()['constraint']; original = vars(constraint).copy()
                calls = []
                def fail_engine(*a, **kw):
                    calls.append(kw)
                    assert kw['constraint'] is constraint and vars(constraint) == original
                    raise RuntimeError('synthetic generation failure')
                namespace = dict(_emit_gateway_generation_start=_emit_gateway_generation_start,
                    _engine_generate=fail_engine, rid=RID, gateway_decision_choice='specific:vmodel_enable_tools',
                    gateway_activated_names=['private_prior_tool'], gateway_force_reason='tool-result-pagination',
                    gateway_constraint=constraint, self=SimpleNamespace(_constraint=constraint, _sampling=None),
                    engine=object(), prompt='PRIVATE PROMPT', max_output_tokens=1024, stop=[],
                    decision_stream=None, on_progress=None, gateway_execution_expert_top_k=0)
                fragment = ast.fix_missing_locations(ast.Module(body=[item,following],type_ignores=[]))
                with pytest.raises(RuntimeError, match='synthetic generation failure'):
                    exec(compile(fragment,'<actual-serving-boundary>','exec'),namespace)
                assert len(calls) == 1 and vars(constraint) == original
                line = capsys.readouterr().out.strip()
                observed = json.loads(line[len(witness.PREFIX):])
                assert observed['available'] and observed['phase'] == phase
                assert observed['prior_activation_count'] == 1
                assert 'PRIVATE' not in line and 'completion' not in observed
    assert sorted(seen) == ['gateway_decision','gateway_execution']
