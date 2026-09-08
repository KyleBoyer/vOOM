"""One explicitly extra-budget diagnostic; no model, HTTP, or live tool I/O."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.fixtures import huihui_captured_action_gate as gate
from tests.fixtures import plex_agent_profile as plex
from tests.fixtures.runtime_profile_http_gate import Pressure
from tests.test_huihui_saved_continuation_pure import setup as original_setup
from tests.test_huihui_captured_action_gate_pure import row


def setup(tmp_path, count=5):
    config, original, _, reference, template = original_setup(tmp_path)
    current = copy.deepcopy(original)
    prior, turns, responses = [], [], []
    for index in range(1, count + 1):
        response = copy.deepcopy(template)
        response['output'][0].update(call_id=f'saved-{index}',
            arguments=json.dumps(dict(limit=500, offset=500*(index-1))))
        path = tmp_path/f'response-{index}.json'
        path.write_text(json.dumps(response))
        wall = index + 0.5
        prior.append(dict(turn=index, request=plex.request_shape(current),
            response_path=str(path), response_sha256=gate.sha(path), row=dict(wall_seconds=wall)))
        turn = plex._response_turn(response, wall, index, current)
        turn['handled_call_count'] = int(index < count)
        turns.append(turn)
        responses.append(response)
        plex._append_call_and_result(current, plex.response_calls(response)[0],
            plex.SYNTHETIC_PAGES[min(index-1, len(plex.SYNTHETIC_PAGES)-1)])
    reference.update(workflow_http=prior, plex=dict(turns=turns,
        tool_results_source='synthetic_two_page_fixture', final_text='',
        protocol_failures=[dict(turn=count, reason='tool_round_limit')],
        completion=dict(passed=False, non_completed_turns=[], error_turns=[],
            terminal_has_unhandled_calls=True, unhandled_call_turns=[count],
            terminal_text_present=False)))
    path = tmp_path/'budget-result.json'
    path.write_text(json.dumps(reference))
    config.pop('saved_continuation')
    config.update(workflow='saved_budget_extension', saved_budget_extension=dict(
        result=str(path), sha256=gate.sha(path), additional_http_calls=1))
    return config, original, current, reference, responses


def repin(config, reference):
    path = Path(config['saved_budget_extension']['result'])
    path.write_text(json.dumps(reference))
    config['saved_budget_extension']['sha256'] = gate.sha(path)


@pytest.mark.parametrize('count', [1, 3, 5])
def test_reconstructs_every_prior_request_and_only_one_extra_http(tmp_path, count):
    config, original, expected, reference, responses = setup(tmp_path, count)
    before = copy.deepcopy((config, reference, responses, original))
    result, wire, metadata = gate.prepare_request(config)
    assert result == expected and json.loads(wire) == expected
    assert metadata['request_sha256'] == hashlib.sha256(wire).hexdigest()
    assert metadata['request_shape'] == plex.request_shape(expected)
    assert metadata['additional_http_calls'] == 1
    assert metadata['original_workflow_turn'] == count+1
    assert metadata['preceding_http_calls_omitted'] == count
    assert metadata['original_handled_tool_round_budget'] == count-1
    assert metadata['appended_fixture_page_indices'] == [0]+[1]*(count-1)
    assert metadata['saved_response_sha256'] == [r['response_sha256'] for r in reference['workflow_http']]
    assert {k:v for k,v in result.items() if k!='input'} == {k:v for k,v in original.items() if k!='input'}
    assert result['input'][:len(original['input'])] == original['input']
    assert metadata['original_capture_request'] == reference['request']
    assert 'extra-budget' in metadata['scope'] and 'prior workflow remains failed' in metadata['scope']
    assert before == (config, reference, responses, original)
    assert gate.sha(config['case']['capture']) == config['case']['sha256']


@pytest.mark.parametrize('invalid', [None, False, True, 0, 2, 1.0, '1'])
def test_explicit_single_additional_call_is_required(tmp_path, invalid):
    config, *_ = setup(tmp_path)
    config['saved_budget_extension']['additional_http_calls'] = invalid
    with pytest.raises(ValueError, match='exactly one explicit'):
        gate.prepare_request(config)


@pytest.mark.parametrize('target', ['result', 0, 2, 4])
def test_every_artifact_is_hash_pinned(tmp_path, target):
    config, _, _, reference, _ = setup(tmp_path)
    path = (config['saved_budget_extension']['result'] if target=='result'
        else reference['workflow_http'][target]['response_path'])
    Path(path).write_text('{}')
    with pytest.raises(ValueError, match='identity mismatch'):
        gate.prepare_request(config)


@pytest.mark.parametrize('field', ['case', 'model', 'profiles', 'profile_digest', 'metadata_hashes'])
def test_model_profile_and_original_capture_may_not_change(tmp_path, field):
    config, *_ = setup(tmp_path)
    if field == 'case': config[field] = {**config[field], 'unexpected': True}
    else: config[field] = 'different'
    with pytest.raises(ValueError, match='configuration mismatch'):
        gate.prepare_request(config)


@pytest.mark.parametrize('mutation', ['empty', 'too_many', 'wrong_order', 'bool_turn',
    'first_shape', 'middle_shape', 'last_shape', 'wrong_source_workflow',
    'non_budget_failure', 'completed_workflow', 'fixture_mismatch', 'handled_final',
    'nonempty_final', 'wrong_page_queue', 'bool_duration', 'nan_duration',
    'integer_completion_bool', 'bool_handled_count', 'float_shape_count'])
def test_incomplete_mixed_or_changed_provenance_rejected(tmp_path, mutation):
    config, _, _, reference, _ = setup(tmp_path)
    prior, fixture = reference['workflow_http'], reference['plex']
    if mutation == 'empty': prior.clear()
    elif mutation == 'too_many': prior.append(copy.deepcopy(prior[-1]))
    elif mutation == 'wrong_order': prior[1]['turn'] = 3
    elif mutation == 'bool_turn': prior[0]['turn'] = True
    elif mutation.endswith('_shape'):
        index = {'first_shape':0, 'middle_shape':2, 'last_shape':4}[mutation]
        prior[index]['request']['canonical_sha256'] = 'a'*64
    elif mutation == 'wrong_source_workflow': reference['config']['workflow'] = 'saved_continuation'
    elif mutation == 'non_budget_failure': fixture['protocol_failures'][0]['reason'] = 'error'
    elif mutation == 'completed_workflow': fixture['completion']['passed'] = True
    elif mutation == 'fixture_mismatch': fixture['turns'][2]['wall_seconds'] += 1
    elif mutation == 'handled_final': fixture['turns'][-1]['handled_call_count'] = 1
    elif mutation == 'nonempty_final': fixture['final_text'] = 'A previous final answer'
    elif mutation == 'wrong_page_queue': fixture['tool_results_source'] = 'live'
    elif mutation == 'bool_duration': prior[0]['row']['wall_seconds'] = True
    elif mutation == 'nan_duration': prior[0]['row']['wall_seconds'] = float('nan')
    elif mutation == 'integer_completion_bool': fixture['completion']['passed'] = 0
    elif mutation == 'bool_handled_count': fixture['turns'][0]['handled_call_count'] = True
    elif mutation == 'float_shape_count': prior[0]['request']['tool_count'] = 1.0
    repin(config, reference)
    with pytest.raises(ValueError):
        gate.prepare_request(config)


@pytest.mark.parametrize('index', [0, 2, 4])
@pytest.mark.parametrize('mutation', ['capped', 'host_rendered', 'failed', 'no_call_id', 'wrong_tool'])
def test_every_saved_call_must_be_natural_model_authored_and_supported(tmp_path, index, mutation):
    config, _, _, reference, responses = setup(tmp_path)
    response = responses[index]
    if mutation == 'capped': response['vmodel_cache_phases'][0]['termination_reason'] = 'max_tokens'
    elif mutation == 'host_rendered': response['vmodel_tool_selection']['gateway_deterministic_policy_rendered'] = 1
    elif mutation == 'failed': response['status'] = 'failed'
    elif mutation == 'no_call_id': response['output'][0].pop('call_id')
    elif mutation == 'wrong_tool': response['output'][0]['name'] = 'different'
    entry = reference['workflow_http'][index]
    path = Path(entry['response_path']); path.write_text(json.dumps(response))
    entry['response_sha256'] = gate.sha(path)
    repin(config, reference)
    with pytest.raises(ValueError):
        gate.prepare_request(config)


@pytest.mark.parametrize('kind', ['terminal', 'empty', 'pending_call', 'malformed_call_with_text'])
def test_one_transport_requires_actual_terminal_answer_never_assigns_plex_grade(tmp_path, monkeypatch, kind):
    config, _, _, _, responses = setup(tmp_path)
    config.update(port=1234, response=str(tmp_path/'diagnostic.json'))
    _, wire, _ = gate.prepare_request(config)
    response = copy.deepcopy(responses[-1])
    if kind != 'pending_call':
        response['output'] = [dict(type='message', content=[dict(type='output_text',
            text='' if kind == 'empty' else 'A terminal answer')])]
    if kind == 'malformed_call_with_text':
        response['output'].append(dict(type='function_call', name='bad', arguments='{bad'))
    sent = []
    def post(url, payload, **kwargs):
        sent.append(payload); assert payload == wire
        kwargs['response_observer'](response)
        result = row(); result['response_status'] = 'completed'
        return result
    monkeypatch.setattr(gate, '_post', post)
    monkeypatch.setattr(gate, '_pressure', lambda: Pressure(6_000_000_000, 0, 0))
    document = dict(failures=[], final_plex_score=None, generated_tools_executed=False)
    gate.run_single_request(config, wire, document, initial_action=False)
    assert len(sent) == 1 and document['final_plex_score'] is None
    assert not document['generated_tools_executed'] and 'plex_rubric_score' not in document
    assert document['checks']['terminal_answer_without_pending_calls'] is (kind == 'terminal')
    assert ('terminal_answer_without_pending_calls' in document['failures']) is (kind != 'terminal')
    assert json.loads(Path(config['response']).read_text()) == response
    assert document['response_sha256'] == gate.sha(config['response'])
