"""Pinned continuation reconstruction and one-call transport, without MLX."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.fixtures import huihui_captured_action_gate as gate
from tests.fixtures import plex_agent_profile as plex
from tests.fixtures.runtime_profile_http_gate import Pressure
from tests.test_huihui_captured_action_gate_pure import row, selection


def setup(tmp_path):
    capture = tmp_path / 'capture.json'
    capture.write_text(json.dumps(dict(model='original', stream=True,
        input=[dict(role='user', content='Original criteria')],
        tools=[dict(type='function', name=plex.PLEX_MEDIA_TOOL,
                    parameters=dict(type='object', properties={}))])))
    config = dict(workflow='plex', model='audit-model', profiles=['audit'],
        profile_digest='digest', metadata_hashes={}, case=dict(capture=str(capture),
            bytes=capture.stat().st_size, sha256=gate.sha(capture), stream=True))
    request, wire, metadata = gate.prepare_case(config['case'], config['model'])
    phase = dict(input_tokens=3, output_tokens=2, generation_max_tokens=1024,
        termination_reason='grammar', generation_witness=dict(available=True,
            prepared_prompt_token_count=3, generated_token_count=2,
            prepared_prompt_token_ids_sha256='a'*64,
            generated_token_ids_sha256='b'*64, engine_text_sha256='c'*64))
    response = dict(status='completed', vmodel_tool_selection=selection(),
        vmodel_cache_phases=[phase], output=[dict(type='function_call',
            name=plex.PLEX_MEDIA_TOOL, arguments='{"limit":500,"offset":0}',
            call_id='saved-call-unchanged')])
    response_path = tmp_path / 'first-response.json'
    response_path.write_text(json.dumps(response))
    continuation = copy.deepcopy(request)
    plex._append_call_and_result(continuation, plex.response_calls(response)[0], plex.SYNTHETIC_PAGES[0])
    reference = dict(schema='voom.huihui-captured-plex.v1', config=copy.deepcopy(config),
        request=metadata, workflow_http=[dict(request=plex.request_shape(request),
            response_path=str(response_path), response_sha256=gate.sha(response_path)),
            dict(request=plex.request_shape(continuation))])
    reference_path = tmp_path / 'reference.json'
    reference_path.write_text(json.dumps(reference))
    config.update(workflow='saved_continuation', saved_continuation=dict(
        result=str(reference_path), sha256=gate.sha(reference_path)))
    return config, request, continuation, reference, response


def repin(config, reference):
    path = Path(config['saved_continuation']['result'])
    path.write_text(json.dumps(reference))
    config['saved_continuation']['sha256'] = gate.sha(path)


def test_exact_continuation_preserves_all_original_fields_and_identity(tmp_path):
    config, original, expected, reference, response = setup(tmp_path)
    before = copy.deepcopy((config, original, reference, response))
    request, wire, metadata = gate.prepare_request(config)
    assert request == expected and json.loads(wire) == expected
    assert request['tools'] == original['tools']
    assert request['input'][:len(original['input'])] == original['input']
    assert request['input'][-2]['arguments'] == response['output'][0]['arguments']
    assert json.loads(request['input'][-1]['output']) == plex.SYNTHETIC_PAGES[0]
    assert {k:v for k,v in request.items() if k!='input'} == {k:v for k,v in original.items() if k!='input'}
    assert metadata['request_shape'] == reference['workflow_http'][1]['request']
    assert metadata['request_sha256'] == hashlib.sha256(wire).hexdigest()
    assert metadata['original_workflow_turn'] == 2
    assert metadata['preceding_http_calls_omitted'] == 1
    assert metadata['tool_results_source'] == 'unchanged_synthetic_first_page'
    assert metadata['original_capture_request'] == reference['request']
    assert before == (config, original, reference, response)
    assert gate.sha(config['case']['capture']) == config['case']['sha256']


@pytest.mark.parametrize('workflow', ['initial_action', 'plex'])
def test_existing_modes_keep_original_request_path(tmp_path, workflow):
    config, *_ = setup(tmp_path)
    config['workflow'] = workflow
    assert gate.prepare_request(config) == gate.prepare_case(config['case'], config['model'])


@pytest.mark.parametrize('target', ['result', 'response'])
def test_pinned_artifacts_cannot_be_changed(tmp_path, target):
    config, _, _, reference, _ = setup(tmp_path)
    path = (config['saved_continuation']['result'] if target == 'result'
            else reference['workflow_http'][0]['response_path'])
    Path(path).write_text('{}')
    with pytest.raises(ValueError, match='identity mismatch'):
        gate.prepare_request(config)


@pytest.mark.parametrize('field', ['model', 'profiles', 'profile_digest', 'metadata_hashes', 'case'])
def test_no_silent_model_profile_or_capture_substitution(tmp_path, field):
    config, *_ = setup(tmp_path)
    if field == 'case':
        config[field] = {**config[field], 'additional_hint': 'different'}
    else:
        config[field] = 'different'
    with pytest.raises(ValueError, match='configuration mismatch'):
        gate.prepare_request(config)


@pytest.mark.parametrize('mutation', ['first_shape', 'second_shape', 'missing_second',
    'status', 'host_rendered', 'capped', 'wrong_tool', 'no_call_id'])
def test_incomplete_or_nonidentical_provenance_is_rejected(tmp_path, mutation):
    config, _, _, reference, response = setup(tmp_path)
    if mutation == 'first_shape':
        reference['workflow_http'][0]['request']['input_items'] += 1
    elif mutation == 'second_shape':
        reference['workflow_http'][1]['request']['canonical_sha256'] = 'd'*64
    elif mutation == 'missing_second':
        reference['workflow_http'].pop()
    else:
        if mutation == 'status': response['status'] = 'failed'
        if mutation == 'host_rendered': response['vmodel_tool_selection']['gateway_deterministic_policy_rendered'] = 1
        if mutation == 'capped': response['vmodel_cache_phases'][0]['termination_reason'] = 'max_tokens'
        if mutation == 'wrong_tool': response['output'][0]['name'] = 'different'
        if mutation == 'no_call_id': response['output'][0].pop('call_id')
        path = Path(reference['workflow_http'][0]['response_path'])
        path.write_text(json.dumps(response))
        reference['workflow_http'][0]['response_sha256'] = gate.sha(path)
    repin(config, reference)
    with pytest.raises(ValueError):
        gate.prepare_request(config)


@pytest.mark.parametrize('completed', [True, False])
def test_single_transport_saves_response_before_checks_without_a_plex_score(tmp_path, monkeypatch, completed):
    config, _, _, _, response = setup(tmp_path)
    config.update(port=1234, response=str(tmp_path/'replayed.json'))
    _, wire, _ = gate.prepare_request(config)
    response['status'] = 'completed' if completed else 'failed'
    calls = []
    def post(url, payload, **kwargs):
        assert payload == wire and kwargs['stream'] is True
        assert kwargs['fail_on_memory_retry'] is True
        calls.append(payload)
        kwargs['response_observer'](response)
        assert Path(config['response']).exists()
        result = row()
        result['response_status'] = response['status']
        return result
    monkeypatch.setattr(gate, '_post', post)
    monkeypatch.setattr(gate, '_pressure', lambda: Pressure(6_000_000_000, 0, 0))
    document = dict(failures=[], generated_tools_executed=False, final_plex_score=None)
    gate.run_single_request(config, wire, document, initial_action=False)
    assert len(calls) == 1
    assert 'listing_calls_only' not in document['checks']
    assert document['checks']['completed'] is completed
    assert ('completed' in document['failures']) is not completed
    assert document['response_sha256'] == gate.sha(config['response'])
    assert json.loads(Path(config['response']).read_text()) == response
    assert Path(config['response']).stat().st_mode & 0o777 == 0o600
    assert document['final_plex_score'] is None and not document['generated_tools_executed']
