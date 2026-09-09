"""Completed-phase diagnostics survive later failure; no MLX/model execution."""

import ast
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime import gateway_phase_witness as witness
from runtime.server import Handler, _emit_gateway_phase_completion

RID = 'resp_' + 'a'*24


def phase():
    return dict(phase='gateway_decision', input_tokens=6, output_tokens=2,
        generation_max_tokens=1024, termination_reason='grammar', cached_tokens=0,
        suffix_prefill_seconds=1.5, decode_seconds=0.5, total_engine_seconds=2.0,
        true_peak_metal_bytes=1234, generation_witness=dict(
            schema='voom.generation-witness.v1', available=True,
            scope='single_engine_generation_before_protocol_parsing',
            token_hash_encoding='compact-json-integer-array-v1', generated_token_count=2,
            prepared_prompt_token_count=6, engine_text_bytes=71,
            generated_token_ids_sha256='b'*64,
            prepared_prompt_token_ids_sha256='c'*64, engine_text_sha256='d'*64))


def test_bounded_copy_omits_unknown_content_and_detaches_original():
    value = phase()
    value.update(text='PRIVATE ANSWER', prompt='PRIVATE PROMPT', tokens=[12345])
    value['generation_witness']['unexpected'] = 'PRIVATE NESTED CONTENT'
    before = copy.deepcopy(value)
    result = witness.completion_record(RID, value)
    assert result['available'] is True and result['scope'] == witness.SCOPE
    assert result['generation_witness']['generated_token_ids_sha256'] == 'b'*64
    assert 'PRIVATE' not in json.dumps(result) and '12345' not in json.dumps(result)
    assert len(json.dumps(result)) < 8192
    assert value == before
    result['generation_witness']['generated_token_count'] = 999
    assert value == before


@pytest.mark.parametrize('reason', ['eos', 'stop', 'grammar', 'max_tokens'])
def test_returned_generation_is_not_a_request_completion_claim(reason):
    value = phase(); value['termination_reason'] = reason
    result = witness.completion_record(RID, value)
    assert result['available'] is True and result['termination_reason'] == reason
    assert 'passed' not in result and 'final_plex_score' not in result
    assert 'not_request_completion' in result['scope']


@pytest.mark.parametrize('key,value', [('input_tokens', True), ('output_tokens', -1),
    ('generation_max_tokens', None), ('suffix_prefill_seconds', float('nan')),
    ('decode_seconds', float('inf')), ('total_engine_seconds', 10**1000),
    ('true_peak_metal_bytes', '1234'), ('termination_reason', 'private error text')])
def test_invalid_metrics_are_unavailable_not_invented_zero(key, value):
    observed = phase(); observed[key] = value
    result = witness.completion_record(RID, observed)
    assert result['available'] is False and 'generation_witness' not in result
    assert 'private error text' not in json.dumps(result)


@pytest.mark.parametrize('key,value', [('prepared_prompt_token_count', 7),
    ('generated_token_count', True), ('engine_text_sha256', 'PRIVATE'),
    ('scope', 'unknown'), ('available', False)])
def test_missing_or_mismatched_raw_witness_fails_closed(key, value):
    observed = phase(); observed['generation_witness'][key] = value
    result = witness.completion_record(RID, observed)
    assert result['available'] is False and 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('request_id', [None, 123, 'PRIVATE', 'resp_' + 'f'*25])
def test_only_server_minted_id_shape_is_logged(request_id):
    result = witness.completion_record(request_id, phase())
    assert result['available'] is False and 'request_id' not in result


def test_weight_bytes_remain_explicitly_logical_not_physical():
    value = phase(); value['weight_store_bytes_read'] = 1234
    assert not witness.completion_record(RID, value)['available']
    value.update(weight_store_bytes_read_source='path_stats',
        weight_store_bytes_read_scope='single_engine_phase_logical_not_physical')
    result = witness.completion_record(RID, value)
    assert result['available'] and result['weight_store_bytes_read'] == 1234
    assert result['weight_store_bytes_read_scope'].endswith('not_physical')


def test_disabled_default_is_silent_and_enabled_emission_does_not_mutate(monkeypatch, capsys):
    monkeypatch.delenv('VMODEL_GENERATION_WITNESS', raising=False)
    value = phase(); before = copy.deepcopy(value)
    _emit_gateway_phase_completion(RID, value)
    assert capsys.readouterr().out == ''
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    _emit_gateway_phase_completion(RID, value)
    line = capsys.readouterr().out.strip()
    assert line.startswith(witness.PREFIX)
    assert json.loads(line[len(witness.PREFIX):]) == witness.completion_record(RID, value)
    assert value == before


def test_broken_observer_sink_cannot_fail_generation(monkeypatch):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    def broken(*args, **kwargs):
        raise OSError('sink unavailable')
    monkeypatch.setattr('builtins.print', broken)
    assert _emit_gateway_phase_completion(RID, phase()) is None


def test_oversized_observation_is_replaced_without_echoing_content(monkeypatch, capsys):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    monkeypatch.setattr(witness, 'completion_record', lambda *args: {'private': 'SECRET'*2000})
    _emit_gateway_phase_completion(RID, phase())
    line = capsys.readouterr().out.strip()
    record = json.loads(line[len(witness.PREFIX):])
    assert not record['available'] and record['reason'] == 'record-limit'
    assert len(line) < 8192 and 'SECRET' not in line


def test_completed_hidden_record_survives_failed_stream_without_fake_public_phase(monkeypatch, capsys):
    monkeypatch.setenv('VMODEL_GENERATION_WITNESS', '1')
    handler = Handler.__new__(Handler)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda *_: None
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None
    engine = SimpleNamespace(cfg=SimpleNamespace(model_type='qwen3_5'))
    def fail(on_token, on_progress):
        _emit_gateway_phase_completion(RID, phase())
        raise RuntimeError('later public generation failed')
    handler._stream_responses('unused', 1024, [], engine, [], lambda *args: {},
        RID, 'fixture', 1, None, 0, None, [], 'msg_test', 'auto', False, generate_fn=fail)
    logs = [json.loads(line[len(witness.PREFIX):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(witness.PREFIX)]
    assert len(logs) == 1 and logs[0]['phase'] == 'gateway_decision' and logs[0]['available']
    events = [json.loads(line[6:]) for line in handler.wfile.getvalue().decode().splitlines()
              if line.startswith('data: ')]
    assert events[-1]['type'] == 'response.failed'
    assert events[-1]['response']['status'] == 'failed'
    assert events[-1]['response']['output'] == []
    assert not any(event['type'] == 'response.completed' for event in events)


def test_real_gateway_hooks_follow_phase_capture_before_protocol_parsing():
    tree = ast.parse((Path(__file__).resolve().parents[1]/'runtime/server.py').read_text())
    seen = []
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            for i, child in enumerate(value):
                if (isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                        and isinstance(child.value.func, ast.Name)
                        and child.value.func.id == '_emit_gateway_phase_completion'):
                    previous, following = value[i-1], value[i+1]
                    assert isinstance(previous, ast.Assign)
                    assert ast.unparse(previous.value.func) == '_cache_phase_telemetry'
                    assert ast.unparse(following.value.func) == '_parse_request_tool_calls'
                    assert ast.unparse(child.value.args[0]) == 'rid'
                    assert ast.unparse(child.value.args[1]) == ast.unparse(previous.targets[0])
                    seen.append(previous.value.args[0].value)
    assert sorted(seen) == ['gateway_decision', 'gateway_execution']
