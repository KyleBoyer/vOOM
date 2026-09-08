"""No MLX: exact wire preservation, terminal semantics and observer boundaries."""

import copy
import hashlib
import io
import json

import pytest

from tests.fixtures import captured_transition_tracking_gate as gate
from tests.fixtures import qwen3_large_agent_replay_gate as replay


def capture(tmp_path, value):
    raw = json.dumps(value).encode()
    path = tmp_path/'capture.json'
    path.write_bytes(raw)
    return dict(capture=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('stream', [True, False, None])
def test_only_declared_fields_change_input_tools_roles_reasoning_preserved(tmp_path, stream):
    source = dict(model='old', input=[dict(role='developer', content='same'),
        dict(role='user', content=[dict(type='input_text', text='same')])],
        tools=[dict(type='function', name='different_tool', parameters={'type': 'object'})],
        reasoning={'effort': 'low'}, max_output_tokens=60, temperature=0.3)
    if stream is not None:
        source['stream'] = stream
    case = {**capture(tmp_path, source), 'stream': bool(stream)}
    request, payload, metadata = gate.prepare_case(case, 'GLM-5.3-Flash')
    expected = {**source, 'model': 'GLM-5.3-Flash', 'max_output_tokens': 1024,
                'temperature': 0.0, 'seed': 64013}
    assert request == expected == json.loads(payload)
    assert metadata['request_changed_fields'] == ['max_output_tokens', 'model', 'seed', 'temperature']


def test_bad_capture_hash_rejected(tmp_path):
    case = capture(tmp_path, dict(input=[]))
    case['sha256'] = '0'*64
    with pytest.raises(ValueError, match='identity'):
        gate.prepare_case(case, 'model')


@pytest.mark.parametrize('extra', [{'messages': []}, {'max_tokens': 60}])
def test_no_silent_chat_conversion(tmp_path, extra):
    with pytest.raises(ValueError, match='native Responses'):
        gate.prepare_case(capture(tmp_path, dict(input=[], **extra)), 'model')


def test_case_cannot_disable_stream_equality_for_a_streaming_capture(tmp_path):
    with pytest.raises(ValueError, match='captured transport'):
        gate.prepare_case(capture(tmp_path, dict(input=[], stream=True)), 'model')


def message(text):
    return dict(type='message', content=[dict(type='output_text', text=text)])


def weather(city='Tokyo', name='get_weather'):
    return dict(type='function_call', name=name, arguments=json.dumps({'city': city}))


@pytest.mark.parametrize('output,ok', [([weather()], True),
    ([weather('Paris')], False), ([weather(name='wrong')], False),
    ([weather(), weather()], False), ([], False),
    ([dict(type='function_call', name='get_weather', arguments='{bad')], False)])
def test_weather_requires_exact_named_single_call_and_arguments(output, ok):
    checks = gate.semantic_checks(dict(output=output), dict(kind='weather_tool', city='Tokyo'))
    assert all(checks.values()) is ok


@pytest.mark.parametrize('text,ok', [('NodeJS Joke', True), ('', False),
    ('NodeJS makes this title too long', False), ('"NodeJS Joke"', False),
    ('NodeJS\nJoke', False), ('Unrelated topic', False)])
def test_title_constraints(text, ok):
    checks = gate.semantic_checks(dict(output=[message(text)]), dict(kind='short_title', topic='node'))
    assert all(checks.values()) is ok


def test_reasoning_is_not_used_as_visible_answer():
    response = dict(output=[dict(type='reasoning', content=[dict(type='reasoning_text', text='NodeJS Joke')])])
    assert gate.visible_text(response) == ''
    assert not all(gate.semantic_checks(response, dict(kind='short_title', topic='node')).values())


def valid_row():
    return dict(http_status=200, response_status='completed', error=None, backend='voom',
        usage={'output_tokens': 4}, runtime_profiles=['test'], runtime_profile_digest='digest',
        runtime_profile_overrides=[], timing={'generation_witness': {'available': True,
            'generated_token_count': 4, 'prepared_prompt_token_ids_sha256': 'abc'},
            'true_peak_metal_bytes': 100, 'memory_prefill_retries': 0},
        pressure_before=dict(available_bytes=7_000_000_000, swap_used_bytes=0, swap_out_bytes=0),
        pressure_after=dict(available_bytes=6_000_000_000, swap_used_bytes=0, swap_out_bytes=0))


@pytest.mark.parametrize('key,value', [('response_status', 'incomplete'), ('http_status', 500),
    ('backend', 'other'), ('runtime_profiles', []), ('runtime_profile_digest', 'wrong')])
def test_good_content_cannot_hide_completion_or_identity_failure(key, value):
    row = valid_row(); row[key] = value
    checks = gate.row_checks(row, dict(output=[message('NodeJS Joke')]),
        dict(kind='short_title', topic='node'), dict(profiles=['test'], profile_digest='digest'))
    assert not all(checks.values())


@pytest.mark.parametrize('tokens', [0, True, 1024, 1025])
def test_cap_or_empty_response_is_not_a_completed_answer_pass(tokens):
    row = valid_row(); row['usage']['output_tokens'] = tokens
    checks = gate.row_checks(row, dict(output=[message('NodeJS Joke')]),
        dict(kind='short_title', topic='node'), dict(profiles=['test'], profile_digest='digest'))
    assert not checks['sufficient_actual_output']


def test_actual_swap_out_failure_even_when_net_usage_does_not_grow():
    row = valid_row(); row['pressure_after']['swap_out_bytes'] = 16_000_001
    checks = gate.row_checks(row, dict(output=[message('NodeJS Joke')]),
        dict(kind='short_title', topic='node'), dict(profiles=['test'], profile_digest='digest'))
    assert checks['swap_used'] and not checks['actual_swap_out']


def factor_timing():
    return dict(qwen_mtp_compact_kda_rollback_enabled=1,
        qwen_mtp_kda_factor_rounds=2, qwen_mtp_kda_factor_bytes_peak=123,
        qwen_mtp_kda_factor_base_bytes_peak=456, qwen_mtp_kda_factor_restores=1,
        qwen_mtp_kda_factor_restore_s=0.1)


def completed_phase(count=2, reason='eos'):
    return dict(input_tokens=7, output_tokens=count, generation_max_tokens=1024,
        termination_reason=reason, generation_witness=dict(available=True,
            generated_token_count=count, prepared_prompt_token_count=7,
            generated_token_ids_sha256='a'*64, prepared_prompt_token_ids_sha256='b'*64,
            engine_text_sha256='c'*64))


@pytest.mark.parametrize('bad', [dict(output_tokens=1024, termination_reason='length'),
    dict(termination_reason=None), dict(termination_reason='unknown'),
    dict(output_tokens=True), dict(generation_max_tokens=True),
    dict(generation_max_tokens=0), dict(generation_max_tokens=None), dict(output_tokens=1025)])
def test_hidden_truncation_cannot_hide_behind_a_completed_public_call(bad):
    hidden = {**completed_phase(), **bad}
    response = dict(vmodel_cache_phases=[hidden, completed_phase()])
    assert not gate.generation_phase_checks(response)['all_phase_natural_termination']


@pytest.mark.parametrize('bad', [None, {}, dict(available=False),
    dict(available=True, generated_token_count=999),
    {**completed_phase()['generation_witness'], 'engine_text_sha256': 'invalid'},
    {**completed_phase()['generation_witness'], 'generated_token_count': True}])
def test_every_phase_needs_its_own_valid_raw_identity(bad):
    hidden = completed_phase(); hidden['generation_witness'] = bad
    checks = gate.generation_phase_checks(dict(vmodel_cache_phases=[hidden, completed_phase()]))
    assert checks['all_phase_natural_termination']
    assert not checks['all_phase_generation_witness']


@pytest.mark.parametrize('phases', [None, [], [None], [{}]])
def test_missing_phase_provenance_is_not_completion(phases):
    assert not any(gate.generation_phase_checks(dict(vmodel_cache_phases=phases)).values())


def test_all_phase_gate_accepts_actual_natural_stop_including_eos_at_budget():
    response = dict(vmodel_cache_phases=[completed_phase(1024),
        completed_phase(19, 'grammar'), completed_phase(2, 'stop_sequence')])
    assert all(gate.generation_phase_checks(response).values())


@pytest.mark.parametrize('bad', [{}, {'qwen_mtp_compact_kda_rollback_enabled': 0},
    {'qwen_mtp_compact_kda_rollback_enabled': True}, {'qwen_mtp_kda_factor_rounds': 0},
    {'qwen_mtp_kda_factor_bytes_peak': 0}, {'qwen_mtp_kda_factor_base_bytes_peak': None},
    {'qwen_mtp_kda_factor_restores': True}, {'qwen_mtp_kda_factor_restore_s': float('nan')}])
def test_optional_factor_gate_requires_actual_typed_path_witness(bad):
    row = valid_row()
    row['timing'].update(factor_timing())
    config = dict(profiles=['test'], profile_digest='digest', require_qwen_factors=True)
    case = dict(kind='short_title', topic='node')
    response = dict(output=[message('NodeJS Joke')])
    assert all(gate.row_checks(row, response, case, config).values())
    if bad:
        row['timing'].update(bad)
    else:
        row['timing'].pop('qwen_mtp_kda_factor_rounds')
    assert not gate.row_checks(row, response, case, config)['qwen_scalar_factor_path']


@pytest.mark.parametrize('phases', [None, [], [{}], [dict(prompt_state_approximate=True)],
    [dict(prompt_state_approximate=0, qwen_lossy_suffix_prefill_early_layers=0,
          qwen_lossy_suffix_prefill_used=1)]])
def test_optional_full_state_gate_rejects_missing_or_approximate_phase(phases):
    row = valid_row()
    config = dict(profiles=['test'], profile_digest='digest', require_full_prompt_state=True)
    case = dict(kind='short_title', topic='node')
    response = dict(output=[message('NodeJS Joke')], vmodel_cache_phases=phases)
    assert not gate.row_checks(row, response, case, config)['all_phases_full_prompt_state']
    response['vmodel_cache_phases'] = [dict(prompt_state_approximate=0,
        qwen_lossy_suffix_prefill_early_layers=0, qwen_lossy_suffix_prefill_used=0)]
    assert all(gate.row_checks(row, response, case, config).values())


@pytest.mark.parametrize('stream', [False, True])
def test_client_observer_gets_actual_terminal_without_altering_wire_or_summary(monkeypatch, stream):
    terminal = dict(status='completed', output=[message('NodeJS Joke')])
    wire = b'{"same":"wire"}'
    body = (b'data: '+json.dumps(dict(type='response.completed', response=terminal)).encode()+b'\n\n'
            if stream else json.dumps(terminal).encode())
    def open_response(request, timeout):
        assert request.data == wire
        return io.BytesIO(body)
    monkeypatch.setattr(replay.urllib.request, 'urlopen', open_response)
    observed, summarized = [], []
    def summary(response, **kwargs):
        summarized.append(copy.deepcopy(response))
        return {'unchanged_summary': True}
    monkeypatch.setattr(replay, '_summary', summary)
    row = replay._post('http://127.0.0.1/test', wire, 1, stream,
                       response_observer=lambda r: observed.append(copy.deepcopy(r)))
    assert observed == summarized == [terminal]
    assert row == {'unchanged_summary': True}


def test_incomplete_sse_does_not_invent_a_terminal_observation(monkeypatch):
    monkeypatch.setattr(replay.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b': waiting\n'))
    observed = []
    row = replay._post('http://127.0.0.1/test', b'{}', 1, True, response_observer=observed.append)
    assert row['http_status'] == 599 and not observed


def native(available=6_000_000_000, swapout=0, observed=True):
    return '[process-memory] '+json.dumps(dict(system_available_bytes=available,
        system_swap_used_bytes=0, system_swap_out_bytes=swapout,
        process=dict(available=observed, physical_footprint_bytes=100,
                     internal_compressed_ledger_bytes=20)))


@pytest.mark.parametrize('log,passed', [('', False), (native(), True),
    (native(observed=False), False), (native(available=5_299_999_999), False),
    (native()+'\n'+native(swapout=16_000_001), False)])
def test_periodic_pressure_cannot_be_replaced_by_good_terminal_sample(log, passed):
    assert gate.native_pressure_summary(log)['passed'] is passed
