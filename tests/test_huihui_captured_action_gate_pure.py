import json
import copy
import hashlib
from pathlib import Path

import pytest

from runtime.profiles import apply_runtime_profiles
from tests.fixtures import huihui_captured_action_gate as gate
from tests.fixtures import plex_agent_profile as plex
from tests.fixtures.runtime_profile_http_gate import Pressure


def call(arguments, name='plugin__plex__plex_list_library'):
    return dict(type='function_call', name=name, arguments=json.dumps(arguments))


def selection():
    return dict(gateway_deterministic_policy_rendered=0, gateway_pagination_host_routed=0,
        gateway_initial_pagination_defaults_applied=0, gateway_literal_arguments_grounded=0)


@pytest.mark.parametrize('arguments', [{}, {'limit': 50, 'offset': 0},
    {'limit': None, 'offset': None}, {'limit': 20.0, 'offset': 0.0}])
def test_initial_listing_only_is_not_final_plex_grade(arguments):
    assert all(gate.action_checks(dict(output=[call(arguments)])).values())


@pytest.mark.parametrize('output', [[], [call({}, 'wrong')],
    [call({})]*3, [call([])], [call({'limit': 101})],
    [call({'limit': True})], [call({'limit': 0})],
    [call({'limit': float('nan')})], [call({'offset': float('inf')})],
    [call({'offset': 1})], [call({'offset': False})],
    [dict(type='function_call', name='plugin__plex__plex_list_library', arguments='{bad')],
    [dict(type='message', content=[dict(type='output_text', text='Done')])]])
def test_invalid_or_non_listing_first_action_rejected(output):
    assert not all(gate.action_checks(dict(output=output)).values())


def test_evaluation_profile_only_disables_host_actions_render_and_prompt_cache():
    base, audit = {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-fast-agent-mtpquant'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-fast-agent-model-only-audit'], environ=audit)
    # Profile identity fields naturally differ; settings remain otherwise exact.
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(audit) == {**settings(base),
        'VMODEL_FAST_TOOL_GATEWAY_DETERMINISTIC_POLICY': '0',
        'VMODEL_FAST_TOOL_GATEWAY_HOST_ROUTE': '0',
        'VMODEL_QWEN35_HOT_KV': '0', 'VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST': '0'}
    assert audit['VMODEL_FAST_TOOL_GATEWAY'] == '1'
    assert audit['VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL'] == '16:1024'


def row():
    return dict(http_status=200, response_status='completed', error=None,
        backend='voom', runtime_profiles=['audit'], runtime_profile_digest='digest',
        runtime_profile_overrides=[], streamed_text_matches_final=True,
        usage=dict(output_tokens=60, input_tokens_details=dict(cached_tokens=0)),
        timing=dict(generation_witness=dict(available=True, generated_token_count=55,
            prepared_prompt_token_ids_sha256='hash'), true_peak_metal_bytes=100,
            memory_prefill_retries=0),
        pressure_before=dict(available_bytes=7_000_000_000, swap_used_bytes=0, swap_out_bytes=0),
        pressure_after=dict(available_bytes=6_000_000_000, swap_used_bytes=0, swap_out_bytes=0))


def test_full_state_audit_changes_only_the_mixed_depth_switch():
    base, full = {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-fast-agent-model-only-audit'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-model-only-audit'], environ=full)
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(full) == {**settings(base), 'VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL': 'off'}
    from runtime.server import _qwen_lossy_suffix_prefill_policy
    assert _qwen_lossy_suffix_prefill_policy('off', mode='fast', total_layers=64,
        layer_types=['linear_attention'] * 3 + ['full_attention']) == (0, 0, 0)


def full_state_checks(phases):
    return gate.acceptance(row(), dict(status='completed', output=[call({})],
        vmodel_tool_selection=selection(), vmodel_cache_phases=phases),
        dict(profiles=['audit'], profile_digest='digest', require_full_prompt_state=True))


def phase():
    return dict(prompt_state_approximate=0, qwen_lossy_suffix_prefill_early_layers=0,
        qwen_lossy_suffix_prefill_used=0, true_peak_metal_bytes=100,
        weight_store_bytes_read=123, weight_store_bytes_read_source='path_stats',
        weight_store_bytes_read_scope='single_engine_phase_logical_not_physical')


def test_full_state_witness_accepts_both_model_generations():
    assert all(full_state_checks([phase(), phase()]).values())


@pytest.mark.parametrize('phases', [None, [], {}, [None], [phase(), {}],
    [{**phase(), 'prompt_state_approximate': 1}, phase()],
    [{**phase(), 'qwen_lossy_suffix_prefill_early_layers': 16}, phase()],
    [{**phase(), 'qwen_lossy_suffix_prefill_used': 1}, phase()],
    [{**phase(), 'prompt_state_approximate': False}]])
def test_full_state_witness_fails_closed_on_any_hidden_phase(phases):
    assert not full_state_checks(phases)['full_prompt_state']


@pytest.mark.parametrize('bad', [
    {'weight_store_bytes_read_source': 'unavailable'},
    {'weight_store_bytes_read': 0}, {'weight_store_bytes_read': True},
    {'weight_store_bytes_read_scope': 'physical'}])
def test_phase_io_needs_positive_measured_logical_reads(bad):
    assert not full_state_checks([phase(), {**phase(), **bad}])['phase_io_witness']


@pytest.mark.parametrize('peak', [None, 0, True, 8_500_000_001])
def test_hidden_phase_metal_cannot_be_hidden_by_public_peak(peak):
    assert not full_state_checks([{**phase(), 'true_peak_metal_bytes': peak}, phase()])['all_phase_metal']


def check(value, status='completed'):
    return gate.acceptance(value, dict(status=status, output=[call({})], vmodel_tool_selection=selection()),
        dict(profiles=['audit'], profile_digest='digest'))


def test_gateway_witness_is_single_generation_not_aggregate_token_equality():
    assert all(check(row()).values())


@pytest.mark.parametrize('count', [0, True, 1024, 1025])
def test_output_cap_not_a_completed_action(count):
    value = row(); value['usage']['output_tokens'] = count
    assert not check(value)['not_output_capped']


def test_incomplete_actual_response_not_hidden_by_summary():
    assert not check(row(), 'incomplete')['completed']


def test_prompt_cache_reuse_rejects_cold_claim():
    value = row(); value['usage']['input_tokens_details']['cached_tokens'] = 1
    assert not check(value)['no_prompt_reuse']


def test_physical_swap_out_fails_despite_unchanged_swap_usage():
    value = row(); value['pressure_after']['swap_out_bytes'] = 16_000_001
    assert not check(value)['actual_swap_out']


def workflow_setup(tmp_path, monkeypatch, *, incomplete=False, retry_evidence=False,
                   abort_on_memory_retry=True):
    request = dict(model='test', stream=True, temperature=0, seed=64013,
        max_output_tokens=1024, input=[dict(role='user', content='unaltered')],
        tools=[dict(type='function', name=plex.PLEX_TOOL, parameters={})])
    baseline = copy.deepcopy(request)
    args = dict(mediaType='all', ratingOperator='lte', movieRatingValue='PG-13',
        showRatingValue='TV-Y7', excludeRootFolderPath='/Kids/', limit=50, offset=0)
    responses = [dict(status='completed', output=[dict(call(args), call_id='a')]),
        dict(status='completed', output=[dict(call({**args, 'offset': 50}), call_id='b')]),
        dict(status='completed', output=[dict(type='message', content=[
            dict(type='output_text', text=', '.join(plex.ELIGIBLE_TITLES))])])]
    if incomplete:
        responses[0]['status'] = 'incomplete'
    for response in responses:
        response['vmodel_tool_selection'] = selection()
    config = dict(port=1234, response=str(tmp_path/'reply.json'),
        result=str(tmp_path/'result.json'), profiles=['audit'], profile_digest='digest',
        abort_on_memory_retry=abort_on_memory_retry,
        wire_sha256=hashlib.sha256(json.dumps(request, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest())
    wires = []
    def post(url, wire, **kwargs):
        current = json.loads(wire)
        assert current['tools'] == baseline['tools']
        assert current['input'][:1] == baseline['input']
        assert len(current['input']) == 1 + len(wires)*2
        assert kwargs['stream'] and kwargs['fail_on_memory_retry'] is abort_on_memory_retry
        assert current['max_output_tokens'] == 1024
        wires.append(wire)
        response = responses[len(wires)-1]
        kwargs['response_observer'](response)
        value = row()
        value.update(wall_seconds=1.5, response_status=response['status'])
        if retry_evidence and len(wires) == 1:
            value['prefill_progress'] = [{'phase': 'memory_retry', 'completed': 1}]
        return value
    monkeypatch.setattr(gate, '_post', post)
    monkeypatch.setattr(gate, '_pressure', lambda: Pressure(6_000_000_000, 0, 0))
    return request, config, responses, wires


def test_workflow_preserves_requests_scores_unchanged_and_saves_each_response(tmp_path, monkeypatch):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    original_post = plex._post
    document = dict(failures=[])
    gate.run_plex_workflow(config, request, document)
    assert plex._post is original_post
    assert len(wires) == 3 and not document['failures']
    assert document['final_plex_score'] == 100
    assert document['plex']['completion']['passed']
    assert document['plex']['tool_results_source'] == 'synthetic_two_page_fixture'
    for receipt, response in zip(document['workflow_http'], responses):
        path = Path(receipt['response_path'])
        assert json.loads(path.read_text()) == response
        assert gate.sha(path) == receipt['response_sha256']
        assert path.stat().st_mode & 0o777 == 0o600
        assert all(receipt['checks'].values())
    assert json.loads((tmp_path/'result.turn3.progress.json').read_text())['turns'] == document['workflow_http']
    for index in (1, 2):
        progress = json.loads((tmp_path/f'result.turn{index}.progress.json').read_text())
        assert progress['turns'] == document['workflow_http'][:index]


def test_incomplete_workflow_stops_after_durable_receipt_and_restores_post(tmp_path, monkeypatch):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch, incomplete=True)
    original_post = plex._post
    document = dict(failures=[])
    with pytest.raises(RuntimeError, match='naturally complete'):
        gate.run_plex_workflow(config, request, document)
    assert plex._post is original_post and len(wires) == 1
    assert json.loads((tmp_path/'reply.turn1.json').read_text()) == responses[0]
    assert 'final_plex_score' not in document
    assert not document['workflow_http'][0]['checks']['completed']


@pytest.mark.parametrize('with_witness', [True, False])
def test_full_state_workflow_requires_phase_witness_before_continuation(tmp_path, monkeypatch, with_witness):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    config['require_full_prompt_state'] = True
    if with_witness:
        for response in responses:
            response['vmodel_cache_phases'] = [phase()]
        document = dict(failures=[])
        gate.run_plex_workflow(config, request, document)
        assert len(wires) == 3 and not document['failures']
    else:
        document = dict(failures=[])
        with pytest.raises(RuntimeError, match='per-phase witnesses'):
            gate.run_plex_workflow(config, request, document)
        assert len(wires) == 1 and 'final_plex_score' not in document
        assert not document['workflow_http'][0]['checks']['full_prompt_state']
        assert (tmp_path/'reply.turn1.json').exists()


def test_workflow_rejects_tool_schema_rewrite_before_http(tmp_path, monkeypatch):
    request, config, _, wires = workflow_setup(tmp_path, monkeypatch)
    original_post = plex._post
    def rewritten(current, url, **kwargs):
        current['tools'] = []
        return plex._post(url, current, 10)
    monkeypatch.setattr(plex, 'run_profile', rewritten)
    with pytest.raises(AssertionError):
        gate.run_plex_workflow(config, request, dict(failures=[]))
    assert not wires and plex._post is original_post


@pytest.mark.parametrize('key', list(selection()))
def test_positive_model_tokens_do_not_hide_host_action_or_argument_repair(key):
    observed = selection(); observed[key] = 1
    checks = gate.acceptance(row(), dict(status='completed', output=[call({})],
        vmodel_tool_selection=observed), dict(profiles=['audit'], profile_digest='digest'))
    assert checks['raw_witness'] and not checks['model_authored_output']


def test_missing_host_action_provenance_is_not_model_only_proof():
    checks = gate.acceptance(row(), dict(status='completed', output=[call({})]),
        dict(profiles=['audit'], profile_digest='digest'))
    assert not checks['model_authored_output']


@pytest.mark.parametrize('branch', ['direct', 'tool'])
def test_direct_model_branch_precedes_execution_only_host_transforms(branch):
    observed = dict(gateway_phase='direct', gateway_host_routed=0,
        gateway_decision_branch=branch, gateway_deterministic_policy_rendered=0)
    assert gate.model_authored_output(dict(vmodel_tool_selection=observed))
    observed['gateway_pagination_host_routed'] = 1
    assert not gate.model_authored_output(dict(vmodel_tool_selection=observed))


def test_partial_or_host_direct_branch_is_not_model_only_proof():
    observed = dict(gateway_phase='direct', gateway_decision_branch='direct',
        gateway_deterministic_policy_rendered=0)
    assert not gate.model_authored_output(dict(vmodel_tool_selection=observed))
    observed['gateway_host_routed'] = 1
    assert not gate.model_authored_output(dict(vmodel_tool_selection=observed))


@pytest.mark.parametrize('evidence', [
    {'aborted_on_memory_retry': True}, {'error': 'aborted_on_memory_retry'},
    {'prefill_progress': [{'phase': 'memory_retry', 'completed': 1}]}])
def test_observed_retry_cannot_be_hidden_by_missing_terminal_timing(evidence):
    value = row(); value['timing'] = {}; value.update(evidence)
    assert not check(value)['no_retry']


def test_ordinary_prefill_progress_does_not_count_as_retry():
    value = row(); value['prefill_progress'] = [{'phase': 'prefill_layer', 'completed': 64}]
    assert check(value)['no_retry']


def test_quality_retry_mode_can_score_completion_without_passing_latency_gate(tmp_path, monkeypatch):
    request, config, _, wires = workflow_setup(tmp_path, monkeypatch,
        retry_evidence=True, abort_on_memory_retry=False)
    document = dict(failures=[])
    gate.run_plex_workflow(config, request, document)
    assert len(wires) == 3 and document['final_plex_score'] == 100
    assert document['plex']['completion']['passed']
    assert document['failures'] == ['turn1:no_retry']
