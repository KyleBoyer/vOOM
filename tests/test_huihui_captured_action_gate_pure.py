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


def full_state_checks(phases, **config):
    return gate.acceptance(row(), dict(status='completed', output=[call({})],
        vmodel_tool_selection=selection(), vmodel_cache_phases=phases),
        dict(profiles=['audit'], profile_digest='digest', require_full_prompt_state=True, **config))


def phase():
    return dict(prompt_state_approximate=0, qwen_lossy_suffix_prefill_early_layers=0,
        qwen_lossy_suffix_prefill_used=0, true_peak_metal_bytes=100,
        weight_store_bytes_read=123, weight_store_bytes_read_source='path_stats',
        weight_store_bytes_read_scope='single_engine_phase_logical_not_physical')


def test_full_state_witness_accepts_both_model_generations():
    assert all(full_state_checks([phase(), phase()]).values())


def test_paged_audit_only_changes_full_attention_residency():
    base, paged = {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-model-only-audit'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-paged256-audit'], environ=paged)
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(paged) == {**settings(base), 'VMODEL_QWEN35_KV_MAX_MB': '256'}


def paged_phase():
    return dict(**phase(), kv_layout='paged', paged_kv_budget_bytes=256_000_000,
        hybrid_recurrent_cache_attached=1, paged_kv_spills=12, paged_kv_reloads=8,
        qwen35_paged_online_attention=0, qwen35_paged_online_page_native=0)


def test_compact_factors_audit_changes_only_flat_mtp_rollback_storage():
    base, factors = {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-paged256-audit'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-factors-audit'], environ=factors)
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(factors) == {**settings(base), 'VMODEL_QWEN_MTP_COMPACT_KDA_ROLLBACK': '1'}


def test_direct_audit_only_removes_gateway_and_factor_arm_only_changes_rollback():
    base, direct, factors = {}, {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-paged256-audit'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-direct-audit'], environ=direct)
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-direct-factors-audit'], environ=factors)
    settings = lambda env: {k: v for k, v in env.items() if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(direct) == {**settings(base), 'VMODEL_FAST_TOOL_GATEWAY': '0'}
    assert settings(factors) == {**settings(direct), 'VMODEL_QWEN_MTP_COMPACT_KDA_ROLLBACK': '1'}


def test_full_workflow_lifetime_audit_preserves_gateway_and_other_policies():
    base, audit, direct = {}, {}, {}
    apply_runtime_profiles(['huihui-qwen38-27b-full-state-factors-audit'], environ=base)
    apply_runtime_profiles(['huihui-qwen38-27b-full-workflow-lifetime-audit'], environ=audit)
    apply_runtime_profiles(['huihui-qwen38-27b-hermes-factors-phase-head-audit'], environ=direct)
    settings = lambda env: {k: v for k, v in env.items()
        if k.startswith('VMODEL_') and k != 'VMODEL_PROFILE'}
    assert settings(audit) == {**settings(base), 'VMODEL_QWEN35_DENSE_HERMES_TOOLS': '1',
        'VMODEL_QWEN35_MIN_AVAILABLE_MB': '5600',
        'VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD_MIN_PROMPT_TOKENS': '0'}
    assert settings(audit) == {**settings(direct), 'VMODEL_FAST_TOOL_GATEWAY': '1'}


def lifetime_phase():
    from tests.test_captured_transition_tracking_gate_pure import completed_phase, factor_timing
    return dict(**paged_phase(), **completed_phase(), **factor_timing(),
        qwen_mtp_used=1, qwen35_serial_verify_suspend_lm_head=1,
        qwen35_serial_verify_suspend_lm_head_min_prompt_tokens=0,
        qwen35_serial_verify_suspend_lm_head_request_active=1,
        qwen35_serial_verify_head_restore_calls=0,
        qwen_mtp_target_head_restore_calls=3,
        qwen_mtp_target_head_suspend_enabled=1,
        qwen_mtp_target_head_suspend_request_active=1)


@pytest.mark.parametrize('bad,key', [
    ({}, 'all_phase_scalar_factors'),
    ({'qwen_mtp_kda_factor_rounds': 0}, 'all_phase_scalar_factors'),
    ({'qwen_mtp_kda_factor_restore_s': float('nan')}, 'all_phase_scalar_factors'),
    ({'qwen_mtp_target_head_restore_calls': None}, 'all_phase_head_lifetime'),
    ({'qwen_mtp_target_head_restore_calls': True}, 'all_phase_head_lifetime'),
    ({'qwen_mtp_target_head_suspend_request_active': 0}, 'all_phase_head_lifetime'),
    ({'qwen35_serial_verify_suspend_lm_head_min_prompt_tokens': 4096}, 'all_phase_head_lifetime')])
def test_lifetime_gate_checks_hidden_phase_not_just_public_summary(bad, key):
    good = lifetime_phase()
    config = dict(require_all_phase_completion=True, require_paged_kv=True,
        require_qwen_factors=True, require_qwen_phase_head=True)
    assert all(full_state_checks([good, good], **config).values())
    suspect = {**good, **bad} if bad else {}
    assert not full_state_checks([suspect, good], **config)[key]


@pytest.mark.parametrize('phases', [None, [], [None]])
def test_lifetime_gate_does_not_invent_missing_phase_evidence(phases):
    checks = full_state_checks(phases, require_qwen_factors=True, require_qwen_phase_head=True)
    assert not checks['all_phase_scalar_factors'] and not checks['all_phase_head_lifetime']


@pytest.mark.parametrize('flag,key,bad', [
    ('require_qwen_factors', 'qwen_mtp_kda_factor_rounds', 0),
    ('require_qwen_phase_head', 'qwen_mtp_target_head_restore_calls', 0)])
def test_lifetime_failure_is_saved_before_stopping_workflow(tmp_path, monkeypatch, flag, key, bad):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    config.update(require_all_phase_completion=True, **{flag: True})
    for response in responses:
        response['vmodel_cache_phases'] = [lifetime_phase(), lifetime_phase()]
    responses[0]['vmodel_cache_phases'][0][key] = bad
    original_post = plex._post
    document = dict(failures=[])
    with pytest.raises(RuntimeError, match='lifetime path'):
        gate.run_plex_workflow(config, request, document)
    assert plex._post is original_post and len(wires) == 1
    assert 'final_plex_score' not in document
    assert json.loads((tmp_path/'reply.turn1.json').read_text()) == responses[0]
    assert json.loads((tmp_path/'result.turn1.progress.json').read_text())['turns'] == document['workflow_http']


def test_paged_witness_requires_actual_spill_and_reload_on_both_phases():
    assert all(full_state_checks([paged_phase(), paged_phase()], require_paged_kv=True).values())


@pytest.mark.parametrize('bad', [{}, {'kv_layout': 'concatenated'},
    {'paged_kv_budget_bytes': 768_000_000}, {'hybrid_recurrent_cache_attached': 0},
    {'hybrid_recurrent_cache_attached': True}, {'paged_kv_spills': 0},
    {'paged_kv_reloads': 0}, {'paged_kv_reloads': True},
    {'qwen35_paged_online_attention': 1}, {'qwen35_paged_online_page_native': 1}])
def test_paged_witness_rejects_missing_or_wrong_hidden_phase(bad):
    suspect = {**paged_phase(), **bad} if bad else {}
    assert not full_state_checks([suspect, paged_phase()], require_paged_kv=True)['paged_kv_witness']


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
    assert document['plex_rubric_score'] == 100
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


def test_serial_recovery_overlay_is_single_setting_and_keeps_full_workflow_contract():
    base, candidate = {}, {}
    profiles = ['huihui-qwen38-27b-full-workflow-lifetime-audit',
                'generation-witness', 'host-activity-witness']
    apply_runtime_profiles(profiles, environ=base)
    apply_runtime_profiles(profiles + ['qwen35-serial-kv-reclaim'], environ=candidate)
    assert candidate == {**base, 'VMODEL_QWEN35_SERIAL_KV_RECLAIM': '1'}


@pytest.mark.parametrize('bad', [None, 'hidden', 'final'])
def test_serial_recovery_gate_preserves_receipts_before_continuing_or_stopping(tmp_path, monkeypatch, bad):
    from tests.test_qwen_kv_reclaim_witness_pure import response as recovery_response, trace, log, record
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    config.update(require_serial_kv_reclaim=True, serial_kv_budget_bytes=256_000_000,
                  workflow='plex', server_log=str(tmp_path/'server.log'))
    for response in responses:
        response.update(recovery_response(trace(), {}))
    if bad == 'hidden':
        responses[0]['vmodel_cache_phases'][0].pop('qwen35_serial_kv_reclaim')
    elif bad == 'final':
        responses[0]['vmodel_timing']['qwen35_serial_kv_reclaim'] = trace()
    post = gate._post
    def traced(*args, **kwargs):
        value = post(*args, **kwargs)
        value['timing'].update(responses[len(wires)-1]['vmodel_timing'])
        return value
    monkeypatch.setattr(gate, '_post', traced)
    document = dict(failures=[])
    if bad is None:
        gate.run_plex_workflow(config, request, document)
        assert len(wires) == 3 and not document['failures']
        assert document['final_plex_score'] == 100  # mocked answers; not a model score
        Path(config['server_log']).write_text(log(*[record() for _ in responses]))
        covered = gate.serial_recovery_coverage(config, document)
        assert covered['passed'] and covered['phase_attempts'] == 3
        Path(document['workflow_http'][0]['response_path']).write_text('{}')
        assert not gate.serial_recovery_coverage(config, document)['passed']
    else:
        with pytest.raises(RuntimeError, match='serial KV recovery witness'):
            gate.run_plex_workflow(config, request, document)
        assert len(wires) == 1 and 'final_plex_score' not in document
        assert json.loads((tmp_path/'reply.turn1.json').read_text()) == responses[0]
        assert json.loads((tmp_path/'result.turn1.progress.json').read_text())['turns'] == document['workflow_http']


@pytest.mark.parametrize('mode', ['extra_event', 'no_response', 'bad_hash', 'missing_log'])
def test_recovery_finalization_fails_closed_on_orphan_events_or_missing_artifacts(tmp_path, mode):
    from tests.test_qwen_kv_reclaim_witness_pure import response, record, log
    reply, server_log = tmp_path/'reply.json', tmp_path/'server.log'
    reply.write_text(json.dumps(response({})))
    server_log.write_text(log(record()) if mode == 'extra_event' else '')
    config = dict(workflow='plex', serial_kv_budget_bytes=256_000_000,
                  server_log=str(server_log))
    document = dict(workflow_http=[dict(response_path=str(reply), response_sha256=gate.sha(reply))])
    if mode == 'no_response':
        document['workflow_http'] = []
    elif mode == 'bad_hash':
        document['workflow_http'][0]['response_sha256'] = '0'*64
    elif mode == 'missing_log':
        config['server_log'] = str(tmp_path/'absent.log')
    assert not gate.serial_recovery_coverage(config, document)['passed']


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


def test_paging_gate_stops_after_receipt_when_full_state_is_not_actually_paged(tmp_path, monkeypatch):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    config.update(require_full_prompt_state=True, require_paged_kv=True)
    for response in responses:
        response['vmodel_cache_phases'] = [phase()]
    document = dict(failures=[])
    with pytest.raises(RuntimeError, match='Paged-state comparison'):
        gate.run_plex_workflow(config, request, document)
    assert len(wires) == 1 and 'final_plex_score' not in document
    assert (tmp_path/'reply.turn1.json').exists()


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


def test_tool_round_limit_preserves_partial_rubric_but_never_a_final_score(tmp_path, monkeypatch):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    template = copy.deepcopy(responses[0])
    responses.clear()
    for index, (kind, offset) in enumerate((('all', 0), ('movie', 0),
            ('show', 0), ('movie', 500), ('show', 500))):
        args = dict(mediaType=kind, ratingOperator='lte', limit=500, offset=offset)
        if kind == 'all':
            args.update(movieRatingValue='PG-13', showRatingValue='TV-Y7')
        else:
            args['ratingValue'] = 'PG-13' if kind == 'movie' else 'TV-Y7'
        response = copy.deepcopy(template)
        response['output'] = [dict(call(args), call_id=f'budget-{index}')]
        responses.append(response)
    document = dict(failures=[])
    gate.run_plex_workflow(config, request, document)
    assert len(wires) == 5 and len(document['workflow_http']) == 5
    assert document['plex']['protocol_failures'] == [dict(turn=5, reason='tool_round_limit')]
    assert document['plex']['completion']['unhandled_call_turns'] == [5]
    assert document['plex']['final_text'] == ''
    assert document['plex_rubric_score'] == document['plex']['rubric']['score'] == 68
    assert document['plex']['rubric']['exclusion_points'] == 15
    assert document['final_plex_score'] is None
    assert document['failures'] == ['completed_plex_quality_or_protocol']


@pytest.mark.parametrize('final_text', ['', 'ALPHA_G'])
def test_final_score_requires_completion_not_a_passing_rubric(tmp_path, monkeypatch, final_text):
    request, config, responses, wires = workflow_setup(tmp_path, monkeypatch)
    responses[-1]['output'][0]['content'][0]['text'] = final_text
    document = dict(failures=[])
    gate.run_plex_workflow(config, request, document)
    assert len(wires) == 3 and document['failures'] == ['completed_plex_quality_or_protocol']
    assert document['plex']['rubric']['passed'] is False
    assert document['plex_rubric_score'] == document['plex']['rubric']['score']
    if final_text:
        assert document['plex']['completion']['passed'] is True
        assert document['final_plex_score'] == document['plex_rubric_score']
    else:
        assert document['plex']['completion']['passed'] is False
        assert document['final_plex_score'] is None
