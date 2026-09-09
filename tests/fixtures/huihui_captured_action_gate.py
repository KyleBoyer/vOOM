#!/usr/bin/env python3
"""Completed initial captured action; never a final paginated Plex score.

Original input/tools/stream/reasoning; only model/max1024/temp0/seed overrides.
The selected lossy gateway profile transforms the model's prepared prompt.
All generated tools remain unexecuted. No final-answer rendering or repair.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.captured_transition_tracking_gate import (
    generation_phase_checks, native_pressure_summary, prepare_case,
    qwen_all_prompt_phase_head_path, qwen_scalar_factor_path, sha)
from tests.fixtures.qwen3_large_agent_replay_gate import _post
from tests.fixtures.qwen4_hot_boundary_http_probe import _atomic_write_private
from tests.fixtures.runtime_profile_http_gate import (
    _port_is_free, _pressure, _stop_server, _wait_ready)


def action_checks(response):
    """Only initial read-only listing validity, not final-answer intelligence."""
    output = response.get('output') or []
    calls = [x for x in output if isinstance(x, dict) and x.get('type') == 'function_call']
    arguments = []
    for call in calls:
        try:
            value = json.loads(call.get('arguments', ''))
            arguments.append(value if isinstance(value, dict) else None)
        except (ValueError, TypeError):
            arguments.append(None)
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value == int(value)
    return dict(
        listing_calls_only=1 <= len(calls) <= 2 and all(
            c.get('name') == 'plugin__plex__plex_list_library' for c in calls),
        object_arguments=bool(arguments) and all(isinstance(a, dict) for a in arguments),
        bounded_first_page=bool(arguments) and all(isinstance(a, dict)
            and (a.get('offset') is None or (number(a['offset']) and a['offset'] == 0))
            and (a.get('limit') is None or (number(a['limit']) and 0 < a['limit'] <= 100))
            for a in arguments))


def model_authored_output(response):
    selection = response.get('vmodel_tool_selection') or {}
    flags = ('gateway_pagination_host_routed',
        'gateway_initial_pagination_defaults_applied', 'gateway_literal_arguments_grounded')
    if selection.get('gateway_deterministic_policy_rendered') != 0:
        return False
    if any(selection.get(key, 0) != 0 for key in flags):
        return False
    # server.py's direct decision returns before execution pagination/defaults/
    # grounding. Those execution-only flags are absent on this explicit branch.
    if (selection.get('gateway_phase') == 'direct'
            and selection.get('gateway_host_routed') == 0
            and selection.get('gateway_decision_branch') in ('direct', 'tool')):
        return True
    return all(selection.get(key) == 0 for key in flags)


def acceptance(row, response, config, *, initial_action=True):
    t, usage = row.get('timing') or {}, row.get('usage') or {}
    witness = t.get('generation_witness') or {}
    before, after = row['pressure_before'], row['pressure_after']
    checks = action_checks(response) if initial_action else {}
    expected_description = config.get('gateway_enable_description_profile')
    if expected_description is not None:
        checks['gateway_enable_description_profile'] = (
            expected_description in ('legacy', 'external-only-v1')
            and (response.get('vmodel_tool_selection') or {}).get(
                'gateway_enable_description_profile') == expected_description)
    if config.get('workflow') == 'saved_budget_extension':
        from tests.fixtures import plex_agent_profile as plex
        checks['terminal_answer_without_pending_calls'] = (
            isinstance(response.get('output'), list)
            and bool(plex.response_text(response).strip())
            and not plex.response_calls(response)
            and not any(isinstance(item, dict) and item.get('type') == 'function_call'
                for item in response.get('output', [])))
    if config.get('require_all_phase_completion', False):
        checks.update(generation_phase_checks(response))
    if config.get('require_serial_kv_reclaim', False):
        from tests.fixtures.qwen_kv_reclaim_witness import phase_checks
        checks.update(phase_checks(response, t,
            budget_bytes=config['serial_kv_budget_bytes'],
            topup_required=config.get('require_serial_kv_reclaim_topup', False)))
    checks.update(
        completed=row.get('http_status') == 200 and row.get('response_status') == 'completed'
            and not row.get('error') and response.get('status') == 'completed',
        not_output_capped=type(usage.get('output_tokens')) is int
            and 0 < usage['output_tokens'] < 1024,
        raw_witness=witness.get('available') is True
            and witness.get('prepared_prompt_token_ids_sha256') is not None
            and type(witness.get('generated_token_count')) is int
            and witness['generated_token_count'] > 0,
        model_authored_output=model_authored_output(response),
        # Gateway workflows may generate more than once. This hash witnesses
        # the exposed engine generation, not aggregate workflow token equality.
        profile=row.get('runtime_profiles') == config['profiles']
            and row.get('runtime_profile_digest') == config['profile_digest']
            and not row.get('runtime_profile_overrides'),
        backend=row.get('backend') == 'voom',
        no_retry=not row.get('aborted_on_memory_retry')
            and row.get('error') != 'aborted_on_memory_retry'
            and not any(event.get('phase') == 'memory_retry'
                for event in row.get('prefill_progress', []) if isinstance(event, dict))
            and int(t.get('memory_prefill_retries') or 0) == 0,
        no_prompt_reuse=usage.get('input_tokens_details', {}).get('cached_tokens') == 0,
        metal=type(t.get('true_peak_metal_bytes')) in (int, float)
            and 0 < t['true_peak_metal_bytes'] <= 8_500_000_000,
        terminal_available=after['available_bytes'] >= 5_300_000_000,
        swap_used=after['swap_used_bytes'] - before['swap_used_bytes'] <= 16_000_000,
        actual_swap_out=after['swap_out_bytes'] - before['swap_out_bytes'] <= 16_000_000,
        stream_matches=row.get('streamed_text_matches_final') is True)
    if config.get('require_full_prompt_state', False):
        phases = response.get('vmodel_cache_phases')
        valid = isinstance(phases, list) and bool(phases) and all(
            isinstance(phase, dict) for phase in phases)
        state_keys = ('prompt_state_approximate',
            'qwen_lossy_suffix_prefill_early_layers', 'qwen_lossy_suffix_prefill_used')
        checks['full_prompt_state'] = valid and all(
            type(phase.get(key)) is int and phase[key] == 0
            for phase in phases for key in state_keys)
        checks['phase_io_witness'] = valid and all(
            phase.get('weight_store_bytes_read_source') == 'path_stats'
            and phase.get('weight_store_bytes_read_scope') == 'single_engine_phase_logical_not_physical'
            and type(phase.get('weight_store_bytes_read')) is int
            and phase['weight_store_bytes_read'] > 0 for phase in phases)
        checks['all_phase_metal'] = valid and all(
            type(phase.get('true_peak_metal_bytes')) is int
            and 0 < phase['true_peak_metal_bytes'] <= 8_500_000_000 for phase in phases)
    if config.get('require_paged_kv', False):
        phases = response.get('vmodel_cache_phases')
        valid = isinstance(phases, list) and bool(phases) and all(
            isinstance(phase, dict) for phase in phases)
        checks['paged_kv_witness'] = valid and all(
            phase.get('kv_layout') == 'paged'
            and type(phase.get('paged_kv_budget_bytes')) is int
            and phase.get('paged_kv_budget_bytes') == 256_000_000
            and type(phase.get('hybrid_recurrent_cache_attached')) is int
            and phase['hybrid_recurrent_cache_attached'] == 1
            and all(type(phase.get(key)) is int and phase[key] > 0
                for key in ('paged_kv_spills', 'paged_kv_reloads'))
            and all(type(phase.get(key)) is int and phase[key] == 0
                for key in ('qwen35_paged_online_attention', 'qwen35_paged_online_page_native'))
            for phase in phases)
    for flag, key, predicate in (
        ('require_qwen_factors', 'all_phase_scalar_factors', qwen_scalar_factor_path),
        ('require_qwen_phase_head', 'all_phase_head_lifetime', qwen_all_prompt_phase_head_path)):
        if config.get(flag, False):
            phases = response.get('vmodel_cache_phases')
            checks[key] = (isinstance(phases, list) and bool(phases)
                and all(isinstance(phase, dict) and predicate(phase) for phase in phases))
    return checks


def run_plex_workflow(config, request, document):
    """Retain the legacy fixed two-page rubric, with immutable HTTP receipts.

The synthetic mixed-page queue is not live Plex or independent movie/show
pagination. Its unchanged rubric has known separate-media strategy limitations;
report those independently of actual final-title errors, never repair a score.
"""
    from tests.fixtures import plex_agent_profile as plex

    base = copy.deepcopy(request)
    receipts = document['workflow_http'] = []
    def recording_post(url, current, timeout):
        index = len(receipts) + 1
        assert 1 <= index <= 5
        assert current['tools'] == base['tools']
        assert current['input'][:len(base['input'])] == base['input']
        assert {k: v for k, v in current.items() if k != 'input'} == {
            k: v for k, v in base.items() if k != 'input'}
        wire = json.dumps(current, ensure_ascii=False, separators=(',', ':')).encode()
        if index == 1:
            import hashlib
            assert hashlib.sha256(wire).hexdigest() == config['wire_sha256']
        response_path = Path(config['response']).with_name(
            Path(config['response']).stem + f'.turn{index}.json')
        assert not response_path.exists()
        terminal = []
        def observe(response):
            assert not terminal
            _atomic_write_private(response_path, response)
            terminal.append(response)
        before = asdict(_pressure())
        row = _post(url, wire, timeout=timeout, stream=True, print_progress=True,
            fail_on_memory_retry=config.get('abort_on_memory_retry', True),
            response_observer=observe)
        row.update(pressure_before=before, pressure_after=asdict(_pressure()))
        response = terminal[0] if terminal else {}
        checks = acceptance(row, response, config, initial_action=False)
        receipts.append(dict(turn=index, request=plex.request_shape(current),
            response_path=str(response_path),
            response_sha256=sha(response_path) if terminal else None,
            row=row, checks=checks))
        # Persist each completed HTTP receipt even if a later turn fails.
        _atomic_write_private(Path(config['result']).with_suffix(f'.turn{index}.progress.json'),
            dict(schema='voom.huihui-plex-progress.v1', state='running', turns=receipts))
        print(json.dumps(dict(plex_turn=index, wall_seconds=row.get('wall_seconds'),
            output_tokens=(row.get('usage') or {}).get('output_tokens'),
            checks=checks)), flush=True)
        if not checks['completed'] or not checks['not_output_capped'] or not checks['model_authored_output']:
            raise RuntimeError('Plex turn did not naturally complete with model-authored output')
        if config.get('require_all_phase_completion', False) and not all(
                checks[key] for key in ('all_phase_natural_termination', 'all_phase_generation_witness')):
            raise RuntimeError('Plex turn has incomplete or unwitnessed hidden/public generation')
        if config.get('require_full_prompt_state', False) and not all(
                checks[key] for key in ('full_prompt_state', 'phase_io_witness', 'all_phase_metal')):
            raise RuntimeError('Full-state comparison lacks required per-phase witnesses')
        if config.get('require_paged_kv', False) and not checks['paged_kv_witness']:
            raise RuntimeError('Paged-state comparison lacks required per-phase witnesses')
        for flag, key in (('require_qwen_factors', 'all_phase_scalar_factors'),
                          ('require_qwen_phase_head', 'all_phase_head_lifetime')):
            if config.get(flag, False) and not checks[key]:
                raise RuntimeError('Required lifetime path is absent from a hidden/public phase')
        if config.get('require_serial_kv_reclaim', False) and not all(checks[key]
                for key in ('all_phase_serial_kv_reclaim_witness',
                            'final_serial_kv_reclaim_trace_matches')):
            raise RuntimeError('Required serial KV recovery witness is incomplete')
        return response, row['wall_seconds']

    # Scope this observer to one synchronous, single-server fixture call.
    # No production code, page contents or model-authored response is changed.
    with patch.object(plex, '_post', recording_post):
        result = plex.run_profile(request, f'http://127.0.0.1:{config["port"]}/v1/responses',
            timeout=1800, max_tool_rounds=4)
    document['plex'] = result
    # Planning points (and vacuous exclusion points on empty text) are not
    # a completed-answer grade. Retain the unchanged rubric diagnostically,
    # but never label it final while tool calls or the final answer are pending.
    document['plex_rubric_score'] = result['rubric']['score']
    document['final_plex_score'] = (result['rubric']['score']
        if result.get('completion', {}).get('passed') is True else None)
    if not result['passed']:
        document['failures'].append('completed_plex_quality_or_protocol')
    for receipt in receipts:
        document['failures'] += [f'turn{receipt["turn"]}:{key}'
            for key, ok in receipt['checks'].items() if not ok]
    before, after = receipts[0]['row']['pressure_before'], receipts[-1]['row']['pressure_after']
    if max(after['swap_out_bytes'] - before['swap_out_bytes'],
           after['swap_used_bytes'] - before['swap_used_bytes']) > 16_000_000:
        document['failures'].append('whole_workflow_http_swap_growth')


def serial_recovery_coverage(config, document):
    """Verify immutable terminal files and the independent complete server log."""
    from tests.fixtures.qwen_kv_reclaim_witness import log_coverage
    try:
        if config.get('workflow', 'initial_action') == 'plex':
            files = [(r['response_path'], r['response_sha256'])
                for r in document.get('workflow_http', []) if r.get('response_sha256')]
        else:
            files = [(config['response'], document.get('response_sha256'))]
        responses = []
        for path, digest in files:
            if not digest or sha(path) != digest:
                raise ValueError('terminal response identity mismatch')
            responses.append(json.loads(Path(path).read_text()))
        return log_coverage(responses, Path(config['server_log']).read_text(),
            budget_bytes=config['serial_kv_budget_bytes'],
            topup_required=config.get('require_serial_kv_reclaim_topup', False))
    except (OSError, ValueError, TypeError, KeyError):
        return dict(passed=False, complete_phase_coverage=False)


def prepare_request(config):
    """Optionally isolate the exact second HTTP of a pinned prior workflow.

    Omitting the first HTTP's process history is the experimental variable,
    not a prompt/tool rewrite or a completed-workflow latency comparison.
    """
    from tests.fixtures import plex_agent_profile as plex

    request, wire, metadata = prepare_case(config['case'], config['model'])
    if config.get('workflow') == 'saved_budget_extension':
        return prepare_budget_extension(config, request, metadata)
    if config.get('workflow', 'initial_action') != 'saved_continuation':
        return request, wire, metadata
    source = config['saved_continuation']
    if sha(source['result']) != source['sha256']:
        raise ValueError('saved workflow result identity mismatch')
    reference = json.loads(Path(source['result']).read_text())
    ref_config = reference['config']
    if (reference.get('schema') != 'voom.huihui-captured-plex.v1'
            or any(config.get(key) != ref_config.get(key) for key in (
                'case', 'model', 'profiles', 'profile_digest', 'metadata_hashes'))
            or metadata != reference.get('request')):
        raise ValueError('saved workflow source configuration mismatch')
    prior = reference['workflow_http']
    if len(prior) < 2 or prior[0]['request'] != plex.request_shape(request):
        raise ValueError('saved workflow does not witness the first two requests')
    first = prior[0]
    if sha(first['response_path']) != first['response_sha256']:
        raise ValueError('saved first response identity mismatch')
    response = json.loads(Path(first['response_path']).read_text())
    if (response.get('status') != 'completed' or not model_authored_output(response)
            or not all(generation_phase_checks(response).values())):
        raise ValueError('saved first response is not naturally completed/model authored')
    calls = plex.response_calls(response)
    if (len(calls) != 1 or calls[0]['name'] not in plex.PLEX_PAGINATION_TOOLS
            or not isinstance(calls[0]['arguments'], dict) or not calls[0]['call_id']):
        raise ValueError('requires one exact saved pagination call')
    continuation = copy.deepcopy(request)
    plex._append_call_and_result(continuation, calls[0], plex.SYNTHETIC_PAGES[0])
    shape = plex.request_shape(continuation)
    if shape != prior[1]['request']:
        raise ValueError('reconstructed continuation differs from saved second HTTP')
    wire = json.dumps(continuation, ensure_ascii=False, separators=(',', ':')).encode()
    metadata = dict(request_sha256=hashlib.sha256(wire).hexdigest(),
        request_bytes=len(wire), original_capture_request=metadata,
        request_shape=shape, saved_workflow_result_sha256=source['sha256'],
        saved_first_response_sha256=first['response_sha256'],
        original_workflow_turn=2, preceding_http_calls_omitted=1,
        tool_results_source='unchanged_synthetic_first_page',
        request_change='append_exact_saved_call_and_unchanged_first_fixture_page',
        scope='Identical saved second HTTP in a fresh server; not full-workflow latency or Plex score')
    return continuation, wire, metadata


def prepare_budget_extension(config, request, metadata):
    """One extra diagnostic HTTP after a pinned tool-budget failure, not a replay.

    Reconstruct every prior request from its exact saved model call and the
    unchanged legacy page queue. No new instructions, tools, sampling choices
    or output budget are introduced; previous process/cache history is omitted.
    This cannot change the prior workflow verdict or produce a final Plex grade.
    """
    from tests.fixtures import plex_agent_profile as plex

    source = config['saved_budget_extension']
    def same_json(left, right):
        # Python equality otherwise aliases True/1 and False/0 in provenance.
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
    if type(source.get('additional_http_calls')) is not int or source['additional_http_calls'] != 1:
        raise ValueError('budget extension requires exactly one explicit additional HTTP')
    if sha(source['result']) != source['sha256']:
        raise ValueError('saved workflow result identity mismatch')
    reference = json.loads(Path(source['result']).read_text())
    ref_config = reference['config']
    if (reference.get('schema') != 'voom.huihui-captured-plex.v1'
            or ref_config.get('workflow') != 'plex'
            or any(config.get(key) != ref_config.get(key) for key in (
                'case', 'model', 'profiles', 'profile_digest', 'metadata_hashes'))
            or metadata != reference.get('request')):
        raise ValueError('saved workflow source configuration mismatch')
    prior = reference.get('workflow_http')
    if not isinstance(prior, list) or not 1 <= len(prior) <= 5:
        raise ValueError('requires one to five witnessed prior HTTPs')
    fixture = reference.get('plex', {})
    count = len(prior)
    if (fixture.get('tool_results_source') != 'synthetic_two_page_fixture'
            or fixture.get('protocol_failures') != [dict(turn=count, reason='tool_round_limit')]
            or not same_json(fixture.get('completion'), dict(passed=False, non_completed_turns=[],
                error_turns=[], terminal_has_unhandled_calls=True,
                unhandled_call_turns=[count], terminal_text_present=False))
            or fixture.get('final_text') != ''
            or not isinstance(fixture.get('turns'), list) or len(fixture['turns']) != count):
        raise ValueError('source must end only with an unhandled tool-budget call')
    current = copy.deepcopy(request)
    hashes, page_indices = [], []
    for index, entry in enumerate(prior, 1):
        if type(entry.get('turn')) is not int or entry['turn'] != index:
            raise ValueError('saved HTTP order mismatch')
        if not same_json(entry.get('request'), plex.request_shape(current)):
            raise ValueError('reconstructed request differs from saved HTTP')
        if sha(entry['response_path']) != entry['response_sha256']:
            raise ValueError('saved response identity mismatch')
        response = json.loads(Path(entry['response_path']).read_text())
        if (response.get('status') != 'completed' or not model_authored_output(response)
                or not all(generation_phase_checks(response).values())):
            raise ValueError('saved response is not naturally completed/model authored')
        calls = plex.response_calls(response)
        if (len(calls) != 1 or calls[0]['name'] not in plex.PLEX_PAGINATION_TOOLS
                or not isinstance(calls[0]['arguments'], dict) or not calls[0]['call_id']):
            raise ValueError('requires one exact saved pagination call per HTTP')
        wall = entry.get('row', {}).get('wall_seconds')
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError('invalid saved HTTP duration')
        turn = plex._response_turn(response, wall, index, current)
        turn['handled_call_count'] = int(index < count)
        if not same_json(turn, fixture['turns'][index - 1]):
            raise ValueError('saved fixture turn provenance mismatch')
        page_index = min(index - 1, len(plex.SYNTHETIC_PAGES) - 1)
        plex._append_call_and_result(current, calls[0], plex.SYNTHETIC_PAGES[page_index])
        hashes.append(entry['response_sha256'])
        page_indices.append(page_index)
    wire = json.dumps(current, ensure_ascii=False, separators=(',', ':')).encode()
    return current, wire, dict(request_sha256=hashlib.sha256(wire).hexdigest(),
        request_bytes=len(wire), original_capture_request=metadata,
        request_shape=plex.request_shape(current), saved_workflow_result_sha256=source['sha256'],
        saved_response_sha256=hashes, appended_fixture_page_indices=page_indices,
        original_workflow_turn=count + 1, preceding_http_calls_omitted=count,
        additional_http_calls=1, original_handled_tool_round_budget=count - 1,
        tool_results_source='unchanged_synthetic_two_page_queue',
        request_change='append_exact_saved_calls_and_unchanged_fixture_pages',
        scope='One explicitly extra-budget diagnostic HTTP in a fresh server; prior workflow remains failed. Not original-workflow replay, complete-workflow latency or Plex score.')


def run_single_request(config, wire, document, *, initial_action):
    """Durably save one HTTP response before applying its unchanged gates."""
    terminal = []
    def observe(value):
        assert not terminal
        _atomic_write_private(Path(config['response']), value)
        terminal.append(value)
    before = asdict(_pressure())
    row = _post(f'http://127.0.0.1:{config["port"]}/v1/responses', wire,
        timeout=1800, stream=True, print_progress=True,
        fail_on_memory_retry=config.get('abort_on_memory_retry', True),
        response_observer=observe)
    row.update(pressure_before=before, pressure_after=asdict(_pressure()))
    document['row'] = row
    document['checks'] = acceptance(row, terminal[0] if terminal else {}, config,
        initial_action=initial_action)
    document['failures'] += [k for k, v in document['checks'].items() if not v]
    if terminal:
        document['response_sha256'] = sha(config['response'])


def run(config):
    assert config.get('require_all_phase_completion') is True
    # Quality-only retries retain the runtime governor and full charged wall;
    # no_retry still fails. The default remains fail-fast for latency audits.
    assert type(config.get('abort_on_memory_retry', True)) is bool
    assert type(config.get('require_full_prompt_state', False)) is bool
    assert type(config.get('require_paged_kv', False)) is bool
    for flag in ('require_qwen_factors', 'require_qwen_phase_head', 'require_serial_kv_reclaim'):
        assert type(config.get(flag, False)) is bool
    assert not config.get('require_paged_kv', False) or config.get('require_full_prompt_state', False)
    assert not any(k.startswith('VMODEL_') for k in os.environ)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == config['source_commit']
    pre = json.loads(Path(config['preflight']).read_text())
    assert pre['passed'] and pre['sample_seconds'] >= 30
    assert 0 <= time.monotonic() - pre['end']['monotonic_s'] < 120
    assert pre['known_transcoders']['passed'] is True
    assert pre['end']['root_free_bytes'] >= 10_000_000_000
    env = {}
    profile = apply_runtime_profiles(config['profiles'], environ=env)
    assert profile.profile_digest == config['profile_digest']
    if config.get('gateway_enable_description_profile') is not None:
        from runtime.server import _hidden_gateway_enable_description_policy
        assert config['gateway_enable_description_profile'] == (
            _hidden_gateway_enable_description_policy(env.get(
                'VMODEL_FAST_TOOL_GATEWAY_ENABLE_EXTERNAL_ONLY', '0')))
    serial_kv_required = env.get('VMODEL_QWEN35_SERIAL_KV_RECLAIM') == '1'
    from tests.fixtures.qwen_kv_reclaim_witness import validate_config
    validate_config(config, env)
    if serial_kv_required:
        assert config.get('require_paged_kv') is True
        assert type(config.get('serial_kv_budget_bytes')) is int
        assert config['serial_kv_budget_bytes'] == 256_000_000
    assert env['VMODEL_FAST_TOOL_GATEWAY_DETERMINISTIC_POLICY'] == '0'
    assert env['VMODEL_FAST_TOOL_GATEWAY_HOST_ROUTE'] == '0'
    assert env['VMODEL_QWEN35_HOT_KV'] == '0'
    assert env['VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST'] == '0'
    if config.get('require_full_prompt_state', False):
        assert env['VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL'] == 'off'
    if config.get('require_paged_kv', False):
        assert env['VMODEL_QWEN35_KV_MAX_MB'] == '256'
        assert env.get('VMODEL_QWEN35_PAGED_ONLINE_ATTENTION', '0') == '0'
        assert env.get('VMODEL_QWEN35_PAGED_ONLINE_PAGE_NATIVE', '0') == '0'
        assert env.get('VMODEL_QWEN35_PAGED_KV_PERSIST', '0') == '0'
        assert env['VMODEL_QWEN35_FP8_KV_CACHE'] == '0'
    assert env['VMODEL_GENERATION_WITNESS'] == env['VMODEL_HOST_ACTIVITY_WITNESS'] == '1'
    if config.get('require_qwen_factors', False):
        assert env['VMODEL_QWEN_MTP_COMPACT_KDA_ROLLBACK'] == '1'
    if config.get('require_qwen_phase_head', False):
        assert env['VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD'] == '1'
        assert env['VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD_MIN_PROMPT_TOKENS'] == '0'
    assert _port_is_free(config['port'])
    for key in ('result', 'response', 'server_log'):
        assert not Path(config[key]).exists()
    workflow = config.get('workflow', 'initial_action')
    assert workflow in ('initial_action', 'plex', 'saved_continuation', 'saved_budget_extension')
    if workflow == 'plex':
        for index in range(1, 6):
            assert not Path(config['result']).with_suffix(f'.turn{index}.progress.json').exists()
            assert not Path(config['response']).with_name(
                Path(config['response']).stem + f'.turn{index}.json').exists()
    request, wire, metadata = prepare_request(config)
    assert metadata['request_sha256'] == config['wire_sha256']
    assert len(request['tools']) == 134 and request['stream'] is True
    for path, digest in config['metadata_hashes'].items():
        assert sha(path) == digest
    started = time.perf_counter()
    document = dict(schema='voom.huihui-captured-initial-action.v1', passed=False,
        scope=__doc__, config=config, preflight_sha256=sha(config['preflight']),
        request=metadata, generated_tools_executed=False, final_plex_score=None,
        failures=[], source_commit=config['source_commit'])
    if workflow == 'plex':
        document.update(schema='voom.huihui-captured-plex.v1',
            scope='Model-only capped-five-response Plex workflow: original134-tool HTTP catalog/history/stream; model/max1024/temp0/seed64013 overrides. Existing lossy gateway prepares a smaller catalog/prompt. Legacy synthetic mixed two-page fixture and unchanged rubric, with known separate-media strategy limitations. No live tool execution, host rendering, full-schema model replay or BF16 proof.')
    elif workflow == 'saved_continuation':
        document.update(schema='voom.huihui-saved-continuation.v1',
            scope='Diagnostic fresh-server replay of the exact saved second HTTP, reconstructed from the original134-tool capture plus the exact first model call and unchanged synthetic first page. First HTTP process/cache history is intentionally omitted. Same model/max1024/temp0/seed64013 and explicit lossy gateway preparation. No live tools, complete-workflow timing, Plex score, full-schema model replay or BF16 proof.')
    elif workflow == 'saved_budget_extension':
        document.update(schema='voom.huihui-saved-budget-extension.v1',
            scope='One explicitly additional HTTP beyond a pinned prior tool-round budget. Every previous request/call/fixture page is reconstructed and checked; unchanged original134-tool capture and sampling/output settings. Prior process/cache history is omitted, prior workflow remains failed. Require a natural terminal answer with no pending calls; no Plex score, live tool execution, original-workflow latency or BF16 claim.')
    server = None
    try:
        with open(config['server_log'], 'x') as log:
            command = [sys.executable, '-m', 'runtime.server', '--port', str(config['port'])]
            for name in config['profiles']:
                command += ['--profile', name]
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            document['server_pid'] = server.pid
            registry = _wait_ready(server, config['port'], 120)
            assert registry['vmodel_runtime_profile_digest'] == config['profile_digest']
            assert registry['vmodel_runtime_profiles'] == config['profiles']
            assert not registry.get('vmodel_runtime_profile_overrides')
            if workflow == 'plex':
                run_plex_workflow(config, request, document)
            else:
                run_single_request(config, wire, document,
                    initial_action=workflow == 'initial_action')
    except BaseException as error:
        document['failures'].append('driver_error:' + type(error).__name__)
    finally:
        if server is not None:
            _stop_server(server)
            document['server_returncode'] = server.returncode
        try:
            document['server_log_sha256'] = sha(config['server_log'])
            document['native_pressure'] = native_pressure_summary(Path(config['server_log']).read_text())
        except (OSError, ValueError, TypeError, KeyError):
            document['native_pressure'] = dict(available=False, passed=False)
        if not document['native_pressure'].get('passed'):
            document['failures'].append('whole_run_pressure')
        if not document['native_pressure'].get('known_transcoders', {}).get('passed'):
            document['failures'].append('known_transcoder_isolation')
        if serial_kv_required:
            document['serial_kv_reclaim_coverage'] = serial_recovery_coverage(config, document)
            if not document['serial_kv_reclaim_coverage']['passed']:
                document['failures'].append('serial_kv_recovery_event_phase_coverage')
        document['metadata_unchanged'] = all(
            sha(path) == digest for path, digest in config['metadata_hashes'].items())
        if not document['metadata_unchanged']:
            document['failures'].append('model_metadata_changed')
        document['wall_seconds'] = time.perf_counter() - started
        document['passed'] = not document['failures']
        _atomic_write_private(Path(config['result']), document)
    print(json.dumps(dict(passed=document['passed'], failures=document['failures'],
        wall_seconds=document['wall_seconds'])), flush=True)
    return 0 if document['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', required=True)
    raise SystemExit(run(json.loads(parser.parse_args().config_json)))
