"""Recovery evidence must cover all phases and match the independent log."""

import copy
import json

import pytest

from tests.fixtures.qwen_kv_reclaim_witness import valid_trace, phase_checks, log_coverage
from tests.fixtures.captured_transition_tracking_gate import row_checks
from tests.test_captured_transition_tracking_gate_pure import valid_row


BUDGET = 256_000_000
KEY = 'qwen35_serial_kv_reclaim'


def record():
    return dict(schema='voom.qwen35-serial-kv-reclaim.v1', outcome='admitted',
        reservation_retried=True, layer=7, start_offset=5046, verifier_positions=5,
        incoming_bytes=90, margin_bytes=20, kv_budget_bytes=BUDGET,
        logical_before_bytes=100, logical_after_bytes=40, logical_reclaimed_bytes=60,
        requested_bytes=60, spill_pages=1, metal_active_released_bytes=60,
        before=dict(metal_active_bytes=100, system_available_bytes=1000,
                    ceiling_bytes=150, deficit_bytes=60),
        after_reclaim=dict(metal_active_bytes=40, system_available_bytes=1000,
                           ceiling_bytes=150, deficit_bytes=0),
        reclaim_seconds=0.2, spill_seconds=0.1, wall_seconds=0.3)


def trace(rows=None):
    rows = [record()] if rows is None else rows
    return dict(attempts=len(rows), admitted=len(rows), records=rows,
        **{key: sum(row[key] for row in rows) for key in (
            'logical_reclaimed_bytes', 'spill_pages', 'spill_seconds', 'wall_seconds')})


def phase(value):
    return {KEY: value, KEY+'_enabled': 1, 'paged_kv_budget_bytes': BUDGET}


def response(*values):
    phases = [phase(v) for v in values]
    return dict(vmodel_cache_phases=phases, vmodel_timing=copy.deepcopy(phases[-1]))


def log(*rows):
    return '\n'.join('[qwen35-serial-kv-reclaim] '+json.dumps(r) for r in rows)


def test_explicit_empty_is_inactive_coverage_never_a_memory_win():
    doc = response({}, {})
    assert valid_trace({}, budget_bytes=BUDGET)
    checks = phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET)
    assert all(checks.values())
    observed = log_coverage([doc], '', budget_bytes=BUDGET)
    assert observed['passed'] and observed['phase_attempts'] == 0
    assert observed['positive_active_release_attempts'] == 0
    assert observed['metal_active_released_bytes'] == 0


@pytest.mark.parametrize('value', [None, [], '', 0, True, {'attempts': 0},
    {'attempts': 0, 'admitted': 0, 'records': []}, {'records': []}])
def test_absent_malformed_or_partial_trace_is_not_explicit_empty(value):
    assert not valid_trace(value, budget_bytes=BUDGET)


def test_success_requires_both_phases_and_exact_final_and_log_order():
    first, second = record(), {**record(), 'layer': 11, 'start_offset': 5049}
    doc = response(trace([first]), trace([second]))
    assert all(phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET).values())
    summary = log_coverage([doc], log(first, second), budget_bytes=BUDGET)
    assert summary['passed'] and summary['phase_attempts'] == 2
    assert summary['logical_reclaimed_bytes'] == summary['metal_active_released_bytes'] == 120
    assert not log_coverage([doc], log(second, first), budget_bytes=BUDGET)['passed']
    assert not log_coverage([doc], log(first), budget_bytes=BUDGET)['passed']
    assert not log_coverage([doc], log(first, second, second), budget_bytes=BUDGET)['passed']


@pytest.mark.parametrize('field,value', [('attempts', True), ('attempts', 2),
    ('admitted', 0), ('admitted', 1.0), ('records_dropped', 1), ('records_dropped', True),
    ('log_errors', 1), ('refused', 1), ('error', 1), ('no_candidates', 1),
    ('logical_reclaimed_bytes', 61), ('spill_pages', 2), ('spill_seconds', float('nan')),
    ('wall_seconds', float('inf'))])
def test_inconsistent_capped_failed_or_nonfinite_summary_fails(field, value):
    stats = trace()
    stats[field] = value
    assert not valid_trace(stats, budget_bytes=BUDGET)


@pytest.mark.parametrize('field,value', [('schema', 'wrong'), ('outcome', 'refused'),
    ('reservation_retried', False), ('reservation_retried', 1), ('layer', True),
    ('verifier_positions', 0), ('incoming_bytes', -1), ('kv_budget_bytes', BUDGET-1),
    ('logical_after_bytes', 41), ('requested_bytes', 61), ('spill_pages', 0),
    ('metal_active_released_bytes', 61), ('metal_active_released_bytes', None),
    ('metal_active_released_bytes', 60.0), ('error_type', 'OSError'),
    ('wall_seconds', 0.1), ('reclaim_seconds', float('nan')),
    ('spill_seconds', 0.5), ('before', {}), ('after_reclaim', None)])
def test_bad_record_never_passes_even_with_recomputed_aggregates(field, value):
    row = {**record(), field: value}
    assert not valid_trace(trace([row]), budget_bytes=BUDGET)


def test_deficits_are_recomputed_including_margin():
    row = record()
    row['before']['deficit_bytes'] = row['requested_bytes'] = 40
    assert not valid_trace(trace([row]), budget_bytes=BUDGET)


@pytest.mark.parametrize('physical', [0, -10])
def test_signed_active_delta_not_mislabeled_as_a_kv_win(physical):
    row = record()
    row['metal_active_released_bytes'] = physical
    row['after_reclaim']['metal_active_bytes'] = 100-physical
    row['after_reclaim']['deficit_bytes'] = 60-physical
    # The second ordinary reserve succeeded later; this non-atomic earlier
    # observation alone cannot attribute that recovery to the spilled pages.
    doc = response(trace([row]))
    result = log_coverage([doc], log(row), budget_bytes=BUDGET)
    assert result['passed'] and result['positive_active_release_attempts'] == 0
    assert result['metal_active_released_bytes'] == physical


@pytest.mark.parametrize('bad', ['missing_hidden', 'missing_field', 'missing_flag',
    'wrong_budget', 'bool_flag', 'stale_final', 'hidden_error'])
def test_public_success_cannot_hide_missing_or_failed_hidden_trace(bad):
    doc = response(trace(), {})
    hidden = doc['vmodel_cache_phases'][0]
    if bad == 'missing_hidden':
        doc['vmodel_cache_phases'][0] = None
    elif bad == 'missing_field':
        hidden.pop(KEY)
    elif bad == 'missing_flag':
        hidden.pop(KEY+'_enabled')
    elif bad == 'wrong_budget':
        hidden['paged_kv_budget_bytes'] -= 1
    elif bad == 'bool_flag':
        hidden[KEY+'_enabled'] = True
    elif bad == 'stale_final':
        doc['vmodel_timing'][KEY] = trace()
    else:
        hidden[KEY]['error'] = 1
    assert not all(phase_checks(doc, doc['vmodel_timing'], budget_bytes=BUDGET).values())
    assert not log_coverage([doc], log(record()), budget_bytes=BUDGET)['passed']


@pytest.mark.parametrize('raw', ['[qwen35-serial-kv-reclaim] broken',
    '[qwen35-serial-kv-reclaim]{}', '[qwen35-serial-kv-reclaim] []'])
def test_malformed_independent_log_is_failure(raw):
    assert not log_coverage([response({})], raw, budget_bytes=BUDGET)['passed']


def test_log_bool_cannot_equal_integer_by_python_coercion():
    row = {**record(), 'layer': 1}
    logged = {**row, 'layer': True}
    assert not log_coverage([response(trace([row]))], log(logged), budget_bytes=BUDGET)['passed']


def test_logged_event_cannot_disappear_into_empty_phase_metadata():
    assert not log_coverage([response({})], log(record()), budget_bytes=BUDGET)['passed']
    assert not log_coverage([], '', budget_bytes=BUDGET)['passed']


def test_opt_in_row_wiring_and_disabled_legacy_path():
    row = valid_row()
    doc = response({})
    row['timing'].update(doc['vmodel_timing'])
    config = dict(profiles=['test'], profile_digest='digest',
                  require_serial_kv_reclaim=True, serial_kv_budget_bytes=BUDGET)
    check = row_checks(row, doc, dict(kind='short_title', topic='node'), config)
    assert check['all_phase_serial_kv_reclaim_witness'] and check['final_serial_kv_reclaim_trace_matches']
    config.pop('require_serial_kv_reclaim')
    check = row_checks(row, {}, dict(kind='short_title', topic='node'), config)
    assert 'all_phase_serial_kv_reclaim_witness' not in check
