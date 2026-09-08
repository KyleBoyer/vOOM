"""Pure, fail-closed checks for complete Qwen serial-KV recovery telemetry.

An explicitly present empty trace means no recovery fired. It is never a
physical-memory win. Records must also match the independently saved log.
"""

import json
import math


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _valid_record(row, budget):
    if not isinstance(row, dict):
        return False
    if (row.get('schema') != 'voom.qwen35-serial-kv-reclaim.v1'
            or row.get('outcome') != 'admitted'
            or row.get('reservation_retried') is not True or 'error_type' in row):
        return False
    integers = ('layer', 'start_offset', 'incoming_bytes', 'margin_bytes',
                'kv_budget_bytes', 'logical_before_bytes', 'logical_after_bytes',
                'logical_reclaimed_bytes', 'requested_bytes', 'spill_pages')
    if not all(_integer(row.get(k)) for k in integers):
        return False
    if (not _integer(row.get('verifier_positions'), 1)
            or row['kv_budget_bytes'] != budget
            or type(row.get('metal_active_released_bytes')) is not int):
        return False
    if not all(_seconds(row.get(k)) for k in ('reclaim_seconds', 'spill_seconds', 'wall_seconds')):
        return False
    if not row['wall_seconds'] >= row['reclaim_seconds'] >= row['spill_seconds'] - 1e-6:
        return False
    for name in ('before', 'after_reclaim'):
        sample = row.get(name)
        if not isinstance(sample, dict) or not all(_integer(sample.get(k)) for k in (
                'metal_active_bytes', 'system_available_bytes', 'ceiling_bytes', 'deficit_bytes')):
            return False
        if sample['deficit_bytes'] != max(0, sample['metal_active_bytes']
                + row['incoming_bytes'] + row['margin_bytes'] - sample['ceiling_bytes']):
            return False
    released = row['logical_reclaimed_bytes']
    return (row['requested_bytes'] == row['before']['deficit_bytes']
        and released == row['logical_before_bytes'] - row['logical_after_bytes']
        and (released > 0) == (row['spill_pages'] > 0)
        and (row['requested_bytes'] > 0 or released == 0)
        and row['metal_active_released_bytes'] == row['before']['metal_active_bytes']
            - row['after_reclaim']['metal_active_bytes'])


def valid_trace(value, *, budget_bytes):
    """Validate a successful phase, including explicit no-recovery metadata.

    Physical deltas may be zero/negative despite admission: a later governor
    sample can recover independently. Never relabel those as a KV memory win.
    Capped/missing/error traces cannot qualify a performance result.
    """
    if not isinstance(value, dict) or not _integer(budget_bytes, 1):
        return False
    if value == {}:
        return True
    records = value.get('records')
    if (not isinstance(records, list) or not 1 <= len(records) <= 64
            or not _integer(value.get('attempts'), 1)
            or not _integer(value.get('admitted'), 1)
            or value['attempts'] != len(records) or value['admitted'] != len(records)):
        return False
    for key in ('refused', 'error', 'no_candidates', 'records_dropped', 'log_errors'):
        if key in value and (not _integer(value[key]) or value[key] != 0):
            return False
    if not all(_valid_record(row, budget_bytes) for row in records):
        return False
    for key in ('logical_reclaimed_bytes', 'spill_pages', 'spill_seconds', 'wall_seconds'):
        check = _seconds if key.endswith('seconds') else _integer
        if not check(value.get(key)) or value[key] != sum(row[key] for row in records):
            return False
    return True


def phase_checks(response, timing, *, budget_bytes):
    phases = response.get('vmodel_cache_phases')
    valid = isinstance(phases, list) and bool(phases)
    key = 'qwen35_serial_kv_reclaim'
    enabled = key + '_enabled'
    def witnessed(value, *, require_budget=True):
        return (isinstance(value, dict) and type(value.get(enabled)) is int
            and value[enabled] == 1
            and (not require_budget or (
                type(value.get('paged_kv_budget_bytes')) is int
                and value['paged_kv_budget_bytes'] == budget_bytes)) and key in value
            and valid_trace(value[key], budget_bytes=budget_bytes))
    complete = valid and all(witnessed(phase) for phase in phases)
    # The existing protocol carries KV budget on phases, not flat timing.
    # Keep every phase's budget mandatory; compare flat trace/flag separately.
    return dict(all_phase_serial_kv_reclaim_witness=bool(complete),
        final_serial_kv_reclaim_trace_matches=bool(complete and witnessed(timing, require_budget=False)
            and timing[key] == phases[-1][key]))


def log_coverage(responses, log_text, *, budget_bytes):
    """Require exact chronological event identity across every hidden/public phase."""
    prefix = '[qwen35-serial-kv-reclaim] '
    try:
        if not responses:
            raise ValueError('no responses')
        expected = []
        for response in responses:
            timing = response.get('vmodel_timing', {})
            if not all(phase_checks(response, timing, budget_bytes=budget_bytes).values()):
                raise ValueError('incomplete phases')
            for phase in response['vmodel_cache_phases']:
                expected.extend(phase['qwen35_serial_kv_reclaim'].get('records', []))
        lines = [line for line in log_text.splitlines()
                 if line.startswith('[qwen35-serial-kv-reclaim]')]
        if any(not line.startswith(prefix) for line in lines):
            raise ValueError('malformed recovery log prefix')
        observed = [json.loads(line[len(prefix):]) for line in lines]
        passed = (all(_valid_record(row, budget_bytes) for row in observed)
                  and observed == expected)
        return dict(passed=passed, complete_phase_coverage=True,
            phase_attempts=len(expected), logged_attempts=len(observed),
            positive_active_release_attempts=sum(r['metal_active_released_bytes'] > 0 for r in expected),
            logical_reclaimed_bytes=sum(r['logical_reclaimed_bytes'] for r in expected),
            metal_active_released_bytes=sum(r['metal_active_released_bytes'] for r in expected),
            scope='Exact event coverage; zero attempts is inactive-path proof only; signed non-atomic deltas are not allocation credit.')
    except (ValueError, TypeError, KeyError, AttributeError):
        return dict(passed=False, complete_phase_coverage=False)
