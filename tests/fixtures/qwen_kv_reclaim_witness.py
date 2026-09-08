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


def validate_config(config, env):
    """Selected recovery profiles must opt into all matching evidence gates."""
    for flag, setting in (
            ('require_serial_kv_reclaim', 'VMODEL_QWEN35_SERIAL_KV_RECLAIM'),
            ('require_serial_kv_reclaim_topup', 'VMODEL_QWEN35_SERIAL_KV_RECLAIM_TOPUP')):
        assert type(config.get(flag, False)) is bool
        assert env.get(setting, '0') in ('0', '1')
        assert config.get(flag, False) is (env.get(setting, '0') == '1')
    assert not config.get('require_serial_kv_reclaim_topup', False) or config.get('require_serial_kv_reclaim', False)


def _valid_snapshot(sample, incoming, margin):
    return (isinstance(sample, dict) and all(_integer(sample.get(k)) for k in (
        'metal_active_bytes', 'system_available_bytes', 'ceiling_bytes', 'deficit_bytes'))
        and sample['deficit_bytes'] == max(0,
            sample['metal_active_bytes'] + incoming + margin - sample['ceiling_bytes']))


def _valid_topup(row):
    """Reconcile both spill passes without equating physical and host deltas."""
    if row.get('topup_enabled') is not True:
        return False
    passes = row.get('reclaim_passes')
    if not isinstance(passes, list) or not 1 <= len(passes) <= 2 or 'topup_check' not in row:
        return False
    for part in passes:
        if not isinstance(part, dict) or not all(_integer(part.get(k)) for k in (
                'requested_bytes', 'logical_before_bytes', 'logical_after_bytes',
                'logical_reclaimed_bytes', 'spill_pages')):
            return False
        if (not all(_seconds(part.get(k)) for k in ('spill_seconds', 'reclaim_seconds'))
                or part['reclaim_seconds'] < part['spill_seconds'] - 1e-6
                or type(part.get('metal_active_released_bytes')) is not int):
            return False
        if not all(_valid_snapshot(part.get(k), row['incoming_bytes'], row['margin_bytes'])
                   for k in ('before', 'after_reclaim')):
            return False
        if (part['requested_bytes'] != part['before']['deficit_bytes']
                or part['logical_reclaimed_bytes'] != part['logical_before_bytes'] - part['logical_after_bytes']
                or (part['logical_reclaimed_bytes'] > 0) != (part['spill_pages'] > 0)
                or (part['requested_bytes'] == 0 and part['logical_reclaimed_bytes'] != 0)
                or part['metal_active_released_bytes'] != part['before']['metal_active_bytes'] - part['after_reclaim']['metal_active_bytes']):
            return False
    first, last = passes[0], passes[-1]
    if (first['before'] != row['before'] or last['after_reclaim'] != row['after_reclaim']
            or first['logical_before_bytes'] != row['logical_before_bytes']
            or last['logical_after_bytes'] != row['logical_after_bytes']):
        return False
    for key in ('logical_reclaimed_bytes', 'spill_pages', 'spill_seconds', 'reclaim_seconds'):
        total = sum(p[key] for p in passes)
        if key.endswith('seconds'):
            if not math.isclose(row[key], total, rel_tol=1e-12, abs_tol=1e-6):
                return False
        elif row[key] != total:
            return False
    eligible = (first['logical_reclaimed_bytes'] > 0
        and first['metal_active_released_bytes'] > 0 and first['after_reclaim']['deficit_bytes'] > 0)
    check = row['topup_check']
    if not eligible:
        return check is None and len(passes) == 1
    if not _valid_snapshot(check, row['incoming_bytes'], row['margin_bytes']):
        return False
    if not check['deficit_bytes']:
        return len(passes) == 1
    return (len(passes) == 2 and last['before'] == check
        and last['logical_before_bytes'] == first['logical_after_bytes'])


def _valid_record(row, budget):
    if not isinstance(row, dict):
        return False
    if (row.get('schema') not in ('voom.qwen35-serial-kv-reclaim.v1', 'voom.qwen35-serial-kv-reclaim.v2')
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
            - row['after_reclaim']['metal_active_bytes']
        and (_valid_topup(row) if row['schema'].endswith('.v2') else not any(
            k in row for k in ('topup_enabled', 'topup_check', 'reclaim_passes'))))


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


def phase_checks(response, timing, *, budget_bytes, topup_required=False):
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
            and valid_trace(value[key], budget_bytes=budget_bytes)
            and (not topup_required or (
                type(value.get(key + '_topup_enabled')) is int
                and value[key + '_topup_enabled'] == 1
                and all(r.get('schema') == 'voom.qwen35-serial-kv-reclaim.v2'
                    for r in value[key].get('records', [])))))
    complete = valid and all(witnessed(phase) for phase in phases)
    # The existing protocol carries KV budget on phases, not flat timing.
    # Keep every phase's budget mandatory; compare flat trace/flag separately.
    return dict(all_phase_serial_kv_reclaim_witness=bool(complete),
        final_serial_kv_reclaim_trace_matches=bool(complete and witnessed(timing, require_budget=False)
            and timing[key] == phases[-1][key]))


def log_coverage(responses, log_text, *, budget_bytes, topup_required=False):
    """Require exact chronological event identity across every hidden/public phase."""
    prefix = '[qwen35-serial-kv-reclaim] '
    try:
        if not responses:
            raise ValueError('no responses')
        expected = []
        for response in responses:
            timing = response.get('vmodel_timing', {})
            if not all(phase_checks(response, timing, budget_bytes=budget_bytes,
                    topup_required=topup_required).values()):
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
