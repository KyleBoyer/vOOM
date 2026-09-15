"""Actual protocol serializers and fail-closed applied-path acceptance."""
import pytest
from runtime.server import _cache_phase_telemetry, _vision_protocol_timing
from tests.fixtures.captured_transition_tracking_gate import row_checks
from tests.test_captured_transition_tracking_gate_pure import valid_row

KEY = 'qwen_mtp_kda_base_spill'


def witness():
    return dict(rounds=4, closed=4, bytes_written=600_000_000,
                logical_bytes_peak=150_000_000, snapshot_resident_bytes_peak=0,
                reloads=2, bytes_read=300_000_000, write_s=.2, reload_s=.1)


@pytest.mark.parametrize('stats', [{}, witness()])
def test_structured_protocol_metric(stats):
    result = dict(path_stats={KEY: stats})
    assert _vision_protocol_timing(result)[KEY] == stats
    assert _cache_phase_telemetry('gateway_decision', result)[KEY] == stats
    assert KEY not in _vision_protocol_timing({})


@pytest.mark.parametrize('change,passed', [({}, True), ({'closed': 3}, False),
    ({'rounds': 0}, False), ({'bytes_written': 0}, False),
    ({'snapshot_resident_bytes_peak': 1}, False), ({'logical_bytes_peak': 0}, False)])
def test_actual_path_required(change, passed):
    row = valid_row()
    row['timing'][KEY] = {**witness(), **change}
    checks = row_checks(row, {}, dict(kind='short_title', topic='node'),
        dict(profiles=['test'], profile_digest='digest', require_factor_base_disk=True))
    assert checks['factor_base_disk_used_and_closed'] is passed
