"""Pure read-only inventory, preflight veto and post-response observer tests."""

import json
import sys
from types import SimpleNamespace

import pytest

from runtime import host_activity_witness as host
from runtime import memory_preflight as preflight
from runtime import process_memory_witness as memory
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.captured_transition_tracking_gate import native_pressure_summary


def process(name='ffmpeg', pid=17, created=123.5):
    return SimpleNamespace(info=dict(name=name, pid=pid, create_time=created,
        cmdline='PRIVATE', environ='PRIVATE', open_files='PRIVATE'))


def scan(*processes):
    def iterator(*, attrs):
        assert attrs == ['pid', 'name', 'create_time']
        return iter(processes)
    return host.sample_known_transcoders(process_iter=iterator)


def test_inventory_only_emits_allowed_identity_scalars():
    result = scan(process(), process('HandBrakeCLI', pid=18), process('Tdarr_Node'), process('PRIVATE'))
    assert result['available'] and result['scanned'] == 4
    assert result['transcoders'] == [dict(pid=17, created=123.5, kind='ffmpeg'),
                                     dict(pid=18, created=123.5, kind='HandBrakeCLI')]
    assert 'PRIVATE' not in json.dumps(result)
    assert result['observation_seconds'] >= 0


@pytest.mark.parametrize('name', ['ffmpeg-worker', '/private/ffmpeg', 'ffmpeg PRIVATE', 'Plex Media Server'])
def test_allowlist_not_arbitrary_process_content(name):
    assert scan(process(name))['transcoders'] == []


@pytest.mark.parametrize('item', [process(name=None), process(pid=True), process(pid=0),
    process(created=None), process(created=float('nan')), process(created=-1)])
def test_unidentified_or_invalid_identity_cannot_certify_quiet(item):
    row = scan(item)
    assert not row['available']
    assert not host.summarize_known_transcoders([row])['passed']


def test_inventory_error_is_redacted_and_unavailable():
    def fail(**kwargs):
        raise OSError('PRIVATE')
    row = host.sample_known_transcoders(process_iter=fail)
    assert not row['available'] and row['reason'] == 'inventory-error'
    assert 'PRIVATE' not in json.dumps(row)


def test_scan_and_match_caps_are_not_quiet(monkeypatch):
    monkeypatch.setattr(host, 'MAX_PROCESSES', 2)
    assert scan(process('other'), process('other'), process())['reason'] == 'process-limit'
    monkeypatch.setattr(host, 'MAX_MATCHES', 1)
    row = scan(process(), process(pid=18))
    assert row['reason'] == 'match-limit' and len(row['transcoders']) == 1


def test_transient_and_pid_reuse_are_retained_across_quiet_last_sample():
    rows = [scan(process()), scan(process(created=999)), scan()]
    result = host.summarize_known_transcoders(rows)
    assert result['available'] and not result['passed']
    assert [r['created'] for r in result['transcoders']] == [123.5, 999]


def test_empty_unknown_or_capped_history_cannot_certify_quiet(monkeypatch):
    assert not host.summarize_known_transcoders([])['passed']
    assert not host.summarize_known_transcoders([{'available': False}])['passed']
    monkeypatch.setattr(host, 'MAX_WINDOW_SAMPLES', 1)
    assert not host.summarize_known_transcoders([scan(), scan()])['passed']


@pytest.mark.parametrize('sample', [None, {'available': True, 'transcoders': None},
    {'available': True, 'transcoders': [None]},
    {'available': True, 'transcoders': [{'pid': 1, 'created': float('nan'), 'kind': 'ffmpeg'}]},
    {'available': True, 'transcoders': [{'pid': 1, 'created': 2, 'kind': 'PRIVATE'}]}])
def test_malformed_aggregate_is_redacted_and_unavailable(sample):
    result = host.summarize_known_transcoders([sample])
    assert not result['available'] and not result['passed']
    assert 'PRIVATE' not in json.dumps(result)


def test_aggregate_does_not_forward_extra_process_fields():
    row = scan(process())
    row['transcoders'][0]['secret'] = 'PRIVATE'
    assert 'PRIVATE' not in json.dumps(host.summarize_known_transcoders([row]))


@pytest.mark.parametrize('seconds', [True, -1, float('nan'), float('inf'), 3601])
def test_window_is_finite_bounded_before_observation(seconds):
    with pytest.raises(ValueError):
        host.sample_transcoder_window(seconds, sample=lambda: pytest.fail('must not scan'))


def test_window_polls_both_endpoints_and_keeps_midwindow_activity():
    now = [0.0]
    sleeps = []
    samples = iter([scan(), scan(process()), scan()])
    def sleep(seconds):
        sleeps.append(seconds); now[0] += seconds
    result = host.sample_transcoder_window(3, sample=lambda: next(samples),
        clock=lambda: now[0], sleep=sleep)
    assert sleeps == [2, 1] and result['samples'] == 3
    assert not result['passed'] and len(result['transcoders']) == 1


def test_stalled_clock_is_bounded_and_unavailable(monkeypatch):
    monkeypatch.setattr(host, 'MAX_WINDOW_SAMPLES', 3)
    result = host.sample_transcoder_window(1, sample=scan, clock=lambda: 0, sleep=lambda _: None)
    assert not result['passed'] and result['samples'] == 3


@pytest.mark.parametrize('enabled', [False, True])
def test_preflight_opt_in_veto_does_not_change_memory_thresholds(monkeypatch, tmp_path, enabled):
    result = tmp_path/'preflight.json'
    args = ['memory_preflight', '--result', str(result), '--sample-seconds', '0', '--min-root-free-gb', '10']
    if enabled:
        args.append('--require-no-transcoders')
    monkeypatch.setattr(sys, 'argv', args)
    snapshot = preflight.PressureSnapshot(1.0, 8_000_000_000, 2_000_000_000,
        0, 2_000_000_000, 0, 0, 20_000_000_000, 100_000_000_000)
    monkeypatch.setattr(preflight, 'capture', lambda _: snapshot)
    def window(_):
        assert enabled
        return host.summarize_known_transcoders([scan(process())])
    monkeypatch.setattr(host, 'sample_transcoder_window', window)
    assert preflight.main() == int(enabled)
    output = json.loads(result.read_text())
    assert output['clean_swap'] and output['root_ok']
    assert output['thresholds']['max_swap_out_growth_bytes'] == 16_000_000
    assert output['passed'] is (not enabled)
    assert ('known_transcoders' in output) is enabled
    if enabled:
        assert output['admission_path'] == 'none'
        assert output['reasons'] == ['known_transcoders_active']


def test_overlay_only_adds_activity_and_required_memory_observer():
    env = {}
    apply_runtime_profiles(['host-activity-witness'], environ=env)
    assert env == {'VMODEL_PROCESS_MEMORY_WITNESS': '1', 'VMODEL_HOST_ACTIVITY_WITNESS': '1'}


@pytest.mark.parametrize('enabled', [False, True])
def test_periodic_scan_is_opt_in_and_after_native_observation(monkeypatch, capsys, enabled):
    actions = []
    monkeypatch.setenv(host.FLAG, '1' if enabled else '0')
    monkeypatch.setattr(memory, 'sample_self_memory', lambda: actions.append('native') or {'available': True})
    def activity():
        assert actions == ['native']
        actions.append('activity')
        return scan(process())
    monkeypatch.setattr(host, 'sample_known_transcoders', activity)
    observer = memory.GovernorProcessMemoryObserver()
    observer.record(governor_monotonic_s=1, system_available_bytes=6_000_000_000,
        system_swap_used_bytes=0, system_swap_out_bytes=0, metal_active_bytes=100,
        cache_budget_bytes_after_response=100, swap_pressure_response=False)
    row = json.loads(capsys.readouterr().out.removeprefix('[process-memory] '))
    assert ('known_transcoders' in row) is enabled
    assert actions == (['native', 'activity'] if enabled else ['native'])


def test_activity_failure_preserves_native_record(monkeypatch, capsys):
    monkeypatch.setenv(host.FLAG, '1')
    monkeypatch.setattr(memory, 'sample_self_memory', lambda: {'available': True})
    def fail():
        raise RuntimeError('PRIVATE')
    monkeypatch.setattr(host, 'sample_known_transcoders', fail)
    observer = memory.GovernorProcessMemoryObserver()
    observer.record(governor_monotonic_s=1, system_available_bytes=6_000_000_000,
        system_swap_used_bytes=0, system_swap_out_bytes=0, metal_active_bytes=100,
        cache_budget_bytes_after_response=100, swap_pressure_response=False)
    raw = capsys.readouterr().out
    row = json.loads(raw.removeprefix('[process-memory] '))
    assert row['process']['available'] and not row['known_transcoders']['available']
    assert 'PRIVATE' not in raw


def test_late_activity_invalidates_isolation_without_faking_memory_failure():
    def row(activity):
        return '[process-memory] '+json.dumps(dict(system_available_bytes=6_000_000_000,
            system_swap_used_bytes=0, system_swap_out_bytes=0,
            process=dict(available=True, physical_footprint_bytes=100, internal_compressed_ledger_bytes=0),
            known_transcoders=activity))
    result = native_pressure_summary(row(scan())+'\n'+row(scan(process()))+'\n'+row(scan()))
    assert result['passed'] and result['known_transcoders']['available']
    assert not result['known_transcoders']['passed']
