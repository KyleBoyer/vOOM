"""Pure gates for stable-stale-swap admission; no MLX import."""

from __future__ import annotations

from dataclasses import replace
import json
import sys

import pytest

from runtime import memory_preflight as preflight
from runtime.memory_preflight import PressureSnapshot, evaluate, sample_pressure_window


GB = int(1e9)
MB = int(1e6)


def snapshot(*, available=8 * GB, swap_used=300 * MB,
             swap_free=1700 * MB, swap_out=10 * GB,
             root_free=100 * GB):
    return PressureSnapshot(
        monotonic_s=0.0,
        system_available_bytes=available,
        swap_total_bytes=2 * GB,
        swap_used_bytes=swap_used,
        swap_free_bytes=swap_free,
        swap_in_bytes=0,
        swap_out_bytes=swap_out,
        root_free_bytes=root_free,
        workspace_free_bytes=400 * GB,
    )


def decide(start, end, intermediate=()):
    return evaluate(
        start,
        end,
        min_clean_swap_free_bytes=2 * GB,
        min_stable_available_bytes=6 * GB,
        min_root_free_bytes=5 * GB,
        max_swap_growth_bytes=16 * MB,
        max_swap_out_growth_bytes=16 * MB,
        intermediate=intermediate,
    )


def test_clean_swap_passes_without_stable_swap_exception():
    start = snapshot(available=4 * GB, swap_used=0, swap_free=2 * GB)
    result = decide(start, start)
    assert result["passed"]
    assert result["admission_path"] == "clean_swap"


def test_stale_swap_passes_when_available_is_high_and_counters_are_stable():
    start = snapshot()
    end = snapshot(swap_used=start.swap_used_bytes,
                   swap_out=start.swap_out_bytes)
    result = decide(start, end)
    assert result["passed"]
    assert result["admission_path"] == "stable_stale_swap"


def test_stale_swap_fails_when_available_memory_is_low():
    result = decide(snapshot(available=5 * GB), snapshot(available=5 * GB))
    assert not result["passed"]
    assert "system_available_below_stable_swap_minimum" in result["reasons"]


def test_stale_swap_fails_when_swap_usage_grows():
    start = snapshot()
    end = snapshot(swap_used=start.swap_used_bytes + 17 * MB)
    result = decide(start, end)
    assert not result["passed"]
    assert "swap_usage_growing" in result["reasons"]


def test_small_swap_out_churn_is_not_misclassified_as_net_deterioration():
    start = snapshot()
    end = snapshot(swap_out=start.swap_out_bytes + 6 * MB)
    result = decide(start, end)
    assert result["passed"]


def test_stale_swap_fails_when_swap_out_churn_exceeds_bound():
    start = snapshot()
    end = snapshot(swap_out=start.swap_out_bytes + 17 * MB)
    result = decide(start, end)
    assert not result["passed"]
    assert "swap_outs_active" in result["reasons"]


def test_root_floor_remains_mandatory_on_both_paths():
    start = snapshot(root_free=4 * GB)
    result = decide(start, start)
    assert not result["passed"]
    assert "root_free_below_minimum" in result["reasons"]


@pytest.mark.parametrize('change,reason', [
    ({'available': 5 * GB}, 'system_available_below_stable_swap_minimum'),
    ({'swap_used': 317 * MB}, 'swap_usage_growing'),
    ({'swap_out': 10 * GB + 17 * MB}, 'swap_outs_active'),
    ({'root_free': 4 * GB}, 'root_free_below_minimum'),
])
def test_interior_pressure_cannot_be_erased_by_healthy_endpoints(change, reason):
    start = snapshot()
    assert decide(start, start)['passed']  # Previous endpoint-only evidence.
    result = decide(start, start, (snapshot(**change),))
    assert not result['passed'] and reason in result['reasons']


def test_clean_and_stale_paths_cannot_be_mixed_to_hide_interior_pressure():
    start = snapshot(available=4 * GB, swap_used=0, swap_free=2 * GB)
    assert decide(start, start)['passed']
    result = decide(start, start, (snapshot(),))
    assert not result['passed'] and not result['clean_swap']
    assert not result['stable_stale_swap']


def test_window_preserves_clean_swap_rule_and_stale_swap_bounds():
    clean = snapshot(available=4 * GB, swap_used=0, swap_free=2 * GB)
    assert decide(clean, clean, (clean,))['admission_path'] == 'clean_swap'
    start = snapshot()
    middle = snapshot(available=6 * GB, swap_used=316 * MB,
                      swap_out=10 * GB + 16 * MB)
    result = decide(start, start, (middle,))
    assert result['admission_path'] == 'stable_stale_swap'
    assert result['swap_growth_bytes'] == result['swap_out_growth_bytes'] == 16 * MB


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def read(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def sample(self):
        return replace(snapshot(), monotonic_s=self.now)


@pytest.mark.parametrize('seconds,count', [(0, 1), (3, 3), (30, 16)])
def test_pressure_window_samples_bounded_cadence_with_final_partial_interval(seconds, count):
    clock = Clock()
    activity = []
    def scan():
        activity.append(clock.now)
        return {'available': True, 'transcoders': []}
    result = sample_pressure_window(seconds, sample=clock.sample, activity=scan,
        clock=clock.read, sleep=clock.sleep)
    assert result['complete'] and result['reason'] is None
    assert len(result['snapshots']) == len(result['activity']) == count
    assert [p.monotonic_s for p in result['snapshots']] == activity
    assert sum(clock.sleeps) == seconds and all(0 < t <= 2 for t in clock.sleeps)


@pytest.mark.parametrize('seconds', [-1, 3601, float('nan'), float('inf'), True, '30'])
def test_invalid_window_never_samples(seconds):
    with pytest.raises(ValueError):
        sample_pressure_window(seconds, sample=lambda: pytest.fail('must not sample'))


def test_nonadvancing_clock_hits_cap_and_cannot_certify_pressure_coverage():
    clock = Clock()
    result = sample_pressure_window(1, sample=clock.sample,
        clock=clock.read, sleep=lambda _: None)
    assert not result['complete'] and result['reason'] == 'pressure-sample-limit'
    assert len(result['snapshots']) == 1802


@pytest.mark.parametrize('point', [None, replace(snapshot(), monotonic_s=float('nan')),
    replace(snapshot(), monotonic_s=-1), replace(snapshot(), monotonic_s=True),
    replace(snapshot(), system_available_bytes=-1),
    replace(snapshot(), swap_used_bytes=True)])
def test_invalid_pressure_samples_fail_closed(point):
    result = sample_pressure_window(1, sample=lambda: point, clock=lambda: 0,
        sleep=lambda _: pytest.fail('must not sleep'))
    assert not result['complete'] and result['reason'] == 'invalid-pressure-sample'


def test_capture_error_retains_prior_evidence_without_exception_message():
    clock = Clock()
    def sample():
        if clock.now:
            raise OSError('PRIVATE')
        return clock.sample()
    result = sample_pressure_window(3, sample=sample, clock=clock.read, sleep=clock.sleep)
    assert not result['complete'] and len(result['snapshots']) == 1
    assert result['reason'] == 'pressure-sampling-error' and result['error_type'] == 'OSError'
    assert 'PRIVATE' not in repr(result)


def test_backward_sampler_clock_cannot_certify_coverage():
    ticks = iter([0.0, -1.0])
    result = sample_pressure_window(1, sample=snapshot, clock=lambda: next(ticks))
    assert not result['complete'] and result['reason'] == 'invalid-pressure-clock'


def test_activity_error_also_prevents_pressure_window_completion():
    clock = Clock()
    def fail():
        raise OSError('PRIVATE')
    result = sample_pressure_window(0, sample=clock.sample, activity=fail,
        clock=clock.read, sleep=clock.sleep)
    assert not result['complete'] and len(result['snapshots']) == 1
    assert result['reason'] == 'pressure-sampling-error' and result['activity'] == []


@pytest.mark.parametrize('seconds', ['nan', 'inf', '-1', '3601'])
def test_cli_rejects_invalid_memory_window_before_any_capture(monkeypatch, tmp_path, seconds):
    monkeypatch.setattr(sys, 'argv', ['memory_preflight', '--result', str(tmp_path/'none.json'),
        '--sample-memory-window', '--sample-seconds', seconds])
    monkeypatch.setattr(preflight, 'capture', lambda _: pytest.fail('must not capture'))
    with pytest.raises(SystemExit) as error:
        preflight.main()
    assert error.value.code == 2


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_cli_opt_in_preserves_endpoints_but_vetoes_interior_pressure(
        monkeypatch, tmp_path, enabled):
    path = tmp_path/'result.json'
    argv = ['memory_preflight', '--result', str(path), '--sample-seconds', '0',
            '--min-root-free-gb', '10']
    if enabled:
        argv.append('--sample-memory-window')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(preflight, 'capture', lambda _: snapshot())
    def window(*args, **kwargs):
        assert enabled
        return dict(complete=True, reason=None, snapshots=[snapshot(available=5 * GB)], activity=[])
    monkeypatch.setattr(preflight, 'sample_pressure_window', window)
    assert preflight.main() == int(enabled)
    result = json.loads(path.read_text())
    assert result['start']['system_available_bytes'] == result['end']['system_available_bytes'] == 8 * GB
    assert ('pressure_window' in result) is enabled
    assert result['thresholds']['min_root_free_bytes'] == 10 * GB
    if enabled:
        assert result['pressure_window']['sample_count'] == 3
        assert result['pressure_window']['minimum_available_bytes'] == 5 * GB
        assert not result['passed']


@pytest.mark.parametrize('complete,transcoder', [(False, False), (True, True), (True, False)])
def test_actual_cli_requires_both_window_coverage_and_transcoder_clearance(
        monkeypatch, tmp_path, complete, transcoder):
    path = tmp_path/'result.json'
    monkeypatch.setattr(sys, 'argv', ['memory_preflight', '--result', str(path),
        '--sample-seconds', '0', '--sample-memory-window', '--require-no-transcoders'])
    monkeypatch.setattr(preflight, 'capture', lambda _: snapshot())
    def window(*args, **kwargs):
        assert callable(kwargs['activity'])
        return dict(complete=complete, reason=None if complete else 'pressure-sample-limit',
            snapshots=[snapshot()], activity=[{'available': True, 'transcoders':
                [dict(pid=17, created=123.5, kind='Plex Transcoder')] if transcoder else []}])
    monkeypatch.setattr(preflight, 'sample_pressure_window', window)
    passed = complete and not transcoder
    assert preflight.main() == int(not passed)
    result = json.loads(path.read_text())
    assert result['passed'] is passed
    assert result['known_transcoders']['passed'] is (not transcoder)
    assert result['pressure_window']['complete'] is complete
    if not complete:
        assert 'pressure_window_unavailable' in result['reasons']
