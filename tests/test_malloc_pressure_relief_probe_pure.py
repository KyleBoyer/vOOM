"""Pure allocator-relief call contract/order tests; no native release or MLX."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent/'fixtures'/'malloc_pressure_relief_probe.py'
spec = importlib.util.spec_from_file_location('malloc_pressure_relief_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_sdk_oracle_embeds_escaped_json_and_checks_exact_signature():
    source = probe.abi_source()
    assert 'printf("{\\"size_t_bytes\\":%zu,\\"pointer_bytes\\":%zu}"' in source
    assert 'size_t (*)(malloc_zone_t *, size_t)' in source
    assert '_Static_assert(__builtin_types_compatible_p(' in source


@pytest.mark.parametrize('bad', [0, -1, True, False, 1.0, '256', None, 268435457])
def test_invalid_goals_rejected_before_native_call(bad):
    def forbidden(*args):
        raise AssertionError('native call not allowed')
    with pytest.raises(ValueError):
        probe.pressure_relief(bad, fn=forbidden)


@pytest.mark.parametrize('goal', [1, 67108864, 268435456])
def test_null_zone_exact_goal_and_best_effort_return(goal):
    calls = []
    def native(zone, size):
        calls.append((zone, size))
        return goal + 16384  # a goal is NOT an upper bound on release.
    receipt = probe.pressure_relief(goal, fn=native)
    assert calls == [(None, goal)]
    assert receipt['reported_released_bytes'] == goal + 16384
    assert not receipt['goal_is_hard_cap']
    assert not receipt['physical_footprint_delta_is_api_return']


def test_no_reclamation_is_not_fabricated_as_a_failure_or_successful_byte_count():
    assert probe.pressure_relief(1, fn=lambda *args: 0)['reported_released_bytes'] == 0


@pytest.mark.parametrize('bad', [None, True, -1, 3.0])
def test_invalid_native_byte_returns_fail(bad):
    with pytest.raises(RuntimeError):
        probe.pressure_relief(1, fn=lambda *args: bad)


def test_native_error_propagates_without_fake_receipt():
    def fail(*args):
        raise RuntimeError('allocator error')
    with pytest.raises(RuntimeError, match='allocator error'):
        probe.pressure_relief(1, fn=fail)


def test_idle_before_release_without_hidden_gc_clear_or_sync():
    calls = []
    def observe():
        calls.append('observe')
        return len(calls)
    def release(size):
        calls.append(('release', size))
        return {'reported_released_bytes': 64}
    stages, receipt = probe.idle_then_relieve(
        64, observe=observe, relieve=release, sleep=lambda t: calls.append(('idle', t)))
    assert calls == ['observe', ('idle', 1.0), 'observe', ('release', 64), 'observe']
    assert list(stages) == ['before_idle', 'after_idle', 'after_relief']
    assert receipt['reported_released_bytes'] == 64


def test_failed_call_does_not_emit_post_relief_snapshot():
    observations = []
    def fail(size):
        raise RuntimeError('allocator error')
    with pytest.raises(RuntimeError):
        probe.idle_then_relieve(64, observe=lambda: observations.append(1),
                               relieve=fail, sleep=lambda t: None)
    assert len(observations) == 2
