"""Pure bounded-lifetime fixture tests; no actual MLX import or allocations."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent / 'fixtures' / 'metal_resource_lifetime_probe.py'
spec = importlib.util.spec_from_file_location('metal_resource_lifetime_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize('mib', [16, 128, 256])
def test_shape_is_exact_bounded_16bit_payload(mib):
    rows, columns = probe.weight_shape(mib)
    assert rows * columns * 2 == mib * 1024 * 1024
    assert columns == 4096


@pytest.mark.parametrize('invalid', [True, False, None, '256', 256.0, 0, 15, 257])
def test_bad_size_rejected_before_mlx_import(invalid):
    with pytest.raises(ValueError):
        probe.weight_shape(invalid)


def test_drop_idle_controls_precede_same_stream_marker():
    events = []
    holder = [object()]
    def observe():
        assert holder == []
        events.append('observe')
        return {'ordinal': len(events)}
    stages, times = probe.drop_and_observe(
        holder, observe=observe,
        clear_cache=lambda: events.append('clear'),
        marker=lambda: events.append('marker'),
        sleep=lambda delay: events.append(delay), clock=lambda: 0.0)
    assert events == ['observe', 'clear', 'observe', .1, 'observe', 1.0,
                      'observe', 'marker', 'observe']
    assert len(stages) == len(times) == 5
    assert all(value == 0 for value in times.values())


def test_marker_errors_propagate_and_do_not_fabricate_final_snapshot():
    observations = []
    def fail():
        raise RuntimeError('device failure')
    with pytest.raises(RuntimeError, match='device failure'):
        probe.drop_and_observe([object()], observe=lambda: observations.append(1),
                              clear_cache=lambda: None, marker=fail,
                              sleep=lambda delay: None)
    assert len(observations) == 4


@pytest.mark.parametrize('holder', [[], [1, 2]])
def test_only_one_explicit_fixture_owner_can_be_dropped(holder):
    with pytest.raises(ValueError):
        probe.drop_and_observe(holder, observe=None, clear_cache=None, marker=None)


def sample(**changes):
    return dict({'process': {'available': True}, 'regions': {'coverage_complete': True},
                 'system_available_bytes': 6_000_000_000,
                 'system_swap_used_bytes': 1_000_000_000,
                 'system_swap_out_bytes': 10_000_000_000,
                 'metal_peak_bytes': 300_000_000, 'root_free_bytes': 15_000_000_000},
                **changes)


@pytest.mark.parametrize('key,value,reason', [
    ('system_available_bytes', 5_299_999_999, 'available memory'),
    ('system_swap_used_bytes', 1_016_000_001, 'swap-used'),
    ('system_swap_out_bytes', 10_016_000_001, 'swap-out'),
    ('metal_peak_bytes', 8_500_000_001, 'Metal peak'),
    ('root_free_bytes', 9_999_999_999, 'root free'),
    ('process', {'available': False}, 'native observation'),
    ('regions', {'coverage_complete': False}, 'native observation'),
])
def test_pressure_failures_are_not_waived(key, value, reason):
    assert any(reason in error for error in
               probe.pressure_failures([sample(), sample(**{key: value})]))


def test_growth_not_absolute_stale_swap_and_exact_thresholds_pass():
    assert not probe.pressure_failures([sample(), sample(
        system_available_bytes=5_300_000_000, system_swap_used_bytes=1_016_000_000,
        system_swap_out_bytes=10_016_000_000, metal_peak_bytes=8_500_000_000,
        root_free_bytes=10_000_000_000)])
    assert probe.pressure_failures([])
