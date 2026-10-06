import copy

import pytest

from runtime.huihui_serve import PROFILE, REQUIRED_SETTINGS, validate_preflight, validate_profile
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.huihui_memory_policy import validate


def test_low_memory_profile_removes_draft_not_live_reserve_or_context():
    env = {}
    apply_runtime_profiles([PROFILE], environ=env)
    validate_profile(env)
    assert env['VMODEL_HTTP_MAX_CONTEXT_TOKENS'] == '16384'
    pre = dict(thresholds=dict(min_stable_available_bytes=5_000_000_000),
               pressure_window=dict(minimum_available_bytes=5_000_000_000))
    config = dict(profiles=[PROFILE], minimum_available_bytes=4_500_000_000)
    assert validate(config, env, pre) == 4_500_000_000
    pre['pressure_window']['minimum_available_bytes'] -= 1
    with pytest.raises(AssertionError):
        validate(config, env, pre)


@pytest.mark.parametrize('key', list(REQUIRED_SETTINGS))
def test_low_memory_profile_is_fail_closed_on_override(key):
    env = dict(REQUIRED_SETTINGS)
    env[key] = 'unqualified'
    with pytest.raises(ValueError):
        validate_profile(env)


def _preflight():
    return dict(passed=True, sample_seconds=30, end=dict(monotonic_s=100),
                pressure_window=dict(complete=True, minimum_available_bytes=5_000_000_000,
                                     minimum_root_free_bytes=10_000_000_000),
                swap_growth_bytes=0, swap_out_growth_bytes=0,
                known_transcoders=dict(passed=True))


def test_valid_fresh_preflight():
    validate_preflight(_preflight(), now=101)


@pytest.mark.parametrize('field,value', [('passed', False), ('sample_seconds', 29),
    ('swap_growth_bytes', 16_000_001), ('swap_out_growth_bytes', 16_000_001)])
def test_invalid_preflight_rejected(field, value):
    pre = _preflight()
    pre[field] = value
    with pytest.raises(ValueError):
        validate_preflight(pre, now=101)


def test_clean_swap_cannot_bypass_available_or_disk_floor():
    for field, value in [('minimum_available_bytes', 4_999_999_999),
                         ('minimum_root_free_bytes', 9_999_999_999), ('complete', False)]:
        pre = copy.deepcopy(_preflight())
        pre['pressure_window'][field] = value
        with pytest.raises(ValueError):
            validate_preflight(pre, now=101)
    with pytest.raises(ValueError):
        validate_preflight(_preflight(), now=220)
    with pytest.raises(ValueError):
        validate_preflight(_preflight(), now=99)
    pre = _preflight()
    pre['known_transcoders']['passed'] = False
    with pytest.raises(ValueError):
        validate_preflight(pre, now=101)
