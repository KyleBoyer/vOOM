import pytest
from runtime.profiles import apply_runtime_profiles
from tests.fixtures.huihui_memory_policy import available_floor, validate


def test_profile_only_changes_authorized_reserve():
    before={};after={}
    apply_runtime_profiles(['huihui-qwen38-27b-harness-preview'],environ=before)
    apply_runtime_profiles(['huihui-qwen38-27b-harness-preview','qwen35-reserve4500-audit'],environ=after)
    assert after=={**before,'VMODEL_QWEN35_MIN_AVAILABLE_MB':'4500'}


@pytest.mark.parametrize('bad',[True,4_500_000_000.0,4_400_000_000,0,'4500000000'])
def test_unqualified_threshold_rejected(bad):
    with pytest.raises(ValueError):available_floor({'minimum_available_bytes':bad})


def test_historical_floor_unchanged():
    assert available_floor({})==5_300_000_000


@pytest.mark.parametrize('setting,threshold,observed,passed',[
    ('4500',5_500_000_000,5_500_000_000,True),
    ('4400',5_500_000_000,5_500_000_000,False),
    ('4500',5_400_000_000,5_500_000_000,False),
    ('4500',5_500_000_000,5_499_999_999,False)])
def test_new_policy_requires_matching_runtime_and_launch_proof(setting,threshold,observed,passed):
    config={'minimum_available_bytes':4_500_000_000,'profiles':['qwen35-reserve4500-audit']}
    env={'VMODEL_QWEN35_MIN_AVAILABLE_MB':setting}
    pre={'thresholds':{'min_stable_available_bytes':threshold},'pressure_window':{'minimum_available_bytes':observed}}
    if passed:assert validate(config,env,pre)==4_500_000_000
    else:
        with pytest.raises(AssertionError):validate(config,env,pre)
