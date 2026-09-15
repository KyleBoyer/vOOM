import pytest
from runtime.qwen_paged_persist_policy import CHECKPOINTS, MAX_MB, limits, request_identity
from runtime.profiles import apply_runtime_profiles


def test_legacy_defaults_unchanged_without_explicit_limits():
    assert limits({},default_checkpoints=64,default_max_mb=0)==(64,0)
    assert request_identity({})==('0',64,0)


def test_cache_mode_and_storage_limits_have_distinct_manager_identities():
    from pathlib import Path
    assert request_identity({})!=request_identity({'VMODEL_QWEN35_PAGED_KV_PERSIST':'1'})
    assert request_identity({})!=request_identity({MAX_MB:'4096'})
    source=(Path(__file__).resolve().parents[1]/'runtime/server.py').read_text()
    assert source.count('            qwen_paged_persist_identity,')==2


def test_named_profile_bounds_new_journal():
    env={};apply_runtime_profiles(['qwen35-paged-prefix-cache'],environ=env)
    assert limits(env,default_checkpoints=64,default_max_mb=0)==(8,4096)
    assert env['VMODEL_QWEN35_FUSED_BOUNDARY_SCAFFOLD_PREFILL']=='0'
    assert env['VMODEL_QWEN35_PAGED_KV_PERSIST']=='1'


@pytest.mark.parametrize('key,value',[(CHECKPOINTS,'0'),(CHECKPOINTS,'65'),
    (MAX_MB,'16385'),(MAX_MB,'-1'),(MAX_MB,'4096.0'),(MAX_MB,True),(MAX_MB,'４')])
def test_invalid_explicit_limits_fail_closed(key,value):
    with pytest.raises(ValueError):limits({key:value},default_checkpoints=64,default_max_mb=0)
