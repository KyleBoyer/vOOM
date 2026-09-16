from types import SimpleNamespace as NS
import ast
from pathlib import Path
import pytest
from runtime.qwen_mtp_draft_lifetime import configure, FLAG
from runtime.profiles import apply_runtime_profiles

class Native:
    request_weight_representation='mxfp4-q4-g32'

def target():
    return NS(cfg=NS(model_type='qwen3_5',num_experts=0),
              rc=NS(qwen35_mxfp4_head_rows=8192))

def test_default_off_and_explicit_reset():
    d=Native();configure(d,target(),Native,{})
    assert not d._head_release_enabled
    configure(d,target(),Native,{FLAG:'1'})
    assert d._head_release_enabled and d._head_release_stats['releases']==0
    d._head_release_stats['releases']=99
    configure(d,target(),Native,{FLAG:'0'})
    assert not d._head_release_enabled and d._head_release_stats['releases']==0

@pytest.mark.parametrize('bad',['auto','',True,1,None])
def test_strict_flag(bad):
    with pytest.raises(ValueError):configure(Native(),target(),Native,{FLAG:bad})

@pytest.mark.parametrize('bad',['custom','moe','model','head','representation'])
def test_fail_closed_unsupported_paths(bad):
    d=Native();t=target()
    if bad=='custom':d=NS()
    if bad=='moe':t.cfg.num_experts=8
    if bad=='model':t.cfg.model_type='qwen4_exp'
    if bad=='head':t.rc.qwen35_mxfp4_head_rows=0
    if bad=='representation':d.request_weight_representation='demand-cache'
    with pytest.raises(ValueError):configure(d,t,Native,{FLAG:'1'})

def test_only_lifetime_flag_and_real_early_validation():
    before={};after={};base=['huihui-qwen38-27b-harness-preview']
    apply_runtime_profiles(base,environ=before)
    apply_runtime_profiles(base+['qwen35-draft-head-release-audit'],environ=after)
    assert after=={**before,FLAG:'1'}
    tree=ast.parse(Path('runtime/qwen35_mtp.py').read_text())
    f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate')
    src=ast.unparse(f)
    assert src.index('configure_draft_lifetime(')<src.index('bootstrap =')
    assert "path_stats['qwen_mtp_draft_head_lifetime'] = dict(self.drafter._head_release_stats)" in src

def test_actual_protocol_preserves_structured_lifetime_witness():
    from runtime.server import _cache_phase_telemetry, _vision_protocol_timing
    key='qwen_mtp_draft_head_lifetime'
    stats=dict(enabled=1,releases=4,reloads=3,release_s=.01)
    result={'path_stats':{key:stats}}
    assert _vision_protocol_timing(result)[key]==stats
    assert _cache_phase_telemetry('gateway_decision',result)[key]==stats

@pytest.mark.parametrize('change,expected',[({},True),({'enabled':True},False),
    ({'releases':0},False),({'reloads':0},False),
    ({'observed_active_released_bytes':0},False),({'logical_released_bytes':0},False)])
def test_fixture_requires_applied_release_not_only_flag(change,expected):
    from tests.fixtures.captured_transition_tracking_gate import row_checks
    from tests.test_captured_transition_tracking_gate_pure import valid_row
    row=valid_row()
    row['timing']['qwen_mtp_draft_head_lifetime']=dict(enabled=1,releases=8,reloads=6,
        logical_released_bytes=8000,observed_active_released_bytes=7990)
    row['timing']['qwen_mtp_draft_head_lifetime'].update(change)
    checks=row_checks(row,{},dict(kind='short_title',topic='node'),
        dict(profiles=['test'],profile_digest='digest',require_draft_head_release=True))
    assert checks['draft_head_weights_released_and_reloaded'] is expected

@pytest.mark.parametrize('phases,expected',[(None,False),([],False),([{}],False),
    ([{'qwen_mtp_draft_head_lifetime':dict(enabled=1,releases=4,reloads=3,
        logical_released_bytes=100,observed_active_released_bytes=90)}],True)])
def test_full_workflow_requires_every_phase(phases,expected):
    tree=ast.parse(Path('tests/fixtures/huihui_captured_action_gate.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="config.get('require_draft_head_release') is True")
    scope=dict(config={'require_draft_head_release':True},checks={},
               response={'vmodel_cache_phases':phases})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                 '<actual full workflow acceptance>','exec'),scope)
    assert scope['checks']['all_phase_draft_head_release'] is expected
