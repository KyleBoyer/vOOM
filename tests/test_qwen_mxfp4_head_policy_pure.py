"""Native head serving guards and honest phase accounting, without MLX."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from runtime import qwen_mxfp4_head_policy as policy
from runtime.profiles import apply_runtime_profiles
from runtime.server import _cache_phase_telemetry, _vision_protocol_timing

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rc = NS(qwen35_mxfp4_head_rows=32768, governor=True,
        pin_lm_head=False, rerank_lm_head=False, quant_lm_head=False,
        qwen35_serial_verify_suspend_lm_head=False, qwen35_phase_head_pre_admit=False,
        qwen4_phase_lm_head=False, glm53_phase_lm_head=False, grammar_jump_forward_lossy=False)
    cfg = NS(model_type='qwen3_5', num_experts=0, tie_word_embeddings=False,
             hidden_size=5120, vocab_size=248320)
    store = NS(vpack2=None, packed=None, gguf=None, fast_dirs=[],
        _raw_fast_tier_manifest=None, k3_scale_sidecar=None, bf16_nf12_sidecar=None,
        _ct_int4_aux={}, _ct_mxfp4_aux={}, _glm53_fp8_aux={}, _dsv4_aux={},
        _qwen4_fused_expert_slices={}, _ensure_raw_fast_tier_loaded=lambda:None,
        _quant_aux={'lm_head.weight':NS(bits=4,group_size=32,mode='mxfp4',
            scales='lm_head.scales',biases=None)})
    return rc,cfg,store


@pytest.mark.parametrize('rows', policy.ROWS)
def test_rows_and_identity_keep_disabled_namespace_unchanged(rows):
    assert policy.parse_rows(rows)==policy.parse_rows(str(rows))==rows
    assert bool(policy.identity(rows))==bool(rows)
    rc,cfg,store=fixture(); rc.qwen35_mxfp4_head_rows=rows
    policy.validate(rc,cfg,store)


@pytest.mark.parametrize('value',[True,False,None,-1,16384,32768.0,'auto','032768',' 32768'])
def test_bad_rows_fail(value):
    with pytest.raises(ValueError): policy.parse_rows(value)


@pytest.mark.parametrize('flag',['pin_lm_head','rerank_lm_head','quant_lm_head',
    'qwen35_serial_verify_suspend_lm_head','qwen35_phase_head_pre_admit',
    'qwen4_phase_lm_head','glm53_phase_lm_head','grammar_jump_forward_lossy'])
def test_incompatible_modes_fail(flag):
    rc,cfg,store=fixture(); setattr(rc,flag,True)
    with pytest.raises(ValueError,match=flag): policy.validate(rc,cfg,store)


@pytest.mark.parametrize('flag',['vpack2','packed','gguf','fast_dirs',
    '_raw_fast_tier_manifest','k3_scale_sidecar','bf16_nf12_sidecar',
    '_ct_int4_aux','_ct_mxfp4_aux','_glm53_fp8_aux','_dsv4_aux','_qwen4_fused_expert_slices'])
def test_source_overlay_cannot_be_silently_bypassed(flag):
    rc,cfg,store=fixture(); setattr(store,flag,object())
    with pytest.raises(ValueError,match='overlay'): policy.validate(rc,cfg,store)


def test_lazy_overlay_is_resolved_and_missing_metadata_fails_closed():
    rc,cfg,store=fixture()
    store._ensure_raw_fast_tier_loaded=lambda:setattr(store,'_raw_fast_tier_manifest',{'path':'overlay'})
    with pytest.raises(ValueError,match='overlay'): policy.validate(rc,cfg,store)
    rc,cfg,store=fixture(); del store._ct_int4_aux
    with pytest.raises(ValueError,match='overlay'): policy.validate(rc,cfg,store)
    rc,cfg,store=fixture(); store._quant_aux={}
    with pytest.raises(ValueError,match='packed head pair'): policy.validate(rc,cfg,store)


@pytest.mark.parametrize('key,value',[('model_type','qwen3_5_moe'),('num_experts',1),
    ('tie_word_embeddings',True),('hidden_size',4096),('vocab_size',100)])
def test_geometry_or_architecture_fails(key,value):
    rc,cfg,store=fixture(); setattr(cfg,key,value)
    with pytest.raises(ValueError,match='Huihui'): policy.validate(rc,cfg,store)


def test_no_governor_fails_and_disabled_policy_does_not_touch_source():
    rc,cfg,store=fixture(); rc.governor=False
    with pytest.raises(ValueError,match='governor'): policy.validate(rc,cfg,store)
    policy.validate(NS(qwen35_mxfp4_head_rows=0),None,None)


@pytest.mark.parametrize('serial,width,passes',[(False,1,True),(False,2,False),
    (True,1,True),(True,64,True),(True,65,False),(True,0,False)])
def test_forward_window_guards(serial,width,passes):
    if passes: policy.validate_forward(32768,width,serial=serial)
    else:
        with pytest.raises(ValueError): policy.validate_forward(32768,width,serial=serial)
    policy.validate_forward(0,width,serial=serial)


def counts(n):
    return {k:n for k in policy.COUNTERS}


def test_request_accounting_keeps_head_bytes_and_target_draft_phases():
    rc,_,_=fixture(); e=NS(rc=rc,_streamed_lm_head=NS(full_scan_telemetry=lambda:counts(15)))
    stats={'weight_store_bytes_read':100}; draft={}
    policy.accumulate(draft,counts(4),counts(6)); policy.accumulate(draft,counts(9),counts(12))
    policy.publish(e,stats,counts(2),counts(4),draft=draft)
    v=stats['qwen35_mxfp4_head_io']
    assert v['total']==counts(13) and v['prefill']==counts(2) and v['decode']==counts(11)
    assert v['draft']==counts(5) and v['target']==counts(8)
    assert v['weight_store_plus_head_bytes']==113
    for result in (_cache_phase_telemetry('generation',{'path_stats':stats}),
                   _vision_protocol_timing({'path_stats':stats})):
        assert result['qwen35_mxfp4_head_io']==v
    assert 'qwen35_mxfp4_head_io' not in _vision_protocol_timing({})


def test_missing_or_regressed_telemetry_fails_and_disabled_remains_missing():
    for before,after in ((None,counts(1)),(counts(1),None),(counts(2),counts(1))):
        with pytest.raises(ValueError): policy.delta(before,after)
    e=NS(rc=NS(qwen35_mxfp4_head_rows=0)); stats={}
    assert policy.snapshot(e) is None
    policy.publish(e,stats,None,None)
    assert stats=={}


def test_opt_in_profiles_and_two_engine_keys_and_prompt_identity():
    env={}; apply_runtime_profiles(['huihui-qwen38-27b-direct-head-rows-audit'],environ=env)
    assert env['VMODEL_QWEN35_MXFP4_HEAD_ROWS']=='32768'
    assert env['VMODEL_QWEN35_PIN_LM_HEAD']==env['VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD']=='0'
    assert env['VMODEL_QWEN35_MIN_AVAILABLE_MB']=='5600'
    assert env['VMODEL_FAST_TOOL_GATEWAY']=='0'
    source=(ROOT/'runtime/server.py').read_text(); tree=ast.parse(source)
    keys=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets)
        and isinstance(n.value,ast.Tuple) and any(isinstance(x,ast.Name)
            and x.id=='qwen35_suspend_lm_head_request' for x in n.value.elts)]
    assert len(keys)==2
    assert all(sum(isinstance(x,ast.Name) and x.id=='qwen35_mxfp4_head_rows'
        for x in n.value.elts)==1 for n in keys)
    engine=(ROOT/'runtime/engine.py').read_text()
    assert 'mxfp4_head_policy.identity(self.rc.qwen35_mxfp4_head_rows)' in engine
    assert 'qwen35_mxfp4_head_rows: int = 0' in engine
    assert engine.index('mxfp4_head_policy.validate(self.rc')<engine.index('persistent = self.cache.pin')
    tree=ast.parse(engine)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamingEngine')
    for name in ('forward_tokens','forward_tokens_serial_positions'):
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.unparse(method.body[1]).startswith('mxfp4_head_policy.validate_forward(')
