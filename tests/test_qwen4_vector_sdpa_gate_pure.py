"""Native-vector candidate remains fixture-only, bounded and bit-gated."""
from types import SimpleNamespace

import pytest

from tests.fixtures.qwen4_vector_sdpa_gate import CASES, ORDER, should_stop, vector_plan, vector_prefill
from tests.test_qwen4_sdpa_tiling_pure import Array


@pytest.mark.parametrize('tile', [1, 2])
@pytest.mark.parametrize('total', [9, 17, 33, 129, 1024])
def test_vector_dispatch_spans_are_complete_and_respect_gqa_bound(tile, total):
    spans = vector_plan(total, tile, 24, 2)
    assert spans[0][0] == 0 and spans[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(0 < b-a <= tile and (b-a)*12 <= 32 for a, b in spans)


@pytest.mark.parametrize('args', [(True,1,24,2), (8,1,24,2), (1025,1,24,2),
    (17,3,24,2), (17,1,24,0), (17,1,25,2), (17,2,64,2), (17,1,24.0,2)])
def test_invalid_or_unsupported_vector_shapes_fail(args):
    with pytest.raises(ValueError): vector_plan(*args)


@pytest.mark.parametrize('rank', [2, 4])
def test_exact_mask_slices_and_shared_unmodified_full_kv(rank):
    events=[]; q=Array((1,24,17,256),'q',events);k=Array((1,2,257,256),'k',events)
    v=Array(k.shape,'v',events); mask=Array((17,257) if rank==2 else (1,1,17,257),'mask',events)
    def sdpa(query, key, value, **kw):
        assert key is k and value is v and kw['mask'].shape[-2] == query.shape[2]
        events.append(('native', query.shape[2]));return query
    mx=SimpleNamespace(bfloat16='bf16',float16='fp16',fast=SimpleNamespace(scaled_dot_product_attention=sdpa),
        eval=lambda value:events.append(('eval',value.shape[2])),concatenate=lambda values,axis:events.append(('concat',axis)))
    vector_prefill(mx,q,k,v,scale=.0625,mask=mask,tile=2)
    assert [e for e in events if e[0]=='native']==[('native',2)]*8+[('native',1)]
    assert not any(e[0]=='slice' and e[1] in ('k','v') for e in events)
    assert events[-1]==('concat',2)
    with pytest.raises(ValueError):vector_prefill(mx,q,k,v,scale=.0625,mask='causal',tile=2)


def test_shape_ladder_and_bit_mismatch_stop_are_preregistered():
    assert len(CASES)==len({seed for _,_,seed in CASES})==3
    assert CASES[0][:2]==(17,257) and CASES[-1][1]==32768
    assert ORDER==ORDER[::-1] and ORDER==(0,1,2,2,1,0)
    assert not should_stop([{'arms':[{'exact':True}]}])
    assert should_stop([{'arms':[{'exact':True},{'exact':False}]}])
