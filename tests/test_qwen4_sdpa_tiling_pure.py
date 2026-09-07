"""Pure bounds/real helper scheduling tests; never import MLX."""

from types import SimpleNamespace

import pytest

from runtime.qwen4_sdpa_tiling import query_tiles, query_tiled_sdpa


@pytest.mark.parametrize('tile', [0, 128, 256, 512])
@pytest.mark.parametrize('total', [1, 17, 128, 129, 257, 513, 1024, 1025, 1031, 1033])
def test_partition_is_complete_and_no_new_vector_tail(total, tile):
    spans = query_tiles(total, tile)
    assert spans[0][0] == 0 and spans[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(a < b for a, b in spans)
    if len(spans) > 1:
        assert all(8 < b - a <= tile + 8 for a, b in spans)


@pytest.mark.parametrize('total,tile', [(True,128),(0,128),(-1,128),(1_048_577,128),
    (100,True),(100,-1),(100,64),(100,128.0),(100,'128')])
def test_invalid_plans_fail_closed(total, tile):
    with pytest.raises(ValueError): query_tiles(total, tile)


class Array:
    def __init__(self, shape, label, events, dtype='bf16'):
        self.shape, self.label, self.events, self.dtype = shape, label, events, dtype
        self.ndim = len(shape)
    def __getitem__(self, index):
        if len(index)==2:
            a,b=index[0].start,index[0].stop
            self.events.append(('slice',self.label,a,b))
            return Array((b-a,self.shape[-1]),self.label,self.events,self.dtype)
        a, b = index[2].start, index[2].stop
        self.events.append(('slice', self.label, a, b))
        return Array((*self.shape[:2], b-a, self.shape[3]), self.label, self.events, self.dtype)


@pytest.mark.parametrize('mask_kind', ['none', 'broadcast', 'full','causal2d','broadcast2d'])
def test_actual_helper_shares_full_kv_and_evaluates_each_chunk_before_next(mask_kind):
    events = []
    q = Array((1,24,257,256), 'query', events)
    k = Array((1,2,1024,256), 'keys', events)
    v = Array((1,2,1024,256), 'values', events)
    mask = None if mask_kind == 'none' else Array((1,1,1 if mask_kind=='broadcast' else 257,1024), 'mask', events)
    if mask_kind.endswith('2d'):
        mask=Array((1 if mask_kind=='broadcast2d' else 257,1024),'mask',events)
    def sdpa(query, keys, values, **kwargs):
        assert keys is k and values is v and kwargs['scale'] == 0.0625
        if mask_kind not in ('full','causal2d'): assert kwargs['mask'] is mask
        events.append(('sdpa', query.shape[2]))
        return query
    mx = SimpleNamespace(bfloat16='bf16', float16='fp16', fast=SimpleNamespace(scaled_dot_product_attention=sdpa),
        eval=lambda a: events.append(('eval',a.shape[2])),
        concatenate=lambda arrays, axis: events.append(('concatenate',axis,len(arrays))))
    query_tiled_sdpa(mx,q,k,v,scale=0.0625,mask=mask,tile_queries=128)
    assert [e for e in events if e[0] != 'slice'] == [('sdpa',128),('eval',128),('sdpa',129),('eval',129),('concatenate',2,2)]
    assert all(e[1] not in ('keys','values') for e in events if e[0]=='slice')


@pytest.mark.parametrize('tile,total', [(0,257),(128,17),(256,129),(512,512)])
def test_single_call_is_unchanged_and_not_evaluated(tile,total):
    q = SimpleNamespace(shape=(1,24,total,256))
    sentinel=object();calls=[]
    mx=SimpleNamespace(fast=SimpleNamespace(scaled_dot_product_attention=lambda *a,**k: calls.append((a,k)) or sentinel))
    assert query_tiled_sdpa(mx,q,'k','v',scale=.0625,mask='unchanged',tile_queries=tile) is sentinel
    assert calls==[((q,'k','v'),dict(scale=.0625,mask='unchanged'))]


@pytest.mark.parametrize('mask', ['causal', SimpleNamespace(ndim=2),
    SimpleNamespace(ndim=4,shape=(1,1,128,1024)), SimpleNamespace(ndim=4,shape=(1,1,257,1023))])
def test_sliced_attention_rejects_implicit_or_wrong_offset_mask(mask):
    events=[];q=Array((1,24,257,256),'q',events);k=Array((1,2,1024,256),'k',events)
    mx=SimpleNamespace(bfloat16='bf16',float16='fp16')
    with pytest.raises(ValueError,match='precomputed'):
        query_tiled_sdpa(mx,q,k,k,scale=.0625,mask=mask,tile_queries=128)
    assert events==[]


def test_component_pressure_is_separate_and_strict():
    from tests.fixtures.qwen4_sdpa_tiling_gate import pressure_failures
    clean=dict(native={'available':True},available_bytes=6_000_000_000,
               swap_used_bytes=0,swap_out_bytes=0,root_free_bytes=11_000_000_000)
    assert pressure_failures([clean],8_500_000_000)==[]
    for key,value,reason in [('available_bytes',5_299_999_999,'available'),
                            ('swap_used_bytes',16_000_001,'net swap'),
                            ('swap_out_bytes',16_000_001,'actual swap'),
                            ('root_free_bytes',9_999_999_999,'root free')]:
        assert any(reason in error for error in pressure_failures([clean,{**clean,key:value}],0))
    assert pressure_failures([clean],8_500_000_001)
    assert pressure_failures([],0)
    assert pressure_failures([{**clean,'native':{'available':False}}],0)


def test_component_corpus_has_independent_seeds_shapes_and_balanced_order():
    from tests.fixtures.qwen4_sdpa_tiling_gate import CASES,ORDER
    assert len(CASES)==len({case[-1] for case in CASES})==5
    assert {case[0] for case in CASES}=={1,2}
    assert max(case[2] for case in CASES)==32768
    assert any(case[1]%128 in (1,7) and case[1]>128 for case in CASES)
    assert ORDER==ORDER[::-1] and all(ORDER.count(tile)==2 for tile in (0,128,256,512))
