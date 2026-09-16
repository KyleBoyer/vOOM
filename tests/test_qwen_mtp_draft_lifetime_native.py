"""Real native draft math with tiny tensors; not full-model qualification."""
from types import SimpleNamespace as NS
import numpy as np
import mlx.core as mx
import pytest
from runtime import quant
from runtime.kv_cache import KVCache
from runtime.lm_head_stream import StreamedLMHead
from runtime.qwen35_mtp import QwenMTPDrafter
from runtime.qwen_mtp_draft_lifetime import configure, FLAG


def make_drafter(active, packed, seed):
    rng=np.random.default_rng(seed)
    shapes={
        'mtp.fc.weight':(64,128),
        'mtp.layers.0.self_attn.q_proj.weight':(128,64),
        'mtp.layers.0.self_attn.k_proj.weight':(64,64),
        'mtp.layers.0.self_attn.v_proj.weight':(64,64),
        'mtp.layers.0.self_attn.o_proj.weight':(64,64),
        'mtp.layers.0.mlp.gate_proj.weight':(128,64),
        'mtp.layers.0.mlp.up_proj.weight':(128,64),
        'mtp.layers.0.mlp.down_proj.weight':(64,128)}
    raw={k:(rng.standard_normal(shape)*.05).astype(np.float32) for k,shape in shapes.items()}
    raw.update({k:(rng.standard_normal(64)*.01).astype(np.float32)
                for k in QwenMTPDrafter._PACKED_NORM_NAMES})
    def weights():
        result={k:mx.array(v).astype(mx.bfloat16) for k,v in raw.items()}
        if packed:
            for k in shapes:
                q,s=mx.quantize(result[k],group_size=32,bits=4,mode='mxfp4')
                result[k]=quant.QTensor(q,s,None,4,32,'mxfp4')
        return result
    class Cache:
        total_bytes=0
        page=None
        def prepare_for(self,n):pass
        def get(self,key,names,**kw):
            if self.page is None:self.page=weights()
            return self.page
        def discard(self,key,names):
            existed=self.page is not None;self.page=None;return existed
    cache=Cache()
    representation='mxfp4-q4-g32' if packed else 'released-bf16'
    store=NS(names_with_prefix=lambda prefix:list(raw),
        mtplx_mtp_sidecar=None if packed else 'tiny',mtp_proposal_representation=representation,
        _mtplx_mtp_sidecar_layout={k:(None,None,int(v.size*2)) for k,v in raw.items()},
        mlx_quantized_resident_bytes=lambda names:100000)
    cfg=NS(model_type='qwen3_5',num_experts=0,rms_norm_eps=1e-6,
        vocab_size=64,num_attention_heads=1,num_key_value_heads=1,
        head_dim=64,partial_rotary_factor=.5,rope_theta=10000)
    embed=mx.array(rng.standard_normal((64,64)).astype(np.float32)).astype(mx.bfloat16)
    head_weight=mx.array(rng.standard_normal((64,64)).astype(np.float32)).astype(mx.bfloat16)
    observations=[]
    class Head(StreamedLMHead):
        def logits(self,h):
            observations.append(cache.page is None)
            return quant.matmul(h,head_weight)
    head=Head.__new__(Head)
    engine=NS(store=store,cache=cache,cfg=cfg,rc=NS(qwen35_mxfp4_head_rows=8192),
        _embed=lambda tokens:embed[mx.array(tokens)][None],_lm_head_weight=lambda:head)
    drafter=QwenMTPDrafter(engine)
    configure(drafter,engine,QwenMTPDrafter,{FLAG:str(int(active))})
    return drafter,cache,observations


def raw_bytes(value):
    mx.eval(value)
    return np.asarray(value.astype(mx.float32)).tobytes()


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('seed',[11,71])
def test_recurrent_draft_logits_hidden_kv_rng_are_identical(packed,seed):
    outcomes=[]
    for active in (False,True):
        d,cache,observations=make_drafter(active,packed,seed)
        w=d.prepare_request_weights();kv=KVCache(1)
        hidden=mx.full((1,1,64),.125,dtype=mx.bfloat16)
        mx.random.seed(431)
        rows=[]
        for step in range(4):
            logits,hidden=d.draft_step(hidden,step+2,kv,step,w)
            rows.append((raw_bytes(logits),raw_bytes(hidden),
                         raw_bytes(kv.keys[0]),raw_bytes(kv.values[0])))
            assert bool(w) is not active
        outcomes.append((rows,raw_bytes(mx.random.uniform(shape=(4,)))))
        assert observations==[active]*4
        if active:
            assert d._head_release_stats['releases']==4
            assert d._head_release_stats['reloads']==3
            assert cache.page is None
        d.release_request_weights(w)
    assert outcomes[0]==outcomes[1]


def test_head_failure_leaves_mapping_and_cache_released(monkeypatch):
    d,cache,_=make_drafter(True,True,11);w=d.prepare_request_weights()
    def fail(*args):raise MemoryError('ordinary head reservation refused')
    monkeypatch.setattr('runtime.qwen35_mtp.final_logits',fail)
    with pytest.raises(MemoryError):
        d.draft_step(mx.zeros((1,1,64),dtype=mx.bfloat16),1,KVCache(1),0,w)
    assert w=={} and cache.page is None
