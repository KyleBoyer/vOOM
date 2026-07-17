"""F33 harness, milestone 2b: verify runtime/glm_dsa.py's DSAState top-k
SELECTION (the set of chosen positions) matches HF's real GlmMoeDsaIndexer
at S > index_topk, where real DSA sparsity engages (no NAS, no real
GLM-5.2 weights needed).

Avoids needing an HF Cache object: calls the indexer in ONE forward pass
over the full S-length sequence (attention_mask=None, causal handled via
position_ids), then compares the LAST query row's selection -- the same
set a decode step at position S-1 would compute against a fully-populated
cache.

**This test caught a real production bug (2026-07-14):** runtime/glm_dsa.py's
`DSAState._rope_idx` split the indexer's rope/pass-through halves as
nope-first/rope-last (matching the MAIN MLA attention's convention), but
the actual official `GlmMoeDsaIndexer.forward()` (transformers==5.13.0)
splits/concatenates rope-FIRST/pass-SECOND for the indexer specifically --
the opposite convention, confirmed by reading the reference source
directly. The bug was silent: a wrong-but-internally-consistent top-k
selection is still a well-formed, in-range index set, so no crash or
shape error revealed it -- only comparing against the real reference
caught it. Confirmed via 13 random seeds that swapping the split order
(now the production code) turns mismatches into exact matches, not just
one lucky case. This directly affects Goal 2 (long-context validation):
any real GLM-5.2 run that exceeds index_topk=2048 tokens would have
selected the WRONG top-k positions before this fix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm_dsa import DSAState
from runtime.layer_runner import _linear

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf_mod  # noqa: E402

HIDDEN = 48
Q_LORA = 32
DR = 8  # qk_rope_head_dim (shared between attention and indexer in this runtime)
INDEX_N_HEADS = 32   # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_HEAD_DIM = 128  # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_TOPK = 4
S = 7  # > INDEX_TOPK, so real sparsity engages


def _build_hf_indexer():
    torch.manual_seed(2)  # a seed that mismatched before the fix
    hf_cfg = GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=1, first_k_dense_replace=0,
        num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        kv_lora_rank=16, q_lora_rank=Q_LORA,
        qk_rope_head_dim=DR, qk_nope_head_dim=8, v_head_dim=16,
        index_topk=INDEX_TOPK, index_n_heads=INDEX_N_HEADS, index_head_dim=INDEX_HEAD_DIM,
        indexer_types=["full"], attn_implementation="eager",
        rope_theta=10000.0,
    )
    indexer = hf_mod.GlmMoeDsaIndexer(hf_cfg, layer_idx=0)
    indexer.eval()
    with torch.no_grad():
        for p in indexer.parameters():
            p.normal_(mean=0.0, std=0.3)
    rope = hf_mod.GlmMoeDsaRotaryEmbedding(hf_cfg)
    q_a_proj = torch.nn.Linear(HIDDEN, Q_LORA, bias=False)
    q_a_layernorm = hf_mod.GlmMoeDsaRMSNorm(Q_LORA)  # released latent eps=1e-6
    with torch.no_grad():
        q_a_proj.weight.normal_(mean=0.0, std=0.3)
    return hf_cfg, indexer, rope, q_a_proj, q_a_layernorm


def _runtime_config(hf_cfg) -> ModelConfig:
    return ModelConfig(
        model_type="glm_moe_dsa", hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=64, rms_norm_eps=hf_cfg.rms_norm_eps, rope_theta=10000.0,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
        head_dim=16, eos_token_ids=(0,), torch_dtype="float32",
        qk_nope_head_dim=8, qk_rope_head_dim=DR, v_head_dim=16,
        rope_interleave=True, index_topk=INDEX_TOPK, indexer_types=("full",),
    )


def _hf_last_row_selection(hf_cfg, indexer, rope, q_a_proj, q_a_layernorm, h_torch) -> set:
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)
    with torch.no_grad():
        q_resid = q_a_layernorm(q_a_proj(h_torch))
        hf_topk = indexer(h_torch, q_resid, (cos, sin), None, position_ids, past_key_values=None)
    return set(int(i) for i in hf_topk[0, -1].tolist())


def _runtime_last_row_selection(hf_cfg, indexer, h_torch) -> set:
    sd = indexer.state_dict()
    w = {}
    for name in ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"]:
        w[f"layer0.self_attn.indexer.{name}"] = mx.array(sd[name].numpy())
    # q_a_proj/q_a_layernorm are set by the caller into w before this runs
    cfg = _runtime_config(hf_cfg)
    dsa = DSAState(cfg)
    h_mx = mx.array(h_torch.numpy())
    dsa.observe(0, "full", h_mx, w, "layer0", offset=0)

    h_last = h_mx[:, -1:, :]
    q_a_last = mx.fast.rms_norm(
        _linear(h_last, w, "layer0.self_attn.q_a_proj"),
        w["layer0.self_attn.q_a_layernorm.weight"], cfg.mla_latent_norm_eps,
    )
    sel = dsa.update_and_select(0, "full", h_last, q_a_last, w, "layer0", offset=S - 1)
    return set(int(i) for i in sel[0, 0].tolist()), w


def test_dsa_indexer_topk_selection_matches_hf():
    hf_cfg, indexer, rope, q_a_proj, q_a_layernorm = _build_hf_indexer()
    torch.manual_seed(20)
    h_torch = torch.randn(1, S, HIDDEN)

    hf_selection = _hf_last_row_selection(hf_cfg, indexer, rope, q_a_proj, q_a_layernorm, h_torch)

    sd = indexer.state_dict()
    w = {}
    for name in ["wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"]:
        w[f"layer0.self_attn.indexer.{name}"] = mx.array(sd[name].numpy())
    w["layer0.self_attn.q_a_proj.weight"] = mx.array(q_a_proj.weight.detach().numpy())
    w["layer0.self_attn.q_a_layernorm.weight"] = mx.array(q_a_layernorm.weight.detach().numpy())

    cfg = _runtime_config(hf_cfg)
    dsa = DSAState(cfg)
    h_mx = mx.array(h_torch.numpy())
    dsa.observe(0, "full", h_mx, w, "layer0", offset=0)
    h_last = h_mx[:, -1:, :]
    q_a_last = mx.fast.rms_norm(
        _linear(h_last, w, "layer0.self_attn.q_a_proj"),
        w["layer0.self_attn.q_a_layernorm.weight"], cfg.mla_latent_norm_eps,
    )
    sel = dsa.update_and_select(0, "full", h_last, q_a_last, w, "layer0", offset=S - 1)
    runtime_selection = set(int(i) for i in sel[0, 0].tolist())

    assert runtime_selection == hf_selection, (
        f"DSA indexer top-k mismatch: HF selected {sorted(hf_selection)}, "
        f"runtime selected {sorted(runtime_selection)}"
    )


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
