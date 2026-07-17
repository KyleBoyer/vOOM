"""F33 harness, milestone 2a: verify runtime/glm.py's _mla_attention matches
HF's real GlmMoeDsaAttention, using a tiny synthetic config (no NAS, no real
GLM-5.2 weights, no BRIEF 0 gate needed) -- same approach as milestone 1
(tests/test_f33_moe_layout_conversion.py).

Uses `index_topk >= S` so HF's indexer-driven top-k mask is a no-op (every
position gets selected) -- both sides reduce to plain causal MLA attention,
letting this test isolate the q_a/q_b/kv_a/kv_b/RoPE/output-projection math
without needing DSAState/compressed-latent-cache scaffolding (that's
milestone 2b: real sparse selection at S > index_topk during decode, which
also needs an HF `Cache` object and this runtime's `DSAState` wired in
lockstep across a prefill+decode sequence -- a bigger, not-yet-attempted
task).

Empirically confirmed (2026-07-14) on the first attempt: max abs diff
~3e-6 between HF and this runtime's dense MLA attention output, confirming
runtime/glm.py's interleaved-RoPE convention and MLA head-splitting exactly
match the released architecture's official implementation -- not just
self-consistency, an actual external oracle check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm import _mla_attention
from runtime.kv_cache import KVCache

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf_mod  # noqa: E402

HIDDEN = 48
N_HEADS = 4
DN, DR, DV = 8, 8, 16  # qk_nope, qk_rope, v_head
S = 5
ROPE_THETA = 10000.0


def _build_hf_attention():
    torch.manual_seed(0)
    hf_cfg = GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=1, first_k_dense_replace=0,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        kv_lora_rank=16, q_lora_rank=32,
        qk_rope_head_dim=DR, qk_nope_head_dim=DN, v_head_dim=DV,
        index_topk=8,  # >= S=5: indexer selection is a no-op (selects all)
        indexer_types=["full"], attn_implementation="eager",
        rope_theta=ROPE_THETA,
    )
    attn = hf_mod.GlmMoeDsaAttention(hf_cfg, layer_idx=0)
    attn.eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(mean=0.0, std=0.3)
    rope = hf_mod.GlmMoeDsaRotaryEmbedding(hf_cfg)
    return hf_cfg, attn, rope


def _runtime_config(rms_norm_eps: float) -> ModelConfig:
    return ModelConfig(
        model_type="glm_moe_dsa", hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        vocab_size=64, rms_norm_eps=rms_norm_eps, rope_theta=ROPE_THETA,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
        head_dim=DN + DR, eos_token_ids=(0,), torch_dtype="float32",
        qk_nope_head_dim=DN, qk_rope_head_dim=DR, v_head_dim=DV,
        rope_interleave=True, index_topk=0,  # 0: skip DSA entirely on this side too
    )


def test_dense_mla_attention_matches_hf():
    hf_cfg, attn, rope = _build_hf_attention()

    torch.manual_seed(1)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)

    with torch.no_grad():
        hf_out, _, topk = attn(h_torch, (cos, sin), None, past_key_values=None, position_ids=position_ids)
    # Sanity: with index_topk >= S, every position must be selected for every
    # query row (a no-op mask) -- if this ever fails, the "dense" premise of
    # this test broke and the comparison below is no longer apples-to-apples.
    for row in topk[0].tolist():
        assert set(row) == set(range(S))

    sd = attn.state_dict()
    w = {}
    for name in ["q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
                 "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
                 "o_proj.weight"]:
        w[f"layer0.self_attn.{name}"] = mx.array(sd[name].numpy())

    cfg = _runtime_config(hf_cfg.rms_norm_eps)
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _mla_attention(h_mx, w, "layer0", cfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-4, f"HF vs runtime MLA attention mismatch: max abs diff {max_abs_diff}"


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
