"""F33 harness, milestone 2c: verify runtime/glm.py's COMPACT sparse attention
output (gather selected latent rows, dense-attend over just that compact set)
numerically matches HF's real GlmMoeDsaAttention at S > index_topk, closing
the "sparse attention output, not only membership" gap named in STATUS.md's
"Current truth" section.

Builds on milestone 2a (tests/test_f33_mla_attention.py: dense MLA attention
matches HF, using index_topk >= S to keep the indexer a no-op) and milestone
2b (tests/test_f33_dsa_indexer.py: the top-k SELECTION SET matches HF at
S > index_topk, but only compared as a `set()` -- order discarded, and only
the indexer's own output, not the downstream attention computation).

The key subtlety this test actually exercises: HF's eager/SDPA backend does
NOT gather only the selected keys -- it computes attention over the FULL
causal history and additively masks non-selected positions to -inf (see
`GlmMoeDsaAttention.forward()`: `attention_mask.masked_fill(index_mask, -inf)`,
with the actual gathered/compact path reserved for the `flash-mla` kernel,
per that method's own comment "consumed by flash_mla_with_kvcache; ignored by
eager/SDPA"). This runtime's F21/F22 path instead genuinely GATHERS the
selected rows (`mx.take(lat_all[0], sel[0, 0], axis=0)`) and computes a
SMALLER dense attention over just the topk set -- mathematically equivalent
in exact arithmetic (masked positions contribute exp(-inf)=0 either way),
but floating-point summation is not associative, so the two code paths sum
in different orders and could in principle diverge more than float noise if
something else were wrong. `runtime/glm_dsa.py`'s `update_and_select` already
sorts the selection back into chronological (ascending-position) order
specifically so its reduction order matches HF's natural 0..S-1 iteration
order (see the comment there) -- this test is the first time that claim is
checked against a real numeric comparison, not just reasoned about.

Empirically confirmed (2026-07-14): max abs diff 3.58e-7 between HF's
masked-dense and this runtime's gathered-compact attention output at the
decode step where S=7 > index_topk=4 (real sparsity), using the same tiny
synthetic GlmMoeDsaConfig family as milestones 2a/2b -- float32-noise scale,
comfortably inside the 1e-4 gate this project's other F33 milestones use.

Honesty note on the chronological-sort claim specifically: an earlier version
of this file also tried to empirically demonstrate that skipping glm_dsa.py's
`mx.sort` (using raw argpartition order instead) measurably changes the
output. It doesn't, at any scale tried here (topk=4 through topk=64): the
sorted-vs-unsorted delta is ~3e-8-4e-8, indistinguishable from float32 noise,
and not even directionally consistent (unsorted was occasionally CLOSER to
HF than sorted, by noise). That ablation is not included -- asserting it
would have overstated what was actually measured. The sort may still matter
at real GLM scale (topk=2048, thousands of summed terms, where float
non-associativity compounds differently), but this synthetic harness cannot
demonstrate that scale, so the sort's necessity here rests on the source
comment's reasoning, not an independent measurement.
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
from runtime.glm_dsa import DSAState
from runtime.kv_cache import KVCache

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf_mod  # noqa: E402

HIDDEN = 48
N_HEADS = 4
DN, DR, DV = 8, 8, 16  # qk_nope, qk_rope, v_head
Q_LORA = 32
KV_LORA = 16
INDEX_N_HEADS = 32   # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_HEAD_DIM = 128  # hardcoded in runtime/glm_dsa.py -- not configurable
INDEX_TOPK = 4
S = 7  # > INDEX_TOPK, so real sparsity engages at the last (decode) row
ROPE_THETA = 10000.0

ATTN_WEIGHT_NAMES = [
    "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
    "o_proj.weight",
]
INDEXER_WEIGHT_NAMES = [
    "indexer.wq_b.weight", "indexer.wk.weight", "indexer.k_norm.weight",
    "indexer.k_norm.bias", "indexer.weights_proj.weight",
]


def _build_hf_attention(seed: int):
    torch.manual_seed(seed)
    hf_cfg = GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=1, first_k_dense_replace=0,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        kv_lora_rank=KV_LORA, q_lora_rank=Q_LORA,
        qk_rope_head_dim=DR, qk_nope_head_dim=DN, v_head_dim=DV,
        index_topk=INDEX_TOPK, index_n_heads=INDEX_N_HEADS, index_head_dim=INDEX_HEAD_DIM,
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


def _runtime_config(hf_cfg) -> ModelConfig:
    return ModelConfig(
        model_type="glm_moe_dsa", hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        vocab_size=64, rms_norm_eps=hf_cfg.rms_norm_eps, rope_theta=ROPE_THETA,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
        head_dim=DN + DR, eos_token_ids=(0,), torch_dtype="float32",
        qk_nope_head_dim=DN, qk_rope_head_dim=DR, v_head_dim=DV,
        rope_interleave=True, index_topk=INDEX_TOPK, indexer_types=("full",),
    )


def _weights_from_hf(attn) -> dict:
    sd = attn.state_dict()
    w = {}
    for name in ATTN_WEIGHT_NAMES + INDEXER_WEIGHT_NAMES:
        w[f"layer0.self_attn.{name}"] = mx.array(sd[name].numpy())
    return w


def _runtime_last_row_output(w, cfg, h_mx) -> np.ndarray:
    kv = KVCache(num_layers=1)
    kv.compressed_mla = True
    kv.dsa = DSAState(cfg)
    _ = _mla_attention(h_mx[:, : S - 1], w, "layer0", cfg, kv, layer=0, offset=0)
    out = _mla_attention(h_mx[:, S - 1 :], w, "layer0", cfg, kv, layer=0, offset=S - 1)
    mx.eval(out)
    return np.array(out)


def test_compact_sparse_attention_output_matches_hf_masked_dense():
    hf_cfg, attn, rope = _build_hf_attention(seed=3)
    torch.manual_seed(4)
    h_torch = torch.randn(1, S, HIDDEN)
    position_ids = torch.arange(S)[None, :]
    cos, sin = rope(h_torch, position_ids)

    with torch.no_grad():
        hf_out_all, _, hf_topk_all = attn(
            h_torch, (cos, sin), None, past_key_values=None, position_ids=position_ids
        )
    last_row_sel = set(int(i) for i in hf_topk_all[0, -1].tolist())
    assert last_row_sel != set(range(S)), (
        "expected real DSA sparsity at the last row (S > index_topk) -- "
        "if this fails the comparison below is not exercising the sparse path"
    )
    hf_last_out = hf_out_all[:, -1:, :].detach().numpy()

    w = _weights_from_hf(attn)
    cfg = _runtime_config(hf_cfg)
    h_mx = mx.array(h_torch.numpy())
    runtime_np = _runtime_last_row_output(w, cfg, h_mx)

    assert hf_last_out.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_last_out - runtime_np))
    assert max_abs_diff < 1e-4, (
        f"compact sparse attention output mismatch at S={S} > index_topk={INDEX_TOPK}: "
        f"max abs diff {max_abs_diff}"
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
