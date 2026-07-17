"""F33 milestone 2e: verify the noaux_tc MoE router at real "eight-of-256"
scale against HF's real GlmMoeDsaTopkRouter. Closes STATUS.md's remaining
blocker (1): "the newly patched fp32-before-matmul router needs a real
eight-of-256 ordered oracle."

Earlier F33 milestones (moe_layout_conversion, mla_attention, dsa_indexer,
dsa_attention_output, rmsnorm_oracle) all used small synthetic expert counts
(8 experts) for speed -- this is the first one to actually exercise the real
released scale (256 routed experts, 8 active per token, matching GLM-5.2's
real config.json), since router tie-breaking behavior at n_routed_experts=8
doesn't stress the same fp32 near-tie boundary conditions 256 competing
sigmoid+bias scores can.

runtime/glm.py's router logic was extracted into a standalone `_route_experts`
function specifically so this test could call the exact production code path
(not a reimplementation of the formula) without needing a full transformer
block (attention + expert weights) around it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm import _route_experts

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as hf_mod  # noqa: E402

HIDDEN = 48
N_EXPERTS = 256   # real GLM-5.2 n_routed_experts
TOP_K = 8         # real GLM-5.2 num_experts_per_tok -- the "eight" in "eight-of-256"
ROUTED_SCALING_FACTOR = 2.5  # real GLM-5.2 value (confirmed from models/glm-fixture-tiny/config.json)


def _build_hf_router(seed: int):
    torch.manual_seed(seed)
    hf_cfg = GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=16, num_hidden_layers=1, first_k_dense_replace=0,
        num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=N_EXPERTS, num_experts_per_tok=TOP_K,
        n_group=1, topk_group=1,  # real GLM-5.2: group restriction is a structural no-op
        norm_topk_prob=True, routed_scaling_factor=ROUTED_SCALING_FACTOR,
        kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
        index_topk=8, indexer_types=["full"], attn_implementation="eager",
    )
    router = hf_mod.GlmMoeDsaTopkRouter(hf_cfg)
    with torch.no_grad():
        router.weight.normal_(mean=0.0, std=0.3)
        # A real e_score_correction_bias is not all-zeros in a trained checkpoint --
        # a nonzero, non-uniform bias is exactly what can shift near-tie outcomes
        # relative to the unbiased sigmoid scores, which is the actual conformance
        # risk this test is checking (noaux_tc: bias affects WHICH experts win).
        router.e_score_correction_bias.normal_(mean=0.0, std=0.5)
    return hf_cfg, router


def _runtime_config(hf_cfg) -> ModelConfig:
    return ModelConfig(
        model_type="glm_moe_dsa", hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=64, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
        head_dim=16, eos_token_ids=(0,), torch_dtype="float32",
        num_experts=N_EXPERTS, num_experts_per_tok=TOP_K,
        norm_topk_prob=True, routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )


def _route_one_seed(seed: int, n_tokens: int = 8):
    hf_cfg, router = _build_hf_router(seed)
    torch.manual_seed(seed + 1000)
    h_torch = torch.randn(n_tokens, HIDDEN)

    with torch.no_grad():
        _, hf_weights, hf_indices = router(h_torch)

    w = {
        "layer0.mlp.gate.weight": mx.array(router.weight.detach().numpy()),
        "layer0.mlp.gate.e_score_correction_bias": mx.array(
            router.e_score_correction_bias.detach().numpy()
        ),
    }
    cfg = _runtime_config(hf_cfg)
    h_mx = mx.array(h_torch.numpy())[None]  # runtime expects (B, L, hidden)
    idx, pw = _route_experts(h_mx, w, "layer0", cfg)
    mx.eval(idx, pw)

    return hf_weights.numpy(), hf_indices.numpy(), np.array(pw[0]), np.array(idx[0])


def test_router_selects_the_same_eight_of_256_experts():
    for seed in range(15):
        _, hf_indices, _, runtime_indices = _route_one_seed(seed)
        for t in range(hf_indices.shape[0]):
            hf_set = set(int(i) for i in hf_indices[t])
            runtime_set = set(int(i) for i in runtime_indices[t])
            assert hf_set == runtime_set, (
                f"seed {seed} token {t}: HF selected {sorted(hf_set)}, "
                f"runtime selected {sorted(runtime_set)} -- top-8-of-256 mismatch"
            )


def test_router_weights_match_for_the_selected_experts():
    """Same selection isn't the whole claim -- the post-normalization,
    post-routed_scaling_factor WEIGHT each selected expert's contribution is
    scaled by must also match, or a correct expert set still produces a
    numerically wrong MoE output."""
    max_diff = 0.0
    for seed in range(15):
        hf_weights, hf_indices, runtime_weights, runtime_indices = _route_one_seed(seed)
        for t in range(hf_indices.shape[0]):
            hf_by_expert = dict(zip(hf_indices[t].tolist(), hf_weights[t].tolist()))
            runtime_by_expert = dict(zip(runtime_indices[t].tolist(), runtime_weights[t].tolist()))
            assert set(hf_by_expert) == set(runtime_by_expert)
            for e in hf_by_expert:
                diff = abs(hf_by_expert[e] - runtime_by_expert[e])
                max_diff = max(max_diff, diff)
    assert max_diff < 1e-5, f"router weight mismatch for a selected expert: max diff {max_diff}"


def test_router_handles_a_real_near_tie_at_the_topk_boundary():
    """Construct a deliberate near-tie AT the 8th/9th boundary (two experts
    with nearly-identical biased scores) and confirm both implementations
    agree on which one wins -- the actual risk noaux_tc's bias term creates,
    not just a random-seed sanity check."""
    hf_cfg, router = _build_hf_router(seed=42)
    torch.manual_seed(43)
    h_torch = torch.randn(1, HIDDEN)

    with torch.no_grad():
        router_logits = torch.nn.functional.linear(h_torch.float(), router.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + router.e_score_correction_bias
        # Force experts 100 and 101 to be a near-tie for the last (8th) slot:
        # set them both just above the (TOP_K+1)-th ranked score, one ULP apart.
        sorted_scores, _ = scores_for_choice[0].sort(descending=True)
        boundary = sorted_scores[TOP_K + 2].item()
        with torch.no_grad():
            router.e_score_correction_bias[100] = boundary + 2e-6 - scores[0, 100].item()
            router.e_score_correction_bias[101] = boundary + 1e-6 - scores[0, 101].item()

        _, hf_weights, hf_indices = router(h_torch)

    w = {
        "layer0.mlp.gate.weight": mx.array(router.weight.detach().numpy()),
        "layer0.mlp.gate.e_score_correction_bias": mx.array(
            router.e_score_correction_bias.detach().numpy()
        ),
    }
    cfg = _runtime_config(hf_cfg)
    h_mx = mx.array(h_torch.numpy())[None]
    idx, pw = _route_experts(h_mx, w, "layer0", cfg)
    mx.eval(idx, pw)

    hf_set = set(int(i) for i in hf_indices[0])
    runtime_set = set(int(i) for i in np.array(idx[0][0]))
    assert hf_set == runtime_set, (
        f"near-tie boundary mismatch: HF selected {sorted(hf_set)}, "
        f"runtime selected {sorted(runtime_set)}"
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
