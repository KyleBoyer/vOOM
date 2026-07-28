"""F128: Kimi K3's real MoE gate fixed a real aliasing bug present in the
original Kimi Linear 48B checkpoint's own released code.

Kimi Linear 48B's real, bundled `modeling_kimi.py` computes:

    scores_for_choice = scores.view(bsz * seq_len, -1)
    scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
    ...
    topk_weight = scores.gather(1, topk_idx)

`scores_for_choice` is a VIEW of `scores`; the in-place `+=` mutates the
underlying storage, so `scores` itself is bias-corrected by the time
`topk_weight = scores.gather(...)` runs -- the routing WEIGHT (not just
selection) ends up computed from the biased score, contradicting noaux_tc's
whole design intent. F92 (2026-07-18) found this, confirmed it against the
real reference, and ported it verbatim into
`runtime.kimi_linear._route_experts` (the `else` branch, gathering from
`biased`) -- correct for Kimi Linear 48B and Kimi K2.5's `kimi_k25`
(same-shape gate, GLM's own `_route_experts` handles that one).

Kimi K3's real, bundled `modeling_kimi_linear.py` (checked 2026-07-28,
`models/Kimi-K3/modeling_kimi_linear.py`) computes the SAME logical step
differently:

    scores = scores.view(bsz * seq_len, -1)
    scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
    ...
    topk_weight = scores.gather(1, topk_idx)

`scores + bias` (regular addition, not `+=`) always allocates a NEW
tensor -- `scores` itself is never mutated, so `topk_weight = scores.gather(...)`
genuinely reads the UNBIASED scores. Moonshot fixed the aliasing bug
between the two releases. This test transcribes both real formulas
verbatim and checks `_route_experts` picks the right one per
`cfg.model_type` (`"kimi_k3"` -> fixed/unbiased; `"kimi_linear"` -> the
original aliasing bug, unchanged and regression-tested here too).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from runtime.config import ModelConfig
from runtime.kimi_linear import _route_experts

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")


def _real_gate_aliasing_bug(scores, bias, top_k):
    """Verbatim from the real Kimi-Linear-48B-A3B-Instruct
    modeling_kimi.py::KimiMoEGate.forward (num_expert_group=1 path, the
    group-masking branch this checkpoint's config makes a no-op)."""
    scores_for_choice = scores.view(scores.shape[0], -1)
    scores_for_choice += bias.unsqueeze(0)  # in-place -- ALIASES scores
    tmp_scores = scores_for_choice
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)  # scores already mutated above
    return topk_idx, topk_weight


def _real_gate_fixed(scores, bias, top_k):
    """Verbatim from the real, bundled models/Kimi-K3/modeling_kimi_linear.py
    ::KimiMoEGate.forward (num_expert_group=1 path)."""
    scores = scores.view(scores.shape[0], -1)
    scores_for_choice = scores + bias.unsqueeze(0)  # fresh tensor
    tmp_scores = scores_for_choice
    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)  # scores was never mutated
    return topk_idx, topk_weight


def _tiny_cfg(model_type: str, hidden: int, num_experts: int, top_k: int) -> ModelConfig:
    return ModelConfig(
        model_type=model_type, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        vocab_size=100, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=hidden, eos_token_ids=(), torch_dtype="float32",
        num_experts=num_experts, num_experts_per_tok=top_k,
        norm_topk_prob=False, routed_scaling_factor=1.0,
    )


@_torch_skip
@pytest.mark.parametrize("model_type,real_ref", [
    ("kimi_k3", _real_gate_fixed),
    ("kimi_linear", _real_gate_aliasing_bug),
])
def test_route_experts_matches_real_reference_per_model_type(model_type, real_ref):
    rng = np.random.default_rng(0 if model_type == "kimi_k3" else 1)
    hidden, num_experts, top_k, n_tokens = 8, 12, 3, 5

    h_np = rng.standard_normal((1, n_tokens, hidden)).astype(np.float32)
    gate_weight_np = rng.standard_normal((num_experts, hidden)).astype(np.float32)
    bias_np = rng.standard_normal((num_experts,)).astype(np.float32)

    logits = h_np.reshape(n_tokens, hidden) @ gate_weight_np.T
    scores_np = 1.0 / (1.0 + np.exp(-logits))  # sigmoid, matching moe_router_activation_func

    ref_idx, ref_weight = real_ref(
        torch.from_numpy(scores_np.copy()), torch.from_numpy(bias_np), top_k)
    ref_idx_np = ref_idx.numpy()
    ref_weight_np = ref_weight.numpy()

    cfg = _tiny_cfg(model_type, hidden, num_experts, top_k)
    w = {
        "layer0.gate.weight": mx.array(gate_weight_np),
        "layer0.gate.e_score_correction_bias": mx.array(bias_np),
    }
    idx, pw = _route_experts(mx.array(h_np), w, "layer0", cfg)
    mx.eval(idx, pw)

    # argpartition/topk don't guarantee the SAME order for ties, but real
    # random float scores make ties astronomically unlikely -- compare as
    # sets of (expert, weight) pairs per token, sorted by expert id.
    for t in range(n_tokens):
        mine_pairs = sorted(zip(idx[0, t].tolist(), np.array(pw[0, t]).tolist()))
        ref_pairs = sorted(zip(ref_idx_np[t].tolist(), ref_weight_np[t].tolist()))
        mine_experts = [e for e, _ in mine_pairs]
        ref_experts = [e for e, _ in ref_pairs]
        assert mine_experts == ref_experts, (
            f"{model_type} token {t}: selected experts differ, "
            f"mine={mine_experts} ref={ref_experts}")
        for (_, mw), (_, rw) in zip(mine_pairs, ref_pairs):
            assert abs(mw - rw) < 1e-5, (
                f"{model_type} token {t}: routing weight differs, "
                f"mine={mw} ref={rw}")


@_torch_skip
def test_kimi_k3_and_kimi_linear_genuinely_differ_on_the_same_inputs():
    """Direct proof the two branches are not accidentally equivalent --
    same scores/bias/top_k, different routing WEIGHTS (selection may or
    may not coincide, but the aliasing bug measurably changes at least one
    weight whenever the bias is non-negligible relative to the score)."""
    rng = np.random.default_rng(2)
    hidden, num_experts, top_k, n_tokens = 8, 12, 3, 5
    h_np = rng.standard_normal((1, n_tokens, hidden)).astype(np.float32)
    gate_weight_np = rng.standard_normal((num_experts, hidden)).astype(np.float32)
    bias_np = (rng.standard_normal((num_experts,)).astype(np.float32) * 2.0)

    w = {
        "layer0.gate.weight": mx.array(gate_weight_np),
        "layer0.gate.e_score_correction_bias": mx.array(bias_np),
    }
    idx_k3, pw_k3 = _route_experts(
        mx.array(h_np), w, "layer0", _tiny_cfg("kimi_k3", hidden, num_experts, top_k))
    idx_kl, pw_kl = _route_experts(
        mx.array(h_np), w, "layer0", _tiny_cfg("kimi_linear", hidden, num_experts, top_k))
    mx.eval(idx_k3, pw_k3, idx_kl, pw_kl)

    max_diff = mx.max(mx.abs(pw_k3.astype(mx.float32) - pw_kl.astype(mx.float32)))
    mx.eval(max_diff)
    assert max_diff.item() > 1e-3, (
        "expected the fixed (kimi_k3) and buggy (kimi_linear) gates to "
        "produce genuinely different routing weights on inputs with a "
        "non-negligible bias -- if they match, the branch dispatch may "
        "not be wired correctly")
