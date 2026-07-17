"""F33 harness, milestone 1: verify the HF <-> runtime MoE weight-layout
conversion is correct, using a tiny synthetic GlmMoeDsaForCausalLM (no NAS, no
real GLM-5.2 weights, no BRIEF 0 gate needed).

Investigation (2026-07-14): instantiating a tiny real `GlmMoeDsaConfig` /
`GlmMoeDsaForCausalLM` (transformers==5.13.0, installed this session) and
inspecting its actual `state_dict()` showed attention (MLA), the DSA indexer,
the shared expert, and the router are all named IDENTICALLY to what
runtime/glm.py and runtime/glm_dsa.py already expect
(`self_attn.q_a_proj`, `self_attn.indexer.wq_b`, `mlp.shared_experts.*`,
`mlp.gate.weight`/`e_score_correction_bias`) -- direct compatibility, no
conversion needed there.

The ONE real mismatch is the ROUTED experts: HF's `GlmMoeDsaExperts` stores
them as two batched 3D tensors, `gate_up_proj` (n_experts, 2*inter, hidden)
and `down_proj` (n_experts, hidden, inter) -- confirmed against the actual
`GlmMoeDsaExperts.forward()` source: `F.linear(x, gate_up_proj[e]).chunk(2,
dim=-1)` gives (gate, up) from the FIRST/SECOND half of the 2*inter output
rows, and `F.linear(y, down_proj[e])` -- both already in this runtime's
`x @ w.T` / (out, in) convention per-expert, just batched-by-expert and with
gate+up fused. runtime/glm.py's `mlp.experts.{e}.gate_proj/up_proj/down_proj`
wants them UNFUSED and per-expert. This test verifies the exact,
now-precisely-known slicing: `gate_proj = gate_up_proj[e][:inter]`,
`up_proj = gate_up_proj[e][inter:]`, `down_proj = down_proj[e]` (no
transpose needed for any of the three).

Also confirmed against the real GLM-5.2 config.json (tiny file read, not a
weight/NAS-heavy operation): `n_group=1, topk_group=1`, so HF's
group-restricted top-k router is a no-op for this model -- runtime/glm.py's
simpler flat top-k (no grouping) is mathematically equivalent for GLM-5.2
specifically, not a missing feature.

This is milestone 1 only (isolated MoE block: router + shared + routed
experts, given an already-computed hidden state). It does NOT yet cover
attention/MLA/DSA end-to-end, which needs KV-cache scaffolding to test in
isolation -- a reasonable milestone 2, not attempted here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.layer_runner import _swiglu

transformers = pytest.importorskip("transformers")
from transformers import GlmMoeDsaConfig  # noqa: E402
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaMoE  # noqa: E402

HIDDEN = 48
INTER = 16
N_EXPERTS = 8
TOP_K = 2


def _tiny_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=64,
        moe_intermediate_size=INTER, num_hidden_layers=2, first_k_dense_replace=1,
        num_attention_heads=4, n_shared_experts=1, n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K, kv_lora_rank=16, q_lora_rank=32,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, index_topk=8,
    )


def _convert_expert_weights(sd: dict, prefix: str) -> dict:
    """The verified HF -> runtime conversion for one MoE block's routed
    experts, plus direct passthrough for the router and shared expert."""
    w: dict[str, mx.array] = {}
    w[f"{prefix}.gate.weight"] = mx.array(sd["gate.weight"].numpy())
    w[f"{prefix}.gate.e_score_correction_bias"] = mx.array(sd["gate.e_score_correction_bias"].numpy())
    for sub in ("gate_proj", "up_proj", "down_proj"):
        w[f"{prefix}.shared_experts.{sub}.weight"] = mx.array(sd[f"shared_experts.{sub}.weight"].numpy())

    gate_up = sd["experts.gate_up_proj"]  # (E, 2*inter, hidden)
    down = sd["experts.down_proj"]  # (E, hidden, inter)
    inter = gate_up.shape[1] // 2
    for e in range(gate_up.shape[0]):
        w[f"{prefix}.experts.{e}.gate_proj.weight"] = mx.array(gate_up[e, :inter, :].numpy())
        w[f"{prefix}.experts.{e}.up_proj.weight"] = mx.array(gate_up[e, inter:, :].numpy())
        w[f"{prefix}.experts.{e}.down_proj.weight"] = mx.array(down[e].numpy())
    return w


def _runtime_moe_forward(h: mx.array, w: dict, prefix: str, num_experts_per_tok: int,
                          norm_topk_prob: bool, routed_scaling_factor: float) -> mx.array:
    """Mirrors runtime/glm.py:156-176's MoE branch (router select + shared +
    weighted routed experts) exactly, given weights already in this
    runtime's flat-dict format. Not calling run_glm_block directly here
    since that also requires attention/KV-cache scaffolding out of scope for
    this milestone."""
    scores = mx.sigmoid(
        h.astype(mx.float32) @ w[f"{prefix}.gate.weight"].astype(mx.float32).T
    )
    biased = scores + w[f"{prefix}.gate.e_score_correction_bias"]
    k = num_experts_per_tok
    idx = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    pw = mx.take_along_axis(scores, idx, axis=-1)
    if norm_topk_prob:
        pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * routed_scaling_factor
    mx.eval(idx, pw)

    L = h.shape[1]
    groups: dict[int, list[tuple[int, float]]] = {}
    for pos in range(L):
        for j in range(k):
            groups.setdefault(int(idx[0, pos, j]), []).append((pos, float(pw[0, pos, j])))

    out = mx.zeros_like(h)
    for e in sorted(groups):
        plist = groups[e]
        positions = [p for p, _ in plist]
        weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
        y = _swiglu(h[:, positions, :], w, f"{prefix}.experts.{e}")
        contribution = (y * weights[None, :, None]).astype(h.dtype)
        out = out.at[:, positions, :].add(contribution)
        mx.eval(out)
    return out + _swiglu(h, w, f"{prefix}.shared_experts")


def test_hf_and_runtime_moe_outputs_match():
    torch.manual_seed(0)
    cfg = _tiny_config()
    moe = GlmMoeDsaMoE(cfg)
    moe.eval()
    # Random-but-reasonable weights (default init), not zeros, so a layout
    # bug (wrong slice/transpose) would show up as a real numeric mismatch.
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(mean=0.0, std=0.5)
        moe.gate.e_score_correction_bias.normal_(mean=0.0, std=0.3)

    torch.manual_seed(1)
    h_torch = torch.randn(1, 5, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        y_hf = moe(h_torch)

    sd = moe.state_dict()
    w = _convert_expert_weights(sd, "mlp")
    h_mx = mx.array(h_torch.numpy())
    y_runtime = _runtime_moe_forward(
        h_mx, w, "mlp", num_experts_per_tok=TOP_K,
        norm_topk_prob=cfg.norm_topk_prob, routed_scaling_factor=cfg.routed_scaling_factor,
    )
    mx.eval(y_runtime)

    y_hf_np = y_hf.detach().numpy()
    y_runtime_np = np.array(y_runtime)
    assert y_hf_np.shape == y_runtime_np.shape
    max_abs_diff = np.max(np.abs(y_hf_np - y_runtime_np))
    assert max_abs_diff < 1e-4, f"HF vs runtime MoE output mismatch: max abs diff {max_abs_diff}"


def test_router_top_k_selection_matches():
    """Isolates just the routing decision (which experts, what weights) --
    if this passes but the full test above fails, the bug is in the expert
    compute/conversion, not the router."""
    torch.manual_seed(2)
    cfg = _tiny_config()
    moe = GlmMoeDsaMoE(cfg)
    moe.eval()
    with torch.no_grad():
        moe.gate.weight.normal_(mean=0.0, std=0.5)
        moe.gate.e_score_correction_bias.normal_(mean=0.0, std=0.3)

    torch.manual_seed(3)
    h_torch = torch.randn(1, 4, HIDDEN, dtype=torch.float32)
    with torch.no_grad():
        _, hf_weights, hf_indices = moe.gate(h_torch)

    w = {
        "gate.weight": mx.array(moe.gate.weight.detach().numpy()),
        "gate.e_score_correction_bias": mx.array(moe.gate.e_score_correction_bias.detach().numpy()),
    }
    h_mx = mx.array(h_torch.numpy())
    scores = mx.sigmoid(
        h_mx.astype(mx.float32) @ w["gate.weight"].astype(mx.float32).T
    )
    biased = scores + w["gate.e_score_correction_bias"]
    idx = mx.argpartition(-biased, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
    pw = mx.take_along_axis(scores, idx, axis=-1)
    pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * cfg.routed_scaling_factor
    mx.eval(idx, pw)

    for pos in range(h_torch.shape[1]):
        hf_set = set(int(i) for i in hf_indices[pos].tolist())
        our_set = set(int(i) for i in idx[0, pos].tolist())
        assert hf_set == our_set, f"position {pos}: HF selected {hf_set}, runtime selected {our_set}"
        hf_w_by_idx = {int(i): float(wt) for i, wt in zip(hf_indices[pos].tolist(), hf_weights[pos].tolist())}
        our_w_by_idx = {int(i): float(wt) for i, wt in zip(idx[0, pos].tolist(), pw[0, pos].tolist())}
        for e in hf_set:
            assert abs(hf_w_by_idx[e] - our_w_by_idx[e]) < 1e-5, (
                f"position {pos} expert {e}: HF weight {hf_w_by_idx[e]} vs runtime {our_w_by_idx[e]}"
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
