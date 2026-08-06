"""F217: oracle the composed MoE against the released MoE.forward.

F211 verified moe_gate, expert_swiglu and moe_combine individually. The
composition -- routing, batched paging, the shared expert, and the accumulation
across groups -- is what ``StreamingEngine._deepseek_v4_ffn`` does, and it has
never been checked end to end.

Attention is now oracled correct in both prefill (F216) and decode, and
compression is inert on short prompts, so this is the remaining candidate for
the post-first-token degeneration.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INFERENCE = ROOT / "models" / "DeepSeek-V4-Flash-0731" / "inference"

pytestmark = pytest.mark.skipif(
    not (INFERENCE / "model.py").is_file(),
    reason="DeepSeek-V4-Flash-0731 inference/ not present")

DIM, INTER, EXPERTS, TOPK = 32, 16, 8, 3


def _reference():
    stub = sys.modules.get("kernel") or types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        if not hasattr(stub, name):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
    sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference

    return reference


def _build(reference):
    import torch

    torch.manual_seed(0)
    args = reference.ModelArgs()
    args.dim, args.moe_inter_dim = DIM, INTER
    args.n_routed_experts, args.n_activated_experts = EXPERTS, TOPK
    args.n_shared_experts = 1
    args.n_hash_layers = 0
    args.swiglu_limit = 10.0
    args.score_func = "sqrtsoftplus"
    args.route_scale = 1.0
    with reference.set_dtype(torch.float32):
        moe = reference.MoE(0, args)
    with torch.no_grad():
        moe.gate.weight = torch.nn.Parameter(
            torch.randn(EXPERTS, DIM, dtype=torch.float32) * 0.4)
        moe.gate.bias = torch.nn.Parameter(
            torch.randn(EXPERTS, dtype=torch.float32) * 0.3)
        for expert in list(moe.experts) + [moe.shared_experts]:
            for layer in (expert.w1, expert.w2, expert.w3):
                layer.weight = torch.nn.Parameter(
                    torch.randn(*layer.weight.shape, dtype=torch.float32) * 0.2)
    return moe, args


def _mlx_ffn(moe, x, batch):
    """Mirror _deepseek_v4_ffn: route, fetch in groups, accumulate."""
    import mlx.core as mx

    from runtime.deepseek_v4 import expert_swiglu, moe_combine, moe_gate

    flat = x.reshape(-1, x.shape[-1])
    weights, indices = moe_gate(
        flat, mx.array(moe.gate.weight.detach().numpy()),
        mx.array(moe.gate.bias.detach().numpy()),
        topk=TOPK, score_func="sqrtsoftplus")

    def routed(expert, rows, scale):
        e = moe.experts[expert]
        return expert_swiglu(
            rows, mx.array(e.w1.weight.detach().numpy()),
            mx.array(e.w2.weight.detach().numpy()),
            mx.array(e.w3.weight.detach().numpy()),
            swiglu_limit=10.0, weights=scale)

    def shared(rows):
        e = moe.shared_experts
        return expert_swiglu(
            rows, mx.array(e.w1.weight.detach().numpy()),
            mx.array(e.w2.weight.detach().numpy()),
            mx.array(e.w3.weight.detach().numpy()), swiglu_limit=10.0)

    expert_ids = sorted({int(e) for row in indices.tolist() for e in row})
    out = shared(flat).astype(mx.float32)
    for start in range(0, len(expert_ids), batch):
        group = set(expert_ids[start:start + batch])
        out = out + moe_combine(
            flat[None], routed, weights, indices, None,
            n_routed_experts=EXPERTS,
            only_experts=group).reshape(flat.shape).astype(mx.float32)
    return np.array(out.reshape(x.shape))


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_composed_moe_matches_the_released_forward(batch):
    """Batched paging must not change the result, at any group size."""
    import mlx.core as mx
    import torch

    reference = _reference()
    moe, _args = _build(reference)

    rng = np.random.default_rng(2)
    x = (rng.normal(size=(1, 6, DIM)) * 0.4).astype(np.float32)
    ids = torch.zeros(1, 6, dtype=torch.long)
    expected = moe(torch.tensor(x), ids).detach().numpy()

    got = _mlx_ffn(moe, mx.array(x), batch)
    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1e-6)
    assert diff / scale < 5e-3, (
        f"batch={batch}: composed MoE diverged, max abs {diff}, "
        f"relative {diff/scale:.5f}")
