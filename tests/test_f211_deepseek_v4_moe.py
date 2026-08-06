"""F211: oracle for DeepSeek V4's MoE routing and experts.

References are the checkpoint's own ``Gate`` and ``Expert`` modules run in
torch. Only the TileLang imports are stubbed; the routing and SwiGLU
arithmetic under test is the released code.

Three details a plausible port gets wrong, all pinned:

* the released default ``score_func`` is ``sqrtsoftplus``, not softmax, and
  only the softmax branch skips the weight renormalization -- assuming softmax
  therefore breaks twice;
* the gate ``bias`` shifts scores for *selection* only, never for the returned
  weights;
* ``swiglu_limit`` clamps the up branch on both sides but the gate branch only
  from above.
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


@pytest.fixture(scope="module")
def reference():
    if "kernel" not in sys.modules:
        stub = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                     "sparse_attn", "hc_split_sinkhorn"):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
        sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference_module

    return reference_module


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "sigmoid", "softmax"])
def test_gate_matches_the_released_module(reference, score_func):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import moe_gate

    torch.manual_seed(0)
    args = reference.ModelArgs()
    args.dim = 32
    args.n_routed_experts = 16
    args.n_activated_experts = 4
    args.score_func = score_func
    args.route_scale = 2.5
    args.n_hash_layers = 0
    gate = reference.Gate(0, args)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(16, 32) * 0.3)
        gate.bias.copy_(torch.randn(16) * 0.2)

    x = torch.randn(5, 32) * 0.5
    ref_weights, ref_indices = gate(x)

    got_weights, got_indices = moe_gate(
        mx.array(x.numpy()), mx.array(gate.weight.detach().numpy()),
        mx.array(gate.bias.detach().numpy()), topk=4,
        score_func=score_func, route_scale=2.5)

    # top-k order is unspecified between implementations; compare as sets and
    # match the weights back to their expert ids.
    ref_map = [dict(zip(i.tolist(), w.tolist()))
               for i, w in zip(ref_indices, ref_weights)]
    got_map = [dict(zip(i.tolist(), w.tolist()))
               for i, w in zip(np.array(got_indices), np.array(got_weights))]
    for row, (expected, actual) in enumerate(zip(ref_map, got_map)):
        assert set(expected) == set(actual), (
            f"row {row}: selected {sorted(actual)} != {sorted(expected)}")
        for expert, value in expected.items():
            assert abs(actual[expert] - value) < 1e-4, (
                f"row {row} expert {expert}: {actual[expert]} != {value}")


def test_bias_changes_selection_but_not_the_returned_weights(reference):
    """Folding the bias into the weights is a silent quality regression."""
    import mlx.core as mx

    from runtime.deepseek_v4 import moe_gate

    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(6, 16)).astype(np.float32))
    weight = mx.array(rng.normal(size=(8, 16)).astype(np.float32) * 0.4)
    bias = mx.array((rng.normal(size=(8,)) * 3.0).astype(np.float32))

    unbiased_w, unbiased_i = moe_gate(x, weight, None, topk=3)
    biased_w, biased_i = moe_gate(x, weight, bias, topk=3)

    assert not np.array_equal(np.array(unbiased_i), np.array(biased_i)), (
        "the bias did not change selection, so this test proves nothing")
    # For any expert selected in BOTH cases, the weight must be identical.
    for row in range(x.shape[0]):
        shared = (set(np.array(unbiased_i)[row].tolist())
                  & set(np.array(biased_i)[row].tolist()))
        for expert in shared:
            a = float(np.array(unbiased_w)[row][
                list(np.array(unbiased_i)[row]).index(expert)])
            b = float(np.array(biased_w)[row][
                list(np.array(biased_i)[row]).index(expert)])
            # Renormalization differs when the selected set differs, so only
            # compare rows whose whole selection matched.
            if set(np.array(unbiased_i)[row].tolist()) == set(
                    np.array(biased_i)[row].tolist()):
                assert abs(a - b) < 1e-5, (
                    f"row {row} expert {expert}: bias leaked into the weight")


def test_expert_matches_the_released_module(reference):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import expert_swiglu

    torch.manual_seed(2)
    # The released Linear defaults to bfloat16; build in float32 so the
    # comparison isolates the SwiGLU arithmetic from bf16 rounding.
    expert = reference.Expert(24, 48, dtype=torch.float32, swiglu_limit=10.0)
    with torch.no_grad():
        for layer in (expert.w1, expert.w2, expert.w3):
            layer.weight.copy_(
                torch.randn(*layer.weight.shape, dtype=torch.float32) * 0.2)

    x = (torch.randn(4, 24) * 3.0).to(torch.float32)
    expected = expert(x).detach().numpy()
    got = np.array(expert_swiglu(
        mx.array(x.numpy()),
        mx.array(expert.w1.weight.detach().numpy()),
        mx.array(expert.w2.weight.detach().numpy()),
        mx.array(expert.w3.weight.detach().numpy()),
        swiglu_limit=10.0))
    assert np.allclose(got, expected, atol=1e-4), (
        f"max abs diff {np.abs(got - expected).max()}")


def test_swiglu_clamp_is_asymmetric_on_the_gate_branch():
    """Gate clamps only from above; symmetric clamping is a different function."""
    import mlx.core as mx

    from runtime.deepseek_v4 import expert_swiglu

    dim, inter = 8, 8
    identity = mx.eye(dim)
    # Drive the gate strongly negative and the up branch strongly positive.
    x = mx.full((1, dim), -50.0)
    out = np.array(expert_swiglu(x, identity, identity, -identity,
                                 swiglu_limit=10.0))
    assert np.isfinite(out).all()
    # If the gate were clamped from below at -10, silu(-10) ~= -4.5e-4; with no
    # lower clamp silu(-50) ~= -9.6e-21. The outputs differ by many orders.
    clamped = np.array(expert_swiglu(mx.full((1, dim), -10.0), identity,
                                     identity, -identity, swiglu_limit=10.0))
    assert np.abs(out).max() < np.abs(clamped).max() * 1e-6, (
        "the gate appears to be clamped from below, which it must not be")


def test_moe_combine_sums_routed_and_shared(reference):
    import mlx.core as mx

    from runtime.deepseek_v4 import moe_combine

    x = mx.ones((1, 3, 4))
    weights = mx.array([[0.5, 0.5]] * 3)
    indices = mx.array([[0, 1], [1, 2], [0, 2]])

    seen = []

    def routed(expert, rows, scale):
        seen.append(expert)
        return mx.ones(rows.shape) * float(expert + 1) * scale

    def shared(rows):
        return mx.ones(rows.shape) * 100.0

    out = np.array(moe_combine(x, routed, weights, indices, shared,
                               n_routed_experts=3))
    assert sorted(seen) == [0, 1, 2], "every selected expert must be visited"
    # row 0: experts 0,1 -> 0.5*1 + 0.5*2 = 1.5, plus shared 100
    assert np.allclose(out[0, 0], 101.5), out[0, 0]
    # row 1: experts 1,2 -> 0.5*2 + 0.5*3 = 2.5
    assert np.allclose(out[0, 1], 102.5), out[0, 1]


def test_moe_combine_never_visits_an_unselected_expert():
    """Paging depends on this: an unselected expert must not be fetched."""
    import mlx.core as mx

    from runtime.deepseek_v4 import moe_combine

    visited = []

    def routed(expert, rows, scale):
        visited.append(expert)
        return mx.zeros(rows.shape)

    moe_combine(mx.ones((1, 1, 4)), routed, mx.array([[1.0]]),
                mx.array([[7]]), None, n_routed_experts=256)
    assert visited == [7], f"visited {visited} instead of only expert 7"


def test_moe_combine_rejects_an_out_of_range_expert():
    import mlx.core as mx

    from runtime.deepseek_v4 import moe_combine

    with pytest.raises(ValueError, match="out of range"):
        moe_combine(mx.ones((1, 1, 4)), lambda *_: mx.zeros((1, 4)),
                    mx.array([[1.0]]), mx.array([[9]]), None,
                    n_routed_experts=4)
