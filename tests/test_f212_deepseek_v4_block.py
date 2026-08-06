"""F212: oracle for DeepSeek V4 block topology.

The eight subsystems before this were verified in isolation. Composition is
where ordering errors live, and the hyper-connection topology has three
orderings that a plausible rewrite gets wrong while remaining
shape-compatible:

* ``residual`` is captured BEFORE ``hc_pre``, so ``hc_post`` mixes the original
  four streams rather than the reduced tensor;
* the attention half uses the ``attn`` HC parameters and the FFN half the
  ``ffn`` ones -- separate learned projections of identical shape;
* the norm applies to the reduced tensor, after ``hc_pre``.

The reference is the checkpoint's own ``Block.forward``, with its ``attn`` and
``ffn`` submodules replaced by deterministic stubs. That isolates the topology
under test from sublayer arithmetic already covered by F206-F211, and means a
mismatch here can only be a composition error.
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

HC, DIM, SEQ = 4, 16, 3
ITERS, EPS, NORM_EPS = 20, 1e-6, 1e-6


@pytest.fixture(scope="module")
def reference():
    if "kernel" not in sys.modules:
        stub = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                     "sparse_attn"):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)

        def _sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20,
                      eps=1e-6):
            """Torch transcription; F205 already verified our MLX version."""
            import torch

            hc = hc_mult
            pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0]
                                + hc_base[:hc]) + eps
            post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1]
                                     + hc_base[hc:2 * hc])
            comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:])
            comb = comb.reshape(*comb.shape[:-1], hc, hc)
            comb = comb.softmax(dim=-1) + eps
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
            for _ in range(sinkhorn_iters - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
            return pre, post, comb
        stub.hc_split_sinkhorn = _sinkhorn
        sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference_module

    return reference_module


def _build_block(reference):
    """A real Block with random HC parameters and stubbed sublayers."""
    import torch

    torch.manual_seed(0)
    args = reference.ModelArgs()
    args.dim = DIM
    args.hc_mult = HC
    args.hc_sinkhorn_iters = ITERS
    args.hc_eps = EPS
    args.norm_eps = NORM_EPS

    block = object.__new__(reference.Block)
    torch.nn.Module.__init__(block)
    block.layer_id = 0
    block.norm_eps = NORM_EPS
    block.hc_mult = HC
    block.hc_sinkhorn_iters = ITERS
    block.hc_eps = EPS
    mix = (2 + HC) * HC
    for name, shape in (("hc_attn_fn", (mix, HC * DIM)),
                        ("hc_ffn_fn", (mix, HC * DIM)),
                        ("hc_attn_base", (mix,)), ("hc_ffn_base", (mix,)),
                        ("hc_attn_scale", (3,)), ("hc_ffn_scale", (3,))):
        setattr(block, name,
                torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32)
                                   * 0.3))
    block.attn_norm = reference.RMSNorm(DIM, NORM_EPS)
    block.ffn_norm = reference.RMSNorm(DIM, NORM_EPS)
    with torch.no_grad():
        block.attn_norm.weight.copy_(torch.randn(DIM) * 0.1 + 1.0)
        block.ffn_norm.weight.copy_(torch.randn(DIM) * 0.1 + 1.0)

    # Deterministic, order-sensitive stubs: distinct affine maps so swapping
    # the halves or the norms changes the result.
    torch.manual_seed(1)
    attn_matrix = torch.randn(DIM, DIM) * 0.2
    ffn_matrix = torch.randn(DIM, DIM) * 0.2
    block.attn = lambda t, *_a, **_k: t @ attn_matrix + 0.1
    block.ffn = lambda t, *_a, **_k: t @ ffn_matrix - 0.2
    return block, attn_matrix, ffn_matrix


def _mlx_pieces(block, attn_matrix, ffn_matrix):
    import mlx.core as mx

    hc = {name: mx.array(getattr(block, f"hc_{name}").detach().numpy())
          for name in ("attn_fn", "attn_scale", "attn_base",
                       "ffn_fn", "ffn_scale", "ffn_base")}
    norms = {"attn": mx.array(block.attn_norm.weight.detach().numpy()),
             "ffn": mx.array(block.ffn_norm.weight.detach().numpy())}
    a = mx.array(attn_matrix.numpy())
    f = mx.array(ffn_matrix.numpy())
    return hc, norms, (lambda t: t @ a + 0.1), (lambda t: t @ f - 0.2)


def test_block_topology_matches_the_released_forward(reference):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import run_deepseek_v4_block

    block, attn_matrix, ffn_matrix = _build_block(reference)
    rng = np.random.default_rng(3)
    x = (rng.normal(size=(1, SEQ, HC, DIM)) * 0.3).astype(np.float32)

    expected = block(torch.tensor(x), 0, None).detach().numpy()
    hc, norms, attention, ffn = _mlx_pieces(block, attn_matrix, ffn_matrix)
    got = np.array(run_deepseek_v4_block(
        mx.array(x), hc, norms, attention, ffn, hc_mult=HC,
        norm_eps=NORM_EPS, sinkhorn_iters=ITERS, hc_eps=EPS))

    assert got.shape == expected.shape
    diff = np.abs(got - expected).max()
    assert diff < 2e-4, f"block topology diverged, max abs diff {diff}"


def test_swapping_the_attn_and_ffn_hc_parameters_changes_the_output(reference):
    """They are shape-identical, so only a test can catch the swap."""
    import mlx.core as mx

    from runtime.deepseek_v4 import run_deepseek_v4_block

    block, attn_matrix, ffn_matrix = _build_block(reference)
    hc, norms, attention, ffn = _mlx_pieces(block, attn_matrix, ffn_matrix)
    rng = np.random.default_rng(4)
    x = mx.array((rng.normal(size=(1, SEQ, HC, DIM)) * 0.3).astype(np.float32))

    common = dict(hc_mult=HC, norm_eps=NORM_EPS, sinkhorn_iters=ITERS,
                  hc_eps=EPS)
    correct = np.array(run_deepseek_v4_block(
        x, hc, norms, attention, ffn, **common))
    swapped_hc = dict(hc)
    for suffix in ("fn", "scale", "base"):
        swapped_hc[f"attn_{suffix}"] = hc[f"ffn_{suffix}"]
        swapped_hc[f"ffn_{suffix}"] = hc[f"attn_{suffix}"]
    swapped = np.array(run_deepseek_v4_block(
        x, swapped_hc, norms, attention, ffn, **common))
    assert not np.allclose(correct, swapped, atol=1e-5), (
        "swapping the halves' HC parameters had no effect")


def test_residual_is_the_pre_reduction_stream(reference):
    """hc_post must mix the original streams, not the reduced tensor."""
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_post, hc_pre, run_deepseek_v4_block

    block, attn_matrix, ffn_matrix = _build_block(reference)
    hc, norms, attention, ffn = _mlx_pieces(block, attn_matrix, ffn_matrix)
    rng = np.random.default_rng(5)
    x = mx.array((rng.normal(size=(1, SEQ, HC, DIM)) * 0.3).astype(np.float32))
    common = dict(hc_mult=HC, norm_eps=NORM_EPS, sinkhorn_iters=ITERS,
                  eps=EPS)

    correct = np.array(run_deepseek_v4_block(
        x, hc, norms, attention, ffn, hc_mult=HC, norm_eps=NORM_EPS,
        sinkhorn_iters=ITERS, hc_eps=EPS))

    # Wrong variant: broadcast the reduced tensor as the residual.
    reduced, post, comb = hc_pre(
        x, hc["attn_fn"], hc["attn_scale"], hc["attn_base"], **common)
    normed = mx.fast.rms_norm(reduced, norms["attn"], NORM_EPS)
    wrong_residual = mx.broadcast_to(reduced[:, :, None, :], x.shape)
    wrong = np.array(hc_post(attention(normed), wrong_residual, post, comb))
    assert not np.allclose(correct[:, :, :, :], wrong, atol=1e-5), (
        "using the reduced tensor as the residual matched, so the test "
        "cannot detect that error")


def test_stream_shape_is_preserved_across_stacked_blocks(reference):
    """The hyper-connection stream never collapses between layers."""
    import mlx.core as mx

    from runtime.deepseek_v4 import run_deepseek_v4_block

    block, attn_matrix, ffn_matrix = _build_block(reference)
    hc, norms, attention, ffn = _mlx_pieces(block, attn_matrix, ffn_matrix)
    x = mx.array(np.random.default_rng(6).normal(
        size=(1, SEQ, HC, DIM)).astype(np.float32) * 0.2)
    for _ in range(3):
        x = run_deepseek_v4_block(
            x, hc, norms, attention, ffn, hc_mult=HC, norm_eps=NORM_EPS,
            sinkhorn_iters=ITERS, hc_eps=EPS)
        assert x.shape == (1, SEQ, HC, DIM)
    assert np.isfinite(np.array(x)).all()
