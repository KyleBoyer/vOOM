"""F207: oracle for DeepSeek V4's learned KV compression pooling.

Beyond the sliding window the model keeps a compressed KV: every
``compress_ratio`` consecutive positions are pooled into one entry by a learned
softmax gate, with an absolute position embedding (``ape``) added to the gate
scores. At ratio 4 the windows overlap, so each pooled entry also sees the
previous group's overlap half.

The reference is the checkpoint's own ``Compressor`` module, instantiated with
random weights and run in torch. Only its TileLang imports are stubbed; the
pooling arithmetic under test is the released code itself.

RoPE is out of scope here by construction -- ``compress_prefill`` deliberately
does not apply it, because a compressed entry stands for original position
``j * ratio`` under a different (YaRN) theta than the sliding-window layers
use. A consecutive-position RoPE folded in here would still produce plausible
activations, so the split is enforced rather than trusted.
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


def _install_kernel_stub(act_quant_identity=True):
    """Import the released model.py without its TileLang kernels.

    ``act_quant`` is the one entry given a real (identity) behaviour: it is a
    QAT round-trip that the pooling test deliberately isolates from. Every
    other entry raises, so a test that reaches genuine kernel code fails
    loudly instead of silently using a placeholder.
    """
    if "kernel" in sys.modules:
        return
    stub = types.ModuleType("kernel")
    for name in ("fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn",
                 "hc_split_sinkhorn"):
        def _unavailable(*_a, __name=name, **_k):
            raise RuntimeError(f"kernel {__name!r} unavailable in this test")
        setattr(stub, name, _unavailable)

    def _act_quant(x, *_a, **_k):
        return x  # isolate pooling from the QAT round-trip
    stub.act_quant = _act_quant if act_quant_identity else _unavailable
    sys.modules["kernel"] = stub


@pytest.fixture(scope="module")
def reference_module():
    _install_kernel_stub()
    sys.path.insert(0, str(INFERENCE))
    import model as reference

    return reference


def _build(reference, ratio, dim=64, head_dim=32, seed=0):
    import torch

    torch.manual_seed(seed)
    args = reference.ModelArgs()
    args.dim = dim
    args.rope_head_dim = 8
    args.norm_eps = 1e-6
    args.max_batch_size = 1
    compressor = reference.Compressor(args, compress_ratio=ratio,
                                      head_dim=head_dim)
    coff = 1 + (ratio == 4)
    with torch.no_grad():
        compressor.ape.copy_(torch.randn(ratio, coff * head_dim) * 0.5)
        compressor.wkv.weight.copy_(
            torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.1)
        compressor.wgate.weight.copy_(
            torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.1)
        compressor.norm.weight.copy_(torch.randn(head_dim) * 0.1 + 1.0)
    return compressor, args


def _reference_pool(compressor, x, ratio, head_dim):
    """Run only the released pooling half of Compressor.forward (start_pos=0).

    Transcribed from model.py:322-370 up to the norm, stopping before RoPE and
    the cache write -- the parts compress_prefill deliberately excludes.
    """
    import torch

    x = x.float()
    kv = compressor.wkv(x)
    score = compressor.wgate(x)
    seqlen = x.size(1)
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if remainder > 0:
        kv = kv[:, :cutoff]
        score = score[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + compressor.ape
    if compressor.overlap:
        kv = compressor.overlap_transform(kv, 0)
        score = compressor.overlap_transform(score, float("-inf"))
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    return compressor.norm(pooled.to(torch.float32))


@pytest.mark.parametrize("ratio,seqlen", [(4, 16), (4, 18), (128, 256),
                                          (128, 300)])
def test_pooling_matches_the_released_compressor(reference_module, ratio,
                                                 seqlen):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import compress_prefill

    head_dim, dim = 32, 64
    compressor, args = _build(reference_module, ratio, dim, head_dim)
    x = torch.randn(1, seqlen, dim, dtype=torch.float32) * 0.4

    expected = _reference_pool(compressor, x, ratio, head_dim)
    got, remainder = compress_prefill(
        mx.array(x.numpy()),
        mx.array(compressor.wkv.weight.detach().numpy()),
        mx.array(compressor.wgate.weight.detach().numpy()),
        mx.array(compressor.ape.detach().numpy()),
        mx.array(compressor.norm.weight.detach().numpy()),
        ratio=ratio, head_dim=head_dim, norm_eps=args.norm_eps)

    assert remainder == seqlen % ratio
    assert got.shape == tuple(expected.shape), (
        f"{got.shape} != {tuple(expected.shape)}")
    diff = np.abs(np.array(got) - expected.detach().numpy()).max()
    assert diff < 2e-4, f"pooling diverged, max abs diff {diff}"


def test_overlap_transform_matches_the_released_one(reference_module):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import compressor_overlap_transform

    ratio, head_dim, groups = 4, 32, 5
    compressor, _ = _build(reference_module, ratio, head_dim=head_dim)
    tensor = torch.randn(1, groups, ratio, 2 * head_dim)

    for fill in (0.0, float("-inf")):
        expected = compressor.overlap_transform(tensor, fill).numpy()
        got = np.array(compressor_overlap_transform(
            mx.array(tensor.numpy()), head_dim, ratio, fill))
        # -inf must land in exactly the same slots.
        assert np.array_equal(np.isinf(got), np.isinf(expected))
        finite = ~np.isinf(expected)
        assert np.allclose(got[finite], expected[finite], atol=1e-6), fill


def test_group_zero_has_no_predecessor_overlap(reference_module):
    """The first group's overlap half must keep the fill, not wrap around."""
    import mlx.core as mx

    from runtime.deepseek_v4 import compressor_overlap_transform

    ratio, head_dim = 4, 8
    tensor = mx.ones((1, 3, ratio, 2 * head_dim))
    got = np.array(compressor_overlap_transform(tensor, head_dim, ratio, 0.0))
    assert np.all(got[0, 0, :ratio] == 0.0), (
        "group 0 pulled an overlap half from nowhere")
    assert np.all(got[0, 1, :ratio] == 1.0)


def test_short_prompt_below_one_group_compresses_nothing(reference_module):
    import mlx.core as mx

    from runtime.deepseek_v4 import compress_prefill

    ratio, head_dim, dim = 4, 32, 64
    compressor, args = _build(reference_module, ratio, dim, head_dim)
    got, remainder = compress_prefill(
        mx.zeros((1, 3, dim)),
        mx.array(compressor.wkv.weight.detach().numpy()),
        mx.array(compressor.wgate.weight.detach().numpy()),
        mx.array(compressor.ape.detach().numpy()),
        mx.array(compressor.norm.weight.detach().numpy()),
        ratio=ratio, head_dim=head_dim, norm_eps=args.norm_eps)
    assert got is None and remainder == 3, (
        "a prompt shorter than one group must report all positions leftover")


def test_gate_softmax_is_over_the_group_axis(reference_module):
    """Softmaxing features instead of group members is the classic slip."""
    import mlx.core as mx

    from runtime.deepseek_v4 import compress_prefill

    ratio, head_dim, dim = 128, 32, 64
    compressor, args = _build(reference_module, ratio, dim, head_dim, seed=3)
    x = mx.array(
        (np.random.default_rng(0).normal(size=(1, ratio, dim)) * 0.4
         ).astype(np.float32))
    pooled, _ = compress_prefill(
        x,
        mx.array(compressor.wkv.weight.detach().numpy()),
        mx.array(compressor.wgate.weight.detach().numpy()),
        mx.array(compressor.ape.detach().numpy()),
        mx.array(compressor.norm.weight.detach().numpy()),
        ratio=ratio, head_dim=head_dim, norm_eps=args.norm_eps)
    # One group in, one entry out.
    assert pooled.shape == (1, 1, head_dim)
