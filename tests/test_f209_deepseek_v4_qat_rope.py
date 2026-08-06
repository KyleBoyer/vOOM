"""F209: oracle for DeepSeek V4's activation QAT and rotary embedding.

Both references here are pure torch in the released checkpoint
(``apply_rotary_emb`` and ``precompute_freqs_cis`` in model.py, and the
documented arithmetic of ``act_quant_kernel_`` in kernel.py), so they are used
directly rather than approximated.

The detail that silently breaks a port: ``apply_rotary_emb`` views the last
axis as complex via ``unflatten(-1, (-1, 2))``, so it rotates *interleaved*
adjacent pairs. Most of this runtime -- including the LFM2 port -- uses the
half-split convention. Substituting one for the other preserves every shape
and every vector norm while rotating the wrong element pairs, so it is
checked against the released function rather than reasoned about.
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
has_reference = (INFERENCE / "model.py").is_file()


def _reference():
    if "kernel" in sys.modules:
        pass
    else:
        stub = types.ModuleType("kernel")
        for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                     "sparse_attn", "hc_split_sinkhorn"):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
        sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference

    return reference


# ---- activation QAT --------------------------------------------------------


def reference_act_quant(x, block_size):
    """Arithmetic of act_quant_kernel_ with inplace=True, round_scale=False."""
    import ml_dtypes

    n = x.shape[-1]
    grouped = x.reshape(*x.shape[:-1], n // block_size, block_size)
    amax = np.maximum(np.abs(grouped).max(axis=-1, keepdims=True), 1e-4)
    scale = amax / 448.0
    clamped = np.clip(grouped / scale, -448.0, 448.0)
    roundtrip = clamped.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    return (roundtrip * scale).reshape(x.shape)


@pytest.mark.parametrize("block_size", [64, 128])
def test_act_quant_matches_the_released_arithmetic(block_size):
    import mlx.core as mx

    from runtime.deepseek_v4 import act_quant_simulate

    rng = np.random.default_rng(0)
    x = (rng.normal(size=(3, 4 * block_size)) * 2.0).astype(np.float32)
    expected = reference_act_quant(x, block_size)
    got = np.array(act_quant_simulate(mx.array(x), block_size))
    assert np.allclose(got, expected, atol=1e-5), (
        f"max abs diff {np.abs(got - expected).max()}")


def test_act_quant_is_lossy_but_bounded():
    """It must actually quantize -- an identity would silently pass elsewhere."""
    import mlx.core as mx

    from runtime.deepseek_v4 import act_quant_simulate

    rng = np.random.default_rng(1)
    x = (rng.normal(size=(2, 256)) * 2.0).astype(np.float32)
    got = np.array(act_quant_simulate(mx.array(x), 128))
    assert not np.allclose(got, x, atol=1e-6), "act_quant behaved as identity"
    # E4M3 has ~2 decimal digits; per-block relative error stays small.
    relative = np.abs(got - x).max() / np.abs(x).max()
    assert relative < 0.1, relative


def test_all_zero_block_survives_the_amax_floor():
    """Without the 1e-4 floor an empty block divides by zero and yields NaN."""
    import mlx.core as mx

    from runtime.deepseek_v4 import act_quant_simulate

    got = np.array(act_quant_simulate(mx.zeros((1, 128)), 128))
    assert np.isfinite(got).all()
    assert np.allclose(got, 0.0)


def test_act_quant_rejects_a_misaligned_axis():
    import mlx.core as mx

    from runtime.deepseek_v4 import act_quant_simulate

    with pytest.raises(ValueError, match="divisible"):
        act_quant_simulate(mx.zeros((1, 100)), 128)


# ---- rotary embedding ------------------------------------------------------


@pytest.mark.skipif(not has_reference, reason="checkpoint inference/ absent")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("ndim", [3, 4])
def test_rope_matches_the_released_apply_rotary_emb(inverse, ndim):
    import mlx.core as mx
    import torch

    from runtime.deepseek_v4 import apply_rope_interleaved

    reference = _reference()
    seqlen, dim = 6, 16
    rng = np.random.default_rng(2)
    shape = (1, seqlen, dim) if ndim == 3 else (1, seqlen, 3, dim)
    x = rng.normal(size=shape).astype(np.float32)

    freqs = reference.precompute_freqs_cis(
        dim, seqlen, 0, 10000.0, 1.0, 32, 1)[:seqlen]
    expected = reference.apply_rotary_emb(
        torch.tensor(x.copy()), freqs, inverse).numpy()

    angles = np.angle(freqs.numpy())
    got = np.array(apply_rope_interleaved(
        mx.array(x), mx.array(np.cos(angles)), mx.array(np.sin(angles)),
        inverse=inverse))
    assert np.allclose(got, expected, atol=1e-4), (
        f"ndim={ndim} inverse={inverse}: max abs diff "
        f"{np.abs(got - expected).max()}")


def test_inverse_rope_undoes_the_forward_rotation():
    """The attention epilogue relies on this exactly."""
    import mlx.core as mx

    from runtime.deepseek_v4 import apply_rope_interleaved, yarn_freqs

    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(size=(1, 5, 2, 16)).astype(np.float32))
    cos, sin = yarn_freqs(16, 5, 0, 10000.0, 1.0, 32, 1)
    forward = apply_rope_interleaved(x, cos, sin)
    back = apply_rope_interleaved(forward, cos, sin, inverse=True)
    assert np.allclose(np.array(back), np.array(x), atol=1e-5)


def test_interleaved_differs_from_the_half_split_convention():
    """Guard the convention: the wrong one preserves shapes and norms."""
    import mlx.core as mx

    from runtime.deepseek_v4 import apply_rope_interleaved, yarn_freqs

    rng = np.random.default_rng(4)
    x = mx.array(rng.normal(size=(1, 4, 16)).astype(np.float32))
    cos, sin = yarn_freqs(16, 4, 0, 10000.0, 1.0, 32, 1)
    interleaved = np.array(apply_rope_interleaved(x, cos, sin))
    half_split = np.array(mx.fast.rope(
        x[:, :, None], 16, traditional=False, base=10000.0, scale=1.0,
        offset=0)[:, :, 0])
    assert not np.allclose(interleaved, half_split, atol=1e-4), (
        "the two RoPE conventions agree here, so this test cannot detect "
        "substituting one for the other")
    assert np.allclose(np.linalg.norm(interleaved, axis=-1),
                       np.linalg.norm(half_split, axis=-1), atol=1e-4), (
        "norms should match even though the rotations differ -- which is "
        "exactly why the mistake is hard to see")


@pytest.mark.skipif(not has_reference, reason="checkpoint inference/ absent")
def test_yarn_frequencies_match_the_released_precompute():
    import mlx.core as mx

    from runtime.deepseek_v4 import yarn_freqs

    reference = _reference()
    dim, seqlen = 64, 4096
    original, base, factor = 512, 160000.0, 16.0
    expected = reference.precompute_freqs_cis(
        dim, seqlen, original, base, factor, 32, 1).numpy()
    cos, sin = yarn_freqs(dim, seqlen, original, base, factor, 32, 1)
    got = np.array(cos) + 1j * np.array(sin)
    assert np.allclose(got, expected, atol=1e-4), (
        f"max abs diff {np.abs(got - expected).max()}")
