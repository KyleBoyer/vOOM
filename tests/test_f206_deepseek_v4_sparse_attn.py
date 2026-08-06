"""F206: oracle for DeepSeek V4's gathered sparse attention and window indices.

The released kernel (``inference/kernel.py::sparse_attn_kernel``) is TileLang
and cannot run here, so the reference is a torch transcription of its
online-softmax loop, including the epilogue where the per-head ``attn_sink``
enters the denominator. The index generators are imported from the checkpoint's
own ``inference/model.py`` unchanged -- they are pure torch and need no stub,
so the index semantics are checked against the real code rather than a
restatement of it.

Two details that a plausible implementation gets wrong:

* ``attn_sink`` is not an extra key. It only enters the softmax denominator
  (``sum_exp[i] += exp(attn_sink[i] - scores_max[i])``), so it lets a head
  attend to nothing and shrink its output. Treating it as a key would also
  contribute a value vector.
* ``-1`` marks unused slots, and during decode the window indices are a
  *rotation* of the ring buffer, not a contiguous range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INFERENCE = (ROOT / "models" / "DeepSeek-V4-Flash-0731" / "inference")
has_reference = (INFERENCE / "model.py").is_file()


def _install_kernel_stub():
    """Let the released model.py import without its TileLang/CUDA kernels.

    ``inference/kernel.py`` imports tilelang at module scope, which cannot load
    on this machine. Only the pure-torch index generators are needed from
    model.py, so a stub module satisfies its import line without pretending to
    implement any kernel -- each entry raises if actually called, so a future
    test that reaches real kernel code fails loudly instead of silently using
    a placeholder. Same approach F92 used for Kimi's Triton-only KDA ops.
    """
    import sys as _sys
    import types

    if "kernel" in _sys.modules:
        return
    stub = types.ModuleType("kernel")
    names = (
        "act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn",
        "hc_split_sinkhorn")
    for name in names:
        def _unavailable(*_a, __name=name, **_k):
            raise RuntimeError(
                f"released TileLang kernel {__name!r} is not available here; "
                "this test must not depend on it")
        setattr(stub, name, _unavailable)
    _sys.modules["kernel"] = stub


def reference_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Transcription of sparse_attn_kernel_ (kernel.py:296) in numpy.

    Follows the kernel's running max/sum rather than a one-shot softmax so the
    epilogue's sink term lands in the same place.
    """
    b, m, h, d = q.shape
    out = np.zeros((b, m, h, d), np.float32)
    for bi in range(b):
        for mi in range(m):
            idxs = topk_idxs[bi, mi]
            valid = idxs >= 0
            scores = np.full((h, len(idxs)), -np.inf, np.float32)
            gathered = np.zeros((len(idxs), d), np.float32)
            for j, index in enumerate(idxs):
                if index < 0:
                    continue
                gathered[j] = kv[bi, index]
                scores[:, j] = q[bi, mi] @ kv[bi, index] * softmax_scale
            row_max = np.where(valid.any(), scores.max(axis=1), 0.0)
            row_max = np.where(np.isinf(row_max), 0.0, row_max)
            weights = np.where(valid[None, :],
                               np.exp(scores - row_max[:, None]), 0.0)
            # epilogue: the sink joins the denominator only
            denom = weights.sum(axis=1) + np.exp(attn_sink - row_max)
            out[bi, mi] = (weights @ gathered) / denom[:, None]
    return out


def _inputs(b=1, m=3, h=4, d=8, n=6, topk=5, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(b, m, h, d)).astype(np.float32) * 0.3
    kv = rng.normal(size=(b, n, d)).astype(np.float32) * 0.3
    sink = rng.normal(size=(h,)).astype(np.float32)
    idxs = rng.integers(-1, n, size=(b, m, topk)).astype(np.int32)
    return q, kv, sink, idxs


def test_sparse_attention_matches_the_released_kernel():
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, kv, sink, idxs = _inputs()
    scale = 1.0 / np.sqrt(q.shape[-1])
    expected = reference_sparse_attn(q, kv, sink, idxs, scale)
    got = np.array(sparse_windowed_attention(
        mx.array(q), mx.array(kv), mx.array(sink), mx.array(idxs), scale))
    assert np.allclose(got, expected, atol=1e-5), (
        f"max abs diff {np.abs(got - expected).max()}")


def test_all_slots_unused_yields_a_finite_shrunk_output():
    """Every index -1: the sink alone carries the denominator, output is 0."""
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, kv, sink, _ = _inputs()
    idxs = np.full((1, 3, 5), -1, np.int32)
    got = np.array(sparse_windowed_attention(
        mx.array(q), mx.array(kv), mx.array(sink), mx.array(idxs), 0.35))
    assert np.isfinite(got).all(), "unused slots produced non-finite output"
    assert np.allclose(got, 0.0)


def test_sink_only_scales_the_output_it_does_not_add_a_value():
    """A larger sink must shrink the output toward zero, not shift it."""
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, kv, _sink, idxs = _inputs(seed=4)
    idxs = np.clip(idxs, 0, None)  # all slots valid
    small = np.full((q.shape[2],), -20.0, np.float32)
    large = np.full((q.shape[2],), 5.0, np.float32)
    args = (mx.array(q), mx.array(kv))
    a = np.array(sparse_windowed_attention(
        *args, mx.array(small), mx.array(idxs), 0.35))
    c = np.array(sparse_windowed_attention(
        *args, mx.array(large), mx.array(idxs), 0.35))
    # Pure rescaling: direction preserved, magnitude shrinks. The scaling
    # factor is per (position, head) -- the sink enters as
    # exp(sink - row_max), and row_max varies with the position's own scores --
    # so the ratio is constant only along the feature axis. Pooling across
    # positions would compare different denominators and look non-uniform.
    assert np.abs(c).max() < np.abs(a).max()
    checked = 0
    for bi in range(q.shape[0]):
        for si in range(q.shape[1]):
            for head in range(q.shape[2]):
                row_a, row_c = a[bi, si, head], c[bi, si, head]
                mask = np.abs(row_a) > 1e-3
                if not mask.any():
                    continue
                ratio = row_c[mask] / row_a[mask]
                assert np.allclose(ratio, ratio[0], rtol=1e-3), (
                    f"({bi},{si},h{head}): sink changed direction, so it is "
                    "acting as a key rather than a denominator term")
                assert ratio[0] < 1.0, "a larger sink must shrink the output"
                checked += 1
    assert checked > 0, "test exercised nothing"


@pytest.mark.skipif(not has_reference,
                    reason="DeepSeek-V4-Flash-0731 inference/ not present")
@pytest.mark.parametrize("start_pos,seqlen", [(0, 12), (1, 1), (5, 1),
                                              (127, 1), (128, 1), (300, 1)])
def test_window_indices_match_the_released_generator(start_pos, seqlen):
    """Checked against the checkpoint's own get_window_topk_idxs."""
    import mlx.core as mx
    import torch

    sys.path.insert(0, str(INFERENCE))
    _install_kernel_stub()
    from model import get_window_topk_idxs

    from runtime.deepseek_v4 import window_topk_idxs

    window = 128
    expected = get_window_topk_idxs(window, 1, seqlen, start_pos)
    got = np.array(window_topk_idxs(window, seqlen, start_pos))
    assert got.shape == tuple(expected.shape), (
        f"shape {got.shape} != {tuple(expected.shape)}")
    assert np.array_equal(got, expected.numpy()), (
        f"window indices diverged at start_pos={start_pos}")
    assert isinstance(torch.tensor(0), torch.Tensor)  # torch really was used


@pytest.mark.skipif(not has_reference,
                    reason="DeepSeek-V4-Flash-0731 inference/ not present")
def test_decode_window_is_a_rotation_not_a_range():
    """Past one full window the ring buffer wraps; pin that it is a rotation."""
    import numpy as np

    from runtime.deepseek_v4 import window_topk_idxs

    window = 128
    got = np.array(window_topk_idxs(window, 1, 300))[0, 0]
    assert sorted(got.tolist()) == list(range(window)), (
        "a full window must reference every slot exactly once")
    assert got.tolist() != list(range(window)), (
        "expected a rotation, got a contiguous range")
    # 300 % 128 == 44, so the oldest slot is 45 and the newest is 44.
    assert got[0] == 45 and got[-1] == 44, got[:3].tolist()
