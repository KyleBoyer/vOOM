"""F205: oracle for DeepSeek V4's hyper-connection mixing.

Hyper-connections replace the residual stream with ``hc_mult`` parallel copies,
reduced to one per sub-layer and re-expanded afterwards, using weights produced
per token by a Sinkhorn normalization. The released implementation is a fused
TileLang kernel (``inference/kernel.py::hc_split_sinkhorn_kernel``) that cannot
run here, so the reference below is a direct scalar transcription of that
kernel -- deliberately written in numpy loops that mirror its structure rather
than reusing the MLX implementation's own vectorized form.

``comb`` is softmaxed over its last axis, column-normalized once, and only then
enters the ``sinkhorn_iters - 1`` row/column loop. Measured here: at the
released 20 iterations that pre-loop column step makes no difference, because
Sinkhorn has already reached its fixed point -- so it is NOT load-bearing as
first assumed. It is at one iteration, which is what the convergence test
pins, so lowering ``hc_sinkhorn_iters`` for speed would make the ordering
matter again and fail that test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HC = 4
MIX = (2 + HC) * HC
EPS = 1e-6
ITERS = 20


def reference_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC,
                             iters=ITERS, eps=EPS):
    """Scalar transcription of hc_split_sinkhorn_kernel_ (kernel.py:378)."""
    n = mixes.shape[0]
    pre = np.empty((n, hc), np.float32)
    post = np.empty((n, hc), np.float32)
    comb = np.empty((n, hc, hc), np.float32)
    for i in range(n):
        row = mixes[i]
        for j in range(hc):
            pre[i, j] = 1.0 / (1.0 + np.exp(-(row[j] * hc_scale[0]
                                              + hc_base[j]))) + eps
        for j in range(hc):
            z = row[j + hc] * hc_scale[1] + hc_base[j + hc]
            post[i, j] = 2.0 / (1.0 + np.exp(-z))
        block = np.empty((hc, hc), np.float32)
        for j in range(hc):
            for k in range(hc):
                idx = j * hc + k + hc * 2
                block[j, k] = row[idx] * hc_scale[2] + hc_base[idx]
        # comb = comb.softmax(-1) + eps
        row_max = block.max(axis=1, keepdims=True)
        block = np.exp(block - row_max)
        block = block / block.sum(axis=1, keepdims=True) + eps
        # comb = comb / (comb.sum(-2) + eps)   <- once, before the loop
        block = block / (block.sum(axis=0, keepdims=True) + eps)
        for _ in range(iters - 1):
            block = block / (block.sum(axis=1, keepdims=True) + eps)
            block = block / (block.sum(axis=0, keepdims=True) + eps)
        comb[i] = block
    return pre, post, comb


def _mixes(n=7, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, MIX)) * 1.5).astype(np.float32)


def test_split_sinkhorn_matches_the_released_kernel():
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_split_sinkhorn

    rng = np.random.default_rng(1)
    mixes = _mixes()
    hc_scale = rng.normal(size=(3,)).astype(np.float32)
    hc_base = rng.normal(size=(MIX,)).astype(np.float32)

    ref_pre, ref_post, ref_comb = reference_split_sinkhorn(
        mixes, hc_scale, hc_base)
    pre, post, comb = hc_split_sinkhorn(
        mx.array(mixes), mx.array(hc_scale), mx.array(hc_base),
        hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)

    assert np.allclose(np.array(pre), ref_pre, atol=1e-5), "pre diverged"
    assert np.allclose(np.array(post), ref_post, atol=1e-5), "post diverged"
    assert np.allclose(np.array(comb), ref_comb, atol=1e-4), "comb diverged"


def test_comb_is_driven_toward_doubly_stochastic():
    """Sinkhorn's purpose: both marginals approach one."""
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_split_sinkhorn

    rng = np.random.default_rng(2)
    _pre, _post, comb = hc_split_sinkhorn(
        mx.array(_mixes(seed=2)),
        mx.array(rng.normal(size=(3,)).astype(np.float32)),
        mx.array(rng.normal(size=(MIX,)).astype(np.float32)),
        hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS)
    value = np.array(comb)
    assert np.allclose(value.sum(axis=-1), 1.0, atol=5e-3)
    assert np.allclose(value.sum(axis=-2), 1.0, atol=5e-3)


def _comb_without_preloop_column_step(mixes, hc_scale, hc_base, iters):
    """The same routine with the single pre-loop column normalization removed."""
    n = mixes.shape[0]
    out = np.empty((n, HC, HC), np.float32)
    for i in range(n):
        block = np.empty((HC, HC), np.float32)
        for j in range(HC):
            for k in range(HC):
                idx = j * HC + k + HC * 2
                block[j, k] = mixes[i][idx] * hc_scale[2] + hc_base[idx]
        block = np.exp(block - block.max(axis=1, keepdims=True))
        block = block / block.sum(axis=1, keepdims=True) + EPS
        for _ in range(iters - 1):
            block = block / (block.sum(axis=1, keepdims=True) + EPS)
            block = block / (block.sum(axis=0, keepdims=True) + EPS)
        out[i] = block
    return out


def test_sinkhorn_has_converged_by_the_released_iteration_count():
    """At 20 iterations the pre-loop column step is immaterial -- but only there.

    Written as a guard against reducing ``hc_sinkhorn_iters`` for speed. The
    released count is deep enough that Sinkhorn reaches its fixed point, so the
    exact pre-loop ordering washes out; at one iteration it dominates. If a
    future change lowers the count, the second assertion starts failing and
    the ordering becomes load-bearing again.
    """
    rng = np.random.default_rng(3)
    mixes = _mixes(seed=3)
    hc_scale = rng.normal(size=(3,)).astype(np.float32)
    hc_base = rng.normal(size=(MIX,)).astype(np.float32)

    converged = reference_split_sinkhorn(mixes, hc_scale, hc_base,
                                         iters=ITERS)[2]
    assert np.allclose(
        converged,
        _comb_without_preloop_column_step(mixes, hc_scale, hc_base, ITERS),
        atol=1e-4), "expected convergence to make the pre-loop step immaterial"

    shallow = reference_split_sinkhorn(mixes, hc_scale, hc_base, iters=1)[2]
    assert not np.allclose(
        shallow,
        _comb_without_preloop_column_step(mixes, hc_scale, hc_base, 1),
        atol=1e-4), "at one iteration the pre-loop column step must dominate"


def test_hc_pre_and_post_round_trip_shapes_and_stream_mixing():
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_post, hc_pre

    rng = np.random.default_rng(4)
    b, s, d = 1, 3, 16
    x = mx.array((rng.normal(size=(b, s, HC, d)) * 0.1).astype(np.float32))
    hc_fn = mx.array((rng.normal(size=(MIX, HC * d)) * 0.05).astype(np.float32))
    hc_scale = mx.array(rng.normal(size=(3,)).astype(np.float32))
    hc_base = mx.array(rng.normal(size=(MIX,)).astype(np.float32))

    reduced, post, comb = hc_pre(
        x, hc_fn, hc_scale, hc_base, hc_mult=HC, norm_eps=1e-6,
        sinkhorn_iters=ITERS, eps=EPS)
    assert reduced.shape == (b, s, d)
    assert post.shape == (b, s, HC)
    assert comb.shape == (b, s, HC, HC)

    out = hc_post(reduced, x, post, comb)
    assert out.shape == (b, s, HC, d)
    # Every output stream must depend on the new contribution: zeroing it
    # has to change all four.
    zeroed = hc_post(mx.zeros_like(reduced), x, post, comb)
    delta = np.abs(np.array(out) - np.array(zeroed)).sum(axis=(0, 1, 3))
    assert (delta > 0).all(), f"a stream ignored the sub-layer output: {delta}"


def test_hc_pre_rms_is_computed_over_all_streams_jointly():
    """Per-stream normalization would be a different function; pin it."""
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_pre

    rng = np.random.default_rng(5)
    b, s, d = 1, 2, 8
    base = rng.normal(size=(b, s, HC, d)).astype(np.float32) * 0.1
    scaled = base.copy()
    scaled[:, :, 0, :] *= 10.0  # perturb only stream 0

    args = dict(hc_mult=HC, norm_eps=1e-6, sinkhorn_iters=ITERS, eps=EPS)
    hc_fn = mx.array((rng.normal(size=(MIX, HC * d)) * 0.05).astype(np.float32))
    hc_scale = mx.array(rng.normal(size=(3,)).astype(np.float32))
    hc_base = mx.array(rng.normal(size=(MIX,)).astype(np.float32))

    a, _, _ = hc_pre(mx.array(base), hc_fn, hc_scale, hc_base, **args)
    c, _, _ = hc_pre(mx.array(scaled), hc_fn, hc_scale, hc_base, **args)
    # A joint RMS makes stream 0's magnitude affect every stream's weighting,
    # so streams 1..3 of the reduction must move too.
    assert not np.allclose(np.array(a), np.array(c), atol=1e-6)


@pytest.mark.skipif(
    not (ROOT / "models" / "DeepSeek-V4-Flash-0731"
         / "model.safetensors.index.json").is_file(),
    reason="DeepSeek-V4-Flash-0731 not present")
def test_released_hc_tensor_shapes_match_the_implementation():
    """The real checkpoint must supply exactly the shapes hc_pre expects."""
    import json

    import mlx.core as mx

    model = ROOT / "models" / "DeepSeek-V4-Flash-0731"
    index = json.loads(
        (model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((model / "config.json").read_text())
    hc_mult = config["hc_mult"]
    dim = config["hidden_size"]
    mix = (2 + hc_mult) * hc_mult

    shard = mx.load(str(model / index["layers.5.hc_attn_fn"]))
    assert tuple(shard["layers.5.hc_attn_fn"].shape) == (mix, hc_mult * dim)
    assert tuple(shard["layers.5.hc_attn_base"].shape) == (mix,)
    assert tuple(shard["layers.5.hc_attn_scale"].shape) == (3,)
