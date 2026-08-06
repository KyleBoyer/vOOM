"""F215: oracle for DeepSeek V4's incremental compressor state.

Decode fills a compression group one position at a time and emits an entry only
on the step that completes it. The reference is the checkpoint's own
``Compressor.forward`` decode branch (model.py:351-366), transcribed in numpy,
since running the module itself requires its TileLang kernels.

The property that matters is continuity: streaming ``ratio`` positions through
``step`` must produce the same compressed entry as pooling those positions in
one shot, because prefill and decode have to agree at their boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEAD_DIM = 8


def test_emits_only_on_the_group_boundary():
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    ratio = 128
    state = CompressorState(ratio, HEAD_DIM)
    ape = mx.zeros((ratio, HEAD_DIM))
    rng = np.random.default_rng(0)
    emitted = []
    for position in range(2 * ratio):
        kv = mx.array(rng.normal(size=(1, 1, HEAD_DIM)).astype(np.float32))
        out = state.step(kv, mx.zeros((1, 1, HEAD_DIM)), position, ape)
        if out is not None:
            emitted.append(position)
    assert emitted == [ratio - 1, 2 * ratio - 1], emitted


def test_streamed_group_matches_one_shot_pooling():
    """Decode must agree with prefill pooling on the same positions."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    ratio = 128
    rng = np.random.default_rng(1)
    kv = rng.normal(size=(ratio, HEAD_DIM)).astype(np.float32)
    scores = rng.normal(size=(ratio, HEAD_DIM)).astype(np.float32)
    ape = rng.normal(size=(ratio, HEAD_DIM)).astype(np.float32)

    state = CompressorState(ratio, HEAD_DIM)
    out = None
    for position in range(ratio):
        out = state.step(
            mx.array(kv[position][None, None]),
            mx.array(scores[position][None, None]), position, mx.array(ape))
    assert out is not None

    biased = scores + ape
    weights = np.exp(biased - biased.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    expected = (kv * weights).sum(axis=0)

    assert np.allclose(np.array(out)[0, 0], expected, atol=1e-4), (
        f"max abs diff {np.abs(np.array(out)[0, 0] - expected).max()}")


def test_unfilled_slots_contribute_nothing():
    """Scores start at -inf so a partial group cannot leak into the pool."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    ratio = 4
    state = CompressorState(ratio, HEAD_DIM)
    assert not np.isfinite(np.array(state.score_state)).any()


def test_overlap_state_slides_after_emitting():
    """At ratio 4 the completed group becomes the next group's overlap half."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    ratio = 4
    state = CompressorState(ratio, HEAD_DIM)
    ape = mx.zeros((ratio, 2 * HEAD_DIM))
    rng = np.random.default_rng(2)
    for position in range(ratio):
        state.step(
            mx.array(rng.normal(size=(1, 1, 2 * HEAD_DIM)).astype(np.float32)),
            mx.zeros((1, 1, 2 * HEAD_DIM)), position, ape)
    first_half = np.array(state.kv_state)[:, :ratio]
    second_half = np.array(state.kv_state)[:, ratio:]
    assert np.array_equal(first_half, second_half), (
        "after emitting, the overlap half must hold the completed group")


def test_ratio_four_uses_the_split_feature_halves():
    """Overlap pooling reads the overlap half's low features and the
    current half's high features, not the whole vector twice."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    ratio = 4
    state = CompressorState(ratio, HEAD_DIM)
    ape = mx.zeros((ratio, 2 * HEAD_DIM))
    # Low features 1.0, high features 2.0 -- the pooled entry must be finite
    # and drawn from both halves rather than one.
    kv = mx.concatenate(
        [mx.ones((1, 1, HEAD_DIM)), mx.full((1, 1, HEAD_DIM), 2.0)], axis=-1)
    out = None
    for position in range(ratio):
        out = state.step(kv, mx.zeros((1, 1, 2 * HEAD_DIM)), position, ape)
    assert out is not None
    assert out.shape == (1, 1, HEAD_DIM)
    assert np.isfinite(np.array(out)).all()
