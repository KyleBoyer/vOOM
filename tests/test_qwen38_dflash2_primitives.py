"""Synthetic numerical gates for the isolated MIT DFlash2 primitives."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from runtime.dflash2 import (
    CandidateSelector,
    grouped_dynamic_convolve,
    project_out_direction,
)


def _reference_convolve(hidden, dynamic, base, group_size):
    hidden = np.asarray(hidden, dtype=np.float32)
    dynamic = np.asarray(dynamic, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    batch, length, width = hidden.shape
    groups = width // group_size
    output = np.zeros_like(hidden)
    for b in range(batch):
        for position in range(length):
            for offset in range(base.shape[0]):
                source = position - offset
                if source < 0:
                    continue
                for group in range(groups):
                    start = group * group_size
                    end = start + group_size
                    output[b, position, start:end] += (
                        base[offset, start:end]
                        + dynamic[b, position, offset, group]
                    ) * hidden[b, source, start:end]
    return output


def test_grouped_dynamic_convolution_matches_independent_loop_and_is_causal():
    hidden = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4) / 8
    dynamic = mx.arange(16, dtype=mx.float32).reshape(1, 4, 2, 2) / 20
    base = mx.array([
        [1.0, 0.5, -0.5, 0.25],
        [0.2, -0.1, 0.4, 0.3],
    ])

    actual = grouped_dynamic_convolve(hidden, dynamic, base, group_size=2)
    expected = _reference_convolve(
        np.array(hidden), np.array(dynamic), np.array(base), 2)
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-6)

    changed = hidden.at[:, -1].add(100)
    changed_output = grouped_dynamic_convolve(
        changed, dynamic, base, group_size=2)
    mx.eval(changed_output)
    np.testing.assert_array_equal(
        np.array(actual[:, :-1]), np.array(changed_output[:, :-1]))


def test_fused_grouped_dynamic_convolution_matches_reference_formula():
    hidden = (
        mx.arange(2 * 5 * 16, dtype=mx.float32).reshape(2, 5, 16) / 97
    ).astype(mx.bfloat16)
    dynamic = (
        mx.arange(2 * 5 * 2 * 4, dtype=mx.float32).reshape(2, 5, 2, 4)
        / 211
    ).astype(mx.bfloat16)
    base = (
        mx.arange(2 * 16, dtype=mx.float32).reshape(2, 16) / 53
    ).astype(mx.bfloat16)

    reference = grouped_dynamic_convolve(
        hidden, dynamic, base, group_size=4)
    fused = grouped_dynamic_convolve(
        hidden, dynamic, base, group_size=4, fused=True)
    mx.eval(reference, fused)
    np.testing.assert_allclose(
        np.array(fused.astype(mx.float32)),
        np.array(reference.astype(mx.float32)),
        rtol=0,
        # The fused kernel accumulates in FP32 and rounds once; the reference
        # rounds each BF16 multiply/add graph edge.  At this magnitude their
        # measured envelope is one BF16 ULP (2^-6).
        atol=2 ** -6,
    )


def test_direction_projection_removes_only_the_selected_component():
    hidden = mx.array([[[3.0, 4.0], [1.0, -2.0]]])
    direction = mx.array([1.0, 0.0])
    projected = project_out_direction(hidden, direction)
    mx.eval(projected)
    np.testing.assert_array_equal(
        np.array(projected),
        np.array([[[0.0, 4.0], [0.0, -2.0]]], dtype=np.float32),
    )
    assert project_out_direction(hidden, direction, 0.0) is hidden


def test_candidate_selector_walks_parent_conditioned_path_and_returns_sparse_q():
    selector = CandidateSelector(2, 4, 2, 2)
    selector.hidden_projection.weight = mx.eye(2)
    selector.predecessor_codebook.weight = mx.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    selector.successor_codebook.weight = mx.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
    ])
    hidden = mx.array([[[1.0, 0.0], [0.0, 1.0]]])
    logits = mx.array([[
        [-20.0, 1.0, 0.9, -20.0],
        [-20.0, 1.0, -20.0, 0.9],
    ]])
    anchor = mx.array([0])

    path, candidates, q = selector.select(hidden, logits, anchor)
    mx.eval(path, candidates)
    assert path.tolist() == [[2, 3]]
    assert q is None

    mx.random.seed(9)
    _path, stochastic_candidates, q = selector.select(
        hidden, logits, anchor, temperature=0.7)
    mx.eval(stochastic_candidates, q)
    assert q.shape == (1, 2, 2)
    np.testing.assert_allclose(
        np.array(mx.sum(q, axis=-1)), np.ones((1, 2)), atol=1e-6)
