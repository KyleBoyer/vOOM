"""F208: oracle for DeepSeek V4's indexer selection and Hadamard rotation.

The indexer picks which compressed KV entries the sparse attention may gather.
It exists only where ``compress_ratio == 4``.

``rotate_activation`` calls ``fast_hadamard_transform``, which is not installed
here, so the reference is an explicit dense Hadamard matrix built by Sylvester
construction -- unambiguous, and independent of any butterfly implementation.
The selection logic is checked against a numpy transcription of
``Indexer.forward`` (model.py:408).

The masking is applied twice in the released code and both are load-bearing:
once as ``-inf`` before the top-k, and again afterwards, because top-k always
returns k slots even when fewer entries are reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dense_hadamard(n: int) -> np.ndarray:
    """Sylvester construction: H_1 = [[1]], H_2k = [[H,H],[H,-H]]."""
    matrix = np.ones((1, 1), np.float32)
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


@pytest.mark.parametrize("n", [2, 4, 8, 128])
def test_hadamard_transform_matches_the_dense_matrix(n):
    import mlx.core as mx

    from runtime.deepseek_v4 import hadamard_transform

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, n)).astype(np.float32)
    expected = x @ dense_hadamard(n)
    got = np.array(hadamard_transform(mx.array(x)))
    assert np.allclose(got, expected, atol=1e-4), (
        f"n={n}: max abs diff {np.abs(got - expected).max()}")


def test_hadamard_scale_is_applied_once_not_per_stage():
    """The released call passes d**-0.5, making the transform orthonormal."""
    import mlx.core as mx

    from runtime.deepseek_v4 import hadamard_transform

    n = 128
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, n)).astype(np.float32)
    got = np.array(hadamard_transform(mx.array(x), scale=n ** -0.5))
    expected = (x @ dense_hadamard(n)) * (n ** -0.5)
    assert np.allclose(got, expected, atol=1e-4)
    # Orthonormal: norms are preserved.
    assert np.allclose(np.linalg.norm(got, axis=-1),
                       np.linalg.norm(x, axis=-1), rtol=1e-3)


def test_hadamard_rejects_a_non_power_of_two():
    import mlx.core as mx

    from runtime.deepseek_v4 import hadamard_transform

    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_transform(mx.zeros((2, 12)))


def reference_index_score(q, kv, weights):
    """Transcription of Indexer.forward's scoring (model.py:426-428)."""
    score = np.einsum("bshd,btd->bsht", q, kv)
    score = np.maximum(score, 0.0) * weights[..., None]
    return score.sum(axis=2)


def test_index_scores_rectify_before_the_head_sum():
    import mlx.core as mx

    from runtime.deepseek_v4 import index_scores

    rng = np.random.default_rng(2)
    q = rng.normal(size=(1, 4, 3, 8)).astype(np.float32)
    kv = rng.normal(size=(1, 6, 8)).astype(np.float32)
    weights = rng.normal(size=(1, 4, 3)).astype(np.float32)

    expected = reference_index_score(q, kv, weights)
    got = np.array(index_scores(mx.array(q), mx.array(kv), mx.array(weights)))
    assert np.allclose(got, expected, atol=1e-5)

    # Rectifying after the sum would be a different function; show it differs.
    naive = (np.einsum("bshd,btd->bsht", q, kv)
             * weights[..., None]).sum(axis=2)
    assert not np.allclose(np.maximum(naive, 0.0), expected, atol=1e-4)


def test_prefill_selection_never_points_past_the_current_group():
    """Causality: position p may only read compressed indices < (p+1)//ratio."""
    import mlx.core as mx

    from runtime.deepseek_v4 import index_topk_idxs

    ratio, seqlen, entries, offset = 4, 16, 4, 100
    rng = np.random.default_rng(3)
    score = mx.array(rng.normal(size=(1, seqlen, entries)).astype(np.float32))
    idxs = np.array(index_topk_idxs(
        score, seqlen, ratio, offset, index_topk=512,
        end_pos=seqlen, prefill=True))

    for position in range(seqlen):
        reach = (position + 1) // ratio
        used = idxs[0, position]
        used = used[used >= 0] - offset
        assert (used < reach).all(), (
            f"position {position} selected compressed entry {used.max()} "
            f"but may only read below {reach}")


def test_unreachable_slots_are_marked_rather_than_pointing_forward():
    """Early positions have fewer reachable entries than k; surplus must be -1."""
    import mlx.core as mx

    from runtime.deepseek_v4 import index_topk_idxs

    ratio, seqlen, entries = 4, 12, 3
    score = mx.zeros((1, seqlen, entries))
    idxs = np.array(index_topk_idxs(
        score, seqlen, ratio, 0, index_topk=512, end_pos=seqlen,
        prefill=True))
    # Positions 0..2 can reach nothing at all.
    assert (idxs[0, :3] == -1).all(), idxs[0, :3]
    # Position 4 reaches exactly entry 0.
    reachable = idxs[0, 4][idxs[0, 4] >= 0]
    assert set(reachable.tolist()) <= {0}


def test_decode_selection_shifts_by_offset_without_masking():
    import mlx.core as mx

    from runtime.deepseek_v4 import index_topk_idxs

    rng = np.random.default_rng(5)
    score = mx.array(rng.normal(size=(1, 1, 8)).astype(np.float32))
    offset = 128
    idxs = np.array(index_topk_idxs(
        score, 1, 4, offset, index_topk=3, end_pos=32, prefill=False))
    assert idxs.shape == (1, 1, 3)
    assert (idxs >= offset).all(), "decode indices must all be offset"
    top = np.argsort(-np.array(score)[0, 0])[:3] + offset
    assert set(idxs[0, 0].tolist()) == set(top.tolist())


def test_keep_count_is_bounded_by_available_entries():
    import mlx.core as mx

    from runtime.deepseek_v4 import index_topk_idxs

    score = mx.zeros((1, 4, 2))
    idxs = np.array(index_topk_idxs(
        score, 4, 4, 0, index_topk=512, end_pos=8, prefill=True))
    assert idxs.shape[-1] == 2, (
        "cannot select more compressed entries than exist")
