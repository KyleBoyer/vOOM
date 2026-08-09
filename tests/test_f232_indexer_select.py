"""F232: Indexer selection, oracled against a transcription of the released one.

The released model scores every compressed entry with a small projection and
keeps index_topk (512) of them. This runtime attended over the WHOLE
compressed region instead, which is a fidelity gap that widens with length --
the region is seqlen/ratio entries, 12,805 at 51K tokens against a cap of 512.

The reference here is a transcription of Indexer.forward's selection maths
(the released module needs kernel stubs that are unavailable, so it cannot be
imported and run directly). The transcription is deliberately literal, so a
disagreement means our version diverged, not that the two were written
differently.

fp4_act_quant is a kernel stub in the released source, so both sides skip it;
that is stated in indexer_select's own docstring rather than hidden here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEADS, HEAD_DIM, ROPE_DIM, RATIO, TOPK = 4, 8, 4, 4, 3


def _reference(qr, x, wq_b, weights_proj, indexer_kv, cos, sin,
               start_pos, offset):
    """Literal transcription of the released selection, in numpy."""
    from runtime.deepseek_v4 import apply_rope_interleaved, hadamard_transform
    import mlx.core as mx

    b, s, _ = x.shape
    q = np.array(mx.array(qr) @ mx.array(wq_b).T).reshape(b, s, HEADS, HEAD_DIM)
    tail = np.array(apply_rope_interleaved(
        mx.array(q[..., -ROPE_DIM:]), cos, sin))
    q = np.concatenate([q[..., :-ROPE_DIM], tail], axis=-1)
    q = np.array(hadamard_transform(mx.array(q)))

    weights = np.array(x) @ np.array(weights_proj).T
    weights = weights * (HEAD_DIM ** -0.5 * HEADS ** -0.5)

    kv = np.array(indexer_kv)
    score = np.einsum("bshd,btd->bsht", q, kv)
    score = np.maximum(score, 0.0) * weights[..., None]
    score = score.sum(axis=2)                       # [b, s, t]

    end_pos = start_pos + s
    if start_pos == 0:
        reach = (np.arange(1, s + 1) // RATIO)[:, None]
        blocked = np.arange(score.shape[-1])[None, :] >= reach
        score = score + np.where(blocked, -np.inf, 0.0)[None]
    keep = min(TOPK, max(end_pos // RATIO, 1))
    order = np.argsort(-score, axis=-1)[..., :keep]
    if start_pos == 0:
        reach = (np.arange(1, s + 1) // RATIO)[None, :, None]
        order = np.where(order >= reach, -1, order + offset)
    else:
        order = order + offset
    return order


def _case(seed=0, s=8, start_pos=0):
    import mlx.core as mx

    from runtime.deepseek_v4 import yarn_freqs

    rng = np.random.default_rng(seed)
    dim, qlora = 16, 12
    qr = mx.array(rng.normal(size=(1, s, qlora)).astype(np.float32))
    x = mx.array(rng.normal(size=(1, s, dim)).astype(np.float32))
    wq_b = mx.array(rng.normal(size=(HEADS * HEAD_DIM, qlora)).astype(np.float32))
    wproj = mx.array(rng.normal(size=(HEADS, dim)).astype(np.float32))
    entries = max((start_pos + s) // RATIO, 1)
    kv = mx.array(rng.normal(size=(1, entries, HEAD_DIM)).astype(np.float32))
    cos, sin = yarn_freqs(ROPE_DIM, start_pos + s, 0, 10000.0, 1.0, 32, 1)
    return qr, x, wq_b, wproj, kv, cos[start_pos:], sin[start_pos:]


@pytest.mark.parametrize("start_pos,s", [(0, 8), (0, 16), (12, 1)])
def test_selection_matches_the_released_maths(start_pos, s):
    import mlx.core as mx

    from runtime.deepseek_v4 import indexer_select

    qr, x, wq_b, wproj, kv, cos, sin = _case(seed=start_pos + s,
                                             s=s, start_pos=start_pos)
    offset = 100
    got = np.array(indexer_select(
        qr, x, wq_b, wproj, kv, n_heads=HEADS, head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM, cos=cos, sin=sin, ratio=RATIO,
        index_topk=TOPK, start_pos=start_pos, offset=offset))
    want = _reference(qr, x, wq_b, wproj, kv, cos, sin, start_pos, offset)

    # Compare as SETS per row: both keep the same entries; argpartition and
    # argsort need not agree on order within the kept group.
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    for i in range(got.shape[1]):
        assert set(int(v) for v in got[0, i]) == set(int(v) for v in want[0, i]), (
            f"row {i}: {sorted(got[0, i])} != {sorted(want[0, i])}")


def test_selection_never_reaches_an_unseeable_entry():
    """A position may only read entries strictly before its own group."""
    import mlx.core as mx

    from runtime.deepseek_v4 import indexer_select

    s, offset = 20, 50
    qr, x, wq_b, wproj, kv, cos, sin = _case(seed=5, s=s)
    got = np.array(indexer_select(
        qr, x, wq_b, wproj, kv, n_heads=HEADS, head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM, cos=cos, sin=sin, ratio=RATIO,
        index_topk=TOPK, start_pos=0, offset=offset))
    for i in range(s):
        used = [int(v) - offset for v in got[0, i] if v >= 0]
        assert all(u < (i + 1) // RATIO for u in used), (
            f"position {i} selected {used}, beyond its group")


def test_selection_caps_at_index_topk():
    """The point of the Indexer: attention width stops growing."""
    import mlx.core as mx

    from runtime.deepseek_v4 import indexer_select

    qr, x, wq_b, wproj, kv, cos, sin = _case(seed=9, s=40)
    got = indexer_select(
        qr, x, wq_b, wproj, kv, n_heads=HEADS, head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM, cos=cos, sin=sin, ratio=RATIO,
        index_topk=TOPK, start_pos=0, offset=0)
    assert got.shape[-1] == TOPK, (
        f"selected {got.shape[-1]} entries against a cap of {TOPK}; the "
        "compressed region would still grow with the prompt")
