"""F230: a shared key region needs no per-query gather.

The compressed KV region is the same rows for every query in a tile; only how
far along a query may look differs. Gathering it duplicated those rows per
query -- 12,805 of a 13,317-slot gather at 51K tokens -- so the operand
[tile, topk, head_dim] was dominated by identical copies.

Attending to it as a masked matmul instead must be arithmetically the same
thing. These tests check that against the gather path directly, since the
softmax couples the two regions and an independent-softmax rewrite would look
plausible and be wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _case(seed=0, s=6, h=3, d=8, window=5, shared=7):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    q = mx.array(rng.normal(size=(1, s, h, d)).astype(np.float32))
    win_kv = mx.array(rng.normal(size=(1, window, d)).astype(np.float32))
    sh_kv = mx.array(rng.normal(size=(1, shared, d)).astype(np.float32))
    sink = mx.array(rng.normal(size=(h,)).astype(np.float32))
    # each query sees the whole window, and a growing prefix of the shared part
    win_idx = mx.broadcast_to(mx.arange(window)[None, None, :],
                              (1, s, window)).astype(mx.int32)
    reach = mx.array(np.clip(np.arange(1, s + 1), 0, shared).astype(np.int32))
    return q, win_kv, sh_kv, sink, win_idx, reach


def test_shared_region_matches_gathering_the_same_entries():
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, win_kv, sh_kv, sink, win_idx, reach = _case()
    s, window, shared = q.shape[1], win_kv.shape[1], sh_kv.shape[1]

    # Reference: one combined region, per-query gather, unreachable -> -1.
    combined = mx.concatenate([win_kv, sh_kv], axis=1)
    rows = []
    reach_np = np.array(reach)
    for i in range(s):
        row = list(range(window))
        row += [window + j if j < reach_np[i] else -1 for j in range(shared)]
        rows.append(row)
    idx = mx.array(np.array(rows, dtype=np.int32))[None]

    reference = sparse_windowed_attention(q, combined, sink, idx, 0.5, tile=0)
    got = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5, tile=0,
                                    shared_kv=sh_kv, shared_reach=reach)
    mx.eval(reference, got)
    rel = float(mx.max(mx.abs(reference - got))) / float(
        mx.max(mx.abs(reference)))
    assert rel < 1e-5, f"shared-region attention diverged, rel={rel}"


def test_shared_region_respects_its_causal_reach():
    """A query must not see a shared entry beyond its own reach."""
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, win_kv, sh_kv, sink, win_idx, reach = _case(seed=3)
    base = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5, tile=0,
                                     shared_kv=sh_kv, shared_reach=reach)
    # Perturb the LAST shared entry, which only the final query can reach.
    perturbed = mx.array(np.array(sh_kv))
    perturbed[:, -1] = perturbed[:, -1] + 10.0
    other = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5, tile=0,
                                      shared_kv=perturbed, shared_reach=reach)
    mx.eval(base, other)
    early = float(mx.max(mx.abs(base[:, :-1] - other[:, :-1])))
    assert early == 0.0, "a query saw a shared entry past its reach"


def test_tiling_is_invariant_with_a_shared_region():
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, win_kv, sh_kv, sink, win_idx, reach = _case(seed=5, s=9)
    ref = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5, tile=0,
                                    shared_kv=sh_kv, shared_reach=reach)
    for tile in (1, 2, 4, 9):
        got = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5,
                                        tile=tile, shared_kv=sh_kv,
                                        shared_reach=reach)
        mx.eval(ref, got)
        assert float(mx.max(mx.abs(ref - got))) < 1e-5, f"tile={tile}"


def test_empty_shared_region_is_the_plain_path():
    import mlx.core as mx

    from runtime.deepseek_v4 import sparse_windowed_attention

    q, win_kv, sh_kv, sink, win_idx, reach = _case(seed=7)
    plain = sparse_windowed_attention(q, win_kv, sink, win_idx, 0.5, tile=0)
    empty = sparse_windowed_attention(
        q, win_kv, sink, win_idx, 0.5, tile=0,
        shared_kv=mx.zeros((1, 0, q.shape[-1])),
        shared_reach=mx.zeros((q.shape[1],), mx.int32))
    mx.eval(plain, empty)
    assert float(mx.max(mx.abs(plain - empty))) == 0.0
