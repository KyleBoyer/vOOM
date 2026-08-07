"""F228: chunked prefill must see exactly what a single sweep sees.

Chunked prefill was silently wrong: a 1550-token prompt answered correctly in
one chunk and produced three different non-answers at three chunk sizes. Three
defects combined, all of them invisible to the shapes tested until then:

* ``compress_topk_idxs`` sized its row from ``start_pos`` alone, ignoring
  ``seqlen`` (F214 oracled only ``(0, many)`` and ``(many, 1)``);
* ``block_decode_topk_idxs`` was causal but unbounded BELOW, so a block wider
  than the window saw more history than a sliding-window layer may;
* ``window_ring_write`` wrote ``kv[:, :1]`` mid-stream, discarding every
  position after the first, so each chunk left a ring of stale entries.

Fixed, six chunk sizes over a 1550-token prompt -- 1, 2, 4, 7, 16 and 25
chunks -- all emit the same six tokens as a single sweep. This file pins the
composed invariant at unit level so the end-to-end agreement has a cheap
guard behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOW = 8


def _visible_positions(window, seqlen, start_pos):
    """Positions a chunked sweep can actually reach, via ring + block."""
    import mlx.core as mx

    from runtime.deepseek_v4 import block_decode_topk_idxs, window_ring_write

    # Ring holding every position before this chunk.
    ring = mx.zeros((1, window, 1))
    if start_pos:
        prior = mx.arange(start_pos).reshape(1, start_pos, 1).astype(mx.float32)
        ring = window_ring_write(ring, prior, 0, window)
    ring_values = np.array(ring).reshape(-1)

    idxs = np.array(block_decode_topk_idxs(window, seqlen, start_pos))[0]
    out = []
    for i in range(seqlen):
        seen = set()
        for v in idxs[i]:
            v = int(v)
            if v < 0:
                continue
            seen.add(float(ring_values[v]) if v < window
                     else float(start_pos + (v - window)))
        out.append(seen)
    return out


@pytest.mark.parametrize("start_pos,seqlen", [
    (8, 4), (8, 12), (16, 8), (13, 5), (20, 20), (32, 3),
])
def test_chunked_sees_exactly_the_single_sweep_window(start_pos, seqlen):
    """Each query must see its own sliding window, no more and no less."""
    seen = _visible_positions(WINDOW, seqlen, start_pos)
    for i, got in enumerate(seen):
        here = start_pos + i
        want = set(float(p) for p in
                   range(max(0, here - WINDOW + 1), here + 1))
        assert got == want, (
            f"start_pos={start_pos} seqlen={seqlen} query {i} (position "
            f"{here}) saw {sorted(got)}, expected {sorted(want)}")


def test_a_block_wider_than_the_window_is_still_bounded():
    seen = _visible_positions(WINDOW, 5 * WINDOW, 4 * WINDOW)
    for got in seen:
        assert len(got) <= WINDOW


def test_compressed_reach_grows_with_position_inside_a_block():
    """Every query's compressed reach is its own, not the block's first."""
    import mlx.core as mx

    from runtime.deepseek_v4 import compress_topk_idxs

    ratio, seqlen, start_pos, offset = 4, 12, 40, 100
    idxs = np.array(compress_topk_idxs(ratio, seqlen, start_pos, offset))[0]
    assert idxs.shape[0] == seqlen, "one row per query, not a single row"
    counts = [(row >= 0).sum() for row in idxs]
    assert counts == sorted(counts), "reach must be non-decreasing"
    assert counts[-1] > counts[0], (
        "reach identical across the block; the start_pos-only form is back")
    for i, row in enumerate(idxs):
        used = [int(v) - offset for v in row if v >= 0]
        assert all(u < (start_pos + i + 1) // ratio for u in used), (
            f"query {i} read a compressed entry it cannot see")
