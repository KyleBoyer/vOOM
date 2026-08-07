"""F220: the gathered tensor must be the assembled one, not the bare ring.

``gather_indices`` addresses a tensor laid out as [window region][compressed
region], and the window region is the full prompt at prefill but a 128-slot
ring at decode. ``compressed_offset`` is set to whichever was built. Handing
the gather a different tensor than the one the indices were built for is
silent: every index is in range, every shape matches, and the values are
simply wrong.

It is also invisible below ``window_size``, because a ring written at offset 0
places position p in slot p % window, which equals p while p < window. That
coincidence is why short prompts stayed coherent while every prompt past 128
tokens collapsed. These tests pin the boundary from both sides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOW, RATIO = 8, 4


def _max_index(idxs):
    flat = np.array(idxs).reshape(-1)
    return int(flat[flat >= 0].max())


@pytest.mark.parametrize("seqlen", [4, 8, 12, 17, 33])
def test_prefill_indices_require_the_full_latent(seqlen):
    """At prefill the window region is the prompt, so indices reach seqlen-1."""
    from runtime.deepseek_v4 import gather_indices

    compressed_offset = seqlen  # what the engine sets when offset == 0
    idxs = gather_indices(WINDOW, RATIO, seqlen, 0, compressed_offset)
    n_compressed = seqlen // RATIO
    assembled = seqlen + n_compressed

    assert _max_index(idxs) < assembled, (
        "an index escapes the assembled [latent | compressed] tensor")
    if seqlen > WINDOW:
        assert _max_index(idxs) >= WINDOW, (
            f"seqlen={seqlen}: no index exceeds the ring, so gathering from a "
            "ring-sized tensor would go undetected here")


def test_below_the_window_ring_and_latent_indexing_coincide():
    """Documents the coincidence that hid the bug, so it is not mistaken for
    evidence that the ring is an acceptable gather target."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_ring_write

    seqlen = WINDOW  # at the boundary, still coincident
    positions = mx.arange(seqlen).reshape(1, seqlen, 1).astype(mx.float32)
    ring = np.array(window_ring_write(
        mx.zeros((1, WINDOW, 1)), positions, 0, WINDOW)).reshape(-1)
    assert np.array_equal(ring, np.arange(seqlen)), (
        "slot p holds position p only while p < window")


def test_past_the_window_ring_and_latent_indexing_diverge():
    """The other side of the boundary: the same index means different things."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_ring_write

    seqlen = WINDOW + 5
    positions = mx.arange(seqlen).reshape(1, seqlen, 1).astype(mx.float32)
    ring = np.array(window_ring_write(
        mx.zeros((1, WINDOW, 1)), positions, 0, WINDOW)).reshape(-1)
    latent = np.arange(seqlen)

    # A prefill index is an absolute position; reading it out of the ring
    # returns some other position entirely.
    probe = seqlen - 1
    assert latent[probe] == probe
    assert ring[probe % WINDOW] == probe
    assert not np.array_equal(ring[:WINDOW], latent[:WINDOW]), (
        "ring and latent agree past the window, so this test proves nothing")


def test_decode_indices_stay_inside_ring_plus_compressed():
    """At decode the window region IS the ring, and offset is the window."""
    from runtime.deepseek_v4 import gather_indices

    for start_pos in (WINDOW, WINDOW + 3, 4 * WINDOW + 1):
        idxs = gather_indices(WINDOW, RATIO, 1, start_pos, WINDOW)
        n_compressed = (start_pos + 1) // RATIO
        assert _max_index(idxs) < WINDOW + n_compressed, (
            f"start_pos={start_pos}: index escapes [ring | compressed]")


# ---- multi-position decode (speculative verification) ---------------------


def test_multi_position_decode_gives_each_query_its_own_window():
    """Feeding a draft block at once must stay causal ACROSS the block.

    The single-position branch returns one rotated row for every query. Reused
    for a block, position start_pos+i would attend to start_pos+j for j > i --
    a later drafted token leaking backwards, which changes an earlier token's
    output and silently corrupts verification.
    """
    import mlx.core as mx

    from runtime.deepseek_v4 import window_topk_idxs

    start, seqlen = 4 * WINDOW + 1, 5
    idxs = np.array(window_topk_idxs(WINDOW, seqlen, start))[0]
    assert idxs.shape == (seqlen, WINDOW)

    for i in range(seqlen):
        here = start + i
        allowed = {p % WINDOW for p in range(max(0, here - WINDOW + 1), here + 1)}
        used = {int(v) for v in idxs[i] if v >= 0}
        assert used <= allowed, (
            f"query {i} at position {here} read slots {used - allowed}, which "
            "hold positions it must not see")
        assert here % WINDOW in used, "query cannot see its own position"


def test_multi_position_rows_actually_differ():
    """Guard: if all rows matched, the causality test above proves nothing."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_topk_idxs

    idxs = np.array(window_topk_idxs(WINDOW, 5, 4 * WINDOW + 1))[0]
    rows = {tuple(r.tolist()) for r in idxs}
    assert len(rows) == 5, f"expected a distinct window per query, got {rows}"


def test_single_position_decode_is_unchanged():
    """The seqlen==1 path must keep its existing rotation exactly."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_topk_idxs

    for start in (WINDOW, WINDOW + 3, 4 * WINDOW + 1):
        got = np.array(window_topk_idxs(WINDOW, 1, start))
        rotated = start % WINDOW
        expected = np.concatenate([np.arange(rotated + 1, WINDOW),
                                   np.arange(0, rotated + 1)])
        assert np.array_equal(got.reshape(-1), expected)


def test_early_positions_mark_unwritten_slots():
    import mlx.core as mx

    from runtime.deepseek_v4 import window_topk_idxs

    idxs = np.array(window_topk_idxs(WINDOW, 3, 1))[0]
    # Position 1 has only slots 0 and 1 written; the rest must be masked.
    used = {int(v) for v in idxs[0] if v >= 0}
    assert used == {0, 1}, used


def test_block_gather_never_reads_a_slot_the_block_overwrote():
    """The bug this exists for: a ring slot holds ONE position.

    With the block written into the ring first, query 0's oldest window entry
    (position start-window+1) shares a slot with block position start+1. This
    gather addresses [ring | block] instead, so ring indices only ever mean
    positions strictly before the block.
    """
    import mlx.core as mx

    from runtime.deepseek_v4 import block_decode_topk_idxs

    start, seqlen = 4 * WINDOW + 1, 5
    idxs = np.array(block_decode_topk_idxs(WINDOW, seqlen, start))[0]
    assert idxs.shape == (seqlen, WINDOW + seqlen)

    ring_part, block_part = idxs[:, :WINDOW], idxs[:, WINDOW:]
    # Ring slots must correspond to positions strictly before the block.
    for i in range(seqlen):
        for j, slot in enumerate(ring_part[i]):
            if slot < 0:
                continue
            position = start - WINDOW + j
            assert position < start, "a ring index reached into the block"
            assert position % WINDOW == slot
            assert position >= start + i - WINDOW + 1

    # Block indices are causal and offset past the ring region.
    for i in range(seqlen):
        used = [int(v) for v in block_part[i] if v >= 0]
        assert used == [WINDOW + j for j in range(i + 1)], (
            f"query {i} block visibility {used} is not causal")


def test_block_gather_masks_negative_positions_early_in_the_sequence():
    import mlx.core as mx

    from runtime.deepseek_v4 import block_decode_topk_idxs

    idxs = np.array(block_decode_topk_idxs(WINDOW, 3, 2))[0]
    ring_part = idxs[:, :WINDOW]
    for i in range(3):
        for j, slot in enumerate(ring_part[i]):
            position = 2 - WINDOW + j
            if position < 0:
                assert slot == -1, "an unwritten position was addressed"


def test_block_gather_never_exceeds_the_window():
    """A sliding-window layer may see at most window_size positions.

    Causality alone does not give this once the block is wider than the
    window: query i would take every block position up to i. Small blocks
    never reach the bound, which is why a 5-position draft and an
    18-position chunk both looked correct while a 200-position chunk
    diverged by rel 0.698 at a ratio-0 layer.
    """
    import mlx.core as mx

    from runtime.deepseek_v4 import block_decode_topk_idxs

    start, seqlen = 400, 200          # block far wider than WINDOW
    idxs = np.array(block_decode_topk_idxs(WINDOW, seqlen, start))[0]
    for i in range(seqlen):
        visible = int((idxs[i] >= 0).sum())
        assert visible <= WINDOW, (
            f"query {i} sees {visible} positions, more than the "
            f"{WINDOW}-slot window")


def test_block_gather_keeps_the_newest_window_positions():
    """Bounded from the correct side: drop the oldest, keep the newest."""
    import mlx.core as mx

    from runtime.deepseek_v4 import block_decode_topk_idxs

    start, seqlen = 400, 200
    idxs = np.array(block_decode_topk_idxs(WINDOW, seqlen, start))[0]
    probe = 150                        # a query well past the window bound
    block_part = idxs[probe, WINDOW:]
    used = [int(v) - WINDOW for v in block_part if v >= 0]
    assert used[-1] == probe, "query cannot see its own position"
    assert used[0] == probe - WINDOW + 1, (
        f"oldest visible block position is {used[0]}, expected "
        f"{probe - WINDOW + 1}")
    assert (np.array(idxs[probe, :WINDOW]) < 0).all(), (
        "ring entries should be fully masked once the block covers the window")
