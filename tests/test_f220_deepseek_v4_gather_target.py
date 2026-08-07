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
