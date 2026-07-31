"""Exactness and corruption gates for the BF16 rANS research sidecar."""

from __future__ import annotations

import os

import pytest

from formats.bf16_rans import (
    _FRAME,
    _FREQUENCIES,
    _PLANE,
    _TILE,
    decode,
    encode,
)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 257, 8193])
def test_rans_roundtrip_arbitrary_bytes(size):
    raw = os.urandom(size)
    frame, stats = encode(raw, tile_bytes=257)
    assert decode(frame) == raw
    assert stats.raw_bytes == size
    assert stats.tiles == (size + 256) // 257


def test_rans_compresses_structured_bf16_byte_planes():
    # Typical BF16 exponent/sign bytes have much lower entropy than mantissas.
    raw = b"".join(bytes((index & 0xFF, 0x3F)) for index in range(32_768))
    frame, stats = encode(raw, tile_bytes=16_384)

    assert decode(frame) == raw
    assert stats.compressed_tiles == stats.tiles
    assert stats.ratio > 1.2


def test_rans_rejects_truncated_frame():
    frame, _ = encode(bytes(range(256)) * 20, tile_bytes=1024)
    with pytest.raises(ValueError):
        decode(frame[:-3])


def test_rans_rejects_payload_corruption():
    raw = b"".join(bytes((index & 0xFF, 0x3F)) for index in range(8192))
    frame, stats = encode(raw, tile_bytes=len(raw))
    assert stats.compressed_tiles == 1
    corrupted = bytearray(frame)
    first_plane_state = (
        _FRAME.size + _TILE.size + _PLANE.size + _FREQUENCIES.size)
    corrupted[first_plane_state] ^= 0x40

    with pytest.raises(ValueError, match="checksum"):
        decode(bytes(corrupted))
