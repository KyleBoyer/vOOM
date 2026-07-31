"""Exact tile-local rANS probe for BF16 transformer trunks.

This is intentionally a research sidecar, not a production WeightStore path.
Each tile splits BF16 into its low/high byte planes, builds an independent
12-bit static model for each plane, and range-ANS encodes both.  Tiles that do
not beat their raw bytes (including headers) are stored verbatim.  The format
is independently block-decodable and bit-exact.

Promotion gate: decoded payload throughput must beat the physical tier serving
the same tensor, including model-header parsing.  A codec that merely makes a
file smaller but decodes below the current ~1.62 GB/s external-NVMe floor
cannot improve K3 latency and remains a probe.
"""

from __future__ import annotations

import argparse
import collections
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path


_MAGIC = b"VRN1"
_SCALE_BITS = 12
_TOTAL = 1 << _SCALE_BITS
_RANS_L = 1 << 23
_FRAME = struct.Struct("<4sIII")
_TILE = struct.Struct("<BIII")
_PLANE = struct.Struct("<I")
_FREQUENCIES = struct.Struct("<256H")


def _normalize_frequencies(data: bytes) -> list[int]:
    counts = collections.Counter(data)
    frequencies = [0] * 256
    active = sorted(counts)
    if not active:
        return frequencies
    remaining = _TOTAL - len(active)
    assigned = 0
    fractions = []
    length = len(data)
    for symbol in active:
        scaled = counts[symbol] * remaining
        extra = scaled // length
        frequencies[symbol] = 1 + extra
        assigned += extra
        fractions.append((scaled % length, counts[symbol], -symbol, symbol))
    left = remaining - assigned
    for _fraction, _count, _negative_symbol, symbol in sorted(
            fractions, reverse=True)[:left]:
        frequencies[symbol] += 1
    if sum(frequencies) != _TOTAL:
        raise AssertionError("normalized rANS frequencies do not sum to scale")
    return frequencies


def _cumulative(frequencies: list[int]) -> list[int]:
    result = [0] * 256
    total = 0
    for symbol, frequency in enumerate(frequencies):
        result[symbol] = total
        total += frequency
    if total not in (0, _TOTAL):
        raise ValueError("invalid rANS frequency total")
    return result


def _encode_plane(data: bytes) -> bytes:
    if not data:
        return _PLANE.pack(4) + _FREQUENCIES.pack(*([0] * 256)) + struct.pack(
            "<I", _RANS_L)
    frequencies = _normalize_frequencies(data)
    cumulative = _cumulative(frequencies)
    state = _RANS_L
    emitted = bytearray()
    for symbol in reversed(data):
        frequency = frequencies[symbol]
        threshold = ((_RANS_L >> _SCALE_BITS) << 8) * frequency
        while state >= threshold:
            emitted.append(state & 0xFF)
            state >>= 8
        state = (
            (state // frequency) << _SCALE_BITS
        ) + (state % frequency) + cumulative[symbol]
    payload = struct.pack("<I", state) + bytes(reversed(emitted))
    return (
        _PLANE.pack(len(payload))
        + _FREQUENCIES.pack(*frequencies)
        + payload
    )


def _decode_plane(encoded: memoryview, length: int) -> tuple[bytes, int]:
    if len(encoded) < _PLANE.size + _FREQUENCIES.size:
        raise ValueError("truncated rANS plane header")
    payload_size = _PLANE.unpack_from(encoded, 0)[0]
    frequencies = list(_FREQUENCIES.unpack_from(encoded, _PLANE.size))
    header = _PLANE.size + _FREQUENCIES.size
    end = header + payload_size
    if payload_size < 4 or end > len(encoded):
        raise ValueError("truncated rANS plane payload")
    payload = encoded[header:end]
    state = struct.unpack_from("<I", payload, 0)[0]
    cursor = 4
    cumulative = _cumulative(frequencies)
    lookup = [0] * _TOTAL
    for symbol, frequency in enumerate(frequencies):
        if frequency:
            start = cumulative[symbol]
            lookup[start:start + frequency] = [symbol] * frequency
    output = bytearray(length)
    mask = _TOTAL - 1
    for index in range(length):
        slot = state & mask
        symbol = lookup[slot]
        output[index] = symbol
        state = (
            frequencies[symbol] * (state >> _SCALE_BITS)
            + slot - cumulative[symbol]
        )
        while state < _RANS_L and cursor < len(payload):
            state = (state << 8) | payload[cursor]
            cursor += 1
    if cursor != len(payload):
        raise ValueError("rANS plane has trailing renormalization bytes")
    return bytes(output), end


@dataclass(frozen=True)
class RANSFrameStats:
    raw_bytes: int
    encoded_bytes: int
    tiles: int
    compressed_tiles: int

    @property
    def ratio(self) -> float:
        return (
            self.raw_bytes / self.encoded_bytes
            if self.encoded_bytes else 1.0
        )


def encode(data: bytes, *, tile_bytes: int = 1 << 20) -> tuple[bytes, RANSFrameStats]:
    if tile_bytes <= 0:
        raise ValueError("rANS tile_bytes must be positive")
    tiles = [
        data[start:start + tile_bytes]
        for start in range(0, len(data), tile_bytes)
    ]
    frame = bytearray(_FRAME.pack(
        _MAGIC, len(data), tile_bytes, len(tiles)))
    compressed_tiles = 0
    for raw in tiles:
        low = raw[0::2]
        high = raw[1::2]
        coded = _encode_plane(low) + _encode_plane(high)
        checksum = zlib.crc32(raw)
        if len(coded) < len(raw):
            frame.extend(_TILE.pack(1, len(raw), len(coded), checksum))
            frame.extend(coded)
            compressed_tiles += 1
        else:
            frame.extend(_TILE.pack(
                0, len(raw), len(raw), checksum))
            frame.extend(raw)
    result = bytes(frame)
    return result, RANSFrameStats(
        raw_bytes=len(data),
        encoded_bytes=len(result),
        tiles=len(tiles),
        compressed_tiles=compressed_tiles,
    )


def decode(frame: bytes) -> bytes:
    view = memoryview(frame)
    if len(view) < _FRAME.size:
        raise ValueError("truncated rANS frame")
    magic, raw_size, _tile_bytes, tile_count = _FRAME.unpack_from(view, 0)
    if magic != _MAGIC:
        raise ValueError("not a vOOM rANS frame")
    cursor = _FRAME.size
    output = bytearray()
    for _ in range(tile_count):
        if cursor + _TILE.size > len(view):
            raise ValueError("truncated rANS tile header")
        compressed, tile_raw_size, payload_size, checksum = _TILE.unpack_from(
            view, cursor)
        cursor += _TILE.size
        end = cursor + payload_size
        if end > len(view):
            raise ValueError("truncated rANS tile payload")
        payload = view[cursor:end]
        cursor = end
        if compressed == 0:
            if payload_size != tile_raw_size:
                raise ValueError("raw rANS tile size mismatch")
            tile = bytes(payload)
        elif compressed == 1:
            low_size = (tile_raw_size + 1) // 2
            high_size = tile_raw_size // 2
            low, used = _decode_plane(payload, low_size)
            high, used_high = _decode_plane(payload[used:], high_size)
            if used + used_high != len(payload):
                raise ValueError("compressed rANS tile has trailing bytes")
            tile = bytearray(tile_raw_size)
            tile[0::2] = low
            tile[1::2] = high
            tile = bytes(tile)
        else:
            raise ValueError(f"unknown rANS tile mode {compressed}")
        if zlib.crc32(tile) != checksum:
            raise ValueError("rANS tile checksum mismatch")
        output.extend(tile)
    if cursor != len(view) or len(output) != raw_size:
        raise ValueError("rANS frame length mismatch")
    return bytes(output)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-bytes", type=int, default=4 << 20)
    parser.add_argument("--tile-bytes", type=int, default=1 << 20)
    args = parser.parse_args()
    with args.path.open("rb") as source:
        raw = source.read(args.max_bytes)
    t0 = time.perf_counter()
    frame, stats = encode(raw, tile_bytes=args.tile_bytes)
    encode_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    restored = decode(frame)
    decode_s = time.perf_counter() - t0
    if restored != raw:
        raise SystemExit("roundtrip mismatch")
    print(
        f"raw={stats.raw_bytes} encoded={stats.encoded_bytes} "
        f"ratio={stats.ratio:.4f} compressed_tiles="
        f"{stats.compressed_tiles}/{stats.tiles} "
        f"encode={stats.raw_bytes / max(encode_s, 1e-9) / 1e6:.2f}MB/s "
        f"decode={stats.raw_bytes / max(decode_s, 1e-9) / 1e6:.2f}MB/s"
    )


if __name__ == "__main__":
    _main()
