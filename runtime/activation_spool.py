"""Exact bounded-memory activation spools for layer-stationary inference."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import mlx.core as mx
import numpy as np

from .uncached_io import set_darwin_nocache


class DiskBacked16BitTileSpool:
    """Store fixed FP16/BF16 position tiles as raw bits on local disk.

    The spool owns one temporary, pre-sized file.  Every write and read is
    descriptor-level and requests Darwin ``F_NOCACHE`` so a multi-gigabyte
    activation carrier is not silently retained in unified-memory file cache.
    Tile boundaries and shapes are fixed at construction; restoring a tile
    recreates the original rank, dtype, and raw 16-bit payload without a
    floating-point conversion.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        shape: tuple[int, ...],
        spans: list[tuple[int, int]],
        dtype,
    ):
        if dtype not in (mx.float16, mx.bfloat16):
            raise TypeError(
                "disk activation spool supports only FP16/BF16 values, "
                f"got {dtype}")
        if len(shape) < 2 or any(int(size) <= 0 for size in shape):
            raise ValueError(f"invalid activation spool shape {shape}")
        batch, positions, *tail = map(int, shape)
        normalized = [(int(start), int(end)) for start, end in spans]
        cursor = 0
        for start, end in normalized:
            if start != cursor or not start < end <= positions:
                raise ValueError(
                    "activation spool spans must exactly partition the "
                    f"position axis: cursor={cursor}, span={(start, end)}, "
                    f"positions={positions}")
            cursor = end
        if cursor != positions:
            raise ValueError(
                "activation spool spans do not cover the position axis: "
                f"{cursor} != {positions}")

        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="voom-glm53-activation-", dir=root)
        self.directory = Path(self._temporary.name)
        self.path = self.directory / "carrier.u16"
        self.dtype = dtype
        self.shape = (batch, positions, *tail)
        self.spans = tuple(normalized)
        self._tile_shapes = tuple(
            (batch, end - start, *tail) for start, end in normalized)
        self._offsets = []
        offset = 0
        for tile_shape in self._tile_shapes:
            self._offsets.append(offset)
            elements = 1
            for size in tile_shape:
                elements *= int(size)
            offset += elements * 2
        self.logical_bytes = offset
        self._fd = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.ftruncate(self._fd, self.logical_bytes)
        self.uncached_descriptors = int(set_darwin_nocache(self._fd))
        self.bytes_written = 0
        self.bytes_read = 0
        self.write_calls = 0
        self.read_calls = 0
        self.write_seconds = 0.0
        self.read_seconds = 0.0
        self._closed = False

    def _tile(self, index: int) -> tuple[int, tuple[int, ...], int]:
        index = int(index)
        if not 0 <= index < len(self._tile_shapes):
            raise IndexError(
                f"activation spool tile {index} outside "
                f"[0, {len(self._tile_shapes)})")
        shape = self._tile_shapes[index]
        elements = 1
        for size in shape:
            elements *= int(size)
        return self._offsets[index], shape, elements * 2

    def store(self, index: int, value: mx.array) -> int:
        """Replace one tile and return its stable integer handle."""
        import time

        if self._closed:
            raise RuntimeError("activation spool is closed")
        offset, shape, expected_bytes = self._tile(index)
        if tuple(map(int, value.shape)) != shape:
            raise ValueError(
                f"activation spool tile {index} shape {tuple(value.shape)} "
                f"!= {shape}")
        if value.dtype != self.dtype:
            raise TypeError(
                f"activation spool tile {index} dtype {value.dtype} "
                f"!= {self.dtype}")
        started = time.perf_counter()
        mx.eval(value)
        host = np.asarray(value.view(mx.uint16))
        if host.dtype != np.uint16:
            raise TypeError(
                f"activation spool host dtype {host.dtype} is not uint16")
        payload = host.tobytes(order="C")
        if len(payload) != expected_bytes:
            raise IOError(
                f"activation spool tile {index} bytes {len(payload)} "
                f"!= {expected_bytes}")
        written = 0
        while written < expected_bytes:
            count = os.pwrite(
                self._fd, payload[written:], offset + written)
            if count <= 0:
                raise IOError(
                    f"short activation spool write at tile {index}: "
                    f"{written}/{expected_bytes}")
            written += count
        self.bytes_written += written
        self.write_calls += 1
        self.write_seconds += time.perf_counter() - started
        return int(index)

    def load(self, index: int) -> mx.array:
        """Restore one tile as its original FP16/BF16 MLX array."""
        import time

        if self._closed:
            raise RuntimeError("activation spool is closed")
        offset, shape, expected_bytes = self._tile(index)
        started = time.perf_counter()
        payload = bytearray(expected_bytes)
        view = memoryview(payload)
        read = 0
        while read < expected_bytes:
            chunk = os.pread(self._fd, expected_bytes - read, offset + read)
            if not chunk:
                raise IOError(
                    f"short activation spool read at tile {index}: "
                    f"{read}/{expected_bytes}")
            view[read:read + len(chunk)] = chunk
            read += len(chunk)
        host = np.frombuffer(payload, dtype=np.uint16).reshape(shape)
        result = mx.array(host, dtype=mx.uint16).view(self.dtype)
        mx.eval(result)
        self.bytes_read += read
        self.read_calls += 1
        self.read_seconds += time.perf_counter() - started
        return result

    def stats(self) -> dict[str, int | float]:
        return {
            "logical_bytes": self.logical_bytes,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "write_calls": self.write_calls,
            "read_calls": self.read_calls,
            "write_s": self.write_seconds,
            "read_s": self.read_seconds,
            "uncached_descriptors": self.uncached_descriptors,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)
        self._temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
