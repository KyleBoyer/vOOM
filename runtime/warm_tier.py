"""F04: compressed-RAM warm tier between the Metal weight cache and disk.

Evicted pages are zstd-L1-compressed (measured 1.44-1.46x on bf16 weight classes,
decode 2.2-2.6 GB/s — far above any disk here) into a plain-RAM heap. A cache miss
checks the warm tier before disk. Incompressible pages (MXFP4 U8 blocks: ~1.0x)
are not admitted — the tier would just be slower RAM for them.

Budgeted separately from the Metal cache (gate #6 in the F-queue doc: an explicit
compressed heap counts against total RAM); the F16 governor can shrink it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from compression import zstd

import mlx.core as mx
import numpy as np


def _page_to_blobs(tensors: dict) -> list[tuple[str, str, tuple, bytes]] | None:
    """Serialize a page's mx arrays to (name, kind, shape, raw bytes). Returns
    None if any tensor type is unsupported (e.g. QTensor) — such pages skip the
    warm tier."""
    out = []
    for name, arr in tensors.items():
        if not isinstance(arr, mx.array):
            return None
        if arr.dtype == mx.bfloat16:
            raw = np.array(arr.view(mx.uint16), copy=False).tobytes()
            out.append((name, "bf16", tuple(arr.shape), raw))
        elif arr.dtype == mx.uint8:
            out.append((name, "u8", tuple(arr.shape), np.array(arr, copy=False).tobytes()))
        elif arr.dtype == mx.float32:
            out.append((name, "f32", tuple(arr.shape), np.array(arr, copy=False).tobytes()))
        else:
            return None
    return out


def _blobs_to_page(blobs) -> dict:
    tensors = {}
    for name, kind, shape, raw in blobs:
        if kind == "bf16":
            tensors[name] = mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)
        elif kind == "u8":
            tensors[name] = mx.array(np.frombuffer(raw, dtype=np.uint8).reshape(shape))
        else:
            tensors[name] = mx.array(np.frombuffer(raw, dtype=np.float32).reshape(shape))
    mx.eval(list(tensors.values()))
    return tensors


class WarmTier:
    MIN_RATIO = 1.15  # don't admit pages that barely compress

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._store: "OrderedDict[str, list]" = OrderedDict()  # key -> compressed blobs
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.admits = 0
        self.rejects = 0

    def admit(self, key: str, tensors: dict):
        blobs = _page_to_blobs(tensors)
        if blobs is None:
            self.rejects += 1
            return
        comp = []
        raw_total = comp_total = 0
        for name, kind, shape, raw in blobs:
            c = zstd.compress(raw, level=1)
            comp.append((name, kind, shape, c))
            raw_total += len(raw)
            comp_total += len(c)
        if raw_total / max(comp_total, 1) < self.MIN_RATIO:
            self.rejects += 1
            return
        with self._lock:
            if key in self._store:
                return
            self._store[key] = comp
            self._bytes += comp_total
            self.admits += 1
            while self._bytes > self.max_bytes and self._store:
                _, old = self._store.popitem(last=False)
                self._bytes -= sum(len(c) for *_, c in old)

    def take(self, key: str) -> dict | None:
        with self._lock:
            comp = self._store.pop(key, None)
            if comp is None:
                return None
            self._bytes -= sum(len(c) for *_, c in comp)
        self.hits += 1
        return _blobs_to_page([(n, k, s, zstd.decompress(c)) for n, k, s, c in comp])

    def summary(self) -> str:
        return (f"warm tier: {self.hits} hits, {self.admits} admits, {self.rejects} rejects, "
                f"{self._bytes / 1e6:.0f}MB resident")
