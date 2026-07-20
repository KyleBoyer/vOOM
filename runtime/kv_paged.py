"""PagedKVCache: KV cache with fixed-size pages that spill to disk under a RAM
budget and page back in during attention.

Layout: per layer, a list of closed pages (page_positions wide) plus an open tail.
The most recent `resident_pages` closed pages per layer are kept in RAM; older
closed pages are spilled (oldest first, all layers) once the budget is exceeded.
Attention still needs *every* position, so spilled pages are reloaded lazily for
the duration of one attention call and dropped again — correct, but slower, which
is the intended trade for contexts that exceed RAM.

Spill format is safetensors via mx.save_safetensors, so reloads use the same lazy
mx.load path as weights. bf16 round-trips losslessly.

F07 (2026-07-13, opt-in via RuntimeConfig.kv_spill_compress): closed pages may
instead be zstd-L1 compressed before the write (same codec/level as F06's
default weight-pack choice; same byte-plane serialization as warm_tier.py's
_page_to_blobs). Purely a byte-transform of the same bf16 bits — reload
reconstructs the identical tensor, so this changes disk bytes and wall time
only, never a token. Kept opt-in, not default, pending measurement: F04's
compressed warm tier went NEGATIVE when sync compression cost outweighed the
disk savings, and KV activations are not guaranteed to compress like weights.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass
class KVStats:
    spills: int = 0
    reloads: int = 0
    spill_s: float = 0.0
    reload_s: float = 0.0
    spill_bytes_raw: int = 0  # only tracked when compress_spill is on
    spill_bytes_compressed: int = 0

    def summary(self) -> str:
        base = (
            f"kv: {self.spills} pages spilled ({self.spill_s:.2f}s), "
            f"{self.reloads} page reloads ({self.reload_s:.2f}s)"
        )
        if self.spill_bytes_compressed:
            ratio = self.spill_bytes_raw / self.spill_bytes_compressed
            base += (f", compress {ratio:.2f}x ({self.spill_bytes_raw / 1e6:.1f}"
                    f"->{self.spill_bytes_compressed / 1e6:.1f}MB)")
        return base


def _bf16_to_raw(arr: mx.array) -> bytes:
    return np.array(arr.view(mx.uint16), copy=False).tobytes()


def _raw_to_bf16(raw: bytes, shape: tuple) -> mx.array:
    return mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)


class _Page:
    __slots__ = ("k", "v", "path", "nbytes", "_compressed")

    def __init__(self, k: mx.array, v: mx.array):
        self.k, self.v = k, v
        self.path: Path | None = None
        self.nbytes = k.nbytes + v.nbytes
        self._compressed = False

    @property
    def resident(self) -> bool:
        return self.k is not None

    def spill(self, path: Path, compress: bool = False) -> int | None:
        """Returns the compressed byte count if compress=True, else None."""
        if compress:
            from compression import zstd

            shape = tuple(self.k.shape)
            k_c = zstd.compress(_bf16_to_raw(self.k), level=1)
            v_c = zstd.compress(_bf16_to_raw(self.v), level=1)
            path = path.with_suffix(".kvz")
            with open(path, "wb") as f:
                pickle.dump({"shape": shape, "k": k_c, "v": v_c}, f)
            self._compressed = True
            self.path = path
            self.k = self.v = None
            return len(k_c) + len(v_c)
        mx.save_safetensors(str(path), {"k": self.k, "v": self.v})
        self._compressed = False
        self.path = path
        self.k = self.v = None
        return None

    def load(self) -> tuple[mx.array, mx.array]:
        if self.resident:
            return self.k, self.v
        if self._compressed:
            from compression import zstd

            with open(self.path, "rb") as f:
                blob = pickle.load(f)
            k = _raw_to_bf16(zstd.decompress(blob["k"]), blob["shape"])
            v = _raw_to_bf16(zstd.decompress(blob["v"]), blob["shape"])
            mx.eval(k, v)
            return k, v
        lazy = mx.load(str(self.path))
        k, v = lazy["k"], lazy["v"]
        mx.eval(k, v)
        return k, v


class PagedKVCache:
    """Drop-in replacement for KVCache: exposes update()/offset/nbytes."""

    def __init__(
        self,
        num_layers: int,
        max_bytes: int,
        spill_dir: str | Path,
        page_positions: int = 256,
        resident_pages: int = 1,
        compress_spill: bool = False,
    ):
        self.num_layers = num_layers
        self.max_bytes = max_bytes
        self.page_positions = page_positions
        self.resident_pages = resident_pages
        self.compress_spill = compress_spill
        self.spill_dir = Path(spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self._pages: list[list[_Page]] = [[] for _ in range(num_layers)]
        self._tail_k: list[mx.array | None] = [None] * num_layers
        self._tail_v: list[mx.array | None] = [None] * num_layers
        self._offset = 0
        self.stats = KVStats()

    # ---- KVCache API ------------------------------------------------------

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self._tail_k[layer] is None:
            self._tail_k[layer], self._tail_v[layer] = k, v
        else:
            self._tail_k[layer] = mx.concatenate([self._tail_k[layer], k], axis=2)
            self._tail_v[layer] = mx.concatenate([self._tail_v[layer], v], axis=2)

        # close full pages out of the tail
        while self._tail_k[layer].shape[2] >= self.page_positions:
            pk = self._tail_k[layer][:, :, : self.page_positions, :]
            pv = self._tail_v[layer][:, :, : self.page_positions, :]
            mx.eval(pk, pv)
            self._pages[layer].append(_Page(pk, pv))
            self._tail_k[layer] = self._tail_k[layer][:, :, self.page_positions :, :]
            self._tail_v[layer] = self._tail_v[layer][:, :, self.page_positions :, :]
            mx.eval(self._tail_k[layer], self._tail_v[layer])

        if layer == self.num_layers - 1:
            self._offset += k.shape[2]
            self._enforce_budget()

        # assemble full K/V for attention, paging in spilled pages transiently
        parts_k, parts_v = [], []
        for page in self._pages[layer]:
            if not page.resident:
                t0 = time.perf_counter()
                pk, pv = page.load()
                self.stats.reloads += 1
                self.stats.reload_s += time.perf_counter() - t0
            else:
                pk, pv = page.k, page.v
            parts_k.append(pk)
            parts_v.append(pv)
        if self._tail_k[layer].shape[2] > 0 or not parts_k:
            parts_k.append(self._tail_k[layer])
            parts_v.append(self._tail_v[layer])
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    @property
    def offset(self) -> int:
        return self._offset

    def nbytes(self) -> int:
        """Resident bytes only (spilled pages cost disk, not RAM)."""
        total = 0
        for layer in range(self.num_layers):
            total += sum(p.nbytes for p in self._pages[layer] if p.resident)
            if self._tail_k[layer] is not None:
                total += self._tail_k[layer].nbytes + self._tail_v[layer].nbytes
        return total

    def release(self) -> None:
        """Release resident tensors and only this cache's recorded spill files."""
        paths = {
            page.path
            for pages in self._pages
            for page in pages
            if page.path is not None
        }
        self._pages = [[] for _ in range(self.num_layers)]
        self._tail_k = [None] * self.num_layers
        self._tail_v = [None] * self.num_layers
        self._offset = 0
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # ---- spilling -----------------------------------------------------------

    def _enforce_budget(self):
        if self.nbytes() <= self.max_bytes:
            return
        # spill oldest closed pages first, round-robin across layers, but always
        # keep the newest `resident_pages` closed pages of each layer resident
        for page_idx in range(max(len(p) for p in self._pages)):
            for layer in range(self.num_layers):
                pages = self._pages[layer]
                if page_idx >= len(pages) - self.resident_pages:
                    continue
                page = pages[page_idx]
                if not page.resident:
                    continue
                raw_bytes = page.nbytes
                t0 = time.perf_counter()
                comp_bytes = page.spill(self.spill_dir / f"kv_l{layer}_p{page_idx}.safetensors",
                                        compress=self.compress_spill)
                self.stats.spills += 1
                self.stats.spill_s += time.perf_counter() - t0
                if comp_bytes is not None:
                    self.stats.spill_bytes_raw += raw_bytes
                    self.stats.spill_bytes_compressed += comp_bytes
                mx.clear_cache()
                if self.nbytes() <= self.max_bytes:
                    return
