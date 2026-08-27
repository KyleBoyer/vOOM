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
import uuid
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
    page_native_calls: int = 0
    page_native_groups: int = 0
    page_native_positions: int = 0
    page_native_s: float = 0.0

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
        # More than one restored durable prefix may briefly coexist while a
        # request is admitted.  Cache-local names prevent two instances from
        # overwriting each other's exact spill pages in the shared directory.
        self._cache_id = uuid.uuid4().hex
        self._pages: list[list[_Page]] = [[] for _ in range(num_layers)]
        self._tail_k: list[mx.array | None] = [None] * num_layers
        self._tail_v: list[mx.array | None] = [None] * num_layers
        self._offset = 0
        self._persisted_layers: frozenset[int] | None = None
        self.compressed_mla = False
        self.stats = KVStats()
        # Explicit fast-profile-only mode. Qwen sets this after construction;
        # the generic/lossless default remains the established full-history
        # materialization plus MLX SDPA.
        self.online_attention = False
        self.online_attention_tile_positions = 2048
        self.online_attention_page_native = False
        self.online_attention_pages_per_tile = 8

    # ---- KVCache API ------------------------------------------------------

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        self._append_layer(layer, k, v)

        # Match KVCache's global-position contract: mixed-depth Qwen prefill
        # can retain the complete prefix in an early attention layer while a
        # later layer stores only a compact suffix. The global endpoint is the
        # longest local layer, not specifically the final model layer.
        self._offset = max(self._offset, self.layer_positions(layer))
        # Enforce after every layer, not only the final one. Chunk-major
        # execution reaches the final layer before the next chunk, but exact
        # layer-stationary prefill completes one layer's entire context before
        # advancing. Waiting for the final layer there leaves every earlier
        # layer resident and defeats the configured paging budget.
        self._enforce_budget(protected_layer=layer)
        return self.materialize_layer(layer)

    def append_for_online_attention(
        self, layer: int, k: mx.array, v: mx.array,
    ) -> None:
        """Append without constructing a second full-history K/V tensor."""
        self._append_layer(layer, k, v)
        self._offset = max(self._offset, self.layer_positions(layer))
        self._enforce_budget(protected_layer=layer)

    def iter_materialized_layer_chunks(
        self, layer: int, *, max_positions: int,
    ):
        """Yield complete K/V history in bounded contiguous page tiles.

        Spilled BF16 pages are reconstructed exactly. Only the online-softmax
        reduction consuming these tiles is approximate; storage and paging
        remain byte-preserving.
        """
        max_positions = int(max_positions)
        if max_positions <= 0:
            raise ValueError("paged KV chunk width must be positive")
        pending_k, pending_v = [], []
        pending_positions = 0

        def materialize_pending():
            if not pending_k:
                return None
            keys = (pending_k[0] if len(pending_k) == 1
                    else mx.concatenate(pending_k, axis=2))
            values = (pending_v[0] if len(pending_v) == 1
                      else mx.concatenate(pending_v, axis=2))
            return keys, values

        def sources():
            # Load one page at a time. Keeping a Python list of every loaded
            # spill page would recreate the full resident history before the
            # fused tile kernel had a chance to bound it.
            for page in self._pages[layer]:
                if not page.resident:
                    t0 = time.perf_counter()
                    keys, values = page.load()
                    self.stats.reloads += 1
                    self.stats.reload_s += time.perf_counter() - t0
                else:
                    keys, values = page.k, page.v
                yield keys, values
            tail_k, tail_v = self._tail_k[layer], self._tail_v[layer]
            if tail_k is not None and int(tail_k.shape[2]) > 0:
                yield tail_k, tail_v

        for keys, values in sources():
            cursor = 0
            width = int(keys.shape[2])
            while cursor < width:
                take = min(
                    width - cursor, max_positions - pending_positions)
                pending_k.append(keys[:, :, cursor:cursor + take, :])
                pending_v.append(values[:, :, cursor:cursor + take, :])
                pending_positions += take
                cursor += take
                if pending_positions == max_positions:
                    result = materialize_pending()
                    if result is not None:
                        yield result
                    pending_k, pending_v = [], []
                    pending_positions = 0
        result = materialize_pending()
        if result is not None:
            yield result

    def iter_materialized_layer_pages(self, layer: int):
        """Yield exact stored pages without concatenating adjacent pages.

        This is the bounded input surface for page-native attention kernels.
        Spilled pages are still loaded only for the duration of the consumer's
        current page group; they are never installed back into the cache.
        """
        for page in self._pages[layer]:
            if not page.resident:
                t0 = time.perf_counter()
                keys, values = page.load()
                self.stats.reloads += 1
                self.stats.reload_s += time.perf_counter() - t0
            else:
                keys, values = page.k, page.v
            yield keys, values
        tail_k, tail_v = self._tail_k[layer], self._tail_v[layer]
        if tail_k is not None and int(tail_k.shape[2]) > 0:
            yield tail_k, tail_v

    def _append_layer(self, layer: int, k: mx.array, v: mx.array) -> None:
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

    def materialize_layer(self, layer: int) -> tuple[mx.array, mx.array]:
        """Return one layer's complete exact history for attention/testing.

        Spilled pages are loaded transiently and are not installed back into
        the cache, so callers never convert the paged representation into a
        second resident copy of the whole prefix.
        """
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

    def layer_positions(self, layer: int) -> int:
        total = sum(page.k.shape[2] if page.resident else
                    page.nbytes // self._page_position_bytes(layer)
                    for page in self._pages[layer])
        if self._tail_k[layer] is not None:
            total += int(self._tail_k[layer].shape[2])
        return int(total)

    def _page_position_bytes(self, layer: int) -> int:
        for page in self._pages[layer]:
            positions = (int(page.k.shape[2]) if page.resident
                         else self.page_positions)
            if positions:
                return page.nbytes // positions
        raise ValueError(f"layer {layer} has no closed page geometry")

    def persistence_slice(self, start: int, end: int) -> dict[str, mx.array]:
        """Materialize only ``[start:end]`` for durable journal serialization."""
        start, end = int(start), int(end)
        if not 0 <= start < end <= self.offset:
            raise ValueError("paged persistence slice is outside cache prefix")
        arrays: dict[str, mx.array] = {}
        for layer in range(self.num_layers):
            positions = self.layer_positions(layer)
            if positions == 0:
                continue
            if positions != self.offset:
                raise ValueError(
                    f"paged layer {layer} covers {positions}, expected {self.offset}")
            pieces_k, pieces_v = [], []
            cursor = 0
            sources = list(self._pages[layer])
            for page in sources:
                pk, pv = page.load()
                width = int(pk.shape[2])
                lo, hi = max(start, cursor), min(end, cursor + width)
                if lo < hi:
                    pieces_k.append(pk[:, :, lo - cursor:hi - cursor, :])
                    pieces_v.append(pv[:, :, lo - cursor:hi - cursor, :])
                cursor += width
            tail_k, tail_v = self._tail_k[layer], self._tail_v[layer]
            if tail_k is not None:
                width = int(tail_k.shape[2])
                lo, hi = max(start, cursor), min(end, cursor + width)
                if lo < hi:
                    pieces_k.append(tail_k[:, :, lo - cursor:hi - cursor, :])
                    pieces_v.append(tail_v[:, :, lo - cursor:hi - cursor, :])
            if not pieces_k:
                raise ValueError(f"paged layer {layer} has no requested slice")
            arrays[f"k{layer}"] = (pieces_k[0] if len(pieces_k) == 1
                                    else mx.concatenate(pieces_k, axis=2))
            arrays[f"v{layer}"] = (pieces_v[0] if len(pieces_v) == 1
                                    else mx.concatenate(pieces_v, axis=2))
        return arrays

    def append_persisted_segment(
            self, arrays: dict[str, mx.array], start: int, end: int) -> None:
        """Append one verified journal segment without materializing history."""
        start, end = int(start), int(end)
        if start != self.offset or end <= start:
            raise ValueError("paged restore segments must be contiguous")
        keys, values = {}, {}
        for name, value in arrays.items():
            if name.startswith("k") and name[1:].isdigit():
                keys[int(name[1:])] = value
            elif name.startswith("v") and name[1:].isdigit():
                values[int(name[1:])] = value
            else:
                raise ValueError(f"unsupported paged journal tensor {name!r}")
        layers = frozenset(keys)
        if not layers or layers != frozenset(values):
            raise ValueError("paged restore requires paired K/V tensors")
        if any(layer < 0 or layer >= self.num_layers for layer in layers):
            raise ValueError("paged restore layer is outside model geometry")
        if self._persisted_layers is None:
            self._persisted_layers = layers
        elif layers != self._persisted_layers:
            raise ValueError("paged restore segment layer set changed")
        width = end - start
        for layer in sorted(layers):
            k, v = keys[layer], values[layer]
            if (k.ndim != 4 or v.ndim != 4 or k.shape != v.shape
                    or int(k.shape[2]) != width
                    or self.layer_positions(layer) != start):
                raise ValueError("paged restore tensor geometry mismatch")
            self._append_layer(layer, k, v)
            self._enforce_budget(protected_layer=layer)
        self._offset = end

    @property
    def offset(self) -> int:
        return self._offset

    def layer_lengths(self) -> tuple[int, ...]:
        """Return each layer's exact local append length.

        Hybrid Qwen speculative verification updates only full-attention
        layers.  Exposing the same checkpoint surface as ``KVCache`` lets its
        target-authoritative rollback keep paged K/V and recurrent KDA state
        aligned without materializing the whole prefix.
        """
        return tuple(
            self.layer_positions(layer) for layer in range(self.num_layers))

    @staticmethod
    def _unlink_pages(pages: list[_Page]) -> None:
        for page in pages:
            if page.path is None:
                continue
            try:
                page.path.unlink()
            except FileNotFoundError:
                pass

    def _trim_layer_to(self, layer: int, target: int) -> None:
        current = self.layer_positions(layer)
        if not 0 <= target <= current:
            raise ValueError("cannot grow paged KV during rollback")
        if target == current:
            return
        pages = self._pages[layer]
        tail_k, tail_v = self._tail_k[layer], self._tail_v[layer]
        if target == 0:
            self._unlink_pages(pages)
            self._pages[layer] = []
            self._tail_k[layer] = None
            self._tail_v[layer] = None
            return

        full_pages, partial = divmod(target, self.page_positions)
        if full_pages > len(pages):
            raise RuntimeError("paged KV rollback geometry is inconsistent")
        if partial:
            if full_pages < len(pages):
                source = pages[full_pages]
                if not source.resident:
                    started = time.perf_counter()
                    next_k, next_v = source.load()
                    self.stats.reloads += 1
                    self.stats.reload_s += time.perf_counter() - started
                else:
                    next_k, next_v = source.k, source.v
            else:
                if tail_k is None:
                    raise RuntimeError("paged KV rollback omitted its tail")
                next_k, next_v = tail_k, tail_v
            new_tail_k = next_k[:, :, :partial, :]
            new_tail_v = next_v[:, :, :partial, :]
        else:
            if tail_k is None:
                raise RuntimeError("paged KV rollback omitted tail geometry")
            new_tail_k = tail_k[:, :, :0, :]
            new_tail_v = tail_v[:, :, :0, :]

        removed = pages[full_pages:]
        self._pages[layer] = pages[:full_pages]
        self._tail_k[layer] = new_tail_k
        self._tail_v[layer] = new_tail_v
        mx.eval(new_tail_k, new_tail_v)
        self._unlink_pages(removed)

    def trim_layer_lengths(self, lengths) -> None:
        """Roll every paged attention layer back to checkpoint-local lengths."""
        targets = tuple(int(value) for value in lengths)
        if len(targets) != self.num_layers or any(value < 0 for value in targets):
            raise ValueError("invalid per-layer paged KV rollback lengths")
        for layer, target in enumerate(targets):
            self._trim_layer_to(layer, target)
        self._offset = max(targets, default=0)
        mx.clear_cache()

    def trim(self, length: int) -> None:
        """Roll all populated layers back to one absolute sequence length."""
        length = int(length)
        if length < 0 or length > self.offset:
            raise ValueError("invalid paged KV trim length")
        self.trim_layer_lengths(tuple(
            min(self.layer_positions(layer), length)
            for layer in range(self.num_layers)
        ))
        self._offset = length

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
        self._persisted_layers = None
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # ---- spilling -----------------------------------------------------------

    def _enforce_budget(self, protected_layer: int | None = None):
        if self.nbytes() <= self.max_bytes:
            return
        # Spill completed layers before the layer currently being extended.
        # In layer-stationary prefill one complete Qwen full-attention layer
        # is only ~200 MB, so this keeps that layer resident while it is read
        # on every causal tile and tiers older completed layers instead.  The
        # protected layer remains a fallback when it alone exceeds the cap,
        # preserving the hard budget for chunk-major and extreme shapes.
        phases = (
            (False, True)
            if protected_layer is not None
            else (False,)
        )
        max_pages = max(len(p) for p in self._pages)
        for protected_phase in phases:
            for page_idx in range(max_pages):
                for layer in range(self.num_layers):
                    is_protected = layer == protected_layer
                    if is_protected != protected_phase:
                        continue
                    pages = self._pages[layer]
                    if page_idx >= len(pages) - self.resident_pages:
                        continue
                    page = pages[page_idx]
                    if not page.resident:
                        continue
                    raw_bytes = page.nbytes
                    t0 = time.perf_counter()
                    comp_bytes = page.spill(
                        self.spill_dir
                        / (f"kv_{self._cache_id}_l{layer}_p{page_idx}"
                           ".safetensors"),
                        compress=self.compress_spill)
                    self.stats.spills += 1
                    self.stats.spill_s += time.perf_counter() - t0
                    if comp_bytes is not None:
                        self.stats.spill_bytes_raw += raw_bytes
                        self.stats.spill_bytes_compressed += comp_bytes
                    mx.clear_cache()
                    if self.nbytes() <= self.max_bytes:
                        return
