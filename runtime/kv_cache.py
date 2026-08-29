"""Resident per-layer KV caches with one shared update/trim interface."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import mlx.core as mx
import numpy as np

from .uncached_io import set_darwin_nocache


class KVCache:
    """Exact-length concatenating cache; fastest for short conversations."""

    compressed_mla: bool = False

    def __init__(self, num_layers: int):
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        # Per-layer retention bound. ``None`` keeps the whole sequence, which
        # is the only behavior any caller saw before sliding windows existed.
        self._windows: list[int | None] = [None] * num_layers
        # Absolute position of each layer's FIRST retained key. Zero unless a
        # window has dropped something, so ``offset`` and ``trim`` keep their
        # exact previous semantics for every unwindowed cache.
        self._starts: list[int] = [0] * num_layers

    def configure_sliding_windows(self, layer_types, window: int) -> int:
        """Bound layers whose attention provably cannot read older keys.

        gpt-oss alternates 128-token ``sliding_attention`` layers with full
        layers, and its attention already slices/masks to that window -- so on
        an 18.6K-token prompt half the layers retained ~158x more KV than they
        could ever attend to. Dropping those keys is exact, not approximate:
        they are unreachable by construction. Returns how many layers were
        bounded.
        """
        if not layer_types or not window or window <= 0:
            return 0
        bounded = 0
        for layer, kind in enumerate(layer_types):
            if layer < len(self._windows) and str(kind) == "sliding_attention":
                self._windows[layer] = int(window)
                bounded += 1
        return bounded

    def layer_start(self, layer: int) -> int:
        """Absolute position of the first key retained for ``layer``."""
        return self._starts[layer] if layer < len(self._starts) else 0

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = mx.concatenate([self.keys[layer], k], axis=2)
            self.values[layer] = mx.concatenate([self.values[layer], v], axis=2)
        window = self._windows[layer]
        if window is not None:
            # The window is per QUERY, so a prefill tile of L queries still
            # needs window + L - 1 keys: its OLDEST query looks back a full
            # window from L-1 positions earlier. Retaining only ``window``
            # here starves every multi-token tile. Decode (L=1) reduces to
            # exactly ``window``, and the extra tail is dropped on the next
            # append.
            retain = window + k.shape[2] - 1
            length = self.keys[layer].shape[2]
            if length > retain:
                self._starts[layer] += length - retain
                self.keys[layer] = self.keys[layer][:, :, -retain:, :]
                self.values[layer] = self.values[layer][:, :, -retain:, :]
        return self.keys[layer], self.values[layer]

    @property
    def offset(self) -> int:
        for layer, value in enumerate(self.keys):
            if value is None:
                continue
            if self.compressed_mla:
                return value.shape[1]
            # Windowed layers store a suffix, so the true sequence end is the
            # layer's start plus what it retains. ``_starts`` is 0 without a
            # window, which reproduces the previous shape-only result exactly.
            return self._starts[layer] + value.shape[2]
        return 0

    def nbytes(self) -> int:
        total = sum(a.nbytes for a in (*self.keys, *self.values) if a is not None)
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            total += dsa.nbytes()
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        qwen4 = getattr(self, "qwen4_cache", None)
        if qwen4 is not None:
            total += qwen4.nbytes()
        return total

    def allocated_nbytes(self) -> int:
        return self.nbytes()

    def fork(self) -> "KVCache":
        """Create a copy-on-write cache branch without copying tensor bytes.

        ``update`` replaces each list entry with a newly concatenated array,
        so sharing the already-evaluated prefix arrays is safe.  The method is
        intentionally limited to the plain in-memory cache: paged/spilled
        caches own external resources and expose their own lifetime rules.
        """
        branch = KVCache(len(self.keys))
        branch.keys = list(self.keys)
        branch.values = list(self.values)
        branch.compressed_mla = self.compressed_mla
        branch._windows = list(self._windows)
        branch._starts = list(self._starts)
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            fork_recurrent = getattr(recurrent, "fork", None)
            if not callable(fork_recurrent):
                raise TypeError("recurrent KV companion cannot be forked")
            branch.kda_cache = fork_recurrent()
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            fork_dsa = getattr(dsa, "fork", None)
            if not callable(fork_dsa):
                raise TypeError("DSA KV companion cannot be forked")
            branch.dsa = fork_dsa()
        qwen4 = getattr(self, "qwen4_cache", None)
        if qwen4 is not None:
            fork_qwen4 = getattr(qwen4, "fork", None)
            if not callable(fork_qwen4):
                raise TypeError("Qwen4 auxiliary KV state cannot be forked")
            branch.qwen4_cache = fork_qwen4()
        return branch

    def update_latent(self, layer: int, lat):
        """Append compressed MLA state on its architecture-specific axis."""
        if self.keys[layer] is None:
            self.keys[layer] = lat
        else:
            self.keys[layer] = mx.concatenate([self.keys[layer], lat], axis=1)
        return self.keys[layer]

    def trim(self, length: int):
        """Roll back all generation state to the first ``length`` positions."""
        pending = []
        for i in range(len(self.keys)):
            if self.keys[i] is None:
                continue
            start = self._starts[i]
            if start and length < start:
                # A windowed layer physically no longer holds this position.
                # Silently keeping a longer prefix would misalign every later
                # key, so refuse instead of corrupting the rollback.
                raise ValueError(
                    f"cannot trim layer {i} to {length}: its sliding window "
                    f"retains only positions >= {start}")
            if self.compressed_mla:
                if self.keys[i].shape[1] > length:
                    self.keys[i] = self.keys[i][:, :length, :]
                    pending.append(self.keys[i])
            elif start + self.keys[i].shape[2] > length:
                local = length - start
                self.keys[i] = self.keys[i][:, :, :local, :]
                self.values[i] = self.values[i][:, :, :local, :]
                pending.extend((self.keys[i], self.values[i]))
        if pending:
            # Every slice is independent. One barrier keeps the old backing
            # arrays alive until all replacement views are materialized while
            # avoiding fixed dispatch/synchronization cost once per layer.
            mx.eval(*pending)
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            dsa.trim(length)
        qwen4 = getattr(self, "qwen4_cache", None)
        if qwen4 is not None:
            qwen4.trim(length)

    def layer_lengths(self) -> tuple[int, ...]:
        """Return each layer's local append length.

        Mixed-depth prefill can intentionally give upper attention layers a
        compact suffix while a lower layer anchors the global ``offset``.
        Speculative rollback must therefore checkpoint local lengths instead
        of assuming every layer begins at global position zero.
        """
        axis = 1 if self.compressed_mla else 2
        return tuple(
            0 if value is None else int(value.shape[axis])
            for value in self.keys
        )

    def trim_layer_lengths(self, lengths) -> None:
        """Roll each attention layer back to its checkpoint-local length."""
        targets = tuple(int(value) for value in lengths)
        if len(targets) != len(self.keys) or any(value < 0 for value in targets):
            raise ValueError("invalid per-layer KV rollback lengths")
        pending = []
        for index, target in enumerate(targets):
            keys = self.keys[index]
            if keys is None:
                if target:
                    raise ValueError("cannot restore a missing KV layer")
                continue
            axis = 1 if self.compressed_mla else 2
            current = int(keys.shape[axis])
            if target > current:
                raise ValueError("cannot grow KV during rollback")
            if target == current:
                continue
            if self.compressed_mla:
                self.keys[index] = keys[:, :target, :]
                pending.append(self.keys[index])
            else:
                self.keys[index] = keys[:, :, :target, :]
                self.values[index] = self.values[index][:, :, :target, :]
                pending.extend((self.keys[index], self.values[index]))
        if pending:
            mx.eval(*pending)


class Fp8KVCache(KVCache):
    """Same interface and math-facing contract as KVCache, but the resident
    storage for keys/values is native MLX fp8 (e4m3, packed as uint8) instead
    of the model's bf16 compute dtype -- halves ordinary full-attention KV
    memory. Deliberately NOT applied to any hybrid recurrent (DeltaNet/KDA)
    state: that state is small and fixed-size regardless of context length,
    so there is little memory to save there and this class never touches it.

    2026-07-27: added after independently checking TurboQuant (Zandieh et
    al., ICLR 2026) against real-world evaluation reports rather than its
    own paper numbers -- 3-bit modes showed 15-25 point drops on hard
    reasoning benchmarks (AIME25/LiveCodeBench) at long context despite the
    paper's own "quality neutral" framing, and the community's own practical
    default converged on plain FP8, not the fancier rotation+codebook
    scheme. FP8 needs no calibration, no codebook, and no residual
    correction -- mx.to_fp8/mx.from_fp8 are a direct, stateless format
    conversion. See tests/test_qwen35_oracle.py's FP8 case and
    tests/test_fp8_kv_cache_real_model.py for the actual measured precision
    impact; this is genuinely lossy and defaults OFF
    (VMODEL_QWEN35_FP8_KV_CACHE=1 to opt in) until validated more broadly,
    per CLAUDE.md/AGENTS.md's "Avoiding overfit defaults" rule.
    """

    def __init__(self, num_layers: int, compute_dtype=mx.bfloat16):
        super().__init__(num_layers)
        self._compute_dtype = compute_dtype

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        k8 = mx.to_fp8(k.astype(self._compute_dtype))
        v8 = mx.to_fp8(v.astype(self._compute_dtype))
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k8, v8
        else:
            self.keys[layer] = mx.concatenate([self.keys[layer], k8], axis=2)
            self.values[layer] = mx.concatenate([self.values[layer], v8], axis=2)
        return (
            mx.from_fp8(self.keys[layer], self._compute_dtype),
            mx.from_fp8(self.values[layer], self._compute_dtype),
        )


def fork_hybrid_kv_endpoint(kv: "KVCache") -> "KVCache":
    """Share an evaluated hybrid (DeltaNet/KDA + attention) endpoint cheaply.

    Plain KVCache updates replace attention arrays with concatenations, and
    KDAStateCache updates replace recurrent arrays, so the evaluated prompt
    buffers are immutable under subsequent decode/prefill of the ORIGINAL
    cache. SteppedKVCache has an explicit per-layer copy-on-write fork: spare
    capacity remains shared until one owner appends, then that owner performs
    a bit-preserving gather into a private backing buffer before its update.
    """
    if type(kv) not in (KVCache, SteppedKVCache):
        raise TypeError(
            "hybrid endpoint snapshots require plain or stepped KVCache")
    recurrent = getattr(kv, "kda_cache", None)
    if recurrent is None:
        raise ValueError("hybrid endpoint is missing recurrent state")
    snapshot = kv.fork()
    arrays = [
        value for value in (*snapshot.keys, *snapshot.values)
        if value is not None
    ]
    if arrays:
        mx.eval(*arrays)
    snapshot.kda_cache.synchronize()
    return snapshot


class PositionFreePagePool:
    """Engine-wide immutable K/V pages for position-independent reuse.

    One physical page id names the same logical token in every layer.  Layer
    arrays are separate (attention still consumes one layer at a time), but a
    cache therefore needs only one block table and one refcount per token.  The
    first implementation deliberately uses one-token pages: tool spans can start
    at arbitrary token boundaries, so larger pages would either waste edge
    storage or silently make some reordered spans unshareable.

    Pages become immutable after their layer payload is written.  A page may be
    retained by another cache only after every layer has been written, which is
    the ownership invariant that lets PIC release its source immediately after
    constructing a destination block table.
    """

    block_size = 1

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, *,
                 min_capacity: int = 256):
        if num_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("position-free page geometry must be positive")
        if min_capacity <= 0:
            raise ValueError("position-free minimum capacity must be positive")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.min_capacity = int(min_capacity)
        self.key_pages: list[mx.array | None] = [None] * self.num_layers
        self.value_pages: list[mx.array | None] = [None] * self.num_layers
        self._refs: list[int] = []
        self._written_masks: list[int] = []
        self._free: list[int] = []
        self._next_id = 0
        self._live_pages = 0
        self._capacity = 0
        self._dtype = None
        self._rope_entries: dict[tuple, tuple[int, mx.array, mx.array]] = {}
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def live_pages(self) -> int:
        return self._live_pages

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def reference_count(self, page_id: int) -> int:
        if not 0 <= page_id < self._next_id:
            raise IndexError(page_id)
        return self._refs[page_id]

    def _check_open(self):
        if self._closed:
            raise RuntimeError("position-free page pool is closed")

    def _ensure_capacity(self, required: int):
        if required <= self._capacity:
            return
        if self._capacity == 0:
            target = max(required, self.min_capacity)
        else:
            # A 25% geometric step amortizes decode growth without reserving the
            # 2x slack that would be painful for multi-thousand-token KV states.
            target = max(
                required,
                self._capacity + max(self.min_capacity, self._capacity // 4),
            )
        extra = target - self._capacity
        grown = []
        new_keys: list[mx.array | None] = []
        new_values: list[mx.array | None] = []
        for keys, values in zip(self.key_pages, self.value_pages):
            if keys is None:
                new_keys.append(None)
                new_values.append(None)
                continue
            key_tail = mx.zeros(
                (extra, self.num_kv_heads, 1, self.head_dim), dtype=keys.dtype)
            value_tail = mx.zeros(
                (extra, self.num_kv_heads, 1, self.head_dim), dtype=values.dtype)
            next_keys = mx.concatenate((keys, key_tail), axis=0)
            next_values = mx.concatenate((values, value_tail), axis=0)
            new_keys.append(next_keys)
            new_values.append(next_values)
            grown.extend((next_keys, next_values))
        if grown:
            # Materialize before old arrays lose their final pool reference; this
            # bounds growth to one explicit old+new generation rather than a lazy
            # chain spanning several capacity changes.
            mx.eval(*grown)
        self.key_pages = new_keys
        self.value_pages = new_values
        self._refs.extend([0] * extra)
        self._written_masks.extend([0] * extra)
        self._capacity = target

    def allocate(self, count: int) -> tuple[int, ...]:
        """Allocate ``count`` unique pages and transfer one reference to caller."""
        self._check_open()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("position-free allocation count must be non-negative")
        if count == 0:
            return ()
        recycled_count = min(count, len(self._free))
        new_count = count - recycled_count
        self._ensure_capacity(self._next_id + new_count)
        recycled = self._free[-recycled_count:] if recycled_count else []
        if recycled_count:
            del self._free[-recycled_count:]
        fresh = list(range(self._next_id, self._next_id + new_count))
        self._next_id += new_count
        page_ids = recycled + fresh
        for page_id in page_ids:
            if self._refs[page_id] != 0:
                raise RuntimeError("position-free allocator recycled a live page")
            self._refs[page_id] = 1
            self._live_pages += 1
            self._written_masks[page_id] = 0
        return tuple(page_ids)

    def reserve_additional(self, count: int) -> None:
        """Reserve enough physical ids for ``count`` further allocations."""
        self._check_open()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("position-free reserve count must be non-negative")
        fresh_needed = max(0, count - len(self._free))
        self._ensure_capacity(self._next_id + fresh_needed)

    def retain(self, page_ids) -> None:
        """Add one reference per logical occurrence in ``page_ids``."""
        self._check_open()
        full_mask = (1 << self.num_layers) - 1
        ids = tuple(int(value) for value in page_ids)
        for page_id in ids:
            if not 0 <= page_id < self._next_id or self._refs[page_id] <= 0:
                raise ValueError(f"cannot retain inactive page {page_id}")
            if self._written_masks[page_id] != full_mask:
                raise ValueError(f"cannot share incomplete page {page_id}")
        for page_id in ids:
            self._refs[page_id] += 1

    def release(self, page_ids) -> None:
        """Drop one reference per occurrence and return dead pages to the free list."""
        if self._closed:
            return
        for value in page_ids:
            page_id = int(value)
            if not 0 <= page_id < self._next_id or self._refs[page_id] <= 0:
                raise RuntimeError(f"position-free page {page_id} released twice")
            self._refs[page_id] -= 1
            if self._refs[page_id] == 0:
                self._live_pages -= 1
                self._written_masks[page_id] = 0
                self._free.append(page_id)

    def _ensure_layer(self, layer: int, dtype):
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        if self._dtype is None:
            self._dtype = dtype
        elif self._dtype != dtype:
            raise ValueError(
                f"position-free pool dtype changed: {self._dtype} -> {dtype}")
        if self.key_pages[layer] is None:
            shape = (self._capacity, self.num_kv_heads, 1, self.head_dim)
            self.key_pages[layer] = mx.zeros(shape, dtype=dtype)
            self.value_pages[layer] = mx.zeros(shape, dtype=dtype)

    def write(self, layer: int, page_ids, keys: mx.array, values: mx.array) -> None:
        """Write newly allocated pages for one layer exactly once."""
        self._check_open()
        ids = tuple(int(value) for value in page_ids)
        if (keys.ndim != 4 or values.shape != keys.shape or keys.shape[0] != 1
                or keys.shape[1] != self.num_kv_heads
                or keys.shape[2] != len(ids)
                or keys.shape[3] != self.head_dim):
            raise ValueError("position-free K/V write shape mismatch")
        self._ensure_layer(layer, keys.dtype)
        bit = 1 << layer
        for page_id in ids:
            if not 0 <= page_id < self._next_id or self._refs[page_id] != 1:
                raise ValueError(
                    f"position-free page {page_id} is not exclusively writable")
            if self._written_masks[page_id] & bit:
                raise ValueError(
                    f"position-free page {page_id} layer {layer} written twice")
        if ids:
            # [1,Nkv,L,D] -> [L,Nkv,1,D], matching physical page layout.
            physical_keys = keys.transpose(2, 1, 0, 3)
            physical_values = values.transpose(2, 1, 0, 3)
            contiguous = all(
                page_id == ids[0] + index
                for index, page_id in enumerate(ids))
            if contiguous:
                end = ids[0] + len(ids)
                self.key_pages[layer][ids[0]:end] = physical_keys
                self.value_pages[layer][ids[0]:end] = physical_values
            else:
                indices = mx.array(ids, dtype=mx.int32)
                self.key_pages[layer][indices] = physical_keys
                self.value_pages[layer][indices] = physical_values
            for page_id in ids:
                self._written_masks[page_id] |= bit

    def _validate_read(self, layer: int, page_ids):
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        if self.key_pages[layer] is None:
            raise ValueError(f"position-free layer {layer} has no pages")
        bit = 1 << layer
        for page_id in page_ids:
            if (not 0 <= page_id < self._next_id
                    or self._refs[page_id] <= 0
                    or not (self._written_masks[page_id] & bit)):
                raise ValueError(
                    f"position-free layer {layer} page {page_id} is unreadable")

    def pages(self, layer: int) -> tuple[mx.array, mx.array]:
        self._check_open()
        if not 0 <= layer < self.num_layers or self.key_pages[layer] is None:
            raise ValueError(f"position-free layer {layer} has no materialized pages")
        return self.key_pages[layer], self.value_pages[layer]

    def gather(self, layer: int, page_ids) -> tuple[mx.array, mx.array]:
        """Gather logical order as ordinary ``[1,Nkv,S,D]`` attention tensors."""
        self._check_open()
        ids = tuple(int(value) for value in page_ids)
        self._validate_read(layer, ids)
        keys, values = self.pages(layer)
        contiguous = bool(ids) and all(
            page_id == ids[0] + index for index, page_id in enumerate(ids))
        if contiguous:
            selected_keys = keys[ids[0]:ids[0] + len(ids)]
            selected_values = values[ids[0]:ids[0] + len(ids)]
        else:
            indices = mx.array(ids, dtype=mx.int32)
            selected_keys = keys[indices]
            selected_values = values[indices]
        return (
            selected_keys.transpose(2, 1, 0, 3),
            selected_values.transpose(2, 1, 0, 3),
        )

    def rope(self, max_position: int, theta: float, denominators=None):
        """Return grow-only FP32 cos/sin tables used by the Metal kernel.

        Keeping the trigonometric values in FP32 is important: applying them to
        BF16 keys and then rounding the rotated key to BF16 matches MLX RoPE,
        whereas a BF16 trigonometric table differed by up to one BF16 ulp.
        """
        self._check_open()
        if max_position <= 0:
            raise ValueError("position-free RoPE length must be positive")
        identity = (float(theta), id(denominators))
        existing = self._rope_entries.get(identity)
        if existing is not None and existing[0] >= max_position:
            return existing[1], existing[2]
        target = max_position if existing is None else max(
            max_position, existing[0] + max(256, existing[0] // 4))
        from .pic_attention import rope_cache

        cos, sin = rope_cache(
            target, self.head_dim, theta, dtype=mx.float32,
            denominators=denominators)
        self._rope_entries[identity] = (target, cos, sin)
        return cos, sin

    def bytes_per_page(self) -> int:
        itemsize = int(getattr(self._dtype, "size", 2))
        return (
            self.num_layers * 2 * self.num_kv_heads * self.head_dim
            * self.block_size * itemsize
        )

    def live_nbytes(self) -> int:
        return self.live_pages * self.bytes_per_page()

    def allocated_nbytes(self) -> int:
        return sum(
            value.nbytes for value in (*self.key_pages, *self.value_pages)
            if value is not None
        )

    def close(self):
        self.key_pages = [None] * self.num_layers
        self.value_pages = [None] * self.num_layers
        self._rope_entries.clear()
        self._refs.clear()
        self._written_masks.clear()
        self._free.clear()
        self._next_id = 0
        self._live_pages = 0
        self._capacity = 0
        self._closed = True


class PositionFreeKVCache(KVCache):
    """Logical block table backed by a shared :class:`PositionFreePagePool`.

    ``keys``/``values`` intentionally remain empty compatibility sentinels.  A
    position-free cache must never enter serializers written for dense rotated
    arrays; engine configuration rejects those combinations up front.
    """

    position_free: bool = True
    custom_attention_query_limit: int = 4
    rotated_view_min_keys: int = 1024

    def __init__(self, pool: PositionFreePagePool):
        super().__init__(pool.num_layers)
        self.pool = pool
        self._page_ids: list[int] = []
        self._layer_lengths = [0] * pool.num_layers
        self._block_table_cache = None
        self._key_positions_cache = None
        self._rotated_view = None
        self._released = False

    @property
    def page_ids(self) -> tuple[int, ...]:
        return tuple(self._page_ids)

    @property
    def offset(self) -> int:
        return len(self._page_ids)

    @property
    def is_complete(self) -> bool:
        return (
            not self._released
            and all(length == self.offset for length in self._layer_lengths)
        )

    def _check_owned(self):
        if self._released:
            raise RuntimeError("position-free cache has been released")

    def _invalidate_layout_arrays(self):
        self._block_table_cache = None
        self._key_positions_cache = None

    def update_unrotated(self, layer: int, keys: mx.array,
                         values: mx.array) -> None:
        """Append unrotated (but already attention-scaled) K and ordinary V."""
        self._check_owned()
        if not 0 <= layer < len(self._layer_lengths):
            raise IndexError(layer)
        width = int(keys.shape[2]) if keys.ndim == 4 else -1
        previous = self._layer_lengths[layer]
        if previous == self.offset:
            self._page_ids.extend(self.pool.allocate(width))
            self._invalidate_layout_arrays()
        if width < 0 or previous + width != self.offset:
            raise ValueError(
                "position-free layers must append the same complete token span")
        page_ids = self._page_ids[previous:previous + width]
        self.pool.write(layer, page_ids, keys, values)
        self._layer_lengths[layer] = previous + width

    def reserve_growth(self, positions: int) -> None:
        self._check_owned()
        self.pool.reserve_additional(positions)

    def gather_unrotated(self, layer: int) -> tuple[mx.array, mx.array]:
        self._check_owned()
        if self._layer_lengths[layer] != self.offset:
            raise ValueError(f"position-free layer {layer} is incomplete")
        return self.pool.gather(layer, self._page_ids)

    def has_rotated_view(self, layer: int, length: int) -> bool:
        """Whether a request-local pre-rotated view covers ``length`` keys."""
        view = self._rotated_view
        return bool(
            view is not None
            and 0 <= layer < len(view.keys)
            and view.keys[layer] is not None
            and view._layer_length(layer) == length
        )

    def set_rotated_view(self, layer: int, keys: mx.array,
                         values: mx.array) -> None:
        """Retain an already-built logical SDPA view for this active request.

        The shared pool remains authoritative. This duplicate is deliberately
        request-local and is dropped before the cache returns to the hot LRU.
        """
        self._check_owned()
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("position-free rotated view shape mismatch")
        if keys.shape[2] != self.offset:
            raise ValueError("position-free rotated view length mismatch")
        if self._rotated_view is None:
            # Resolved at call time after this module has defined the class.
            self._rotated_view = SteppedKVCache(len(self._layer_lengths))
        self._rotated_view.keys[layer] = keys
        self._rotated_view.values[layer] = values
        self._rotated_view._lengths[layer] = self.offset

    def update_rotated_view(self, layer: int, keys: mx.array,
                            values: mx.array) -> tuple[mx.array, mx.array]:
        self._check_owned()
        if self._rotated_view is None:
            raise ValueError("position-free rotated view is not initialized")
        return self._rotated_view.update(layer, keys, values)

    def drop_rotated_view(self) -> None:
        self._rotated_view = None

    def rotated_view_nbytes(self) -> int:
        return (
            self._rotated_view.nbytes()
            if self._rotated_view is not None else 0)

    def block_table(self) -> mx.array:
        self._check_owned()
        if self._block_table_cache is None:
            self._block_table_cache = mx.array(
                self._page_ids, dtype=mx.int32)
        return self._block_table_cache

    def key_positions(self) -> mx.array:
        self._check_owned()
        if self._key_positions_cache is None:
            self._key_positions_cache = mx.arange(
                self.offset, dtype=mx.int32)
        return self._key_positions_cache

    def paged_attention(self, layer: int, queries: mx.array,
                        query_positions: mx.array, *, theta: float,
                        denominators=None, scale: float):
        self._check_owned()
        if self._layer_lengths[layer] != self.offset:
            raise ValueError(f"position-free layer {layer} is incomplete")
        from .pic_attention import position_free_paged_attention

        keys, values = self.pool.pages(layer)
        cos, sin = self.pool.rope(
            max(self.offset, int(query_positions.size)), theta, denominators)
        return position_free_paged_attention(
            queries, keys, values, self.block_table(), self.key_positions(),
            query_positions, cos, sin, scale=scale)

    @classmethod
    def from_pic_plan(cls, source: "PositionFreeKVCache", plan,
                      length: int) -> "PositionFreeKVCache":
        """Create an incomplete destination whose reused positions share pages.

        Newly selected positions own fresh pages immediately.  Call
        :meth:`write_selected` once per layer to complete them; on any failure the
        caller must release the destination (the PIC helper does this in a
        ``finally`` path).
        """
        source._check_owned()
        if not source.is_complete:
            raise ValueError("PIC source position-free cache is incomplete")
        if length <= 0 or len(plan.selected_positions) <= 0:
            raise ValueError("PIC destination needs selected positions")
        destination = cls(source.pool)
        allocated = source.pool.allocate(len(plan.selected_positions))
        layout: list[int | None] = [None] * length
        try:
            for position, page_id in zip(plan.selected_positions, allocated):
                if not 0 <= position < length or layout[position] is not None:
                    raise ValueError("invalid PIC selected position layout")
                layout[position] = page_id
            retained: list[int] = []
            for reused in plan.reused:
                if (not 0 <= reused.start < reused.end <= length
                        or reused.source_start < 0
                        or reused.source_start + reused.length > source.offset):
                    raise ValueError("invalid PIC reused position layout")
                source_ids = source._page_ids[
                    reused.source_start:reused.source_start + reused.length]
                for logical, page_id in zip(
                        range(reused.start, reused.end), source_ids):
                    if layout[logical] is not None:
                        raise ValueError("overlapping PIC destination layout")
                    layout[logical] = page_id
                    retained.append(page_id)
            if any(value is None for value in layout):
                raise ValueError("PIC destination layout has uncovered positions")
            source.pool.retain(retained)
            destination._page_ids = [int(value) for value in layout]
            destination._invalidate_layout_arrays()
            return destination
        except Exception:
            source.pool.release(allocated)
            raise

    def write_selected(self, layer: int, positions, keys: mx.array,
                       values: mx.array) -> None:
        self._check_owned()
        selected = tuple(int(value) for value in positions)
        if self._layer_lengths[layer] != 0:
            raise ValueError(f"PIC layer {layer} was already completed")
        if keys.ndim != 4 or keys.shape[2] != len(selected):
            raise ValueError("PIC selected K/V shape mismatch")
        try:
            page_ids = [self._page_ids[position] for position in selected]
        except IndexError as error:
            raise ValueError("PIC selected position is outside destination") from error
        self.pool.write(layer, page_ids, keys, values)
        self._layer_lengths[layer] = self.offset

    def nbytes(self) -> int:
        return self.offset * self.pool.bytes_per_page() + self.rotated_view_nbytes()

    def allocated_nbytes(self) -> int:
        # Logical ownership is the useful per-cache number. Pool capacity is
        # exposed separately because several caches may share the same arrays.
        return self.nbytes()

    def trim(self, length: int):
        self._check_owned()
        if isinstance(length, bool) or not isinstance(length, int):
            raise ValueError("position-free trim length must be an integer")
        if not 0 <= length <= self.offset:
            raise ValueError("position-free trim cannot grow the cache")
        if length == self.offset:
            return
        removed = self._page_ids[length:]
        self._page_ids = self._page_ids[:length]
        self.pool.release(removed)
        self._layer_lengths = [min(value, length) for value in self._layer_lengths]
        if self._rotated_view is not None:
            self._rotated_view.trim(length)
        self._invalidate_layout_arrays()
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            dsa.trim(length)

    def release(self):
        if self._released:
            return
        self.pool.release(self._page_ids)
        self._page_ids.clear()
        self._layer_lengths = [0] * len(self._layer_lengths)
        self.drop_rotated_view()
        self._invalidate_layout_arrays()
        self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Destructors are only a final safety net; engine ownership paths call
            # release explicitly and surface invariant violations there.
            pass


class SteppedKVCache(KVCache):
    """Capacity-stepped exact KV for long-context decode.

    Growing in 256-position blocks avoids recopying a multi-thousand-token
    prefix on every generated token. A 3.5K-token OLMoE measurement improved
    from 113 to 186 tok/s with identical tokens. It is selected only when the
    declared request length crosses the runtime threshold because the simpler
    :class:`KVCache` remains faster for short chats.
    """

    step = 256

    def __init__(self, num_layers: int):
        super().__init__(num_layers)
        self._lengths: list[int] = [0] * num_layers
        self._latent_spill_root: Path | None = None
        self._latent_spill_temporary = None
        self._latent_spill_meta: dict[int, tuple[Path, tuple[int, ...], str]] = {}
        self.latent_spill_bytes_written = 0
        self.latent_spill_bytes_read = 0
        self.latent_spill_layers = 0
        self.latent_spill_reloads = 0
        self.latent_spill_uncached_descriptors = 0
        # A fork shares already-evaluated capacity buffers. MLX slice updates
        # replace bytes inside that capacity, so unlike plain KVCache's concat
        # path they require an explicit detach before the next append.
        self._shared_layers: set[int] = set()

    def enable_latent_disk_spill(self, root: str | Path) -> None:
        """Tier exact compressed-MLA arrays and reload them on demand."""
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._latent_spill_root = root
        self._latent_spill_temporary = tempfile.TemporaryDirectory(
            prefix="voom-k3-mla-kv-", dir=root)

    @property
    def latent_spill_enabled(self) -> bool:
        return self._latent_spill_temporary is not None

    def _latent_spill_directory(self) -> Path:
        if self._latent_spill_temporary is None:
            raise RuntimeError("compressed MLA KV spill is not enabled")
        return Path(self._latent_spill_temporary.name)

    @staticmethod
    def _latent_payload(value: mx.array) -> tuple[bytes, str]:
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            return np.asarray(value.view(mx.uint16)).tobytes(order="C"), "bf16"
        if value.dtype == mx.float16:
            return np.asarray(value).astype(np.float16, copy=False).tobytes(order="C"), "f16"
        if value.dtype == mx.float32:
            return np.asarray(value).astype(np.float32, copy=False).tobytes(order="C"), "f32"
        raise TypeError(f"unsupported compressed MLA spill dtype {value.dtype}")

    @staticmethod
    def _latent_from_payload(
        payload: bytes, shape: tuple[int, ...], dtype: str,
    ) -> mx.array:
        host_dtype = {
            "bf16": np.uint16,
            "f16": np.float16,
            "f32": np.float32,
        }.get(dtype)
        if host_dtype is None:
            raise TypeError(f"unsupported compressed MLA dtype token {dtype}")
        host = np.frombuffer(payload, dtype=host_dtype).reshape(shape)
        value = mx.array(host)
        return value.view(mx.bfloat16) if dtype == "bf16" else value

    def spill_latent_layer(self, layer: int) -> bool:
        """Write one completed MLA layer exactly, then release its array."""
        if not self.latent_spill_enabled or not self.compressed_mla:
            return False
        value = self.keys[layer]
        if value is None:
            return False
        payload, dtype = self._latent_payload(value)
        path = self._latent_spill_directory() / f"layer-{layer:03d}.bin"
        with path.open("wb", buffering=0) as output:
            self.latent_spill_uncached_descriptors += int(
                set_darwin_nocache(output.fileno()))
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        self._latent_spill_meta[layer] = (
            path, tuple(map(int, value.shape)), dtype)
        self.latent_spill_bytes_written += len(payload)
        self.latent_spill_layers += 1
        self.keys[layer] = None
        return True

    def _reload_latent_layer(self, layer: int) -> None:
        metadata = self._latent_spill_meta.get(layer)
        if metadata is None or self.keys[layer] is not None:
            return
        path, shape, dtype = metadata
        with path.open("rb", buffering=0) as source:
            self.latent_spill_uncached_descriptors += int(
                set_darwin_nocache(source.fileno()))
            payload = source.read()
        value = self._latent_from_payload(payload, shape, dtype)
        mx.eval(value)
        self.keys[layer] = value
        self.latent_spill_bytes_read += len(payload)
        self.latent_spill_reloads += 1

    def materialize_latent_layer_for_persistence(self, layer: int):
        """Return an exact MLA layer even after layer-stationary disk spill.

        Durable hot-KV snapshots are written after prefill, when completed K3
        MLA layers may intentionally be absent from ``keys``.  Persistence is
        not allowed to mistake that absence for an architecture without an
        attention layer; reload the immutable spill payload before slicing.
        """
        self._reload_latent_layer(layer)
        return self.keys[layer]

    def latent_spill_stats(self) -> dict[str, int]:
        return {
            "layers": self.latent_spill_layers,
            "bytes_written": self.latent_spill_bytes_written,
            "bytes_read": self.latent_spill_bytes_read,
            "reloads": self.latent_spill_reloads,
            "resident_bytes": sum(
                value.nbytes for value in self.keys if value is not None),
            "uncached_descriptors": self.latent_spill_uncached_descriptors,
        }

    @classmethod
    def from_cache(cls, cache: KVCache) -> "KVCache":
        if isinstance(cache, cls):
            return cache
        result = cls(len(cache.keys))
        result.keys = list(cache.keys)
        result.values = list(cache.values)
        result.compressed_mla = cache.compressed_mla
        for layer, key in enumerate(result.keys):
            if key is not None:
                result._lengths[layer] = (
                    key.shape[1] if result.compressed_mla else key.shape[2]
                )
        # F92: kda_cache is a KDAStateCache (Kimi Linear), structurally
        # unrelated to the token-indexed key/value arrays this method
        # rebuilds -- must be carried over unchanged or it's silently
        # dropped, leaving KDA layers stateless with no error.
        for attribute in (
            "dsa",
            "mla_absorbed",
            "mla_absorbed_prefill",
            "mla_absorbed_key_tile_size",
            "glm53_sparse_absorbed_mla",
            "glm53_sparse_fused_attention",
            "kda_cache",
        ):
            if hasattr(cache, attribute):
                setattr(result, attribute, getattr(cache, attribute))
        return result

    def fork(self) -> "SteppedKVCache":
        """Share an exact endpoint and detach each layer on first mutation.

        Temporary latent-spill files have one cleanup owner and therefore
        cannot safely be shared. GLM-5.3's resident compressed-latent cache is
        the intended path; K3 spill continues to fail closed until its disk
        payload has reference-counted ownership.
        """
        if self.latent_spill_enabled or self._latent_spill_meta:
            raise TypeError(
                "stepped KV snapshots do not support temporary latent spill")
        branch = SteppedKVCache(len(self.keys))
        branch.keys = list(self.keys)
        branch.values = list(self.values)
        branch.compressed_mla = self.compressed_mla
        branch._windows = list(self._windows)
        branch._starts = list(self._starts)
        branch._lengths = list(self._lengths)
        for attribute in (
            "mla_absorbed",
            "mla_absorbed_prefill",
            "mla_absorbed_key_tile_size",
            "glm53_sparse_absorbed_mla",
            "glm53_sparse_fused_attention",
        ):
            if hasattr(self, attribute):
                setattr(branch, attribute, getattr(self, attribute))
        for attribute, label in (
            ("kda_cache", "recurrent KV"),
            ("dsa", "DSA KV"),
            ("qwen4_cache", "Qwen4 auxiliary KV"),
        ):
            companion = getattr(self, attribute, None)
            if companion is None:
                continue
            fork_companion = getattr(companion, "fork", None)
            if not callable(fork_companion):
                raise TypeError(f"{label} companion cannot be forked")
            setattr(branch, attribute, fork_companion())
        shared = {
            layer for layer, value in enumerate(self.keys)
            if value is not None
        }
        self._shared_layers.update(shared)
        branch._shared_layers.update(shared)
        arrays = [
            value for value in (*branch.keys, *branch.values)
            if value is not None
        ]
        if arrays:
            mx.eval(*arrays)
        return branch

    @staticmethod
    def _exact_buffer_copy(value: mx.array, axis: int) -> mx.array:
        """Allocate a byte-identical MLX buffer without floating arithmetic."""
        indices = mx.arange(int(value.shape[axis]), dtype=mx.int32)
        copied = mx.take(value, indices, axis=axis)
        mx.eval(copied)
        return copied

    def _detach_shared_layer(self, layer: int) -> None:
        if layer not in self._shared_layers:
            return
        key = self.keys[layer]
        if key is not None:
            axis = 1 if self.compressed_mla else 2
            self.keys[layer] = self._exact_buffer_copy(key, axis)
            value = self.values[layer]
            if value is not None:
                self.values[layer] = self._exact_buffer_copy(value, axis)
        self._shared_layers.discard(layer)

    def _layer_length(self, layer: int) -> int:
        length = self._lengths[layer]
        value = self.keys[layer]
        if not length and value is not None:
            length = value.shape[1] if self.compressed_mla else value.shape[2]
            self._lengths[layer] = length
        return length

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        self._detach_shared_layer(layer)
        previous = self._layer_length(layer)
        end = previous + k.shape[2]
        current = self.keys[layer]
        if current is None or end > current.shape[2]:
            blocks = (self.step + k.shape[2] - 1) // self.step
            new_k = mx.zeros(
                (*k.shape[:2], blocks * self.step, k.shape[3]), dtype=k.dtype)
            new_v = mx.zeros(
                (*v.shape[:2], blocks * self.step, v.shape[3]), dtype=v.dtype)
            if current is not None:
                old_k = current
                old_v = self.values[layer]
                if previous % self.step:
                    old_k = old_k[..., :previous, :]
                    old_v = old_v[..., :previous, :]
                new_k = mx.concatenate([old_k, new_k], axis=2)
                new_v = mx.concatenate([old_v, new_v], axis=2)
            self.keys[layer], self.values[layer] = new_k, new_v
        self.keys[layer][..., previous:end, :] = k
        self.values[layer][..., previous:end, :] = v
        self._lengths[layer] = end
        return (
            self.keys[layer][..., :end, :],
            self.values[layer][..., :end, :],
        )

    def update_latent(self, layer: int, lat: mx.array) -> mx.array:
        """Append exact compressed MLA latents into spare axis-1 capacity.

        Compressed MLA stores ``[c_kv | k_rope]`` as ``(B, positions,
        latent_width)`` rather than ordinary per-head K/V.  The previous
        :class:`KVCache` implementation concatenated the complete prefix for
        every prefill tile and decode token.  Supporting the architecture's
        native axis here preserves the exact latent bytes while reducing that
        quadratic copy schedule to one growth copy per ``step`` positions.
        """
        if not self.compressed_mla:
            raise ValueError(
                "update_latent requires compressed_mla=True"
            )
        self._reload_latent_layer(layer)
        self._detach_shared_layer(layer)
        previous = self._layer_length(layer)
        incoming = int(lat.shape[1])
        end = previous + incoming
        current = self.keys[layer]
        if current is None or end > current.shape[1]:
            blocks = (self.step + incoming - 1) // self.step
            grown = mx.zeros(
                (lat.shape[0], blocks * self.step, lat.shape[2]),
                dtype=lat.dtype,
            )
            if current is not None:
                old = current
                if previous % self.step:
                    old = old[:, :previous, :]
                grown = mx.concatenate([old, grown], axis=1)
            self.keys[layer] = grown
        self.keys[layer][:, previous:end, :] = lat
        # MLX updates are functional graphs. Materialize the capacity buffer
        # now so thousands of long-prefill appends cannot retain a chain of
        # prior update expressions even though the logical bytes are bounded.
        mx.eval(self.keys[layer])
        self._lengths[layer] = end
        return self.keys[layer][:, :end, :]

    @property
    def offset(self) -> int:
        # A cache restored by assigning exact arrays predates the private
        # capacity-length side table.  Adopt the first materialized layer on
        # demand, as ``update`` already does through ``_layer_length``.
        return next(
            (length for layer in range(len(self.keys))
             if (length := self._layer_length(layer))),
            0,
        )

    def nbytes(self) -> int:
        total = 0
        for layer, key in enumerate(self.keys):
            if key is None:
                continue
            length = self._layer_length(layer)
            if self.compressed_mla:
                total += key[:, :length, :].nbytes
            else:
                total += key[..., :length, :].nbytes
                total += self.values[layer][..., :length, :].nbytes
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            total += dsa.nbytes()
        qwen4 = getattr(self, "qwen4_cache", None)
        if qwen4 is not None:
            total += qwen4.nbytes()
        return total

    def allocated_nbytes(self) -> int:
        total = sum(a.nbytes for a in (*self.keys, *self.values) if a is not None)
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            total += dsa.nbytes()
        qwen4 = getattr(self, "qwen4_cache", None)
        if qwen4 is not None:
            total += qwen4.nbytes()
        return total

    def trim(self, length: int):
        pending = []
        for layer, key in enumerate(self.keys):
            if key is None and self._layer_length(layer) > length:
                self._reload_latent_layer(layer)
                key = self.keys[layer]
            if key is None or self._layer_length(layer) <= length:
                continue
            if self.compressed_mla:
                self.keys[layer] = key[:, :length, :]
                pending.append(self.keys[layer])
            else:
                self.keys[layer] = key[..., :length, :]
                self.values[layer] = self.values[layer][..., :length, :]
                pending.extend((self.keys[layer], self.values[layer]))
            self._lengths[layer] = length
            self._shared_layers.discard(layer)
        if pending:
            mx.eval(*pending)
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            dsa.trim(length)

    def trim_layer_lengths(self, lengths) -> None:
        """Restore mixed-depth lengths and keep stepped side tables exact."""
        targets = tuple(int(value) for value in lengths)
        if len(targets) != len(self.keys) or any(value < 0 for value in targets):
            raise ValueError("invalid per-layer KV rollback lengths")
        pending = []
        for layer, target in enumerate(targets):
            current = self._layer_length(layer)
            if target > current:
                raise ValueError("cannot grow KV during rollback")
            if target == current:
                continue
            key = self.keys[layer]
            if key is None:
                if target:
                    raise ValueError("cannot restore a missing KV layer")
                self._lengths[layer] = 0
                continue
            if self.compressed_mla:
                self.keys[layer] = key[:, :target, :]
                pending.append(self.keys[layer])
            else:
                self.keys[layer] = key[..., :target, :]
                self.values[layer] = self.values[layer][..., :target, :]
                pending.extend((self.keys[layer], self.values[layer]))
            self._lengths[layer] = target
            self._shared_layers.discard(layer)
        if pending:
            mx.eval(*pending)

    def close_latent_spill(self) -> None:
        self._latent_spill_meta.clear()
        if self._latent_spill_temporary is not None:
            self._latent_spill_temporary.cleanup()
            self._latent_spill_temporary = None

    def __del__(self):
        try:
            self.close_latent_spill()
        except Exception:
            pass
