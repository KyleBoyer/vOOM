"""Resident per-layer KV caches with one shared update/trim interface."""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """Exact-length concatenating cache; fastest for short conversations."""

    compressed_mla: bool = False

    def __init__(self, num_layers: int):
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = mx.concatenate([self.keys[layer], k], axis=2)
            self.values[layer] = mx.concatenate([self.values[layer], v], axis=2)
        return self.keys[layer], self.values[layer]

    @property
    def offset(self) -> int:
        first = next((value for value in self.keys if value is not None), None)
        if first is None:
            return 0
        return first.shape[1] if self.compressed_mla else first.shape[2]

    def nbytes(self) -> int:
        total = sum(a.nbytes for a in (*self.keys, *self.values) if a is not None)
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        return total

    def allocated_nbytes(self) -> int:
        return self.nbytes()

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
            if self.compressed_mla:
                if self.keys[i].shape[1] > length:
                    self.keys[i] = self.keys[i][:, :length, :]
                    pending.append(self.keys[i])
            elif self.keys[i].shape[2] > length:
                self.keys[i] = self.keys[i][:, :, :length, :]
                self.values[i] = self.values[i][:, :, :length, :]
                pending.extend((self.keys[i], self.values[i]))
        if pending:
            # Every slice is independent. One barrier keeps the old backing
            # arrays alive until all replacement views are materialized while
            # avoiding fixed dispatch/synchronization cost once per layer.
            mx.eval(*pending)
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            dsa.trim(length)


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

    @classmethod
    def from_cache(cls, cache: KVCache) -> "KVCache":
        if isinstance(cache, cls) or cache.compressed_mla:
            return cache
        result = cls(len(cache.keys))
        result.keys = list(cache.keys)
        result.values = list(cache.values)
        for layer, key in enumerate(result.keys):
            if key is not None:
                result._lengths[layer] = key.shape[2]
        # F92: kda_cache is a KDAStateCache (Kimi Linear), structurally
        # unrelated to the token-indexed key/value arrays this method
        # rebuilds -- must be carried over unchanged or it's silently
        # dropped, leaving KDA layers stateless with no error.
        for attribute in ("dsa", "mla_absorbed", "kda_cache"):
            if hasattr(cache, attribute):
                setattr(result, attribute, getattr(cache, attribute))
        return result

    def _layer_length(self, layer: int) -> int:
        length = self._lengths[layer]
        value = self.keys[layer]
        if not length and value is not None:
            length = value.shape[1] if self.compressed_mla else value.shape[2]
            self._lengths[layer] = length
        return length

    def update(self, layer: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
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

    @property
    def offset(self) -> int:
        first = next((i for i, value in enumerate(self.keys)
                      if value is not None), None)
        return 0 if first is None else self._layer_length(first)

    def nbytes(self) -> int:
        total = 0
        for layer, key in enumerate(self.keys):
            if key is None:
                continue
            length = self._layer_length(layer)
            total += key[..., :length, :].nbytes
            total += self.values[layer][..., :length, :].nbytes
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        return total

    def allocated_nbytes(self) -> int:
        total = sum(a.nbytes for a in (*self.keys, *self.values) if a is not None)
        recurrent = getattr(self, "kda_cache", None)
        if recurrent is not None:
            total += recurrent.nbytes()
        return total

    def trim(self, length: int):
        pending = []
        for layer, key in enumerate(self.keys):
            if key is None or self._layer_length(layer) <= length:
                continue
            self.keys[layer] = key[..., :length, :]
            self.values[layer] = self.values[layer][..., :length, :]
            self._lengths[layer] = length
            pending.extend((self.keys[layer], self.values[layer]))
        if pending:
            mx.eval(*pending)
        dsa = getattr(self, "dsa", None)
        if dsa is not None:
            dsa.trim(length)
