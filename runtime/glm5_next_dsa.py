"""GLM-5.3 pooled DeepSeek Sparse Attention indexer state."""

from __future__ import annotations

import time

import mlx.core as mx

from . import quant
from .config import ModelConfig
from .layer_runner import _linear


class GLM5NextDSAState:
    """Per-layer cached k-pool inputs and chronological sparse selections."""

    packed_step = 1024
    pool_step = 256

    def __init__(
            self, cfg: ModelConfig, *, incremental_pool_cache: bool = False):
        self.cfg = cfg
        self.incremental_pool_cache = bool(incremental_pool_cache)
        # Compatibility name used by generic KV diagnostics/persistence.
        # Values are [key, compression-gate, valid] packed states, not the
        # older GLM-5.2 index key alone.
        self.k_idx: dict[int, mx.array] = {}
        # The incremental candidate also keeps a stepped backing allocation for
        # raw packed rows. The first pool-cache version still concatenated and
        # recopied the complete prefix every 32-position tile: ~1,465 copies at
        # the real 46.8K capture. A logical k_idx view remains authoritative for
        # persistence/diagnostics; spare rows are zero capacity only.
        self._packed_capacity: dict[int, mx.array] = {}
        self._packed_lengths: dict[int, int] = {}
        # Complete k-pools never change after their final raw row is appended.
        # The released reference rebuilds every prior pool for each prefill
        # tile; this opt-in cache retains only those immutable derived keys.
        # An incomplete tail is still rebuilt from the current raw rows so its
        # causal behavior and eventual completed value remain authoritative.
        self.pool_keys: dict[int, mx.array] = {}
        self._pool_capacity: dict[int, mx.array] = {}
        self._pool_lengths: dict[int, int] = {}
        self.selection: mx.array | None = None
        self.sel_layer = -1
        self.stats = {
            "observations": 0,
            "sparse_selects": 0,
            # Generic GLM telemetry exposes the older GLM-5.2 shared-indexer
            # counter. GLM-5.3's strict config requires every indexer be full,
            # so this field is intentionally and truthfully always zero.
            "shared_reuses": 0,
            "pools_scored": 0,
            "raw_indices_selected": 0,
            "pool_rows_computed": 0,
            "pool_rows_reused": 0,
            "pool_build_s": 0.0,
            "selection_s": 0.0,
            "packed_capacity_grows": 0,
            "packed_rows_copied": 0,
            "packed_rows_appended": 0,
            "packed_capacity_rows_peak": 0,
            "pool_capacity_grows": 0,
            "pool_rows_copied": 0,
            "pool_capacity_rows_peak": 0,
            "pool_metadata_rows_avoided": 0,
        }

    def nbytes(self) -> int:
        capacity_layers = set(self._packed_capacity)
        pool_capacity_layers = set(self._pool_capacity)
        return (
            sum(value.nbytes for value in self._packed_capacity.values())
            + sum(
                value.nbytes for layer, value in self.k_idx.items()
                if layer not in capacity_layers)
            + sum(value.nbytes for value in self._pool_capacity.values())
            + sum(
                value.nbytes for layer, value in self.pool_keys.items()
                if layer not in pool_capacity_layers)
        )

    def fork(self) -> "GLM5NextDSAState":
        branch = GLM5NextDSAState(
            self.cfg, incremental_pool_cache=self.incremental_pool_cache)
        if self.incremental_pool_cache:
            # Do not share a stepped backing allocation across
            # speculative/hot branches. Copy only the logical rows; the first
            # append builds independent spare capacity.
            for layer, packed in self.k_idx.items():
                copied = mx.take(
                    packed,
                    mx.arange(int(packed.shape[1]), dtype=mx.int32),
                    axis=1,
                )
                mx.eval(copied)
                branch.k_idx[layer] = copied
                branch._packed_lengths[layer] = int(copied.shape[1])
        else:
            # Preserve the established immutable-array COW behavior for the
            # default route; it is heavily used by native MTP/hot endpoints.
            branch.k_idx = dict(self.k_idx)
        if self.incremental_pool_cache:
            for layer, pooled in self.pool_keys.items():
                copied = mx.take(
                    pooled,
                    mx.arange(int(pooled.shape[1]), dtype=mx.int32),
                    axis=1,
                )
                mx.eval(copied)
                branch.pool_keys[layer] = copied
                branch._pool_lengths[layer] = int(copied.shape[1])
        else:
            branch.pool_keys = dict(self.pool_keys)
        branch.selection = self.selection
        branch.sel_layer = self.sel_layer
        branch.stats = dict(self.stats)
        return branch

    def trim(self, length: int) -> None:
        length = int(length)
        for layer, packed in list(self.k_idx.items()):
            if packed.shape[1] > length:
                backing = self._packed_capacity.get(layer)
                self.k_idx[layer] = (
                    backing[:, :length, :]
                    if backing is not None else packed[:, :length, :])
                self._packed_lengths[layer] = length
                mx.eval(self.k_idx[layer])
            cached = self.pool_keys.get(layer)
            complete = length // self.cfg.index_kpool
            if cached is not None and int(cached.shape[1]) > complete:
                if complete:
                    backing = self._pool_capacity.get(layer)
                    self.pool_keys[layer] = (
                        backing[:, :complete, :]
                        if backing is not None else cached[:, :complete, :])
                    self._pool_lengths[layer] = complete
                    mx.eval(self.pool_keys[layer])
                else:
                    self.pool_keys.pop(layer, None)
                    self._pool_lengths[layer] = 0
        self.selection = None
        self.sel_layer = -1

    def observe(
            self, layer: int, indexer_type: str, hidden: mx.array,
            w: dict, prefix: str, offset: int = 0) -> None:
        if indexer_type != "full":
            raise ValueError("GLM-5.3 only supports full per-layer indexers")
        k_new = _linear(hidden, w, f"{prefix}.self_attn.indexer.wk")
        k_new = mx.fast.layer_norm(
            k_new,
            w[f"{prefix}.self_attn.indexer.k_norm.weight"],
            w[f"{prefix}.self_attn.indexer.k_norm.bias"],
            1e-6,
        )
        # This checkpoint entry is a bare Parameter, not an nn.Linear module,
        # so its canonical tensor name deliberately has no ``.weight`` suffix.
        gate_weight = w[
            f"{prefix}.self_attn.indexer.index_kpool_compress_gate"]
        gate_new = hidden.astype(gate_weight.dtype) @ gate_weight.T
        valid = mx.ones((*k_new.shape[:-1], 1), dtype=k_new.dtype)
        packed = mx.concatenate([k_new, gate_new, valid], axis=-1)
        self._append_packed(layer, packed, offset)
        self.stats["observations"] += 1

    def _append_packed(
            self, layer: int, packed: mx.array, offset: int) -> None:
        """Append already-derived index rows using the configured schedule.

        Keeping the storage operation separate makes its copy volume and
        exactness independently measurable at the released 257-value row
        width; the model-facing ``observe`` path remains the sole projection
        producer and observation counter.
        """
        previous = self.k_idx.get(layer)
        if previous is not None and int(previous.shape[1]) != int(offset):
            raise ValueError(
                "GLM-5.3 indexer cache offset does not match its packed "
                f"history: offset={offset}, cached={previous.shape[1]}")
        if previous is None and int(offset) != 0:
            raise ValueError(
                "GLM-5.3 indexer cannot begin at a nonzero offset")
        if self.incremental_pool_cache:
            incoming = int(packed.shape[1])
            end = int(offset) + incoming
            backing = self._packed_capacity.get(layer)
            if backing is None or int(backing.shape[1]) < end:
                target = (
                    (end + self.packed_step - 1) // self.packed_step
                    * self.packed_step)
                tail = mx.zeros(
                    (int(packed.shape[0]), target - int(offset),
                     int(packed.shape[2])),
                    dtype=packed.dtype,
                )
                backing = (
                    tail if previous is None
                    else mx.concatenate((previous, tail), axis=1))
                self.stats["packed_capacity_grows"] += 1
                self.stats["packed_rows_copied"] += int(offset)
            backing[:, int(offset):end, :] = packed
            mx.eval(backing)
            self._packed_capacity[layer] = backing
            self._packed_lengths[layer] = end
            self.k_idx[layer] = backing[:, :end, :]
            self.stats["packed_rows_appended"] += incoming
            self.stats["packed_capacity_rows_peak"] = max(
                self.stats["packed_capacity_rows_peak"],
                int(backing.shape[1]),
            )
        else:
            self.k_idx[layer] = (
                packed if previous is None
                else mx.concatenate([previous, packed], axis=1))
            mx.eval(self.k_idx[layer])

    def _pooled_states_reference(
            self, packed: mx.array, w: dict, prefix: str, *,
            include_metadata: bool = True,
    ) -> tuple[mx.array, mx.array | None, mx.array | None]:
        cfg = self.cfg
        dim = cfg.index_head_dim
        pool = cfg.index_kpool
        keys = packed[..., :dim]
        gates = packed[..., dim:2 * dim]
        valid = packed[..., 2 * dim] > 0
        batch, length, _ = keys.shape
        pool_count = (length + pool - 1) // pool
        padded = pool_count * pool
        if padded != length:
            pad = padded - length
            keys = mx.concatenate([
                keys, mx.zeros((batch, pad, dim), dtype=keys.dtype)], axis=1)
            gates = mx.concatenate([
                gates, mx.zeros((batch, pad, dim), dtype=gates.dtype)], axis=1)
            valid = mx.concatenate([
                valid, mx.zeros((batch, pad), dtype=mx.bool_)], axis=1)
        grouped_keys = keys.reshape(batch, pool_count, pool, dim)
        grouped_gates = gates.reshape(batch, pool_count, pool, dim)
        grouped_valid = valid.reshape(batch, pool_count, pool)
        if include_metadata:
            offsets = mx.arange(padded, dtype=mx.int32).reshape(
                pool_count, pool)
            indices = mx.broadcast_to(
                offsets[None], (batch, pool_count, pool))
            indices = mx.where(grouped_valid, indices, -1)
            pool_valid = mx.all(grouped_valid, axis=-1)
        else:
            indices = None
            pool_valid = None

        ape = w[
            f"{prefix}.self_attn.indexer.index_kpool_compress_ape"]
        logits = grouped_gates.astype(mx.float32) + ape.astype(mx.float32)[
            None, None, :, :]
        logits = mx.where(
            grouped_valid[..., None], logits, float("-inf"))
        probabilities = mx.nan_to_num(
            mx.softmax(logits, axis=2)).astype(grouped_keys.dtype)
        pool_keys = mx.sum(probabilities * grouped_keys, axis=2)
        return pool_keys, indices, pool_valid

    def _pooled_states(
            self, packed: mx.array, w: dict, prefix: str, *, layer: int,
            include_metadata: bool = True,
    ) -> tuple[mx.array, mx.array | None, mx.array | None]:
        """Return released pooled keys, reusing only immutable full pools.

        Pool compression reduces solely along the fixed ``index_kpool`` axis;
        complete pools are independent of later rows.  The metadata and any
        incomplete tail are still rebuilt at the full current length.  This
        makes the cache a scheduling optimization rather than a change to the
        selected candidate set.  It remains explicit until real-model greedy
        and endpoint gates establish byte-identical behavior on Metal.
        """
        if not self.incremental_pool_cache:
            return self._pooled_states_reference(
                packed, w, prefix, include_metadata=include_metadata)

        pool = self.cfg.index_kpool
        length = int(packed.shape[1])
        complete = length // pool
        cached = self.pool_keys.get(layer)
        cached_count = 0 if cached is None else int(cached.shape[1])
        if cached_count > complete:
            cached = cached[:, :complete, :]
            cached_count = complete

        if cached_count < complete:
            build_started = time.perf_counter()
            start = cached_count * pool
            end = complete * pool
            derived, _indices, _valid = self._pooled_states_reference(
                packed[:, start:end, :], w, prefix,
                include_metadata=False)
            mx.eval(derived)
            backing = self._pool_capacity.get(layer)
            if backing is None or int(backing.shape[1]) < complete:
                target = (
                    (complete + self.pool_step - 1) // self.pool_step
                    * self.pool_step)
                tail = mx.zeros(
                    (int(derived.shape[0]), target - cached_count,
                     int(derived.shape[2])),
                    dtype=derived.dtype,
                )
                backing = (
                    tail if cached is None
                    else mx.concatenate((cached, tail), axis=1))
                self.stats["pool_capacity_grows"] += 1
                self.stats["pool_rows_copied"] += cached_count
            backing[:, cached_count:complete, :] = derived
            mx.eval(backing)
            self._pool_capacity[layer] = backing
            self._pool_lengths[layer] = complete
            cached = backing[:, :complete, :]
            self.pool_keys[layer] = cached
            self.stats["pool_rows_computed"] += complete - cached_count
            self.stats["pool_build_s"] += time.perf_counter() - build_started
        elif cached is not None:
            backing = self._pool_capacity.get(layer)
            if backing is not None:
                cached = backing[:, :complete, :]
                self._pool_lengths[layer] = complete
            self.pool_keys[layer] = cached
        backing = self._pool_capacity.get(layer)
        if backing is not None:
            self.stats["pool_capacity_rows_peak"] = max(
                self.stats["pool_capacity_rows_peak"],
                int(backing.shape[1]),
            )
        self.stats["pool_rows_reused"] += cached_count

        # Recreate current chronological indices/validity exactly as the
        # released full calculation does.  This metadata is tiny and avoids a
        # separate candidate-order implementation.
        if length % pool:
            build_started = time.perf_counter()
            tail_keys, _indices, _valid = self._pooled_states_reference(
                packed[:, complete * pool:, :], w, prefix,
                include_metadata=False)
            mx.eval(tail_keys)
            self.stats["pool_build_s"] += time.perf_counter() - build_started
            self.stats["pool_rows_computed"] += 1
            pool_keys = (
                tail_keys if cached is None
                else mx.concatenate([cached, tail_keys], axis=1))
            # Tail-local metadata has relative indices; use the full reference
            # metadata while discarding its recomputed keys.  Computing that
            # full key tensor would defeat the optimization, so construct the
            # equivalent integer/valid arrays directly below.
        else:
            pool_keys = cached

        if not include_metadata:
            return pool_keys, None, None

        batch = int(packed.shape[0])
        pool_count = (length + pool - 1) // pool
        padded = pool_count * pool
        valid = packed[..., 2 * self.cfg.index_head_dim] > 0
        if padded != length:
            valid = mx.concatenate([
                valid,
                mx.zeros((batch, padded - length), dtype=mx.bool_),
            ], axis=1)
        grouped_valid = valid.reshape(batch, pool_count, pool)
        offsets = mx.arange(padded, dtype=mx.int32).reshape(pool_count, pool)
        indices = mx.broadcast_to(
            offsets[None], (batch, pool_count, pool))
        indices = mx.where(grouped_valid, indices, -1)
        pool_valid = mx.all(grouped_valid, axis=-1)
        return pool_keys, indices, pool_valid

    def update_and_select(
            self, layer: int, indexer_type: str, hidden: mx.array,
            q_resid: mx.array, w: dict, prefix: str, offset: int,
    ) -> mx.array | None:
        if indexer_type != "full":
            raise ValueError("GLM-5.3 only supports full per-layer indexers")
        selection_started = time.perf_counter()
        packed = self.k_idx.get(layer)
        if packed is None or packed.shape[1] <= self.cfg.index_topk:
            self.selection = None
            return None

        cfg = self.cfg
        batch, query_length, _ = hidden.shape
        key_length = int(packed.shape[1])
        heads, dim = cfg.index_n_heads, cfg.index_head_dim
        query = _linear(
            q_resid, w, f"{prefix}.self_attn.indexer.wq_b").reshape(
                batch, query_length, heads, dim)
        pool_keys, _pool_indices, _pool_valid = self._pooled_states(
            packed, w, prefix, layer=layer, include_metadata=False)
        pool_count = int(pool_keys.shape[1])
        pool = int(cfg.index_kpool)
        scores = mx.einsum(
            "blhd,bpd->blhp", query.astype(mx.float32),
            pool_keys.astype(mx.float32))
        scores = mx.maximum(scores * (dim ** -0.5), 0.0)

        weights_proj = w[
            f"{prefix}.self_attn.indexer.weights_proj.weight"]
        if isinstance(weights_proj, quant.QTensor):
            head_weights = quant.matmul(hidden, weights_proj)
        else:
            head_weights = hidden.astype(
                weights_proj.dtype) @ weights_proj.T
        head_weights = head_weights.astype(mx.float32) * (heads ** -0.5)
        scores = mx.sum(
            scores * head_weights[..., None], axis=2)

        # A pool is visible only once its final raw token is causal for that
        # query. Incomplete pools are never selected; their visible suffix is
        # appended explicitly below.
        # Every packed row comes from ``observe`` and carries valid=1. Pool
        # metadata is therefore an exact arithmetic mapping: pool p owns raw
        # rows p*k..p*k+k-1, and only a complete/causal pool is eligible.
        # Avoid rebuilding [B, pools, k] indices and broadcasting them across
        # every query tile at long context.
        pool_end = (
            mx.arange(pool_count, dtype=mx.int32) * pool + (pool - 1))
        pool_valid = pool_end < key_length
        query_pos = mx.arange(
            offset, offset + query_length, dtype=mx.int32)
        visible = (
            pool_end[None, None, :] <= query_pos[None, :, None])
        candidates = visible & pool_valid[None, None, :]
        scores = mx.where(candidates, scores, float("-inf"))
        select_count = min(cfg.index_topk // cfg.index_kpool, pool_count)
        chosen = mx.argpartition(
            -scores, kth=select_count - 1, axis=-1)[..., :select_count]
        chosen_valid = mx.take_along_axis(candidates, chosen, axis=-1)
        expanded = (
            chosen[..., None] * pool
            + mx.arange(pool, dtype=mx.int32)[None, None, None, :])
        expanded = mx.where(
            chosen_valid[..., None], expanded, -1).reshape(
                batch, query_length, select_count * pool)
        self.stats["pool_metadata_rows_avoided"] += pool_count

        if cfg.index_kpool_always_select_tail and cfg.index_kpool > 1:
            tail_width = cfg.index_kpool - 1
            visible_count = query_pos + 1
            tail_count = visible_count % cfg.index_kpool
            tail_start = visible_count - tail_count
            tail_offsets = mx.arange(tail_width, dtype=mx.int32)
            tail = tail_start[:, None] + tail_offsets[None, :]
            tail_valid = (
                tail_offsets[None, :] < tail_count[:, None]) & (
                    tail < key_length)
            tail = mx.where(tail_valid, tail, -1)
            tail = mx.broadcast_to(
                tail[None], (batch, query_length, tail_width))
            expanded = mx.concatenate([expanded, tail], axis=-1)

        # Official attention scatters IDs into a mask over chronological K/V.
        # A compact gather must restore that chronological order explicitly.
        sentinel = mx.array(key_length, dtype=mx.int32)
        ordered = mx.sort(mx.where(expanded >= 0, expanded, sentinel), axis=-1)
        ordered = mx.where(ordered < key_length, ordered, -1)
        mx.eval(ordered)
        self.selection = ordered
        self.sel_layer = layer
        self.stats["sparse_selects"] += 1
        self.stats["pools_scored"] += pool_count * query_length
        self.stats["raw_indices_selected"] += int(
            mx.sum(ordered >= 0).item())
        self.stats["selection_s"] += time.perf_counter() - selection_started
        return ordered
