"""GLM-5.3 pooled DeepSeek Sparse Attention indexer state."""

from __future__ import annotations

import time

import mlx.core as mx

from . import quant
from .config import ModelConfig
from .layer_runner import _linear


class GLM5NextDSAState:
    """Per-layer cached k-pool inputs and chronological sparse selections."""

    def __init__(
            self, cfg: ModelConfig, *, incremental_pool_cache: bool = False):
        self.cfg = cfg
        self.incremental_pool_cache = bool(incremental_pool_cache)
        # Compatibility name used by generic KV diagnostics/persistence.
        # Values are [key, compression-gate, valid] packed states, not the
        # older GLM-5.2 index key alone.
        self.k_idx: dict[int, mx.array] = {}
        # Complete k-pools never change after their final raw row is appended.
        # The released reference rebuilds every prior pool for each prefill
        # tile; this opt-in cache retains only those immutable derived keys.
        # An incomplete tail is still rebuilt from the current raw rows so its
        # causal behavior and eventual completed value remain authoritative.
        self.pool_keys: dict[int, mx.array] = {}
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
        }

    def nbytes(self) -> int:
        return (
            sum(value.nbytes for value in self.k_idx.values())
            + sum(value.nbytes for value in self.pool_keys.values())
        )

    def fork(self) -> "GLM5NextDSAState":
        branch = GLM5NextDSAState(
            self.cfg, incremental_pool_cache=self.incremental_pool_cache)
        branch.k_idx = dict(self.k_idx)
        branch.pool_keys = dict(self.pool_keys)
        branch.selection = self.selection
        branch.sel_layer = self.sel_layer
        branch.stats = dict(self.stats)
        return branch

    def trim(self, length: int) -> None:
        length = int(length)
        for layer, packed in list(self.k_idx.items()):
            if packed.shape[1] > length:
                self.k_idx[layer] = packed[:, :length, :]
                mx.eval(self.k_idx[layer])
            cached = self.pool_keys.get(layer)
            complete = length // self.cfg.index_kpool
            if cached is not None and int(cached.shape[1]) > complete:
                if complete:
                    self.pool_keys[layer] = cached[:, :complete, :]
                    mx.eval(self.pool_keys[layer])
                else:
                    self.pool_keys.pop(layer, None)
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
        previous = self.k_idx.get(layer)
        if previous is not None and int(previous.shape[1]) != int(offset):
            raise ValueError(
                "GLM-5.3 indexer cache offset does not match its packed "
                f"history: offset={offset}, cached={previous.shape[1]}")
        if previous is None and int(offset) != 0:
            raise ValueError(
                "GLM-5.3 indexer cannot begin at a nonzero offset")
        self.k_idx[layer] = (
            packed if previous is None
            else mx.concatenate([previous, packed], axis=1))
        mx.eval(self.k_idx[layer])
        self.stats["observations"] += 1

    def _pooled_states_reference(
            self, packed: mx.array, w: dict, prefix: str,
    ) -> tuple[mx.array, mx.array, mx.array]:
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
        offsets = mx.arange(padded, dtype=mx.int32).reshape(
            pool_count, pool)
        indices = mx.broadcast_to(
            offsets[None], (batch, pool_count, pool))
        indices = mx.where(grouped_valid, indices, -1)
        pool_valid = mx.all(grouped_valid, axis=-1)

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
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return released pooled keys, reusing only immutable full pools.

        Pool compression reduces solely along the fixed ``index_kpool`` axis;
        complete pools are independent of later rows.  The metadata and any
        incomplete tail are still rebuilt at the full current length.  This
        makes the cache a scheduling optimization rather than a change to the
        selected candidate set.  It remains explicit until real-model greedy
        and endpoint gates establish byte-identical behavior on Metal.
        """
        if not self.incremental_pool_cache:
            return self._pooled_states_reference(packed, w, prefix)

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
                packed[:, start:end, :], w, prefix)
            mx.eval(derived)
            cached = (
                derived if cached is None
                else mx.concatenate([cached, derived], axis=1))
            mx.eval(cached)
            self.pool_keys[layer] = cached
            self.stats["pool_rows_computed"] += complete - cached_count
            self.stats["pool_build_s"] += time.perf_counter() - build_started
        elif cached is not None:
            self.pool_keys[layer] = cached
        self.stats["pool_rows_reused"] += cached_count

        # Recreate current chronological indices/validity exactly as the
        # released full calculation does.  This metadata is tiny and avoids a
        # separate candidate-order implementation.
        if length % pool:
            build_started = time.perf_counter()
            tail_keys, _indices, _valid = self._pooled_states_reference(
                packed[:, complete * pool:, :], w, prefix)
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
        pool_keys, pool_indices, pool_valid = self._pooled_states(
            packed, w, prefix, layer=layer)
        pool_count = int(pool_keys.shape[1])
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
        pool_end = mx.maximum(pool_indices[..., -1], 0)
        query_pos = mx.arange(
            offset, offset + query_length, dtype=mx.int32)
        visible = (
            pool_end[:, None, :] <= query_pos[None, :, None])
        candidates = visible & pool_valid[:, None, :]
        scores = mx.where(candidates, scores, float("-inf"))
        select_count = min(cfg.index_topk // cfg.index_kpool, pool_count)
        chosen = mx.argpartition(
            -scores, kth=select_count - 1, axis=-1)[..., :select_count]
        chosen_valid = mx.take_along_axis(candidates, chosen, axis=-1)
        expanded = mx.take_along_axis(
            mx.broadcast_to(
                pool_indices[:, None, :, :],
                (batch, query_length, pool_count, cfg.index_kpool)),
            mx.broadcast_to(
                chosen[..., None],
                (batch, query_length, select_count, cfg.index_kpool)),
            axis=2)
        expanded = mx.where(
            chosen_valid[..., None], expanded, -1).reshape(
                batch, query_length, select_count * cfg.index_kpool)

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
