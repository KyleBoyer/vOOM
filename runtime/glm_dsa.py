"""F22: GLM-5.2 DSA (DeepSeek Sparse Attention) indexer + IndexShare.

Checkpoint weights per layer: self_attn.indexer.{wq_b [4096,2048], wk [128,6144],
k_norm (+bias) [128], weights_proj [32,6144]}. Config: index_n_heads=32,
index_head_dim=128, index_topk=2048, indexer_types per layer ('full'|'shared'),
index_topk_freq=4 (a 'full' layer's selection is reused by following 'shared'
layers), indexer_rope_interleave=True.

Lightning-indexer scoring (DeepSeek-V3.2 family):
    q_idx  = wq_b(q_a_latent)            -> (B, L, 32, 128)
    k_idx  = k_norm(wk(h))               -> (B, S, 128)      (cached per full layer)
    w_head = weights_proj(h)             -> (B, L, 32)
    score(t, s) = sum_j w_head[t,j] * relu(q_idx[t,j] . k_idx[s])
    keep top index_topk positions per query t (causal).

EXACTNESS: for S <= index_topk every position is selected, so the dense path is
mathematically identical — the sparse branch only activates beyond 2,048 cached
positions. Above that boundary selection is scored in bounded key tiles and
merged with an explicit total order (higher score, then lower absolute key
position). Compact attention restores chronological key order. This remains a
conformance candidate until the F33/F75 real-model output and token gates pass.

Integration: glm._mla_attention consults DSAState when S > index_topk. Decode
(L=1) gathers the selected latent rows before kv_b expansion, so the F21
compressed cache also cuts the EXPANSION cost to top-k rows only.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

import mlx.core as mx
import numpy as np

from .config import ModelConfig
from . import quant
from .layer_runner import _linear


class DSAState:
    """Per-generation indexer state: cached k_idx per 'full' layer and the
    selection shared with subsequent 'shared' layers."""

    def __init__(
        self, cfg: ModelConfig, *, key_tile_size: int = 256,
        index_step_size: int = 0, selection_query_tile_size: int = 0,
        index_preallocate: bool = False,
        selection_spill_dir: str = "",
    ):
        if key_tile_size <= 0:
            raise ValueError("DSA key_tile_size must be positive")
        if index_step_size < 0:
            raise ValueError("DSA index_step_size must be non-negative")
        if selection_query_tile_size < 0:
            raise ValueError(
                "DSA selection_query_tile_size must be non-negative")
        self.cfg = cfg
        self.key_tile_size = int(key_tile_size)
        self.index_step_size = int(index_step_size)
        self.selection_query_tile_size = int(selection_query_tile_size)
        self.index_preallocate = bool(index_preallocate)
        self._index_capacity_hint = 0
        self.selection_spill_dir = (
            Path(selection_spill_dir).expanduser().resolve()
            if selection_spill_dir else None)
        self._selection_file = None
        self._selection_file_size = 0
        self._selection_file_dirty = False
        self.k_idx: dict[int, mx.array] = {}  # full-layer -> (B, S, 128)
        self._k_idx_capacity: dict[int, mx.array] = {}
        self._k_idx_lengths: dict[int, int] = {}
        self.selection: mx.array | None = None  # (B, L, topk) indices from last full layer
        self.sel_layer: int = -1
        self._preselected_layer: int = -1
        # Layer-stationary prefill finishes every query tile of a full indexer
        # layer before entering the following shared layers. IndexShare must
        # therefore retain the selection for *each* absolute query range, not
        # merely the last tile. A dense range is represented separately so a
        # missing sparse entry can fail closed instead of silently attending
        # through the wrong full-layer selection.
        self.selection_ranges: dict[
            tuple[int, int], mx.array | tuple[int, tuple[int, ...]]
        ] = {}
        self.dense_ranges: set[tuple[int, int]] = set()
        # F69: proof-carrying telemetry -- configuring a feature (e.g. a long
        # enough context) is not evidence it actually EXERCISED the sparse
        # path this run. A GLM run with a short prompt looks identical to one
        # with a real long-context sparse selection unless something actually
        # counts what ran, not just what was configured (exactly the gap that
        # let this session's own real-GLM validation script silently skip its
        # own chunking path while its docstring called chunking "tested").
        self.stats = {
            "observations": 0,
            "sparse_selects": 0,
            "shared_reuses": 0,
            "score_tiles": 0,
            "score_syncs": 0,
            "score_candidate_id_sorts_avoided": 0,
            "score_final_sorts_avoided": 0,
            "multi_query_selects": 0,
            "selection_ranges_peak": 0,
            "selection_bytes_peak": 0,
            "preselection_groups": 0,
            "preselection_queries": 0,
            "preselection_attention_ranges": 0,
            "index_capacity_grows": 0,
            "index_capacity_preallocations": 0,
            "index_capacity_preallocated_rows": 0,
            "index_rows_copied": 0,
            "index_rows_appended": 0,
            "index_capacity_rows_peak": 0,
            "selection_spill_bytes_written": 0,
            "selection_spill_bytes_read": 0,
            "selection_spill_reads": 0,
            "selection_spill_flushes": 0,
            "selection_spill_write_s": 0.0,
            "selection_spill_read_s": 0.0,
            "selection_score_s": 0.0,
            "preselection_s": 0.0,
            "index_observe_s": 0.0,
        }

    def nbytes(self) -> int:
        capacity_layers = set(self._k_idx_capacity)
        return (
            sum(value.nbytes for value in self._k_idx_capacity.values())
            + sum(
                value.nbytes for layer, value in self.k_idx.items()
                if layer not in capacity_layers)
            + sum(
                value.nbytes for value in self.selection_ranges.values()
                if not isinstance(value, tuple))
        )

    def fork(self) -> "DSAState":
        branch = DSAState(
            self.cfg,
            key_tile_size=self.key_tile_size,
            index_step_size=self.index_step_size,
            selection_query_tile_size=self.selection_query_tile_size,
            index_preallocate=self.index_preallocate,
            selection_spill_dir=(
                str(self.selection_spill_dir)
                if self.selection_spill_dir is not None else ""),
        )
        if self._selection_file_size:
            raise TypeError(
                "DSA snapshots do not support temporary selection spill")
        if self.index_step_size:
            # A stepped branch gets independent logical index bytes. It may
            # establish its own spare backing on first append without sharing
            # mutable capacity.
            for layer, keys in self.k_idx.items():
                copied = mx.take(
                    keys,
                    mx.arange(int(keys.shape[1]), dtype=mx.int32),
                    axis=1,
                )
                mx.eval(copied)
                branch.k_idx[layer] = copied
                branch._k_idx_lengths[layer] = int(copied.shape[1])
        else:
            # Preserve the established cheap immutable-array fork for the
            # default short-context path.
            branch.k_idx = dict(self.k_idx)
        branch.selection = self.selection
        branch.sel_layer = self.sel_layer
        branch._preselected_layer = self._preselected_layer
        branch.selection_ranges = dict(self.selection_ranges)
        branch.dense_ranges = set(self.dense_ranges)
        branch.stats = dict(self.stats)
        branch._index_capacity_hint = self._index_capacity_hint
        return branch

    def set_index_capacity_hint(self, end: int) -> None:
        """Publish a content-independent upper bound for this request sweep.

        Layer-stationary prefill knows its final absolute position before the
        first index-key tile is observed.  Each full indexer layer can therefore
        allocate the same stepped capacity it would eventually reach, once,
        instead of copying its complete prefix on every 1,024-row growth.  The
        hint changes only backing allocation; ``k_idx`` continues to expose
        exactly the initialized logical prefix.
        """
        end = int(end)
        if end < 0:
            raise ValueError("DSA index capacity hint must be non-negative")
        self._index_capacity_hint = max(self._index_capacity_hint, end)

    def clear_selections(self) -> None:
        """Release full-layer query selections after all IndexShare users."""
        self.selection = None
        self.sel_layer = -1
        self._preselected_layer = -1
        self.selection_ranges.clear()
        self.dense_ranges.clear()
        if self._selection_file is not None:
            self._selection_file.seek(0)
            self._selection_file.truncate(0)
        self._selection_file_size = 0
        self._selection_file_dirty = False

    def _begin_selection_layer(self, layer: int) -> None:
        if self.sel_layer == layer:
            return
        self.selection = None
        self.selection_ranges.clear()
        self.dense_ranges.clear()
        if self._selection_file is not None:
            self._selection_file.seek(0)
            self._selection_file.truncate(0)
        self._selection_file_size = 0
        self._selection_file_dirty = False
        self.sel_layer = int(layer)
        self._preselected_layer = -1

    def is_preselected(self, layer: int) -> bool:
        return self._preselected_layer == int(layer)

    def _selection_spill_file(self):
        if self.selection_spill_dir is None:
            return None
        if self._selection_file is None:
            self.selection_spill_dir.mkdir(parents=True, exist_ok=True)
            self._selection_file = tempfile.TemporaryFile(
                prefix="voom-glm-dsa-selection-",
                dir=self.selection_spill_dir,
                mode="w+b",
                buffering=1024 * 1024,
            )
        return self._selection_file

    def _store_selection_range(
        self, query_range: tuple[int, int], selection: mx.array,
    ) -> None:
        spill = self._selection_spill_file()
        if spill is None:
            self.selection_ranges[query_range] = selection
            return
        started = time.perf_counter()
        payload = np.asarray(selection, dtype=np.int32).tobytes(order="C")
        file_offset = self._selection_file_size
        spill.seek(file_offset)
        spill.write(payload)
        self._selection_file_dirty = True
        self._selection_file_size += len(payload)
        self.selection_ranges[query_range] = (
            file_offset, tuple(int(v) for v in selection.shape))
        self.stats["selection_spill_bytes_written"] += len(payload)
        self.stats["selection_spill_write_s"] += (
            time.perf_counter() - started)

    def _load_selection_range(
        self, entry: mx.array | tuple[int, tuple[int, ...]],
    ) -> mx.array:
        if not isinstance(entry, tuple):
            return entry
        if self._selection_file is None:
            raise RuntimeError("DSA selection spill metadata has no open file")
        started = time.perf_counter()
        file_offset, shape = entry
        size = int(np.prod(shape, dtype=np.int64)) * 4
        if self._selection_file_dirty:
            # Switching a buffered update stream from writes to reads needs
            # one flush. Later shared IndexShare layers only seek/read the
            # immutable selection generation; flushing every tile would add
            # ~92K redundant syscalls at the 46.8K harness shape.
            self._selection_file.flush()
            self._selection_file_dirty = False
            self.stats["selection_spill_flushes"] += 1
        self._selection_file.seek(file_offset)
        payload = self._selection_file.read(size)
        if len(payload) != size:
            raise RuntimeError("DSA selection spill payload is truncated")
        host = np.frombuffer(payload, dtype=np.int32).copy().reshape(shape)
        value = mx.array(host)
        mx.eval(value)
        self.stats["selection_spill_bytes_read"] += size
        self.stats["selection_spill_reads"] += 1
        self.stats["selection_spill_read_s"] += (
            time.perf_counter() - started)
        return value

    def _rope_idx(self, x: mx.array, offset: int) -> mx.array:
        """F33 (2026-07-14 correction): confirmed against the actual official
        `GlmMoeDsaIndexer.forward()` source (transformers==5.13.0) that the
        indexer splits/concatenates rope-FIRST, pass-through-SECOND
        (`q_rot, q_pass = split(q, [qk_rope_head_dim, head_dim-qk_rope_head_dim])`,
        `cat([q_rot, q_pass])`) -- the OPPOSITE convention from the main MLA
        attention module (which is nope-first/rope-last, and IS correct --
        see tests/test_f33_mla_attention.py). The previous version of this
        function assumed the same nope-first/rope-last convention for the
        indexer too, which was never checked against the reference and was
        WRONG: it silently produced incorrect top-k selections any time real
        DSA sparsity engaged (S > index_topk), which only a cross-
        implementation oracle check (not this project's own self-consistency
        tests) could catch, since a wrong-but-consistent selection still
        produces a well-formed, in-range set of indices with no crash or
        shape error. Confirmed via tests/test_f33_dsa_indexer.py: swapping
        this split order changed a mismatching top-k selection into an
        exact match against HF's reference, across a small deterministic
        MoE-style config, not just one lucky seed.
        x: (B, H, L, 128)."""
        dr = self.cfg.qk_rope_head_dim
        rot, nope = x[..., :dr], x[..., dr:]
        rot = mx.fast.rope(rot, dr, traditional=True, base=self.cfg.rope_theta,
                           scale=1.0, offset=offset)
        return mx.concatenate([rot, nope], axis=-1)

    def trim(self, length: int):
        """Roll indexer state back to an accepted speculative prefix.

        Target KV rollback used to trim only MLA/KV tensors.  Rejected verify
        lanes therefore remained in ``k_idx`` and could be selected by a later
        decode once context exceeded ``index_topk``.  A cached selection is tied
        to the pre-trim history as well, so invalidate it unconditionally.
        """
        for layer, keys in list(self.k_idx.items()):
            if keys.shape[1] > length:
                backing = self._k_idx_capacity.get(layer)
                self.k_idx[layer] = (
                    backing[:, :length, :]
                    if backing is not None else keys[:, :length, :])
                self._k_idx_lengths[layer] = length
                mx.eval(self.k_idx[layer])
        self.clear_selections()

    @staticmethod
    def _stable_topk(
        scores: mx.array, ids: mx.array, keep: int,
    ) -> tuple[mx.array, mx.array]:
        """Top-k with a tile-independent total order and sorted retained IDs.

        ``_tiled_select`` presents IDs in ascending order: the retained set is
        re-sorted chronologically here and every following key tile contains
        only larger IDs. MLX argsort is stable, so a stable descending score
        sort directly preserves the lower-ID tie rule. Re-sorting only the
        retained K IDs prepares the next merge and is cheaper than re-sorting
        all K+tile candidate IDs on every iteration.
        """
        keep = min(int(keep), int(scores.shape[-1]))
        score_order = mx.argsort(-scores, axis=-1)[..., :keep]
        kept_scores = mx.take_along_axis(scores, score_order, axis=-1)
        kept_ids = mx.take_along_axis(ids, score_order, axis=-1)
        id_order = mx.argsort(kept_ids, axis=-1)
        return (
            mx.take_along_axis(kept_scores, id_order, axis=-1),
            mx.take_along_axis(kept_ids, id_order, axis=-1),
        )

    def _tiled_select(
        self, q_idx: mx.array, w_head: mx.array, k_all: mx.array,
        *, offset: int,
    ) -> mx.array:
        """Score/merge official DSA candidates without allocating BxLxHxS."""
        cfg = self.cfg
        B, L, n_heads, head_dim = q_idx.shape
        S = int(k_all.shape[1])
        keep = int(cfg.index_topk)
        q_pos = mx.arange(offset, offset + L, dtype=mx.int32)[:, None]
        running_scores = None
        running_ids = None
        for tile_index, start in enumerate(range(0, S, self.key_tile_size)):
            end = min(start + self.key_tile_size, S)
            k_tile = k_all[:, start:end, :].astype(mx.float32)
            scores = mx.einsum(
                "blje,bse->bljs", q_idx.astype(mx.float32), k_tile)
            scores = mx.maximum(scores * (head_dim ** -0.5), 0.0)
            scores = (scores * w_head[..., None]).sum(axis=2)
            ids = mx.broadcast_to(
                mx.arange(start, end, dtype=mx.int32)[None, None, :],
                (B, L, end - start),
            )
            scores = mx.where(
                (ids <= q_pos[None]), scores,
                mx.array(float("-inf"), dtype=mx.float32),
            )
            if running_scores is not None:
                scores = mx.concatenate((running_scores, scores), axis=-1)
                ids = mx.concatenate((running_ids, ids), axis=-1)
            running_scores, running_ids = self._stable_topk(
                scores, ids, keep)
            self.stats["score_candidate_id_sorts_avoided"] += 1
            # Bound the lazy score graph, but do not synchronize every key
            # tile. At Q=8/P=1024, eight tiles retain only about 32 MB of FP32
            # dot output and cut host/Metal boundaries by up to 8x.
            if (tile_index + 1) % 8 == 0:
                mx.eval(running_scores, running_ids)
                self.stats["score_syncs"] += 1
            self.stats["score_tiles"] += 1
        if running_ids is None:
            raise RuntimeError("DSA selection received an empty key history")
        # Compact MLA attention must consume valid keys in their original
        # temporal order. Queries in a tile that straddles the K boundary may
        # have fewer than K causal keys; retain -1 padding so the caller can
        # mask it instead of accidentally gathering future rows.
        # Only the causal mask creates ``-inf`` padding.  Do not use
        # ``isfinite`` here: a mathematically valid +inf score (while not
        # expected from the released finite checkpoint) must still outrank
        # finite rows exactly as the official top-k would.
        valid = running_scores > mx.array(float("-inf"), dtype=mx.float32)
        # _stable_topk already keeps retained IDs chronological. Causal IDs
        # form the prefix and masked future fillers the suffix for each query.
        self.stats["score_final_sorts_avoided"] += 1
        return mx.where(valid, running_ids, -1)

    def observe(self, layer: int, indexer_type: str, h: mx.array, w: dict, prefix: str,
                offset: int = 0):
        """Accumulate the indexer k-cache for 'full' layers on EVERY call (prefill
        and decode, below and above threshold) — selection later needs the full
        history. k gets interleaved RoPE at its absolute positions (F33)."""
        if indexer_type != "full":
            return
        started = time.perf_counter()
        k_new = _linear(h, w, f"{prefix}.self_attn.indexer.wk")
        k_new = mx.fast.layer_norm(
            k_new, w[f"{prefix}.self_attn.indexer.k_norm.weight"],
            w[f"{prefix}.self_attn.indexer.k_norm.bias"], 1e-6,
        )
        k_new = self._rope_idx(k_new[:, None], offset)[:, 0]  # (B, L, 128) via 1-head view
        prev = self.k_idx.get(layer)
        if prev is not None and int(prev.shape[1]) != int(offset):
            raise ValueError(
                "GLM DSA index cache offset does not match its history: "
                f"offset={offset}, cached={prev.shape[1]}")
        if prev is None and int(offset) != 0:
            raise ValueError("GLM DSA index cache cannot begin at nonzero offset")
        if self.index_step_size:
            incoming = int(k_new.shape[1])
            end = int(offset) + incoming
            backing = self._k_idx_capacity.get(layer)
            if backing is None or int(backing.shape[1]) < end:
                ordinary_target = (
                    (end + self.index_step_size - 1)
                    // self.index_step_size * self.index_step_size)
                hinted_target = ordinary_target
                if self.index_preallocate and self._index_capacity_hint:
                    hinted_target = (
                        (self._index_capacity_hint + self.index_step_size - 1)
                        // self.index_step_size * self.index_step_size)
                target = max(ordinary_target, hinted_target)
                tail = mx.zeros(
                    (int(k_new.shape[0]), target - int(offset),
                     int(k_new.shape[2])),
                    dtype=k_new.dtype,
                )
                backing = (
                    tail if prev is None
                    else mx.concatenate((prev, tail), axis=1))
                self.stats["index_capacity_grows"] += 1
                self.stats["index_rows_copied"] += int(offset)
                if target > ordinary_target:
                    self.stats["index_capacity_preallocations"] += 1
                    self.stats["index_capacity_preallocated_rows"] += (
                        target - ordinary_target)
            backing[:, int(offset):end, :] = k_new
            mx.eval(backing)
            self._k_idx_capacity[layer] = backing
            self._k_idx_lengths[layer] = end
            self.k_idx[layer] = backing[:, :end, :]
            self.stats["index_rows_appended"] += incoming
            self.stats["index_capacity_rows_peak"] = max(
                self.stats["index_capacity_rows_peak"],
                int(backing.shape[1]),
            )
        else:
            self.k_idx[layer] = (
                k_new if prev is None
                else mx.concatenate([prev, k_new], axis=1))
            mx.eval(self.k_idx[layer])
        self.stats["observations"] += 1
        self.stats["index_observe_s"] += time.perf_counter() - started

    def _selection_inputs(
        self, h: mx.array, q_a: mx.array, w: dict, prefix: str, offset: int,
    ) -> tuple[mx.array, mx.array]:
        cfg = self.cfg
        B, L, _ = h.shape
        n_heads = cfg.index_n_heads
        head_dim = cfg.index_head_dim
        q_idx = _linear(q_a, w, f"{prefix}.self_attn.indexer.wq_b").reshape(
            B, L, n_heads, head_dim
        )
        q_idx = self._rope_idx(q_idx.transpose(0, 2, 1, 3), offset).transpose(0, 2, 1, 3)
        # Official forward casts h to weights_proj.weight.dtype, performs the
        # projection, then casts its result to FP32. Transformers' non-strict
        # `_keep_in_fp32_modules` promotes this module for FP16 loads only; a
        # released BF16 load keeps the checkpoint weight BF16. Do not promote
        # the GEMM itself to FP32 (the synthetic FP32 oracle used to hide this).
        weights_proj = w[f"{prefix}.self_attn.indexer.weights_proj.weight"]
        if isinstance(weights_proj, quant.QTensor):
            w_head = quant.matmul(h, weights_proj).astype(mx.float32)
        else:
            w_head = (h.astype(weights_proj.dtype) @ weights_proj.T).astype(mx.float32)
        # Include the released positive scales even though real-number top-k is
        # invariant: finite FP32 rounding can break a near tie differently.
        w_head = w_head * (n_heads ** -0.5)
        return q_idx, w_head

    def selection_for_range(
        self, layer: int, offset: int, length: int, *, shared: bool,
    ) -> mx.array | None:
        if self.sel_layer < 0:
            raise RuntimeError("DSA IndexShare has no preceding full selection")
        if not shared and self.sel_layer != int(layer):
            raise RuntimeError(
                "DSA preselected selection belongs to a different full layer")
        if shared:
            self.stats["shared_reuses"] += 1
        query_range = (int(offset), int(length))
        if query_range in self.dense_ranges:
            self.selection = None
            return None
        try:
            entry = self.selection_ranges[query_range]
        except KeyError as error:
            label = "IndexShare" if shared else "preselected full layer"
            raise RuntimeError(
                f"DSA {label} is missing the full-layer selection for "
                f"query range offset={offset}, length={length}"
            ) from error
        self.selection = self._load_selection_range(entry)
        return self.selection

    def preselect_full_layer(
        self, layer: int, h: mx.array, w: dict, prefix: str, offset: int,
        *, attention_tile_width: int,
    ) -> None:
        """Batch index scores independently of compact-attention width.

        The q/k projections retain the ordinary attention-tile outer shape.
        Only the score/merge operation combines several adjacent query tiles,
        reducing millions of tiny Metal launches at 49K while keeping selected
        rows file-backed in the original attention-sized ranges.
        """
        preselection_started = time.perf_counter()
        selection_width = int(self.selection_query_tile_size)
        attention_width = int(attention_tile_width)
        total = int(h.shape[1])
        if selection_width <= attention_width:
            raise ValueError("DSA preselection width must exceed attention width")
        if selection_width % attention_width:
            raise ValueError(
                "DSA preselection width must be divisible by attention width")
        self._begin_selection_layer(layer)

        # Preserve the already-gated projection outer shape exactly. Index
        # keys are cheap and remain resident; selection below scores them in
        # wider query batches without changing the head-dimension reduction.
        for start in range(0, total, attention_width):
            end = min(start + attention_width, total)
            self.observe(
                layer, "full", h[:, start:end], w, prefix, offset + start)

        cfg = self.cfg
        k_all = self.k_idx[layer]
        for start in range(0, total, selection_width):
            end = min(start + selection_width, total)
            q_parts = []
            head_parts = []
            for sub_start in range(start, end, attention_width):
                sub_end = min(sub_start + attention_width, end)
                h_sub = h[:, sub_start:sub_end]
                q_a_sub = mx.fast.rms_norm(
                    _linear(h_sub, w, f"{prefix}.self_attn.q_a_proj"),
                    w[f"{prefix}.self_attn.q_a_layernorm.weight"],
                    cfg.mla_latent_norm_eps,
                )
                q_idx, w_head = self._selection_inputs(
                    h_sub, q_a_sub, w, prefix, offset + sub_start)
                q_parts.append(q_idx)
                head_parts.append(w_head)
            q_group = (
                q_parts[0] if len(q_parts) == 1
                else mx.concatenate(q_parts, axis=1))
            head_group = (
                head_parts[0] if len(head_parts) == 1
                else mx.concatenate(head_parts, axis=1))
            global_end = int(offset) + end
            if global_end <= int(cfg.index_topk):
                for sub_start in range(start, end, attention_width):
                    sub_end = min(sub_start + attention_width, end)
                    self.dense_ranges.add(
                        (int(offset) + sub_start, sub_end - sub_start))
                continue
            score_started = time.perf_counter()
            selected = self._tiled_select(
                q_group, head_group, k_all[:, :global_end],
                offset=int(offset) + start)
            mx.eval(selected)
            self.stats["selection_score_s"] += (
                time.perf_counter() - score_started)
            self.stats["score_syncs"] += 1
            self.stats["preselection_groups"] += 1
            self.stats["preselection_queries"] += end - start
            for sub_start in range(start, end, attention_width):
                sub_end = min(sub_start + attention_width, end)
                query_range = (
                    int(offset) + sub_start, sub_end - sub_start)
                relative_start = sub_start - start
                relative_end = sub_end - start
                if int(offset) + sub_end <= int(cfg.index_topk):
                    self.dense_ranges.add(query_range)
                    continue
                selection = selected[:, relative_start:relative_end]
                self._store_selection_range(query_range, selection)
                self.selection = selection
                self.stats["sparse_selects"] += 1
                self.stats["multi_query_selects"] += int(
                    relative_end - relative_start > 1)
                self.stats["preselection_attention_ranges"] += 1
            self.stats["selection_ranges_peak"] = max(
                self.stats["selection_ranges_peak"],
                len(self.selection_ranges))
            self.stats["selection_bytes_peak"] = max(
                self.stats["selection_bytes_peak"],
                self._selection_file_size or sum(
                    value.nbytes for value in self.selection_ranges.values()
                    if not isinstance(value, tuple)),
            )
        self._preselected_layer = int(layer)
        self.stats["preselection_s"] += (
            time.perf_counter() - preselection_started)

    def update_and_select(
        self, layer: int, indexer_type: str, h: mx.array, q_a: mx.array,
        w: dict, prefix: str, offset: int,
    ) -> mx.array | None:
        """Returns (B, L, topk) selected key indices, or None when everything is
        selected (S <= topk) so the caller can use the dense path unchanged."""
        cfg = self.cfg
        query_range = (int(offset), int(h.shape[1]))
        if indexer_type == "shared":
            return self.selection_for_range(
                layer, offset, int(h.shape[1]), shared=True)

        self._begin_selection_layer(layer)
        k_all = self.k_idx.get(layer)
        if k_all is None or k_all.shape[1] <= cfg.index_topk:
            self.selection = None
            self.dense_ranges.add(query_range)
            return None
        q_idx, w_head = self._selection_inputs(
            h, q_a, w, prefix, offset)
        score_started = time.perf_counter()
        sel = self._tiled_select(q_idx, w_head, k_all, offset=offset)
        mx.eval(sel)
        self.stats["selection_score_s"] += (
            time.perf_counter() - score_started)
        self.stats["score_syncs"] += 1
        self.selection = sel
        self._store_selection_range(query_range, sel)
        self.stats["sparse_selects"] += 1
        self.stats["multi_query_selects"] += int(int(h.shape[1]) > 1)
        self.stats["selection_ranges_peak"] = max(
            self.stats["selection_ranges_peak"], len(self.selection_ranges))
        self.stats["selection_bytes_peak"] = max(
            self.stats["selection_bytes_peak"],
            self._selection_file_size or sum(
                value.nbytes for value in self.selection_ranges.values()
                if not isinstance(value, tuple)),
        )
        return sel

    def __del__(self):
        try:
            self.close_selection_spill()
        except Exception:
            pass

    def close_selection_spill(self) -> None:
        if self._selection_file is not None:
            self._selection_file.close()
            self._selection_file = None
        self._selection_file_size = 0
        self._selection_file_dirty = False
