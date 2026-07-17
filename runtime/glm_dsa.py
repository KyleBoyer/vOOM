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
positions. Above that boundary the current L=1 implementation is a conformance
candidate, not a proven released path: ordered selection, compact-attention
arithmetic, and sparse L>1 still require F33/F75 output and token gates.

Integration: glm._mla_attention consults DSAState when S > index_topk. Decode
(L=1) gathers the selected latent rows before kv_b expansion, so the F21
compressed cache also cuts the EXPANSION cost to top-k rows only.
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig
from . import quant
from .layer_runner import _linear


class DSAState:
    """Per-generation indexer state: cached k_idx per 'full' layer and the
    selection shared with subsequent 'shared' layers."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.k_idx: dict[int, mx.array] = {}  # full-layer -> (B, S, 128)
        self.selection: mx.array | None = None  # (B, L, topk) indices from last full layer
        self.sel_layer: int = -1
        # F69: proof-carrying telemetry -- configuring a feature (e.g. a long
        # enough context) is not evidence it actually EXERCISED the sparse
        # path this run. A GLM run with a short prompt looks identical to one
        # with a real long-context sparse selection unless something actually
        # counts what ran, not just what was configured (exactly the gap that
        # let this session's own real-GLM validation script silently skip its
        # own chunking path while its docstring called chunking "tested").
        self.stats = {"observations": 0, "sparse_selects": 0, "shared_reuses": 0}

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
                self.k_idx[layer] = keys[:, :length, :]
                mx.eval(self.k_idx[layer])
        self.selection = None
        self.sel_layer = -1

    def observe(self, layer: int, indexer_type: str, h: mx.array, w: dict, prefix: str,
                offset: int = 0):
        """Accumulate the indexer k-cache for 'full' layers on EVERY call (prefill
        and decode, below and above threshold) — selection later needs the full
        history. k gets interleaved RoPE at its absolute positions (F33)."""
        if indexer_type != "full":
            return
        k_new = _linear(h, w, f"{prefix}.self_attn.indexer.wk")
        k_new = mx.fast.layer_norm(
            k_new, w[f"{prefix}.self_attn.indexer.k_norm.weight"],
            w[f"{prefix}.self_attn.indexer.k_norm.bias"], 1e-6,
        )
        k_new = self._rope_idx(k_new[:, None], offset)[:, 0]  # (B, L, 128) via 1-head view
        prev = self.k_idx.get(layer)
        self.k_idx[layer] = k_new if prev is None else mx.concatenate([prev, k_new], axis=1)
        mx.eval(self.k_idx[layer])
        self.stats["observations"] += 1

    def update_and_select(
        self, layer: int, indexer_type: str, h: mx.array, q_a: mx.array,
        w: dict, prefix: str, offset: int,
    ) -> mx.array | None:
        """Returns (B, L, topk) selected key indices, or None when everything is
        selected (S <= topk) so the caller can use the dense path unchanged."""
        cfg = self.cfg
        if indexer_type == "shared":
            self.stats["shared_reuses"] += 1
            return self.selection

        k_all = self.k_idx.get(layer)
        if k_all is None or k_all.shape[1] <= cfg.index_topk:
            self.selection = None
            return None
        S = k_all.shape[1]

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
        scores = mx.einsum("blje,bse->bljs", q_idx.astype(mx.float32), k_all.astype(mx.float32))
        scores = mx.maximum(scores * (head_dim ** -0.5), 0.0)
        w_head = w_head * (n_heads ** -0.5)
        scores = (scores * w_head[..., None]).sum(axis=2)
        # causal: query t (global pos offset+t) may select s <= offset+t
        q_pos = mx.arange(offset, offset + L)[:, None]
        s_pos = mx.arange(S)[None, :]
        scores = mx.where((s_pos <= q_pos)[None], scores, mx.array(float("-inf")))
        sel = mx.argpartition(-scores, kth=cfg.index_topk - 1, axis=-1)[..., : cfg.index_topk]
        # HF eager/SDPA scatters the selected IDs into a mask over the original
        # chronological key tensor. Our compact gather must restore that order;
        # feeding argpartition order to attention changes the floating reduction.
        sel = mx.sort(sel, axis=-1)
        mx.eval(sel)
        self.selection = sel
        self.sel_layer = layer
        self.stats["sparse_selects"] += 1
        return sel
