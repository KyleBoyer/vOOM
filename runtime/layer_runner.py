"""Stateless forward math for Llama-family decoder blocks (covers Llama, SmolLM2,
TinyLlama, Qwen2 with attention_bias). Weights are passed in as plain dicts keyed by
HF tensor names so the runner never owns residency decisions.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx

from . import quant
from .config import ModelConfig, effective_expert_top_k
from .lm_head_stream import StreamedLMHead


def embed(tokens: mx.array, embed_weight: mx.array) -> mx.array:
    """tokens (L,) int -> (1, L, hidden)"""
    if isinstance(embed_weight, quant.QTensor):
        # MLX quantizes embeddings row-wise. Select packed rows first, then
        # dequantize only those rows; materializing the complete vocabulary
        # matrix would erase the memory and cold-start benefit of an on-disk
        # quantized checkpoint.
        return mx.dequantize(
            embed_weight.wq[tokens],
            scales=embed_weight.scales[tokens],
            biases=(embed_weight.biases[tokens]
                    if embed_weight.biases is not None else None),
            group_size=embed_weight.group_size,
            bits=embed_weight.bits,
            mode=embed_weight.mode,
        )[None]
    return embed_weight[tokens][None]


def _linear(x: mx.array, w: dict, name: str) -> mx.array:
    y = quant.matmul(x, w[f"{name}.weight"])
    bias = w.get(f"{name}.bias")
    return y + bias if bias is not None else y


@partial(mx.compile, shapeless=True)
def _silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """Shape-polymorphic fused SwiGLU activation (matmuls stay outside)."""
    return mx.sigmoid(gate) * gate * up


def _attention(
    h: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    kv: "object",
    layer: int,
    offset: int,
    rope_freqs: mx.array | None = None,
    rope_mscale: float = 1.0,
) -> mx.array:
    """Attention sub-block on pre-normed input h. Returns o_proj output (no residual)."""
    B, L, _ = h.shape
    n_h, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    q = _linear(h, w, f"{prefix}.self_attn.q_proj")
    k = _linear(h, w, f"{prefix}.self_attn.k_proj")
    v = _linear(h, w, f"{prefix}.self_attn.v_proj")
    # QK-norm variants, distinguished by weight width:
    #   full-width (OLMoE): weight spans n_h*hd, applied pre-reshape;
    #   per-head (Qwen3/Qwen3-VL): weight spans hd, applied post-reshape.
    q_norm = w.get(f"{prefix}.self_attn.q_norm.weight")
    k_norm = w.get(f"{prefix}.self_attn.k_norm.weight")
    per_head_norm = q_norm is not None and q_norm.shape[0] == hd
    if q_norm is not None and not per_head_norm:
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
    q = q.reshape(B, L, n_h, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    if per_head_norm:  # Qwen3: norm each head's vector before rope
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)

    position_free = bool(getattr(kv, "position_free", False))
    # Canonical static YaRN scales Q/K before rotation. Position-free pages keep
    # that scaled K, but deliberately defer only the position-dependent rotation.
    if rope_mscale != 1.0:
        q = q * rope_mscale
        k = k * rope_mscale
    if rope_freqs is None:
        q = mx.fast.rope(q, hd, traditional=False, base=cfg.rope_theta,
                         scale=1.0, offset=offset)
        if not position_free:
            k = mx.fast.rope(k, hd, traditional=False, base=cfg.rope_theta,
                             scale=1.0, offset=offset)
    else:
        q = mx.fast.rope(q, hd, traditional=False, base=None, freqs=rope_freqs,
                         scale=1.0, offset=offset)
        if not position_free:
            k = mx.fast.rope(k, hd, traditional=False, base=None,
                             freqs=rope_freqs, scale=1.0, offset=offset)

    query_positions = mx.arange(offset, offset + L, dtype=mx.int32)
    if position_free:
        had_rotated_view = kv.has_rotated_view(layer, offset)
        kv.update_unrotated(layer, k, v)
        # The fused kernel avoids gathering/copying shared pages for decode and
        # very small verification windows. MLX SDPA remains substantially faster
        # for wide prefill, so that path gathers a temporary logical view and
        # rotates it once without mutating the shared physical pages.
        if had_rotated_view:
            if rope_freqs is None:
                rotated_new_keys = mx.fast.rope(
                    k, hd, traditional=False, base=cfg.rope_theta,
                    scale=1.0, offset=offset)
            else:
                rotated_new_keys = mx.fast.rope(
                    k, hd, traditional=False, base=None, freqs=rope_freqs,
                    scale=1.0, offset=offset)
            keys, values = kv.update_rotated_view(
                layer, rotated_new_keys, v)
        else:
            materialize_rotated_view = (
                L > int(getattr(kv, "custom_attention_query_limit", 0))
                or kv.offset >= int(getattr(kv, "rotated_view_min_keys", 0)))
            use_paged_kernel = (
                not materialize_rotated_view
                and L <= int(getattr(kv, "custom_attention_query_limit", 0))
                and hd % 32 == 0
                and mx.metal.is_available()
            )
            if use_paged_kernel:
                attn = kv.paged_attention(
                    layer, q, query_positions, theta=cfg.rope_theta,
                    denominators=rope_freqs, scale=hd ** -0.5)
                attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * hd)
                return _linear(attn, w, f"{prefix}.self_attn.o_proj")
            keys, values = kv.gather_unrotated(layer)
            if rope_freqs is None:
                keys = mx.fast.rope(
                    keys, hd, traditional=False, base=cfg.rope_theta,
                    scale=1.0, offset=0)
            else:
                keys = mx.fast.rope(
                    keys, hd, traditional=False, base=None, freqs=rope_freqs,
                    scale=1.0, offset=0)
            if materialize_rotated_view:
                kv.set_rotated_view(layer, keys, values)
    else:
        keys, values = kv.update(layer, k, v)

    # Single query token attends to everything. Multi-token windows need an
    # explicit causal mask aligned to the cache offset ("causal" assumes offset 0).
    mask = None
    if L > 1:
        q_pos = query_positions[:, None]
        k_pos = mx.arange(keys.shape[2])[None, :]
        mask = mx.where(k_pos <= q_pos, 0.0, float("-inf")).astype(q.dtype)

    attn = mx.fast.scaled_dot_product_attention(
        q, keys, values, scale=hd**-0.5, mask=mask)
    attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * hd)
    return _linear(attn, w, f"{prefix}.self_attn.o_proj")


def _swiglu(h: mx.array, w: dict, prefix: str, *, fused: bool = False) -> mx.array:
    gate = _linear(h, w, f"{prefix}.gate_proj")
    up = _linear(h, w, f"{prefix}.up_proj")
    activated = (_silu_mul(gate, up) if fused
                 else mx.sigmoid(gate) * gate * up)
    return _linear(activated, w, f"{prefix}.down_proj")


def stack_expert_weights(weights: list):
    """Stack one projection from independently pageable expert weights.

    The disk/runtime representation stays per expert for large out-of-core MoE
    checkpoints.  A small quantized checkpoint that fully fits can fuse those
    pages once and use MLX's gather kernels without changing the on-disk layout.
    """
    if not weights:
        raise ValueError("cannot stack an empty expert set")
    first = weights[0]
    if isinstance(first, quant.QTensor):
        if not all(
            isinstance(weight, quant.QTensor)
            and (weight.bits, weight.group_size, weight.mode)
            == (first.bits, first.group_size, first.mode)
            and (weight.biases is None) == (first.biases is None)
            for weight in weights
        ):
            raise ValueError("expert quantization layouts differ within one projection")
        return quant.QTensor(
            mx.stack([weight.wq for weight in weights]),
            mx.stack([weight.scales for weight in weights]),
            (mx.stack([weight.biases for weight in weights])
             if first.biases is not None else None),
            first.bits,
            first.group_size,
            first.mode,
        )
    if any(isinstance(weight, quant.QTensor) for weight in weights):
        raise ValueError("cannot mix quantized and dense experts in one projection")
    return mx.stack(weights)


def _gather_expert_linear(x: mx.array, weight, indices: mx.array) -> mx.array:
    """Selected expert matmul; returns ``(..., top_k, output_dims)``."""
    expanded = (mx.expand_dims(x, (-2, -3))
                if x.ndim == 2 else mx.expand_dims(x, -2))
    if isinstance(weight, quant.QTensor):
        result = mx.gather_qmm(
            expanded,
            weight.wq,
            weight.scales,
            weight.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=weight.group_size,
            bits=weight.bits,
            mode=weight.mode,
        )
    else:
        result = mx.gather_mm(
            expanded,
            weight.swapaxes(-1, -2),
            rhs_indices=indices,
        )
    return result.squeeze(-2)


def run_fused_moe_mlp(
    x: mx.array,
    w: dict,
    fused_experts: dict[str, object],
    prefix: str,
    cfg: ModelConfig,
    layer: int,
    *,
    fused_swiglu: bool = False,
    mlx_router_semantics: bool = False,
) -> mx.array:
    """Apply the resident OLMoE router/experts to post-attention states."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"],
                         cfg.rms_norm_eps)
    flat = h.reshape(-1, h.shape[-1])
    logits = quant.matmul(flat, w[f"{prefix}.mlp.gate.weight"])
    if not mlx_router_semantics:
        # Released Transformers OLMoE chooses top-k from FP32 probabilities.
        logits = logits.astype(mx.float32)
    # Standard MLX checkpoints are independently executable by MLX-LM, whose
    # precise softmax computes accurately but returns the router's BF16 dtype.
    # The resident prequantized path opts into that reference arithmetic so a
    # portable converted artifact produces the same token stream in both runtimes.
    # Top-k is discontinuous, so this distinction must be explicit.
    probs = mx.softmax(logits, axis=-1, precise=True)
    k = effective_expert_top_k(cfg, layer)
    indices = mx.stop_gradient(
        mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])
    scores = mx.take_along_axis(probs, indices, axis=-1)
    if cfg.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)

    gate = _gather_expert_linear(flat, fused_experts["gate_proj"], indices)
    up = _gather_expert_linear(flat, fused_experts["up_proj"], indices)
    activated = (_silu_mul(gate, up) if fused_swiglu
                 else mx.sigmoid(gate) * gate * up)
    down = _gather_expert_linear(
        activated, fused_experts["down_proj"], indices)
    out = (down * scores[..., None].astype(down.dtype)).sum(axis=-2)
    return x + out.reshape(h.shape)


def run_fused_moe_block(
    x: mx.array,
    w: dict,
    fused_experts: dict[str, object],
    prefix: str,
    cfg: ModelConfig,
    kv: "object",
    layer: int,
    offset: int,
    mlp_last_only: bool = False,
    rope_freqs: mx.array | None = None,
    rope_mscale: float = 1.0,
    fused_swiglu: bool = False,
    mlx_router_semantics: bool = False,
) -> mx.array:
    """Resident OLMoE block using one gathered kernel per expert projection."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    x = x + _attention(h, w, prefix, cfg, kv, layer, offset,
                       rope_freqs=rope_freqs, rope_mscale=rope_mscale)
    if mlp_last_only:
        x = x[:, -1:, :]
    return run_fused_moe_mlp(
        x, w, fused_experts, prefix, cfg, layer,
        fused_swiglu=fused_swiglu,
        mlx_router_semantics=mlx_router_semantics)


def run_block(
    x: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    kv: "object",
    layer: int,
    offset: int,
    mlp_last_only: bool = False,
    rope_freqs: mx.array | None = None,
    rope_mscale: float = 1.0,
    fused_swiglu: bool = False,
) -> mx.array:
    """One dense decoder block. x: (1, L, hidden).

    mlp_last_only (F36): once this layer's KV is built, earlier positions' MLP
    outputs are dead if only the last position's logits are consumed — attention
    runs full-width, then the residual is sliced to the last position."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    x = x + _attention(h, w, prefix, cfg, kv, layer, offset,
                       rope_freqs=rope_freqs, rope_mscale=rope_mscale)
    if mlp_last_only:
        x = x[:, -1:, :]
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    return x + _swiglu(h, w, f"{prefix}.mlp", fused=fused_swiglu)


def run_moe_mlp(
    x: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    layer: int,
    get_experts,
) -> mx.array:
    """Apply pageable OLMoE routing/experts to post-attention states."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    # The side-quest quantizes OLMoE's router along with the expert matrices.
    # QTensor deliberately does not pretend to be an mx.array (and therefore
    # has no ``.T``); use the common weight dispatch just like every other
    # projection in this block.
    logits = quant.matmul(h, w[f"{prefix}.mlp.gate.weight"]).astype(mx.float32)
    probs = mx.softmax(logits, axis=-1, precise=True)
    k = effective_expert_top_k(cfg, layer)
    idx = mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k]  # (1, L, k)
    pw = mx.take_along_axis(probs, idx, axis=-1)  # (1, L, k)
    if cfg.norm_topk_prob:
        pw = pw / pw.sum(axis=-1, keepdims=True)
    mx.eval(idx, pw)

    # Materialize each routed tensor once. Converting individual MLX scalars in
    # the nested loop below creates a tiny gather/evaluation barrier per route.
    # Keep the same position-major/top-k-major insertion order after the bulk
    # transfer so accumulation order and exact token behavior are unchanged.
    # Convert the already-evaluated arrays themselves, then remove the batch
    # dimension in Python. ``idx[0].tolist()`` would first enqueue a new MLX
    # gather and give back much of the synchronization win on one-token decode.
    index_rows = idx.tolist()[0]
    weight_rows = pw.tolist()[0]

    # Group positions by expert so each selected expert runs once per sweep.
    groups: dict[int, list[tuple[int, float]]] = {}
    for pos, (experts, weights) in enumerate(zip(index_rows, weight_rows)):
        for expert, weight in zip(experts, weights):
            groups.setdefault(int(expert), []).append((pos, float(weight)))

    out = mx.zeros_like(h)
    experts = get_experts(layer, sorted(groups), positions={e: [pt for pt, _ in v] for e, v in groups.items()})
    for e, plist in groups.items():
        ew = experts[e]
        positions = [p for p, _ in plist]
        weights = mx.array([wt for _, wt in plist]).astype(h.dtype)
        y = _swiglu(h[:, positions, :], ew, f"{prefix}.mlp.experts.{e}")
        out = out.at[:, positions, :].add(y * weights[None, :, None])
    return x + out


def run_moe_block(
    x: mx.array,
    w: dict,
    prefix: str,
    cfg: ModelConfig,
    kv: "object",
    layer: int,
    offset: int,
    get_experts,
    mlp_last_only: bool = False,
    rope_freqs: mx.array | None = None,
    rope_mscale: float = 1.0,
) -> mx.array:
    """MoE decoder block (OLMoE-style: softmax router over all experts, top-k).
    `get_experts(layer, [ids]) -> {id: weight dict}` is the paging hook — only the
    routed experts are materialized, batch-fetched in one disk pass after routing.

    mlp_last_only (F36): slice to the last position after attention so the
    routed union collapses from up-to-L*k experts to exactly k."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    x = x + _attention(h, w, prefix, cfg, kv, layer, offset,
                       rope_freqs=rope_freqs, rope_mscale=rope_mscale)
    if mlp_last_only:
        x = x[:, -1:, :]
    return run_moe_mlp(x, w, prefix, cfg, layer, get_experts)


def final_logits(x: mx.array, norm_weight: mx.array, lm_head_weight, eps: float) -> mx.array:
    """x: (1, L, hidden) -> logits (vocab,) for the last position only."""
    h = mx.fast.rms_norm(x[:, -1:, :], norm_weight, eps)
    if isinstance(lm_head_weight, StreamedLMHead):
        return lm_head_weight.logits(h)[0, 0]
    return quant.matmul(h, lm_head_weight)[0, 0]


def all_logits(x: mx.array, norm_weight: mx.array, lm_head_weight, eps: float) -> mx.array:
    """x: (1, L, hidden) -> logits (L, vocab) for every position (verification)."""
    h = mx.fast.rms_norm(x, norm_weight, eps)
    if isinstance(lm_head_weight, StreamedLMHead):
        return lm_head_weight.logits(h)[0]
    return quant.matmul(h, lm_head_weight)[0]
