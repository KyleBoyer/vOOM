"""Released Qwen3.5/Qwen3.6 hybrid text-trunk math.

Qwen3.6-35B-A3B keeps the ``qwen3_5_moe`` architecture identifier.  It is
not compatible with the ordinary Qwen3 decoder: three Gated DeltaNet layers
alternate with one gated full-attention layer, every layer has routed and
shared experts, full attention uses partial RoPE, and decoder RMSNorm weights
are zero-centered (the executed scale is ``1 + weight``).

The recurrent implementation below follows the official Transformers
``Qwen3_5MoeGatedDeltaNet`` fallback formula.  It is correctness-first and
uses the same bounded lazy-graph checkpoints as runtime.kimi_linear's already
measured sequential KDA path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import quant
from .config import ModelConfig, effective_expert_top_k
from .expert_batching import consume_expert_batches
from .glm import _group_routes
from .kda_state import KDAStateCache
from .kimi_linear import _causal_depthwise_conv1d
from .layer_runner import _linear, _swiglu
from .lm_head_stream import StreamedLMHead


def qwen35_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Official zero-centered decoder RMSNorm: norm(x.float) * (1+w)."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    normalized = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (normalized * (1.0 + weight.astype(mx.float32))).astype(source_dtype)


def _silu_gated_rms_norm(
    x: mx.array, gate: mx.array, weight: mx.array, eps: float,
) -> mx.array:
    """DeltaNet's ordinary-scale RMSNorm followed by a SiLU output gate."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    normalized = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    gate32 = gate.astype(mx.float32)
    silu_gate = gate32 * mx.sigmoid(gate32)
    return (normalized * weight.astype(mx.float32) * silu_gate).astype(source_dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_partial_rope(
    q: mx.array, k: mx.array, offset: int, cfg: ModelConfig,
    positions3: np.ndarray | mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply released rotate-half RoPE to the leading partial head width.

    Text positions use equal T/H/W ids, so Qwen3.6's interleaved M-RoPE
    frequency selection reduces exactly to the ordinary one-dimensional
    sequence below.  Multimodal positions remain explicitly unsupported by
    this text-trunk module.
    """
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > cfg.head_dim or rotary_dim % 2:
        raise ValueError(
            f"invalid Qwen3.5 partial rotary width {rotary_dim} "
            f"for head_dim={cfg.head_dim}")
    dims = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    inv_freq = 1.0 / (cfg.rope_theta ** (dims / rotary_dim))
    if positions3 is None:
        positions = mx.arange(offset, offset + q.shape[2], dtype=mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]
    else:
        positions = mx.array(positions3).astype(mx.float32)
        if positions.ndim != 2 or positions.shape != (3, q.shape[2]):
            raise ValueError(
                "Qwen3.5 multimodal positions must have shape "
                f"(3, {q.shape[2]}), got {positions.shape}")
        sections = (cfg.rope_scaling or {}).get("mrope_section")
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(not isinstance(value, int) or value < 0
                       for value in sections)
                or sum(sections) != rotary_dim // 2):
            raise ValueError(
                "Qwen3.5 mrope_section must contain three non-negative "
                f"integers summing to {rotary_dim // 2}")
        # Official apply_interleaved_mrope starts with temporal positions and
        # replaces frequency indices 1,4,... with H and 2,5,... with W up to
        # their declared section lengths. This is partial RoPE: the remaining
        # 192 head dimensions in Qwen3.6 pass through untouched.
        components = np.zeros(rotary_dim // 2, dtype=np.int32)
        components[1:3 * sections[1]:3] = 1
        components[2:3 * sections[2]:3] = 2
        selected = positions[mx.array(components)]
        freqs = selected.T * inv_freq[None, :]
    embedding = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(embedding).astype(q.dtype)[None, None, :, :]
    sin = mx.sin(embedding).astype(q.dtype)[None, None, :, :]

    def apply(x):
        rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
        rotated = rotated * cos + _rotate_half(rotated) * sin
        return mx.concatenate([rotated, passthrough], axis=-1)

    return apply(q), apply(k)


def _full_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int,
    positions3: np.ndarray | mx.array | None = None,
) -> mx.array:
    batch, length, _ = h.shape
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    projected = _linear(h, w, f"{prefix}.self_attn.q_proj")
    projected = projected.reshape(batch, length, heads, 2 * head_dim)
    q = projected[..., :head_dim]
    output_gate = projected[..., head_dim:].reshape(
        batch, length, heads * head_dim)
    k = _linear(h, w, f"{prefix}.self_attn.k_proj").reshape(
        batch, length, kv_heads, head_dim)
    v = _linear(h, w, f"{prefix}.self_attn.v_proj").reshape(
        batch, length, kv_heads, head_dim)

    q = qwen35_rms_norm(
        q, w[f"{prefix}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k = qwen35_rms_norm(
        k, w[f"{prefix}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    q, k = _apply_partial_rope(q, k, offset, cfg, positions3)
    keys, values = kv.update(layer, k, v)

    mask = None
    if length > 1:
        q_pos = mx.arange(offset, offset + length, dtype=mx.int32)[:, None]
        k_pos = mx.arange(keys.shape[2], dtype=mx.int32)[None, :]
        mask = mx.where(k_pos <= q_pos, 0.0, float("-inf")).astype(q.dtype)
    attended = mx.fast.scaled_dot_product_attention(
        q, keys, values, scale=head_dim ** -0.5, mask=mask)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch, length, heads * head_dim)
    attended = attended * mx.sigmoid(output_gate)
    return _linear(attended, w, f"{prefix}.self_attn.o_proj")


def _gated_delta_net(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig,
    state_cache: KDAStateCache | None, layer: int,
) -> mx.array:
    batch, length, _ = h.shape
    key_heads = cfg.linear_num_key_heads
    value_heads = cfg.linear_num_value_heads
    key_dim = cfg.linear_key_head_dim
    value_dim = cfg.linear_value_head_dim
    kernel = cfg.linear_conv_kernel_dim
    if min(key_heads, value_heads, key_dim, value_dim, kernel) <= 0:
        raise ValueError("incomplete Qwen3.5 Gated DeltaNet configuration")
    if value_heads % key_heads:
        raise ValueError("Qwen3.5 value heads must be divisible by key heads")

    key_width = key_heads * key_dim
    value_width = value_heads * value_dim
    mixed = _linear(h, w, f"{prefix}.linear_attn.in_proj_qkv")
    history = None
    cached_history = (
        state_cache.conv_history(layer) if state_cache is not None else None)
    if cached_history is not None:
        history = cached_history[0]
    mixed, new_history = _causal_depthwise_conv1d(
        mixed, w[f"{prefix}.linear_attn.conv1d.weight"], history, kernel)
    q, k, v = mx.split(mixed, (key_width, 2 * key_width), axis=-1)
    q = q.reshape(batch, length, key_heads, key_dim)
    k = k.reshape(batch, length, key_heads, key_dim)
    v = v.reshape(batch, length, value_heads, value_dim)
    repeats = value_heads // key_heads
    if repeats > 1:
        q = mx.repeat(q, repeats, axis=2)
        k = mx.repeat(k, repeats, axis=2)

    def l2norm(value):
        value = value.astype(mx.float32)
        return value * mx.rsqrt(
            mx.sum(value * value, axis=-1, keepdims=True) + 1e-6)

    q = l2norm(q) * (key_dim ** -0.5)
    k = l2norm(k)
    v = v.astype(mx.float32)
    beta = mx.sigmoid(_linear(
        h, w, f"{prefix}.linear_attn.in_proj_b").astype(mx.float32))
    a = _linear(h, w, f"{prefix}.linear_attn.in_proj_a").astype(mx.float32)
    dt_bias = w[f"{prefix}.linear_attn.dt_bias"].astype(mx.float32)
    softplus = mx.logaddexp(
        a + dt_bias.reshape(1, 1, value_heads),
        mx.zeros_like(a))
    decay = -mx.exp(
        w[f"{prefix}.linear_attn.A_log"].astype(mx.float32)
    ).reshape(1, 1, value_heads) * softplus

    state = state_cache.state(layer) if state_cache is not None else None
    if state is None:
        state = mx.zeros(
            (batch, value_heads, key_dim, value_dim), dtype=mx.float32)
    outputs = []
    for position in range(length):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * mx.exp(decay[:, position])[..., None, None]
        predicted = mx.sum(k_t[..., None] * state, axis=-2)
        delta = (v_t - predicted) * beta[:, position, :, None]
        state = state + k_t[..., None] * delta[..., None, :]
        outputs.append(mx.sum(q_t[..., None] * state, axis=-2))
        if (position + 1) % 32 == 0:
            mx.eval(state)
    output = mx.stack(outputs, axis=1)
    if state_cache is not None:
        mx.eval(state)
        state_cache.set_state(layer, state)
        state_cache.set_conv_history(layer, (new_history,))

    z = _linear(h, w, f"{prefix}.linear_attn.in_proj_z").reshape(
        batch, length, value_heads, value_dim)
    output = _silu_gated_rms_norm(
        output, z, w[f"{prefix}.linear_attn.norm.weight"],
        cfg.rms_norm_eps)
    output = output.reshape(batch, length, value_width)
    return _linear(output, w, f"{prefix}.linear_attn.out_proj")


def _route_experts(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
) -> tuple[mx.array, mx.array]:
    router_logits = quant.matmul(h, w[f"{prefix}.mlp.gate.weight"])
    probs = mx.softmax(router_logits.astype(mx.float32), axis=-1, precise=True)
    top_k = effective_expert_top_k(cfg, layer)
    indices = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(probs, indices, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return indices, scores.astype(router_logits.dtype)


def _moe(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None,
) -> mx.array:
    indices, scores = _route_experts(h, w, prefix, cfg, layer)
    mx.eval(indices, scores)
    groups = _group_routes(indices, scores)
    routed = mx.zeros_like(h)
    expert_ids = sorted(groups)
    positions_by_expert = {
        expert: [position for position, _ in groups[expert]]
        for expert in expert_ids
    }
    if iter_expert_batches is None:
        experts = get_experts(
            layer, expert_ids, positions=positions_by_expert)
        batches = ((expert_ids, experts),)
    else:
        batches = iter_expert_batches(
            layer, expert_ids, positions=positions_by_expert)

    def consume_batch(batch_ids, experts):
        nonlocal routed
        for expert in batch_ids:
            plist = groups[expert]
            positions = [position for position, _ in plist]
            route_weights = mx.array(
                [weight for _, weight in plist], dtype=h.dtype)
            expert_prefix = f"{prefix}.mlp.experts.{expert}"
            contribution = _swiglu(
                h[:, positions, :], experts[expert], expert_prefix)
            routed = routed.at[:, positions, :].add(
                contribution * route_weights[None, :, None])
        mx.eval(routed)

    consume_expert_batches(batches, consume_batch)
    shared = _swiglu(h, w, f"{prefix}.mlp.shared_expert")
    shared_gate = mx.sigmoid(_linear(
        h, w, f"{prefix}.mlp.shared_expert_gate"))
    return routed + shared_gate * shared


def run_qwen35_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, get_experts, mlp_last_only: bool = False,
    iter_expert_batches=None,
    positions3: np.ndarray | mx.array | None = None,
) -> mx.array:
    residual = x
    h = qwen35_rms_norm(
        x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    layer_type = cfg.layer_types[layer]
    if layer_type == "linear_attention":
        mixed = _gated_delta_net(
            h, w, prefix, cfg, getattr(kv, "kda_cache", None), layer)
    elif layer_type == "full_attention":
        mixed = _full_attention(
            h, w, prefix, cfg, kv, layer, offset, positions3)
    else:
        raise ValueError(f"unsupported Qwen3.5 layer type {layer_type!r}")
    x = residual + mixed
    if mlp_last_only:
        x = x[:, -1:, :]
    h = qwen35_rms_norm(
        x, w[f"{prefix}.post_attention_layernorm.weight"],
        cfg.rms_norm_eps)
    # 2026-07-20: Qwen3.5/3.6's dense sibling checkpoints (bare "qwen3_5"
    # model_type -- Qwen3.5-4B/9B, Qwen3.6-27B) share this exact hybrid
    # DeltaNet/full-attention layer layout but have num_experts=0 (a plain
    # per-layer MLP under {prefix}.mlp.* instead of routed/shared experts).
    # _swiglu is the same generic gate/up/down-proj helper the shared-expert
    # path above already reuses for a single dense FFN; the tensor names it
    # reads (gate_proj/up_proj/down_proj under the given prefix) are the
    # real released names confirmed directly from Qwen/Qwen3.5-4B's own
    # config.json/weight layout, not inferred.
    if not cfg.num_experts:
        return x + _swiglu(h, w, f"{prefix}.mlp")
    return x + _moe(
        h, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches)


def multimodal_prefill(
    engine, tokens: list[int], image_embeds: mx.array,
    positions3: np.ndarray, kv,
) -> mx.array:
    """Exact Qwen3.5/3.6 hybrid prefill with vision embeddings spliced in.

    DeltaNet layers consume the sequence in ordinary causal order and carry no
    RoPE. Full-attention layers receive the released 3D partial/interleaved
    positions. Qwen3.6 declares no DeepStack injection points, so the vision
    tower contributes only its final merged embeddings.
    """
    cfg = engine.cfg
    vision_tokens = {cfg.image_token_id, cfg.video_token_id} - {0}
    is_vision = np.isin(np.asarray(tokens), list(vision_tokens))
    x = engine._embed(list(tokens))
    if is_vision.any():
        indexes = mx.array(np.nonzero(is_vision)[0])
        if image_embeds is None or image_embeds.shape[0] != indexes.shape[0]:
            raise ValueError(
                "Qwen3.5 vision embedding count does not match expanded "
                "placeholder tokens")
        copied = mx.zeros_like(x) + x
        copied[0, indexes, :] = image_embeds.astype(x.dtype)
        x = copied

    offset = kv.offset
    for layer in range(cfg.num_hidden_layers):
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer))
        x = run_qwen35_block(
            x, weights, f"model.layers.{layer}", cfg, kv, layer, offset,
            engine._get_experts,
            iter_expert_batches=engine._iter_expert_batches,
            positions3=positions3,
        )
        mx.eval(x)
    logits = engine._final_logits(x)
    mx.eval(logits)
    return logits


def multimodal_suffix_prefill(
    engine, tokens: list[int], positions3: np.ndarray, kv, prefix_tokens: int,
) -> mx.array:
    """Extend an exact hybrid multimodal endpoint with text-only tokens.

    The full-attention quarter uses the suffix's released M-RoPE positions,
    while the DeltaNet layers advance their exact recurrent matrices and conv
    histories from the retained prompt endpoint.  Neither state kind is
    rewound or approximated.
    """
    suffix = tokens[prefix_tokens:]
    if not suffix:
        raise ValueError("Qwen3.5 vision prompt-cache suffix must not be empty")
    cfg = engine.cfg
    vision_tokens = {cfg.image_token_id, cfg.video_token_id} - {0}
    if any(token in vision_tokens for token in suffix):
        raise ValueError("Qwen3.5 vision prompt-cache suffix must be text-only")
    suffix_positions = np.asarray(positions3)[:, prefix_tokens:]
    if suffix_positions.shape != (3, len(suffix)):
        raise ValueError("Qwen3.5 vision suffix position metadata mismatch")

    x = engine._embed(suffix)
    offset = kv.offset
    if offset != prefix_tokens:
        raise ValueError(
            f"Qwen3.5 vision endpoint offset {offset} != prefix {prefix_tokens}")
    for layer in range(cfg.num_hidden_layers):
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer))
        x = run_qwen35_block(
            x, weights, f"model.layers.{layer}", cfg, kv, layer, offset,
            engine._get_experts,
            iter_expert_batches=engine._iter_expert_batches,
            positions3=suffix_positions,
        )
        mx.eval(x)
    logits = engine._final_logits(x)
    mx.eval(logits)
    return logits


def final_logits(
    x: mx.array, norm_weight: mx.array, lm_head_weight,
    eps: float,
) -> mx.array:
    h = qwen35_rms_norm(x[:, -1:, :], norm_weight, eps)
    if isinstance(lm_head_weight, StreamedLMHead):
        return lm_head_weight.logits(h)[0, 0]
    return quant.matmul(h, lm_head_weight)[0, 0]


def all_logits(
    x: mx.array, norm_weight: mx.array, lm_head_weight,
    eps: float,
) -> mx.array:
    h = qwen35_rms_norm(x, norm_weight, eps)
    if isinstance(lm_head_weight, StreamedLMHead):
        return lm_head_weight.logits(h)[0]
    return quant.matmul(h, lm_head_weight)[0]
