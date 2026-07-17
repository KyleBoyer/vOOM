"""Experimental position-independent KV reuse for dense tool catalogs.

Exact prefix reuse remains the preferred path.  This module implements the
lossy EPIC/LegoLink shape needed only after a catalog edit: reuse the unchanged
tail of each cached tool span, relocate its post-RoPE keys, and recompute the
first ``repair_tokens`` positions of every reused span plus all uncached prompt
positions.  The selective sweep is layer-by-layer so repaired tokens attend to
the complete assembled cache instead of merely concatenating stale KV.

The path supports dense Qwen and OLMoE. OLMoE recomputes routing for every
selected position and uses either its resident gathered experts or the normal
page-fetch hook. GLM MLA/DSA and multimodal M-RoPE have different state
semantics and must fail closed rather than silently receiving this
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from . import layer_runner
from .kv_cache import KVCache, PositionFreeKVCache


@dataclass(frozen=True)
class ToolCapsuleSpan:
    identity: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ReusedRange:
    start: int
    end: int
    source_start: int
    kind: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ToolPICPlan:
    reused: tuple[ReusedRange, ...]
    selected_positions: tuple[int, ...]
    exact_prefix_tokens: int
    capsule_tokens_reused: int
    capsule_tokens_repaired: int

    @property
    def selected_tokens(self) -> int:
        return len(self.selected_positions)


def _validate_spans(spans, length: int, label: str) -> tuple[ToolCapsuleSpan, ...]:
    normalized = tuple(
        span if isinstance(span, ToolCapsuleSpan) else ToolCapsuleSpan(*span)
        for span in spans)
    previous = 0
    identities = set()
    for span in normalized:
        if not span.identity or span.identity in identities:
            raise ValueError(f"{label} capsule identities must be unique")
        if not 0 <= span.start < span.end <= length or span.start < previous:
            raise ValueError(f"invalid/overlapping {label} capsule span: {span}")
        identities.add(span.identity)
        previous = span.end
    return normalized


def build_pic_plan(
        tokens, capsules, source_tokens, source_capsules, *,
        exact_prefix_tokens: int = 0, repair_tokens: int = 16,
) -> ToolPICPlan | None:
    """Plan selective recomputation against one prior catalog KV.

    ``exact_prefix_tokens`` is already-proven prefix state and is always reused
    first.  A tool contributes only when its identity, length, and exact token
    IDs match.  Every other position is recomputed.  ``None`` means detached
    reuse cannot save any work over the exact-prefix baseline.
    """
    tokens = tuple(tokens)
    source_tokens = tuple(source_tokens)
    current = _validate_spans(capsules, len(tokens), "current")
    source = _validate_spans(source_capsules, len(source_tokens), "source")
    if repair_tokens < 0:
        raise ValueError("repair_tokens must be non-negative")
    if not 0 <= exact_prefix_tokens <= min(len(tokens), len(source_tokens)):
        raise ValueError("exact_prefix_tokens is outside the token sequences")
    if tokens[:exact_prefix_tokens] != source_tokens[:exact_prefix_tokens]:
        raise ValueError("declared exact prefix does not match source tokens")

    source_by_id = {span.identity: span for span in source}
    reusable: list[ReusedRange] = []
    if exact_prefix_tokens:
        reusable.append(ReusedRange(
            0, exact_prefix_tokens, 0, "exact_prefix"))

    repaired = 0
    capsule_reused = 0
    for span in current:
        old = source_by_id.get(span.identity)
        if old is None or old.length != span.length:
            continue
        if tokens[span.start:span.end] != source_tokens[old.start:old.end]:
            continue
        repair_end = min(span.end, span.start + repair_tokens)
        reuse_start = max(exact_prefix_tokens, repair_end)
        if reuse_start >= span.end:
            continue
        source_start = old.start + (reuse_start - span.start)
        reusable.append(ReusedRange(
            reuse_start, span.end, source_start, "tool_capsule"))
        capsule_reused += span.end - reuse_start
        repaired += max(0, repair_end - max(span.start, exact_prefix_tokens))

    reusable.sort(key=lambda value: value.start)
    non_overlapping: list[ReusedRange] = []
    for value in reusable:
        if non_overlapping and value.start < non_overlapping[-1].end:
            raise ValueError("planned PIC ranges overlap")
        non_overlapping.append(value)
    if capsule_reused <= 0:
        return None

    selected = []
    cursor = 0
    for value in non_overlapping:
        selected.extend(range(cursor, value.start))
        cursor = value.end
    selected.extend(range(cursor, len(tokens)))
    # Generation needs the prompt endpoint hidden state. A malformed template
    # ending inside a detached span is not safe for this implementation.
    if not selected or selected[-1] != len(tokens) - 1:
        return None
    return ToolPICPlan(
        reused=tuple(non_overlapping),
        selected_positions=tuple(selected),
        exact_prefix_tokens=exact_prefix_tokens,
        capsule_tokens_reused=capsule_reused,
        capsule_tokens_repaired=repaired,
    )


def _rotate_half(value):
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _rope_cos_sin(positions, head_dim: int, theta: float, denominators, dtype):
    if denominators is None:
        half = head_dim // 2
        denominators = theta ** (mx.arange(half, dtype=mx.float32) / half)
    else:
        denominators = denominators.astype(mx.float32)
    angles = positions.astype(mx.float32)[:, None] / denominators[None, :]
    angles = mx.concatenate((angles, angles), axis=-1)
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def _apply_rope(value, positions, cfg, denominators, *, scale: float = 1.0):
    cos, sin = _rope_cos_sin(
        positions, cfg.head_dim, cfg.rope_theta, denominators, value.dtype)
    if scale != 1.0:
        value = value * scale
    return value * cos[None, None] + _rotate_half(value) * sin[None, None]


def _layout_parts(plan: ToolPICPlan, length: int):
    """Yield a complete ordered selected/reused partition of the prompt."""
    cursor = 0
    selected_cursor = 0
    for reused in plan.reused:
        if cursor < reused.start:
            width = reused.start - cursor
            yield "selected", cursor, reused.start, selected_cursor
            selected_cursor += width
        yield "reused", reused.start, reused.end, reused.source_start
        cursor = reused.end
    if cursor < length:
        yield "selected", cursor, length, selected_cursor


def prefill_with_tool_capsules(engine, tokens, source_kv: KVCache,
                               plan: ToolPICPlan):
    """Build a complete prompt KV with selective EPIC-style recomputation.

    Returns ``(kv, logits)``.  The caller owns admission, source-slot lifetime,
    telemetry, and the policy decision comparing this work with exact-prefix
    suffix prefill.
    """
    tokens = tuple(tokens)
    cfg = engine.cfg
    supported_moe = cfg.model_type == "olmoe" and bool(cfg.num_experts)
    if cfg.model_type not in ("qwen2", "qwen3", "olmoe"):
        raise ValueError(
            f"tool PIC is unsupported for model_type={cfg.model_type!r}")
    if cfg.num_experts and not supported_moe:
        raise ValueError(
            f"tool PIC is unsupported for model_type={cfg.model_type!r}")
    if cfg.vision_config:
        raise ValueError("tool PIC is unsupported for multimodal M-RoPE")
    source_needed = max(
        (value.source_start + value.length for value in plan.reused),
        default=0)
    if source_kv.compressed_mla or source_kv.offset < source_needed:
        raise ValueError("tool PIC requires a complete dense source KV")
    if len(source_kv.keys) != cfg.num_hidden_layers:
        raise ValueError("tool PIC source KV layer count mismatch")
    if not plan.selected_positions:
        raise ValueError("tool PIC requires at least one recomputed position")
    if isinstance(source_kv, PositionFreeKVCache):
        return _prefill_with_shared_tool_capsules(
            engine, tokens, source_kv, plan)

    selected_positions = mx.array(plan.selected_positions, dtype=mx.int32)
    selected_tokens = [tokens[position] for position in plan.selected_positions]
    x = engine._embed(selected_tokens)
    destination = KVCache(cfg.num_hidden_layers)
    layout = tuple(_layout_parts(plan, len(tokens)))

    for layer in range(cfg.num_hidden_layers):
        source_keys = source_kv.keys[layer]
        source_values = source_kv.values[layer]
        if source_keys is None or source_values is None:
            raise ValueError(f"tool PIC source is missing layer {layer} KV")
        resident_moe = (
            engine._resident_moe_layers[layer]
            if supported_moe
            and getattr(engine, "_resident_moe_layers", None) is not None
            else None)
        weights = (resident_moe[0] if resident_moe is not None
                   else engine.cache.get(
                       engine._layer_key(layer), engine._layer_names(layer)))
        prefix = f"model.layers.{layer}"
        hidden = mx.fast.rms_norm(
            x, weights[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
        linear = layer_runner._linear
        batch, selected_count, _ = hidden.shape
        q = linear(hidden, weights, f"{prefix}.self_attn.q_proj")
        k = linear(hidden, weights, f"{prefix}.self_attn.k_proj")
        v = linear(hidden, weights, f"{prefix}.self_attn.v_proj")
        q_norm = weights.get(f"{prefix}.self_attn.q_norm.weight")
        k_norm = weights.get(f"{prefix}.self_attn.k_norm.weight")
        per_head_norm = q_norm is not None and q_norm.shape[0] == cfg.head_dim
        if q_norm is not None and not per_head_norm:
            q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
            k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
        q = q.reshape(
            batch, selected_count, cfg.num_attention_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(
            batch, selected_count, cfg.num_key_value_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(
            batch, selected_count, cfg.num_key_value_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        if per_head_norm:
            q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
            k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
        q = _apply_rope(
            q, selected_positions, cfg, engine._rope_freqs,
            scale=engine._mscale)
        k = _apply_rope(
            k, selected_positions, cfg, engine._rope_freqs,
            scale=engine._mscale)

        key_parts = []
        value_parts = []
        for kind, start, end, source_or_selected in layout:
            width = end - start
            if kind == "selected":
                key_parts.append(k[:, :, source_or_selected:source_or_selected + width, :])
                value_parts.append(v[:, :, source_or_selected:source_or_selected + width, :])
                continue
            old_start = source_or_selected
            old_keys = source_keys[:, :, old_start:old_start + width, :]
            if old_start != start:
                delta = mx.full((width,), start - old_start, dtype=mx.int32)
                old_keys = _apply_rope(
                    old_keys, delta, cfg, engine._rope_freqs)
            key_parts.append(old_keys)
            value_parts.append(
                source_values[:, :, old_start:old_start + width, :])
        keys = key_parts[0] if len(key_parts) == 1 else mx.concatenate(
            key_parts, axis=2)
        values = value_parts[0] if len(value_parts) == 1 else mx.concatenate(
            value_parts, axis=2)
        destination.keys[layer] = keys
        destination.values[layer] = values

        key_positions = mx.arange(len(tokens), dtype=mx.int32)[None, :]
        mask = mx.where(
            key_positions <= selected_positions[:, None],
            0.0, float("-inf")).astype(q.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=cfg.head_dim ** -0.5, mask=mask)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, selected_count, cfg.num_attention_heads * cfg.head_dim)
        x = x + linear(attended, weights, f"{prefix}.self_attn.o_proj")
        if supported_moe:
            if resident_moe is not None:
                x = layer_runner.run_fused_moe_mlp(
                    x, weights, resident_moe[1], prefix, cfg, layer,
                    fused_swiglu=engine.rc.fused_swiglu,
                    mlx_router_semantics=True)
            else:
                x = layer_runner.run_moe_mlp(
                    x, weights, prefix, cfg, layer, engine._get_experts)
        else:
            hidden = mx.fast.rms_norm(
                x, weights[f"{prefix}.post_attention_layernorm.weight"],
                cfg.rms_norm_eps)
            x = x + layer_runner._swiglu(
                hidden, weights, f"{prefix}.mlp", fused=engine.rc.fused_swiglu)
        mx.eval(x, keys, values)

    logits = layer_runner.final_logits(
        x[:, -1:, :], engine._norm_w, engine._lm_head_weight(),
        cfg.rms_norm_eps)
    mx.eval(logits)
    engine._h_window = x[:, -1:, :]
    engine._h_last = x[:, -1:, :]
    return destination, logits


def _prefill_with_shared_tool_capsules(
        engine, tokens, source_kv: PositionFreeKVCache, plan: ToolPICPlan):
    """Selective PIC sweep that shares immutable, unrotated physical pages.

    Reused positions are represented only by page ids. Selected/repair positions
    receive fresh pages, and each layer writes those pages before attending to the
    complete logical block table. Wide selective sweeps gather a temporary K/V
    view for MLX SDPA; small sweeps use the custom page-table kernel directly.
    """
    tokens = tuple(tokens)
    cfg = engine.cfg
    supported_moe = cfg.model_type == "olmoe" and bool(cfg.num_experts)
    selected_tuple = tuple(plan.selected_positions)
    selected_positions = mx.array(selected_tuple, dtype=mx.int32)
    selected_tokens = [tokens[position] for position in selected_tuple]
    x = engine._embed(selected_tokens)
    destination = PositionFreeKVCache.from_pic_plan(
        source_kv, plan, len(tokens))
    try:
        for layer in range(cfg.num_hidden_layers):
            resident_moe = (
                engine._resident_moe_layers[layer]
                if supported_moe
                and getattr(engine, "_resident_moe_layers", None) is not None
                else None)
            weights = (resident_moe[0] if resident_moe is not None
                       else engine.cache.get(
                           engine._layer_key(layer), engine._layer_names(layer)))
            prefix = f"model.layers.{layer}"
            hidden = mx.fast.rms_norm(
                x, weights[f"{prefix}.input_layernorm.weight"],
                cfg.rms_norm_eps)
            linear = layer_runner._linear
            batch, selected_count, _ = hidden.shape
            q = linear(hidden, weights, f"{prefix}.self_attn.q_proj")
            k = linear(hidden, weights, f"{prefix}.self_attn.k_proj")
            v = linear(hidden, weights, f"{prefix}.self_attn.v_proj")
            q_norm = weights.get(f"{prefix}.self_attn.q_norm.weight")
            k_norm = weights.get(f"{prefix}.self_attn.k_norm.weight")
            per_head_norm = (
                q_norm is not None and q_norm.shape[0] == cfg.head_dim)
            if q_norm is not None and not per_head_norm:
                q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
                k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
            q = q.reshape(
                batch, selected_count, cfg.num_attention_heads,
                cfg.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(
                batch, selected_count, cfg.num_key_value_heads,
                cfg.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(
                batch, selected_count, cfg.num_key_value_heads,
                cfg.head_dim).transpose(0, 2, 1, 3)
            if per_head_norm:
                q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
                k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)

            # YaRN's attention scale is position-independent and therefore part
            # of the stored key. Only RoPE itself is deferred to logical read.
            if engine._mscale != 1.0:
                k = k * engine._mscale
            q = _apply_rope(
                q, selected_positions, cfg, engine._rope_freqs,
                scale=engine._mscale)
            destination.write_selected(layer, selected_tuple, k, v)

            materialize_rotated_view = (
                selected_count > destination.custom_attention_query_limit
                or len(tokens) >= destination.rotated_view_min_keys)
            use_paged_kernel = (
                not materialize_rotated_view
                and selected_count
                <= destination.custom_attention_query_limit
                and cfg.head_dim % 32 == 0
                and mx.metal.is_available()
            )
            if use_paged_kernel:
                attended = destination.paged_attention(
                    layer, q, selected_positions, theta=cfg.rope_theta,
                    denominators=engine._rope_freqs,
                    scale=cfg.head_dim ** -0.5)
            else:
                keys, values = destination.gather_unrotated(layer)
                if engine._rope_freqs is None:
                    keys = mx.fast.rope(
                        keys, cfg.head_dim, traditional=False,
                        base=cfg.rope_theta, scale=1.0, offset=0)
                else:
                    keys = mx.fast.rope(
                        keys, cfg.head_dim, traditional=False, base=None,
                        freqs=engine._rope_freqs, scale=1.0, offset=0)
                if materialize_rotated_view:
                    destination.set_rotated_view(layer, keys, values)
                key_positions = mx.arange(len(tokens), dtype=mx.int32)[None, :]
                mask = mx.where(
                    key_positions <= selected_positions[:, None],
                    0.0, float("-inf")).astype(q.dtype)
                attended = mx.fast.scaled_dot_product_attention(
                    q, keys, values, scale=cfg.head_dim ** -0.5, mask=mask)
            attended = attended.transpose(0, 2, 1, 3).reshape(
                batch, selected_count,
                cfg.num_attention_heads * cfg.head_dim)
            x = x + linear(attended, weights, f"{prefix}.self_attn.o_proj")
            if supported_moe:
                if resident_moe is not None:
                    x = layer_runner.run_fused_moe_mlp(
                        x, weights, resident_moe[1], prefix, cfg, layer,
                        fused_swiglu=engine.rc.fused_swiglu,
                        mlx_router_semantics=True)
                else:
                    x = layer_runner.run_moe_mlp(
                        x, weights, prefix, cfg, layer, engine._get_experts)
            else:
                hidden = mx.fast.rms_norm(
                    x, weights[f"{prefix}.post_attention_layernorm.weight"],
                    cfg.rms_norm_eps)
                x = x + layer_runner._swiglu(
                    hidden, weights, f"{prefix}.mlp",
                    fused=engine.rc.fused_swiglu)
            mx.eval(x)

        logits = layer_runner.final_logits(
            x[:, -1:, :], engine._norm_w, engine._lm_head_weight(),
            cfg.rms_norm_eps)
        mx.eval(logits)
        engine._h_window = x[:, -1:, :]
        engine._h_last = x[:, -1:, :]
        return destination, logits
    except Exception:
        destination.release()
        raise
