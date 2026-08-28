"""GLM-5.3-Flash text block composed from vOOM's exact primitives.

The released model combines four-stream mHC residuals, Kimi Delta Attention
on 34/45 layers, NoPE MLA+pooled DSA on the other 11, and GLM noaux_tc MoE.
This module owns only that composition and the model's asymmetric SwiGLU
clamp; mHC, KDA, MLA, routing, paging, and cache state remain in their
independently tested implementations.
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig
from .deepseek_v4 import run_deepseek_v4_block
from .expert_batching import consume_expert_batches
from .glm import _group_routes, _mla_attention, _route_experts
from .kimi_linear import _kda_attention
from .layer_runner import _linear


def glm5_next_swiglu(
        x: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> mx.array:
    """Released GLM-5.3 SwiGLU with its asymmetric activation clamp."""
    gate = _linear(x, w, f"{prefix}.gate_proj")
    up = _linear(x, w, f"{prefix}.up_proj")
    limit = float(cfg.swiglu_limit)
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    activated = (mx.sigmoid(gate) * gate) * up
    return _linear(activated, w, f"{prefix}.down_proj")


def glm5_next_mlp(
        h: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
        get_experts, *, iter_expert_batches=None, profile=None) -> mx.array:
    """Dense/MoE sublayer output before mHC re-expansion."""
    is_dense = (
        cfg.mlp_layer_types[layer] == "dense"
        if layer < len(cfg.mlp_layer_types)
        else layer < cfg.first_k_dense_replace
    )
    if is_dense:
        return glm5_next_swiglu(h, w, f"{prefix}.mlp", cfg)

    router_t0 = profile.start_substep() if profile is not None else None
    idx, route_weights = _route_experts(h, w, prefix, cfg)
    if not (profile is not None and profile.finish_substep(
            "router", layer, router_t0, idx, route_weights,
            positions=int(h.shape[1]))):
        mx.eval(idx, route_weights)
    groups = _group_routes(idx, route_weights)

    # The released eager loop starts from zero, adds routed experts in
    # ascending expert-id order, then adds the always-on shared expert.
    out = mx.zeros_like(h)
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
        nonlocal out
        for expert in batch_ids:
            placements = groups[expert]
            positions = [position for position, _ in placements]
            weights = mx.array(
                [weight for _, weight in placements]).astype(mx.float32)
            value = glm5_next_swiglu(
                h[:, positions, :], experts[expert],
                f"{prefix}.mlp.experts.{expert}", cfg)
            contribution = (
                value * weights[None, :, None]).astype(h.dtype)
            out = out.at[:, positions, :].add(contribution)
        # Bound strong references to the current expert batch before the next
        # storage fetch, matching the existing GLM/Kimi paging lifetime.
        mx.eval(out)

    consume_expert_batches(batches, consume_batch)
    return out + glm5_next_swiglu(
        h, w, f"{prefix}.mlp.shared_experts", cfg)


def glm5_next_mlp_layer_stationary_tiles(
        hidden_tiles: list[mx.array], w: dict, prefix: str,
        cfg: ModelConfig, layer: int, get_experts, *,
        iter_expert_batches=None, profile=None) -> list[mx.array]:
    """Run GLM-5.3 MLPs at their original tile GEMM shapes.

    For MoE layers, routing is still evaluated independently per tile. The
    loop nest is then inverted from ``tile -> expert page`` to ``expert page
    -> tiles`` so one released page serves every tile that selected it. Within
    each tile experts remain accumulated in ascending ID order, with the same
    materialization boundary used by the q=1 reference profile.
    """
    if not hidden_tiles:
        return []
    is_dense = (
        cfg.mlp_layer_types[layer] == "dense"
        if layer < len(cfg.mlp_layer_types)
        else layer < cfg.first_k_dense_replace
    )
    if is_dense:
        outputs = []
        for tile in hidden_tiles:
            value = glm5_next_swiglu(
                tile, w, f"{prefix}.mlp", cfg)
            mx.eval(value)
            outputs.append(value)
        return outputs

    groups_by_tile = []
    global_positions: dict[int, list[int]] = {}
    position_base = 0
    for tile in hidden_tiles:
        router_t0 = profile.start_substep() if profile is not None else None
        idx, route_weights = _route_experts(tile, w, prefix, cfg)
        if not (profile is not None and profile.finish_substep(
                "router", layer, router_t0, idx, route_weights,
                positions=int(tile.shape[1]))):
            mx.eval(idx, route_weights)
        groups = _group_routes(idx, route_weights)
        groups_by_tile.append(groups)
        for expert, placements in groups.items():
            global_positions.setdefault(expert, []).extend(
                position_base + position for position, _ in placements)
        position_base += int(tile.shape[1])

    expert_ids = sorted(global_positions)
    routed = [mx.zeros_like(tile) for tile in hidden_tiles]
    # Shared outputs use precisely the same per-tile GEMM shapes as the
    # chunk-major reference and are evaluated before routed page streaming.
    shared = []
    for tile in hidden_tiles:
        value = glm5_next_swiglu(
            tile, w, f"{prefix}.mlp.shared_experts", cfg)
        mx.eval(value)
        shared.append(value)

    if iter_expert_batches is None:
        experts = get_experts(
            layer, expert_ids, positions=global_positions)
        batches = ((expert_ids, experts),)
    else:
        batches = iter_expert_batches(
            layer, expert_ids, positions=global_positions)

    def consume_batch(batch_ids, experts):
        touched = []
        for expert in batch_ids:
            for tile_index, (tile, groups) in enumerate(zip(
                    hidden_tiles, groups_by_tile)):
                placements = groups.get(expert)
                if not placements:
                    continue
                positions = [position for position, _ in placements]
                weights = mx.array(
                    [weight for _, weight in placements]).astype(mx.float32)
                value = glm5_next_swiglu(
                    tile[:, positions, :], experts[expert],
                    f"{prefix}.mlp.experts.{expert}", cfg)
                contribution = (
                    value * weights[None, :, None]).astype(tile.dtype)
                routed[tile_index] = routed[tile_index].at[
                    :, positions, :].add(contribution)
                touched.append(tile_index)
        # The production q=1 fetch profile materializes after every one-page
        # batch. Preserve that boundary independently for each affected tile.
        for tile_index in sorted(set(touched)):
            mx.eval(routed[tile_index])

    consume_expert_batches(batches, consume_batch)
    outputs = []
    for routed_value, shared_value in zip(routed, shared):
        value = routed_value + shared_value
        mx.eval(value)
        outputs.append(value)
    return outputs


def glm5_next_mla_attention(
        hidden: mx.array, w: dict, prefix: str, cfg: ModelConfig,
        kv, layer: int, offset: int, *,
        indexer_type_override: str | None = None) -> mx.array:
    """NoPE MLA with row-specific pooled-DSA gather for every query width."""
    if not getattr(kv, "compressed_mla", False):
        return _mla_attention(hidden, w, prefix, cfg, kv, layer, offset)

    batch, length, _ = hidden.shape
    heads = cfg.num_attention_heads
    key_dim = cfg.qk_nope_head_dim
    value_dim = cfg.v_head_dim
    q_resid = _linear(hidden, w, f"{prefix}.self_attn.q_a_proj")
    q_resid = mx.fast.rms_norm(
        q_resid, w[f"{prefix}.self_attn.q_a_layernorm.weight"],
        cfg.mla_latent_norm_eps)
    query = _linear(
        q_resid, w, f"{prefix}.self_attn.q_b_proj").reshape(
            batch, length, heads, key_dim).transpose(0, 2, 1, 3)

    latent = _linear(
        hidden, w, f"{prefix}.self_attn.kv_a_proj_with_mqa")
    latent = mx.fast.rms_norm(
        latent, w[f"{prefix}.self_attn.kv_a_layernorm.weight"],
        cfg.mla_latent_norm_eps)
    latent_all = kv.update_latent(layer, latent)
    key_length = int(latent_all.shape[1])

    selection = None
    dsa = getattr(kv, "dsa", None)
    if dsa is not None:
        indexer_type = (
            indexer_type_override
            if indexer_type_override is not None
            else cfg.indexer_types[layer])
        dsa.observe(layer, indexer_type, hidden, w, prefix, offset)
        if key_length > cfg.index_topk:
            selection = dsa.update_and_select(
                layer, indexer_type, hidden, q_resid, w, prefix, offset)

    if selection is None:
        expanded = _linear(
            latent_all, w, f"{prefix}.self_attn.kv_b_proj").reshape(
                batch, key_length, heads, key_dim + value_dim).transpose(
                    0, 2, 1, 3)
        keys, values = expanded[..., :key_dim], expanded[..., key_dim:]
        mask = None
        if length > 1:
            query_positions = mx.arange(
                offset, offset + length, dtype=mx.int32)[:, None]
            key_positions = mx.arange(
                key_length, dtype=mx.int32)[None, :]
            mask = mx.where(
                key_positions <= query_positions,
                0.0, float("-inf")).astype(query.dtype)
        output = mx.fast.scaled_dot_product_attention(
            query, keys, values, scale=key_dim ** -0.5, mask=mask)
    else:
        if selection.shape[:2] != (batch, length):
            raise ValueError(
                "GLM-5.3 DSA selection is not query-position aligned")
        safe = mx.where(selection >= 0, selection, 0)
        # B=1 is the serving shape, but retain a correct bounded batch path;
        # each row's index set is inherently different and cannot be expressed
        # as one shared axis-1 take.
        gathered = mx.stack([
            mx.take(latent_all[b], safe[b], axis=0)
            for b in range(batch)
        ], axis=0)
        width = int(gathered.shape[2])
        expanded = _linear(
            gathered, w, f"{prefix}.self_attn.kv_b_proj").reshape(
                batch, length, width, heads, key_dim + value_dim).transpose(
                    0, 3, 1, 2, 4)
        keys, values = expanded[..., :key_dim], expanded[..., key_dim:]
        scores = mx.sum(
            query[..., None, :].astype(mx.float32)
            * keys.astype(mx.float32), axis=-1) * (key_dim ** -0.5)
        scores = mx.where(
            (selection >= 0)[:, None, :, :], scores, float("-inf"))
        probabilities = mx.softmax(scores, axis=-1).astype(values.dtype)
        output = mx.sum(probabilities[..., None] * values, axis=-2)

    output = output.transpose(0, 2, 1, 3).reshape(
        batch, length, heads * value_dim)
    return _linear(output, w, f"{prefix}.self_attn.o_proj")


def run_glm5_next_mtp_block(
        x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
        layer: int, offset: int, get_experts, *,
        iter_expert_batches=None, profile=None) -> mx.array:
    """Run the released layer-45 NextN decoder block.

    The checkpoint's NextN layer deliberately has no mHC parameters.  It is a
    conventional pre-norm residual DSA+MoE block fed by ``eh_proj``; using the
    trunk's four-stream composer would therefore both look up nonexistent
    weights and change the released draft distribution.  The draft remains a
    proposal source only—every emitted token is verified by the target.
    """
    attn_input = mx.fast.rms_norm(
        x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    attn = glm5_next_mla_attention(
        attn_input, w, prefix, cfg, kv, layer, offset,
        indexer_type_override="full")
    hidden = (x + attn).astype(mx.bfloat16)
    mlp_input = mx.fast.rms_norm(
        hidden, w[f"{prefix}.post_attention_layernorm.weight"],
        cfg.rms_norm_eps)
    mlp = glm5_next_mlp(
        mlp_input, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
    return (hidden + mlp).astype(mx.bfloat16)


def run_glm5_next_block(
        x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
        layer: int, offset: int, get_experts, *, mlp_last_only: bool = False,
        iter_expert_batches=None, native_fused_kda_decode: bool = False,
        native_fused_kda_prefill: bool = False,
        compiled_kda_prefill: bool = False, profile=None) -> mx.array:
    """Run one released GLM-5.3 decoder block over four mHC streams."""
    hc = {
        "attn_fn": w[f"{prefix}.hc_attn_fn"],
        "attn_scale": w[f"{prefix}.hc_attn_scale"],
        "attn_base": w[f"{prefix}.hc_attn_base"],
        "ffn_fn": w[f"{prefix}.hc_ffn_fn"],
        "ffn_scale": w[f"{prefix}.hc_ffn_scale"],
        "ffn_base": w[f"{prefix}.hc_ffn_base"],
    }
    norms = {
        "attn": w[f"{prefix}.input_layernorm.weight"],
        "ffn": w[f"{prefix}.post_attention_layernorm.weight"],
    }

    positions = int(x.shape[1])

    def attention(hidden):
        started = profile.start_substep() if profile is not None else None
        if layer in cfg.kda_layers:
            result = _kda_attention(
                hidden, w, prefix, cfg,
                getattr(kv, "kda_cache", None), layer,
                native_fused_decode=native_fused_kda_decode,
                native_fused_prefill=native_fused_kda_prefill,
                compiled_prefill=compiled_kda_prefill,
                released_output_dtype=True,
                profile=profile,
            )
        elif layer in cfg.full_attn_layers:
            result = glm5_next_mla_attention(
                hidden, w, prefix, cfg, kv, layer, offset)
        else:
            raise ValueError(
                f"GLM-5.3 layer {layer} is in neither KDA nor DSA layout")
        if profile is not None:
            profile.finish_substep(
                "attention", layer, started, result, positions=positions)
        return result

    def ffn(hidden):
        started = profile.start_substep() if profile is not None else None
        result = glm5_next_mlp(
            hidden, w, prefix, cfg, layer, get_experts,
            iter_expert_batches=iter_expert_batches, profile=profile)
        if profile is not None:
            profile.finish_substep(
                "mlp", layer, started, result,
                positions=int(hidden.shape[1]))
        return result

    result = run_deepseek_v4_block(
        x, hc, norms, attention, ffn,
        hc_mult=cfg.hc_mult,
        norm_eps=cfg.rms_norm_eps,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
    )
    if mlp_last_only and result.shape[1] > 1:
        result = result[:, -1:, :, :]
    return result
