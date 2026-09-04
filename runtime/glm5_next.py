"""GLM-5.3-Flash text block composed from vOOM's exact primitives.

The released model combines four-stream mHC residuals, Kimi Delta Attention
on 34/45 layers, NoPE MLA+pooled DSA on the other 11, and GLM noaux_tc MoE.
This module owns only that composition and the model's asymmetric SwiGLU
clamp; mHC, KDA, MLA, routing, paging, and cache state remain in their
independently tested implementations.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import quant
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
        iter_expert_batches=None, profile=None, memory_guard=None,
        coalesce_expert_positions: bool = False,
        coalesced_expert_max_positions: int = 512,
        coalesced_stats: dict | None = None) -> list[mx.array]:
    """Run GLM-5.3 MLPs at their original tile GEMM shapes.

    For MoE layers, routing is still evaluated independently per tile. The
    loop nest is then inverted from ``tile -> expert page`` to ``expert page
    -> tiles`` so one released page serves every tile that selected it. Within
    each tile experts remain accumulated in ascending ID order, with the same
    materialization boundary used by the q=1 reference profile.
    """
    if not hidden_tiles:
        return []
    coalesced_expert_max_positions = int(
        coalesced_expert_max_positions)
    if (coalesce_expert_positions
            and coalesced_expert_max_positions <= 0):
        raise ValueError(
            "coalesced_expert_max_positions must be positive")
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
            if memory_guard is not None:
                memory_guard("dense_mlp_tile")
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
        if memory_guard is not None:
            memory_guard("router_tile")
        groups = _group_routes(idx, route_weights)
        groups_by_tile.append(groups)
        for expert, placements in groups.items():
            global_positions.setdefault(expert, []).extend(
                position_base + position for position, _ in placements)
        position_base += int(tile.shape[1])

    expert_ids = sorted(global_positions)
    routed = [mx.zeros_like(tile) for tile in hidden_tiles]
    route_widths = [
        len(placements)
        for groups in groups_by_tile
        for placements in groups.values()
    ]
    if coalesced_stats is not None:
        coalesced_stats["exact_expert_layers"] = int(
            coalesced_stats.get("exact_expert_layers", 0)) + 1
        coalesced_stats["exact_expert_tiles"] = int(
            coalesced_stats.get("exact_expert_tiles", 0)) + len(hidden_tiles)
        coalesced_stats["exact_expert_swiglu_calls"] = int(
            coalesced_stats.get("exact_expert_swiglu_calls", 0)
        ) + len(route_widths)
        coalesced_stats["exact_expert_rows"] = int(
            coalesced_stats.get("exact_expert_rows", 0)
        ) + sum(route_widths)
        coalesced_stats["exact_expert_max_rows"] = max(
            int(coalesced_stats.get("exact_expert_max_rows", 0)),
            max(route_widths, default=0))
        for label, lower, upper in (
                ("rows_1", 1, 1), ("rows_2", 2, 2),
                ("rows_3_4", 3, 4), ("rows_5_8", 5, 8),
                ("rows_9_16", 9, 16), ("rows_17_32", 17, 32),
                ("rows_33_plus", 33, None)):
            matches = sum(
                width >= lower and (upper is None or width <= upper)
                for width in route_widths)
            key = f"exact_expert_{label}_calls"
            coalesced_stats[key] = int(
                coalesced_stats.get(key, 0)) + matches
    if coalesced_stats is not None and coalesce_expert_positions:
        route_assignments = sum(
            len(positions) for positions in global_positions.values())
        coalesced_stats["layers"] = int(
            coalesced_stats.get("layers", 0)) + 1
        coalesced_stats["input_positions"] = int(
            coalesced_stats.get("input_positions", 0)) + position_base
        coalesced_stats["route_assignments"] = int(
            coalesced_stats.get("route_assignments", 0)) + route_assignments
        coalesced_stats["unique_experts"] = int(
            coalesced_stats.get("unique_experts", 0)) + len(expert_ids)
        coalesced_stats["max_unique_experts"] = max(
            int(coalesced_stats.get("max_unique_experts", 0)),
            len(expert_ids))
        coalesced_stats["max_expert_routes"] = max(
            int(coalesced_stats.get("max_expert_routes", 0)),
            max((len(global_positions[expert]) for expert in expert_ids),
                default=0))

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
            if coalesce_expert_positions:
                inputs = []
                destinations = []
                for tile_index, (tile, groups) in enumerate(zip(
                        hidden_tiles, groups_by_tile)):
                    placements = groups.get(expert)
                    if not placements:
                        continue
                    positions = [position for position, _ in placements]
                    placement_weights = [
                        weight for _, weight in placements]
                    for placement_start in range(
                            0, len(positions),
                            coalesced_expert_max_positions):
                        placement_end = min(
                            placement_start
                            + coalesced_expert_max_positions,
                            len(positions))
                        bounded_positions = positions[
                            placement_start:placement_end]
                        weights = mx.array(placement_weights[
                            placement_start:placement_end]).astype(mx.float32)
                        inputs.append(tile[:, bounded_positions, :])
                        destinations.append((
                            tile_index, bounded_positions, weights,
                            len(bounded_positions)))
                if not inputs:
                    continue
                # A full 46.8K prompt can route many thousands of rows to one
                # hot expert. One all-context GEMM was fast at 8K but learned a
                # 2.5+ GB transient and was correctly refused at 46.8K. Keep
                # the expert page resident while bounding only its gathered
                # operand. Oversized tile placements are split contiguously;
                # scatter order and per-position routing weights remain
                # unchanged.
                chunks = []
                chunk_inputs = []
                chunk_destinations = []
                chunk_width = 0
                for input_value, destination in zip(inputs, destinations):
                    width = int(destination[3])
                    if (chunk_destinations
                            and chunk_width + width
                            > coalesced_expert_max_positions):
                        chunks.append((
                            chunk_inputs, chunk_destinations, chunk_width))
                        chunk_inputs = []
                        chunk_destinations = []
                        chunk_width = 0
                    chunk_inputs.append(input_value)
                    chunk_destinations.append(destination)
                    chunk_width += width
                if chunk_destinations:
                    chunks.append((
                        chunk_inputs, chunk_destinations, chunk_width))

                if coalesced_stats is not None:
                    coalesced_stats["gemm_calls"] = int(
                        coalesced_stats.get("gemm_calls", 0)) + len(chunks)
                    coalesced_stats["gemm_input_positions"] = int(
                        coalesced_stats.get("gemm_input_positions", 0)) + sum(
                            width for _inputs, _destinations, width in chunks)
                    coalesced_stats["gemm_full_chunks"] = int(
                        coalesced_stats.get("gemm_full_chunks", 0)) + sum(
                            width == coalesced_expert_max_positions
                            for _inputs, _destinations, width in chunks)
                    coalesced_stats["max_positions"] = max(
                        int(coalesced_stats.get("max_positions", 0)),
                        max(width for _inputs, _destinations, width in chunks))
                    if len(chunks) > 1:
                        coalesced_stats["split_experts"] = int(
                            coalesced_stats.get("split_experts", 0)) + 1

                for chunk_inputs, chunk_destinations, _width in chunks:
                    expert_input = (
                        chunk_inputs[0] if len(chunk_inputs) == 1
                        else mx.concatenate(chunk_inputs, axis=1))
                    value = glm5_next_swiglu(
                        expert_input, experts[expert],
                        f"{prefix}.mlp.experts.{expert}", cfg)
                    mx.eval(value)
                    cursor = 0
                    chunk_touched = []
                    for (tile_index, positions, weights,
                         width) in chunk_destinations:
                        contribution = (
                            value[:, cursor:cursor + width]
                            * weights[None, :, None]).astype(
                                hidden_tiles[tile_index].dtype)
                        routed[tile_index] = routed[tile_index].at[
                            :, positions, :].add(contribution)
                        touched.append(tile_index)
                        chunk_touched.append(tile_index)
                        cursor += width
                    # Materialize before constructing the next bounded chunk;
                    # otherwise MLX retains every prior expert intermediate
                    # in one lazy graph and defeats the position ceiling.
                    for tile_index in sorted(set(chunk_touched)):
                        mx.eval(routed[tile_index])
                continue
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
        if memory_guard is not None:
            memory_guard("routed_expert_batch")

    consume_expert_batches(batches, consume_batch)
    for tile_index, (routed_value, tile) in enumerate(zip(
            routed, hidden_tiles)):
        # Compute the independent shared expert only when this tile's routed
        # accumulation is complete.  The released result is still exactly
        # ``routed + shared`` with the same GEMM shapes and addition order, but
        # a 46,849-token prompt no longer retains every shared output at once
        # (about 384 MB of avoidable BF16 Metal storage on GLM-5.3-Flash).
        shared_value = glm5_next_swiglu(
            tile, w, f"{prefix}.mlp.shared_experts", cfg)
        mx.eval(shared_value)
        if memory_guard is not None:
            memory_guard("shared_mlp_tile")
        value = routed_value + shared_value
        mx.eval(value)
        # Replace the completed routed buffer immediately. Keeping a separate
        # output list would retain both full-context tensors until return,
        # another ~384 MB peak on the real 46,849-token harness capture.
        routed[tile_index] = value
    return routed


def _glm5_next_sparse_mla_attention(
        query: mx.array, latent_all: mx.array, selection: mx.array,
        w: dict, prefix: str, *, heads: int, key_dim: int, value_dim: int,
        query_tile_size: int = 4) -> mx.array:
    """Evaluate the released row-specific DSA gather in bounded query tiles.

    GLM-5.3 selects 2,048 different latent rows for every query. Expanding a
    32-query prefill tile in one expression materializes roughly 4.3 GB of
    per-head K/V before scores/probabilities and crossed 12.7 GB peak Metal on
    the 16 GB host. Query rows are independent after their indices are fixed;
    slicing only that outer batch axis preserves each row's operations and
    selected-key order while bounding the live gather. Materialize every
    result tile so MLX cannot retain all sparse expansion graphs until the
    caller's later synchronization point.
    """
    batch, query_heads, length, query_dim = query.shape
    if query_heads != heads or query_dim != key_dim:
        raise ValueError("GLM-5.3 sparse MLA query shape is inconsistent")
    if selection.shape[:2] != (batch, length):
        raise ValueError(
            "GLM-5.3 DSA selection is not query-position aligned")
    tile_size = max(1, int(query_tile_size))
    outputs = []
    scale = key_dim ** -0.5
    for start in range(0, length, tile_size):
        end = min(start + tile_size, length)
        selected = selection[:, start:end]
        safe = mx.where(selected >= 0, selected, 0)
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
                batch, end - start, width, heads,
                key_dim + value_dim).transpose(0, 3, 1, 2, 4)
        keys, values = expanded[..., :key_dim], expanded[..., key_dim:]
        scores = mx.sum(
            query[:, :, start:end, None, :].astype(mx.float32)
            * keys.astype(mx.float32), axis=-1) * scale
        scores = mx.where(
            (selected >= 0)[:, None, :, :], scores, float("-inf"))
        probabilities = mx.softmax(scores, axis=-1).astype(values.dtype)
        output = mx.sum(probabilities[..., None] * values, axis=-2)
        mx.eval(output)
        outputs.append(output)
    result = (
        outputs[0] if len(outputs) == 1
        else mx.concatenate(outputs, axis=2))
    mx.eval(result)
    return result


def _glm5_next_sparse_absorbed_mla_attention(
        query: mx.array, latent_all: mx.array, selection: mx.array,
        w: dict, prefix: str, *, heads: int, key_dim: int, value_dim: int,
        query_tile_size: int = 32) -> mx.array:
    """Official serving-layout sparse MLA without per-key K/V expansion.

    Weight absorption applies the two exact real-arithmetic identities
    ``q @ (c @ Wk.T).T = (q @ Wk) @ c.T`` and
    ``p @ (c @ Wv.T) = (p @ c) @ Wv.T``. The official GLM sparse serving
    layout uses the compact latent as K/V for precisely this reason. Floating
    association differs from eager expand-then-attend, so this remains an
    explicit runtime candidate until real greedy/state gates clear it.
    """
    batch, query_heads, length, query_dim = query.shape
    if query_heads != heads or query_dim != key_dim:
        raise ValueError("GLM-5.3 absorbed MLA query shape is inconsistent")
    if selection.shape[:2] != (batch, length):
        raise ValueError(
            "GLM-5.3 DSA selection is not query-position aligned")
    kv_b = w[f"{prefix}.self_attn.kv_b_proj.weight"]
    if isinstance(kv_b, quant.QTensor):
        raise ValueError(
            "GLM-5.3 absorbed MLA requires a dense kv_b_proj weight")
    latent_dim = int(latent_all.shape[-1])
    kv_b = kv_b.reshape(heads, key_dim + value_dim, latent_dim)
    w_key = kv_b[:, :key_dim, :]
    w_value = kv_b[:, key_dim:, :]
    # Project Q into the compact latent space only once per query/head.
    query_latent = mx.einsum("bhqd,hdc->bhqc", query, w_key)
    scale = key_dim ** -0.5
    tile_size = max(1, int(query_tile_size))
    outputs = []
    for start in range(0, length, tile_size):
        end = min(start + tile_size, length)
        selected = selection[:, start:end]
        safe = mx.where(selected >= 0, selected, 0)
        gathered = mx.stack([
            mx.take(latent_all[b], safe[b], axis=0)
            for b in range(batch)
        ], axis=0)
        # [B,H,Q,1,C] @ [B,1,Q,C,K] keeps C out of the score working set.
        scores = mx.matmul(
            query_latent[:, :, start:end, None, :].astype(mx.float32),
            gathered[:, None].swapaxes(-1, -2).astype(mx.float32),
        ).squeeze(-2) * scale
        scores = mx.where(
            (selected >= 0)[:, None, :, :], scores, float("-inf"))
        probabilities = mx.softmax(scores, axis=-1).astype(gathered.dtype)
        weighted_latent = mx.matmul(
            probabilities[..., None, :], gathered[:, None]
        ).squeeze(-2)
        output = mx.einsum(
            "bhqc,hdc->bhqd", weighted_latent, w_value)
        mx.eval(output)
        outputs.append(output)
    result = (
        outputs[0] if len(outputs) == 1
        else mx.concatenate(outputs, axis=2))
    mx.eval(result)
    return result


def _glm5_next_dense_absorbed_mla_attention(
        query: mx.array, latent_all: mx.array, w: dict, prefix: str, *,
        heads: int, key_dim: int, value_dim: int, offset: int) -> mx.array:
    """Absorbed MLA for the causal dense prefix before DSA activates."""
    batch, query_heads, length, query_dim = query.shape
    if query_heads != heads or query_dim != key_dim:
        raise ValueError("GLM-5.3 dense absorbed MLA query shape is inconsistent")
    kv_b = w[f"{prefix}.self_attn.kv_b_proj.weight"]
    if isinstance(kv_b, quant.QTensor):
        raise ValueError(
            "GLM-5.3 absorbed MLA requires a dense kv_b_proj weight")
    latent_dim = int(latent_all.shape[-1])
    kv_b = kv_b.reshape(heads, key_dim + value_dim, latent_dim)
    w_key = kv_b[:, :key_dim, :]
    w_value = kv_b[:, key_dim:, :]
    query_latent = mx.einsum("bhqd,hdc->bhqc", query, w_key)
    scores = mx.matmul(
        query_latent.astype(mx.float32),
        latent_all[:, None].swapaxes(-1, -2).astype(mx.float32),
    ) * (key_dim ** -0.5)
    if length > 1:
        query_positions = mx.arange(
            offset, offset + length, dtype=mx.int32)[:, None]
        key_positions = mx.arange(
            latent_all.shape[1], dtype=mx.int32)[None, :]
        scores = mx.where(
            (key_positions <= query_positions)[None, None],
            scores, float("-inf"))
    probabilities = mx.softmax(scores, axis=-1).astype(latent_all.dtype)
    weighted_latent = mx.matmul(probabilities, latent_all[:, None])
    output = mx.einsum("bhqc,hdc->bhqd", weighted_latent, w_value)
    mx.eval(output)
    return output


def _glm5_next_update_expanded_prefill_kv(
        latent: mx.array, w: dict, prefix: str, *, layer: int,
        cache: dict, heads: int, key_dim: int, value_dim: int,
        step: int = 256) -> tuple[mx.array, mx.array]:
    """Project each released latent row once into exact prefill K/V.

    This is a request-local, single-active-layer cache. It exists only while
    layer-stationary prefill is processing one DSA layer and is discarded
    before that layer's MLP. The durable cache remains compact MLA latents.
    """
    batch, incoming, _ = latent.shape
    expanded = _linear(
        latent, w, f"{prefix}.self_attn.kv_b_proj").reshape(
            batch, incoming, heads, key_dim + value_dim).transpose(0, 2, 1, 3)
    new_keys = expanded[..., :key_dim]
    new_values = expanded[..., key_dim:]
    mx.eval(new_keys, new_values)
    entry = cache.get(layer)
    previous = 0 if entry is None else int(entry[2])
    end = previous + incoming
    if entry is None:
        capacity = max(step, ((incoming + step - 1) // step) * step)
        keys = mx.zeros(
            (batch, heads, capacity, key_dim), dtype=new_keys.dtype)
        values = mx.zeros(
            (batch, heads, capacity, value_dim), dtype=new_values.dtype)
    else:
        keys, values, _ = entry
        capacity = int(keys.shape[2])
        if end > capacity:
            added = ((end - capacity + step - 1) // step) * step
            keys = mx.concatenate([
                keys, mx.zeros(
                    (batch, heads, added, key_dim), dtype=keys.dtype)
            ], axis=2)
            values = mx.concatenate([
                values, mx.zeros(
                    (batch, heads, added, value_dim), dtype=values.dtype)
            ], axis=2)
    keys[..., previous:end, :] = new_keys
    values[..., previous:end, :] = new_values
    mx.eval(keys, values)
    cache[layer] = [keys, values, end]
    return keys[..., :end, :], values[..., :end, :]


def _glm5_next_update_host_expanded_prefill_kv(
        latent: mx.array, w: dict, prefix: str, *, layer: int,
        cache: dict, heads: int, key_dim: int, value_dim: int,
        step: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Project once, then retain the exact BF16 payload outside Metal.

    Host arrays contain raw uint16 bits.  Capacity is allocated once for the
    known request length, while only written pages become resident.  Returned
    slices are views over initialized rows; no floating-point conversion is
    involved.
    """
    batch, incoming, _ = latent.shape
    expanded = _linear(
        latent, w, f"{prefix}.self_attn.kv_b_proj").reshape(
            batch, incoming, heads, key_dim + value_dim).transpose(0, 2, 1, 3)
    new_keys = expanded[..., :key_dim]
    new_values = expanded[..., key_dim:]
    if new_keys.dtype != mx.bfloat16 or new_values.dtype != mx.bfloat16:
        raise TypeError(
            "GLM-5.3 expanded K/V host spool requires BF16 projections")
    mx.eval(new_keys, new_values)
    entry = cache.get(layer)
    previous = 0 if entry is None else int(entry[2])
    end = previous + incoming
    migrated_key_bits = migrated_value_bits = None
    if entry is not None and not isinstance(entry[0], np.ndarray):
        # Dense attention is cheap enough to keep its <=top-k expanded prefix
        # on Metal. At the first genuinely sparse tile, migrate those already
        # projected BF16 bits once, then append only new rows on the host.
        migrated_key_bits = np.array(
            np.asarray(entry[0][..., :previous, :].view(mx.uint16)),
            dtype=np.uint16, copy=True)
        migrated_value_bits = np.array(
            np.asarray(entry[1][..., :previous, :].view(mx.uint16)),
            dtype=np.uint16, copy=True)
        entry = None
    if entry is None:
        requested = int(getattr(cache, "capacity_rows", 0) or 0)
        capacity = (
            max(end, requested)
            if requested > 0
            else max(end, ((incoming + step - 1) // step) * step)
        )
        keys = np.empty(
            (batch, heads, capacity, key_dim), dtype=np.uint16)
        values = np.empty(
            (batch, heads, capacity, value_dim), dtype=np.uint16)
        cache.capacity_grows = int(getattr(
            cache, "capacity_grows", 0)) + 1
        if previous:
            keys[..., :previous, :] = migrated_key_bits
            values[..., :previous, :] = migrated_value_bits
            migrated_bytes = int(
                migrated_key_bits.nbytes + migrated_value_bits.nbytes)
            cache.rows_copied = int(getattr(
                cache, "rows_copied", 0)) + previous
            cache.rows_written = int(getattr(
                cache, "rows_written", 0)) + previous
            cache.host_bytes_written = int(getattr(
                cache, "host_bytes_written", 0)) + migrated_bytes
    else:
        keys, values, _ = entry
        capacity = int(keys.shape[2])
        if end > capacity:
            added = ((end - capacity + step - 1) // step) * step
            grown_keys = np.empty(
                (batch, heads, capacity + added, key_dim), dtype=np.uint16)
            grown_values = np.empty(
                (batch, heads, capacity + added, value_dim), dtype=np.uint16)
            grown_keys[..., :previous, :] = keys[..., :previous, :]
            grown_values[..., :previous, :] = values[..., :previous, :]
            cache.rows_copied = int(getattr(
                cache, "rows_copied", 0)) + previous
            cache.capacity_grows = int(getattr(
                cache, "capacity_grows", 0)) + 1
            keys, values = grown_keys, grown_values
            capacity += added
    key_bits = np.asarray(new_keys.view(mx.uint16))
    value_bits = np.asarray(new_values.view(mx.uint16))
    keys[..., previous:end, :] = key_bits
    values[..., previous:end, :] = value_bits
    written = int(key_bits.nbytes + value_bits.nbytes)
    cache.rows_written = int(getattr(
        cache, "rows_written", 0)) + incoming
    cache.host_bytes_written = int(getattr(
        cache, "host_bytes_written", 0)) + written
    cache.capacity_rows_peak = max(
        int(getattr(cache, "capacity_rows_peak", 0)), capacity)
    cache.host_logical_bytes_peak = max(
        int(getattr(cache, "host_logical_bytes_peak", 0)),
        int(keys.nbytes + values.nbytes))
    cache[layer] = [keys, values, end]
    return keys[..., :end, :], values[..., :end, :]


def _glm5_next_restore_host_bf16(
        value: np.ndarray, *, cache=None) -> mx.array:
    """Upload one raw host payload and restore its BF16 interpretation."""
    if value.dtype != np.uint16:
        raise TypeError("GLM-5.3 host K/V payload must be raw uint16")
    result = mx.array(value, dtype=mx.uint16).view(mx.bfloat16)
    mx.eval(result)
    if cache is not None:
        cache.host_bytes_uploaded = int(getattr(
            cache, "host_bytes_uploaded", 0)) + int(value.nbytes)
        cache.host_upload_calls = int(getattr(
            cache, "host_upload_calls", 0)) + 1
    return result


def _glm5_next_quantize_expanded_rows_int8(
        value: mx.array) -> tuple[mx.array, mx.array]:
    """Symmetric per-row int8 storage for explicitly lossy fused DSA K/V."""
    wide = value.astype(mx.float32)
    scale = mx.max(mx.abs(wide), axis=-1, keepdims=True) / 127.0
    safe_scale = mx.where(scale > 0, scale, 1.0)
    quantized = mx.round(wide / safe_scale)
    quantized = mx.minimum(mx.maximum(quantized, -127), 127).astype(mx.int8)
    # Keep scales in FP32: they are only one scalar per head/row (<1% of the
    # compact payload) and avoid compounding the deliberate int8 error.
    return quantized, scale.astype(mx.float32)


def _glm5_next_update_expanded_prefill_kv_int8(
        latent: mx.array, w: dict, prefix: str, *, layer: int,
        cache: dict, heads: int, key_dim: int, value_dim: int,
        step: int = 256,
        ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Project once, then retain per-head/per-row scaled int8 K/V."""
    batch, incoming, _ = latent.shape
    expanded = _linear(
        latent, w, f"{prefix}.self_attn.kv_b_proj").reshape(
            batch, incoming, heads, key_dim + value_dim).transpose(0, 2, 1, 3)
    new_keys, new_key_scales = _glm5_next_quantize_expanded_rows_int8(
        expanded[..., :key_dim])
    new_values, new_value_scales = _glm5_next_quantize_expanded_rows_int8(
        expanded[..., key_dim:])
    mx.eval(new_keys, new_values, new_key_scales, new_value_scales)
    entry = cache.get(layer)
    previous = 0 if entry is None else int(entry[4])
    end = previous + incoming
    if entry is None:
        capacity = max(step, ((incoming + step - 1) // step) * step)
        keys = mx.zeros(
            (batch, heads, capacity, key_dim), dtype=mx.int8)
        values = mx.zeros(
            (batch, heads, capacity, value_dim), dtype=mx.int8)
        key_scales = mx.zeros(
            (batch, heads, capacity, 1), dtype=mx.float32)
        value_scales = mx.zeros(
            (batch, heads, capacity, 1), dtype=mx.float32)
    else:
        keys, values, key_scales, value_scales, _ = entry
        capacity = int(keys.shape[2])
        if end > capacity:
            added = ((end - capacity + step - 1) // step) * step
            keys = mx.concatenate([
                keys, mx.zeros(
                    (batch, heads, added, key_dim), dtype=mx.int8)
            ], axis=2)
            values = mx.concatenate([
                values, mx.zeros(
                    (batch, heads, added, value_dim), dtype=mx.int8)
            ], axis=2)
            key_scales = mx.concatenate([
                key_scales, mx.zeros(
                    (batch, heads, added, 1), dtype=mx.float32)
            ], axis=2)
            value_scales = mx.concatenate([
                value_scales, mx.zeros(
                    (batch, heads, added, 1), dtype=mx.float32)
            ], axis=2)
    keys[..., previous:end, :] = new_keys
    values[..., previous:end, :] = new_values
    key_scales[..., previous:end, :] = new_key_scales
    value_scales[..., previous:end, :] = new_value_scales
    mx.eval(keys, values, key_scales, value_scales)
    cache[layer] = [keys, values, key_scales, value_scales, end]
    return (
        keys[..., :end, :], values[..., :end, :],
        key_scales[..., :end, :], value_scales[..., :end, :])


def _glm5_next_sparse_expanded_attention(
        query: mx.array, keys_all: mx.array, values_all: mx.array,
        selection: mx.array, *, key_dim: int,
        query_tile_size: int = 4, host_stats=None) -> mx.array:
    """Row-specific exact attention over already projected K/V."""
    batch, _heads, length, _ = query.shape
    if selection.shape[:2] != (batch, length):
        raise ValueError(
            "GLM-5.3 DSA selection is not query-position aligned")
    outputs = []
    tile_size = max(1, int(query_tile_size))
    scale = key_dim ** -0.5
    for start in range(0, length, tile_size):
        end = min(start + tile_size, length)
        selected = selection[:, start:end]
        safe = mx.where(selected >= 0, selected, 0)
        if isinstance(keys_all, np.ndarray):
            if not isinstance(values_all, np.ndarray):
                raise TypeError("host-expanded GLM K/V placement is inconsistent")
            safe_host = np.asarray(safe, dtype=np.int32)
            if batch == 1:
                key_bits = np.take(keys_all[0], safe_host[0], axis=1)[None]
                value_bits = np.take(
                    values_all[0], safe_host[0], axis=1)[None]
            else:
                key_bits = np.stack([
                    np.take(keys_all[b], safe_host[b], axis=1)
                    for b in range(batch)
                ], axis=0)
                value_bits = np.stack([
                    np.take(values_all[b], safe_host[b], axis=1)
                    for b in range(batch)
                ], axis=0)
            gathered_keys = _glm5_next_restore_host_bf16(
                key_bits, cache=host_stats)
            gathered_values = _glm5_next_restore_host_bf16(
                value_bits, cache=host_stats)
            if host_stats is not None:
                host_stats.host_sparse_gather_rows = int(getattr(
                    host_stats, "host_sparse_gather_rows", 0)) + int(
                        safe_host.size)
        else:
            gathered_keys = mx.stack([
                mx.take(keys_all[b], safe[b], axis=1)
                for b in range(batch)
            ], axis=0)
            gathered_values = mx.stack([
                mx.take(values_all[b], safe[b], axis=1)
                for b in range(batch)
            ], axis=0)
        scores = mx.sum(
            query[:, :, start:end, None, :].astype(mx.float32)
            * gathered_keys.astype(mx.float32), axis=-1) * scale
        scores = mx.where(
            (selected >= 0)[:, None, :, :], scores, float("-inf"))
        probabilities = mx.softmax(scores, axis=-1).astype(
            gathered_values.dtype)
        output = mx.sum(
            probabilities[..., None] * gathered_values, axis=-2)
        mx.eval(output)
        outputs.append(output)
    result = (
        outputs[0] if len(outputs) == 1
        else mx.concatenate(outputs, axis=2))
    mx.eval(result)
    return result


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

    expanded_prefill = getattr(kv, "_glm53_expanded_prefill", None)
    absorbed_sparse = bool(
        getattr(kv, "glm53_sparse_absorbed_mla", False)
        and selection is not None)
    if absorbed_sparse and expanded_prefill is not None:
        # Dense attention through the first released top-k boundary retains
        # the ordinary expanded formulation exactly. Once DSA selection is
        # authoritative, absorbed MLA consumes compact latent rows and this
        # layer's expanded prefix is dead; release it before the long-context
        # allocation peak rather than building/appending unused K/V.
        expanded_prefill.pop(layer, None)
        expanded_prefill = None
    expanded_keys = expanded_values = None
    expanded_key_scales = expanded_value_scales = None
    host_expanded_kv = False
    if expanded_prefill is not None:
        # A stable-boundary/hot-prefix continuation begins with an already
        # populated compact latent cache but a new request-local expanded
        # cache. Seed it from the complete exact prefix once; later tiles append
        # only their new rows. Without this arm, suffix attention would expose
        # only the new K/V rows while constructing a mask for the full history.
        expanded_input = (
            latent_all
            if layer not in expanded_prefill and int(offset) > 0
            else latent
        )
        host_expanded_kv = bool(getattr(
            expanded_prefill, "host_spool", False))
        if host_expanded_kv and getattr(
                kv, "glm53_sparse_fused_attention", False):
            raise ValueError(
                "GLM-5.3 host-expanded K/V is incompatible with fused "
                "sparse attention")
        if getattr(kv, "glm53_sparse_fused_kv_int8", False):
            (expanded_keys, expanded_values,
             expanded_key_scales,
             expanded_value_scales) = (
                _glm5_next_update_expanded_prefill_kv_int8(
                    expanded_input, w, prefix, layer=layer,
                    cache=expanded_prefill, heads=heads,
                    key_dim=key_dim, value_dim=value_dim))
        elif host_expanded_kv and selection is not None:
            expanded_keys, expanded_values = (
                _glm5_next_update_host_expanded_prefill_kv(
                    expanded_input, w, prefix, layer=layer,
                    cache=expanded_prefill, heads=heads,
                    key_dim=key_dim, value_dim=value_dim))
        else:
            expanded_keys, expanded_values = (
                _glm5_next_update_expanded_prefill_kv(
                    expanded_input, w, prefix, layer=layer,
                    cache=expanded_prefill, heads=heads,
                    key_dim=key_dim, value_dim=value_dim))

    if selection is None:
        if expanded_keys is not None:
            keys, values = expanded_keys, expanded_values
            if isinstance(keys, np.ndarray):
                keys = _glm5_next_restore_host_bf16(
                    keys, cache=expanded_prefill)
                values = _glm5_next_restore_host_bf16(
                    values, cache=expanded_prefill)
            if expanded_key_scales is not None:
                keys = keys.astype(query.dtype) * expanded_key_scales.astype(
                    query.dtype)
                values = (
                    values.astype(query.dtype)
                    * expanded_value_scales.astype(query.dtype))
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
            expanded = _linear(
                latent_all, w, f"{prefix}.self_attn.kv_b_proj").reshape(
                    batch, key_length, heads,
                    key_dim + value_dim).transpose(0, 2, 1, 3)
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
        if absorbed_sparse:
            output = _glm5_next_sparse_absorbed_mla_attention(
                query, latent_all, selection, w, prefix,
                heads=heads, key_dim=key_dim, value_dim=value_dim,
                query_tile_size=32)
        elif (expanded_keys is not None
              and getattr(kv, "glm53_sparse_fused_attention", False)):
            from .glm5_next_sparse_fused import (
                glm5_next_sparse_fused_attention,
                glm5_next_sparse_fused_attention_int8,
            )

            if expanded_key_scales is not None:
                output = glm5_next_sparse_fused_attention_int8(
                    query, expanded_keys, expanded_values,
                    expanded_key_scales, expanded_value_scales,
                    selection, key_dim=key_dim)
            else:
                output = glm5_next_sparse_fused_attention(
                    query, expanded_keys, expanded_values, selection,
                    key_dim=key_dim)
            kv._glm53_sparse_fused_calls = int(getattr(
                kv, "_glm53_sparse_fused_calls", 0)) + 1
            kv._glm53_sparse_fused_positions = int(getattr(
                kv, "_glm53_sparse_fused_positions", 0)) + int(length)
            kv._glm53_sparse_fused_selected_rows = int(getattr(
                kv, "_glm53_sparse_fused_selected_rows", 0)) + (
                    int(length) * int(selection.shape[-1]))
        elif expanded_keys is not None:
            if expanded_key_scales is not None:
                raise ValueError(
                    "GLM-5.3 int8 expanded K/V requires fused sparse attention")
            output = _glm5_next_sparse_expanded_attention(
                query, expanded_keys, expanded_values, selection,
                key_dim=key_dim, query_tile_size=4,
                host_stats=expanded_prefill)
        else:
            output = _glm5_next_sparse_mla_attention(
                query, latent_all, selection, w, prefix,
                heads=heads, key_dim=key_dim, value_dim=value_dim,
                query_tile_size=4)

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
        compiled_kda_prefill: bool = False,
        compiled_kda_prefill_segment: int = 32,
        profile=None) -> mx.array:
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
                compiled_prefill_segment=compiled_kda_prefill_segment,
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
