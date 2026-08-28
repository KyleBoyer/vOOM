"""Released Qwen3.8 Flash-Next (Qwen4-Exp) text-trunk math.

This is a streaming-runtime adaptation of the public Qwen/Transformers and
MLX-VLM implementations.  It preserves the four hyper-connection streams,
PLE recurrence, QSA block selection, sigmoid-gated DeltaNet, and released MoE
routing while allowing vOOM to page weights and PLE rows independently.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import numpy as np

from . import quant
from .config import ModelConfig
from .layer_runner import _linear, _swiglu
from .qwen35 import (
    _apply_partial_rope,
    _cache_local_causal_mask,
    _gated_delta_net,
    _moe,
    _route_experts,
)
from .glm import _group_routes
from .qwen4_exp_ple_rows import Qwen4ExpPLERowStore
from .qwen4_exp_state import Qwen4ExpStateCache
from .lm_head_stream import StreamedLMHead


def qwen4_rms_norm(
    x: mx.array, weight: mx.array, eps: float, *, group_size: int | None = None,
) -> mx.array:
    """Zero-centered Qwen4 RMSNorm, optionally normalized per HC stream."""
    source_dtype = x.dtype
    value = x.astype(mx.float32)
    if group_size is not None:
        if group_size <= 0 or x.shape[-1] % group_size:
            raise ValueError("Qwen4 grouped RMSNorm has invalid stream width")
        value = value.reshape(*value.shape[:-1], -1, group_size)
        scale = weight.reshape(-1, group_size).astype(mx.float32)
    else:
        scale = weight.astype(mx.float32)
    value = value * mx.rsqrt(
        mx.mean(value * value, axis=-1, keepdims=True) + eps)
    value = value * (1.0 + scale)
    return value.reshape(x.shape).astype(source_dtype)


def hyper_connection_mix(
    hidden: mx.array,
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    *,
    inject: bool = True,
):
    """Mix four normalized streams and optionally return branch gates."""
    count = int(cfg.qwen4_hc_count)
    expected = count * int(cfg.hidden_size)
    if hidden.shape[-1] != expected:
        raise ValueError(
            f"Qwen4 hyper input width {hidden.shape[-1]} != {expected}")
    normalized = qwen4_rms_norm(
        hidden, weights[f"{prefix}.hc_norm.weight"], cfg.rms_norm_eps,
        group_size=cfg.hidden_size)
    down = _linear(normalized, weights, f"{prefix}.input_mix_weight_down")
    # PyTorch keeps BF16 for tensor/scalar division under the released eager
    # path. MLX promotes BF16 / Python int to FP32; without the cast every
    # injection promotes the four-stream residual to FP32, doubling long-
    # context memory and changing the served target after layer zero.
    down = (down / count).astype(hidden.dtype)
    mixed_gate = mx.sigmoid(_linear(
        down * mx.sigmoid(down),
        weights, f"{prefix}.input_mix_weight_up"))
    mixed_gate = mixed_gate.reshape(
        *hidden.shape[:-1], count, cfg.hidden_size)
    streams = normalized.reshape(
        *hidden.shape[:-1], count, cfg.hidden_size)
    mixed = mx.mean(mixed_gate * streams, axis=-2)
    if not inject:
        return mixed
    injection_input = (_linear(
        normalized, weights,
        f"{prefix}.block_inject_weight") / count).astype(hidden.dtype)
    injection = (2 * mx.sigmoid(injection_input)).astype(hidden.dtype)
    return mixed, hidden, injection


def hyper_connection_inject(
    branch: mx.array, hyper_input: mx.array, injection: mx.array,
) -> mx.array:
    if injection.shape[:-1] != branch.shape[:-1]:
        raise ValueError("Qwen4 hyper injection position shape mismatch")
    if branch.dtype != hyper_input.dtype or injection.dtype != hyper_input.dtype:
        raise TypeError(
            "Qwen4 released hyper injection requires one activation dtype: "
            f"branch={branch.dtype}, residual={hyper_input.dtype}, "
            f"injection={injection.dtype}")
    update = branch[..., None, :] * injection[..., None]
    return (hyper_input + update.reshape(hyper_input.shape)).astype(
        hyper_input.dtype)


def _dilated_causal_depthwise_conv1d(
    x: mx.array,
    weight: mx.array,
    history: mx.array | None,
    *,
    kernel_size: int,
    dilation: int,
) -> tuple[mx.array, mx.array]:
    """PyTorch depthwise Conv1d correlation with causal left state."""
    batch, length, channels = x.shape
    state_len = (kernel_size - 1) * dilation
    if history is None:
        history = mx.zeros((batch, state_len, channels), dtype=x.dtype)
    if tuple(history.shape) != (batch, state_len, channels):
        raise ValueError("Qwen4 PLE convolution history shape mismatch")
    padded = mx.concatenate([history, x], axis=1)
    taps = weight.reshape(channels, kernel_size)
    output = mx.zeros((batch, length, channels), dtype=mx.float32)
    for tap in range(kernel_size):
        start = tap * dilation
        output = output + (
            padded[:, start:start + length].astype(mx.float32)
            * taps[:, tap].astype(mx.float32))
    new_history = (
        padded[:, -state_len:] if state_len
        else mx.zeros((batch, 0, channels), dtype=x.dtype))
    return (output * mx.sigmoid(output)).astype(x.dtype), new_history


def apply_ple(
    hidden: mx.array,
    input_ids: Sequence[int],
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    layer: int,
    row_store: Qwen4ExpPLERowStore,
    state: Qwen4ExpStateCache | None,
) -> mx.array:
    """Add exact direct-paged PLE output to the four-stream hidden state."""
    if hidden.shape[0] != 1:
        raise ValueError("initial Qwen4 PLE pager supports batch size one")
    tokens = tuple(int(token) for token in input_ids)
    if len(tokens) != hidden.shape[1]:
        raise ValueError("Qwen4 PLE token/hidden length mismatch")
    previous = state.ple_context[layer] if state is not None else None
    row_ids = row_store.layout.row_ids(tokens, previous_context=previous)
    storage = row_store.read_rows(row_ids)
    # numpy uint16 carries the exact BF16 bit pattern; view changes only dtype.
    embeddings = mx.array(storage).view(mx.bfloat16).reshape(
        1, len(tokens), cfg.qwen4_ple_embed_dim)
    count = cfg.qwen4_hc_count
    keys = qwen4_rms_norm(
        _linear(embeddings, weights, f"{prefix}.key_proj"),
        weights[f"{prefix}.norm_key.weight"], cfg.rms_norm_eps,
        group_size=cfg.hidden_size).reshape(
            1, len(tokens), count, cfg.hidden_size)
    values = _linear(embeddings, weights, f"{prefix}.value_proj")
    queries = qwen4_rms_norm(
        hidden, weights[f"{prefix}.norm_query.weight"], cfg.rms_norm_eps,
        group_size=cfg.hidden_size).reshape(
            1, len(tokens), count, cfg.hidden_size)
    source_dtype = hidden.dtype
    gate = mx.sum(keys * queries, axis=-1, keepdims=True)
    gate = (gate / math.sqrt(cfg.hidden_size)).astype(source_dtype)
    gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
    gate = gate.astype(source_dtype)
    gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(
        hidden.shape).astype(source_dtype)
    normalized = qwen4_rms_norm(
        gated, weights[f"{prefix}.norm_conv.weight"], cfg.rms_norm_eps,
        group_size=cfg.hidden_size)
    history = state.ple_conv[layer] if state is not None else None
    convolved, new_history = _dilated_causal_depthwise_conv1d(
        normalized, weights[f"{prefix}.conv1d.weight"], history,
        kernel_size=cfg.qwen4_ple_conv_kernel_size,
        dilation=cfg.qwen4_ngram_size)
    if state is not None:
        state.ple_conv[layer] = new_history
        context_len = row_store.layout.context_len
        combined = (previous or (row_store.layout.eos_token_id,) * context_len) + tokens
        state.ple_context[layer] = combined[-context_len:]
        state.ple_lengths[layer] += len(tokens)
    return (gated + convolved).astype(source_dtype)


def _apply_rope_positions(
    x: mx.array, positions: mx.array, cfg: ModelConfig,
) -> mx.array:
    """Apply Qwen4 partial text RoPE to arbitrary (including pooled) positions."""
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError("invalid Qwen4 partial rotary width")
    dims = mx.arange(0, rotary_dim, 2, dtype=mx.float32)
    inv = 1.0 / (cfg.rope_theta ** (dims / rotary_dim))
    position_values = positions.astype(mx.float32)
    frequencies = position_values[..., None] * inv
    embedding = mx.concatenate([frequencies, frequencies], axis=-1)
    cosine = mx.cos(embedding).astype(x.dtype)[:, None]
    sine = mx.sin(embedding).astype(x.dtype)[:, None]
    rotated = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = mx.concatenate(
        [-rotated[..., half:], rotated[..., :half]], axis=-1)
    value = rotated * cosine + rotated_half * sine
    return mx.concatenate([value, x[..., rotary_dim:]], axis=-1)


def _qsa_selection_mask(
    hidden: mx.array,
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    layer: int,
    offset: int,
    state: Qwen4ExpStateCache,
) -> mx.array | None:
    batch, length, _ = hidden.shape
    heads = cfg.qwen4_indexer_n_heads
    kv_heads = cfg.qwen4_indexer_kv_heads
    dim = cfg.qwen4_indexer_head_dim
    ratio = cfg.qwen4_indexer_compress_ratio
    block_topk = cfg.qwen4_indexer_budget // ratio
    if block_topk <= 0:
        raise ValueError("Qwen4 QSA budget must cover a compressed block")
    projected = _linear(
        hidden, weights, f"{prefix}.self_attn.indexer.index_qk_proj")
    projected = projected.reshape(batch, length, heads + kv_heads, dim)
    query = projected[:, :, :heads]
    raw_keys = projected[:, :, heads:]
    if kv_heads != 1:
        raise ValueError("initial Qwen4 QSA path requires one index KV head")
    raw_keys = raw_keys.squeeze(2)
    query = qwen4_rms_norm(
        query,
        weights[f"{prefix}.self_attn.indexer.q_layernorm.weight"],
        cfg.rms_norm_eps).transpose(0, 2, 1, 3)
    positions = mx.broadcast_to(
        mx.arange(offset, offset + length, dtype=mx.int32)[None],
        (batch, length))
    raw_keys, full_positions = state.update_qsa(
        layer, raw_keys, positions)
    key_len = int(raw_keys.shape[1])
    complete_blocks = key_len // ratio
    if complete_blocks <= block_topk:
        return None
    query = _apply_rope_positions(query, positions, cfg)
    complete_key_len = complete_blocks * ratio
    pooled = raw_keys[:, :complete_key_len].reshape(
        batch, complete_blocks, ratio, dim)
    pooled = mx.mean(pooled.astype(mx.float32), axis=2).astype(raw_keys.dtype)
    pooled = qwen4_rms_norm(
        pooled,
        weights[f"{prefix}.self_attn.indexer.k_layernorm.weight"],
        cfg.rms_norm_eps)[:, None]
    starts = mx.arange(complete_blocks, dtype=mx.int32) * ratio
    block_positions = full_positions[..., starts]
    pooled = _apply_rope_positions(pooled, block_positions, cfg)
    scores = query.astype(mx.float32) @ pooled.astype(mx.float32).transpose(
        0, 1, 3, 2)
    scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(dim)
    query_ends = positions + 1
    complete_counts = query_ends // ratio
    valid = (
        mx.arange(complete_blocks)[None, None, :]
        < complete_counts[..., None])
    scores = mx.where(valid, scores, -mx.inf)
    selected = mx.argpartition(
        scores, kth=-block_topk, axis=-1)[..., -block_topk:]
    hits = mx.put_along_axis(
        mx.zeros((batch, length, complete_blocks), dtype=mx.bool_),
        selected, mx.array(True), axis=-1)
    selected_tokens = mx.repeat(hits, ratio, axis=-1)
    if complete_key_len < key_len:
        selected_tokens = mx.concatenate([
            selected_tokens,
            mx.zeros(
                (batch, length, key_len - complete_key_len), dtype=mx.bool_),
        ], axis=-1)
    token_positions = mx.arange(key_len)[None, None]
    tail_starts = complete_counts * ratio
    tail = (
        (token_positions >= tail_starts[..., None])
        & (token_positions < query_ends[..., None]))
    causal = token_positions < query_ends[..., None]
    selected_tokens = mx.where(
        (complete_counts > block_topk)[..., None],
        selected_tokens | tail,
        causal)
    return selected_tokens[:, None]


def _qsa_attention(
    hidden: mx.array,
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    kv,
    layer: int,
    offset: int,
    state: Qwen4ExpStateCache,
) -> mx.array:
    batch, length, _ = hidden.shape
    heads, kv_heads, dim = (
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    qsa_mask = _qsa_selection_mask(
        hidden, weights, prefix, cfg, layer, offset, state)
    projected = _linear(hidden, weights, f"{prefix}.self_attn.q_proj")
    projected = projected.reshape(batch, length, heads, 2 * dim)
    query = projected[..., :dim]
    output_gate = projected[..., dim:].reshape(batch, length, heads * dim)
    key = _linear(hidden, weights, f"{prefix}.self_attn.k_proj").reshape(
        batch, length, kv_heads, dim)
    value = _linear(hidden, weights, f"{prefix}.self_attn.v_proj").reshape(
        batch, length, kv_heads, dim)
    query = qwen4_rms_norm(
        query, weights[f"{prefix}.self_attn.q_norm.weight"],
        cfg.rms_norm_eps).transpose(0, 2, 1, 3)
    key = qwen4_rms_norm(
        key, weights[f"{prefix}.self_attn.k_norm.weight"],
        cfg.rms_norm_eps).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    query, key = _apply_partial_rope(query, key, offset, cfg)
    keys, values = kv.update(layer, key, value)
    if qsa_mask is None:
        mask = (
            _cache_local_causal_mask(length, int(keys.shape[2]), query.dtype)
            if length > 1 else None)
    else:
        mask = mx.where(qsa_mask, 0.0, -mx.inf).astype(query.dtype)
    attended = mx.fast.scaled_dot_product_attention(
        query, keys, values, scale=dim ** -0.5, mask=mask)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch, length, heads * dim)
    attended = attended * mx.sigmoid(output_gate)
    return _linear(attended, weights, f"{prefix}.self_attn.o_proj")


def qwen4_attention_branch(
    hidden: mx.array,
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    kv,
    layer: int,
    offset: int,
    *,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
) -> mx.array:
    layer_type = cfg.layer_types[layer]
    if layer_type == "linear_attention":
        return _gated_delta_net(
            hidden, weights, prefix, cfg,
            getattr(kv, "kda_cache", None), layer,
            compiled_delta_prefill=compiled_delta_prefill,
            native_fused_delta_prefill=native_fused_delta_prefill)
    if layer_type == "full_attention":
        state = getattr(kv, "qwen4_cache", None)
        if not isinstance(state, Qwen4ExpStateCache):
            raise ValueError("Qwen4 QSA layer is missing auxiliary state")
        return _qsa_attention(
            hidden, weights, prefix, cfg, kv, layer, offset, state)
    raise ValueError(f"unsupported Qwen4 layer type {layer_type!r}")


def qwen4_attention_residual(
    hidden: mx.array,
    input_ids: Sequence[int],
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    kv,
    layer: int,
    offset: int,
    *,
    row_store: Qwen4ExpPLERowStore | None = None,
    profile=None,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
) -> mx.array:
    """PLE plus attention half of one released Qwen4 decoder block.

    Keeping this boundary explicit lets an exact speculative verifier advance
    every recurrent state one position at a time, route each resulting row,
    and then reuse the union of immutable expert weight pages.  No activation
    rows are batched through hidden-width reductions here.
    """
    positions = int(hidden.shape[1])
    if layer in cfg.qwen4_ple_layers:
        if row_store is None:
            raise ValueError("Qwen4 PLE layer is missing its direct-row store")
        ple_started = profile.start_substep() if profile is not None else None
        hidden = hidden + apply_ple(
            hidden, input_ids, weights, f"{prefix}.ple", cfg, layer,
            row_store, getattr(kv, "qwen4_cache", None))
        if profile is not None:
            profile.finish_substep(
                "ple", layer, ple_started, hidden, positions=positions)

    attention_started = profile.start_substep() if profile is not None else None
    mixed, hyper_input, injection = hyper_connection_mix(
        hidden, weights, f"{prefix}.attn_hyper_connection", cfg)
    branch = qwen4_attention_branch(
        mixed, weights, prefix, cfg, kv, layer, offset,
        compiled_delta_prefill=compiled_delta_prefill,
        native_fused_delta_prefill=native_fused_delta_prefill)
    hidden = hyper_connection_inject(branch, hyper_input, injection)
    if profile is not None:
        profile.finish_substep(
            "attention", layer, attention_started, hidden,
            positions=positions)
    return hidden


def qwen4_mlp_route(
    hidden: mx.array,
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    layer: int,
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    dict[int, list[tuple[int, float]]],
]:
    """Return the exact per-row MoE route and HC injection operands."""
    mixed, hyper_input, injection = hyper_connection_mix(
        hidden, weights, f"{prefix}.mlp_hyper_connection", cfg)
    indices, scores = _route_experts(mixed, weights, prefix, cfg, layer)
    mx.eval(indices, scores)
    return mixed, hyper_input, injection, _group_routes(indices, scores)


def qwen4_mlp_from_groups(
    route: tuple[
        mx.array,
        mx.array,
        mx.array,
        dict[int, list[tuple[int, float]]],
    ],
    experts: dict[int, dict],
    weights: dict,
    prefix: str,
) -> mx.array:
    """Evaluate one routed row from already-fetched exact expert pages.

    The accumulation order and materialization boundary mirror ``_moe``:
    ascending expert id, BF16 route weights, one routed-output evaluation,
    then the shared expert and hyper-connection injection.  Only storage
    lifetime changes; each verifier position retains ordinary decode math.
    """
    mixed, hyper_input, injection, groups = route
    routed = mx.zeros_like(mixed)
    for expert in sorted(groups):
        if expert not in experts:
            raise ValueError(f"missing routed Qwen4 expert {expert}")
        plist = groups[expert]
        positions = [position for position, _ in plist]
        route_weights = mx.array(
            [weight for _, weight in plist], dtype=mixed.dtype)
        contribution = _swiglu(
            mixed[:, positions, :],
            experts[expert],
            f"{prefix}.mlp.experts.{expert}",
        )
        routed = routed.at[:, positions, :].add(
            contribution * route_weights[None, :, None])
    mx.eval(routed)
    shared = _swiglu(mixed, weights, f"{prefix}.mlp.shared_expert")
    shared_gate = mx.sigmoid(_linear(
        mixed, weights, f"{prefix}.mlp.shared_expert_gate"))
    branch = routed + shared_gate * shared
    return hyper_connection_inject(branch, hyper_input, injection)


def qwen4_mlp_from_group_batches(
    routes: Sequence[tuple[
        mx.array,
        mx.array,
        mx.array,
        dict[int, list[tuple[int, float]]],
    ]],
    expert_batches,
    weights: dict,
    prefix: str,
) -> list[mx.array]:
    """Evaluate verifier rows while releasing each exact expert batch.

    Expert IDs are consumed in ascending order, so every row retains the same
    routed accumulation order as :func:`qwen4_mlp_from_groups`.  The explicit
    batch barrier is the lifetime boundary that permits the next immutable
    expert-page read to overlap current Metal compute without retaining the
    complete routed union.  This path is selected only by the existing
    explicit expert-batch-prefetch option and is bitwise-gated against the
    ordinary whole-union implementation.
    """
    routed = [mx.zeros_like(route[0]) for route in routes]
    previous_expert = -1
    for expert_ids, experts in expert_batches:
        ordered = tuple(int(expert) for expert in expert_ids)
        if tuple(sorted(ordered)) != ordered:
            raise ValueError("Qwen4 verifier expert batches must be sorted")
        if ordered and ordered[0] <= previous_expert:
            raise ValueError(
                "Qwen4 verifier expert batches must be globally ordered")
        for expert in ordered:
            if expert not in experts:
                raise ValueError(f"missing routed Qwen4 expert {expert}")
            for route_index, route in enumerate(routes):
                mixed, _hyper_input, _injection, groups = route
                plist = groups.get(expert)
                if not plist:
                    continue
                positions = [position for position, _ in plist]
                route_weights = mx.array(
                    [weight for _, weight in plist], dtype=mixed.dtype)
                contribution = _swiglu(
                    mixed[:, positions, :],
                    experts[expert],
                    f"{prefix}.mlp.experts.{expert}",
                )
                routed[route_index] = routed[route_index].at[
                    :, positions, :
                ].add(contribution * route_weights[None, :, None])
            previous_expert = expert
        # This is both the proof boundary and the ownership boundary: after it
        # returns, no lazy graph retains the current batch's weight arrays.
        mx.eval(*routed)
        del experts

    outputs = []
    for route, routed_output in zip(routes, routed, strict=True):
        mixed, hyper_input, injection, _groups = route
        shared = _swiglu(mixed, weights, f"{prefix}.mlp.shared_expert")
        shared_gate = mx.sigmoid(_linear(
            mixed, weights, f"{prefix}.mlp.shared_expert_gate"))
        branch = routed_output + shared_gate * shared
        outputs.append(hyper_connection_inject(
            branch, hyper_input, injection))
    return outputs


def run_qwen4_block(
    hidden: mx.array,
    input_ids: Sequence[int],
    weights: dict,
    prefix: str,
    cfg: ModelConfig,
    kv,
    layer: int,
    offset: int,
    get_experts,
    *,
    row_store: Qwen4ExpPLERowStore | None = None,
    iter_expert_batches=None,
    profile=None,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
) -> mx.array:
    positions = int(hidden.shape[1])
    hidden = qwen4_attention_residual(
        hidden, input_ids, weights, prefix, cfg, kv, layer, offset,
        row_store=row_store,
        profile=profile,
        compiled_delta_prefill=compiled_delta_prefill,
        native_fused_delta_prefill=native_fused_delta_prefill)

    mlp_started = profile.start_substep() if profile is not None else None
    mixed, hyper_input, injection = hyper_connection_mix(
        hidden, weights, f"{prefix}.mlp_hyper_connection", cfg)
    branch = _moe(
        mixed, weights, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
    hidden = hyper_connection_inject(branch, hyper_input, injection)
    if profile is not None:
        profile.finish_substep(
            "mlp", layer, mlp_started, hidden, positions=positions)
    return hidden


def final_hidden(
    hidden: mx.array, weights: dict, cfg: ModelConfig,
) -> mx.array:
    return hyper_connection_mix(
        hidden, weights, "model.hyper_connection_mixer", cfg, inject=False)


def final_logits(
    hidden: mx.array, mixer_weights: dict, head, cfg: ModelConfig,
) -> mx.array:
    mixed = final_hidden(hidden[:, -1:], mixer_weights, cfg)
    if isinstance(head, StreamedLMHead):
        return head.logits(mixed)[0, 0]
    return quant.matmul(mixed, head)[0, 0]


def all_logits(
    hidden: mx.array, mixer_weights: dict, head, cfg: ModelConfig,
) -> mx.array:
    mixed = final_hidden(hidden, mixer_weights, cfg)
    if isinstance(head, StreamedLMHead):
        return head.logits(mixed)[0]
    return quant.matmul(mixed, head)[0]
