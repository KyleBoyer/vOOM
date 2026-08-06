"""LFM2 (Liquid Foundation Model 2) decoder blocks.

LFM2.5-2.6B is a 30-layer hybrid: 22 gated short-convolution layers and 8
full-attention layers, selected by ``config.json``'s ``layer_types``. Every
layer shares the same residual shape::

    h   = x + operator(operator_norm(x))       # operator = short conv | attention
    out = h + feed_forward(ffn_norm(h))        # SwiGLU

The short-convolution layers carry a fixed ``conv_L_cache - 1`` position
history instead of a growing KV cache, so their per-layer state is bounded and
independent of context length -- structurally the same situation as Kimi's
causal convolution, whose ``KDAStateCache.conv_history`` slot this reuses
rather than introducing a second state container. That reuse is what gives
LFM2 fork/restore, disk spill, and therefore suffix-decoding rollback for free.

Ported against mlx-lm's ``models/lfm2.py`` and gated by a numerical oracle
(``tests/test_f202_lfm2_oracle.py``) rather than by inspection.
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig


def layer_is_attention(cfg: ModelConfig, layer: int) -> bool:
    """True when ``layer`` is a full-attention layer, per released layer_types."""
    types = getattr(cfg, "layer_types", None) or ()
    if layer < len(types):
        return str(types[layer]) == "full_attention"
    return False


def _rms_norm(x: mx.array, weight, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps) if weight is not None else x


def _lfm2_attention(x: mx.array, w: dict, prefix: str, cfg: ModelConfig,
                    kv, layer: int, offset: int) -> mx.array:
    """One full-attention layer: GQA with per-head q/k RMSNorm, then RoPE.

    The head-dim norms are applied *before* the transpose to (B, H, L, D) and
    before RoPE, matching the released module order; normalizing after RoPE
    would rescale the rotated pair and is not the same function.
    """
    batch, length, _ = x.shape
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads or heads
    head_dim = cfg.hidden_size // heads

    queries = x @ w[f"{prefix}.self_attn.q_proj.weight"].T
    keys = x @ w[f"{prefix}.self_attn.k_proj.weight"].T
    values = x @ w[f"{prefix}.self_attn.v_proj.weight"].T

    queries = _rms_norm(
        queries.reshape(batch, length, heads, head_dim),
        w.get(f"{prefix}.self_attn.q_layernorm.weight"), cfg.rms_norm_eps,
    ).transpose(0, 2, 1, 3)
    keys = _rms_norm(
        keys.reshape(batch, length, kv_heads, head_dim),
        w.get(f"{prefix}.self_attn.k_layernorm.weight"), cfg.rms_norm_eps,
    ).transpose(0, 2, 1, 3)
    values = values.reshape(
        batch, length, kv_heads, head_dim).transpose(0, 2, 1, 3)

    queries = mx.fast.rope(queries, head_dim, traditional=False,
                           base=cfg.rope_theta, scale=1.0, offset=offset)
    keys = mx.fast.rope(keys, head_dim, traditional=False,
                        base=cfg.rope_theta, scale=1.0, offset=offset)

    keys, values = kv.update(layer, keys, values)
    mask = "causal" if length > 1 else None
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=head_dim ** -0.5, mask=mask)
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return output @ w[f"{prefix}.self_attn.out_proj.weight"].T


def _lfm2_short_conv(x: mx.array, w: dict, prefix: str, cfg: ModelConfig,
                     state_cache, layer: int) -> mx.array:
    """One gated short-convolution layer.

    ``in_proj`` produces three equal parts (B, C, x); the convolution runs over
    ``B * x`` and its output is gated by ``C``. The depthwise kernel is applied
    as an explicit shifted sum rather than through ``nn.Conv1d`` so the
    ``conv_L_cache - 1`` carry-over is the same array in prefill and decode --
    a Conv1d module would need its own padding convention on each path, and the
    two are easy to get subtly out of step.

    Released weight layout is ``(channels, 1, kernel)``. Position ``t`` reads
    inputs ``t-K+1 .. t`` with ``weight[:, K-1]`` multiplying ``x[t]``; this is
    cross-correlation, matching MLX's Conv1d, not a flipped convolution.
    """
    projected = x @ w[f"{prefix}.conv.in_proj.weight"].T
    gate_b, gate_c, value = mx.split(projected, 3, axis=-1)
    bx = gate_b * value

    kernel = w[f"{prefix}.conv.conv.weight"]
    if kernel.ndim == 3:
        # (channels, 1, kernel) -> (channels, kernel)
        kernel = kernel.reshape(kernel.shape[0], -1)
    taps = kernel.shape[-1]
    keep = taps - 1

    if state_cache is None:
        # Without the companion cache every decode step would restart from a
        # zero history, which does not raise -- it silently returns fluent
        # nonsense. Observed on the paged-KV path, whose cache object carries
        # no ``kda_cache``: output degenerated to a repeated token pair while
        # every counter still looked healthy. Refuse instead.
        raise ValueError(
            f"LFM2 short-conv layer {layer} has no conv-state cache; the "
            "active KV layout does not carry the required companion state "
            "(paged KV is not yet supported for hybrid conv layers)")
    history = state_cache.conv_history(layer)
    if history is None:
        state = mx.zeros((bx.shape[0], keep, bx.shape[2]), dtype=bx.dtype)
    else:
        state = history[0]
    padded = mx.concatenate([state, bx], axis=1)

    conv_out = None
    for tap in range(taps):
        window = padded[:, tap:tap + bx.shape[1], :]
        term = window * kernel[:, tap]
        conv_out = term if conv_out is None else conv_out + term

    if state_cache is not None:
        state_cache.set_conv_history(layer, (padded[:, -keep:, :],))

    gated = gate_c * conv_out
    return gated @ w[f"{prefix}.conv.out_proj.weight"].T


def _lfm2_mlp(x: mx.array, w: dict, prefix: str) -> mx.array:
    """SwiGLU feed-forward: ``w2(silu(w1(x)) * w3(x))``."""
    gate = x @ w[f"{prefix}.feed_forward.w1.weight"].T
    up = x @ w[f"{prefix}.feed_forward.w3.weight"].T
    return (mx.sigmoid(gate) * gate * up) @ w[
        f"{prefix}.feed_forward.w2.weight"].T


def _lfm2_operator_residual(x: mx.array, w: dict, prefix: str,
                            cfg: ModelConfig, kv, layer: int, offset: int,
                            state_cache) -> mx.array:
    """``x + operator(operator_norm(x))`` -- short conv or attention.

    Split from the FFN half so the serial verifier can drive one position at a
    time and snapshot the recurrent endpoint between the two, exactly as the
    qwen/kimi families do.
    """
    normed = _rms_norm(x, w.get(f"{prefix}.operator_norm.weight"),
                       cfg.rms_norm_eps)
    if layer_is_attention(cfg, layer):
        return x + _lfm2_attention(normed, w, prefix, cfg, kv, layer, offset)
    return x + _lfm2_short_conv(normed, w, prefix, cfg, state_cache, layer)


def _lfm2_mlp_residual(x: mx.array, w: dict, prefix: str, cfg: ModelConfig,
                       mlp_last_only: bool = False) -> mx.array:
    """``x + feed_forward(ffn_norm(x))``, optionally for the last row only."""
    if mlp_last_only and x.shape[1] > 1:
        # Only the final position's logits are consumed and the FFN is
        # position-independent, so earlier rows cannot affect the output.
        x = x[:, -1:, :]
    return x + _lfm2_mlp(
        _rms_norm(x, w.get(f"{prefix}.ffn_norm.weight"), cfg.rms_norm_eps),
        w, prefix)


def run_lfm2_block(x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
                   layer: int, offset: int, state_cache=None,
                   mlp_last_only: bool = False, profile=None) -> mx.array:
    """One LFM2 decoder block, conv or attention depending on layer_types."""
    positions = int(x.shape[1])
    operator_t0 = profile.start_substep() if profile is not None else None
    x = _lfm2_operator_residual(x, w, prefix, cfg, kv, layer, offset,
                                state_cache)
    if profile is not None:
        profile.finish_substep(
            "attention" if layer_is_attention(cfg, layer) else "short_conv",
            layer, operator_t0, x, positions=positions)

    mlp_t0 = profile.start_substep() if profile is not None else None
    x = _lfm2_mlp_residual(x, w, prefix, cfg, mlp_last_only=mlp_last_only)
    if profile is not None:
        profile.finish_substep("mlp", layer, mlp_t0, x, positions=positions)
    return x
