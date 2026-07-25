"""Jet-Nemotron (jet-ai/Jet-Nemotron-4B, Jet-Nemotron-2B): a Qwen2.5-vocab
dense LM whose ``layer_types`` mix three block kinds per layer -- "jet"
(JetBlock, a gated-delta-rule linear-attention variant with an added
per-token DYNAMICALLY-GENERATED depthwise convolution on V), "swa" (ordinary
sliding-window attention), and "attn" (ordinary full attention). See
docs/future_lossless_techniques.md for the architecture citation and port
scoping.

JetBlock's core recurrence is the SAME gated-delta-rule math already
oracle-verified in this codebase for Qwen3.5's DeltaNet
(``runtime/qwen35.py::_gated_delta_net``) -- confirmed by direct comparison
of both real HF sources: ``decay = -exp(A_log) * softplus(a_proj(h) +
dt_bias)``, sigmoid beta gate, L2-normalized Q/K inside the recurrence
(``use_qk_l2norm_in_kernel=True``), identical state-update/output formula.
Two real differences from Qwen3.5's version, confirmed against the real
``jet_block.py`` (NVIDIA, Apache-2.0, downloaded 2026-07-22):
  1. Q and K get an explicit SiLU BEFORE the L2-norm (Qwen3.5's version has
     no SiLU on Q/K at all, only L2-norm).
  2. The short causal conv is on V alone, with a kernel DYNAMICALLY
     GENERATED per position from the layer's raw input (a small
     w1->SiLU->w2 MLP), not a single static learned depthwise kernel shared
     across all positions and applied to the fused QKV projection the way
     Qwen3.5's ``_causal_depthwise_conv1d`` works.

The gated output norm (``FusedRMSNormGated``, real fla-org default
``activation="swish"``, confirmed via the real
fla/modules/fused_norm_gate.py source 2026-07-22:
``norm(x) * weight * (gate * sigmoid(gate))``) is exactly
``runtime/qwen35.py::_silu_gated_rms_norm``, reused directly here.

"swa"/"attn" layers are ordinary Qwen2-style GQA with **bias=True on
Q/K/V** (unlike Qwen3's bias-free convention), **no QK-norm at all**, and
standard (non-partial) rotate-half RoPE -- fully covered by
``layer_runner.run_block``/``_attention`` already; sliding-window layers
need only the same causal+window mask/KV-truncation gpt-oss's own
"sliding_attention" layer type already implements
(``runtime/gptoss.py``), not a new KV class.
"""

from __future__ import annotations

import mlx.core as mx

from . import layer_runner
from .config import ModelConfig
from .layer_runner import _attention, _linear, _swiglu


def _get_dynamic_conv_kernel(
    generator_input: mx.array, w: dict, prefix: str,
    value_dim: int, kernel_size: int,
) -> mx.array:
    """The per-position, per-channel depthwise conv kernel JetBlock's
    DynamicShortConvolution generates from the layer's RAW input (not from
    V) via a small w1->SiLU->w2 MLP. Real formula:
    ``dynamic_conv.py::DynamicShortConvolution.get_kernel``."""
    hidden = _linear(
        generator_input, w, f"{prefix}.self_attn.dynamic_conv1d.kernel_generator.w1")
    hidden = hidden * mx.sigmoid(hidden)  # SiLU
    flat = _linear(
        hidden, w, f"{prefix}.self_attn.dynamic_conv1d.kernel_generator.w2")
    batch, length, _ = generator_input.shape
    return flat.reshape(batch, length, value_dim, kernel_size)


def _dynamic_causal_conv1d(
    x: mx.array, kernels: mx.array, kernel_size: int,
    history: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Real formula: dynamic_conv.py::DynamicShortConvolution._forward_naive
    (no cache) / ``_step_naive`` (with cache) -- x: (B, T, D), kernels:
    (B, T, D, W) -> (B, T, D). Equivalent to ``unfold``-then-multiply-then-sum;
    implemented as explicit shifted taps (same style as this project's
    existing static ``_causal_depthwise_conv1d``/the F92 KDA oracle's
    ShortConvolution stub) since MLX has no unfold primitive.

    ``history``: (B, W-1, D) carried from a previous call (the real trailing
    V-projections of the last W-1 positions), or None (zero-padded) for a
    fresh conversation -- REQUIRED for correct multi-step decode: unlike a
    fixed shared kernel, each position's kernel is freshly generated, but
    the CONVOLUTION WINDOW itself still needs genuine prior V values, not
    zeros, once decode moves past the first token. (A single one-shot
    prefill call with history=None already matches this exactly since the
    real model's own first call also has no cache.) Returns
    (output, new_history) -- mirrors ``_causal_depthwise_conv1d``'s own
    history contract exactly.
    """
    batch, length, dim = x.shape
    if history is None:
        history = mx.zeros((batch, kernel_size - 1, dim), dtype=x.dtype)
    padded = mx.concatenate([history, x], axis=1)  # (B, L+W-1, D)
    taps = [padded[:, tap:tap + length, :] for tap in range(kernel_size)]
    stacked = mx.stack(taps, axis=-1)  # (B, T, D, W)
    out = (stacked * kernels).sum(axis=-1)
    new_history = (padded[:, length:, :] if kernel_size > 1
                   else mx.zeros((batch, 0, dim), dtype=x.dtype))
    return out, new_history


def _jet_block(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig,
    state_cache: "object | None", layer: int,
) -> mx.array:
    """JetBlock ("jet" layer type): gated-delta-rule linear attention with
    a dynamically-generated causal conv on V. h: (1, L, hidden)."""
    batch, length, _ = h.shape
    num_heads = cfg.jet_num_heads
    head_dim = cfg.jet_head_dim
    kernel_size = cfg.jet_conv_kernel_size
    key_dim = num_heads * head_dim
    head_v_dim = cfg.jet_head_v_dim
    value_dim = num_heads * head_v_dim

    q = _linear(h, w, f"{prefix}.self_attn.q_proj")
    k = _linear(h, w, f"{prefix}.self_attn.k_proj")
    q = q * mx.sigmoid(q)  # SiLU -- present here, absent in Qwen3.5's own DeltaNet
    k = k * mx.sigmoid(k)

    v_raw = _linear(h, w, f"{prefix}.self_attn.v_proj")
    kernels = _get_dynamic_conv_kernel(h, w, prefix, value_dim, kernel_size)
    conv_history = None
    if state_cache is not None:
        cached = state_cache.conv_history(layer)
        conv_history = cached[0] if cached is not None else None
    v, new_conv_history = _dynamic_causal_conv1d(
        v_raw, kernels, kernel_size, history=conv_history)
    v = v * mx.sigmoid(v)  # DynamicShortConvolution's own SiLU activation

    q = q.reshape(batch, length, num_heads, head_dim)
    k = k.reshape(batch, length, num_heads, head_dim)
    v = v.reshape(batch, length, num_heads, head_v_dim).astype(mx.float32)

    beta = mx.sigmoid(_linear(h, w, f"{prefix}.self_attn.b_proj").astype(mx.float32))
    a = _linear(h, w, f"{prefix}.self_attn.a_proj").astype(mx.float32)
    dt_bias = w[f"{prefix}.self_attn.dt_bias"].astype(mx.float32)
    softplus = mx.logaddexp(a + dt_bias.reshape(1, 1, num_heads), mx.zeros_like(a))
    decay = -mx.exp(w[f"{prefix}.self_attn.A_log"].astype(mx.float32)).reshape(
        1, 1, num_heads) * softplus

    def l2norm(value):
        value = value.astype(mx.float32)
        return value * mx.rsqrt(
            mx.sum(value * value, axis=-1, keepdims=True) + 1e-6)

    q = l2norm(q) * (head_dim ** -0.5)
    k = l2norm(k)

    state = state_cache.state(layer) if state_cache is not None else None
    if state is None:
        state = mx.zeros((batch, num_heads, head_dim, head_v_dim), dtype=mx.float32)
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
        mx.eval(new_conv_history)
        state_cache.set_conv_history(layer, (new_conv_history,))

    from .qwen35 import _silu_gated_rms_norm

    gate = _linear(h, w, f"{prefix}.self_attn.g_proj").reshape(
        batch, length, num_heads, head_v_dim)
    output = _silu_gated_rms_norm(
        output, gate, w[f"{prefix}.self_attn.o_norm.weight"], cfg.rms_norm_eps)
    output = output.reshape(batch, length, value_dim)
    return _linear(output, w, f"{prefix}.self_attn.o_proj")


def run_jet_nemotron_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig,
    kv: "object", layer: int, offset: int,
    rope_freqs: mx.array | None = None,
    mlp_last_only: bool = False,
) -> mx.array:
    """One Jet-Nemotron decoder block, dispatching on
    ``cfg.layer_types[layer]`` ("jet" | "swa" | "attn"). Mirrors
    ``layer_runner.run_block``'s residual/norm wiring exactly -- only the
    attention sub-block differs per layer type. "attn"/"swa" reuse
    ``layer_runner._attention`` unchanged (bias-based QKV, no QK-norm,
    standard rope, both already handled generically); "swa" additionally
    passes ``sliding_window`` (added to ``_attention`` specifically to
    support this model, mirroring gpt-oss's own "sliding_attention"
    handling). "jet" layers carry NoPE recurrent state instead of
    positional KV -- ``kv`` must expose a KDAStateCache-compatible
    ``.state(layer)``/``.set_state(layer, ...)`` interface for those
    layers (see runtime/kda_state.py; the same interface Qwen3.5/Kimi
    Linear's own hybrid layers already use).
    """
    layer_type = cfg.layer_types[layer] if cfg.layer_types else "attn"
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    if layer_type == "jet":
        recurrent_state = getattr(kv, "kda_cache", None)
        x = x + _jet_block(h, w, prefix, cfg, recurrent_state, layer)
    else:
        sliding_window = cfg.sliding_window if layer_type == "swa" else 0
        x = x + _attention(
            h, w, prefix, cfg, kv, layer, offset,
            rope_freqs=rope_freqs, sliding_window=sliding_window)
    if mlp_last_only:
        x = x[:, -1:, :]
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    return x + _swiglu(h, w, f"{prefix}.mlp")
