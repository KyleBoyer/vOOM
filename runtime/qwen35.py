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


def _zmlx_deltanet_kernels():
    """Lazy import so zmlx stays an optional dependency for everyone who
    doesn't set rc.zmlx_fused_deltanet_decode (SQ26)."""
    from zmlx.kernels import deltanet
    return deltanet


def _zmlx_causal_depthwise_conv1d(
    x: mx.array, weight: mx.array, history: mx.array | None, kernel_size: int,
) -> tuple[mx.array, mx.array]:
    """zmlx-fused drop-in for _causal_depthwise_conv1d, decode (L=1) only.

    Real greedy-token quality gate (2026-07-23,
    tests/test_zmlx_fused_deltanet_decode.py): 24 real generated tokens
    against Qwen3.5-4B, byte-identical to the existing float32-accumulated
    implementation despite a genuine per-call bf16 precision difference
    (SQ26) -- the difference doesn't flip any observed argmax. Callers must
    only use this for L==1; falls back to the exact existing
    implementation otherwise as a safety net, not a silent shape assumption.
    """
    b, length, channels = x.shape
    if length != 1:
        return _causal_depthwise_conv1d(x, weight, history, kernel_size)
    if history is None:
        history = mx.zeros((b, kernel_size - 1, channels), dtype=x.dtype)
    conv_input = mx.concatenate([history, x], axis=1)  # (B, K, C)
    out, new_history = _zmlx_deltanet_kernels().fused_conv1d_silu(
        conv_input, weight)
    return out, new_history


def _zmlx_silu_gated_rms_norm(
    x: mx.array, gate: mx.array, weight: mx.array, eps: float,
) -> mx.array:
    """zmlx-fused drop-in for _silu_gated_rms_norm -- see
    _zmlx_causal_depthwise_conv1d's docstring for the quality-gate proof.
    Unlike the conv fusion, zmlx's gated_rmsnorm_silu has no shape
    restriction, but this is still only ever called for L==1 (see
    _gated_delta_net's own decode-only gating of zmlx_fused_decode)."""
    return _zmlx_deltanet_kernels().gated_rmsnorm_silu(
        x, gate, weight, eps=eps)


_NATIVE_CONV1D_STEP_SOURCE = """
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    uint C = padded_shape[2];
    uint K = padded_shape[1];
    if (c >= C) return;

    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        acc += float(padded[(b * K + k) * C + c]) * float(taps[c * K + k]);
    }
    float silu = acc / (1.0 + exp(-acc));
    out[b * C + c] = T(silu);
"""

_native_conv1d_step_kernel = mx.fast.metal_kernel(
    name="qwen35_conv1d_decode_step",
    input_names=["padded", "taps"],
    output_names=["out"],
    source=_NATIVE_CONV1D_STEP_SOURCE,
)


def _native_fused_causal_conv1d(
    x: mx.array, weight: mx.array, history: mx.array | None, kernel_size: int,
) -> tuple[mx.array, mx.array]:
    """Native mx.fast.metal_kernel drop-in for _causal_depthwise_conv1d,
    decode (L=1) only -- fuses the K-tap weighted sum + SiLU into one
    dispatch instead of K elementwise-multiply-adds + a separate sigmoid*x.
    F103 (2026-07-24): verified in isolation against the reference math to
    max abs diff 4.8e-7 (near machine precision, tighter than the DeltaNet
    step kernel's 1.9e-5 since there's no float32-state-accumulation order
    difference here, just a reduction over K=4 taps), ~3.9x in an isolated
    back-to-back microbenchmark -- NOT trusted alone, see
    tests/test_native_fused_deltanet_decode.py for the real end-to-end
    verdict this specific fusion produced. Callers must only use this for
    L==1; falls back to the exact existing implementation otherwise,
    mirroring _zmlx_causal_depthwise_conv1d's own safety-net convention.
    """
    b, length, channels = x.shape
    if length != 1:
        return _causal_depthwise_conv1d(x, weight, history, kernel_size)
    if history is None:
        history = mx.zeros((b, kernel_size - 1, channels), dtype=x.dtype)
    padded = mx.concatenate([history, x], axis=1)  # (B, K, C)
    taps = weight.reshape(channels, kernel_size)
    out = _native_conv1d_step_kernel(
        inputs=[padded, taps],
        template=[("T", x.dtype)],
        grid=(channels, b, 1),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, channels)],
        output_dtypes=[x.dtype],
    )[0][:, None, :]
    new_history = (
        padded[:, 1:, :] if kernel_size > 1
        else mx.zeros((b, 0, channels), dtype=x.dtype))
    return out, new_history


def qwen35_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Official zero-centered decoder RMSNorm: norm(x.float) * (1+w).

    mx.fast.rms_norm computes x*rsqrt(mean(x^2)+eps)*weight -- no native
    "(1+w)" variant exists, but passing weight+1 as ITS weight argument
    computes exactly this formula through the same single fused Metal
    dispatch instead of the 5+ separate ops (square/mean/rsqrt/add/mul)
    the manual composite used. Verified byte-identical (0.0 max abs diff,
    not just close) against the original composite across representative
    shapes -- the previous rms_norm+residual FUSION attempt (a from-
    scratch custom kernel, F103's third target) failed and was correctly
    abandoned because it competed against this exact native primitive
    directly; this instead just routes qwen3.5's own formula THROUGH that
    primitive, which the original composite never did."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32) + 1.0
    return mx.fast.rms_norm(x32, w32, eps).astype(source_dtype)


def _silu_gated_rms_norm(
    x: mx.array, gate: mx.array, weight: mx.array, eps: float,
) -> mx.array:
    """DeltaNet's ordinary-scale RMSNorm followed by a SiLU output gate.

    Same native-primitive-reuse idea as qwen35_rms_norm above: the norm+
    weight portion is plain (no 1+w offset here), so it maps onto
    mx.fast.rms_norm directly with no weight transform at all -- only the
    SiLU gate multiply stays a separate op (it isn't part of RMSNorm's
    own math). Verified byte-identical against the original composite."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    normalized = mx.fast.rms_norm(x32, w32, eps)
    gate32 = gate.astype(mx.float32)
    silu_gate = gate32 * mx.sigmoid(gate32)
    return (normalized * silu_gate).astype(source_dtype)


def _sigmoid_gated_rms_norm(
    x: mx.array, gate: mx.array, weight: mx.array, eps: float,
) -> mx.array:
    """Qwen4-Exp's released DeltaNet output gate (plain sigmoid)."""
    source_dtype = x.dtype
    normalized = mx.fast.rms_norm(
        x.astype(mx.float32), weight.astype(mx.float32), eps)
    return (
        normalized * mx.sigmoid(gate.astype(mx.float32))
    ).astype(source_dtype)


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
    online_paged = bool(
        length == 1 and getattr(kv, "online_attention", False))
    if online_paged:
        kv.append_for_online_attention(layer, k, v)
        if getattr(kv, "online_attention_page_native", False):
            from .qwen35_paged_attention import page_native_paged_attention

            attended = page_native_paged_attention(
                q,
                kv,
                layer,
                pages_per_tile=int(getattr(
                    kv, "online_attention_pages_per_tile", 8)),
            )
        else:
            from .qwen35_paged_attention import tiled_paged_attention

            attended = tiled_paged_attention(
                q,
                kv,
                layer,
                tile_positions=int(getattr(
                    kv, "online_attention_tile_positions", 2048)),
            )
    else:
        keys, values = kv.update(layer, k, v)

    mask = None
    if not online_paged and length > 1:
        # The cache normally begins at global position zero, making its local
        # length and the RoPE offset identical. Mixed-depth lossy prefill can
        # intentionally begin an upper layer at a later global position:
        # RoPE must still use ``offset``, while the causal mask must index the
        # compact layer-local cache. Deriving the query start from the updated
        # key length is identical on ordinary caches and correct on compact
        # suffix caches (including subsequent multi-token extensions).
        mask = _cache_local_causal_mask(
            length, int(keys.shape[2]), q.dtype)
    if not online_paged:
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=head_dim ** -0.5, mask=mask)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch, length, heads * head_dim)
    attended = attended * mx.sigmoid(output_gate)
    return _linear(attended, w, f"{prefix}.self_attn.o_proj")


def _cache_local_causal_mask(
        query_length: int, key_length: int, dtype) -> mx.array:
    """Lower-right causal mask for an append-only, possibly compact KV."""
    query_length = int(query_length)
    key_length = int(key_length)
    cache_local_offset = key_length - query_length
    if query_length <= 0 or cache_local_offset < 0:
        raise ValueError("invalid Qwen causal-mask cache dimensions")
    q_pos = mx.arange(
        cache_local_offset, cache_local_offset + query_length,
        dtype=mx.int32)[:, None]
    k_pos = mx.arange(key_length, dtype=mx.int32)[None, :]
    return mx.where(
        k_pos <= q_pos, 0.0, float("-inf")).astype(dtype)


# Chunked-parallel DeltaNet (fast/lossy, 2026-07-23): replaces the sequential
# per-position recurrence for multi-position sweeps (prefill, speculative
# verify) with the chunkwise WY form below. F100's real greedy 9B gate measured
# ~4x and the private captured-request profile in F121 independently measured
# 86.62s -> 48.40s on the lossy 9B artifact. The chunkwise reassociation is
# close but not activation-identical across arbitrary prompt-checkpoint
# boundaries, so the exact endpoint continuation oracle must pass under the
# default sequential recurrence. Admission is an engine-local runtime setting:
# fast/lossy routes default on, lossless routes default off, and switching model
# IDs in one server cannot leak a process-global choice across engines.


def _sequential_gated_delta_rule(q, k, v, beta, decay, state):
    """The verbatim per-position recurrence (q,k,v: (B,L,H,dim) float32;
    beta/decay: (B,L,H); state: (B,H,K,V)). Factored out so the chunked
    twin below can be math-A/B'd directly against it -- this loop is the
    path already oracle-verified against real HF sources."""
    length = q.shape[1]
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
    return mx.stack(outputs, axis=1), state


@mx.compile
def _compiled_gated_delta_segment(q, k, v, beta, decay, state):
    """Trace one bounded segment of the reference recurrence unchanged."""
    outputs = []
    for position in range(q.shape[1]):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
        state = state * mx.exp(decay[:, position])[..., None, None]
        predicted = mx.sum(k_t[..., None] * state, axis=-2)
        delta = (v_t - predicted) * beta[:, position, :, None]
        state = state + k_t[..., None] * delta[..., None, :]
        outputs.append(mx.sum(q_t[..., None] * state, axis=-2))
    return mx.stack(outputs, axis=1), state


def _compiled_gated_delta_rule(
    q, k, v, beta, decay, state, segment: int = 32,
):
    """Compile the reference recurrence in bounded, byte-identical segments.

    The segment body deliberately repeats `_sequential_gated_delta_rule`'s
    operators and their order verbatim. Segment width 32 also preserves its
    state-materialization cadence; unlike the WY path below, this does not
    reassociate the FP32 recurrence. The final partial segment stays lazy just
    as the reference path does until its caller materializes the returned
    state.
    """
    length = int(q.shape[1])
    if length <= 0 or segment <= 0:
        raise ValueError(
            "compiled Qwen DeltaNet requires positions and a positive segment")
    outputs = []
    for start in range(0, length, segment):
        end = min(start + segment, length)
        output, state = _compiled_gated_delta_segment(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            beta[:, start:end],
            decay[:, start:end],
            state,
        )
        if end % segment == 0:
            mx.eval(state)
        outputs.append(output)
    if len(outputs) == 1:
        return outputs[0], state
    return mx.concatenate(outputs, axis=1), state


_NATIVE_DELTANET_STEP_SOURCE = """
    uint dv = thread_position_in_grid.x;
    uint h  = thread_position_in_grid.y;
    uint b  = thread_position_in_grid.z;
    uint Dk = state_shape[2];
    uint Dv = state_shape[3];
    uint H  = state_shape[1];
    if (dv >= Dv) return;

    float dec = exp(float(decay[(b * H) + h]));
    float bet = float(beta[(b * H) + h]);
    float vt  = float(v[(b * H + h) * Dv + dv]);
    uint state_base = ((b * H + h) * Dk) * Dv + dv;

    float predicted = 0.0;
    for (uint dk = 0; dk < Dk; dk++) {
        float s = float(state[state_base + dk * Dv]) * dec;
        out_state[state_base + dk * Dv] = T(s);
        predicted += float(k[(b * H + h) * Dk + dk]) * s;
    }
    float delta = (vt - predicted) * bet;

    float o = 0.0;
    for (uint dk = 0; dk < Dk; dk++) {
        float kk = float(k[(b * H + h) * Dk + dk]);
        float s = float(out_state[state_base + dk * Dv]) + kk * delta;
        out_state[state_base + dk * Dv] = T(s);
        o += float(q[(b * H + h) * Dk + dk]) * s;
    }
    out[(b * H + h) * Dv + dv] = T(o);
"""

_native_deltanet_step_kernel = mx.fast.metal_kernel(
    name="qwen35_deltanet_decode_step",
    input_names=["q", "k", "v", "beta", "decay", "state"],
    output_names=["out", "out_state"],
    source=_NATIVE_DELTANET_STEP_SOURCE,
)


def _native_fused_gated_delta_step(q, k, v, beta, decay, state):
    """Hand-written `mx.fast.metal_kernel` fusion of the ENTIRE single-position
    recurrence body (decay-scale, predicted-value dot, delta, state update,
    output dot) into one Metal dispatch, for decode (L=1) only.

    F103 (2026-07-24): this is a from-scratch custom kernel, NOT zmlx (SQ26's
    third-party library, found to be a net decode slowdown despite winning an
    isolated microbenchmark -- attributed to per-call dispatch/abstraction
    overhead that a real, interleaved generate() loop doesn't get to amortize
    the way a 200-iteration back-to-back loop does). Verified in isolation
    against `_sequential_gated_delta_rule`'s single-step body at real
    Qwen3.5-4B dimensions (H=32, Dk=Dv=128): max abs diff ~1.9e-5 (float32
    accumulation-order noise, same class of tolerance this project already
    accepts for zmlx's conv/rmsnorm fusions), ~2.1x faster in that same
    isolated back-to-back loop. Given the zmlx precedent, an isolated win is
    NOT proof of a real decode-loop win -- real end-to-end A/B on an actual
    model is the only claim this project trusts (see
    tests/test_native_fused_deltanet_decode.py and the dated STATUS.md entry
    for whichever verdict that real test produced).

    Callers must only invoke this for L==1 (single decode position); q/k/v
    here are already squeezed to (B,H,dim) -- the L axis is the caller's
    responsibility, mirroring _zmlx_causal_depthwise_conv1d's convention.
    """
    B, H, Dk = q.shape
    Dv = v.shape[-1]
    outputs = _native_deltanet_step_kernel(
        inputs=[q, k, v, beta, decay, state],
        template=[("T", state.dtype)],
        grid=(Dv, H, B),
        threadgroup=(min(Dv, 256), 1, 1),
        output_shapes=[(B, H, Dv), state.shape],
        output_dtypes=[state.dtype, state.dtype],
    )
    return outputs[0], outputs[1]


_NATIVE_DELTANET_PREFILL_SOURCE = """
    uint dv = thread_position_in_grid.x;
    uint h  = thread_position_in_grid.y;
    uint b  = thread_position_in_grid.z;
    uint L  = q_shape[1];
    uint H  = q_shape[2];
    uint Dk = q_shape[3];
    uint Dv = v_shape[3];
    if (dv >= Dv) return;

    uint state_base = ((b * H + h) * Dk) * Dv + dv;
    for (uint t = 0; t < L; t++) {
        uint bh = (b * L + t) * H + h;
        float dec = exp(float(decay[bh]));
        float bet = float(beta[bh]);
        float vt = float(v[bh * Dv + dv]);

        float predicted = 0.0f;
        for (uint dk = 0; dk < Dk; dk++) {
            uint index = state_base + dk * Dv;
            float previous = (
                t == 0 ? float(state[index]) : float(out_state[index]));
            float decayed = previous * dec;
            out_state[index] = decayed;
            predicted += float(k[bh * Dk + dk]) * decayed;
        }

        float delta = (vt - predicted) * bet;
        float value = 0.0f;
        for (uint dk = 0; dk < Dk; dk++) {
            uint index = state_base + dk * Dv;
            float kk = float(k[bh * Dk + dk]);
            float updated = float(out_state[index]) + kk * delta;
            out_state[index] = updated;
            value += float(q[bh * Dk + dk]) * updated;
        }
        out[bh * Dv + dv] = value;
    }
"""

_native_deltanet_prefill_kernel = mx.fast.metal_kernel(
    name="qwen35_deltanet_serial_prefill",
    input_names=["q", "k", "v", "beta", "decay", "state"],
    output_names=["out", "out_state"],
    source=_NATIVE_DELTANET_PREFILL_SOURCE,
)


def _native_fused_gated_delta_prefill(q, k, v, beta, decay, state):
    """Run a complete serial DeltaNet tile in one Metal dispatch.

    One thread owns one recurrent-state value column and loops over positions
    in released causal order. This removes the Python/per-position dispatch
    chain but uses an explicit scalar reduction rather than MLX's ``mx.sum``
    tree, so ordinary float32 association can differ. It is consequently a
    lossy, explicit candidate even when a finite token corpus agrees.
    """
    B, L, H, Dk = q.shape
    Dv = int(v.shape[-1])
    if L <= 1:
        raise ValueError(
            "native fused DeltaNet prefill requires more than one position")
    if (k.shape != q.shape or v.shape[:3] != (B, L, H)
            or beta.shape != (B, L, H) or decay.shape != (B, L, H)
            or state.shape != (B, H, Dk, Dv)):
        raise ValueError("invalid native fused DeltaNet prefill geometry")
    outputs = _native_deltanet_prefill_kernel(
        inputs=[q, k, v, beta, decay, state],
        grid=(Dv, H, B),
        threadgroup=(min(Dv, 256), 1, 1),
        output_shapes=[(B, L, H, Dv), state.shape],
        output_dtypes=[state.dtype, state.dtype],
    )
    return outputs[0], outputs[1]


def _chunked_gated_delta_rule(q, k, v, beta, decay, state, chunk: int = 64):
    """Chunkwise (WY/UT-transform) form of the SAME recurrence.

    Standard parallelization of the gated delta rule (Gated DeltaNet paper;
    fla's chunk_gated_delta_rule implements the same algebra in Triton):
    within a chunk, unrolling S_j = a_j*S_{j-1} + k_j u_j^T with
    u_j = beta_j (v_j - (a_j S_{j-1})^T k_j) gives a unit-lower-triangular
    system (I + M) U = R with
      M[j,l] = beta_j * (A_j/A_l) * (k_j . k_l)   (l < j)
      R_j    = beta_j * (v_j - A_j * k_j^T S_0)
      A_j    = exp(cumsum(decay)_j)
    solved by forward substitution (C tiny steps instead of C full
    (K,V)-state updates), after which outputs and the end-of-chunk state
    are plain matmuls:
      o_j = A_j * q_j^T S_0 + sum_{l<=j} (A_j/A_l)(q_j . k_l) u_l
      S_C = A_C * S_0 + Kc^T diag(A_C/A_l) U
    Inputs/outputs and semantics identical to _sequential_gated_delta_rule
    (decay applied before the read, output taken from the post-update
    state) -- proven equivalent to <=1e-4 in
    tests/test_chunked_delta_rule_oracle.py.
    """
    B, L, H, K = k.shape
    outputs = []
    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        C = end - start
        qc = q[:, start:end].transpose(0, 2, 1, 3)      # (B,H,C,K)
        kc = k[:, start:end].transpose(0, 2, 1, 3)
        vc = v[:, start:end].transpose(0, 2, 1, 3)      # (B,H,C,V)
        bc = beta[:, start:end].transpose(0, 2, 1)      # (B,H,C)
        cs = mx.cumsum(decay[:, start:end].transpose(0, 2, 1), axis=-1)
        A = mx.exp(cs)                                   # (B,H,C) = A_j
        # G[j,l] = A_j / A_l = exp(cs_j - cs_l); bounded <= 1 for l <= j
        # since decay <= 0. For l > j (never a valid term -- both M and the
        # output masks zero it out) the exponent is >= 0 and can be large
        # enough to overflow to +inf at real model scale; inf * 0 = nan,
        # which then poisons the whole chunk (caught live: real prefill
        # produced all-token-0 garbage despite the small-scale oracle test
        # passing -- its random decay range never got large enough to
        # overflow). Mask the exponent itself, not just the product, so an
        # invalid entry becomes exp(-inf) = 0 exactly, never inf * 0.
        incl_lower = mx.tri(C, k=0, dtype=q.dtype)
        neg_inf = mx.array(-float("inf"), dtype=cs.dtype)
        exponent = mx.where(
            incl_lower.astype(mx.bool_), cs[..., :, None] - cs[..., None, :],
            neg_inf)
        G = mx.exp(exponent)                             # (B,H,C,C), 0 above diag
        kk = kc @ kc.swapaxes(-1, -2)                    # (B,H,C,C)
        strict_lower = mx.tri(C, k=-1, dtype=q.dtype)
        M = bc[..., :, None] * G * kk * strict_lower
        k_s0 = kc @ state                                # (B,H,C,V)
        R = bc[..., None] * (vc - A[..., None] * k_s0)
        # Forward substitution on the unit-lower-triangular system: C tiny
        # steps -- the sequential dependency shrinks from full-state updates
        # to C-length dot products.
        u_rows = []
        for j in range(C):
            u_j = R[..., j, :]
            if j:
                stacked = mx.stack(u_rows, axis=-2)       # (B,H,j,V)
                u_j = u_j - mx.sum(
                    M[..., j, :j, None] * stacked, axis=-2)
            u_rows.append(u_j)
        U = mx.stack(u_rows, axis=-2)                     # (B,H,C,V)
        qk = qc @ kc.swapaxes(-1, -2)
        o = A[..., None] * (qc @ state) + (G * qk * incl_lower) @ U
        outputs.append(o.transpose(0, 2, 1, 3))           # (B,C,H,V)
        A_last = A[..., -1]                               # (B,H)
        carry = mx.exp(cs[..., -1, None] - cs)            # A_C/A_l, (B,H,C)
        state = A_last[..., None, None] * state + (
            kc.swapaxes(-1, -2) @ (carry[..., None] * U))
        mx.eval(state)
    return mx.concatenate(outputs, axis=1), state


def _gated_delta_net(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig,
    state_cache: KDAStateCache | None, layer: int,
    zmlx_fused_decode: bool = False,
    defer_state_eval: bool = False,
    native_fused_decode: bool = False,
    chunked_delta_prefill: bool = False,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
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
    conv_fn = _causal_depthwise_conv1d
    if length == 1:
        if native_fused_decode:
            conv_fn = _native_fused_causal_conv1d
        elif zmlx_fused_decode:
            conv_fn = _zmlx_causal_depthwise_conv1d
    mixed, new_history = conv_fn(
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
    if sum(map(bool, (
            chunked_delta_prefill,
            compiled_delta_prefill,
            native_fused_delta_prefill))) > 1:
        raise ValueError(
            "chunked, compiled, and native-fused Qwen DeltaNet prefill are "
            "mutually exclusive")
    if chunked_delta_prefill and length > 1:
        output, state = _chunked_gated_delta_rule(q, k, v, beta, decay, state)
    elif compiled_delta_prefill and length > 1:
        output, state = _compiled_gated_delta_rule(
            q, k, v, beta, decay, state)
    elif native_fused_delta_prefill and length > 1:
        output, state = _native_fused_gated_delta_prefill(
            q, k, v, beta, decay, state)
    elif native_fused_decode and length == 1:
        # F103: hand-written mx.fast.metal_kernel fusion of the single-step
        # recurrence body. See _native_fused_gated_delta_step's docstring.
        step_out, state = _native_fused_gated_delta_step(
            q[:, 0], k[:, 0], v[:, 0], beta[:, 0], decay[:, 0], state)
        output = step_out[:, None]
    else:
        output, state = _sequential_gated_delta_rule(q, k, v, beta, decay, state)
    # The released Transformers kernels accumulate the DeltaNet recurrence in
    # FP32 but cast ``core_attn_out`` back to the input Q/K dtype before the
    # gated RMSNorm and output projection.  ``q`` above is deliberately
    # promoted for the reference recurrence, so leaving its result in FP32
    # silently promotes every Qwen4 hyper residual after a linear-attention
    # layer and doubles long-context activation storage.
    output = output.astype(h.dtype)
    if state_cache is not None:
        if state_cache.factor_capture_active:
            if length != 1:
                raise ValueError(
                    "compact Qwen DeltaNet factor capture requires serial "
                    "positions")
            state_cache.capture_factor_step(
                layer,
                gate=decay[:, 0],
                key=k[:, 0],
                value=v[:, 0],
                beta=beta[:, 0],
                conv_history=(new_history,),
            )
        # Per-layer state eval is a bounded-lazy-graph checkpoint for long
        # prefill sweeps, but at decode (L=1) it is one of ~24 pure GPU sync
        # points per token. The resident hybrid fast path (engine._sweep)
        # defers it and batch-evals every layer's updated state in ONE call
        # at the sweep boundary instead -- identical arithmetic, different
        # eval boundary only.
        if not defer_state_eval:
            mx.eval(state)
        state_cache.set_state(layer, state)
        state_cache.set_conv_history(layer, (new_history,))

    z = _linear(h, w, f"{prefix}.linear_attn.in_proj_z").reshape(
        batch, length, value_heads, value_dim)
    output_gate_type = getattr(cfg, "qwen4_output_gate_type", "")
    if output_gate_type == "sigmoid":
        norm_fn = _sigmoid_gated_rms_norm
    else:
        norm_fn = (
            _zmlx_silu_gated_rms_norm if zmlx_fused_decode and length == 1
            else _silu_gated_rms_norm)
    output = norm_fn(
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
    get_experts, iter_expert_batches=None, profile=None,
) -> mx.array:
    router_t0 = profile.start_substep() if profile is not None else None
    indices, scores = _route_experts(h, w, prefix, cfg, layer)
    if not (profile is not None and profile.finish_substep(
            "router", layer, router_t0, indices, scores,
            positions=int(h.shape[1]))):
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


def _qwen35_attention_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, mlp_last_only: bool = False,
    positions3: np.ndarray | mx.array | None = None,
    zmlx_fused_decode: bool = False,
    defer_state_eval: bool = False,
    native_fused_decode: bool = False,
    chunked_delta_prefill: bool = False,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
) -> mx.array:
    """DeltaNet-or-full-attention + residual only, no MLP/MoE -- split out of
    run_qwen35_block (2026-07-25) so a caller can run attention per-tile
    (DeltaNet state and ordinary KV both still need causal tile order,
    unchanged) while deferring MoE routing to run once per layer, mirroring
    _kimi_linear_attention_residual/_glm_attention_residual's own split for
    the same reason -- this is what lets F35's layer-stationary technique
    extend to qwen3_5_moe (Qwen3.5-35B-A3B, Qwen3.6-27B/35B-A3B's routed
    layers), not just the bare dense "qwen3_5" F94 already covered."""
    residual = x
    h = qwen35_rms_norm(
        x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    layer_type = cfg.layer_types[layer]
    if layer_type == "linear_attention":
        mixed = _gated_delta_net(
            h, w, prefix, cfg, getattr(kv, "kda_cache", None), layer,
            zmlx_fused_decode=zmlx_fused_decode,
            defer_state_eval=defer_state_eval,
            native_fused_decode=native_fused_decode,
            chunked_delta_prefill=chunked_delta_prefill,
            compiled_delta_prefill=compiled_delta_prefill,
            native_fused_delta_prefill=native_fused_delta_prefill)
    elif layer_type == "full_attention":
        mixed = _full_attention(
            h, w, prefix, cfg, kv, layer, offset, positions3)
    else:
        raise ValueError(f"unsupported Qwen3.5 layer type {layer_type!r}")
    x = residual + mixed
    if mlp_last_only:
        x = x[:, -1:, :]
    return x


def _qwen35_mlp_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None, profile=None,
) -> mx.array:
    """MLP (dense) or MoE + residual only, given x already post-attention --
    the other half of run_qwen35_block's split, see
    _qwen35_attention_residual."""
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
        iter_expert_batches=iter_expert_batches, profile=profile)


def run_qwen35_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, get_experts, mlp_last_only: bool = False,
    iter_expert_batches=None,
    positions3: np.ndarray | mx.array | None = None,
    zmlx_fused_decode: bool = False,
    defer_state_eval: bool = False,
    native_fused_decode: bool = False,
    chunked_delta_prefill: bool = False,
    compiled_delta_prefill: bool = False,
    native_fused_delta_prefill: bool = False,
    profile=None,
) -> mx.array:
    """One Qwen3.5/3.6 decoder block (chunk-major / ordinary use). Thin
    wrapper over the attention/MLP split above -- see
    StreamingEngine._layer_stationary_qwen35_sweep in engine.py for the
    layer-major caller that uses the split directly, now for MoE variants
    too, not just the dense case F94 originally built it for."""
    positions = int(x.shape[1])
    attention_t0 = profile.start_substep() if profile is not None else None
    x = _qwen35_attention_residual(
        x, w, prefix, cfg, kv, layer, offset, mlp_last_only=mlp_last_only,
        positions3=positions3, zmlx_fused_decode=zmlx_fused_decode,
        defer_state_eval=defer_state_eval,
        native_fused_decode=native_fused_decode,
        chunked_delta_prefill=chunked_delta_prefill,
        compiled_delta_prefill=compiled_delta_prefill,
        native_fused_delta_prefill=native_fused_delta_prefill)
    if profile is not None:
        profile.finish_substep(
            "attention", layer, attention_t0, x, positions=positions)
    mlp_t0 = profile.start_substep() if profile is not None else None
    x = _qwen35_mlp_residual(
        x, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
    if profile is not None:
        profile.finish_substep(
            "mlp", layer, mlp_t0, x, positions=int(x.shape[1]))
    return x


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
            chunked_delta_prefill=engine.rc.qwen_chunked_delta_prefill,
            compiled_delta_prefill=engine.rc.qwen_compiled_delta_prefill,
            native_fused_delta_prefill=(
                engine.rc.qwen_native_fused_delta_prefill),
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
            chunked_delta_prefill=engine.rc.qwen_chunked_delta_prefill,
            compiled_delta_prefill=engine.rc.qwen_compiled_delta_prefill,
            native_fused_delta_prefill=(
                engine.rc.qwen_native_fused_delta_prefill),
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
