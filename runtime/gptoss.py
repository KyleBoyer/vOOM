"""gpt-oss (GptOssForCausalLM) block math: YaRN RoPE, per-head attention sinks,
alternating 128-token sliding windows, and MXFP4 MoE (128 experts, top-4, fused
gate_up with clamped swiglu variant).

MXFP4 verification (2026-07-10, real checkpoint bytes): viewing the HF
`*_blocks` uint8 tensors as uint32 and the `*_scales` uint8 as-is makes
mx.dequantize / mx.quantized_matmul(mode="mxfp4", group_size=32, bits=4) agree
with a manual OCP-spec decode to max|diff| = 0.0 — no repacking is needed.

Expert tensors are stored fused ([128, ...]); formats/packed.py unfuses them into
per-expert pages at pack time, which is what makes expert paging possible here.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .config import ModelConfig
from .layer_runner import _linear


def yarn_params(cfg: ModelConfig) -> tuple[mx.array, float]:
    """Return (rope freqs for mx.fast.rope, attention scaling mscale)."""
    rs = cfg.rope_scaling
    dim = cfg.head_dim
    base = cfg.rope_theta
    inv = base ** (mx.arange(0, dim, 2) / dim)  # mx.fast.rope wants freqs, not inv_freq
    if not rs or rs.get("rope_type") != "yarn":
        return inv, 1.0
    factor = rs["factor"]
    orig_max = rs["original_max_position_embeddings"]
    beta_fast, beta_slow = rs.get("beta_fast", 32.0), rs.get("beta_slow", 1.0)

    def correction_dim(num_rot):
        return dim * math.log(orig_max / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    low = math.floor(correction_dim(beta_fast))
    high = math.ceil(correction_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    ramp = mx.clip((mx.arange(dim // 2) - low) / max(high - low, 1e-3), 0.0, 1.0)
    # YaRN: dims below `low` (high frequency, wavelength << context) EXTRAPOLATE
    # (keep original denominators); dims above `high` INTERPOLATE (denominator
    # * factor = slower rotation). The first release of this function had the
    # blend swapped, which scrambled positions progressively with distance and
    # degenerated every gpt-oss generation past ~40 tokens.
    freqs = inv * (1 - ramp) + inv * factor * ramp
    mscale = 0.1 * math.log(factor) + 1.0
    return freqs, mscale


def _attention_gptoss(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
    freqs: mx.array, mscale: float,
) -> mx.array:
    B, L, _ = h.shape
    n_h, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    q = _linear(h, w, f"{prefix}.self_attn.q_proj").reshape(B, L, n_h, hd).transpose(0, 2, 1, 3)
    k = _linear(h, w, f"{prefix}.self_attn.k_proj").reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    v = _linear(h, w, f"{prefix}.self_attn.v_proj").reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)

    q = mx.fast.rope(q, hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=freqs)
    k = mx.fast.rope(k, hd, traditional=False, base=None, scale=1.0, offset=offset, freqs=freqs)
    if mscale != 1.0:  # YaRN attention scaling applies to both q and k
        q = q * mscale
        k = k * mscale

    keys, values = kv.update(layer, k, v)
    sliding = bool(cfg.layer_types) and cfg.layer_types[layer] == "sliding_attention"
    if sliding and L == 1 and keys.shape[2] > cfg.sliding_window:
        # decode fast path: a sliding layer only ever sees the last `window` keys —
        # unmasked full attention here corrupts generations past ~window tokens
        keys = keys[:, :, -cfg.sliding_window :, :]
        values = values[:, :, -cfg.sliding_window :, :]
    S = keys.shape[2]
    rep = n_h // n_kv
    keys = mx.repeat(keys, rep, axis=1)
    values = mx.repeat(values, rep, axis=1)

    scores = (q * hd**-0.5) @ keys.transpose(0, 1, 3, 2)  # (B, n_h, L, S)

    if L > 1:  # prefill: causal (+ sliding) mask; decode L=1 needs none (see slice above)
        q_pos = mx.arange(offset, offset + L)[:, None]
        k_pos = mx.arange(S)[None, :]
        allowed = k_pos <= q_pos
        if sliding:
            allowed = allowed & (k_pos > q_pos - cfg.sliding_window)
        scores = mx.where(allowed[None, None], scores, mx.array(float("-inf")))

    # per-head sink logit joins the softmax denominator (never attended to)
    sinks = w[f"{prefix}.self_attn.sinks"].reshape(1, n_h, 1, 1).astype(scores.dtype)
    m = mx.maximum(scores.max(axis=-1, keepdims=True), sinks)
    p = mx.exp(scores - m)
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(sinks - m)
    attn = (p / denom) @ values
    attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * hd)
    return _linear(attn, w, f"{prefix}.self_attn.o_proj")


def _mxfp4_linear(x: mx.array, blocks: mx.array, scales: mx.array, bias: mx.array) -> mx.array:
    rows = blocks.shape[0]
    wq = blocks.reshape(rows, -1).view(mx.uint32)
    return mx.quantized_matmul(
        x, wq, scales=scales, transpose=True, group_size=32, bits=4, mode="mxfp4"
    ) + bias


def _group_routes(idx: mx.array, weights: mx.array
                  ) -> dict[int, list[tuple[int, float]]]:
    """Transfer evaluated routes once, preserving position/top-k insertion order."""
    index_rows = idx.tolist()[0]
    weight_rows = weights.tolist()[0]
    groups: dict[int, list[tuple[int, float]]] = {}
    for position, (experts, route_weights) in enumerate(
            zip(index_rows, weight_rows)):
        for expert, route_weight in zip(experts, route_weights):
            groups.setdefault(int(expert), []).append(
                (position, float(route_weight)))
    return groups


def run_gptoss_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
    get_experts, freqs: mx.array, mscale: float, mlp_last_only: bool = False,
) -> mx.array:
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    x = x + _attention_gptoss(h, w, prefix, cfg, kv, layer, offset, freqs, mscale)
    if mlp_last_only:  # F36: KV is built; only the last position feeds the logits
        x = x[:, -1:, :]

    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    logits = _linear(h, w, f"{prefix}.mlp.router")  # (1, L, E), bf16 + bias
    k = cfg.num_experts_per_tok
    idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    sel = mx.take_along_axis(logits, idx, axis=-1)
    pw = mx.softmax(sel.astype(mx.float32), axis=-1)  # gpt-oss: softmax over the top-k logits
    mx.eval(idx, pw)

    groups = _group_routes(idx, pw)

    limit = cfg.swiglu_limit
    out = mx.zeros_like(h)
    experts = get_experts(layer, sorted(groups), positions={e: [pt for pt, _ in v] for e, v in groups.items()})
    for e, plist in groups.items():
        ew = experts[e]
        p = f"{prefix}.mlp.experts.{e}"
        positions = [pt for pt, _ in plist]
        weights = mx.array([wt for _, wt in plist]).astype(h.dtype)
        hx = h[:, positions, :]
        gu = _mxfp4_linear(hx, ew[f"{p}.gate_up_blocks"], ew[f"{p}.gate_up_scales"], ew[f"{p}.gate_up_bias"])
        gate, up = gu[..., 0::2], gu[..., 1::2]
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
        glu = gate * mx.sigmoid(gate * 1.702)
        y = _mxfp4_linear((up + 1) * glu, ew[f"{p}.down_blocks"], ew[f"{p}.down_scales"], ew[f"{p}.down_bias"])
        out = out.at[:, positions, :].add(y * weights[None, :, None])
    return x + out
