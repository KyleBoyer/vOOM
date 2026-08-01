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
    """Return released-model-correct YaRN parameters for ``mx.fast.rope``.

    Transformers expresses YaRN as a linear blend of *inverse* frequencies,
    whereas MLX's ``freqs=`` argument expects their reciprocal (the RoPE
    denominators).  Blending the denominators directly is not equivalent, so
    perform the reference blend first and invert only the final result.
    GPT-OSS also publishes ``truncate: false`` and therefore uses the floating
    correction bounds rather than the more common floor/ceil variant.
    """
    rs = cfg.rope_scaling
    dim = cfg.head_dim
    base = cfg.rope_theta
    pos_freqs = base ** (
        mx.arange(0, dim, 2, dtype=mx.float32) / dim
    )  # mx.fast.rope wants denominators, not inverse frequencies
    if not rs or rs.get("rope_type") != "yarn":
        return pos_freqs, 1.0
    factor = rs["factor"]
    orig_max = rs["original_max_position_embeddings"]
    beta_fast, beta_slow = rs.get("beta_fast", 32.0), rs.get("beta_slow", 1.0)

    def correction_dim(num_rot):
        return dim * math.log(orig_max / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if rs.get("truncate", True):
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    denominator = high - low
    if denominator == 0:
        denominator = 0.001
    ramp = mx.clip(
        (mx.arange(dim // 2, dtype=mx.float32) - low) / denominator,
        0.0,
        1.0,
    )

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    freqs = 1.0 / inv_freq
    mscale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
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
        # A windowed layer stores a SUFFIX of the sequence, so its first key is
        # at absolute position ``base``, not 0. Without this the causal and
        # sliding masks would be applied to the wrong positions.
        base = kv.layer_start(layer) if hasattr(kv, "layer_start") else 0
        k_pos = mx.arange(base, base + S)[None, :]
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


def _gptoss_attention_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int,
    offset: int, freqs: mx.array, mscale: float,
) -> mx.array:
    """Run GPT-OSS's input norm and attention residual for one causal tile.

    This split is intentionally just the first half of ``run_gptoss_block``.
    Keeping it public-to-the-runtime lets the layer-stationary prefill path
    advance each layer's KV in bounded position tiles before routing the
    complete post-attention sequence once.  No parameter, normalization, RoPE,
    sink, masking, or residual arithmetic differs from the ordinary block.
    """
    h = mx.fast.rms_norm(
        x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    return x + _attention_gptoss(
        h, w, prefix, cfg, kv, layer, offset, freqs, mscale)


def _gptoss_mlp_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, *, iter_expert_batches=None, profile=None,
) -> mx.array:
    """Run GPT-OSS's router and MXFP4 MoE with bounded expert lifetime.

    Route selection and contribution order are identical to the historical
    monolithic block.  When ``iter_expert_batches`` is supplied, each bounded
    page mapping is consumed and the accumulated output is materialized before
    requesting the next mapping.  This is the same F74-v2 ownership boundary
    used by the other paged MoE runners and prevents a full 128-expert union
    from remaining strongly referenced during long prefill ranges.
    """
    h = mx.fast.rms_norm(
        x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    logits = _linear(h, w, f"{prefix}.mlp.router")
    k = cfg.num_experts_per_tok
    idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    sel = mx.take_along_axis(logits, idx, axis=-1)
    pw = mx.softmax(sel.astype(mx.float32), axis=-1)
    mx.eval(idx, pw)

    groups = _group_routes(idx, pw)
    # Compute contributions in the exact historical insertion order.  Fetch
    # order was previously sorted, but the old loop consumed ``groups.items``;
    # the bounded producer may fetch in this same consumption order without
    # changing any arithmetic.
    expert_ids = list(groups)
    positions_by_expert = {
        expert: [position for position, _weight in groups[expert]]
        for expert in expert_ids
    }
    if iter_expert_batches is None:
        pages = get_experts(
            layer, sorted(expert_ids), positions=positions_by_expert)
        batches = ((expert_ids, pages),)
    else:
        batches = iter_expert_batches(
            layer, expert_ids, positions=positions_by_expert)

    limit = cfg.swiglu_limit
    out = mx.zeros_like(h)
    for batch_ids, pages in batches:
        for expert in batch_ids:
            plist = groups[expert]
            ew = pages[expert]
            p = f"{prefix}.mlp.experts.{expert}"
            positions = [position for position, _weight in plist]
            route_weights = mx.array(
                [weight for _position, weight in plist]).astype(h.dtype)
            hx = h[:, positions, :]
            gu = _mxfp4_linear(
                hx, ew[f"{p}.gate_up_blocks"],
                ew[f"{p}.gate_up_scales"], ew[f"{p}.gate_up_bias"])
            gate, up = gu[..., 0::2], gu[..., 1::2]
            gate = mx.minimum(gate, limit)
            up = mx.clip(up, -limit, limit)
            glu = gate * mx.sigmoid(gate * 1.702)
            y = _mxfp4_linear(
                (up + 1) * glu, ew[f"{p}.down_blocks"],
                ew[f"{p}.down_scales"], ew[f"{p}.down_bias"])
            out = out.at[:, positions, :].add(
                y * route_weights[None, :, None])
        if iter_expert_batches is not None:
            # Materialize the accumulated activation before the producer is
            # resumed so no lazy graph can retain this batch's expert pages.
            mx.eval(out)
        del pages
    return x + out


def _gptoss_tiled_mlp_residual(
    tiles: list[mx.array], w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, *, iter_expert_batches=None, profile=None,
) -> mx.array:
    """Evaluate a tile-equivalent MoE while fetching each expert union once.

    Layer-stationary prefill must retain the ordinary fixed tile's matrix
    shapes as well as route/contribution order: changing a router or expert
    GEMM from (tile, hidden) to (whole_prompt, hidden) can select a different
    floating kernel. Route and normalize every tile independently, materialize
    each expert contribution while its bounded page is live, then reconstruct
    each tile in its original first-seen expert order after all pages are gone.
    The retained contribution volume is only routed activation output
    (positions * top-k * hidden), never expert weights.
    """
    if not tiles:
        raise ValueError("GPT-OSS tiled MLP needs at least one tile")
    routed = []
    expert_ids: list[int] = []
    positions_by_expert: dict[int, list[int]] = {}
    position_base = 0
    for x in tiles:
        h = mx.fast.rms_norm(
            x, w[f"{prefix}.post_attention_layernorm.weight"],
            cfg.rms_norm_eps)
        logits = _linear(h, w, f"{prefix}.mlp.router")
        k = cfg.num_experts_per_tok
        idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        sel = mx.take_along_axis(logits, idx, axis=-1)
        pw = mx.softmax(sel.astype(mx.float32), axis=-1)
        mx.eval(idx, pw)
        groups = _group_routes(idx, pw)
        for expert, plist in groups.items():
            if expert not in positions_by_expert:
                expert_ids.append(expert)
                positions_by_expert[expert] = []
            positions_by_expert[expert].extend(
                position_base + position for position, _weight in plist)
        routed.append((x, h, groups))
        position_base += int(x.shape[1])

    if iter_expert_batches is None:
        pages = get_experts(
            layer, sorted(expert_ids), positions=positions_by_expert)
        batches = ((expert_ids, pages),)
    else:
        batches = iter_expert_batches(
            layer, expert_ids, positions=positions_by_expert)

    limit = cfg.swiglu_limit
    contributions: dict[tuple[int, int], tuple[list[int], mx.array]] = {}
    for batch_ids, pages in batches:
        for expert in batch_ids:
            ew = pages[expert]
            p = f"{prefix}.mlp.experts.{expert}"
            for tile_index, (_x, h, groups) in enumerate(routed):
                plist = groups.get(expert)
                if not plist:
                    continue
                positions = [position for position, _weight in plist]
                route_weights = mx.array(
                    [weight for _position, weight in plist]).astype(h.dtype)
                hx = h[:, positions, :]
                gu = _mxfp4_linear(
                    hx, ew[f"{p}.gate_up_blocks"],
                    ew[f"{p}.gate_up_scales"], ew[f"{p}.gate_up_bias"])
                gate, up = gu[..., 0::2], gu[..., 1::2]
                gate = mx.minimum(gate, limit)
                up = mx.clip(up, -limit, limit)
                glu = gate * mx.sigmoid(gate * 1.702)
                contribution = _mxfp4_linear(
                    (up + 1) * glu, ew[f"{p}.down_blocks"],
                    ew[f"{p}.down_scales"], ew[f"{p}.down_bias"])
                contribution = contribution * route_weights[None, :, None]
                mx.eval(contribution)
                contributions[(tile_index, expert)] = (
                    positions, contribution)
        del pages

    outputs = []
    for tile_index, (x, h, groups) in enumerate(routed):
        out = mx.zeros_like(h)
        for expert in groups:
            positions, contribution = contributions.pop((tile_index, expert))
            out = out.at[:, positions, :].add(contribution)
        outputs.append(x + out)
    if contributions:
        raise RuntimeError("unconsumed GPT-OSS tiled expert contributions")
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)


def run_gptoss_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
    get_experts, freqs: mx.array, mscale: float, mlp_last_only: bool = False,
    iter_expert_batches=None, profile=None,
) -> mx.array:
    x = _gptoss_attention_residual(
        x, w, prefix, cfg, kv, layer, offset, freqs, mscale)
    if mlp_last_only:  # F36: KV is built; only the last position feeds the logits
        x = x[:, -1:, :]
    return _gptoss_mlp_residual(
        x, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
