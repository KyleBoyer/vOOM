"""DeepSeek V4 hyper-connections and block plumbing.

DeepSeek-V4-Flash replaces the ordinary residual stream with **hyper-connections
(HC)**: the hidden state is carried as ``hc_mult`` (4) parallel copies, and
every sub-layer reduces those copies to one on the way in and re-expands to
four on the way out. Nothing else in this runtime has that topology -- every
other architecture here is ``x = x + f(norm(x))`` -- so it is implemented and
oracled separately before any block runner is built on top of it.

The reduction/expansion weights are not parameters directly; they are produced
per token by a Sinkhorn normalization of a learned projection of the hidden
state. The released ``inference/kernel.py`` implements that as a fused
TileLang kernel; this is the same arithmetic in MLX, checked against a
transcription of the kernel in ``tests/test_f205_deepseek_v4_hc.py``.

Ordering matters and is not obvious from the shapes: ``comb`` is softmaxed over
its last axis, then column-normalized *once*, and only then does the
``sinkhorn_iters - 1`` row/column loop run. Getting that first column step
wrong still produces a doubly-stochastic-looking matrix.
"""

from __future__ import annotations

import mlx.core as mx


def hc_split_sinkhorn(mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
                      hc_mult: int = 4, sinkhorn_iters: int = 20,
                      eps: float = 1e-6):
    """Split the per-token mix vector into ``(pre, post, comb)``.

    ``mixes`` is ``[..., (2 + hc) * hc]``. The first ``hc`` entries become the
    input reduction weights, the next ``hc`` the output expansion weights, and
    the remaining ``hc * hc`` the residual combination matrix, which is driven
    toward doubly stochastic by Sinkhorn iteration.
    """
    hc = int(hc_mult)
    pre = mx.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * mx.sigmoid(
        mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])

    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:])
    comb = comb.reshape(*comb.shape[:-1], hc, hc)

    comb = mx.softmax(comb, axis=-1) + eps
    # One column normalization BEFORE the loop, matching the released kernel.
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(int(sinkhorn_iters) - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(x: mx.array, hc_fn: mx.array, hc_scale: mx.array,
           hc_base: mx.array, *, hc_mult: int, norm_eps: float,
           sinkhorn_iters: int, eps: float):
    """Reduce ``[b, s, hc, d]`` streams to a single ``[b, s, d]`` input.

    The mix projection runs in float32 over the *flattened* stream dimension,
    and its RMS factor is computed over all ``hc * d`` values jointly -- not
    per stream. Normalizing per stream would be a different function.
    """
    shape = x.shape
    flat = x.reshape(*shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True)
                     + norm_eps)
    mixes = (flat @ hc_fn.astype(mx.float32).T) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes, hc_scale.astype(mx.float32), hc_base.astype(mx.float32),
        hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)
    reduced = mx.sum(pre[..., None] * flat.reshape(shape), axis=2)
    return reduced.astype(x.dtype), post, comb


def hc_post(x: mx.array, residual: mx.array, post: mx.array,
            comb: mx.array) -> mx.array:
    """Expand a ``[b, s, d]`` sub-layer output back to ``[b, s, hc, d]``.

    ``comb`` mixes the *incoming* streams among themselves while ``post``
    scales the new contribution into each stream. The sum runs over the first
    of ``comb``'s two stream axes.
    """
    scaled = post[..., None] * x[..., None, :]
    mixed = mx.sum(comb[..., None] * residual[..., None, :], axis=2)
    return (scaled + mixed).astype(x.dtype)


def hc_head(x: mx.array, hc_fn: mx.array, hc_scale: mx.array,
            hc_base: mx.array, *, norm_eps: float, eps: float) -> mx.array:
    """Final reduction before the LM head.

    Unlike ``hc_pre`` this needs no Sinkhorn: every element of the projected
    mix is used directly as a sigmoid gate, so the head consumes the whole
    ``mix_hc`` vector rather than the first ``hc`` entries.
    """
    shape = x.shape
    flat = x.reshape(*shape[:2], -1).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True)
                     + norm_eps)
    mixes = (flat @ hc_fn.astype(mx.float32).T) * rsqrt
    pre = mx.sigmoid(mixes * hc_scale + hc_base) + eps
    return mx.sum(pre[..., None] * flat.reshape(shape), axis=2).astype(x.dtype)


# ---- sparse windowed attention (F206) --------------------------------------
#
# DeepSeek V4's attention gathers an explicit per-position index list rather
# than masking a dense score matrix. ``-1`` marks an unused slot. A single
# shared KV vector of ``head_dim`` serves every head (config's
# ``num_key_value_heads: 1``), so the gathered values are [topk, d], not
# per-head.
#
# The learned per-head ``attn_sink`` is NOT an extra key: it contributes only
# to the softmax denominator, letting a head attend to "nothing" and shrink its
# output. Adding it as a key instead would also add its value vector, which is
# a different function.


def sparse_windowed_attention(q: mx.array, kv: mx.array, attn_sink: mx.array,
                              topk_idxs: mx.array,
                              softmax_scale: float) -> mx.array:
    """Gathered sparse attention with per-head sinks.

    ``q``  is ``[b, s, h, d]``; ``kv`` is ``[b, n, d]`` shared across heads;
    ``topk_idxs`` is ``[b, s, topk]`` with ``-1`` for unused slots.
    """
    b, s, h, d = q.shape
    topk = topk_idxs.shape[-1]
    valid = topk_idxs >= 0
    safe = mx.maximum(topk_idxs, 0).astype(mx.int32)

    # [b, s, topk, d]
    gathered = mx.take_along_axis(
        mx.broadcast_to(kv[:, None, :, :], (b, s, kv.shape[1], d)),
        safe[..., None], axis=2)

    scores = mx.einsum("bshd,bskd->bshk", q.astype(mx.float32),
                       gathered.astype(mx.float32)) * softmax_scale
    scores = mx.where(valid[:, :, None, :], scores, -mx.inf)

    row_max = mx.max(scores, axis=-1, keepdims=True)
    # A row whose slots are all unused would otherwise propagate -inf; the sink
    # alone then carries the denominator.
    row_max = mx.where(mx.isinf(row_max), mx.zeros_like(row_max), row_max)
    weights = mx.where(valid[:, :, None, :], mx.exp(scores - row_max), 0.0)
    denominator = mx.sum(weights, axis=-1, keepdims=True) + mx.exp(
        attn_sink.reshape(1, 1, h, 1).astype(mx.float32) - row_max)
    out = mx.einsum("bshk,bskd->bshd", weights, gathered.astype(mx.float32))
    return (out / denominator).astype(q.dtype)


def window_topk_idxs(window_size: int, seqlen: int, start_pos: int
                     ) -> mx.array:
    """Sliding-window index list, matching ``get_window_topk_idxs``.

    Decode reuses a ring buffer, so the index list is a rotation rather than a
    contiguous range; the ``-1`` padding marks slots not yet written.
    """
    if start_pos >= window_size - 1:
        rotated = start_pos % window_size
        matrix = mx.concatenate([
            mx.arange(rotated + 1, window_size),
            mx.arange(0, rotated + 1),
        ])[None, :]
    elif start_pos > 0:
        head = mx.arange(start_pos + 1)
        pad = mx.full((window_size - start_pos - 1,), -1, dtype=head.dtype)
        matrix = mx.concatenate([head, pad])[None, :]
    else:
        base = mx.arange(seqlen)[:, None]
        width = min(seqlen, window_size)
        matrix = mx.maximum(base - window_size + 1, 0) + mx.arange(width)
        matrix = mx.where(matrix > base, -1, matrix)
    return matrix.astype(mx.int32)[None]


# ---- KV compression (F207) -------------------------------------------------
#
# Beyond the sliding window, DeepSeek V4 keeps a *compressed* KV: every
# ``compress_ratio`` consecutive positions are pooled into one entry by a
# learned softmax gate. ``compress_ratios`` varies per layer (0, 4, or 128 in
# the released config), and only ratio 4 uses overlapping windows.
#
# The QAT activation round-trip the released code applies afterwards
# (``act_quant``) is injectable rather than assumed: it is a separate,
# separately-testable step, and passing ``None`` keeps this function exactly
# the pooling arithmetic so an oracle can isolate it.


def compressor_overlap_transform(tensor: mx.array, head_dim: int,
                                 ratio: int, fill: float) -> mx.array:
    """Interleave each group with the previous group's overlap half.

    ``tensor`` is ``[b, n, ratio, 2 * head_dim]``: the first ``head_dim``
    features carry the overlapping window and the second ``head_dim`` the
    ordinary one. The result is ``[b, n, 2 * ratio, head_dim]`` whose later
    ``ratio`` slots hold this group's ordinary half and whose first ``ratio``
    slots hold the *previous* group's overlap half -- so group zero has no
    predecessor and keeps the fill value.
    """
    b, n = tensor.shape[0], tensor.shape[1]
    out = mx.full((b, n, 2 * ratio, head_dim), fill, dtype=tensor.dtype)
    out[:, :, ratio:] = tensor[..., head_dim:]
    if n > 1:
        out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return out


def compress_prefill(x: mx.array, wkv: mx.array, wgate: mx.array,
                     ape: mx.array, norm_weight: mx.array, *,
                     ratio: int, head_dim: int, norm_eps: float,
                     act_quant=None):
    """Pool a prompt into compressed KV entries (the ``start_pos == 0`` path).

    Returns ``(compressed, leftover_positions)``. Positions past the last whole
    group are not compressed; the released code parks them in a state buffer
    for the decode path to finish, so the count is reported rather than
    silently dropped.

    Pooling runs in float32 -- the released module holds ``wkv``/``wgate`` in
    float32 for exactly this reason -- and the gate softmax is taken over the
    group axis, not the feature axis.
    """
    overlap = ratio == 4
    coff = 2 if overlap else 1
    b, seqlen, _ = x.shape
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if cutoff < ratio:
        return None, seqlen

    values = x.astype(mx.float32) @ wkv.astype(mx.float32).T
    scores = x.astype(mx.float32) @ wgate.astype(mx.float32).T
    values = values[:, :cutoff]
    scores = scores[:, :cutoff]

    groups = cutoff // ratio
    values = values.reshape(b, groups, ratio, coff * head_dim)
    scores = scores.reshape(b, groups, ratio, coff * head_dim) + ape.astype(
        mx.float32)

    if overlap:
        values = compressor_overlap_transform(values, head_dim, ratio, 0.0)
        scores = compressor_overlap_transform(
            scores, head_dim, ratio, float("-inf"))

    pooled = mx.sum(values * mx.softmax(scores, axis=2), axis=2)
    pooled = mx.fast.rms_norm(pooled.astype(x.dtype), norm_weight, norm_eps)

    # RoPE is deliberately NOT applied here. Compressed entry j stands for
    # original position ``j * ratio``, not j, and compressed layers use YaRN
    # with ``compress_rope_theta`` rather than the base theta the
    # sliding-window layers use. Both belong to the caller that knows the
    # layer's compress_ratio; folding a consecutive-position RoPE in here
    # would be wrong in a way that still produces plausible activations.
    if act_quant is not None:
        pooled = act_quant(pooled)
    return pooled, remainder


# ---- indexer: top-k selection over compressed KV (F208) ---------------------
#
# Present only where ``compress_ratio == 4``. It scores every compressed entry
# against a per-head projection of the query and keeps ``index_topk`` of them,
# which become the sparse attention's gather list alongside the sliding window.


def hadamard_transform(x: mx.array, scale: float | None = None) -> mx.array:
    """Walsh-Hadamard transform over the last axis, which must be a power of two.

    Matches ``fast_hadamard_transform.hadamard_transform``: the *unnormalized*
    butterfly, then a caller-supplied scale (the released code passes
    ``d ** -0.5``, making it orthonormal). Applying the normalization inside
    the butterfly instead would scale by ``d ** -0.5`` per stage rather than
    once, which is a different transform.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"Hadamard transform needs a power-of-two axis, got {n}")
    leading = x.shape[:-1]
    y = x.astype(mx.float32)
    step = 1
    while step < n:
        y = y.reshape(*leading, n // (2 * step), 2, step)
        lower, upper = y[..., 0, :], y[..., 1, :]
        y = mx.stack([lower + upper, lower - upper], axis=-2)
        step *= 2
    y = y.reshape(*leading, n)
    return (y * scale if scale is not None else y).astype(x.dtype)


def index_scores(q: mx.array, compressed_kv: mx.array, weights: mx.array
                 ) -> mx.array:
    """Per-position scores over compressed entries.

    ``relu`` is applied to the per-head scores *before* the head-weighted sum,
    so a head can only ever add evidence for an entry, never veto another
    head's. Summing first and then rectifying would let heads cancel.
    """
    scored = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                       compressed_kv.astype(mx.float32))
    scored = mx.maximum(scored, 0.0) * weights.astype(mx.float32)[..., None]
    return mx.sum(scored, axis=2)


def index_topk_idxs(score: mx.array, seqlen: int, ratio: int, offset: int,
                    index_topk: int, end_pos: int, *, prefill: bool
                    ) -> mx.array:
    """Select compressed entries, masking those a position cannot see yet.

    A position ``p`` may only read compressed entries strictly before its own
    group, i.e. index ``< (p + 1) // ratio``. During prefill that bound is
    applied twice -- once as ``-inf`` before the top-k so unreachable entries
    are not selected, and again afterwards, because when fewer than ``k``
    entries are reachable the top-k still returns ``k`` slots and the surplus
    must be marked unused rather than silently pointing at future context.
    """
    keep = min(int(index_topk), max(end_pos // ratio, 1))
    if prefill:
        reach = (mx.arange(1, seqlen + 1) // ratio)[:, None]
        columns = mx.arange(score.shape[-1])[None, :]
        blocked = columns >= reach
        score = score + mx.where(blocked, -mx.inf, 0.0)[None]
    # argpartition yields unsigned indices; -1 marks unused slots, so the
    # cast has to happen before any masking or it overflows.
    idxs = mx.argpartition(
        -score, kth=keep - 1, axis=-1)[..., :keep].astype(mx.int32)
    if prefill:
        reach = (mx.arange(1, seqlen + 1) // ratio)[None, :, None]
        idxs = mx.where(idxs >= reach, -1, idxs + offset)
    else:
        idxs = idxs + offset
    return idxs.astype(mx.int32)


# ---- activation QAT and rotary embedding (F209) ----------------------------


_FP8_MAX = 448.0


def act_quant_simulate(x: mx.array, block_size: int = 128) -> mx.array:
    """Fused FP8 quantize/dequantize round-trip, as the released kernel does.

    This is QAT simulation, not storage: the released model runs activations
    through an E4M3 round-trip so inference matches training. Skipping it
    changes the numbers, so it is applied rather than treated as optional.

    Per block of ``block_size`` along the last axis: ``s = max(|x|, 1e-4) /
    448`` then ``e4m3(clamp(x / s, -448, 448)) * s``. The 1e-4 floor keeps an
    all-zero block from dividing by zero; without it the round-trip returns
    NaN rather than zero.
    """
    n = x.shape[-1]
    if n % block_size:
        raise ValueError(
            f"act_quant needs a last axis divisible by {block_size}, got {n}")
    leading = x.shape[:-1]
    grouped = x.astype(mx.float32).reshape(*leading, n // block_size, block_size)
    amax = mx.maximum(mx.max(mx.abs(grouped), axis=-1, keepdims=True), 1e-4)
    scale = amax / _FP8_MAX
    clamped = mx.clip(grouped / scale, -_FP8_MAX, _FP8_MAX)
    # Round-trip through the real E4M3 grid rather than approximating it.
    quantized = mx.from_fp8(mx.to_fp8(clamped), mx.float32)
    return (quantized * scale).reshape(*leading, n).astype(x.dtype)


def yarn_freqs(dim: int, seqlen: int, original_seq_len: int, base: float,
               factor: float, beta_fast: float, beta_slow: float):
    """Port of ``precompute_freqs_cis``: NTK-by-parts (YaRN) interpolation.

    Returns ``(cos, sin)`` of shape ``[seqlen, dim // 2]``. When
    ``original_seq_len`` is zero the correction is skipped entirely, which is
    how the released code disables YaRN on pure sliding-window layers -- those
    use the base theta instead of ``compress_rope_theta``.
    """
    import math

    freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    if original_seq_len and seqlen > original_seq_len:
        def correction_dim(rotations):
            return (dim * math.log(
                original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base)))

        low = math.floor(correction_dim(beta_fast))
        high = math.ceil(correction_dim(beta_slow))
        low, high = max(low, 0), min(high, dim - 1)
        ramp = mx.clip(
            (mx.arange(dim // 2).astype(mx.float32) - low)
            / max(high - low, 1e-3), 0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = mx.arange(seqlen).astype(mx.float32)[:, None] * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


def apply_rope_interleaved(x: mx.array, cos: mx.array, sin: mx.array,
                           inverse: bool = False) -> mx.array:
    """Rotary embedding over *interleaved* pairs, matching the released code.

    ``apply_rotary_emb`` views the last axis as complex via
    ``unflatten(-1, (-1, 2))``, so pairs are adjacent ``(x0, x1), (x2, x3)``.
    That is the "traditional" convention -- LFM2 and most of this runtime use
    the half-split form instead, and substituting one for the other rotates
    the wrong element pairs while keeping every shape and norm intact.

    ``inverse`` conjugates the rotation, which the attention epilogue uses to
    de-rotate its output before the projection.
    """
    shape = x.shape
    pairs = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // 2, 2)
    real, imaginary = pairs[..., 0], pairs[..., 1]
    if x.ndim == 4:  # [b, s, h, d] -> broadcast over heads
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    else:            # [b, s, d]
        cos, sin = cos[None], sin[None]
    if inverse:
        sin = -sin
    rotated = mx.stack(
        [real * cos - imaginary * sin, real * sin + imaginary * cos], axis=-1)
    return rotated.reshape(shape).astype(x.dtype)


# ---- attention output projection (F210) ------------------------------------


def attention_output_projection(o: mx.array, wo_a: mx.array, wo_b: mx.array,
                                *, n_groups: int, o_lora_rank: int
                                ) -> mx.array:
    """Grouped low-rank output projection.

    ``o`` is ``[b, s, heads, head_dim]``. Heads are split into ``n_groups``
    contiguous groups; each group has its own ``[o_lora_rank, group_width]``
    slice of ``wo_a``, and the concatenated per-group codes then pass through
    the shared ``wo_b``.

    ``wo_a`` is stored ``[n_groups * o_lora_rank, group_width]`` and must be
    viewed as ``[n_groups, o_lora_rank, group_width]`` -- group-major on the
    OUTPUT axis. Reshaping the other way pairs each group with another group's
    projection, which keeps every shape valid.

    The caller applies the inverse RoPE to ``o``'s rotary dimensions before
    this; the released code de-rotates the attention output before projecting.
    """
    b, s = o.shape[0], o.shape[1]
    grouped = o.reshape(b, s, n_groups, -1)
    per_group = wo_a.reshape(n_groups, o_lora_rank, grouped.shape[-1])
    codes = mx.einsum("bsgd,grd->bsgr", grouped.astype(mx.float32),
                      per_group.astype(mx.float32))
    return (codes.reshape(b, s, n_groups * o_lora_rank)
            @ wo_b.astype(mx.float32).T).astype(o.dtype)


# ---- MoE routing and experts (F211) ----------------------------------------


def moe_gate(x: mx.array, weight: mx.array, bias: mx.array | None, *,
             topk: int, score_func: str = "sqrtsoftplus",
             route_scale: float = 1.0):
    """Route each token to ``topk`` experts.

    ``bias`` shifts scores for *selection only*; the returned weights come from
    the unbiased scores. That is the noaux_tc pattern GLM and Kimi also use,
    and folding the bias into the weights is a silent quality regression rather
    than an error.

    The released default is ``sqrtsoftplus`` (``sqrt(softplus(logits))``), not
    softmax. Only the softmax variant skips the renormalization below, so a
    port that assumes softmax both computes the wrong scores and skips a
    division.
    """
    scores = x.astype(mx.float32) @ weight.astype(mx.float32).T
    if score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    elif score_func == "sqrtsoftplus":
        scores = mx.sqrt(mx.logaddexp(scores, mx.zeros_like(scores)))
    else:
        raise ValueError(f"unknown score_func {score_func!r}")

    original = scores
    if bias is not None:
        scores = scores + bias.astype(mx.float32)
    indices = mx.argpartition(-scores, kth=topk - 1, axis=-1)[..., :topk]
    weights = mx.take_along_axis(original, indices, axis=-1)
    if score_func != "softmax":
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights * route_scale, indices


def expert_swiglu(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array, *,
                  swiglu_limit: float = 0.0,
                  weights: mx.array | None = None) -> mx.array:
    """One expert's SwiGLU, with the released asymmetric clamp.

    ``swiglu_limit`` clamps the up branch on BOTH sides but the gate branch
    only from above (``torch.clamp(gate, max=limit)``). Clamping the gate
    symmetrically would suppress the negative tail that ``silu`` is there to
    pass, so the asymmetry is deliberate and reproduced exactly.
    """
    gate = (x.astype(mx.float32) @ w1.astype(mx.float32).T)
    up = (x.astype(mx.float32) @ w3.astype(mx.float32).T)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    activated = (gate * mx.sigmoid(gate)) * up
    if weights is not None:
        activated = weights * activated
    return activated @ w2.astype(mx.float32).T


def moe_combine(x: mx.array, routed, weights: mx.array, indices: mx.array,
                shared=None, *, n_routed_experts: int) -> mx.array:
    """Sum the routed experts' weighted outputs plus the shared expert.

    ``routed(expert_id, rows, scale)`` returns that expert's already-scaled
    output for the selected token rows, so the caller owns paging: only experts
    a token actually selected are ever materialized. Selection is resolved on
    the host because the engine needs the expert id list anyway to schedule
    weight fetches.

    A token may select the same expert only once (top-k over distinct ids), so
    each (row, expert) pair contributes exactly one weight.
    """
    flat = x.reshape(-1, x.shape[-1])
    rows_total = flat.shape[0]
    selected = indices.reshape(rows_total, -1).tolist()
    scales = weights.reshape(rows_total, -1)

    by_expert: dict[int, list[tuple[int, int]]] = {}
    for row, experts in enumerate(selected):
        for slot, expert in enumerate(experts):
            by_expert.setdefault(int(expert), []).append((row, slot))

    out = mx.zeros(flat.shape, dtype=mx.float32)
    for expert in sorted(by_expert):
        if not 0 <= expert < n_routed_experts:
            raise ValueError(f"routed to expert {expert} out of range")
        pairs = by_expert[expert]
        rows = mx.array([row for row, _ in pairs])
        slots = mx.array([slot for _, slot in pairs])
        scale = mx.take_along_axis(
            scales[rows], slots[:, None], axis=-1)
        contribution = routed(expert, flat[rows], scale).astype(mx.float32)
        out = out + mx.zeros(flat.shape, dtype=mx.float32).at[rows].add(
            contribution)
    if shared is not None:
        out = out + shared(flat).astype(mx.float32)
    return out.reshape(x.shape).astype(x.dtype)
