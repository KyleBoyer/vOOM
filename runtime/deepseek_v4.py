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

from dataclasses import dataclass

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
                              topk_idxs: mx.array, softmax_scale: float,
                              tile: int = 128) -> mx.array:
    """Gathered sparse attention with per-head sinks.

    ``q``  is ``[b, s, h, d]``; ``kv`` is ``[b, n, d]`` shared across heads;
    ``topk_idxs`` is ``[b, s, topk]`` with ``-1`` for unused slots.

    Query positions are processed in tiles because the gathered operand is
    ``[b, tile, topk, d]``: materializing it for every position at once is
    ~108MB at a 271-token prompt and would be ~19GB at the 46K-token harness
    capture. The released implementation is a fused FlashAttention-style kernel
    that never materializes it at all; tiling is the same bound without a
    custom kernel. Tiling changes only the order in which independent rows are
    computed, so results are unchanged.
    """
    b, s, h, d = q.shape
    if tile <= 0 or s <= tile:
        return _sparse_windowed_attention_tile(
            q, kv, attn_sink, topk_idxs, softmax_scale)
    parts = [
        _sparse_windowed_attention_tile(
            q[:, start:start + tile], kv, attn_sink,
            topk_idxs[:, start:start + tile], softmax_scale)
        for start in range(0, s, tile)
    ]
    return mx.concatenate(parts, axis=1)


def _sparse_windowed_attention_tile(q, kv, attn_sink, topk_idxs,
                                    softmax_scale):
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    safe = mx.maximum(topk_idxs, 0).astype(mx.int32)

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

    # Project in the weights' own dtype and widen the result; upcasting the
    # weight matrices themselves allocates a full float32 copy per call.
    values = (x.astype(wkv.dtype) @ wkv.T).astype(mx.float32)
    scores = (x.astype(wgate.dtype) @ wgate.T).astype(mx.float32)
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
             route_scale: float = 1.0, hash_indices: mx.array | None = None):
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
    if hash_indices is not None:
        # The first num_hash_layers layers route by TOKEN ID: gate.tid2eid maps
        # each vocabulary entry to a fixed expert set, and the gate weights only
        # supply the mixing weights. Score-based top-k on these layers routes to
        # entirely different experts.
        indices = hash_indices.astype(mx.uint32)
    else:
        indices = mx.argpartition(-scores, kth=topk - 1, axis=-1)[..., :topk]
    weights = mx.take_along_axis(original, indices, axis=-1)
    if score_func != "softmax":
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights * route_scale, indices


def _packed_matmul(x: mx.array, w) -> mx.array:
    """x @ w.T for either a dense array or a packed MXFP4 expert.

    DeepSeek V4's routed experts ship as E2M1 FP4 codes with E8M0 group scales
    at group_size 32, which is exactly OCP MXFP4: the released bytes feed
    ``mx.quantized_matmul`` after a shape-only uint8 -> uint32 view, with no
    repacking and no value conversion. Verified against the dequantized
    reference on real layer-20 weights at rel 2.3e-07 (float32 rounding).

    This exists because dequantizing is the decode bottleneck, not I/O: a plain
    pread of expert tensors runs at 1.55GB/s while the same fetch with dequant
    runs at 0.51GB/s, and the dequant in isolation processes raw bytes at
    0.85GB/s. Serialized those predict 0.55GB/s, which is what the engine
    measured.
    """
    if isinstance(w, PackedExpert):
        # x stays float32: the dense branch below multiplies in float32, and
        # casting the activation to bfloat16 here cost three orders of
        # magnitude of agreement with the dequantized reference (rel 3.2e-3
        # against 2.3e-7) for no speed benefit.
        return mx.quantized_matmul(
            x.astype(mx.float32), w.codes, scales=w.scales, biases=None,
            transpose=True, group_size=w.group_size, bits=4,
            mode="mxfp4").astype(mx.float32)
    return x.astype(mx.float32) @ w.astype(mx.float32).T


@dataclass
class PackedExpert:
    """A routed expert left in its released MXFP4 form.

    ``codes`` is the checkpoint's own packed nibble stream viewed as uint32;
    ``scales`` its E8M0 bytes, unmodified. Nothing is repacked, so this is a
    representation change only -- but the fused kernel reassociates the
    float32 sums a dequantized matmul would perform in a different order, so
    equivalence is proven by greedy-token equality rather than bit identity.
    """

    codes: mx.array
    scales: mx.array
    group_size: int = 32

    @property
    def nbytes(self) -> int:
        return self.codes.nbytes + self.scales.nbytes

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self.codes.shape[:-1], self.codes.shape[-1] * 8)


def expert_swiglu(x: mx.array, w1, w2, w3, *,
                  swiglu_limit: float = 0.0,
                  weights: mx.array | None = None) -> mx.array:
    """One expert's SwiGLU, with the released asymmetric clamp.

    ``swiglu_limit`` clamps the up branch on BOTH sides but the gate branch
    only from above (``torch.clamp(gate, max=limit)``). Clamping the gate
    symmetrically would suppress the negative tail that ``silu`` is there to
    pass, so the asymmetry is deliberate and reproduced exactly.

    Each weight is either a dense array or a ``PackedExpert``; the two are
    interchangeable here and produce the same result to float rounding.
    """
    gate = _packed_matmul(x, w1)
    up = _packed_matmul(x, w3)
    if swiglu_limit > 0:
        up = mx.clip(up, -swiglu_limit, swiglu_limit)
        gate = mx.minimum(gate, swiglu_limit)
    activated = (gate * mx.sigmoid(gate)) * up
    if weights is not None:
        activated = weights * activated
    return _packed_matmul(activated, w2)


def moe_combine(x: mx.array, routed, weights: mx.array, indices: mx.array,
                shared=None, *, n_routed_experts: int,
                only_experts: set | None = None) -> mx.array:
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
        if only_experts is not None and expert not in only_experts:
            # Bounded fetch: this call handles one group of the routed union.
            continue
        if not 0 <= expert < n_routed_experts:
            raise ValueError(f"routed to expert {expert} out of range")
        pairs = by_expert[expert]
        rows = mx.array([row for row, _ in pairs])
        slots = mx.array([slot for _, slot in pairs])
        scale = mx.take_along_axis(
            scales[rows], slots[:, None], axis=-1)
        contribution = routed(expert, flat[rows], scale).astype(mx.float32)
        # Scatter straight into the accumulator. Materializing a full-size
        # zeros buffer per expert made the measured layer transient ~6.6GB on
        # the real model -- with a ~150-expert routed union that is 150 live
        # [tokens, hidden] float32 temporaries, and the governor refused the
        # request outright.
        out = out.at[rows].add(contribution)
    if shared is not None:
        out = out + shared(flat).astype(mx.float32)
    return out.reshape(x.shape).astype(x.dtype)


# ---- block assembly (F212) -------------------------------------------------


def run_deepseek_v4_block(x: mx.array, hc: dict, norms: dict, attention,
                          ffn, *, hc_mult: int, norm_eps: float,
                          sinkhorn_iters: int, hc_eps: float) -> mx.array:
    """One decoder block over the hyper-connection stream.

    ``x`` is ``[b, s, hc_mult, dim]`` throughout -- the stream never collapses
    between blocks. Each half reduces it, norms, runs its sublayer on the
    single reduced ``[b, s, dim]`` tensor, then re-expands.

    Three orderings this gets right and a plausible rewrite does not:

    * ``residual`` is captured BEFORE ``hc_pre``, so ``hc_post`` mixes the
      original streams, not the reduced tensor;
    * the attention half uses the ``attn`` HC parameters and the FFN half the
      ``ffn`` ones -- they are separate learned projections, and swapping them
      is shape-compatible;
    * the norm is applied to the REDUCED tensor, after ``hc_pre``, not to the
      stream before it.

    ``attention`` and ``ffn`` are callables taking and returning
    ``[b, s, dim]``, so the caller supplies paging-aware implementations and
    this function stays pure topology.
    """
    common = dict(hc_mult=hc_mult, norm_eps=norm_eps,
                  sinkhorn_iters=sinkhorn_iters, eps=hc_eps)

    residual = x
    reduced, post, comb = hc_pre(
        x, hc["attn_fn"], hc["attn_scale"], hc["attn_base"], **common)
    reduced = mx.fast.rms_norm(reduced, norms["attn"], norm_eps)
    x = hc_post(attention(reduced), residual, post, comb)

    residual = x
    reduced, post, comb = hc_pre(
        x, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"], **common)
    reduced = mx.fast.rms_norm(reduced, norms["ffn"], norm_eps)
    return hc_post(ffn(reduced), residual, post, comb)


def deepseek_v4_attention(x: mx.array, w: dict, prefix: str, *,
                          heads: int, head_dim: int, rope_head_dim: int,
                          q_lora_rank: int, o_lora_rank: int, n_groups: int,
                          norm_eps: float, cos: mx.array, sin: mx.array,
                          kv_all: mx.array, topk_idxs: mx.array,
                          act_quant_block: int = 64) -> mx.array:
    """Compose the attention halves verified in F206/F209/F210.

    Order follows the released ``Attention.forward``: q through its LoRA and
    norm, per-head RMS *without* a learned weight, RoPE on the rotary tail
    only, gathered sparse attention, then the INVERSE RoPE on the output before
    the grouped projection. ``kv_all`` and ``topk_idxs`` come from the caller,
    which owns the window/compressed cache.
    """
    b, s, _ = x.shape
    q = x @ w[f"{prefix}.wq_a"].T
    q = mx.fast.rms_norm(q, w[f"{prefix}.q_norm"], norm_eps)
    q = (q @ w[f"{prefix}.wq_b"].T).reshape(b, s, heads, head_dim)
    # A weightless RMS over each head, distinct from the learned q_norm above.
    q = q * mx.rsqrt(mx.mean(mx.square(q.astype(mx.float32)), axis=-1,
                             keepdims=True) + norm_eps).astype(q.dtype)
    tail = apply_rope_interleaved(q[..., -rope_head_dim:], cos, sin)
    q = mx.concatenate([q[..., :-rope_head_dim], tail], axis=-1)

    out = sparse_windowed_attention(
        q, kv_all, w[f"{prefix}.attn_sink"], topk_idxs,
        float(head_dim) ** -0.5)

    # De-rotate before projecting -- the released epilogue's inverse=True.
    tail = apply_rope_interleaved(
        out[..., -rope_head_dim:], cos, sin, inverse=True)
    out = mx.concatenate([out[..., :-rope_head_dim], tail], axis=-1)
    return attention_output_projection(
        out, w[f"{prefix}.wo_a"], w[f"{prefix}.wo_b"],
        n_groups=n_groups, o_lora_rank=o_lora_rank)


# ---- window ring buffer + compressed cache (F214) --------------------------
#
# The released Attention keeps ONE buffer per layer: the first ``window_size``
# slots are a ring over the most recent positions, and everything after them is
# the compressed region. Both are addressed by the same gather list, which is
# why ``get_compress_topk_idxs`` shifts compressed indices by an offset.


def window_ring_write(ring: mx.array, kv: mx.array, start_pos: int,
                      window: int) -> mx.array:
    """Place ``kv`` into the ring exactly as the released cache does.

    Prefill writes the last ``window`` positions but *rotated*: with
    ``cutoff = seqlen % window`` the tail lands at ``[cutoff:window]`` and the
    head wraps to ``[:cutoff]``, so slot ``p % window`` always holds position
    ``p``. Writing them contiguously instead would put every subsequent decode
    step's ring index off by ``cutoff``.
    """
    seqlen = kv.shape[1]
    if start_pos == 0:
        if seqlen <= window:
            return mx.concatenate(
                [kv, ring[:, seqlen:]], axis=1) if seqlen < window else kv
        cutoff = seqlen % window
        tail = kv[:, -window:]
        if cutoff == 0:
            return tail
        return mx.concatenate([tail[:, window - cutoff:],
                               tail[:, :window - cutoff]], axis=1)
    slot = start_pos % window
    return mx.concatenate(
        [ring[:, :slot], kv[:, :1], ring[:, slot + 1:]], axis=1)


def compress_topk_idxs(ratio: int, seqlen: int, start_pos: int, offset: int
                       ) -> mx.array:
    """Compressed-region gather list, matching ``get_compress_topk_idxs``.

    Entries are shifted by ``offset`` because the compressed region shares one
    buffer with the window ring. A position may only read compressed entry
    ``< (p + 1) // ratio``; unreachable slots are ``-1``.
    """
    if start_pos > 0:
        return (mx.arange((start_pos + 1) // ratio)
                + offset).astype(mx.int32)[None, None]
    entries = seqlen // ratio
    columns = mx.broadcast_to(mx.arange(entries)[None, :], (seqlen, entries))
    reach = (mx.arange(1, seqlen + 1) // ratio)[:, None]
    return mx.where(columns >= reach, -1,
                    columns + offset).astype(mx.int32)[None]


def gather_indices(window: int, ratio: int, seqlen: int, start_pos: int,
                   compressed_offset: int) -> mx.array:
    """Concatenate the window and compressed gather lists for one layer.

    ``ratio == 0`` means the layer has no compressed region at all (the
    released ``compress_ratios`` contains 0, 4 and 128), in which case the
    window list is the whole gather.
    """
    windowed = window_topk_idxs(window, seqlen, start_pos)
    if not ratio:
        return windowed
    compressed = compress_topk_idxs(ratio, seqlen, start_pos,
                                    compressed_offset)
    if compressed.shape[1] != windowed.shape[1]:
        compressed = mx.broadcast_to(
            compressed, (compressed.shape[0], windowed.shape[1],
                         compressed.shape[2]))
    return mx.concatenate([windowed, compressed], axis=-1)


# ---- compressor decode state (F215) ----------------------------------------


class CompressorState:
    """Per-layer buffers that carry partial groups across decode steps.

    Prefill leaves ``seqlen % ratio`` trailing positions uncompressed; decode
    fills the group one position at a time and emits a compressed entry only on
    the step that completes it (``(start_pos + 1) % ratio == 0``).

    At ratio 4 the released module keeps ``2 * ratio`` slots: the first ``ratio``
    hold the previous group's overlap half and the second ``ratio`` the current
    group. On emit it takes the overlap half's first ``head_dim`` features and
    the current half's second ``head_dim``, then slides the current group down.
    Scores are initialized to ``-inf`` so an unfilled slot contributes nothing
    through the softmax.
    """

    def __init__(self, ratio: int, head_dim: int, batch: int = 1,
                 dtype=mx.float32):
        self.ratio = int(ratio)
        self.head_dim = int(head_dim)
        self.overlap = self.ratio == 4
        coff = 2 if self.overlap else 1
        slots = coff * self.ratio
        self.kv_state = mx.zeros((batch, slots, coff * head_dim), dtype=dtype)
        self.score_state = mx.full(
            (batch, slots, coff * head_dim), -mx.inf, dtype=dtype)

    def _write(self, slot: int, kv_row: mx.array, score_row: mx.array) -> None:
        self.kv_state = mx.concatenate([
            self.kv_state[:, :slot], kv_row[:, None],
            self.kv_state[:, slot + 1:]], axis=1)
        self.score_state = mx.concatenate([
            self.score_state[:, :slot], score_row[:, None],
            self.score_state[:, slot + 1:]], axis=1)
        # Materialize before returning. These buffers are tiny -- [b, 8, 2048]
        # float32 -- but each write concatenates around the PREVIOUS state, so
        # left lazy they chain one graph link per decode step per layer, and
        # every link pins its inputs: the projections that produced kv_row and
        # score_row, and through them the compressor's own [1024, 4096] wkv and
        # wgate matrices. That is 16.8MB per compressed layer per step, about
        # 0.69GB per token across 41 layers, held for the life of the request
        # and invisible to any Python-side accounting because the retained
        # arrays have no Python wrapper. Measured: 0.52GB/token of otherwise
        # unattributable growth, gone once these two lines evaluate.
        mx.eval(self.kv_state, self.score_state)

    def step(self, kv: mx.array, score: mx.array, start_pos: int,
             ape: mx.array):
        """Absorb one position; return a compressed entry or ``None``.

        ``kv``/``score`` are ``[b, 1, coff * head_dim]`` projections of this
        position's hidden state, before the position embedding is added.
        """
        ratio = self.ratio
        score = score + ape[start_pos % ratio].astype(score.dtype)
        slot = (ratio if self.overlap else 0) + start_pos % ratio
        self._write(slot, kv[:, 0], score[:, 0])

        if (start_pos + 1) % ratio:
            return None

        head_dim = self.head_dim
        if self.overlap:
            values = mx.concatenate(
                [self.kv_state[:, :ratio, :head_dim],
                 self.kv_state[:, ratio:, head_dim:]], axis=1)
            scores = mx.concatenate(
                [self.score_state[:, :ratio, :head_dim],
                 self.score_state[:, ratio:, head_dim:]], axis=1)
        else:
            values, scores = self.kv_state, self.score_state

        pooled = mx.sum(
            values * mx.softmax(scores, axis=1), axis=1, keepdims=True)

        if self.overlap:
            # Slide the completed group into the overlap half for the next one.
            self.kv_state = mx.concatenate(
                [self.kv_state[:, ratio:], self.kv_state[:, ratio:]], axis=1)
            self.score_state = mx.concatenate(
                [self.score_state[:, ratio:], self.score_state[:, ratio:]],
                axis=1)
        return pooled
