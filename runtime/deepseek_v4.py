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
