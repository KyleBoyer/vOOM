"""DSpark: DeepSeek V4's own multi-token draft, stored under ``mtp.*``.

The checkpoint ships three draft stages. Their layout follows the released
``DSparkBlock.__init__`` exactly, and reading it is how the stage roles were
established rather than guessed:

* stage 0 carries ``main_proj``/``main_norm`` (``stage_id == 0``). ``main_proj``
  is [dim, dim * len(dspark_target_layer_ids)] -- measured (4096, 12288) for
  the three target layers [40, 41, 42].
* stage 2 carries ``norm``, ``markov_head``, ``confidence_head`` and the
  ``hc_head_*`` parameters (``stage_id == n_stages - 1``).
* every stage carries a complete block: attention, MoE with 256 routed experts
  plus a shared expert, and both hyper-connection parameter sets.

Why this is worth having: decode is storage-bound, so a target sweep costs
essentially the same for one position as for six. Drafting runs three layers
(~0.7GB) against the target's ~8.3GB, about 8%, and proposes
``dspark_block_size`` (5) tokens that a single target sweep then verifies.

Two things differ from a target layer and both simplify it:

* ``DSparkAttention`` asserts ``compress_ratio == 0``, so there is no
  compressed region and no compressor state -- the gather is the window plus
  the block's own positions.
* within the block the gather is NOT causal: every drafted position attends to
  every other. That is deliberate (the block is filled with noise tokens and
  refined in one pass), and a causal mask here would be a silent divergence.
"""

from __future__ import annotations

import mlx.core as mx

from .deepseek_v4 import (apply_rope_interleaved, attention_output_projection,
                          hc_head, hc_post, hc_pre, sparse_windowed_attention)


def dspark_topk_idxs(window_size: int, block_size: int, start_pos: int
                     ) -> mx.array:
    """Gather list for one draft block, matching ``get_dspark_topk_idxs``.

    Window slots first, then the block's own positions offset by
    ``window_size``, because the caller concatenates [ring | block] before
    gathering. Every query row is identical -- see the module note on why this
    is intentionally not causal.
    """
    if start_pos <= 0:
        raise ValueError("DSpark drafting requires start_pos > 0")
    covered = min(window_size, start_pos + 1)
    row = mx.concatenate([mx.arange(covered),
                          window_size + mx.arange(block_size)])
    return mx.broadcast_to(row[None, None, :],
                           (1, block_size, row.shape[0])).astype(mx.int32)


def draft_input_ids(current_token: int, block_size: int, noise_token_id: int
                    ) -> mx.array:
    """[current, noise, noise, ...] -- the released ``forward_embed`` layout.

    Only the first slot carries a real token; the rest are the noise id the
    draft stages were trained to refine.
    """
    ids = [int(current_token)] + [int(noise_token_id)] * (block_size - 1)
    return mx.array(ids, dtype=mx.int32)[None]


def markov_bias(token_ids: mx.array, markov_w1: mx.array, markov_w2: mx.array
                ) -> tuple[mx.array, mx.array]:
    """Low-rank bigram correction: previous token -> rank -> vocabulary bias.

    Returns ``(bias, embedding)``; the embedding is what the confidence head
    consumes, so both are returned rather than recomputed.
    """
    embed = markov_w1[token_ids]
    return embed @ markov_w2.T, embed


def dspark_main_x(main_hidden: mx.array, main_proj: mx.array,
                  main_norm: mx.array, *, norm_eps: float) -> mx.array:
    """``main_norm(main_proj(main_hidden))`` for the concatenated target states.

    ``main_hidden`` is the target's hidden states at ``dspark_target_layer_ids``
    concatenated on the feature axis, in that order. Order matters and is not
    recoverable from shape alone, since every target layer has the same width.
    """
    projected = main_hidden.astype(mx.float32) @ main_proj.astype(mx.float32).T
    return mx.fast.rms_norm(projected.astype(main_hidden.dtype),
                            main_norm, norm_eps)


def dspark_attention(x: mx.array, main_x: mx.array, w: dict, prefix: str, *,
                     ring: mx.array, start_pos: int, heads: int, head_dim: int,
                     rope_head_dim: int, q_lora_rank: int, o_lora_rank: int,
                     n_groups: int, norm_eps: float, window: int,
                     cos: mx.array, sin: mx.array,
                     main_cos: mx.array, main_sin: mx.array
                     ) -> tuple[mx.array, mx.array]:
    """One draft stage's attention. Returns ``(output, updated_ring)``.

    ``main_x`` supplies the single real position's KV -- the released code
    writes it into slot ``start_pos % window`` -- while ``x`` supplies the
    block's own queries and KV. The block's KV is concatenated after the ring
    rather than written into it, so a rejected draft never touches ring state.
    """
    block = x.shape[1]
    main_kv = mx.fast.rms_norm(
        main_x.astype(x.dtype) @ w[f"{prefix}.wkv.weight"].T,
        w[f"{prefix}.kv_norm.weight"], norm_eps)
    main_tail = apply_rope_interleaved(
        main_kv[..., -rope_head_dim:], main_cos, main_sin)
    main_kv = mx.concatenate(
        [main_kv[..., :-rope_head_dim], main_tail], axis=-1)

    slot = start_pos % window
    ring = mx.concatenate(
        [ring[:, :slot], main_kv, ring[:, slot + 1:]], axis=1)

    q = mx.fast.rms_norm(
        x @ w[f"{prefix}.wq_a.weight"].T, w[f"{prefix}.q_norm.weight"],
        norm_eps) @ w[f"{prefix}.wq_b.weight"].T
    q = q.reshape(x.shape[0], block, heads, head_dim)
    q = q * mx.rsqrt(mx.mean(q.astype(mx.float32) ** 2, axis=-1,
                             keepdims=True) + norm_eps).astype(q.dtype)
    q_tail = apply_rope_interleaved(q[..., -rope_head_dim:], cos, sin)
    q = mx.concatenate([q[..., :-rope_head_dim], q_tail], axis=-1)

    kv = mx.fast.rms_norm(
        x @ w[f"{prefix}.wkv.weight"].T, w[f"{prefix}.kv_norm.weight"],
        norm_eps)
    kv_tail = apply_rope_interleaved(kv[..., -rope_head_dim:], cos, sin)
    kv = mx.concatenate([kv[..., :-rope_head_dim], kv_tail], axis=-1)

    gathered = mx.concatenate([ring, kv], axis=1)
    topk = dspark_topk_idxs(window, block, start_pos)
    out = sparse_windowed_attention(
        q, gathered, w[f"{prefix}.attn_sink"], topk,
        softmax_scale=head_dim ** -0.5)
    out_tail = apply_rope_interleaved(
        out[..., -rope_head_dim:], cos, sin, inverse=True)
    out = mx.concatenate([out[..., :-rope_head_dim], out_tail], axis=-1)
    projected = attention_output_projection(
        out, w[f"{prefix}.wo_a.weight"], w[f"{prefix}.wo_b.weight"],
        n_groups=n_groups, o_lora_rank=o_lora_rank)
    return projected, ring


def run_dspark_stage(x: mx.array, hc: dict, norms: dict, attention, ffn, *,
                     hc_mult: int, norm_eps: float, sinkhorn_iters: int,
                     hc_eps: float) -> mx.array:
    """One draft stage, identical in topology to a target block.

    Kept separate from ``run_deepseek_v4_block`` only because the attention
    signature differs (it takes ``main_x`` and returns an updated ring); the
    hyper-connection order is the same and is what F212 already pinned.
    """
    residual = x
    reduced, post, comb = hc_pre(x, hc["attn_fn"], hc["attn_scale"],
                                 hc["attn_base"], hc_mult=hc_mult,
                                 norm_eps=norm_eps,
                                 sinkhorn_iters=sinkhorn_iters, eps=hc_eps)
    x = hc_post(attention(mx.fast.rms_norm(reduced, norms["attn"], norm_eps)),
                residual, post, comb)

    residual = x
    reduced, post, comb = hc_pre(x, hc["ffn_fn"], hc["ffn_scale"],
                                 hc["ffn_base"], hc_mult=hc_mult,
                                 norm_eps=norm_eps,
                                 sinkhorn_iters=sinkhorn_iters, eps=hc_eps)
    return hc_post(ffn(mx.fast.rms_norm(reduced, norms["ffn"], norm_eps)),
                   residual, post, comb)


def dspark_sample_block(logits: mx.array, current_token: int,
                        markov_w1: mx.array, markov_w2: mx.array
                        ) -> tuple[list[int], mx.array]:
    """Greedy version of the released ``forward_head`` sampling loop.

    Sequential and not vectorizable: each position's Markov bias is a function
    of the token sampled at the previous position, so position i cannot be
    resolved before i-1. Returns the drafted ids and the stacked Markov
    embeddings the confidence head consumes.
    """
    block = logits.shape[1]
    previous = int(current_token)
    drafted: list[int] = []
    embeds = []
    for index in range(block):
        bias, embed = markov_bias(mx.array([previous], dtype=mx.int32),
                                  markov_w1, markov_w2)
        embeds.append(embed)
        adjusted = logits[:, index].astype(mx.float32) + bias.astype(mx.float32)
        previous = int(mx.argmax(adjusted, axis=-1).item())
        drafted.append(previous)
    return drafted, mx.stack(embeds, axis=1)


def accepted_prefix(drafted: list[int], target_tokens: list[int]) -> int:
    """How many drafted tokens the target confirms, longest prefix first.

    ``target_tokens[i]`` is the target's own choice at the position the draft
    filled with ``drafted[i]``. The first disagreement ends acceptance: the
    tokens after it were conditioned on a token the target rejected, so they
    are meaningless even if they happen to match.
    """
    accepted = 0
    for proposed, actual in zip(drafted, target_tokens):
        if proposed != actual:
            break
        accepted += 1
    return accepted
