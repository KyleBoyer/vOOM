"""GLM-5.2 (GlmMoeDsaForCausalLM) block math — the GOAL model.

Architecture (from the real checkpoint, 2026-07-10):
- MLA attention (DeepSeek-style): q = q_b(rms(q_a(h))) with q_lora 2048;
  kv_a_proj_with_mqa -> [c_kv 512 | k_rope 64]; kv_b(rms(c_kv)) -> per-head
  [k_nope 192 | v 256]. 64 heads, qk dim 256 (192 nope + 64 rope), v dim 256.
  RoPE theta 8e6, INTERLEAVED (traditional=True), applied to the 64-dim slice;
  k_rope is single-headed (MQA) and broadcast across heads.
- DSA sparse attention: an indexer picks top-2048 positions per decode query. For
  contexts <= index_topk (2048) every position is selected, so dense attention is
  mathematically exact. Decode-time DSA state/selection is wired for longer
  contexts, including IndexShare reuse, but sparse L>1 prefill and the released-
  implementation oracle/>2048 end-to-end gate are still open (F22/F33).
- FFN: first 3 layers dense SwiGLU (inter 12288). MoE layers: 256 routed experts
  (per-expert tensors — expert paging applies unchanged), top-8 by sigmoid score
  with e_score_correction_bias added FOR SELECTION ONLY (noaux_tc), weights =
  normalized unbiased scores * routed_scaling_factor (2.5), plus 1 always-on
  shared expert (inter 2048).
- MTP: num_nextn_predict_layers=1, applied iteratively for multi-token drafts and
  wired into runtime/speculative.py. State repairs are in place; strict target-only
  token A/B and per-round expert-byte telemetry remain open (F23/F32).

KV note: the ordinary path caches full per-head K/V (~65 KB/token/layer). The
implemented compressed-MLA mode caches c_kv+k_rope (~1.2 KB/token/layer, ~49x
smaller) and re-expands selected latents; exact closed-page spill and absorbed
decode remain F07/F08/F34 work.
"""

from __future__ import annotations

import mlx.core as mx

from . import quant
from .config import ModelConfig
from .expert_batching import consume_expert_batches
from .layer_runner import _linear, _swiglu


def _mla_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int
) -> mx.array:
    B, L, _ = h.shape
    n_h = cfg.num_attention_heads
    dn, dr, dv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim

    if cfg.q_lora_rank:
        q_a = _linear(h, w, f"{prefix}.self_attn.q_a_proj")
        q_a = mx.fast.rms_norm(
            q_a,
            w[f"{prefix}.self_attn.q_a_layernorm.weight"],
            cfg.mla_latent_norm_eps,
        )
        q = _linear(q_a, w, f"{prefix}.self_attn.q_b_proj")
    else:
        # F92: Kimi Linear's MLA layers have q_lora_rank=null -- no Q
        # low-rank compression, a single q_proj straight from hidden_states.
        # Everything downstream (nope/rope split, RoPE, concat) is identical.
        q = _linear(h, w, f"{prefix}.self_attn.q_proj")
    q = q.reshape(B, L, n_h, dn + dr).transpose(0, 2, 1, 3)
    q_nope, q_rope = q[..., :dn], q[..., dn:]

    kv_a = _linear(h, w, f"{prefix}.self_attn.kv_a_proj_with_mqa")
    c_kv, k_rope = kv_a[..., : -dr], kv_a[..., -dr:]
    c_kv = mx.fast.rms_norm(
        c_kv,
        w[f"{prefix}.self_attn.kv_a_layernorm.weight"],
        cfg.mla_latent_norm_eps,
    )

    k_rope = k_rope.reshape(B, L, 1, dr).transpose(0, 2, 1, 3)  # single MQA rope head
    if not cfg.mla_use_nope:
        q_rope = mx.fast.rope(q_rope, dr, traditional=cfg.rope_interleave, base=cfg.rope_theta,
                              scale=1.0, offset=offset)
        k_rope = mx.fast.rope(k_rope, dr, traditional=cfg.rope_interleave, base=cfg.rope_theta,
                              scale=1.0, offset=offset)
    # else: F92 -- Kimi Linear's MLA is NoPE, the "rope" head-dim split is
    # carried through unrotated; position info comes only from KDA layers.
    queries = mx.concatenate([q_nope, q_rope], axis=-1)

    if getattr(kv, "compressed_mla", False):
        # F21: cache only [c_kv | roped k_rope] = 576 floats/token-layer (~50x
        # less RAM) and re-expand ALL cached positions through kv_b each call.
        # The cached latent bytes are exact, but re-expanding a prior row inside
        # a different-S GEMM can select a different Metal kernel/reduction shape.
        # The real probe measured max decode activation delta 0.000244, so this
        # execution path is E evidence—not structural L0—until F87 exact replay
        # residuals or an equivalent released-arithmetic mechanism close it.
        lat = mx.concatenate([c_kv, k_rope.transpose(0, 2, 1, 3).reshape(B, L, dr)], axis=-1)
        lat_all = kv.update_latent(layer, lat)  # (B, S, 512+dr)
        S = lat_all.shape[1]

        # F22 DSA: beyond index_topk cached positions, the indexer selects which
        # latents to expand and attend over (decode path, L=1). Below the
        # threshold every position is selected -> dense path is exact.
        dsa = getattr(kv, "dsa", None)
        itype = (cfg.indexer_types[layer] if cfg.indexer_types and layer < len(cfg.indexer_types)
                 else "shared")
        if dsa is not None and cfg.index_topk:
            dsa.observe(layer, itype, h, w, prefix, offset)  # k-cache accumulates always, roped at absolute positions
        if dsa is not None and cfg.index_topk and L == 1 and S > cfg.index_topk:
            sel = dsa.update_and_select(layer, itype, h, q_a, w, prefix, offset)
            if sel is not None:  # sel: (B, 1, topk) — gather selected latent rows
                lat_all = mx.take(lat_all[0], sel[0, 0], axis=0)[None]
                S = lat_all.shape[1]

        c_all, kr_all = lat_all[..., :-dr], lat_all[..., -dr:]

        if L == 1 and getattr(kv, "mla_absorbed", False):
            # F34: MLA weight absorption (DeepSeek-V3.2 decode path,
            # https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py).
            # The naive path above expands ALL S selected latents through
            # kv_b_proj — an (S x kv_lora x n_h*(dn+dv)) GEMM — to get
            # per-head K/V, purely so attention can score/sum over them.
            # Matrix associativity lets us skip that expansion entirely:
            # fold kv_b_proj's per-head K up-projection into q_nope (so the
            # score is computed DIRECTLY against the compact latent, no k_nope
            # ever materialized), and its V up-projection into the FINAL
            # output (so the attention-weighted SUM happens in latent space,
            # projected to full width only once, not once per cached
            # position). Floating-point association changes vs the naive
            # path (different summation order) — greedy-token-identity is
            # the gate (tests/test_mla_absorbed.py), not bit-identical
            # logits, matching this doc's own note that reassociation is
            # expected here.
            kv_lora = c_all.shape[-1]
            w_kvb = w[f"{prefix}.self_attn.kv_b_proj.weight"].reshape(n_h, dn + dv, kv_lora)
            w_uk, w_uv = w_kvb[:, :dn, :], w_kvb[:, dn:, :]

            q_nope_abs = mx.einsum("bhld,hdc->bhlc", q_nope, w_uk)      # (B,n_h,1,kv_lora)
            score_nope = mx.einsum("bhlc,bsc->bhls", q_nope_abs, c_all)  # (B,n_h,1,S)
            score_rope = mx.einsum("bhld,bsd->bhls", q_rope, kr_all)     # (B,n_h,1,S)
            scale = (dn + dr) ** -0.5
            scores = (score_nope + score_rope).astype(mx.float32) * scale
            attn_w = mx.softmax(scores, axis=-1).astype(q_nope.dtype)

            weighted_c = mx.einsum("bhls,bsc->bhlc", attn_w, c_all)     # (B,n_h,1,kv_lora)
            out = mx.einsum("bhlc,hdc->bhld", weighted_c, w_uv)          # (B,n_h,1,dv)
            attn = out.transpose(0, 2, 1, 3).reshape(B, L, n_h * dv)
            return _linear(attn, w, f"{prefix}.self_attn.o_proj")

        kvb = _linear(c_all, w, f"{prefix}.self_attn.kv_b_proj").reshape(B, S, n_h, dn + dv).transpose(0, 2, 1, 3)
        k_nope, values = kvb[..., :dn], kvb[..., dn:]
        keys = mx.concatenate(
            [k_nope, mx.broadcast_to(kr_all.reshape(B, 1, S, dr), (B, n_h, S, dr))], axis=-1
        )
    else:
        kvb = _linear(c_kv, w, f"{prefix}.self_attn.kv_b_proj").reshape(B, L, n_h, dn + dv).transpose(0, 2, 1, 3)
        k_nope, v = kvb[..., :dn], kvb[..., dn:]
        keys = mx.concatenate([k_nope, mx.broadcast_to(k_rope, (B, n_h, L, dr))], axis=-1)
        keys, values = kv.update(layer, keys, v)

    mask = None
    if L > 1:
        q_pos = mx.arange(offset, offset + L)[:, None]
        k_pos = mx.arange(keys.shape[2])[None, :]
        mask = mx.where(k_pos <= q_pos, 0.0, float("-inf")).astype(queries.dtype)

    attn = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=(dn + dr) ** -0.5, mask=mask
    )
    attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * dv)
    return _linear(attn, w, f"{prefix}.self_attn.o_proj")


def _route_experts(h: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> tuple[mx.array, mx.array]:
    """noaux_tc sigmoid router: bias affects WHICH experts win, not their weights.

    Released Transformers computes F.linear(float32(h), float32(weight)).
    Casting only the RESULT (an earlier version of this code) performs a BF16
    GEMM first and can change the discontinuous top-k routing decision near a
    boundary. Extracted from run_glm_block so an oracle test can call the
    exact production routing math directly (tests/test_f33_router_oracle.py)
    without needing a full block (attention + expert fetch) around it.

    Real GLM-5.2 has n_group=topk_group=1, so the released group-restricted
    top-k is a no-op for this model (every expert is in the single group that
    always wins) -- this function only implements the flat top-k, not general
    group restriction; it is not correct for a config with n_group > 1.
    """
    gate_weight = w[f"{prefix}.mlp.gate.weight"]
    if isinstance(gate_weight, quant.QTensor):
        router_logits = quant.matmul(h.astype(mx.float32), gate_weight)
    else:
        router_logits = h.astype(mx.float32) @ gate_weight.astype(mx.float32).T
    scores = mx.sigmoid(router_logits)
    biased = scores + w[f"{prefix}.mlp.gate.e_score_correction_bias"]
    k = cfg.num_experts_per_tok
    idx = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    pw = mx.take_along_axis(scores, idx, axis=-1)
    if cfg.norm_topk_prob:
        pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * cfg.routed_scaling_factor
    return idx, pw


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


def run_glm_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
    get_experts, mlp_last_only: bool = False, iter_expert_batches=None,
) -> mx.array:
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    x = x + _mla_attention(h, w, prefix, cfg, kv, layer, offset)
    if mlp_last_only:  # F36: KV is built; only the last position feeds the logits
        x = x[:, -1:, :]

    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    is_dense = (
        cfg.mlp_layer_types[layer] == "dense"
        if layer < len(cfg.mlp_layer_types)
        else layer < cfg.first_k_dense_replace
    )
    if is_dense:
        return x + _swiglu(h, w, f"{prefix}.mlp")

    idx, pw = _route_experts(h, w, prefix, cfg)
    mx.eval(idx, pw)
    groups = _group_routes(idx, pw)

    # HF accumulates routed contributions into zeros in ascending expert order,
    # then adds the shared expert. Starting from the shared value changes BF16
    # association and is not the released arithmetic order.
    out = mx.zeros_like(h)
    expert_ids = sorted(groups)  # HF iterates expert_hit.nonzero(): ascending expert id
    positions_by_expert = {e: [pt for pt, _ in groups[e]] for e in expert_ids}
    if iter_expert_batches is None:
        experts = get_experts(layer, expert_ids, positions=positions_by_expert)
        batches = ((expert_ids, experts),)
    else:
        # F74-v2: the fetch lifetime and compute lifetime are the SAME bounded
        # batch. The earlier cache-only sub-batching still returned one mapping
        # containing the complete union, so evicted arrays remained strongly
        # referenced until this function returned and memory was not bounded.
        batches = iter_expert_batches(layer, expert_ids, positions=positions_by_expert)

    def consume_batch(batch_ids, experts):
        nonlocal out
        for e in batch_ids:
            plist = groups[e]
            positions = [p for p, _ in plist]
            # Released path retains FP32 router weights, multiplies the expert
            # output in FP32, then casts the contribution back to the residual
            # dtype immediately before index_add_.
            route_weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
            y = _swiglu(h[:, positions, :], experts[e], f"{prefix}.mlp.experts.{e}")
            contribution = (y * route_weights[None, :, None]).astype(h.dtype)
            out = out.at[:, positions, :].add(contribution)
        # Materialize before the iterator fetches the next batch. This severs the
        # lazy graph's references to the current expert tensors; the cache may now
        # evict/release them without the returned union pinning every prior page.
        mx.eval(out)
        del contribution, y, route_weights

    consume_expert_batches(batches, consume_batch)
    out = out + _swiglu(h, w, f"{prefix}.mlp.shared_experts")
    return x + out
