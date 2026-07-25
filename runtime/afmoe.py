"""Afmoe (Arcee AI's Trinity Nano/Mini) block math.

Real reference: `models/Trinity-Nano-Preview/modeling_afmoe.py`
(`AfmoeForCausalLM`). Genuinely new architecture family for this project --
NOT MLA like GLM/Kimi:

- Ordinary GQA attention (`num_attention_heads` query heads,
  `num_key_value_heads` KV heads, repeated to match), per-head Q/K RMSNorm
  applied BEFORE RoPE, a sigmoid output gate (`self_attn.gate_proj`,
  computed from the SAME pre-attention hidden state, multiplied into the
  attention output before `o_proj`).
- RoPE is applied ONLY to "sliding_attention" (local) layers -- the real
  code's `AfmoeAttention.forward` gates `apply_rotary_pos_emb` on
  `self.is_local_attention`. "full_attention" layers get NO position
  encoding at all (NoPE), matching this project's existing `mla_use_nope`
  precedent (Kimi Linear) in spirit, though this is a different
  architecture entirely.
- Sliding-window causal masking on local layers (`cfg.sliding_window`),
  ordinary causal masking on full layers -- reuses the exact masking
  approach `runtime/gptoss.py::_attention_gptoss` already implements for
  gpt-oss's own alternating local/global layers (decode fast-path slicing
  + prefill masked scores), adapted here without attention sinks (afmoe
  has none).
- "Sandwich"/dual RMSNorm per layer: `input_layernorm` (pre-attn),
  `post_attention_layernorm` (POST-attn, before the residual add),
  `pre_mlp_layernorm`, `post_mlp_layernorm` (POST-mlp, before the residual
  add) -- 4 norms per layer, not GLM/Qwen's usual 2.
- MoE gate: sigmoid scoring + a REAL persisted `mlp.expert_bias` tensor
  used for top-k SELECTION only (top_scores gathered from the unbiased
  `scores`, not the biased ones) -- the same noaux_tc-style
  bias-affects-selection-not-weight pattern GLM/Kimi's `_route_experts`
  already implements, just under different weight names
  (`mlp.router.gate.weight` vs GLM's `mlp.gate.weight`, `mlp.expert_bias`
  vs GLM's `mlp.gate.e_score_correction_bias`) -- hence a local duplicate
  here rather than reusing `runtime.glm._route_experts` directly, same
  precedent as `kimi_linear.py`'s own local `_route_experts`.
- MuP (maximal update parametrization): `hidden_states = hidden_states *
  hidden_size**0.5` applied ONCE at the embedding, before any layer runs
  (`runtime/engine.py::_embed` applies this when `cfg.mup_enabled`, not
  handled in this file).

Numeric oracle: real `modeling_afmoe.py` + torch, calling `AfmoeDecoderLayer.
forward()` directly with manually-built position_embeddings/attention_mask
(bypassing `AfmoeModel.forward()`'s top-level HF utility glue, which hits
several real transformers-version-skew bugs unrelated to this architecture
-- see tests/test_afmoe_oracle.py's module docstring for what was patched
and why).
"""

from __future__ import annotations

import mlx.core as mx

from .config import ModelConfig
from .expert_batching import consume_expert_batches
from .glm import _group_routes
from .layer_runner import _linear, _swiglu


def _afmoe_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
) -> mx.array:
    B, L, _ = h.shape
    n_h, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    is_sliding = bool(cfg.layer_types) and cfg.layer_types[layer] == "sliding_attention"

    q = _linear(h, w, f"{prefix}.self_attn.q_proj").reshape(B, L, n_h, hd)
    k = _linear(h, w, f"{prefix}.self_attn.k_proj").reshape(B, L, n_kv, hd)
    v = _linear(h, w, f"{prefix}.self_attn.v_proj").reshape(B, L, n_kv, hd)
    gate = _linear(h, w, f"{prefix}.self_attn.gate_proj")

    q = mx.fast.rms_norm(q, w[f"{prefix}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k = mx.fast.rms_norm(k, w[f"{prefix}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    if is_sliding:
        # Real reference RoPE convention (rotate_half / GPT-NeoX style,
        # non-interleaved) -- only sliding/local layers get position info.
        q = mx.fast.rope(q, hd, traditional=False, base=cfg.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, hd, traditional=False, base=cfg.rope_theta, scale=1.0, offset=offset)

    keys, values = kv.update(layer, k, v)
    if is_sliding and L == 1 and keys.shape[2] > cfg.sliding_window:
        # Decode fast path (mirrors gptoss.py's own precedent exactly): a
        # sliding layer only ever attends to the last `sliding_window` keys.
        keys = keys[:, :, -cfg.sliding_window:, :]
        values = values[:, :, -cfg.sliding_window:, :]
    S = keys.shape[2]
    rep = n_h // n_kv
    if rep > 1:
        keys = mx.repeat(keys, rep, axis=1)
        values = mx.repeat(values, rep, axis=1)

    mask = None
    if L > 1:
        q_pos = mx.arange(offset, offset + L)[:, None]
        k_pos = mx.arange(S)[None, :]
        allowed = k_pos <= q_pos
        if is_sliding:
            allowed = allowed & (k_pos > q_pos - cfg.sliding_window)
        mask = mx.where(allowed[None, None], 0.0, float("-inf")).astype(q.dtype)

    attn = mx.fast.scaled_dot_product_attention(q, keys, values, scale=hd ** -0.5, mask=mask)
    attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * hd)
    attn = attn * mx.sigmoid(gate)
    return _linear(attn, w, f"{prefix}.self_attn.o_proj")


def _afmoe_route_experts(h: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> tuple[mx.array, mx.array]:
    """Afmoe's sigmoid token-choice router. NOT the same weight names as
    runtime.glm._route_experts, but the same noaux_tc-style math: the real
    persisted `mlp.expert_bias` affects which experts win top-k, not the
    routing WEIGHT applied to their output (real `AfmoeTokenChoiceRouter.
    forward`: `top_scores = scores.gather(...)` reads the unbiased `scores`,
    `selected_experts` comes from `topk(scores + expert_bias)`)."""
    gate_weight = w[f"{prefix}.mlp.router.gate.weight"]
    scores = mx.sigmoid((h.astype(mx.float32) @ gate_weight.astype(mx.float32).T))
    bias = w[f"{prefix}.mlp.expert_bias"].astype(mx.float32)
    k = cfg.num_experts_per_tok
    idx = mx.argpartition(-(scores + bias), kth=k - 1, axis=-1)[..., :k]
    pw = mx.take_along_axis(scores, idx, axis=-1)
    if cfg.norm_topk_prob:
        pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * cfg.routed_scaling_factor
    return idx, pw


def run_afmoe_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv, layer: int, offset: int,
    get_experts, mlp_last_only: bool = False, iter_expert_batches=None,
) -> mx.array:
    residual = x
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    h = _afmoe_attention(h, w, prefix, cfg, kv, layer, offset)
    h = mx.fast.rms_norm(h, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    x = residual + h
    if mlp_last_only:  # F36: KV is built; only the last position feeds the logits
        x = x[:, -1:, :]

    residual = x
    h = mx.fast.rms_norm(x, w[f"{prefix}.pre_mlp_layernorm.weight"], cfg.rms_norm_eps)

    is_dense = layer < cfg.first_k_dense_replace
    if is_dense:
        h = _swiglu(h, w, f"{prefix}.mlp")
    else:
        idx, pw = _afmoe_route_experts(h, w, prefix, cfg)
        mx.eval(idx, pw)
        groups = _group_routes(idx, pw)
        out = mx.zeros_like(h)
        expert_ids = sorted(groups)
        positions_by_expert = {e: [p for p, _ in groups[e]] for e in expert_ids}
        if iter_expert_batches is None:
            experts = get_experts(layer, expert_ids, positions=positions_by_expert)
            batches = ((expert_ids, experts),)
        else:
            batches = iter_expert_batches(layer, expert_ids, positions=positions_by_expert)

        def consume_batch(batch_ids, experts):
            nonlocal out
            for e in batch_ids:
                plist = groups[e]
                positions = [p for p, _ in plist]
                weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
                y = _swiglu(h[:, positions, :], experts[e], f"{prefix}.mlp.experts.{e}")
                contribution = (y.astype(mx.float32) * weights[None, :, None]).astype(h.dtype)
                out = out.at[:, positions, :].add(contribution)
            mx.eval(out)

        consume_expert_batches(batches, consume_batch)
        shared = _swiglu(h, w, f"{prefix}.mlp.shared_experts")
        h = out + shared

    h = mx.fast.rms_norm(h, w[f"{prefix}.post_mlp_layernorm.weight"], cfg.rms_norm_eps)
    return residual + h
