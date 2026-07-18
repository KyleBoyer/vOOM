"""Kimi Linear (KDA hybrid) block math -- Kimi K3 readiness prep.

See docs/future_lossless_techniques.md F92 for the full architecture audit
of the real moonshotai/Kimi-Linear-48B-A3B-Instruct checkpoint, the
implementation plan, and open gates.

VERIFIED against a real numerical oracle as of 2026-07-18
(tests/test_f92_kda_oracle.py): the KDA attention block, the MLA attention
block (NoPE variant), and the MoE gate+expert routing all match the real,
unmodified `modeling_kimi.py` to <1e-3 max abs diff on a tiny random-weight
instance (same methodology as tests/test_f33_mla_attention.py for GLM --
extract a real HF module's state_dict, feed identical weights through this
module, compare outputs). `fla-core`'s ops package unconditionally imports
`triton` at package-init time and there is no Triton wheel for Apple Silicon
macOS, so the oracle test installs pure-PyTorch stand-ins (formulas
transcribed from the real fla-org/flash-linear-attention source, not
reconstructed from memory) for exactly the pieces `fla` would have supplied,
then runs the real released model code around them. This does NOT use the
real 48B-parameter released weights (infeasible to instantiate as PyTorch
nn.Parameters on this machine's RAM) -- see tests/test_kimi_linear_smoke.py
for the separate real-weights shape/plumbing smoke test.

The oracle caught two real bugs this module's first version got wrong:
1. Kimi Linear's MLA is NoPE (`config.mla_use_nope=True`) -- the real
   `KimiMLAAttention.forward` never calls any rotary-embedding function at
   all; position information comes only from the KDA layers' inherent
   sequential recurrence. `runtime.glm._mla_attention` gained an
   `mla_use_nope` branch for this (GLM always applies real RoPE, unaffected).
2. Kimi's real `KimiMoEGate.forward` has `scores_for_choice = scores.view(...);
   scores_for_choice += bias` -- an in-place `+=` on a `.view()`, which
   aliases and mutates the original `scores` tensor. So the released model's
   actual executed routing WEIGHT (not just expert selection) is computed
   from the bias-corrected score, unlike GLM's noaux_tc design where bias
   affects selection only. Verified to 6 decimal places against the real
   gate before fixing `_route_experts` below to match -- this is very
   likely an unintentional aliasing bug in the released reference code, but
   the mission is byte-for-byte replication of AS-RELEASED behavior, not
   presumed design intent.

Architecture (from the real downloaded modeling_kimi.py / config.json):
- 27 layers. cfg.full_attn_layers (7 of 27, 0-indexed) use MLA; every other
  layer (20 of 27, cfg.kda_layers) uses KDA. Layer 0 is dense MLP (all other
  layers >=1 are MoE, first_k_dense_replace=1, moe_layer_freq=1).
- MLA layers are DeepSeek/GLM-5.2-shaped (kv_a_proj_with_mqa -> RMSNorm ->
  kv_b_proj, NoPE) but with q_lora_rank=null (no Q compression, a single
  q_proj) -- runtime.glm._mla_attention was generalized to handle both.
- MoE gate is the same noaux_tc sigmoid+bias-correction flat top-k as GLM's
  (n_group=topk_group=1 in this checkpoint) -- runtime.glm._route_experts /
  _group_routes are reused directly. Per-routed-expert MLPs use w1/w2/w3
  naming (w1=gate, w2=down, w3=up) instead of GLM's gate_proj/up_proj/
  down_proj, so a small local swiglu variant is used for them; the dense
  layer-0 MLP and each MoE layer's shared_experts both use the ordinary
  gate_proj/up_proj/down_proj naming and reuse layer_runner._swiglu.
- KDA (KimiDeltaAttention) recurrence, per this project's 2026-07-18 read of
  fla-org/flash-linear-attention's ops/kda/{gate,naive,fused_recurrent}.py
  (no local copy of that package to import from -- Triton/CUDA-only anyway):
    q, k are L2-normalized per (batch, head, timestep) over head_dim (eps
    1e-6), q additionally scaled by head_dim ** -0.5, both AFTER a causal
    depthwise conv1d (kernel_size=4, SiLU) applied to q/k/v.
    gate g = -exp(A_log) * softplus(f_b_proj(f_a_proj(h)) + dt_bias), shape
    (B, L, H, head_dim) -- a per-(head, key-channel) log-decay.
    beta = sigmoid(b_proj(h)), a per-head scalar (the delta-rule write
    strength).
    Per-timestep state update (S: (B, H, head_dim, head_dim), K-axis is the
    key/decay axis, V-axis is the value axis):
        S_t = S_{t-1} * exp(g_t)[..., None]
        pred_v = sum_K(k_t[..., None] * S_t)
        S_t = S_t + (beta_t * k_t)[..., None] * (v_t - pred_v)[..., None, :]
        o_t = sum_K(q_t[..., None] * S_t)
  This sequential-scan implementation is correctness-first, not the chunked-
  parallel algorithm the real kernel uses for speed (F92's explicit
  correctness-before-speed stop rule). It is O(L) Python-level steps; expect
  it to be impractically slow for anything beyond a short smoke-test prefix.
- Output: a second low-rank gate (g_a_proj -> g_b_proj) feeds a sigmoid-
  gated RMSNorm (normalize o, scale by o_norm.weight, THEN multiply by
  sigmoid(gate) -- gate applied AFTER normalization, per fla's
  modules/fused_norm_gate.py) before o_proj.
"""

from __future__ import annotations

import mlx.core as mx

from . import quant
from .config import ModelConfig
from .expert_batching import consume_expert_batches
from .glm import _group_routes, _mla_attention
from .kda_state import KDAStateCache
from .layer_runner import _linear, _swiglu


def _route_experts(h: mx.array, w: dict, moe_prefix: str, cfg: ModelConfig) -> tuple[mx.array, mx.array]:
    """Kimi's MoE router. NOT the same weight math as runtime.glm._route_experts.

    Gate weight path differs from GLM's hardcoded f"{prefix}.mlp.gate.*"
    (Kimi's MoE module lives under f"{prefix}.block_sparse_moe.gate.*"), so
    this is a local duplicate rather than a reparametrized import -- avoids
    touching glm._route_experts's existing call sites
    (tests/test_f33_router_oracle.py calls it directly).

    F92 oracle finding (2026-07-18, real modeling_kimi.py, verified to 6
    decimal places against the actual released KimiMoEGate): unlike GLM's
    noaux_tc design where the bias affects ONLY which experts are selected,
    Kimi's real released gate computes
        scores_for_choice = scores.view(...); scores_for_choice += bias
    -- an in-place `+=` on a `.view()`, which ALIASES and mutates the
    original `scores` tensor too. So by the time the real code does
    `topk_weight = scores.gather(1, topk_idx)`, `scores` has ALREADY been
    bias-corrected -- the routing WEIGHT (not just the selection) is
    computed from the biased score. This is very likely an unintentional
    aliasing bug in the released reference code, not deliberate design (it
    contradicts the whole point of noaux_tc bias-correction), but this
    project's mission is byte-for-byte replication of the AS-RELEASED
    checkpoint's actual behavior, not the presumed design intent -- so `pw`
    below is deliberately gathered from `biased`, not `scores`. Do not
    "fix" this to look like GLM's version.
    """
    gate_weight = w[f"{moe_prefix}.gate.weight"]
    if isinstance(gate_weight, quant.QTensor):
        router_logits = quant.matmul(h.astype(mx.float32), gate_weight)
    else:
        router_logits = h.astype(mx.float32) @ gate_weight.astype(mx.float32).T
    scores = mx.sigmoid(router_logits)
    biased = scores + w[f"{moe_prefix}.gate.e_score_correction_bias"]
    k = cfg.num_experts_per_tok
    idx = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    pw = mx.take_along_axis(biased, idx, axis=-1)  # F92: biased, not scores -- see docstring
    if cfg.norm_topk_prob:
        pw = pw / (pw.sum(axis=-1, keepdims=True) + 1e-20)
    pw = pw * cfg.routed_scaling_factor
    return idx, pw


def _causal_depthwise_conv1d(
    x: mx.array, weight: mx.array, history: mx.array | None, kernel_size: int,
) -> tuple[mx.array, mx.array]:
    """Per-channel causal conv (PyTorch Conv1d cross-correlation, no flip), SiLU-activated.

    x: (B, L, C). weight: (C, 1, K) HF Conv1d layout. history: (B, K-1, C)
    carried from a previous call, or None (zero-padded) for the first call.
    Returns (silu(conv(x)), new_history).
    """
    B, L, C = x.shape
    K = kernel_size
    if history is None:
        history = mx.zeros((B, K - 1, C), dtype=x.dtype)
    padded = mx.concatenate([history, x], axis=1)  # (B, L+K-1, C)
    taps = weight.reshape(C, K)  # (C, K), tap k=K-1 is the current timestep
    out = mx.zeros((B, L, C), dtype=mx.float32)
    for k in range(K):
        out = out + padded[:, k:k + L, :].astype(mx.float32) * taps[:, k].astype(mx.float32)
    new_history = padded[:, L:, :] if K > 1 else mx.zeros((B, 0, C), dtype=x.dtype)
    activated = (mx.sigmoid(out) * out).astype(x.dtype)
    return activated, new_history


def _gated_rms_norm(x: mx.array, gate: mx.array, weight: mx.array, eps: float) -> mx.array:
    """out = rmsnorm(x) * weight * sigmoid(gate) -- gate applied AFTER norm+scale."""
    x32 = x.astype(mx.float32)
    var = mx.mean(x32 * x32, axis=-1, keepdims=True)
    x_hat = x32 * (1.0 / mx.sqrt(var + eps))
    y = x_hat * weight.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
    return y.astype(x.dtype)


def _kimi_expert_swiglu(h: mx.array, w: dict, prefix: str) -> mx.array:
    """Per-routed-expert MLP: w1=gate, w2=down, w3=up (Mixtral-style naming)."""
    gate = _linear(h, w, f"{prefix}.w1")
    up = _linear(h, w, f"{prefix}.w3")
    activated = mx.sigmoid(gate) * gate * up
    return _linear(activated, w, f"{prefix}.w2")


def _kda_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kda_cache: KDAStateCache | None, layer: int,
) -> mx.array:
    B, L, _ = h.shape
    H = cfg.kda_num_heads
    D = cfg.kda_head_dim
    K = cfg.kda_conv_kernel_size

    q = _linear(h, w, f"{prefix}.self_attn.q_proj")
    k = _linear(h, w, f"{prefix}.self_attn.k_proj")
    v = _linear(h, w, f"{prefix}.self_attn.v_proj")

    q_hist, k_hist, v_hist = (
        kda_cache.conv_history(layer) if kda_cache is not None and kda_cache.conv_history(layer) is not None
        else (None, None, None)
    )
    q, q_hist_new = _causal_depthwise_conv1d(q, w[f"{prefix}.self_attn.q_conv1d.weight"], q_hist, K)
    k, k_hist_new = _causal_depthwise_conv1d(k, w[f"{prefix}.self_attn.k_conv1d.weight"], k_hist, K)
    v, v_hist_new = _causal_depthwise_conv1d(v, w[f"{prefix}.self_attn.v_conv1d.weight"], v_hist, K)

    dt_bias = w[f"{prefix}.self_attn.dt_bias"].reshape(H, D).astype(mx.float32)
    g_raw = _linear(_linear(h, w, f"{prefix}.self_attn.f_a_proj"), w, f"{prefix}.self_attn.f_b_proj")
    g_raw = g_raw.reshape(B, L, H, D).astype(mx.float32) + dt_bias
    softplus_g = mx.logaddexp(g_raw, mx.zeros_like(g_raw))  # log(1 + exp(x)), numerically stable
    A = mx.exp(w[f"{prefix}.self_attn.A_log"].astype(mx.float32)).reshape(1, 1, H, 1)
    gate = -A * softplus_g  # (B, L, H, D) log-decay, <= 0

    beta = mx.sigmoid(_linear(h, w, f"{prefix}.self_attn.b_proj").astype(mx.float32))  # (B, L, H)

    q = q.reshape(B, L, H, D).astype(mx.float32)
    k = k.reshape(B, L, H, D).astype(mx.float32)
    v = v.reshape(B, L, H, D).astype(mx.float32)

    def _l2norm(x):
        return x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)

    q = _l2norm(q) * (D ** -0.5)
    k = _l2norm(k)

    state = kda_cache.state(layer) if kda_cache is not None else None
    if state is None:
        state = mx.zeros((B, H, D, D), dtype=mx.float32)

    outputs = []
    for t in range(L):
        q_t, k_t, v_t, g_t, beta_t = q[:, t], k[:, t], v[:, t], gate[:, t], beta[:, t]
        state = state * mx.exp(g_t)[..., None]                       # (B,H,K,V) decay along K axis
        pred_v = mx.sum(k_t[..., None] * state, axis=-2)             # (B,H,V)
        residual = v_t - pred_v
        state = state + (beta_t[..., None] * k_t)[..., None] * residual[..., None, :]
        o_t = mx.sum(q_t[..., None] * state, axis=-2)                # (B,H,V)
        outputs.append(o_t)
        if (t + 1) % 32 == 0:
            # F92: bound the lazy graph -- a naive Python-level scan otherwise
            # accumulates one node per op per timestep with no eval boundary.
            mx.eval(state)
    o = mx.stack(outputs, axis=1)  # (B, L, H, D) float32

    if kda_cache is not None:
        mx.eval(state)
        kda_cache.set_state(layer, state)
        kda_cache.set_conv_history(layer, (q_hist_new, k_hist_new, v_hist_new))

    g_out = _linear(_linear(h, w, f"{prefix}.self_attn.g_a_proj"), w, f"{prefix}.self_attn.g_b_proj")
    g_out = g_out.reshape(B, L, H, D)
    o = _gated_rms_norm(o, g_out, w[f"{prefix}.self_attn.o_norm.weight"], cfg.rms_norm_eps)
    o = o.reshape(B, L, H * D)
    return _linear(o, w, f"{prefix}.self_attn.o_proj")


def run_kimi_linear_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, get_experts, mlp_last_only: bool = False, iter_expert_batches=None,
) -> mx.array:
    """`kv` carries KDA's recurrent state the same way GLM's MLA carries
    `kv.compressed_mla`/`kv.dsa` -- an ad-hoc `kv.kda_cache` (KDAStateCache)
    attribute set once in Engine.new_kv(), not a separate threaded argument.
    A bare KVCache (or None, as the oracle/smoke tests pass) has no
    `kda_cache` attribute -- getattr defaults to a fresh-each-call None,
    i.e. stateless single-shot behavior, matching those tests' expectations.
    """
    h = mx.fast.rms_norm(x, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)

    if layer in cfg.full_attn_layers:
        x = x + _mla_attention(h, w, prefix, cfg, kv, layer, offset)
    elif layer in cfg.kda_layers:
        kda_cache = getattr(kv, "kda_cache", None)
        x = x + _kda_attention(h, w, prefix, cfg, kda_cache, layer)
    else:
        raise ValueError(
            f"layer {layer} is in neither cfg.full_attn_layers nor cfg.kda_layers")

    if mlp_last_only:  # KV/state is built; only the last position feeds the logits
        x = x[:, -1:, :]

    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    if layer < cfg.first_k_dense_replace:
        return x + _swiglu(h, w, f"{prefix}.mlp")

    moe_prefix = f"{prefix}.block_sparse_moe"
    idx, pw = _route_experts(h, w, moe_prefix, cfg)
    mx.eval(idx, pw)
    groups = _group_routes(idx, pw)

    out = mx.zeros_like(h)
    expert_ids = sorted(groups)
    positions_by_expert = {e: [pt for pt, _ in groups[e]] for e in expert_ids}
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
            route_weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
            y = _kimi_expert_swiglu(h[:, positions, :], experts[e], f"{moe_prefix}.experts.{e}")
            contribution = (y * route_weights[None, :, None]).astype(h.dtype)
            out = out.at[:, positions, :].add(contribution)
        mx.eval(out)
        del contribution, y, route_weights

    consume_expert_batches(batches, consume_batch)
    out = out + _swiglu(h, w, f"{moe_prefix}.shared_experts")
    return x + out
