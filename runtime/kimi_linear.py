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
    if cfg.model_type == "kimi_k3":
        # F128: K3's real bundled modeling_kimi_linear.py FIXED this aliasing
        # bug -- its KimiMoEGate.forward computes
        #     scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        # using regular `+` (a fresh tensor), never `+=` on a `.view()`, so
        # `scores` itself is never mutated; `topk_weight = scores.gather(...)`
        # genuinely reads the UNBIASED scores -- bias affects selection only,
        # matching GLM's noaux_tc design intent. Confirmed by directly
        # reading the real, bundled modeling_kimi_linear.py shipped with the
        # actual downloaded K3 checkpoint (not re-derived/assumed from the
        # original Kimi Linear 48B's own real aliasing bug below).
        pw = mx.take_along_axis(scores, idx, axis=-1)
    else:
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
    """out = rmsnorm(x) * weight * sigmoid(gate) -- gate applied AFTER norm+scale.

    F105-style native-primitive reuse (2026-07-25): this was a hand-rolled
    composite (mean/sqrt/multiply) that never called mx.fast.rms_norm,
    unlike this same file's input_layernorm/post_attention_layernorm calls
    a few lines below, which already did. No formula transform needed here
    (plain weight scaling, no zero-centered offset) -- verified
    byte-identical (0.0 max abs diff) against the original composite.
    Same expectation as F105's qwen3.5 case: this op is tiny relative to
    the matmul/disk costs that dominate a decode step, so it is kept as a
    correctness-preserving simplification, not claimed as a speed win
    without a real measurement to back that claim."""
    source_dtype = x.dtype
    x32 = x.astype(mx.float32)
    w32 = weight.astype(mx.float32)
    normed = mx.fast.rms_norm(x32, w32, eps)
    return (normed * mx.sigmoid(gate.astype(mx.float32))).astype(source_dtype)


def _kimi_expert_swiglu(h: mx.array, w: dict, prefix: str) -> mx.array:
    """Per-routed-expert MLP: w1=gate, w2=down, w3=up (Mixtral-style naming)."""
    gate = _linear(h, w, f"{prefix}.w1")
    up = _linear(h, w, f"{prefix}.w3")
    activated = mx.sigmoid(gate) * gate * up
    return _linear(activated, w, f"{prefix}.w2")


def _situ_and_mul(gate: mx.array, up: mx.array, beta: float, linear_beta: float) -> mx.array:
    """Kimi K3's real "situ" activation (hidden_act="situ"), verbatim from the
    real modeling_kimi_linear.py's SituAndMul.forward: both halves are
    upcast to float32 for the tanh/sigmoid, result cast back to the input
    dtype. `linear_beta` of 0.0 means "unset" (`up` passes through
    untransformed), matching the real code's `if self.linear_beta is not
    None`. NOT used by Kimi Linear 48B or Kimi K2.5 (both plain swiglu,
    cfg.hidden_act defaults to "silu" for them) -- Kimi K3 only."""
    gate32 = gate.astype(mx.float32)
    up32 = up.astype(mx.float32)
    situ_a = beta * mx.tanh(gate32 / beta) * mx.sigmoid(gate32)
    if linear_beta:
        up32 = linear_beta * mx.tanh(up32 / linear_beta)
    return (situ_a * up32).astype(gate.dtype)


def _kimi_expert_mlp(h: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> mx.array:
    """Per-routed-expert MLP, dispatching activation by cfg.hidden_act.
    Kimi Linear/K2.5 (hidden_act="silu") take the exact same swiglu path as
    _kimi_expert_swiglu above (kept separate so that function's existing
    real-oracle test import/call sites are untouched); Kimi K3
    (hidden_act="situ") uses _situ_and_mul instead."""
    if cfg.hidden_act != "situ":
        return _kimi_expert_swiglu(h, w, prefix)
    gate = _linear(h, w, f"{prefix}.w1")
    up = _linear(h, w, f"{prefix}.w3")
    activated = _situ_and_mul(
        gate, up, cfg.activation_situ_beta, cfg.activation_situ_linear_beta)
    return _linear(activated, w, f"{prefix}.w2")


def _kimi_dense_mlp(h: mx.array, w: dict, prefix: str, cfg: ModelConfig) -> mx.array:
    """gate_proj/up_proj/down_proj MLP (dense layer-0, and shared_experts),
    dispatching activation by cfg.hidden_act. Kimi Linear/K2.5 take the
    exact same path as the existing layer_runner._swiglu call sites (kept
    unchanged so their own real-oracle tests are untouched); Kimi K3 uses
    _situ_and_mul instead."""
    if cfg.hidden_act != "situ":
        return _swiglu(h, w, prefix)
    gate = _linear(h, w, f"{prefix}.gate_proj")
    up = _linear(h, w, f"{prefix}.up_proj")
    activated = _situ_and_mul(
        gate, up, cfg.activation_situ_beta, cfg.activation_situ_linear_beta)
    return _linear(activated, w, f"{prefix}.down_proj")


def _kda_attention(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, kda_cache: KDAStateCache | None, layer: int,
    native_fused_decode: bool = False,
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
    # 2026-07-25: KDA's per-channel causal conv (K-tap weighted sum + SiLU)
    # is mathematically IDENTICAL to qwen3.5's own conv1d+SiLU -- this file
    # already provides the plain/shared implementation qwen35.py imports
    # (_causal_depthwise_conv1d, above), and qwen35.py's F103 native fused
    # Metal kernel is a verified-byte-identical drop-in for it (same K-tap
    # weighted sum + sigmoid(acc)*acc SiLU, same (B,L,C)/(C,1,K) shapes).
    # Reused directly here, the same "found a mathematically identical
    # existing loop, reused the kernel unmodified" pattern that already
    # worked for Jet-Nemotron's DeltaNet-step kernel reuse.
    conv_fn = _causal_depthwise_conv1d
    if L == 1 and native_fused_decode:
        from .qwen35 import _native_fused_causal_conv1d
        conv_fn = _native_fused_causal_conv1d
    q, q_hist_new = conv_fn(q, w[f"{prefix}.self_attn.q_conv1d.weight"], q_hist, K)
    k, k_hist_new = conv_fn(k, w[f"{prefix}.self_attn.k_conv1d.weight"], k_hist, K)
    v, v_hist_new = conv_fn(v, w[f"{prefix}.self_attn.v_conv1d.weight"], v_hist, K)

    dt_bias = w[f"{prefix}.self_attn.dt_bias"].reshape(H, D).astype(mx.float32)
    g_raw = _linear(_linear(h, w, f"{prefix}.self_attn.f_a_proj"), w, f"{prefix}.self_attn.f_b_proj")
    g_raw = g_raw.reshape(B, L, H, D).astype(mx.float32) + dt_bias
    # F128: Kimi K3's real checkpoint saves A_log with head_dim elements
    # (128), not num_heads (96) -- confirmed directly against a real
    # downloaded layer's shard (b_proj/dt_bias both correctly reflect
    # num_heads=96 elsewhere in the SAME layer, so this is A_log-specific,
    # not a wrong num_heads reading). Fetched fla-org/flash-linear-
    # attention's real kda/gate.py Triton kernel source (2026-07-27) shows
    # A_log is indexed strictly `A_log + i_h` for i_h in [0, H) -- H passed
    # explicitly by the caller, never inferred from A_log's own tensor
    # size -- so the real kernel silently reads only the first H=96
    # elements regardless of the buffer's true (over-allocated) length.
    # Not runtime-verified against the real Triton kernel itself (no
    # CUDA/Triton on this machine), but this is the only interpretation
    # consistent with every real source available: the kernel's own
    # indexing, and the original Kimi Linear 48B's A_log (no over-
    # allocation, exactly num_heads elements, unaffected by this slice).
    A = mx.exp(w[f"{prefix}.self_attn.A_log"][:H].astype(mx.float32)).reshape(1, 1, H, 1)
    if cfg.kda_gate_lower_bound:
        # F128: Kimi K3's real linear_attn_config sets gate_lower_bound=-5.0
        # (safe_gate=True in the real KimiDeltaAttention.forward) -- ported
        # verbatim from the real kda_gate_fwd_kernel's USE_LOWER_BOUND
        # branch: `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, using
        # the RAW g_raw directly (no softplus at all in this branch, unlike
        # the no-lower-bound formula below).
        gate = cfg.kda_gate_lower_bound * mx.sigmoid(A * g_raw)
    else:
        softplus_g = mx.logaddexp(g_raw, mx.zeros_like(g_raw))  # log(1 + exp(x)), numerically stable
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

    # F128: Kimi K3's real KimiDeltaAttention.forward picks a single
    # full-rank g_proj instead of the low-rank g_a_proj/g_b_proj split when
    # config.linear_attn_config.use_full_rank_gate is true (confirmed
    # present -- true -- on the real checkpoint; absent/false for the
    # original Kimi Linear 48B, which only ever ships g_a_proj/g_b_proj).
    if cfg.kda_use_full_rank_gate:
        g_out = _linear(h, w, f"{prefix}.self_attn.g_proj")
    else:
        g_out = _linear(_linear(h, w, f"{prefix}.self_attn.g_a_proj"), w, f"{prefix}.self_attn.g_b_proj")
    g_out = g_out.reshape(B, L, H, D)
    o = _gated_rms_norm(o, g_out, w[f"{prefix}.self_attn.o_norm.weight"], cfg.rms_norm_eps)
    o = o.reshape(B, L, H * D)
    return _linear(o, w, f"{prefix}.self_attn.o_proj")


def _kimi_linear_attention_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, mlp_last_only: bool = False,
    native_fused_decode: bool = False,
) -> mx.array:
    """Attention (KDA or MLA) + residual only, no MLP/MoE. Split out of
    the original monolithic `run_kimi_linear_block` (F35-prep, 2026-07-24)
    so layer-stationary tiled prefill can call this PER TILE (attention
    must still see tiles in causal/sequential order -- KDA's recurrent
    state and MLA's KV cache both accumulate exactly as before, this split
    changes nothing about that) while calling the MLP/MoE half exactly
    ONCE per layer across the whole prompt instead. `run_kimi_linear_block`
    below is now a thin two-call wrapper preserving the exact original
    behavior for existing (chunk-major) callers.

    `kv` carries KDA's recurrent state the same way GLM's MLA carries
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
        x = x + _kda_attention(
            h, w, prefix, cfg, kda_cache, layer,
            native_fused_decode=native_fused_decode)
    else:
        raise ValueError(
            f"layer {layer} is in neither cfg.full_attn_layers nor cfg.kda_layers")

    if mlp_last_only:  # KV/state is built; only the last position feeds the logits
        x = x[:, -1:, :]
    return x


def _kimi_linear_mlp_residual(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None, profile=None,
) -> mx.array:
    """MLP (dense) or MoE + residual only, given `x` already post-attention.
    See `_kimi_linear_attention_residual`'s docstring for why this is split
    out. `x` may cover any subset of positions (a single tile, or the
    whole prompt) -- routing/expert-fetch always operates on exactly
    whatever positions are present in `x`, which is what lets a
    layer-stationary caller route the WHOLE prompt at once instead of
    per-tile."""
    h = mx.fast.rms_norm(x, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

    if layer < cfg.first_k_dense_replace:
        return x + _kimi_dense_mlp(h, w, f"{prefix}.mlp", cfg)

    return x + _kimi_moe_output(h, w, prefix, cfg, layer, get_experts, iter_expert_batches, profile)


def _kimi_moe_output(
    h: mx.array, w: dict, prefix: str, cfg: ModelConfig, layer: int,
    get_experts, iter_expert_batches=None, profile=None,
) -> mx.array:
    """Routed-experts + shared-experts MoE output ONLY -- no residual add,
    and the caller must already have checked `layer >= cfg.first_k_dense_replace`
    (the dense layer-0 case is not handled here). Factored out of
    `_kimi_linear_mlp_residual` (which just does `x + this`) so the
    AttnRes-aware K3 block runner below can reuse the exact same MoE math
    without duplicating it -- AttnRes replaces what happens to the residual
    stream around each sublayer, not the sublayer's own computation."""
    moe_prefix = f"{prefix}.block_sparse_moe"
    router_t0 = profile.start_substep() if profile is not None else None
    idx, pw = _route_experts(h, w, moe_prefix, cfg)
    if not (profile is not None and profile.finish_substep(
            "router", layer, router_t0, idx, pw,
            positions=int(h.shape[1]))):
        mx.eval(idx, pw)
    groups = _group_routes(idx, pw)

    # F128: Kimi K3's real KimiSparseMoeBlock routes on the FULL hidden
    # state (h, above) but runs each expert in a smaller "latent" space
    # (config.routed_expert_hidden_size) when cfg.moe_latent_hidden_size is
    # set -- confirmed by the real routed_expert_down_proj/_norm/_up_proj
    # tensors on a real downloaded shard. Kimi Linear/K2.5 leave this 0, so
    # h_latent is just h unchanged for them (identical behavior to before
    # this branch existed).
    if cfg.moe_latent_hidden_size:
        h_latent = _linear(h, w, f"{moe_prefix}.routed_expert_down_proj")
    else:
        h_latent = h

    out = mx.zeros_like(h_latent)
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
            y = _kimi_expert_mlp(h_latent[:, positions, :], experts[e], f"{moe_prefix}.experts.{e}", cfg)
            contribution = (y * route_weights[None, :, None]).astype(h_latent.dtype)
            out = out.at[:, positions, :].add(contribution)
        mx.eval(out)
        del contribution, y, route_weights

    consume_expert_batches(batches, consume_batch)

    if cfg.moe_latent_hidden_size:
        if cfg.moe_latent_use_norm:
            out = mx.fast.rms_norm(
                out, w[f"{moe_prefix}.routed_expert_norm.weight"], cfg.rms_norm_eps)
        out = _linear(out, w, f"{moe_prefix}.routed_expert_up_proj")

    return out + _kimi_dense_mlp(h, w, f"{moe_prefix}.shared_experts", cfg)


def run_kimi_linear_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, get_experts, mlp_last_only: bool = False, iter_expert_batches=None,
    native_fused_decode: bool = False,
    profile=None,
) -> mx.array:
    """One Kimi Linear decoder block (chunk-major / ordinary use). Thin
    wrapper over `_kimi_linear_attention_residual` +
    `_kimi_linear_mlp_residual` -- see those functions' docstrings for why
    the split exists (layer-stationary tiled prefill, F35-prep)."""
    positions = int(x.shape[1])
    attention_t0 = profile.start_substep() if profile is not None else None
    x = _kimi_linear_attention_residual(
        x, w, prefix, cfg, kv, layer, offset, mlp_last_only=mlp_last_only,
        native_fused_decode=native_fused_decode)
    if profile is not None:
        profile.finish_substep(
            "attention", layer, attention_t0, x, positions=positions)
    mlp_t0 = profile.start_substep() if profile is not None else None
    x = _kimi_linear_mlp_residual(
        x, w, prefix, cfg, layer, get_experts,
        iter_expert_batches=iter_expert_batches, profile=profile)
    if profile is not None:
        profile.finish_substep(
            "mlp", layer, mlp_t0, x, positions=int(x.shape[1]))
    return x


def _apply_attn_res(
    prefix_sum: mx.array, block_residual: mx.array,
    proj_weight: mx.array, norm_weight: mx.array, eps: float,
) -> mx.array:
    """F128: Kimi K3's "Attention Residuals" (AttnRes, arXiv 2603.15031),
    ported verbatim from the real modeling_kimi_linear.py's module-level
    `_apply_attn_res`: a softmax-attention readout over `block_residual`
    (residual-stream snapshots taken every `cfg.attn_res_block_size`
    layers, one column per snapshot so far) PLUS the current running
    `prefix_sum`, using a shared RMSNorm + single learned scalar projection
    per query/key ("proj"/"norm" are `nn.Linear(hidden,1,bias=False)` and
    an RMSNorm respectively -- NOT the usual QK attention shapes, there is
    only one score per snapshot, not a per-head/per-dim breakdown).

    prefix_sum: (N, hidden). block_residual: (N, num_blocks, hidden), where
    N = batch*positions and num_blocks may be 0 (no snapshot taken yet --
    callers must skip calling this entirely in that case, matching the real
    code's `if block_residual.shape[1] > 0` guard around its first call
    site only; every other call site always has num_blocks >= 1 by
    construction). Returns (N, hidden).
    """
    v = mx.concatenate([block_residual, prefix_sum[:, None, :]], axis=1)
    v32 = v.astype(mx.float32)
    variance = mx.mean(v32 * v32, axis=-1, keepdims=True)
    k = v32 * mx.rsqrt(variance + eps)
    score_weight = norm_weight.astype(mx.float32) * proj_weight.reshape(-1).astype(mx.float32)
    scores = mx.sum(k * score_weight, axis=-1)
    probs = mx.softmax(scores, axis=-1)[:, None, :]
    out = (probs @ v32)[:, 0, :]
    return out.astype(v.dtype)


def attn_res_wrap_layer(
    x: mx.array, block_residual: mx.array, w: dict, prefix: str,
    cfg: ModelConfig, layer: int, attn_fn, mlp_fn,
) -> tuple[mx.array, mx.array]:
    """The AttnRes bookkeeping itself, ported verbatim from the real
    `KimiDecoderLayer._forward_attn_residual`'s control flow, factored out
    from any particular attention/MLP math so it can be unit-tested against
    a real-reference torch transcription with trivial stand-in
    `attn_fn`/`mlp_fn` (see tests/test_f128_k3_attn_res_oracle.py) --
    KDA/MLA/MoE math is already independently oracle-verified elsewhere
    (F92/F93), so this isolates the genuinely new risk: getting the
    block-boundary reset/snapshot bookkeeping itself right.

    `attn_fn`/`mlp_fn` each take the appropriately-normed hidden state and
    return the sublayer's raw output (pre-residual) -- `run_kimi_k3_block`
    below supplies the real KDA/MLA/dense/MoE closures; the oracle test
    supplies simple deterministic stand-ins instead.

    Returns `(new_prefix_sum, new_block_residual)`, both to be threaded
    into the next layer's call exactly like `x` itself already is.
    """
    B, L, H = x.shape
    prefix_sum = x
    hidden_states = x

    if block_residual.shape[1] > 0:
        hidden_states = _apply_attn_res(
            prefix_sum.reshape(-1, H), block_residual,
            w[f"{prefix}.self_attention_res_proj.weight"],
            w[f"{prefix}.self_attention_res_norm.weight"], cfg.rms_norm_eps,
        ).reshape(B, L, H)

    if layer % cfg.attn_res_block_size == 0:
        block_residual = mx.concatenate(
            [block_residual, prefix_sum.reshape(-1, H)[:, None, :]], axis=1)
        prefix_sum = None

    attn_out = attn_fn(mx.fast.rms_norm(
        hidden_states, w[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps))
    prefix_sum = (prefix_sum + attn_out) if prefix_sum is not None else attn_out

    hidden_states = _apply_attn_res(
        prefix_sum.reshape(-1, H), block_residual,
        w[f"{prefix}.mlp_res_proj.weight"],
        w[f"{prefix}.mlp_res_norm.weight"], cfg.rms_norm_eps,
    ).reshape(B, L, H)

    mlp_out = mlp_fn(mx.fast.rms_norm(
        hidden_states, w[f"{prefix}.post_attention_layernorm.weight"], cfg.rms_norm_eps))
    prefix_sum = prefix_sum + mlp_out
    return prefix_sum, block_residual


def run_kimi_k3_block(
    x: mx.array, w: dict, prefix: str, cfg: ModelConfig, kv,
    layer: int, offset: int, block_residual: mx.array, get_experts,
    mlp_last_only: bool = False, iter_expert_batches=None,
    native_fused_decode: bool = False, profile=None,
) -> tuple[mx.array, mx.array]:
    """One Kimi K3 decoder block WITH AttnRes -- thin wrapper supplying the
    real KDA/MLA attention and dense/MoE MLP closures to
    `attn_res_wrap_layer` above (see its docstring for the AttnRes mechanism
    itself). Unlike `run_kimi_linear_block` (whose `x = x + sublayer_out`
    residual this does NOT use), the running accumulator is `prefix_sum`:
    it resets to just the sublayer's own output at every block boundary
    instead of adding onto the prior value. `block_residual` is this
    function's extra piece of state, threaded by the caller exactly like
    `x` itself -- purely an intra-forward-pass, depth-wise accumulator (the
    real reference re-inits it fresh at the top of every `forward()` call
    and never persists it via any KV/cache mechanism), so chunk-major
    callers must start a fresh empty `block_residual` at the top of each
    chunk's layer loop, never carrying it across chunks or decode steps.
    Returns `(x, block_residual)`, both to be threaded into the next
    layer's call.
    """
    def attn_fn(h):
        if layer in cfg.full_attn_layers:
            return _mla_attention(h, w, prefix, cfg, kv, layer, offset)
        if layer in cfg.kda_layers:
            kda_cache = getattr(kv, "kda_cache", None)
            return _kda_attention(
                h, w, prefix, cfg, kda_cache, layer,
                native_fused_decode=native_fused_decode)
        raise ValueError(
            f"layer {layer} is in neither cfg.full_attn_layers nor cfg.kda_layers")

    def mlp_fn(h2):
        if layer < cfg.first_k_dense_replace:
            return _kimi_dense_mlp(h2, w, f"{prefix}.mlp", cfg)
        return _kimi_moe_output(
            h2, w, prefix, cfg, layer, get_experts,
            iter_expert_batches=iter_expert_batches, profile=profile)

    # F128: unlike run_kimi_linear_block's mlp_last_only (which trims BETWEEN
    # attention and MLP to skip MLP compute for positions whose logits are
    # never needed), this function deliberately ignores `mlp_last_only` and
    # always processes the full L positions through both attention and MLP.
    # Trimming here would shrink x's row count out from under block_residual
    # (built up over ALL positions across every earlier layer in this same
    # sweep) with no matching trim on block_residual's own rows, breaking
    # the row-alignment attn_res_wrap_layer's concatenation depends on. The
    # caller (Engine._sweep) trims AFTER the whole layer loop AND
    # apply_output_attn_res have both run instead -- see its own comment.
    # `mlp_last_only` is accepted only so this function's call signature
    # matches run_kimi_linear_block's; wasted MLP compute on discarded
    # positions is a real, un-optimized cost here, not a correctness issue.
    del mlp_last_only
    return attn_res_wrap_layer(
        x, block_residual, w, prefix, cfg, layer, attn_fn, mlp_fn)


def apply_output_attn_res(
    x: mx.array, w: dict, block_residual: mx.array, cfg: ModelConfig,
) -> mx.array:
    """The final AttnRes readout applied once after ALL layers (real
    `KimiLinearModel._apply_output_attn_res`), before the model's final
    RMSNorm -- uses its own dedicated `model.output_attn_res_proj`/
    `model.output_attn_res_norm` weights, distinct from any per-layer ones."""
    B, L, H = x.shape
    return _apply_attn_res(
        x.reshape(-1, H), block_residual,
        w["model.output_attn_res_proj.weight"],
        w["model.output_attn_res_norm.weight"], cfg.rms_norm_eps,
    ).reshape(B, L, H)
