"""F92 numerical oracle: real modeling_kimi.py vs. runtime/kimi_linear.py.

This is the gate F92 (docs/future_lossless_techniques.md) explicitly said was
still open: a real-transformers oracle, not just a shape/plumbing smoke test.

`fla-core`'s ops package unconditionally imports `triton` at package-init
time (`fla/ops/__init__.py` -> `.abc` -> `import triton`), and there is no
Triton wheel for Apple Silicon macOS -- so the real released `modeling_kimi.py`
cannot import as-is on this machine. This test installs pure-PyTorch stand-ins
for exactly the pieces `fla` would have supplied (KDA gate math, the gated
delta-rule recurrence, the short causal conv, the gated RMSNorm) into
`sys.modules` BEFORE importing the real `modeling_kimi.py`/`configuration_kimi.py`
files unmodified from the downloaded checkpoint directory. Every formula in
the stand-ins was pulled from the real fla-org/flash-linear-attention source
(gate.py, naive.py, fused_recurrent.py, modules/fused_norm_gate.py) via
WebFetch on 2026-07-18, not guessed -- see F92's module docstring in
runtime/kimi_linear.py for the citations. The MLA attention, MoE gate/expert
routing, and decoder-layer plumbing all run as 100% real, unmodified
`modeling_kimi.py` code (eager attention backend, no flash-attn needed).

This does NOT use the real 48B-parameter released weights (infeasible to
even instantiate as PyTorch nn.Parameters on this machine's RAM) -- it uses a
tiny random-weight config, extracts the state_dict, and feeds the identical
weights into runtime/kimi_linear.py, exactly the same methodology already
established in tests/test_f33_mla_attention.py for GLM's MLA.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from runtime.config import ModelConfig
from runtime.kda_state import KDAStateCache
from runtime.kimi_linear import _kda_attention, _kimi_expert_swiglu, _route_experts
from runtime.glm import _group_routes, _mla_attention
from runtime.kv_cache import KVCache

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-Linear-48B-A3B-Instruct"
_MODEL_AVAILABLE = (MODEL_DIR / "modeling_kimi.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-Linear-48B-A3B-Instruct's real modeling_kimi.py is not "
           "available locally (a real checkpoint's source file, not fetched in CI)",
)


def _install_fla_stubs() -> None:
    """Pure-PyTorch stand-ins for the Triton-only pieces of fla-core.

    Formulas transcribed from the real fla-org/flash-linear-attention source
    (2026-07-18), not reconstructed from memory -- see the module docstring.
    """
    if "fla" in sys.modules and getattr(sys.modules["fla"], "_f92_stub", False):
        return  # already installed this process

    fla_mod = types.ModuleType("fla")
    fla_mod._f92_stub = True
    sys.modules["fla"] = fla_mod

    fla_utils = types.ModuleType("fla.utils")
    fla_utils.tensor_cache = lambda fn: fn  # identity decorator; never hit (no padding mask in this test)
    sys.modules["fla.utils"] = fla_utils

    fla_ops = types.ModuleType("fla.ops")
    sys.modules["fla.ops"] = fla_ops
    fla_ops_utils = types.ModuleType("fla.ops.utils")
    sys.modules["fla.ops.utils"] = fla_ops_utils
    fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")

    def _unsupported(*a, **k):
        raise NotImplementedError("F92 oracle stub: attention_mask must be None (no padding path)")

    fla_ops_utils_index.prepare_lens_from_mask = _unsupported
    fla_ops_utils_index.prepare_cu_seqlens_from_mask = _unsupported
    sys.modules["fla.ops.utils.index"] = fla_ops_utils_index

    # --- fla.ops.kda.gate.fused_kda_gate ---
    # Real formula (fla/ops/kda/gate.py naive reference):
    #   g = -exp(A_log).unsqueeze(-1) * softplus(g + dt_bias.view(H, K))
    fla_ops_kda_gate = types.ModuleType("fla.ops.kda.gate")

    def fused_kda_gate(g: torch.Tensor, A_log: torch.Tensor, head_dim: int,
                        g_bias: torch.Tensor | None = None) -> torch.Tensor:
        *lead, proj = g.shape
        H = proj // head_dim
        g = g.view(*lead, H, head_dim)
        if g_bias is not None:
            g = g + g_bias.view(H, head_dim)
        softplus_g = F.softplus(g)
        A = torch.exp(A_log.reshape(1, 1, H, 1).to(torch.float32))
        return -A * softplus_g

    fla_ops_kda_gate.fused_kda_gate = fused_kda_gate
    sys.modules["fla.ops.kda.gate"] = fla_ops_kda_gate

    # --- fla.ops.kda.{chunk_kda, fused_recurrent_kda} ---
    # Real recurrence (fla/ops/kda/naive.py naive_recurrent_kda), shapes per
    # fla/ops/kda/fused_recurrent.py's public docstring: q,k [B,T,H,K],
    # v [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV], state [N,HV,K,V].
    # chunk_kda and fused_recurrent_kda are numerically the same recurrence
    # (chunked-parallel vs sequential is an algorithm choice, not a different
    # formula) -- this stub implements it once, sequentially, for both.
    fla_ops_kda = types.ModuleType("fla.ops.kda")

    def _kda_recurrence(q, k, v, g, beta, initial_state=None, output_final_state=False,
                         use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        if cu_seqlens is not None:
            raise NotImplementedError("F92 oracle stub: cu_seqlens (padded batches) not supported")
        B, T, H, K = q.shape
        V = v.shape[-1]
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * (K ** -0.5)

        state = (initial_state.clone() if initial_state is not None
                  else q.new_zeros(B, H, K, V))
        outputs = []
        for t in range(T):
            q_t, k_t, v_t, g_t, beta_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
            state = state * g_t.exp()[..., None]
            pred_v = (k_t[..., None] * state).sum(-2)
            state = state + torch.einsum(
                "bhk,bhv->bhkv", beta_t[..., None] * k_t, v_t - pred_v)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
        o = torch.stack(outputs, dim=1)
        return o, (state if output_final_state else None)

    fla_ops_kda.chunk_kda = _kda_recurrence
    fla_ops_kda.fused_recurrent_kda = _kda_recurrence
    sys.modules["fla.ops.kda"] = fla_ops_kda

    # --- fla.modules.{ShortConvolution, FusedRMSNormGated} ---
    fla_modules = types.ModuleType("fla.modules")

    class ShortConvolution(torch.nn.Module):
        """Causal depthwise conv1d + activation. Matches the real module's
        constructor/call signature for the single-shot (no cache, no
        cu_seqlens) path this oracle exercises."""

        def __init__(self, hidden_size, kernel_size, activation=None, bias=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.kernel_size = kernel_size
            self.activation = activation
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
            self.bias = torch.nn.Parameter(torch.empty(hidden_size)) if bias else None

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
            if cu_seqlens is not None:
                raise NotImplementedError("F92 oracle stub: cu_seqlens not supported")
            B, T, C = x.shape
            K = self.kernel_size
            history = cache if cache is not None else x.new_zeros(B, K - 1, C)
            padded = torch.cat([history, x], dim=1)
            taps = self.weight.view(C, K)
            out = x.new_zeros(B, T, C)
            for k in range(K):
                out = out + padded[:, k:k + T, :] * taps[:, k]
            if self.bias is not None:
                out = out + self.bias
            if self.activation == "silu":
                out = F.silu(out)
            new_cache = padded[:, T:, :] if output_final_state else None
            return out, new_cache

    class FusedRMSNormGated(torch.nn.Module):
        """rmsnorm(x) * weight * sigmoid(gate) -- gate applied AFTER norm+scale
        (fla/modules/fused_norm_gate.py's layer_norm_gated_fwd_kernel: stats
        computed on raw x, gate multiplied in last)."""

        def __init__(self, hidden_size, eps=1e-5, activation="sigmoid"):
            super().__init__()
            assert activation == "sigmoid"
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, gate):
            x32 = x.float()
            var = (x32 * x32).mean(-1, keepdim=True)
            x_hat = x32 * torch.rsqrt(var + self.eps)
            y = x_hat * self.weight.float() * torch.sigmoid(gate.float())
            return y.to(x.dtype)

    fla_modules.ShortConvolution = ShortConvolution
    fla_modules.FusedRMSNormGated = FusedRMSNormGated
    sys.modules["fla.modules"] = fla_modules


def _patch_transformers_generic_shims() -> None:
    """`transformers` 5.13.0 (installed here) removed/renamed `OutputRecorder`
    and `check_model_inputs` from `transformers.utils.generic`, which the
    model's `modeling_kimi.py` (written against transformers>=4.56.0) imports
    at module level. Both are only used in `KimiPreTrainedModel` class-body
    dict literals / a `forward` decorator on `KimiLinearModel` -- neither is
    exercised by this test file (it instantiates individual layer classes,
    never `KimiLinearModel.forward`), so a no-op stand-in is safe and does
    not affect anything this oracle actually measures. Patches the real,
    already-imported `transformers.utils.generic` module in place (reversible,
    process-local) rather than touching the installed package on disk.
    """
    import transformers.utils.generic as generic_mod
    if not hasattr(generic_mod, "OutputRecorder"):
        class _NoOpOutputRecorder:
            def __init__(self, *a, **k):
                pass
        generic_mod.OutputRecorder = _NoOpOutputRecorder
    if not hasattr(generic_mod, "check_model_inputs"):
        generic_mod.check_model_inputs = lambda fn: fn


def _load_real_modeling_kimi():
    _install_fla_stubs()
    _patch_transformers_generic_shims()
    pkg_name = "_f92_kimi_linear_real"
    if pkg_name in sys.modules and hasattr(sys.modules[pkg_name], "_loaded_ok"):
        return sys.modules[f"{pkg_name}.modeling_kimi"], sys.modules[f"{pkg_name}.configuration_kimi"]
    sys.modules.pop(pkg_name, None)
    sys.modules.pop(f"{pkg_name}.modeling_kimi", None)
    sys.modules.pop(f"{pkg_name}.configuration_kimi", None)

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(MODEL_DIR)]
    sys.modules[pkg_name] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", MODEL_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load("configuration_kimi")
    model_mod = _load("modeling_kimi")
    pkg._loaded_ok = True
    return model_mod, cfg_mod


HIDDEN, NUM_HEADS, HEAD_DIM, CONV_K = 64, 4, 16, 4
NUM_EXPERTS, TOP_K, MOE_INTER = 6, 2, 32
KV_LORA, QK_NOPE, QK_ROPE, V_DIM = 24, 12, 8, 16
S = 7


def _tiny_kimi_config(model_mod, cfg_mod):
    cfg = cfg_mod.KimiLinearConfig(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=2, num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_HEADS,
        head_dim=QK_NOPE + QK_ROPE, first_k_dense_replace=0, moe_layer_freq=1,
        num_experts=NUM_EXPERTS, num_experts_per_token=TOP_K, num_shared_experts=1,
        moe_intermediate_size=MOE_INTER, moe_renormalize=True,
        moe_router_activation_func="sigmoid", num_expert_group=1, topk_group=1,
        routed_scaling_factor=1.7, rms_norm_eps=1e-5, rope_theta=10000.0,
        q_lora_rank=None, kv_lora_rank=KV_LORA, qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE, v_head_dim=V_DIM, mla_use_nope=True,
        linear_attn_config={
            "kda_layers": [1], "full_attn_layers": [2],
            "head_dim": HEAD_DIM, "num_heads": NUM_HEADS, "short_conv_kernel_size": CONV_K,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def _runtime_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="kimi_linear", hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=2, num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_HEADS,
        vocab_size=100, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=4096, tie_word_embeddings=False, attention_bias=False,
        head_dim=QK_NOPE + QK_ROPE, eos_token_ids=(), torch_dtype="float32",
        kda_head_dim=HEAD_DIM, kda_num_heads=NUM_HEADS, kda_conv_kernel_size=CONV_K,
        q_lora_rank=0, kv_lora_rank=KV_LORA, qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE, v_head_dim=V_DIM, mla_latent_norm_eps=1e-6,
        n_shared_experts=1, n_group=1, topk_group=1, routed_scaling_factor=1.7,
        norm_topk_prob=True, kda_layers=(0,), full_attn_layers=(1,),
        first_k_dense_replace=0, moe_layer_freq=1, mla_use_nope=True,
        num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K,
    )


def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    """Additive causal mask matching runtime.glm._mla_attention's own
    internally-built mask -- KimiMLAAttention.forward does NOT build one
    itself (that's the full model's job via create_causal_mask), so a bare
    `real(h, attention_mask=None, ...)` call is bidirectional, not causal."""
    q_pos = torch.arange(seq_len)[:, None]
    k_pos = torch.arange(seq_len)[None, :]
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    return mask[None, None, :, :]


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.3)


@_model_skip
def test_kda_attention_matches_real_modeling_kimi():
    model_mod, cfg_mod = _load_real_modeling_kimi()
    cfg = _tiny_kimi_config(model_mod, cfg_mod)

    real = model_mod.KimiDeltaAttention(cfg, layer_idx=0)
    real.eval()
    _randomize(real, seed=1)
    # reset_parameters() etc aren't relevant here (Linear/Parameter modules
    # already got randomized above); A_log must stay positive pre-log-space,
    # matching the real init (torch.log(uniform(1,16))) rather than N(0,0.3).
    with torch.no_grad():
        real.A_log.copy_(torch.log(torch.empty_like(real.A_log).uniform_(1, 16)))

    torch.manual_seed(2)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out = real(h_torch, attention_mask=None, cache_params=None)

    sd = real.state_dict()
    prefix = "layer1"
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}

    rcfg = _runtime_cfg()
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _kda_attention(h_mx, w, prefix, rcfg, None, 0)
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, f"KDA oracle mismatch: max abs diff {max_abs_diff}"


@_model_skip
def test_mla_attention_matches_real_modeling_kimi_no_q_lora():
    """Kimi Linear's MLA has q_lora_rank=None -- the branch runtime/glm.py's
    _mla_attention gained for F92 (a single q_proj, no q_a/q_b split)."""
    model_mod, cfg_mod = _load_real_modeling_kimi()
    cfg = _tiny_kimi_config(model_mod, cfg_mod)
    assert cfg.q_lora_rank is None

    real = model_mod.KimiMLAAttention(cfg, layer_idx=1)
    real.eval()
    _randomize(real, seed=3)

    torch.manual_seed(4)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out = real(h_torch, attention_mask=_causal_mask_torch(S), past_key_values=None)

    sd = real.state_dict()
    prefix = "layer2"
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}

    rcfg = _runtime_cfg()
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _mla_attention(h_mx, w, prefix, rcfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, f"MLA (no q_lora) oracle mismatch: max abs diff {max_abs_diff}"


@_model_skip
def test_moe_routing_and_experts_match_real_modeling_kimi():
    model_mod, cfg_mod = _load_real_modeling_kimi()
    cfg = _tiny_kimi_config(model_mod, cfg_mod)

    real = model_mod.KimiSparseMoeBlock(cfg)
    real.eval()
    _randomize(real, seed=5)
    with torch.no_grad():
        real.gate.e_score_correction_bias.normal_(mean=0.0, std=0.3)

    torch.manual_seed(6)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out = real(h_torch)

    sd = real.state_dict()
    prefix = "layer3.block_sparse_moe"
    w = {}
    for k, v in sd.items():
        w[f"{prefix}.{k}"] = mx.array(v.numpy())

    rcfg = _runtime_cfg()
    h_mx = mx.array(h_torch.numpy())
    idx, pw = _route_experts(h_mx, w, prefix, rcfg)
    mx.eval(idx, pw)
    groups = _group_routes(idx, pw)

    out = mx.zeros_like(h_mx)
    for e in sorted(groups):
        plist = groups[e]
        positions = [p for p, _ in plist]
        route_weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
        y = _kimi_expert_swiglu(h_mx[:, positions, :], w, f"{prefix}.experts.{e}")
        contribution = (y * route_weights[None, :, None]).astype(h_mx.dtype)
        out = out.at[:, positions, :].add(contribution)
    shared_prefix = f"{prefix}.shared_experts"
    from runtime.layer_runner import _swiglu
    out = out + _swiglu(h_mx, w, shared_prefix)
    mx.eval(out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, f"MoE routing/expert oracle mismatch: max abs diff {max_abs_diff}"
