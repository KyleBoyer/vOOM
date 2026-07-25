"""Jet-Nemotron numerical oracle: real jet_block.py vs. runtime/jet_nemotron.py.

`fla-core`'s package unconditionally imports `triton` at import time
(`fla/ops/__init__.py` -> `.abc` -> `import triton`, and even `fla.modules`
transitively pulls in `fla.ops` via its convolution submodule) -- there is
no Triton wheel for Apple Silicon, so the real released `jet_block.py`
cannot import as-is on this machine. This test installs pure-PyTorch
stand-ins for exactly the pieces `fla` would have supplied (the gated
delta-rule recurrence, the gated RMSNorm) into `sys.modules` BEFORE
importing the real, unmodified `jet_block.py`/`dynamic_conv.py` files from
the downloaded checkpoint directory -- the same methodology already
established in tests/test_f92_kda_oracle.py for Kimi Linear's KDA.

Formula provenance (not guessed):
- The gated-delta-rule recurrence itself (state decay/update/output) is the
  SAME math this project's tests/test_qwen35_oracle.py already verified
  against the real `Qwen3_5MoeGatedDeltaNet` (`decay = -exp(A_log) *
  softplus(a_proj(h) + dt_bias)`, sigmoid beta, L2-normalized Q/K inside the
  recurrence) -- confirmed identical by direct comparison of both real HF
  sources (modeling_qwen3_5_moe.py vs. the downloaded jet_block.py).
- `FusedRMSNormGated`'s default gate activation ("swish"/SiLU:
  `norm(x) * weight * (gate * sigmoid(gate))`) was independently confirmed
  via WebFetch of the real fla-org/flash-linear-attention
  fla/modules/fused_norm_gate.py source on 2026-07-22 -- NOT copied from
  test_f92_kda_oracle.py's own stub, which uses plain sigmoid (apparently a
  Kimi-specific override elsewhere, not fla's own default; JetBlock passes
  no explicit `activation=`, so it gets fla's real default, "swish").
- The dynamic (per-position, per-channel) causal convolution needs NO stub
  at all: `dynamic_conv.py::DynamicShortConvolution._forward_naive` is a
  complete, real, pure-PyTorch reference already shipped in the checkpoint
  -- this test imports and runs it directly, unmodified.
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
from runtime.jet_nemotron import _jet_block
from runtime.kda_state import KDAStateCache

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Jet-Nemotron-4B"
_MODEL_AVAILABLE = (MODEL_DIR / "jet_block.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Jet-Nemotron-4B's real jet_block.py is not available locally "
           "(a real checkpoint's source file, not fetched in CI)",
)


def _install_fla_stubs() -> None:
    """Pure-PyTorch stand-ins for the Triton-only pieces of fla-core that
    jet_block.py imports. See this module's own docstring for formula
    provenance."""
    if "fla" in sys.modules and getattr(sys.modules["fla"], "_jet_stub", False):
        return

    fla_mod = types.ModuleType("fla")
    fla_mod._jet_stub = True
    sys.modules["fla"] = fla_mod

    fla_layers = types.ModuleType("fla.layers")
    sys.modules["fla.layers"] = fla_layers
    fla_layers_utils = types.ModuleType("fla.layers.utils")

    def _unsupported(*a, **k):
        raise NotImplementedError(
            "jet oracle stub: attention_mask/padding path not supported "
            "(this test always calls with attention_mask=None)")

    fla_layers_utils.get_unpad_data = _unsupported
    fla_layers_utils.index_first_axis = _unsupported
    fla_layers_utils.pad_input = _unsupported
    sys.modules["fla.layers.utils"] = fla_layers_utils

    # --- fla.modules.FusedRMSNormGated ---
    # Real formula/default confirmed via WebFetch of the real
    # fla-org/flash-linear-attention fla/modules/fused_norm_gate.py source,
    # 2026-07-22: default activation="swish" (SiLU), gate applied as
    # norm(x) * weight * (gate * sigmoid(gate)).
    fla_modules = types.ModuleType("fla.modules")

    class FusedRMSNormGated(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, activation="swish", **kwargs):
            super().__init__()
            assert activation in ("swish", "silu")
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, gate):
            x32 = x.float()
            var = (x32 * x32).mean(-1, keepdim=True)
            x_hat = x32 * torch.rsqrt(var + self.eps)
            gate32 = gate.float()
            y = x_hat * self.weight.float() * (gate32 * torch.sigmoid(gate32))
            return y.to(x.dtype)

    fla_modules.FusedRMSNormGated = FusedRMSNormGated
    sys.modules["fla.modules"] = fla_modules

    # --- fla.ops.gated_delta_rule.{chunk_gated_delta_rule, fused_recurrent_gated_delta_rule} ---
    # Same recurrence already oracle-verified for Qwen3.5's DeltaNet in
    # tests/test_qwen35_oracle.py (confirmed identical formula by direct
    # comparison against the real jet_block.py source) -- chunked vs.
    # sequential is an algorithm choice, not a different formula, so one
    # sequential implementation covers both real entry points.
    fla_ops = types.ModuleType("fla.ops")
    sys.modules["fla.ops"] = fla_ops
    fla_ops_gdr = types.ModuleType("fla.ops.gated_delta_rule")

    def _gated_delta_rule(q, k, v, g, beta, initial_state=None,
                           output_final_state=False, cu_seqlens=None,
                           use_qk_l2norm_in_kernel=False, **kwargs):
        if cu_seqlens is not None:
            raise NotImplementedError(
                "jet oracle stub: cu_seqlens (padded batches) not supported")
        # NOTE: unlike Kimi Linear's KDA (g shape [B,T,H,K], a per-(head,
        # key-dim) gate -- see test_f92_kda_oracle.py's own stub), JetBlock's
        # gate is scalar PER HEAD ([B,T,H], from a_proj: Linear(hidden, num_heads)),
        # matching Qwen3.5's own _gated_delta_net exactly (decay shape
        # (1,1,value_heads), broadcast via [..., None, None] -- TWO new axes,
        # not one). Confirmed by direct comparison of both real HF sources.
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
            state = state * g_t.exp()[..., None, None]
            pred_v = (k_t[..., None] * state).sum(-2)
            state = state + torch.einsum(
                "bhk,bhv->bhkv", beta_t[..., None] * k_t, v_t - pred_v)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
        o = torch.stack(outputs, dim=1)
        return o, (state if output_final_state else None)

    fla_ops_gdr.chunk_gated_delta_rule = _gated_delta_rule
    fla_ops_gdr.fused_recurrent_gated_delta_rule = _gated_delta_rule
    sys.modules["fla.ops.gated_delta_rule"] = fla_ops_gdr


def _load_real_jet_block():
    _install_fla_stubs()
    pkg_name = "_jet_oracle_real"
    if pkg_name in sys.modules and hasattr(sys.modules[pkg_name], "_loaded_ok"):
        return sys.modules[f"{pkg_name}.jet_block"]
    sys.modules.pop(pkg_name, None)
    sys.modules.pop(f"{pkg_name}.jet_block", None)
    sys.modules.pop(f"{pkg_name}.dynamic_conv", None)
    sys.modules.pop(f"{pkg_name}.configuration_jet_nemotron", None)
    sys.modules.pop(f"{pkg_name}.kv_cache", None)

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(MODEL_DIR)]
    sys.modules[pkg_name] = pkg

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{name}", MODEL_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _unsupported(*a, **k):
        raise NotImplementedError(
            "jet oracle stub: Triton-only dynamic-conv path not supported "
            "-- this test always uses implementation='naive'")

    # dynamic_conv.py unconditionally imports these three Triton-only
    # sibling modules at module level even though this oracle only ever
    # exercises implementation="naive" (DynamicShortConvolution._forward_naive,
    # a complete real pure-PyTorch reference already in dynamic_conv.py
    # itself -- these three files are speed-only alternatives of the exact
    # same math, never called here).
    for triton_only in ("dconv_fwdbwd", "dconv_fwd_cache", "dconv_step"):
        stub = types.ModuleType(f"{pkg_name}.{triton_only}")
        stub.__package__ = pkg_name
        for name in ("dynamic_conv_triton_autograd", "dynamic_conv_triton_cache",
                     "causal_conv_step_triton"):
            setattr(stub, name, _unsupported)
        sys.modules[f"{pkg_name}.{triton_only}"] = stub

    _load("configuration_jet_nemotron")
    _load("kv_cache")
    _load("dynamic_conv")
    jet_block_mod = _load("jet_block")
    pkg._loaded_ok = True
    return jet_block_mod


HIDDEN = 32
NUM_HEADS = 4
HEAD_DIM = 8
EXPAND_V = 2
CONV_K = 4
GEN_REDUCTION = 4  # must divide HIDDEN evenly for the kernel generator's w1
S = 11


def _runtime_config() -> ModelConfig:
    return ModelConfig(
        model_type="jet_nemotron", hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=1, num_attention_heads=NUM_HEADS, num_key_value_heads=NUM_HEADS,
        vocab_size=100, rms_norm_eps=1e-6, rope_theta=1000000.0,
        max_position_embeddings=32768, tie_word_embeddings=True, attention_bias=False,
        head_dim=HEAD_DIM, eos_token_ids=(), torch_dtype="float32",
        jet_num_heads=NUM_HEADS, jet_head_dim=HEAD_DIM,
        jet_head_v_dim=HEAD_DIM * EXPAND_V, jet_conv_kernel_size=CONV_K,
        jet_dconv_generator_reduction=GEN_REDUCTION,
    )


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.3)


@_model_skip
def test_jet_block_matches_real_jet_block_source():
    jet_block_mod = _load_real_jet_block()
    JetBlockConfig = jet_block_mod.JetBlockConfig
    JetBlock = jet_block_mod.JetBlock

    jbc = JetBlockConfig(
        mode="fused_recurrent",  # q_len <= 64 forces this branch anyway
        expand_v=EXPAND_V, num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        norm_eps=1e-6, conv_size=CONV_K,
        dconv_generator_reduction=GEN_REDUCTION,
        dconv_implementation="naive",
    )
    real = JetBlock(
        hidden_size=HIDDEN, initializer_range=0.02, layer_idx=0,
        jet_block_config=jbc)
    real.eval()
    _randomize(real, seed=1)
    # A_log/dt_bias must stay in their real init's valid ranges (positive
    # pre-log for A_log; dt_bias is an inverse-softplus, sign-unconstrained
    # but shouldn't be resampled N(0, 0.3) either since that's not what the
    # real init produces -- use the real init's own construction instead).
    with torch.no_grad():
        real.A_log.copy_(torch.log(torch.empty_like(real.A_log).uniform_(0, 16)))

    torch.manual_seed(2)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out, _ = real(h_torch, past_key_value=None, attention_mask=None,
                          use_cache=False)

    sd = real.state_dict()
    prefix = "layer0"  # _jet_block appends ".self_attn" itself, matching the
    # real checkpoint's own model.layers.{i}.self_attn.* naming (confirmed
    # via models/Jet-Nemotron-4B/model.safetensors.index.json)
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}

    rcfg = _runtime_config()
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _jet_block(h_mx, w, prefix, rcfg, None, 0)
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, f"JetBlock oracle mismatch: max abs diff {max_abs_diff}"


@_model_skip
def test_dynamic_conv_matches_real_forward_naive_in_isolation():
    """Narrower, more diagnostic test: JUST the dynamic causal conv, in
    case the combined JetBlock test above ever fails -- isolates whether
    the conv or the recurrence is the source of a mismatch."""
    jet_block_mod = _load_real_jet_block()
    dynamic_conv_mod = sys.modules["_jet_oracle_real.dynamic_conv"]
    DynamicShortConvolution = dynamic_conv_mod.DynamicShortConvolution

    value_dim = NUM_HEADS * HEAD_DIM * EXPAND_V
    real = DynamicShortConvolution(
        hidden_size=value_dim, kernel_size=CONV_K,
        generator_input_size=HIDDEN, generator_reduction=GEN_REDUCTION,
        implementation="naive")
    real.eval()
    _randomize(real, seed=5)

    torch.manual_seed(6)
    generator_input = torch.randn(1, S, HIDDEN)
    x = torch.randn(1, S, value_dim)
    with torch.no_grad():
        hf_out, _ = real(x=x, generator_input=generator_input)

    from runtime.jet_nemotron import _dynamic_causal_conv1d, _get_dynamic_conv_kernel

    sd = real.state_dict()
    block_prefix = "layer0"  # _get_dynamic_conv_kernel appends ".self_attn.dynamic_conv1d" itself
    w = {f"{block_prefix}.self_attn.dynamic_conv1d.{k}": mx.array(v.numpy()) for k, v in sd.items()}
    x_mx = mx.array(x.numpy())
    gen_mx = mx.array(generator_input.numpy())
    kernels = _get_dynamic_conv_kernel(gen_mx, w, block_prefix, value_dim, CONV_K)
    runtime_out, _new_history = _dynamic_causal_conv1d(x_mx, kernels, CONV_K)
    runtime_out = runtime_out * mx.sigmoid(runtime_out)  # the module's own SiLU
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-5, f"dynamic conv oracle mismatch: max abs diff {max_abs_diff}"


@_model_skip
def test_jet_block_multi_step_decode_matches_real_stepwise_cache():
    """Regression test for a real bug an early draft of this port had: a
    dynamic conv's kernel is freshly generated per position, but the
    CONVOLUTION WINDOW still needs genuine prior V values carried across
    calls, not zero-padding every single-token decode step as if it were
    the start of a new sequence (that draft produced a plausible-looking
    but wrong first token followed by rapidly degrading garbage on a real
    checkpoint, exactly this failure signature). Feeds tokens ONE AT A TIME
    through the real JetBlock (with a real JetNemotronCache) and through
    runtime/jet_nemotron.py's _jet_block (with a real KDAStateCache),
    asserting every step's output matches -- not just a single one-shot
    forward pass, which cannot exercise this bug at all."""
    jet_block_mod = _load_real_jet_block()
    JetBlockConfig = jet_block_mod.JetBlockConfig
    JetBlock = jet_block_mod.JetBlock
    kv_cache_mod = sys.modules["_jet_oracle_real.kv_cache"]
    JetNemotronCache = kv_cache_mod.JetNemotronCache

    jbc = JetBlockConfig(
        mode="fused_recurrent", expand_v=EXPAND_V, num_heads=NUM_HEADS,
        head_dim=HEAD_DIM, norm_eps=1e-6, conv_size=CONV_K,
        dconv_generator_reduction=GEN_REDUCTION, dconv_implementation="naive")
    real = JetBlock(
        hidden_size=HIDDEN, initializer_range=0.02, layer_idx=0,
        jet_block_config=jbc)
    real.eval()
    _randomize(real, seed=9)
    with torch.no_grad():
        real.A_log.copy_(torch.log(torch.empty_like(real.A_log).uniform_(0, 16)))

    torch.manual_seed(10)
    h_all = torch.randn(1, S, HIDDEN)

    real_cache = JetNemotronCache()
    hf_outputs = []
    with torch.no_grad():
        for t in range(S):
            step_out, real_cache = real(
                h_all[:, t:t + 1, :], past_key_value=real_cache,
                attention_mask=None, use_cache=True)
            hf_outputs.append(step_out)
    hf_out = torch.cat(hf_outputs, dim=1)

    sd = real.state_dict()
    prefix = "layer0"
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}
    rcfg = _runtime_config()

    state_cache = KDAStateCache(1)
    runtime_outputs = []
    for t in range(S):
        h_step = mx.array(h_all[:, t:t + 1, :].numpy())
        step_out = _jet_block(h_step, w, prefix, rcfg, state_cache, 0)
        mx.eval(step_out)
        runtime_outputs.append(step_out)
    runtime_out = mx.concatenate(runtime_outputs, axis=1)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, (
        f"multi-step decode oracle mismatch: max abs diff {max_abs_diff}")
