"""F93 numerical oracle for Kimi K2.5's MoE gate + expert combination math,
run against the REAL modeling_deepseek.py (MoEGate + DeepseekV3MoE).

Follows the same "verify against real source before reusing a shared code
path, don't assume it transfers" discipline that caught two real bugs
elsewhere in F93 (rope_interleave, K2.5's MoE gate not needing Kimi
Linear's bias-aliasing quirk). This test specifically checks: is the real
DeepseekV3 MoEGate.forward's `scores_for_choice = scores.view(...) +
bias` (an ordinary `+`, confirmed NOT an in-place `+=` by direct source
reading 2026-07-19) actually equivalent to runtime.glm._route_experts, and
does DeepseekV3MoE.forward's combination order (routed-weighted-sum, THEN
add shared experts) match run_glm_block's assumed order.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm import _group_routes, _route_experts
from runtime.layer_runner import _swiglu

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_MODEL_AVAILABLE = (MODEL_DIR / "modeling_deepseek.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K2.5's real modeling_deepseek.py is not available locally "
           "(a real checkpoint's source file, not fetched in CI)",
)

HIDDEN = 32
NUM_EXPERTS, TOP_K, MOE_INTER = 6, 2, 16
N_GROUP, TOPK_GROUP = 1, 1  # matches K2.5's real checkpoint (confirmed 2026-07-18)
S = 7


def _patch_transformers_shims() -> None:
    import transformers.utils.import_utils as iu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False


def _load_real_modeling_deepseek():
    import importlib.util
    import sys
    import types

    _patch_transformers_shims()
    pkg_name = "_f93_k25_real"
    if pkg_name in sys.modules and hasattr(sys.modules[pkg_name], "_loaded_ok"):
        return sys.modules[f"{pkg_name}.modeling_deepseek"], sys.modules[f"{pkg_name}.configuration_deepseek"]
    sys.modules.pop(pkg_name, None)

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

    cfg_mod = _load("configuration_deepseek")
    model_mod = _load("modeling_deepseek")
    pkg._loaded_ok = True
    return model_mod, cfg_mod


def _tiny_config(cfg_mod):
    cfg = cfg_mod.DeepseekV3Config(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        moe_intermediate_size=MOE_INTER, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        routed_scaling_factor=1.7, topk_method="noaux_tc",
        norm_topk_prob=True, scoring_func="sigmoid",
    )
    cfg.rope_scaling = None
    return cfg


def _runtime_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="kimi_k25", hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=100, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=16, eos_token_ids=(), torch_dtype="float32",
        num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K,
        moe_expert_prefix="mlp.experts", n_group=N_GROUP, topk_group=TOPK_GROUP,
        routed_scaling_factor=1.7, norm_topk_prob=True, n_shared_experts=1,
    )


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.3)


@_model_skip
def test_moe_gate_and_experts_match_real_deepseek():
    model_mod, cfg_mod = _load_real_modeling_deepseek()
    cfg = _tiny_config(cfg_mod)

    real = model_mod.DeepseekV3MoE(cfg)
    real.eval()
    _randomize(real, seed=5)
    with torch.no_grad():
        real.gate.e_score_correction_bias.normal_(mean=0.0, std=0.3)

    torch.manual_seed(6)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out = real(h_torch)

    sd = real.state_dict()
    prefix = "layer0"
    w = {f"{prefix}.mlp.{k}": mx.array(v.numpy()) for k, v in sd.items()}

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
        y = _swiglu(h_mx[:, positions, :], w, f"{prefix}.mlp.experts.{e}")
        contribution = (y * route_weights[None, :, None]).astype(h_mx.dtype)
        out = out.at[:, positions, :].add(contribution)
    out = out + _swiglu(h_mx, w, f"{prefix}.mlp.shared_experts")
    mx.eval(out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(out)
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    assert max_abs_diff < 1e-3, f"K2.5 MoE oracle mismatch: max abs diff {max_abs_diff}"
