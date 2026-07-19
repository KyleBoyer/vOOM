"""F93 real-weight, full-scale MoE layer oracle for Kimi K2.5, run against
the REAL modeling_deepseek.py using REAL checkpoint weights (layer 4 --
the same layer test_f93_k25_int4_dequant.py already validates dequant on).

Unlike test_f93_k25_moe_oracle.py (synthetic random weights, a tiny 6-expert
config), this exercises the full production path together at REAL scale:
real gate routing with the checkpoint's actual trained weight + bias values
across all 384 experts, real INT4 expert dequantization via
WeightStore.fetch() (the exact fetch path a live request uses), and the
real per-expert combination order -- catching integration bugs (wrong
prefix, wrong dequant wiring, wrong routing at real n_routed_experts scale)
that a small isolated unit test could miss.

Kept cheap: the S=2 random hidden-state test sequence bounds how many
distinct experts real top-8 routing can select (<=16); only those experts'
real weights are ever fetched/dequantized/instantiated on either side --
mirroring how moe_infer() itself skips any expert with zero routed tokens,
not reconstructing that skip logic separately.
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
from runtime.model_loader import WeightStore

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_MODEL_AVAILABLE = (MODEL_DIR / "modeling_deepseek.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K2.5's real modeling_deepseek.py/checkpoint is not available locally "
           "(a real ~554GB model, not fetched in CI)",
)

try:
    import torch as _torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")

# 2026-07-19 real values, models/Kimi-K2.5/config.json text_config.
LAYER = 4
HIDDEN = 7168
INTERMEDIATE = 18432
MOE_INTER = 2048
N_ROUTED, N_SHARED, TOP_K = 384, 1, 8
N_GROUP, TOPK_GROUP = 1, 1
ROUTED_SCALING = 2.827
S = 2  # small on purpose -- bounds distinct experts real routing can pick to <= S*TOP_K


def _patch_transformers_shims() -> None:
    """Same version-drift shim as the other F93 K2.5 oracle tests."""
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


def _real_config(cfg_mod):
    cfg = cfg_mod.DeepseekV3Config(
        vocab_size=163840, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        moe_intermediate_size=MOE_INTER, num_hidden_layers=1,
        num_attention_heads=64, num_key_value_heads=64,
        n_shared_experts=N_SHARED, n_routed_experts=N_ROUTED,
        num_experts_per_tok=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING, topk_method="noaux_tc",
        norm_topk_prob=True, scoring_func="sigmoid", hidden_act="silu",
    )
    cfg.rope_scaling = None
    return cfg


def _runtime_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="kimi_k25", hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=1, num_attention_heads=64, num_key_value_heads=64,
        vocab_size=163840, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=192, eos_token_ids=(), torch_dtype="bfloat16",
        num_experts=N_ROUTED, num_experts_per_tok=TOP_K,
        n_group=N_GROUP, topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING, norm_topk_prob=True, n_shared_experts=N_SHARED,
    )


def _expert_forward_torch(model_mod, cfg, fetched: dict, prefix_key: str, x: torch.Tensor,
                           intermediate_size: int) -> torch.Tensor:
    """A real DeepseekV3MLP instantiated (and .forward()'d) with real weights
    for exactly one expert -- not the full 384-expert nn.ModuleList (which
    would allocate ~67GB of float32 params on this 16GB machine)."""
    mlp = model_mod.DeepseekV3MLP(cfg, intermediate_size=intermediate_size)
    mlp.eval()
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.from_numpy(
            np.array(fetched[f"{prefix_key}.gate_proj.weight"].astype(mx.float32))))
        mlp.up_proj.weight.copy_(torch.from_numpy(
            np.array(fetched[f"{prefix_key}.up_proj.weight"].astype(mx.float32))))
        mlp.down_proj.weight.copy_(torch.from_numpy(
            np.array(fetched[f"{prefix_key}.down_proj.weight"].astype(mx.float32))))
        return mlp(x)


@_model_skip
@_torch_skip
def test_moe_layer_matches_real_deepseek_on_real_k25_weights():
    model_mod, cfg_mod = _load_real_modeling_deepseek()
    cfg = _real_config(cfg_mod)
    prefix = f"model.layers.{LAYER}"

    store = WeightStore(str(MODEL_DIR))

    # 1. Real gate at full production scale (384 experts, real trained
    #    weight + bias) -- cheap, ~11MB total, no dequant involved (router
    #    weights are plain bf16 in the checkpoint).
    gate_names = [f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.gate.e_score_correction_bias"]
    gate_out, _, _ = store.fetch(gate_names)
    mx.eval(list(gate_out.values()))

    gate = model_mod.MoEGate(cfg)
    gate.eval()
    with torch.no_grad():
        gate.weight.copy_(torch.from_numpy(np.array(gate_out[gate_names[0]].astype(mx.float32))))
        gate.e_score_correction_bias.copy_(
            torch.from_numpy(np.array(gate_out[gate_names[1]].astype(mx.float32))))

    torch.manual_seed(0)
    h_torch = torch.randn(1, S, HIDDEN, dtype=torch.float32) * 0.3
    with torch.no_grad():
        topk_idx, topk_weight = gate(h_torch)

    selected = sorted(set(int(i) for i in topk_idx.reshape(-1).tolist()))
    assert 1 <= len(selected) <= S * TOP_K

    # 2. Real routed + shared expert weights, fetched through the actual
    #    production WeightStore.fetch() path (INT4 dequant included) --
    #    only for the small set of experts real routing just selected.
    fetch_names = [f"{prefix}.mlp.shared_experts.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    for e in selected:
        fetch_names += [f"{prefix}.mlp.experts.{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    fetched, _, _ = store.fetch(fetch_names)
    mx.eval(list(fetched.values()))

    # 3. Real reference combination -- same structure as
    #    DeepseekV3MoE.moe_infer's per-expert loop (weighted sum of real
    #    expert outputs) + shared-expert addition, using real
    #    DeepseekV3MLP.forward() for each individual expert module instead
    #    of the full 384-wide nn.ModuleList.
    with torch.no_grad():
        y = torch.zeros(1, S, HIDDEN, dtype=torch.float32)
        flat_idx = topk_idx.view(-1)
        flat_pos = torch.arange(S).repeat_interleave(TOP_K)
        for e in selected:
            mask = flat_idx == e
            positions = flat_pos[mask]
            weights = topk_weight.view(-1)[mask].view(-1, 1)
            out = _expert_forward_torch(
                model_mod, cfg, fetched, f"{prefix}.mlp.experts.{e}",
                h_torch[:, positions, :], MOE_INTER)
            y[:, positions, :] += out * weights.unsqueeze(0)
        y = y + _expert_forward_torch(
            model_mod, cfg, fetched, f"{prefix}.mlp.shared_experts", h_torch, MOE_INTER * N_SHARED)

    # 4. Our production routing + combination math, driven by the exact
    #    same real weights.
    w = dict(gate_out)
    w.update(fetched)
    rcfg = _runtime_cfg()
    h_mx = mx.array(h_torch.numpy()).astype(mx.bfloat16)
    idx, pw = _route_experts(h_mx, w, prefix, rcfg)
    mx.eval(idx, pw)
    groups = _group_routes(idx, pw)
    assert sorted(groups.keys()) == selected, (
        f"routing mismatch at real 384-expert scale: real picked {selected}, "
        f"ours picked {sorted(groups.keys())}")

    out = mx.zeros_like(h_mx)
    for e in sorted(groups):
        plist = groups[e]
        positions = [p for p, _ in plist]
        route_weights = mx.array([wt for _, wt in plist]).astype(mx.float32)
        ey = _swiglu(h_mx[:, positions, :], w, f"{prefix}.mlp.experts.{e}")
        contribution = (ey * route_weights[None, :, None]).astype(h_mx.dtype)
        out = out.at[:, positions, :].add(contribution)
    out = out + _swiglu(h_mx, w, f"{prefix}.mlp.shared_experts")
    mx.eval(out)

    hf_np = y.numpy()
    runtime_np = np.array(out.astype(mx.float32))
    assert hf_np.shape == runtime_np.shape
    max_abs_diff = np.max(np.abs(hf_np - runtime_np))
    rel_l2 = np.linalg.norm(hf_np - runtime_np) / np.linalg.norm(hf_np)
    # Both sides read the SAME real bf16-native checkpoint weights, but the
    # runtime path computes in bf16 (matching production) while the torch
    # reference computes in float32 -- so this bounds bf16-vs-fp32 rounding
    # accumulated over 7168/2048-wide reductions, not exact equality (unlike
    # the float32-vs-float32 synthetic oracles elsewhere in F93, which use
    # 1e-3). A wiring bug (wrong prefix, wrong expert, wrong combine order)
    # would blow well past this, not sit at bf16-rounding scale.
    assert rel_l2 < 0.05, (
        f"K2.5 real-weight MoE layer oracle mismatch: rel_l2={rel_l2}, "
        f"max_abs_diff={max_abs_diff}")
