"""F93 real-weight MLA attention oracle for Kimi K2.5, run against the REAL
modeling_deepseek.py using REAL checkpoint weights (layer 4 -- the same
layer the MoE and dequant oracles use).

Unlike test_f93_k25_mla_oracle.py (synthetic random weights, a tiny
64-hidden config, a shrunk original_max_position_embeddings to force-
exercise the YaRN ramp region), this loads layer 4's REAL self_attn weights
(hidden=7168, q_lora_rank=1536, kv_lora_rank=512, 64 heads) and the
checkpoint's real YaRN rope_scaling config through the production
WeightStore.fetch() path, and compares against the real
DeepseekV3Attention forward on the exact same weights. This is the
integration-level check: real weight names/shapes/prefix at production
scale, not just the isolated math on a hand-picked tiny config.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from runtime.config import ModelConfig
from runtime.glm import _mla_attention
from runtime.kv_cache import KVCache
from runtime.model_loader import WeightStore

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_MODEL_AVAILABLE = (MODEL_DIR / "modeling_deepseek.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K2.5's real modeling_deepseek.py/checkpoint is not available locally "
           "(a real ~554GB model, not fetched in CI)",
)

# 2026-07-19 real values, models/Kimi-K2.5/config.json text_config.
LAYER = 4
HIDDEN, N_HEADS = 7168, 64
DN, DR, DV = 128, 64, 128  # qk_nope, qk_rope, v_head
KV_LORA, Q_LORA = 512, 1536
ROPE_THETA = 50000.0
YARN_SCALING = {
    "type": "yarn", "factor": 64.0, "beta_fast": 32.0, "beta_slow": 1.0,
    "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 4096,
}
S = 5


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


def _real_config(cfg_mod):
    cfg = cfg_mod.DeepseekV3Config(
        vocab_size=163840, hidden_size=HIDDEN, intermediate_size=18432,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        kv_lora_rank=KV_LORA, q_lora_rank=Q_LORA, qk_rope_head_dim=DR,
        v_head_dim=DV, qk_nope_head_dim=DN, max_position_embeddings=262144,
        rope_theta=ROPE_THETA, rope_scaling=YARN_SCALING,
        attention_bias=False, attention_dropout=0.0,
    )
    cfg.rope_scaling = YARN_SCALING
    return cfg


def _runtime_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="kimi_k25", hidden_size=HIDDEN, intermediate_size=18432,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        vocab_size=163840, rms_norm_eps=1e-5, rope_theta=ROPE_THETA,
        max_position_embeddings=262144, tie_word_embeddings=False, attention_bias=False,
        head_dim=DN + DR, eos_token_ids=(), torch_dtype="bfloat16",
        q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=DN,
        qk_rope_head_dim=DR, v_head_dim=DV, mla_latent_norm_eps=1e-5,
        rope_interleave=True, mla_use_nope=False, rope_scaling=YARN_SCALING,
    )


def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    q_pos = torch.arange(seq_len)[:, None]
    k_pos = torch.arange(seq_len)[None, :]
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    return mask[None, None, :, :]


_ATTN_TENSORS = (
    "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
    "o_proj.weight",
)


@_model_skip
def test_mla_layer_matches_real_deepseek_on_real_k25_weights():
    model_mod, cfg_mod = _load_real_modeling_deepseek()
    cfg = _real_config(cfg_mod)
    prefix = f"model.layers.{LAYER}"

    store = WeightStore(str(MODEL_DIR))
    names = [f"{prefix}.self_attn.{t}" for t in _ATTN_TENSORS]
    fetched, _, _ = store.fetch(names)
    mx.eval(list(fetched.values()))

    real = model_mod.DeepseekV3Attention(cfg, layer_idx=0)
    real.eval()
    with torch.no_grad():
        sd = real.state_dict()
        for t in _ATTN_TENSORS:
            key = f"{prefix}.self_attn.{t}"
            sd[t].copy_(torch.from_numpy(np.array(fetched[key].astype(mx.float32))))

    torch.manual_seed(1)
    h_torch = torch.randn(1, S, HIDDEN, dtype=torch.float32) * 0.1
    with torch.no_grad():
        hf_out, _, _ = real(h_torch, attention_mask=_causal_mask_torch(S), position_ids=None)

    # WeightStore's logical names already match _mla_attention's expected
    # weight keys 1:1 -- no renaming needed.
    w = dict(fetched)

    rcfg = _runtime_cfg()
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy()).astype(mx.bfloat16)
    runtime_out = _mla_attention(h_mx, w, prefix, rcfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    hf_np = hf_out.detach().numpy()
    runtime_np = np.array(runtime_out.astype(mx.float32))
    assert hf_np.shape == runtime_np.shape
    rel_l2 = np.linalg.norm(hf_np - runtime_np) / np.linalg.norm(hf_np)
    # Same bf16-runtime-vs-fp32-reference rationale as the real-weight MoE
    # oracle: both sides read the same real bf16 checkpoint weights, but
    # compute in different precisions, so exact float32-oracle tolerances
    # (1e-3, used by the synthetic same-precision MLA oracle) do not apply.
    assert rel_l2 < 0.05, f"K2.5 real-weight MLA layer oracle mismatch: rel_l2={rel_l2}"
