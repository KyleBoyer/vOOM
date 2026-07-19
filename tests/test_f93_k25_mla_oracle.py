"""F93 numerical oracle for Kimi K2.5's language-model attention/MoE math,
run against the REAL modeling_deepseek.py (no fla-core stubbing needed --
unlike Kimi Linear, K2.5's language model has no Triton-only dependencies).

2026-07-19 finding (later fixed same day): K2.5's real config.json declares
YaRN RoPE scaling (type=yarn, factor=64, beta_fast=32, beta_slow=1,
mscale=1.0, mscale_all_dim=1.0, original_max_position_embeddings=4096) --
runtime/glm.py's _mla_attention originally applied only plain RoPE, no
YaRN at all, which this file's tests first caught at 0.59-0.81 max abs
diff. YaRN support was then added (glm.py::_yarn_rope_params): the
frequency-ramp math and cos/sin scale turned out to be algebraically
identical to runtime/rope.py's existing yarn_parameters() (verified by
direct derivation against the real DeepseekV3YarnRotaryEmbedding, reused
directly) -- but a SEPARATE softmax_scale multiplier
(`yarn_get_mscale(factor, mscale_all_dim) ** 2`, applied in the real
DeepseekV3Attention.__init__) is NOT covered by runtime/rope.py at all and
needed new code. For K2.5's own config values (mscale == mscale_all_dim ==
1.0) the cos/sin-scale ratio is a no-op (1.0) while the softmax multiplier
is not (~2.0) -- both are exercised correctly below (test now expects a
close match, not a mismatch).
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

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_MODEL_AVAILABLE = (MODEL_DIR / "modeling_deepseek.py").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K2.5's real modeling_deepseek.py is not available locally "
           "(a real checkpoint's source file, not fetched in CI)",
)

HIDDEN, N_HEADS = 64, 4
DN, DR, DV = 12, 8, 16  # qk_nope, qk_rope, v_head
KV_LORA, Q_LORA = 24, 32
S = 7

# K2.5's real declared YaRN params (2026-07-19, models/Kimi-K2.5/config.json
# text_config.rope_scaling) -- only original_max_position_embeddings is
# lowered here (4096 -> 64) so a 7-token test sequence still exercises the
# "already past the original context" ramp region instead of always
# landing in the trivial low-position regime.
YARN_SCALING = {
    "type": "yarn", "factor": 64.0, "beta_fast": 32.0, "beta_slow": 1.0,
    "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 64,
}


def _patch_transformers_shims() -> None:
    """`transformers` 5.13.0 (installed here) removed `is_torch_fx_available`
    from `transformers.utils.import_utils` -- modeling_deepseek.py (written
    against an older transformers) imports it at module level but never
    calls it in any code path this test exercises (no FX tracing here).
    Same category of version-drift shim as F92's OutputRecorder/
    check_model_inputs patch for modeling_kimi.py."""
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


def _tiny_config(cfg_mod, *, with_yarn: bool):
    cfg = cfg_mod.DeepseekV3Config(
        vocab_size=100, hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        kv_lora_rank=KV_LORA, q_lora_rank=Q_LORA, qk_rope_head_dim=DR,
        v_head_dim=DV, qk_nope_head_dim=DN, max_position_embeddings=64,
        rope_theta=10000.0, rope_scaling=(YARN_SCALING if with_yarn else None),
        attention_bias=False, attention_dropout=0.0,
    )
    # transformers 5.13.0's PretrainedConfig base class auto-populates
    # rope_scaling with a {"rope_type": "default", ...} dict even when the
    # subclass constructor is explicitly passed None (a generic
    # "rope_parameters" standardization this old DeepSeek-V3 code predates)
    # -- override it back to what was actually requested.
    cfg.rope_scaling = YARN_SCALING if with_yarn else None
    return cfg


def _causal_mask_torch(seq_len: int) -> torch.Tensor:
    q_pos = torch.arange(seq_len)[:, None]
    k_pos = torch.arange(seq_len)[None, :]
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    return mask[None, None, :, :]


def _runtime_cfg() -> ModelConfig:
    return ModelConfig(
        model_type="kimi_k25", hidden_size=HIDDEN, intermediate_size=HIDDEN * 4,
        num_hidden_layers=1, num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS,
        vocab_size=100, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False, attention_bias=False,
        head_dim=DN + DR, eos_token_ids=(), torch_dtype="float32",
        q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=DN,
        qk_rope_head_dim=DR, v_head_dim=DV, mla_latent_norm_eps=1e-6,
        # F93: DeepSeek-V3-family real apply_rotary_pos_emb de-interleaves
        # before rotate_half -- equivalent to traditional=True on the raw
        # checkpoint weights. Verified: False gave 0.81 max diff against
        # the real modeling_deepseek.py, True gives 9.5e-7.
        rope_interleave=True, mla_use_nope=False,
    )


def _randomize(module: torch.nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(mean=0.0, std=0.3)


@_model_skip
def test_mla_matches_real_deepseek_when_no_yarn_is_configured():
    """Sanity baseline: with rope_scaling=None (plain RoPE, matching what
    _mla_attention actually implements), the two should agree -- proves
    the MLA structure/q_lora/RoPE-application itself is correct, isolating
    the YaRN gap as the ONLY known discrepancy (checked next)."""
    model_mod, cfg_mod = _load_real_modeling_deepseek()
    cfg = _tiny_config(cfg_mod, with_yarn=False)

    real = model_mod.DeepseekV3Attention(cfg, layer_idx=0)
    real.eval()
    _randomize(real, seed=1)

    torch.manual_seed(2)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out, _, _ = real(h_torch, attention_mask=_causal_mask_torch(S), position_ids=None)

    sd = real.state_dict()
    prefix = "layer0"
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}

    rcfg = _runtime_cfg()
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _mla_attention(h_mx, w, prefix, rcfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    max_diff = np.max(np.abs(hf_out.detach().numpy() - np.array(runtime_out)))
    assert max_diff < 1e-3, f"no-YaRN MLA baseline mismatch: {max_diff}"


@_model_skip
def test_mla_matches_real_deepseek_with_yarn_implemented():
    """K2.5's ACTUAL config (rope_scaling=yarn) vs runtime/glm.py's
    _mla_attention. 2026-07-19: YaRN support was added to _mla_attention
    (_yarn_rope_params in glm.py) after this test first proved the gap at
    0.59 max abs diff with no YaRN applied -- this now exercises the real
    implementation via cfg.rope_scaling and expects it to match closely."""
    model_mod, cfg_mod = _load_real_modeling_deepseek()
    cfg = _tiny_config(cfg_mod, with_yarn=True)

    real = model_mod.DeepseekV3Attention(cfg, layer_idx=0)
    real.eval()
    _randomize(real, seed=1)

    torch.manual_seed(2)
    h_torch = torch.randn(1, S, HIDDEN)
    with torch.no_grad():
        hf_out, _, _ = real(h_torch, attention_mask=_causal_mask_torch(S), position_ids=None)

    sd = real.state_dict()
    prefix = "layer0"
    w = {f"{prefix}.self_attn.{k}": mx.array(v.numpy()) for k, v in sd.items()}

    rcfg = _runtime_cfg()
    rcfg.rope_scaling = YARN_SCALING
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _mla_attention(h_mx, w, prefix, rcfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    max_diff = np.max(np.abs(hf_out.detach().numpy() - np.array(runtime_out)))
    assert max_diff < 1e-3, f"K2.5 YaRN MLA oracle mismatch: max abs diff {max_diff}"
