"""F93 numerical oracle for Kimi K2.5's language-model attention/MoE math,
run against the REAL modeling_deepseek.py (no fla-core stubbing needed --
unlike Kimi Linear, K2.5's language model has no Triton-only dependencies).

2026-07-19 finding: K2.5's real config.json declares YaRN RoPE scaling
(type=yarn, factor=64, beta_fast=32, beta_slow=1, mscale=1.0,
mscale_all_dim=1.0, original_max_position_embeddings=4096) -- but
runtime/glm.py's _mla_attention (reused unmodified for kimi_k25, see F93)
only ever applies plain RoPE with no YaRN scaling at all, and no YaRN
wiring exists for glm_moe_dsa/kimi_k25 in engine.py (only qwen2 and
gpt_oss have YaRN support today). This is expected to make the MLA oracle
test below FAIL/mismatch -- that failure is the point: it quantifies a
real, previously undiscovered correctness gap rather than leaving it as an
unverified assumption. Do not "fix" this test to pass by disabling YaRN in
the real reference -- that would hide the gap instead of proving it.

Also confirmed by reading the real DeepseekV3Attention.__init__: its YaRN
mscale application (`softmax_scale *= yarn_get_mscale(factor,
mscale_all_dim) ** 2`) is NOT the same formula as runtime/rope.py's
existing yarn_parameters() (which computes a mscale/mscale_all_dim RATIO,
tuned for a different lineage -- MLX-LM/Qwen-style YaRN). For K2.5's own
config values (mscale == mscale_all_dim == 1.0) the two formulas diverge
concretely: runtime/rope.py's ratio gives attention_scale=1.0 (no-op),
while DeepSeek's real formula gives softmax_scale *= ~2.0. Reusing
runtime/rope.py's yarn_parameters() as-is for K2.5 would be WRONG, not a
shortcut -- a correct fix needs a DeepSeek-V3-specific mscale formula, not
attempted here.
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
def test_mla_yarn_gap_is_real_and_quantified():
    """K2.5's ACTUAL config (rope_scaling=yarn) vs runtime/glm.py's
    _mla_attention, which applies no YaRN. Documents the real gap found
    2026-07-19 rather than leaving it as an unverified assumption -- this
    is EXPECTED to fail/mismatch. If this ever starts passing without
    YaRN being added to _mla_attention, something else changed and needs
    investigating (not a silent green light)."""
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

    rcfg = _runtime_cfg()  # no YaRN fields -- matches _mla_attention's actual capability
    kv = KVCache(num_layers=1)
    h_mx = mx.array(h_torch.numpy())
    runtime_out = _mla_attention(h_mx, w, prefix, rcfg, kv, layer=0, offset=0)
    mx.eval(runtime_out)

    max_diff = np.max(np.abs(hf_out.detach().numpy() - np.array(runtime_out)))
    print(f"\nF93 YaRN gap: real-vs-plain-RoPE max abs diff = {max_diff:.6f} "
          "(expected large -- quantifies the missing-YaRN gap, see module docstring)")
    assert max_diff > 1e-2, (
        "expected a large mismatch quantifying the missing-YaRN gap; got "
        f"{max_diff} -- if this is now small, re-investigate before assuming "
        "the gap is fixed (it isn't, per this file's docstring)"
    )
