"""Numeric oracle for the afmoe architecture (Arcee AI's Trinity Nano/Mini),
new to this project 2026-07-24.

Real reference: the checkpoint's own `modeling_afmoe.py`/`configuration_
afmoe.py` (bundled with `models/Trinity-Nano-Preview`, a real downloaded
checkpoint, not a synthetic fixture). Calls the real `AfmoeDecoderLayer.
forward()` directly with manually-built `position_embeddings`/
`attention_mask` tensors, bypassing `AfmoeModel.forward()`'s top-level
wrapper -- NOT to avoid testing real code, but because that wrapper hits
three separate transformers-version-skew bugs unrelated to this
architecture's own math (the bundled file was written against an older
transformers API than the installed 5.13.0):

1. `ROPE_INIT_FUNCTIONS` dropped the `"default"` key (only linear/dynamic/
   yarn/longrope/llama3/proportional remain) -- `AfmoeRotaryEmbedding.
   __init__` looks it up unconditionally for non-scaled RoPE. Patched with
   the standard, well-known non-scaled formula (same one this project
   already uses in runtime/gptoss.py etc.) -- not a workaround for this
   architecture's math, a restoration of what any transformers version
   would have provided.
2. `PreTrainedModel.post_init()`'s newer weight-init machinery expects a
   `compute_default_rope_parameters` method the bundled rotary module
   doesn't implement. Skipped entirely (`AfmoePreTrainedModel.post_init =
   lambda self: None`) -- irrelevant for an oracle, which only needs SOME
   concrete comparable values, not "properly" initialized training
   statistics.
3. `create_causal_mask()`'s newer signature renamed `input_embeds` ->
   `inputs_embeds` -- only hit by `AfmoeModel.forward()`'s own mask-
   construction call, not by `AfmoeDecoderLayer.forward()` itself. Avoided
   by building masks directly (same causal/sliding-window construction
   `runtime/gptoss.py::_attention_gptoss` already uses) and calling the
   decoder layer directly -- this is also the RIGHT oracle granularity
   (matching F92/F93's own "extract state_dict, feed identical weights
   through the mx implementation" methodology) rather than a workaround.

Verified: a dense (`layer < num_dense_layers`)/sliding_attention layer
(layer 0) and a MoE/full_attention layer (layer 2), matching real
`AfmoeDecoderLayer.forward()` to ~1.2e-6 max abs diff (near float32
machine precision) both times -- confirms GQA attention with per-head
Q/K RMSNorm, RoPE gated to sliding layers only (full layers get NoPE),
sigmoid attention-output gating, the 4-norm "sandwich" layer structure,
and the sigmoid-scored/expert-bias-selection-only MoE router are all
correct in `runtime/afmoe.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parent.parent
_TRINITY_DIR = ROOT / "models" / "Trinity-Nano-Preview"
_skip = pytest.mark.skipif(
    not (_TRINITY_DIR / "modeling_afmoe.py").exists(),
    reason="Trinity-Nano-Preview's real modeling_afmoe.py is not available "
           "locally (a real ~11GB checkpoint, not fetched in CI)")

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_skip_no_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed in this venv")


def _build_reference(cfg_kwargs):
    """Import the real reference classes, apply the version-skew shims
    documented in this module's docstring, and return the config class."""
    sys.path.insert(0, str(_TRINITY_DIR))
    import torch
    from configuration_afmoe import AfmoeConfig
    from modeling_afmoe import AfmoeDecoderLayer, AfmoePreTrainedModel
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _default_rope_init(config, device=None, seq_len=None):
        dim = config.head_dim
        base = config.rope_theta
        return (1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)),
                1.0)
    ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    AfmoePreTrainedModel.post_init = lambda self: None

    cfg = AfmoeConfig(pad_token_id=0, attn_implementation="eager", **cfg_kwargs)
    return torch, AfmoeConfig, AfmoeDecoderLayer, cfg


def _position_embeddings(torch, seq_len, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _masks(torch, seq_len, sliding_window):
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    causal = k <= q
    sliding = causal & (k > q - sliding_window)

    def to_float(mask):
        m = torch.zeros(seq_len, seq_len)
        m.masked_fill_(~mask, float("-inf"))
        return m[None, None]
    return to_float(causal), to_float(sliding)


@_skip
@_skip_no_torch
def test_afmoe_dense_sliding_and_moe_full_layers_match_real_reference():
    torch, AfmoeConfig, AfmoeDecoderLayer, cfg = _build_reference(dict(
        num_hidden_layers=6, vocab_size=64, hidden_size=32, intermediate_size=64,
        moe_intermediate_size=16, num_dense_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
        global_attn_every_n_layers=3, sliding_window=4, mup_enabled=True,
    ))
    torch.manual_seed(0)

    L = 10
    cos, sin = _position_embeddings(torch, L, cfg.head_dim, cfg.rope_theta)
    full_mask, sliding_mask = _masks(torch, L, cfg.sliding_window)
    x = torch.randn(1, L, cfg.hidden_size)

    from runtime.config import ModelConfig
    from runtime.kv_cache import KVCache
    from runtime.afmoe import run_afmoe_block

    mx_cfg = ModelConfig(
        model_type="afmoe", hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        vocab_size=cfg.vocab_size, rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings, tie_word_embeddings=False,
        attention_bias=False, eos_token_ids=(), torch_dtype="float32",
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts, num_experts_per_tok=cfg.num_experts_per_tok,
        n_shared_experts=cfg.num_shared_experts,
        first_k_dense_replace=cfg.num_dense_layers,
        norm_topk_prob=cfg.route_norm, routed_scaling_factor=cfg.route_scale,
        layer_types=tuple(cfg.layer_types), sliding_window=cfg.sliding_window,
    )

    def to_mx(state_dict, prefix):
        return {f"{prefix}.{k}": mx.array(v.numpy()) for k, v in state_dict.items()}

    x_mx = mx.array(x.numpy())

    def check(layer_idx, mask):
        layer = AfmoeDecoderLayer(cfg, layer_idx=layer_idx)
        layer.eval()
        with torch.no_grad():
            ref_out = layer(x, attention_mask=mask, position_embeddings=(cos, sin))
        assert not torch.isnan(ref_out).any()

        w = to_mx(layer.state_dict(), f"model.layers.{layer_idx}")

        def get_experts(layer_i, expert_ids, positions=None):
            return {e: w for e in expert_ids}

        kv = KVCache(num_layers=mx_cfg.num_hidden_layers)
        out = run_afmoe_block(
            x_mx, w, f"model.layers.{layer_idx}", mx_cfg, kv, layer_idx, offset=0,
            get_experts=get_experts)
        mx.eval(out)
        diff = float(mx.max(mx.abs(out - mx.array(ref_out.numpy()))))
        assert diff < 1e-3, f"layer {layer_idx}: max abs diff {diff:.3e}"

    check(0, sliding_mask)  # dense, sliding_attention (layer_types[0])
    check(2, full_mask)     # MoE, full_attention (layer_types[2])
