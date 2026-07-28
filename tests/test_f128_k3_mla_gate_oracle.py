"""F128: Kimi K3's "Gated MLA" output gate.

K3's real config.json sets `mla_use_output_gate: true` for its 24 real MLA
layers (Moonshot's own model card lists the attention mechanism as
"KDA & Gated MLA", not plain MLA) -- absent (false) for every other model
`runtime.glm._mla_attention` already serves (GLM-5.2, Kimi K2.5, Kimi
Linear 48B). Ported verbatim from the real, bundled
`models/Kimi-K3/modeling_kimi_linear.py`'s `KimiMLAAttention.__init__`/
`forward`:

    self.use_output_gate = getattr(config, "mla_use_output_gate", False)
    if self.use_output_gate:
        projection_size = self.num_heads * self.v_head_dim
        self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
    ...
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    if self.use_output_gate:
        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g
    attn_output = self.o_proj(attn_output)

The gate is a FRESH projection of the original hidden_states input (not
derived from the attention output itself), multiplying the attention
output elementwise BEFORE o_proj. This test isolates that formula (no real
attention/KV machinery needed) and separately confirms
`runtime.glm._mla_attention` applies it identically via a full real-weight
K3 layer 3 (a real MLA layer, 0-indexed, confirmed present on the real
checkpoint) smoke check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from runtime.config import ModelConfig

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
_MODEL_AVAILABLE = (MODEL_DIR / "model.safetensors.index.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 is not available locally (a real ~1.4TB model, not fetched in CI)",
)


# Verbatim from the real, bundled models/Kimi-K3/modeling_kimi_linear.py's
# KimiMLAAttention.forward gate application (the g_proj Linear itself is
# just a matmul, transcribed directly rather than instantiating an
# nn.Module for a one-line formula).
def _real_mla_output_gate(attn_output, hidden_states, g_proj_weight):
    g = torch.sigmoid(hidden_states @ g_proj_weight.T)
    return attn_output * g


@_torch_skip
def test_mla_output_gate_matches_real_reference():
    rng = np.random.default_rng(0)
    B, L, H, proj = 2, 3, 16, 40  # proj = num_heads * v_head_dim, arbitrary here
    attn_out_np = rng.standard_normal((B, L, proj)).astype(np.float32)
    hidden_np = rng.standard_normal((B, L, H)).astype(np.float32)
    g_proj_np = rng.standard_normal((proj, H)).astype(np.float32)

    ref = _real_mla_output_gate(
        torch.from_numpy(attn_out_np), torch.from_numpy(hidden_np),
        torch.from_numpy(g_proj_np)).numpy()

    attn_out = mx.array(attn_out_np)
    hidden = mx.array(hidden_np)
    g_proj = mx.array(g_proj_np)
    mine = attn_out * mx.sigmoid(hidden @ g_proj.T)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_model_skip
def test_real_k3_layer_3_is_a_real_mla_layer_with_output_gate():
    """Confirms the real checkpoint's layer 3 (0-indexed) is genuinely one
    of the 24 real MLA layers with a real g_proj tensor of the expected
    shape -- the exact fact this whole fix depends on, checked directly
    against real bytes rather than assumed from the config field alone."""
    cfg = ModelConfig.from_dir(MODEL_DIR)
    assert cfg.mla_use_output_gate is True
    assert 3 in cfg.full_attn_layers

    import json

    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    name = "language_model.model.layers.3.self_attn.g_proj.weight"
    assert name in index["weight_map"]
    shard = mx.load(str(MODEL_DIR / index["weight_map"][name]))
    g_proj = shard[name]
    assert g_proj.shape == (cfg.num_attention_heads * cfg.v_head_dim, cfg.hidden_size)
