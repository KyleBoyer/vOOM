"""Kimi Linear (KDA) smoke tests -- F92, docs/future_lossless_techniques.md.

These are shape/plumbing smoke tests, NOT a numerical-correctness oracle.
This venv has no torch/transformers/einops/fla-core (checked 2026-07-18), so
the real released modeling_kimi.py / fla-org kernels cannot run locally as a
reference. Do not read a pass here as proof the KDA math matches the
released model -- only that it runs, produces finite output of the right
shape on real downloaded weights, and that the stateful (chunked-decode)
API agrees with a single-shot call, i.e. the recurrence is genuinely
incremental rather than silently recomputing from scratch each call.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from runtime.config import ModelConfig
from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.kimi_linear import _kda_attention, run_kimi_linear_block

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-Linear-48B-A3B-Instruct"

# A real, ~98GB checkpoint the project may have local access to -- not
# something to fetch in CI or on a fresh clone. Skip gracefully rather than
# downloading or failing on a model that was never expected to be present.
_MODEL_AVAILABLE = (MODEL_DIR / "config.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-Linear-48B-A3B-Instruct is not available locally "
           "(a real ~98GB model, not fetched in CI)",
)


def _tiny_kda_config(hidden: int, num_heads: int, head_dim: int, conv_kernel: int) -> ModelConfig:
    return ModelConfig(
        model_type="kimi_linear", hidden_size=hidden, intermediate_size=hidden * 4,
        num_hidden_layers=1, num_attention_heads=num_heads, num_key_value_heads=num_heads,
        vocab_size=1000, rms_norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=4096, tie_word_embeddings=False, attention_bias=False,
        head_dim=hidden // num_heads, eos_token_ids=(), torch_dtype="bfloat16",
        kda_head_dim=head_dim, kda_num_heads=num_heads, kda_conv_kernel_size=conv_kernel,
    )


def _random_kda_weights(prefix: str, hidden: int, num_heads: int, head_dim: int, conv_kernel: int) -> dict:
    proj = num_heads * head_dim

    def randw(*shape):
        return mx.random.normal(shape).astype(mx.float32)

    return {
        f"{prefix}.self_attn.q_proj.weight": randw(proj, hidden),
        f"{prefix}.self_attn.k_proj.weight": randw(proj, hidden),
        f"{prefix}.self_attn.v_proj.weight": randw(proj, hidden),
        f"{prefix}.self_attn.q_conv1d.weight": randw(proj, 1, conv_kernel) * 0.1,
        f"{prefix}.self_attn.k_conv1d.weight": randw(proj, 1, conv_kernel) * 0.1,
        f"{prefix}.self_attn.v_conv1d.weight": randw(proj, 1, conv_kernel) * 0.1,
        f"{prefix}.self_attn.A_log": mx.log(mx.random.uniform(1, 16, (1, 1, num_heads, 1))),
        f"{prefix}.self_attn.f_a_proj.weight": randw(head_dim, hidden),
        f"{prefix}.self_attn.f_b_proj.weight": randw(proj, head_dim),
        f"{prefix}.self_attn.dt_bias": randw(proj),
        f"{prefix}.self_attn.b_proj.weight": randw(num_heads, hidden),
        f"{prefix}.self_attn.g_a_proj.weight": randw(head_dim, hidden),
        f"{prefix}.self_attn.g_b_proj.weight": randw(proj, head_dim),
        f"{prefix}.self_attn.o_norm.weight": mx.ones((head_dim,)),
        f"{prefix}.self_attn.o_proj.weight": randw(hidden, proj),
    }


def test_kda_attention_chunked_matches_single_shot():
    """Splitting a sequence across two stateful calls must equal one call.

    This does not prove the formula matches the released model (no oracle
    available), but it does prove the state/conv-history carry-over is a
    real incremental recurrence, not something that only happens to work
    for L==1 or silently drops state between calls -- exactly the class of
    bug this project's own F32 rollback-boundary tests exist to catch.
    """
    hidden, num_heads, head_dim, conv_kernel = 64, 4, 16, 4
    cfg = _tiny_kda_config(hidden, num_heads, head_dim, conv_kernel)
    prefix = "layer0"
    w = _random_kda_weights(prefix, hidden, num_heads, head_dim, conv_kernel)

    mx.random.seed(0)
    x = mx.random.normal((1, 5, hidden)).astype(mx.float32)

    single_shot = _kda_attention(x, w, prefix, cfg, None, 0)
    mx.eval(single_shot)

    cache = KDAStateCache(num_layers=1)
    out1 = _kda_attention(x[:, :3, :], w, prefix, cfg, cache, 0)
    out2 = _kda_attention(x[:, 3:, :], w, prefix, cfg, cache, 0)
    mx.eval(out1, out2)
    chunked = mx.concatenate([out1, out2], axis=1)

    assert not bool(mx.any(mx.isnan(single_shot)).item())
    max_diff = float(mx.max(mx.abs(chunked - single_shot)).item())
    assert max_diff < 1e-5, f"chunked decode diverged from single-shot prefill: {max_diff}"


@_model_skip
def test_real_weights_kda_and_mla_layers_produce_finite_output():
    """F92 step 5 smoke gate: run one real KDA layer and one real MLA+MoE
    layer from the actual downloaded checkpoint. Proves shapes/plumbing
    only -- see module docstring."""
    cfg = ModelConfig.from_dir(MODEL_DIR)
    assert cfg.kda_layers and cfg.full_attn_layers, "config parsing produced no layer-type lists"

    kda_layer = cfg.kda_layers[0]
    mla_layer = cfg.full_attn_layers[0]
    import json
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    def shard_for(layer: int, suffix: str) -> str:
        return weight_map[f"model.layers.{layer}.{suffix}"]

    shard_names = {
        shard_for(kda_layer, "input_layernorm.weight"),
        shard_for(kda_layer, "self_attn.q_proj.weight"),
        shard_for(mla_layer, "input_layernorm.weight"),
        shard_for(mla_layer, "self_attn.q_proj.weight"),
        shard_for(mla_layer, "block_sparse_moe.gate.weight"),
    }
    w: dict = {}
    for name in shard_names:
        w.update(mx.load(str(MODEL_DIR / name)))

    kv = KVCache(num_layers=cfg.num_hidden_layers)
    kv.kda_cache = KDAStateCache(num_layers=cfg.num_hidden_layers)

    mx.random.seed(0)
    x = (mx.random.normal((1, 6, cfg.hidden_size)).astype(mx.bfloat16) * 0.02)

    is_dense = kda_layer < cfg.first_k_dense_replace
    out_kda = run_kimi_linear_block(
        x, w, f"model.layers.{kda_layer}", cfg, kv,
        layer=kda_layer, offset=0, get_experts=None if is_dense else _expert_loader(w, MODEL_DIR),
    )
    mx.eval(out_kda)
    assert out_kda.shape == x.shape
    assert not bool(mx.any(mx.isnan(out_kda)).item())

    def get_experts(layer, expert_ids, positions=None):
        loader = _expert_loader(w, MODEL_DIR)
        return loader(layer, expert_ids, positions=positions)

    out_mla = run_kimi_linear_block(
        out_kda, w, f"model.layers.{mla_layer}", cfg, kv,
        layer=mla_layer, offset=0, get_experts=get_experts,
    )
    mx.eval(out_mla)
    assert out_mla.shape == x.shape
    assert not bool(mx.any(mx.isnan(out_mla)).item())


def _expert_loader(w: dict, model_dir: Path):
    import json
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    def load(layer, expert_ids, positions=None):
        out = {}
        for e in expert_ids:
            prefix = f"model.layers.{layer}.block_sparse_moe.experts.{e}"
            for suffix in ("w1.weight", "w2.weight", "w3.weight"):
                key = f"{prefix}.{suffix}"
                if key not in w:
                    shard = weight_map[key]
                    w.update(mx.load(str(model_dir / shard)))
            out[e] = {
                f"{prefix}.w1.weight": w[f"{prefix}.w1.weight"],
                f"{prefix}.w2.weight": w[f"{prefix}.w2.weight"],
                f"{prefix}.w3.weight": w[f"{prefix}.w3.weight"],
            }
        return out
    return load
