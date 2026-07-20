"""Lossless Qwen3.5/Qwen3.6 fused-expert packing gates."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np

from formats.packed import pack_model, read_tensor_bytes, to_mx
from formats.packed2 import build_from_vpack, verify_generation
from runtime.model_loader import WeightStore


def _write_config(path):
    config = {
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "vision_config": {"depth": 1},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 4,
            "moe_intermediate_size": 3,
            "shared_expert_intermediate_size": 3,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "vocab_size": 8,
            "eos_token_id": 7,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "layer_types": ["full_attention"],
            "partial_rotary_factor": 1.0,
            "rope_parameters": {"rope_theta": 10000.0},
        },
    }
    (path / "config.json").write_text(json.dumps(config))


def _fixture(path):
    _write_config(path)
    gate_up = mx.arange(2 * 2 * 3 * 4).reshape(2, 6, 4).astype(mx.bfloat16)
    down = (100 + mx.arange(2 * 4 * 3)).reshape(2, 4, 3).astype(mx.bfloat16)
    tensors = {
        "model.language_model.embed_tokens.weight": (
            mx.arange(8 * 4).reshape(8, 4).astype(mx.bfloat16)),
        "model.language_model.layers.0.input_layernorm.weight": (
            mx.zeros((4,), dtype=mx.bfloat16)),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
        "model.language_model.norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
        "lm_head.weight": mx.arange(8 * 4).reshape(8, 4).astype(mx.bfloat16),
    }
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    return gate_up, down


def _read(vpack, manifest, name):
    head, raw = read_tensor_bytes(vpack, manifest[name])
    value = to_mx(head, raw)
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def test_fused_bf16_experts_split_bit_exact_and_canonicalize(tmp_path):
    gate_up, down = _fixture(tmp_path)
    vpack = pack_model(tmp_path, verify_shards=True)
    manifest = json.loads((vpack / "manifest.json").read_text())
    base = "model.language_model.layers.0.mlp.experts"

    assert f"{base}.gate_up_proj" not in manifest
    assert f"{base}.down_proj" not in manifest
    for expert in range(2):
        np.testing.assert_array_equal(
            _read(vpack, manifest, f"{base}.{expert}.gate_proj.weight"),
            np.array(gate_up[expert, :3].astype(mx.float32)),
        )
        np.testing.assert_array_equal(
            _read(vpack, manifest, f"{base}.{expert}.up_proj.weight"),
            np.array(gate_up[expert, 3:].astype(mx.float32)),
        )
        np.testing.assert_array_equal(
            _read(vpack, manifest, f"{base}.{expert}.down_proj.weight"),
            np.array(down[expert].astype(mx.float32)),
        )

    build_from_vpack(tmp_path)
    report = verify_generation(tmp_path, decode=True)
    assert report["errors"] == []
    assert report["hashed"] == report["tensors"]

    index = json.loads((tmp_path / "weights.vpack2.index.json").read_text())
    names = list(index)
    expert_names = [name for name in names if ".mlp.experts." in name]
    assert expert_names == sorted(
        expert_names,
        key=lambda name: (int(name.split(".experts.", 1)[1].split(".", 1)[0]),
                          name),
    )

    store = WeightStore(tmp_path, require_vpack_hashes=True)
    assert store.packed
    assert store.has("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert store.has("model.layers.0.mlp.experts.1.down_proj.weight")
    fetched, _seconds, _bytes = store.fetch([
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ])
    np.testing.assert_array_equal(
        np.array(fetched[
            "model.layers.0.mlp.experts.0.gate_proj.weight"].astype(mx.float32)),
        np.array(gate_up[0, :3].astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(fetched[
            "model.layers.0.mlp.experts.1.down_proj.weight"].astype(mx.float32)),
        np.array(down[1].astype(mx.float32)),
    )
