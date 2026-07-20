"""Tests for the shard-bounded standard-MLX quantization converter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formats.quantize_mlx import convert_model
from runtime.model_loader import WeightStore
from runtime.quant import QTensor


def _write_source(path: Path) -> None:
    path.mkdir()
    shard1 = {
        "model.layers.0.self_attn.q_proj.weight": mx.eye(64),
        "model.layers.0.mlp.gate.weight": mx.ones((2, 64)),
        "model.layers.0.mlp.experts.0.up_proj.weight": mx.ones((64, 64)),
    }
    shard2 = {
        "model.norm.weight": mx.ones((64,)),
        "lm_head.weight": mx.ones((128, 64)),
    }
    mx.save_safetensors(str(path / "model-00001-of-00002.safetensors"), shard1)
    mx.save_safetensors(str(path / "model-00002-of-00002.safetensors"), shard2)
    weight_map = {
        name: shard
        for shard, tensors in (
            ("model-00001-of-00002.safetensors", shard1),
            ("model-00002-of-00002.safetensors", shard2),
        )
        for name in tensors
    }
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {}, "weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({
        "model_type": "olmoe",
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "vocab_size": 128,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }))
    (path / "tokenizer.json").write_text("fixture-tokenizer")
    (path / "chat_template.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}))


def test_expert_profile_streams_shards_and_preserves_sensitive_weights(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _write_source(source)
    converted = convert_model(source, output)

    assert converted == output
    assert not (output / ".quantize-incomplete.json").exists()
    assert (output / "tokenizer.json").read_text() == "fixture-tokenizer"
    assert json.loads((output / "chat_template.json").read_text()) == {
        "chat_template": "{{ messages }}"}

    config = json.loads((output / "config.json").read_text())
    expert_stem = "model.layers.0.mlp.experts.0.up_proj"
    assert config["quantization"] == {
        "bits": 4, "group_size": 32, "mode": "mxfp4"}
    assert config["voom_quantization"]["profile"] == "experts"
    assert config["voom_quantization"]["quantized_tensors"] == 1
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert f"{expert_stem}.scales" in index["weight_map"]
    assert index["weight_map"][f"{expert_stem}.scales"] == \
        "model-00001-of-00002.safetensors"

    store = WeightStore(output)
    values, _seconds, _nbytes = store.fetch([
        f"{expert_stem}.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ])
    assert isinstance(values[f"{expert_stem}.weight"], QTensor)
    assert not isinstance(values["model.layers.0.mlp.gate.weight"], QTensor)
    assert not isinstance(values["model.layers.0.self_attn.q_proj.weight"], QTensor)
    assert not isinstance(values["lm_head.weight"], QTensor)


def test_resume_rejects_changed_conversion_parameters(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _write_source(source)
    output.mkdir()
    (output / ".quantize-incomplete.json").write_text(json.dumps({
        "version": 2,
        "source": str(source.resolve()),
        "profile": "experts",
        "mode": "mxfp4",
        "group_size": 32,
        "bits": 4,
        "completed_shards": [],
        "weight_map": {},
        "quantized_tensors": 0,
        "total_size": 0,
    }))

    try:
        convert_model(source, output, mode="affine", group_size=64)
    except ValueError as error:
        assert "resume state mismatch" in str(error)
    else:
        raise AssertionError("changed parameters were accepted for a resumed conversion")


def test_qwen_fused_experts_are_split_and_prequantized(tmp_path):
    source, output = tmp_path / "qwen-source", tmp_path / "qwen-output"
    source.mkdir()
    gate_up = mx.arange(2 * 64 * 64, dtype=mx.float32).reshape(
        2, 64, 64).astype(mx.bfloat16) / 100
    down = mx.arange(2 * 64 * 32, dtype=mx.float32).reshape(
        2, 64, 32).astype(mx.bfloat16) / 100
    tensors = {
        "model.language_model.embed_tokens.weight": mx.ones((128, 64)),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
        "model.language_model.norm.weight": mx.ones((64,)),
        "lm_head.weight": mx.ones((128, 64)),
    }
    mx.save_safetensors(str(source / "model.safetensors"), tensors)
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "vision_config": {"depth": 1},
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 128,
            "eos_token_id": 127,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "layer_types": ["full_attention"],
            "partial_rotary_factor": 1.0,
            "rope_parameters": {"rope_theta": 10_000.0},
        },
    }))
    (source / "configuration.json").write_text("custom configuration\n")
    (source / "video_preprocessor_config.json").write_text(
        json.dumps({"video_fps": 2}))

    convert_model(source, output)
    assert (output / "configuration.json").read_text() == "custom configuration\n"
    assert json.loads((output / "video_preprocessor_config.json").read_text()) == {
        "video_fps": 2}
    index = json.loads(
        (output / "model.safetensors.index.json").read_text())["weight_map"]
    physical = "model.language_model.layers.0.mlp.experts"
    assert f"{physical}.gate_up_proj" not in index
    assert f"{physical}.down_proj" not in index
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            stem = f"{physical}.{expert}.{projection}"
            assert f"{stem}.weight" in index
            assert f"{stem}.scales" in index

    store = WeightStore(output)
    values, _seconds, _bytes = store.fetch([
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ])
    assert all(isinstance(value, QTensor) for value in values.values())
