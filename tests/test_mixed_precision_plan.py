from __future__ import annotations

import json

import mlx.core as mx
import pytest

from formats.mixed_precision import (
    MixedPrecisionSpec,
    create_matrix,
    create_plan,
    validate_plan,
)
from formats.quantize_mlx import convert_model
from runtime.model_loader import WeightStore
from runtime.quant import QTensor


def _write_mixed_source(path):
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": mx.ones((64, 32)),
        "model.norm.weight": mx.ones((32,)),
        "lm_head.weight": mx.ones((64, 32)),
        "mtp.fc.weight": mx.ones((32, 32)),
        "mtp.norm.weight": mx.ones((32,)),
    }
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.self_attn.q_proj.weight"] = mx.ones((32, 32))
        tensors[f"{prefix}.mlp.up_proj.weight"] = mx.ones((32, 32))
        tensors[f"{prefix}.input_layernorm.weight"] = mx.ones((32,))
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "layer_types": ["full_attention"] * 4,
    }))
    return tensors


def test_component_plan_keeps_attention_final_layers_and_mtp_independent(tmp_path):
    source = tmp_path / "source"
    _write_mixed_source(source)
    plan = create_plan(source, MixedPrecisionSpec(
        attention="mxfp8", last_bf16_layers=2, mtp="mxfp4"))

    assert plan["tensors"][
        "model.layers.0.self_attn.q_proj.weight"]["storage"] == "mxfp8"
    assert plan["tensors"][
        "model.layers.0.mlp.up_proj.weight"]["storage"] == "mxfp4"
    assert plan["tensors"][
        "model.layers.2.self_attn.q_proj.weight"]["storage"] == "bf16"
    assert plan["tensors"][
        "model.layers.3.mlp.up_proj.weight"]["storage"] == "bf16"
    assert plan["tensors"]["mtp.fc.weight"]["storage"] == "mxfp4"
    assert plan["tensors"]["model.embed_tokens.weight"]["storage"] == "source"
    assert plan["summary"]["estimated_bytes"] == sum(
        value["estimated_bytes"] for value in plan["tensors"].values())
    assert validate_plan(plan, source) == plan


def test_plan_validation_fails_closed_on_tamper_and_matrix_is_complete(tmp_path):
    source = tmp_path / "source"
    _write_mixed_source(source)
    plan = create_plan(source, MixedPrecisionSpec())
    tampered = json.loads(json.dumps(plan))
    tampered["tensors"]["lm_head.weight"]["storage"] = "bf16"
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_plan(tampered, source)

    matrix = create_matrix(source)
    assert len(matrix["plans"]) == 24
    assert {row["spec"]["attention"] for row in matrix["plans"]} == {
        "bf16", "mxfp8"}
    assert {row["spec"]["last_bf16_layers"] for row in matrix["plans"]} == {
        0, 1, 2, 4}
    assert {row["spec"]["mtp"] for row in matrix["plans"]} == {
        "bf16", "mxfp8", "mxfp4"}


def test_tiny_mixed_plan_build_emits_per_tensor_descriptors(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _write_mixed_source(source)
    plan = create_plan(source, MixedPrecisionSpec(
        attention="mxfp8", last_bf16_layers=1, mtp="bf16"))
    convert_model(source, output, precision_plan=plan)

    config = json.loads((output / "config.json").read_text())
    assert config["voom_quantization"]["precision_plan_digest"] == \
        plan["plan_digest"]
    assert config["quantization"][
        "model.layers.0.self_attn.q_proj"] == {
            "bits": 8, "group_size": 32, "mode": "mxfp8"}
    store = WeightStore(output)
    values, _seconds, _bytes = store.fetch([
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.3.mlp.up_proj.weight",
        "mtp.fc.weight",
    ])
    assert isinstance(values[
        "model.layers.0.self_attn.q_proj.weight"], QTensor)
    assert values[
        "model.layers.0.self_attn.q_proj.weight"].mode == "mxfp8"
    assert isinstance(values["model.layers.0.mlp.up_proj.weight"], QTensor)
    assert not isinstance(values["model.layers.3.mlp.up_proj.weight"], QTensor)
    assert not isinstance(values["mtp.fc.weight"], QTensor)
