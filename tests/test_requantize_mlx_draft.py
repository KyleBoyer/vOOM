from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

from formats.requantize_mlx_draft import requantize_affine_draft
from runtime.model_loader import WeightStore
from runtime.quant import QTensor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path, *, profile: str = "all-draft") -> dict[str, mx.array]:
    path.mkdir()
    dense = mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64) / 97
    wq, scales, biases = mx.quantize(
        dense.astype(mx.bfloat16), group_size=64, bits=4, mode="affine")
    tensors = {
        "model.language_model.embed_tokens.weight": wq,
        "model.language_model.embed_tokens.scales": scales,
        "model.language_model.embed_tokens.biases": biases,
        "model.language_model.norm.weight": mx.ones((64,), dtype=mx.bfloat16),
    }
    shard = path / "model.safetensors"
    mx.save_safetensors(str(shard), tensors)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "vocab_size": 16,
            "eos_token_id": 15,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "layer_types": ["full_attention"],
            "partial_rotary_factor": 1.0,
            "rope_parameters": {"rope_theta": 10_000.0},
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "quantization_config": {
            "bits": 4, "group_size": 64, "mode": "affine"},
        "voom_quantization": {
            "profile": profile, "quantized_tensors": 1},
    }))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(x.nbytes for x in tensors.values())},
        "weight_map": {name: shard.name for name in tensors},
    }))
    (path / "tokenizer.json").write_text("proposal tokenizer")
    return tensors


@pytest.mark.parametrize("bits", [2, 3])
def test_requantized_draft_is_standard_mlx_and_smaller(tmp_path, bits):
    source = tmp_path / "source"
    original = _source(source)
    output = requantize_affine_draft(
        source, tmp_path / f"draft-q{bits}", bits=bits)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization"] == {
        "bits": bits, "group_size": 64, "mode": "affine"}
    assert config["voom_draft_requantization"]["proposal_only"] is True
    assert (output / "tokenizer.json").read_text() == "proposal tokenizer"

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] < sum(x.nbytes for x in original.values())
    store = WeightStore(output)
    values, _seconds, _bytes = store.fetch([
        "model.embed_tokens.weight",
        "model.norm.weight",
    ])
    q = values["model.embed_tokens.weight"]
    assert isinstance(q, QTensor)
    assert (q.bits, q.group_size, q.mode) == (bits, 64, "affine")
    assert values["model.norm.weight"].dtype == mx.bfloat16


def test_requantization_is_byte_deterministic(tmp_path):
    source = tmp_path / "source"
    _source(source)
    first = requantize_affine_draft(source, tmp_path / "first", bits=2)
    second = requantize_affine_draft(source, tmp_path / "second", bits=2)
    assert _sha(first / "model.safetensors") == _sha(
        second / "model.safetensors")


@pytest.mark.parametrize("profile", ["all", "experts", "serving"])
def test_requantization_rejects_non_draft_artifacts(tmp_path, profile):
    source = tmp_path / "source"
    _source(source, profile=profile)
    with pytest.raises(ValueError, match="proposal-only"):
        requantize_affine_draft(source, tmp_path / "output", bits=2)
    assert not (tmp_path / "output").exists()


def test_requantization_rejects_existing_output(tmp_path):
    source = tmp_path / "source"
    _source(source)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        requantize_affine_draft(source, output, bits=2)
