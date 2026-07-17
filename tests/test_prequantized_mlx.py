"""Standard MLX on-disk quantization is consumed without BF16 expansion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime import layer_runner, quant
from runtime.engine import RuntimeConfig, _quantization_cache_identity
from runtime.model_loader import WeightStore


def _config() -> dict:
    return {
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }


def test_standard_mlx_triplets_become_logical_qtensors(tmp_path):
    original = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    wq, scales, biases = mx.quantize(original, group_size=64, bits=4)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": wq,
        "model.embed_tokens.scales": scales,
        "model.embed_tokens.biases": biases,
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })
    (tmp_path / "config.json").write_text(json.dumps(_config()))

    store = WeightStore(tmp_path)
    assert store.on_disk_quantized
    assert "model.embed_tokens.scales" not in store._names
    assert "model.embed_tokens.biases" not in store._names

    tensors, _seconds, nbytes = store.fetch([
        "model.embed_tokens.weight", "model.norm.weight"])
    embedded = tensors["model.embed_tokens.weight"]
    assert isinstance(embedded, quant.QTensor)
    assert embedded.mode == "affine"
    assert nbytes == embedded.nbytes + tensors["model.norm.weight"].nbytes

    token_ids = mx.array([1, 6])
    selected = layer_runner.embed(token_ids, embedded)[0]
    expected = mx.dequantize(
        wq, scales=scales, biases=biases, group_size=64, bits=4)[token_ids]
    mx.eval(selected, expected)
    assert mx.allclose(selected, expected)


def test_quant_policy_does_not_requantize_a_disk_qtensor():
    source = mx.ones((8, 64), dtype=mx.float32)
    wq, scales, biases = mx.quantize(source, group_size=64, bits=4)
    disk_tensor = quant.QTensor(wq, scales, biases, 4, 64)
    policy = quant.QuantPolicy(bits=4, group_size=64)
    assert policy.transform("model.layers.0.mlp.up_proj.weight", disk_tensor) is disk_tensor


@pytest.mark.parametrize(
    ("mode", "group_size", "bits"),
    [("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)],
)
def test_bias_free_standard_mlx_modes_round_trip(mode, group_size, bits, tmp_path):
    original = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    packed = mx.quantize(original, group_size=group_size, bits=bits, mode=mode)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": packed[0],
        "model.embed_tokens.scales": packed[1],
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })
    config = _config()
    config["quantization"] = {
        "group_size": group_size, "bits": bits, "mode": mode}
    (tmp_path / "config.json").write_text(json.dumps(config))

    store = WeightStore(tmp_path)
    tensors, _seconds, _nbytes = store.fetch(["model.embed_tokens.weight"])
    embedded = tensors["model.embed_tokens.weight"]
    selected = layer_runner.embed(mx.array([1, 6]), embedded)[0]
    expected = mx.dequantize(
        packed[0], scales=packed[1], group_size=group_size,
        bits=bits, mode=mode)[mx.array([1, 6])]
    mx.eval(selected, expected)

    assert isinstance(embedded, quant.QTensor)
    assert embedded.biases is None
    assert embedded.mode == mode
    assert embedded.shape == original.shape
    assert embedded.dtype == mx.bfloat16
    assert mx.array_equal(selected, expected)


def test_fine_grained_mlx_quantization_uses_per_module_parameters(tmp_path):
    original = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    wq, scales, biases = mx.quantize(original, group_size=64, bits=4)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": wq,
        "model.embed_tokens.scales": scales,
        "model.embed_tokens.biases": biases,
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })
    config = _config()
    config["quantization"] = {
        "model.embed_tokens": {
            "group_size": 64, "bits": 4, "mode": "affine"}}
    (tmp_path / "config.json").write_text(json.dumps(config))

    store = WeightStore(tmp_path)
    tensors, _seconds, _nbytes = store.fetch(["model.embed_tokens.weight"])
    embedded = tensors["model.embed_tokens.weight"]
    assert store.on_disk_quantized
    assert isinstance(embedded, quant.QTensor)
    assert (embedded.group_size, embedded.bits, embedded.mode) == (64, 4, "affine")


def test_unsupported_hf_fp8_layout_fails_closed(tmp_path):
    config = _config()
    config.pop("quantization")
    config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": mx.zeros((8, 64), dtype=mx.uint8),
        "model.embed_tokens.weight_scale_inv": mx.ones((8, 1)),
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })

    with pytest.raises(NotImplementedError, match="unsupported on-disk quantization"):
        WeightStore(tmp_path)


def test_declared_standard_mlx_quantization_without_triplets_fails_closed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.uint32),
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })

    with pytest.raises(NotImplementedError, match="unsupported on-disk quantization"):
        WeightStore(tmp_path)


@pytest.mark.parametrize(
    ("mode", "group_size", "bits", "has_bias"),
    [
        ("affine", 64, 3, True),
        ("mxfp4", 32, 4, False),
        ("mxfp8", 32, 8, False),
        ("nvfp4", 16, 4, False),
    ],
)
def test_quantize_on_load_policy_supports_mlx_modes(
        mode, group_size, bits, has_bias):
    source = mx.arange(8 * 64, dtype=mx.float32).reshape(8, 64) / 100
    policy = quant.QuantPolicy(
        bits=bits, group_size=group_size, mode=mode, min_dim=0)
    packed = policy.transform("lm_head.weight", source)
    got = quant.matmul(mx.ones((1, 64)), packed)
    mx.eval(got)

    assert isinstance(packed, quant.QTensor)
    assert packed.mode == mode
    assert packed.shape == source.shape
    assert (packed.biases is not None) is has_bias
    assert got.shape == (1, 8)


@pytest.mark.parametrize(
    ("bits", "group_size", "mode"),
    [(4, 64, "mxfp4"), (7, 64, "affine")],
)
def test_quant_policy_rejects_invalid_mode_parameters(bits, group_size, mode):
    with pytest.raises(ValueError, match="unsupported MLX quantization parameters"):
        quant.QuantPolicy(bits=bits, group_size=group_size, mode=mode)


def test_standard_mlx_metadata_rejects_unsupported_affine_7bit(tmp_path):
    config = _config()
    config["quantization"] = {
        "group_size": 64, "bits": 7, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": mx.zeros((8, 8), dtype=mx.uint32),
        "model.embed_tokens.scales": mx.ones((8, 1)),
        "model.norm.weight": mx.ones((64,), dtype=mx.float32),
    })

    with pytest.raises(ValueError, match="no usable bits/group_size descriptor"):
        WeightStore(tmp_path)


def test_expert_only_policy_preserves_router_attention_and_lm_head():
    policy = quant.QuantPolicy(
        bits=4,
        group_size=32,
        mode="mxfp4",
        quantize_attention=False,
        quantize_router=False,
        quantize_lm_head=False,
        min_dim=0,
    )
    source = mx.ones((64, 64), dtype=mx.bfloat16)

    assert policy.transform("model.layers.0.self_attn.q_proj.weight", source) is source
    assert policy.transform("model.layers.0.mlp.gate.weight", source) is source
    assert policy.transform("lm_head.weight", source) is source
    assert isinstance(policy.transform(
        "model.layers.0.mlp.experts.0.up_proj.weight", source), quant.QTensor)


def test_selective_store_reports_tensor_level_quantization(tmp_path):
    original = mx.ones((8, 64), dtype=mx.bfloat16)
    packed = mx.quantize(original, group_size=32, bits=4, mode="mxfp4")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "model.embed_tokens.weight": original,
        "model.layers.0.mlp.experts.0.up_proj.weight": packed[0],
        "model.layers.0.mlp.experts.0.up_proj.scales": packed[1],
        "model.layers.0.mlp.gate.weight": original,
        "model.norm.weight": mx.ones((64,), dtype=mx.bfloat16),
    })
    config = _config()
    config["quantization"] = {
        "group_size": 32, "bits": 4, "mode": "mxfp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))

    store = WeightStore(tmp_path)
    assert store.is_quantized("model.layers.0.mlp.experts.0.up_proj.weight")
    assert not store.is_quantized("model.embed_tokens.weight")
    assert store.quantization_ratio(
        "model.layers.0.mlp.experts.0.up_proj.weight") < 0.3
    # The raw router makes a single ratio for the whole MLP family unsafe.
    assert store.uniform_quantization_ratio(".mlp.") == 1.0


def test_disk_kv_identity_includes_load_time_policy():
    store = type("Store", (), {
        "on_disk_quantized": True,
        "quantization_identity": "mlx-example",
    })()
    base = RuntimeConfig(quant_bits=4, quant_attention=False)
    changed = RuntimeConfig(quant_bits=4, quant_attention=True)

    assert _quantization_cache_identity(base, store) != (
        _quantization_cache_identity(changed, store))

    reranked = RuntimeConfig(
        quant_bits=4, quant_attention=False, rerank_lm_head=True)
    assert _quantization_cache_identity(base, store) != (
        _quantization_cache_identity(reranked, store))

    hybrid_attention = RuntimeConfig(
        quant_bits=4, quant_attention=False,
        resident_attention_mode="mxfp8")
    assert _quantization_cache_identity(base, store) != (
        _quantization_cache_identity(hybrid_attention, store))
