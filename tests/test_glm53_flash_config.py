"""Fail-closed parsing for the released GLM-5.3-Flash wrapper config."""

from __future__ import annotations

import copy
import json
import struct

import numpy as np
import pytest

from runtime.config import ModelConfig


def _config() -> dict:
    text = {
        "model_type": "glm5_next_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 0,
        "vocab_size": 64,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1_048_576,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "dtype": "bfloat16",
        "eos_token_id": [31, 32],
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 16,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "mhc": True,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention",
            "deepseek_sparse_attention",
        ],
        "linear_attn_config": {
            # GLM-5.3 uses zero-indexed ids, unlike Kimi's one-indexed list.
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 0,
        "v_head_dim": 8,
        "mla_use_nope": True,
        "index_topk": 16,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "indexer_types": ["full"] * 4,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_rope_interleave": True,
        "num_nextn_predict_layers": 1,
    }
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": text,
        "vision_config": {"model_type": "glm5_next", "hidden_size": 16},
        "image_token_id": 60,
        "video_token_id": 61,
        "tie_word_embeddings": False,
    }


def _load(tmp_path, config: dict) -> ModelConfig:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return ModelConfig.from_dir(tmp_path)


def test_glm53_lifts_released_text_geometry_without_reindexing_layers(tmp_path):
    loaded = _load(tmp_path, _config())

    assert loaded.model_type == "glm5_next"
    assert loaded.architectures == ("Glm5NextForConditionalGeneration",)
    assert loaded.vision_backend == "glm5_next"
    assert loaded.image_token_id == 60
    assert loaded.video_token_id == 61
    assert loaded.kda_layers == (0, 1, 2)
    assert loaded.full_attn_layers == (3,)
    assert loaded.kda_gate_lower_bound == -5.0
    assert loaded.hc_mult == 4
    assert loaded.index_topk == 16
    assert loaded.index_kpool == 4
    assert loaded.index_kpool_compress
    assert loaded.index_kpool_always_select_tail
    assert loaded.indexer_rope_interleave
    assert loaded.qk_rope_head_dim == 0
    assert loaded.mla_use_nope
    assert loaded.swiglu_limit == 10.0


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("architectures",), ["Glm5NextForCausalLM"], "architecture"),
        (("text_config", "layer_types"),
         ["linear_attention"] * 3 + ["full_attention"], "layer type"),
        (("text_config", "linear_attn_config", "kda_layers"),
         [1, 2, 3], "linear_attn_config"),
        (("text_config", "indexer_types"), ["full"] * 3, "indexer_types"),
        (("text_config", "index_kpool"), 3, "pooled indexer"),
        (("text_config", "index_kpool_compress"), False, "pooled indexer"),
        (("text_config", "qk_rope_head_dim"), 2, "NoPE MLA"),
        (("text_config", "hc_mult"), 2, "hyper-connection"),
    ],
)
def test_glm53_incompatible_geometry_fails_closed(
        tmp_path, path, value, match):
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        _load(tmp_path, config)


def _write_safetensor(tmp_path, tensors):
    header = {}
    payload = bytearray()
    for name, (value, dtype) in tensors.items():
        raw = np.ascontiguousarray(value).tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload)


def test_glm53_float_scale_fp8_dequant_matches_explicit_block_reference():
    import mlx.core as mx

    from runtime.quant import dequantize_finegrained_fp8

    codes = np.array([
        [0x00, 0x38, 0x40, 0xB8],
        [0x30, 0xB0, 0x48, 0xC8],
        [0x20, 0xA0, 0x50, 0xD0],
        [0x10, 0x90, 0x58, 0xD8],
    ], dtype=np.uint8)
    scale = np.array([[0.25, 0.5], [1.0, 2.0]], dtype=np.float32)
    packed = mx.array(codes)
    multiplier = mx.array(scale)
    values = mx.from_fp8(packed, mx.float32)
    expected = (values.reshape(2, 2, 2, 2)
                * multiplier[:, None, :, None]).reshape(4, 4).astype(
                    mx.bfloat16)
    got = dequantize_finegrained_fp8(packed, multiplier)
    mx.eval(expected, got)

    assert bool(mx.all(expected == got))


def test_glm53_weight_store_joins_and_hides_scale_inv(tmp_path):
    import mlx.core as mx

    from runtime.model_loader import WeightStore
    from runtime.quant import dequantize_finegrained_fp8

    config = _config()
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [2, 2],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    codes = np.arange(16, dtype=np.uint8).reshape(4, 4)
    scale = np.array([[0.25, 0.5], [1.0, 2.0]], dtype=np.float32)
    physical = "model.language_model.layers.0.mlp.gate_proj"
    _write_safetensor(tmp_path, {
        f"{physical}.weight": (codes, "F8_E4M3"),
        f"{physical}.weight_scale_inv": (scale, "F32"),
    })

    store = WeightStore(tmp_path)
    logical = "model.layers.0.mlp.gate_proj.weight"
    scale_name = "model.layers.0.mlp.gate_proj.weight_scale_inv"
    assert store.has(logical)
    assert scale_name not in store.names_with_prefix(
        "model.layers.0.mlp.gate_proj")
    # Synthetic 2x2 blocks cost one FP8 byte + 4/4 scale bytes per value.
    assert store.expert_storage_bytes_per_weight == pytest.approx(2.0)

    fetched, _seconds, physical_bytes = store.fetch([logical])
    want = dequantize_finegrained_fp8(
        mx.array(codes), mx.array(scale))
    mx.eval(fetched[logical], want)
    assert fetched[logical].dtype == mx.bfloat16
    assert bool(mx.all(fetched[logical] == want))
    assert physical_bytes == codes.nbytes + scale.nbytes


def test_glm53_fp8_scale_dtype_mismatch_fails_closed():
    import mlx.core as mx

    from runtime.quant import dequantize_finegrained_fp8

    with pytest.raises(ValueError, match="weight_scale_inv must be float32"):
        dequantize_finegrained_fp8(
            mx.zeros((4, 4), mx.uint8), mx.ones((2, 2), mx.float16))
