"""Regression gates for the AirLLM Kimi-K3 capabilities retained by vOOM.

AirLLM's useful K3 changes were checkpoint-facing rather than new model math:
the multimodal wrapper prefixes, nested compressed-tensors metadata, direct
per-expert reads, verbatim packed dtypes, and top-level AttnRes/vision modules.
vOOM uses its own MLX scheduler and reads the indexed source shard directly;
this fixture proves that complete contract without creating split copies.
"""

from __future__ import annotations

import json

import mlx.core as mx

from runtime import quant
from runtime.model_loader import WeightStore


def _text_config() -> dict:
    return {
        "model_type": "kimi_linear",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "scale_dtype": "torch.uint8",
                        "symmetric": True,
                        "type": "float",
                    },
                },
            },
        },
    }


def _write_wrapped_k3_checkpoint(path):
    config = {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": _text_config(),
        "vision_config": {
            "patch_size": 14,
            "merge_kernel_size": [2, 2],
            "mm_projector_type": "patchmergerv2",
        },
        "media_placeholder_token_id": 20,
    }
    (path / "config.json").write_text(json.dumps(config))

    shard = "model-00001-of-000001.safetensors"
    expert = (
        "language_model.model.layers.0.block_sparse_moe.experts.7.w1"
    )
    tensors = {
        "language_model.model.embed_tokens.weight": mx.ones(
            (32, 64), dtype=mx.bfloat16),
        "language_model.model.layers.0.input_layernorm.weight": mx.ones(
            (64,), dtype=mx.bfloat16),
        f"{expert}.weight_packed": mx.arange(
            8 * 32, dtype=mx.uint8).reshape(8, 32),
        f"{expert}.weight_scale": mx.full((8, 2), 127, dtype=mx.uint8),
        "language_model.model.output_attn_res_norm.weight": mx.ones(
            (64,), dtype=mx.bfloat16),
        "language_model.model.output_attn_res_proj.weight": mx.ones(
            (1, 64), dtype=mx.bfloat16),
        "language_model.model.norm.weight": mx.ones(
            (64,), dtype=mx.bfloat16),
        "language_model.lm_head.weight": mx.ones(
            (32, 64), dtype=mx.bfloat16),
        "mm_projector.proj.weight": mx.ones(
            (64, 64), dtype=mx.bfloat16),
        "vision_tower.patch_embed.proj.weight": mx.ones(
            (64, 64), dtype=mx.bfloat16),
    }
    mx.save_safetensors(str(path / shard), tensors)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: shard for name in tensors},
    }))
    return shard


def test_wrapped_k3_loader_retains_airllm_checkpoint_capabilities(tmp_path):
    shard = _write_wrapped_k3_checkpoint(tmp_path)

    store = WeightStore(tmp_path, native_ct_mxfp4=True)

    # The quantization descriptor is nested under text_config in the released
    # multimodal wrapper. It must govern the physical expert pair unchanged.
    assert store.quantization["quant_method"] == "compressed-tensors"
    assert store.quantization["format"] == "mxfp4-pack-quantized"
    assert store.expert_storage_bytes_per_weight == 4 / 8 + 1 / 32
    assert store.expert_resident_bytes_per_weight == 4 / 8 + 1 / 32

    # Wrapper paths are exposed through the canonical text runtime contract;
    # non-text modules remain independently addressable for a native adapter.
    required = (
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.output_attn_res_norm.weight",
        "model.output_attn_res_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "mm_projector.proj.weight",
        "model.visual.patch_embed.proj.weight",
    )
    assert all(store.has(name) for name in required)

    logical_expert = (
        "model.layers.0.block_sparse_moe.experts.7.w1.weight"
    )
    assert store.names_with_prefix(
        "model.layers.0.block_sparse_moe.experts.7."
    ) == [logical_expert]

    fetched, _seconds, read_bytes = store.fetch([logical_expert])
    packed = fetched[logical_expert]
    assert isinstance(packed, quant.QTensor)
    assert (packed.mode, packed.bits, packed.group_size) == ("mxfp4", 4, 32)
    assert packed.wq.dtype == mx.uint32
    assert packed.scales.dtype == mx.uint8
    assert read_bytes == packed.nbytes

    # Unlike AirLLM's splitter, vOOM never creates a second checkpoint tree:
    # every canonical/logical name still points at the released source shard.
    assert all(store.weight_map[name] == shard for name in (*required, logical_expert))
    assert not (tmp_path / "splitted_model").exists()
