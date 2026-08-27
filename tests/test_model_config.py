"""Strict HF config/generation metadata loading without model weights."""

from __future__ import annotations

import json

import pytest

from runtime.config import ModelConfig


_MISSING = object()


def _text_config(*, eos_token_id=3, vocab_size=16) -> dict:
    return {
        "model_type": "qwen2",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "head_dim": 8,
        "eos_token_id": eos_token_id,
        "torch_dtype": "bfloat16",
    }


def _write_config(tmp_path, config: dict, *, generation=_MISSING,
                  video_processor=_MISSING):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if generation is not _MISSING:
        (tmp_path / "generation_config.json").write_text(
            json.dumps(generation))
    if video_processor is not _MISSING:
        (tmp_path / "video_preprocessor_config.json").write_text(
            json.dumps(video_processor))


def test_generation_eos_ids_merge_after_config_order_and_deduplicate(tmp_path):
    config = _text_config(eos_token_id=[3, 4, 3])
    _write_config(
        tmp_path, config,
        generation={"eos_token_id": [4, 5, 5]},
    )

    loaded = ModelConfig.from_dir(tmp_path)

    assert loaded.eos_token_ids == (3, 4, 5)


@pytest.mark.parametrize("generation", [_MISSING, {}, {"eos_token_id": None}])
def test_missing_generation_eos_preserves_config_ids(tmp_path, generation):
    config = _text_config(eos_token_id=7)
    _write_config(tmp_path, config, generation=generation)

    assert ModelConfig.from_dir(tmp_path).eos_token_ids == (7,)


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("config", True),
        ("config", "3"),
        ("config", 3.0),
        ("config", -1),
        ("config", 16),
        ("config", [3, False]),
        ("config", [3, "4"]),
        ("generation", True),
        ("generation", "3"),
        ("generation", 3.0),
        ("generation", -1),
        ("generation", 16),
        ("generation", [4, False]),
        ("generation", [4, "5"]),
    ],
)
def test_invalid_eos_ids_fail_closed(tmp_path, source, value):
    config = _text_config()
    generation = _MISSING
    if source == "config":
        config["eos_token_id"] = value
    else:
        generation = {"eos_token_id": value}
    _write_config(tmp_path, config, generation=generation)

    with pytest.raises(ValueError, match=rf"{source}.*eos_token_id"):
        ModelConfig.from_dir(tmp_path)


def test_qwen3vl_loads_video_total_pixel_bounds(tmp_path):
    config = {
        "model_type": "qwen3_vl",
        "text_config": _text_config(eos_token_id=12, vocab_size=32),
        "vision_config": {
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": 20,
        "video_token_id": 21,
        "vision_start_token_id": 22,
        "vision_end_token_id": 23,
    }
    _write_config(
        tmp_path, config,
        generation={"eos_token_id": [12, 13]},
        video_processor={
            "size": {"shortest_edge": 4_096,
                     "longest_edge": 25_165_824},
        },
    )

    loaded = ModelConfig.from_dir(tmp_path)

    assert loaded.eos_token_ids == (12, 13)
    assert loaded.video_min_pixels == 4_096
    assert loaded.video_max_pixels == 25_165_824
    assert loaded.vision_backend == "qwen3vl"


def test_kimi_k3_preserves_outer_architecture_and_vision_contract(tmp_path):
    config = {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": _text_config(eos_token_id=12, vocab_size=32),
        "vision_config": {
            "patch_size": 14,
            "merge_kernel_size": [2, 2],
            "mm_projector_type": "patchmergerv2",
        },
        "media_placeholder_token_id": 20,
    }
    _write_config(tmp_path, config)

    loaded = ModelConfig.from_dir(tmp_path)

    assert loaded.model_type == "kimi_k3"
    assert loaded.vision_backend == "kimi_k3"
    assert loaded.architectures == ("KimiK3ForConditionalGeneration",)
    assert loaded.media_placeholder_token_id == 20
    assert loaded.vision_config == config["vision_config"]


@pytest.mark.parametrize("architectures", ["KimiK3ForConditionalGeneration", [""]])
def test_invalid_architecture_contract_fails_closed(tmp_path, architectures):
    config = _text_config()
    config["architectures"] = architectures
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match="architectures"):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize(
    "size",
    [
        [],
        {"shortest_edge": True},
        {"shortest_edge": 0},
        {"longest_edge": "25165824"},
        {"shortest_edge": 8_192, "longest_edge": 4_096},
    ],
)
def test_invalid_qwen3vl_video_pixel_bounds_fail_closed(tmp_path, size):
    config = {
        "model_type": "qwen3_vl",
        "text_config": _text_config(vocab_size=32),
        "vision_config": {"patch_size": 16},
    }
    _write_config(tmp_path, config, video_processor={"size": size})

    with pytest.raises(ValueError, match="video|size"):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"patch_size": 8},
        {"temporal_patch_size": 4},
        {"merge_size": 4},
        {"patch_size": True},
    ],
)
def test_mismatched_qwen3vl_video_geometry_fails_closed(tmp_path, override):
    config = {
        "model_type": "qwen3_vl",
        "text_config": _text_config(vocab_size=32),
        "vision_config": {
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    processor = {"size": {}, **override}
    _write_config(tmp_path, config, video_processor=processor)

    with pytest.raises(ValueError, match="video .*size"):
        ModelConfig.from_dir(tmp_path)


def _qwen4_exp_config() -> dict:
    text = _text_config(eos_token_id=31, vocab_size=64)
    text.update({
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention", "linear_attention",
            "linear_attention", "full_attention",
        ],
        "full_attention_interval": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "hc_count": 4,
        "hc_lowrank": 8,
        "ple_layer_ids": [2],
        "ple_embed_dim": 32,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 100,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 4,
        "indexer_budget": 16,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 8,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 2,
        "output_gate_type": "sigmoid",
    })
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": text,
        "vision_config": {"model_type": "qwen4_exp", "hidden_size": 16},
    }


def test_qwen4_exp_lifts_exact_text_geometry_and_preserves_outer_family(tmp_path):
    config = _qwen4_exp_config()
    _write_config(tmp_path, config)

    loaded = ModelConfig.from_dir(tmp_path)

    assert loaded.model_type == "qwen4_exp"
    assert loaded.vision_backend == "qwen4_exp"
    assert loaded.architectures == ("Qwen4ExpForConditionalGeneration",)
    assert loaded.qwen4_hc_count == 4
    assert loaded.qwen4_hc_lowrank == 8
    assert loaded.qwen4_ple_layers == (1,)
    assert loaded.qwen4_ple_embed_dim == 32
    assert loaded.qwen4_ngram_size == 3
    assert loaded.qwen4_split_ngram_parts == 4
    assert loaded.qwen4_indexer_budget == 16
    assert loaded.qwen4_indexer_compress_ratio == 4
    assert loaded.qwen4_output_gate_type == "sigmoid"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("ple_layer_ids", [0], "ple_layer_ids"),
        ("ple_layer_ids", [5], "ple_layer_ids"),
        ("layer_types", ["linear_attention"] * 3 + ["dense_attention"],
         "layer type"),
        ("output_gate_type", "relu", "geometry"),
        ("split_ngram_parts", 0, "geometry"),
    ],
)
def test_qwen4_exp_incomplete_geometry_fails_closed(
        tmp_path, key, value, match):
    config = _qwen4_exp_config()
    config["text_config"][key] = value
    _write_config(tmp_path, config)

    with pytest.raises(ValueError, match=match):
        ModelConfig.from_dir(tmp_path)
