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
