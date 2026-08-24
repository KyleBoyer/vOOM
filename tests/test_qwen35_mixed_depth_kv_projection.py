from types import SimpleNamespace

import pytest

from runtime.engine import (
    StreamingEngine,
    _remaining_layer_transient_reserve,
)


def _engine(*, early_layers=16, prefix_tokens=0, suffix_tokens=1024):
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=64,
        layer_types=tuple(
            layer_type
            for _ in range(16)
            for layer_type in (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
        ),
        num_key_value_heads=4,
        head_dim=256,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vision_config=None,
        num_experts=0,
    )
    engine.rc = SimpleNamespace(
        qwen_lossy_suffix_prefill_early_layers=early_layers,
        qwen_lossy_suffix_prefill_prefix_tokens=prefix_tokens,
        qwen_lossy_suffix_prefill_tokens=suffix_tokens,
    )
    return engine


def _recurrent_bytes():
    state = 48 * 48 * 128 * 128 * 4
    conv = 48 * 3 * (2 * 16 * 128 + 48 * 128) * 4
    return state + conv


def test_huihui_mixed_depth_projection_matches_retained_geometry():
    engine = _engine()
    positions = 30_029
    bytes_per_full_layer_position = 2 * 4 * 256 * 4
    expected_attention = (
        4 * positions + 12 * 1024
    ) * bytes_per_full_layer_position

    projected = engine._project_dense_text_kv_bytes(positions)

    assert projected == expected_attention + _recurrent_bytes()
    assert projected == 1_241_546_752


def test_mixed_depth_projection_counts_scaffold_in_deep_layers():
    engine = _engine(prefix_tokens=256, suffix_tokens=1024)
    positions = 30_029
    stable_boundary = 29_900
    deep_positions = 256 + 1024 + (positions - stable_boundary)
    bytes_per_full_layer_position = 2 * 4 * 256 * 4
    expected_attention = (
        4 * positions + 12 * deep_positions
    ) * bytes_per_full_layer_position

    projected = engine._project_dense_text_kv_bytes(
        positions, stable_boundary_positions=stable_boundary)

    assert projected == expected_attention + _recurrent_bytes()


def test_short_prompt_and_disabled_schedule_keep_uniform_projection():
    positions = 512
    bytes_per_full_layer_position = 2 * 4 * 256 * 4
    expected = 16 * positions * bytes_per_full_layer_position + _recurrent_bytes()

    assert _engine()._project_dense_text_kv_bytes(positions) == expected
    assert _engine(early_layers=0)._project_dense_text_kv_bytes(30_029) == (
        16 * 30_029 * bytes_per_full_layer_position + _recurrent_bytes())


def test_remaining_transient_excludes_only_already_live_dense_outputs():
    assert _remaining_layer_transient_reserve(1_230_000_000, 273_612_800) == (
        956_387_200)
    assert _remaining_layer_transient_reserve(10, 20) == 0
    with pytest.raises(ValueError):
        _remaining_layer_transient_reserve(-1, 0)
    with pytest.raises(ValueError):
        _remaining_layer_transient_reserve(1, -1)
