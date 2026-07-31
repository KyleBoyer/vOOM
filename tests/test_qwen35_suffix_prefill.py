"""Focused invariants for Qwen hybrid mixed-depth suffix prefill."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.qwen35 import _cache_local_causal_mask


def test_compact_suffix_causal_mask_is_lower_right():
    mask = np.asarray(_cache_local_causal_mask(3, 5, mx.float32))
    np.testing.assert_array_equal(
        np.isfinite(mask),
        np.array([
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]),
    )


def test_ordinary_causal_mask_is_unchanged():
    mask = np.asarray(_cache_local_causal_mask(4, 4, mx.float32))
    np.testing.assert_array_equal(
        np.isfinite(mask),
        np.tril(np.ones((4, 4), dtype=bool)),
    )


def test_compact_suffix_causal_mask_rejects_impossible_dimensions():
    with pytest.raises(ValueError, match="cache dimensions"):
        _cache_local_causal_mask(5, 4, mx.float32)


def test_endpoint_packing_keeps_fixed_prefix_and_suffix_global_positions():
    engine = object.__new__(StreamingEngine)
    engine.rc = RuntimeConfig(
        qwen_lossy_suffix_prefill_early_layers=4,
        qwen_lossy_suffix_prefill_prefix_tokens=2,
        qwen_lossy_suffix_prefill_tokens=3,
    )
    engine.cfg = type("Config", (), {"num_hidden_layers": 8})()
    calls = []

    def sweep(x, _kv, offset, tile_width, on_progress=None, **kwargs):
        calls.append({
            "values": np.asarray(x),
            "offset": offset,
            "tile_width": tile_width,
            "positions3": (
                None if kwargs.get("positions3") is None
                else np.asarray(kwargs["positions3"])),
            **{key: value for key, value in kwargs.items()
               if key != "positions3"},
        })
        return x

    engine._layer_stationary_qwen35_sweep = sweep
    values = mx.arange(10, dtype=mx.float32).reshape(1, 10, 1)
    result = engine._qwen35_lossy_suffix_prefill_sweep(
        values, object(), offset=50, tile_width=4)

    np.testing.assert_array_equal(
        np.asarray(result).reshape(-1), [0, 1, 7, 8, 9])
    assert calls[0]["layer_end"] == 4
    assert calls[1]["layer_start"] == 4
    assert calls[1]["offset"] == 50
    np.testing.assert_array_equal(
        calls[1]["positions3"],
        np.array([
            [50, 51, 57, 58, 59],
            [50, 51, 57, 58, 59],
            [50, 51, 57, 58, 59],
        ], dtype=np.float32),
    )
