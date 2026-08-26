"""Bounded gates for explicit lossy Qwen paged online attention."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os

import mlx.core as mx
import numpy as np
import pytest

from runtime.kv_paged import PagedKVCache
from runtime.qwen35_paged_attention import (
    theoretical_tile_count,
    tiled_paged_attention,
)


def test_paged_chunk_iterator_reconstructs_spilled_bits_exactly(tmp_path):
    kv = PagedKVCache(
        1, max_bytes=1, spill_dir=tmp_path,
        page_positions=4, resident_pages=0)
    keys = mx.arange(14 * 8, dtype=mx.float32).reshape(1, 2, 14, 4)
    values = keys + 1000
    kv.append_for_online_attention(0, keys, values)

    chunks = list(kv.iter_materialized_layer_chunks(0, max_positions=5))
    reconstructed_keys = mx.concatenate([item[0] for item in chunks], axis=2)
    reconstructed_values = mx.concatenate([item[1] for item in chunks], axis=2)
    mx.eval(reconstructed_keys, reconstructed_values)

    assert [int(item[0].shape[2]) for item in chunks] == [5, 5, 4]
    np.testing.assert_array_equal(np.asarray(reconstructed_keys), np.asarray(keys))
    np.testing.assert_array_equal(
        np.asarray(reconstructed_values), np.asarray(values))
    assert kv.offset == 14
    assert theoretical_tile_count(14, 5) == 3


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_tiled_online_attention_is_close_and_bounded_across_spills(tmp_path):
    mx.random.seed(238)
    kv = PagedKVCache(
        1, max_bytes=1, spill_dir=tmp_path,
        page_positions=32, resident_pages=0)
    q = mx.random.normal((1, 4, 1, 128)).astype(mx.bfloat16)
    keys = mx.random.normal((1, 2, 73, 128)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 73, 128)).astype(mx.bfloat16)
    kv.append_for_online_attention(0, keys, values)

    candidate = tiled_paged_attention(q, kv, 0, tile_positions=32)
    reference = mx.fast.scaled_dot_product_attention(
        q, keys, values, scale=128 ** -0.5)
    mx.eval(candidate, reference)

    actual = np.asarray(candidate.astype(mx.float32))
    expected = np.asarray(reference.astype(mx.float32))
    assert candidate.shape == reference.shape
    assert float(np.max(np.abs(actual - expected))) <= 0.002
    denominator = np.linalg.norm(actual) * np.linalg.norm(expected)
    assert float(np.vdot(actual, expected) / denominator) >= 0.9999
    assert kv.stats.reloads >= 2


def test_server_paged_online_attention_is_strict_and_explicit(tmp_path):
    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(os.environ, {
        "VMODEL_QWEN35_PAGED_ONLINE_ATTENTION": "yes",
    }):
        with pytest.raises(RequestValidationError, match="must be 0 or 1"):
            EngineManager().get(Path("/tmp/not-opened-paged-online"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_PAGED_ONLINE_TILE_POSITIONS": "4096",
    }):
        with pytest.raises(RequestValidationError, match="must be one of"):
            EngineManager().get(Path("/tmp/not-opened-paged-tile"), "fast")
    with patch.dict(os.environ, {
        "VMODEL_QWEN35_KV_PAGE_POSITIONS": "4096",
    }):
        with pytest.raises(RequestValidationError, match="must be one of"):
            EngineManager().get(Path("/tmp/not-opened-kv-page"), "fast")

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            return None

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN35_KV_MAX_MB": "64",
        "VMODEL_QWEN35_KV_SPILL_DIR": str(tmp_path / "spill"),
        "VMODEL_QWEN35_KV_PAGE_POSITIONS": "1024",
        "VMODEL_QWEN35_PAGED_ONLINE_ATTENTION": "1",
        "VMODEL_QWEN35_PAGED_ONLINE_TILE_POSITIONS": "1024",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen-paged-online"), "fast")

    assert captured[0].max_kv_mb == 64
    assert captured[0].kv_page_positions == 1024
    assert captured[0].qwen35_paged_online_attention
    assert captured[0].qwen35_paged_online_tile_positions == 1024
