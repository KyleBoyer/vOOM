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
from runtime.qwen35_tree_verify import QwenTreeKVProxy
from runtime.qwen35_paged_attention import (
    theoretical_tile_count,
    tiled_paged_attention,
)
from runtime.speculative_tree import SpeculativeTree


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


def test_tree_proxy_streams_prompt_then_only_current_ancestor_path(tmp_path):
    kv = PagedKVCache(
        1, max_bytes=1, spill_dir=tmp_path,
        page_positions=4, resident_pages=0)
    kv.online_attention = True
    kv.online_attention_tile_positions = 5
    base_keys = mx.arange(12 * 8, dtype=mx.float32).reshape(1, 2, 12, 4)
    base_values = base_keys + 1000
    kv.append_for_online_attention(0, base_keys, base_values)
    tree = SpeculativeTree(
        token_ids=(1, 2, 3, 4),
        depths=(0, 1, 1, 2),
        parents=(-1, 0, 0, 1),
        children=({2: 1, 3: 2}, {4: 3}, {}, {}),
    )
    proxy = QwenTreeKVProxy(kv, tree, 1)
    node_keys = []
    node_values = []
    for node in range(4):
        proxy.current_node = node
        key = mx.full((1, 2, 1, 4), 100 + node, dtype=mx.float32)
        value = mx.full((1, 2, 1, 4), 200 + node, dtype=mx.float32)
        proxy.append_for_online_attention(0, key, value)
        node_keys.append(key)
        node_values.append(value)

    proxy.current_node = 3
    chunks = list(proxy.iter_materialized_layer_chunks(
        0, max_positions=5))
    actual_keys = mx.concatenate([item[0] for item in chunks], axis=2)
    actual_values = mx.concatenate([item[1] for item in chunks], axis=2)
    expected_keys = mx.concatenate(
        [base_keys, node_keys[0], node_keys[1], node_keys[3]], axis=2)
    expected_values = mx.concatenate(
        [base_values, node_values[0], node_values[1], node_values[3]], axis=2)
    mx.eval(actual_keys, actual_values, expected_keys, expected_values)

    assert proxy.online_attention
    assert proxy.online_attention_tile_positions == 5
    assert proxy.layer_positions(0) == 15
    assert [int(item[0].shape[2]) for item in chunks] == [5, 5, 2, 3]
    np.testing.assert_array_equal(
        np.asarray(actual_keys), np.asarray(expected_keys))
    np.testing.assert_array_equal(
        np.asarray(actual_values), np.asarray(expected_values))
    # The immutable base prompt is never extended by speculative nodes.
    assert kv.layer_positions(0) == 12


def test_tree_proxy_online_commit_appends_only_selected_path(tmp_path):
    kv = PagedKVCache(
        1, max_bytes=1, spill_dir=tmp_path / "source",
        page_positions=4, resident_pages=0)
    kv.online_attention = True
    base_keys = mx.zeros((1, 2, 8, 4), dtype=mx.float32)
    base_values = base_keys + 1
    kv.append_for_online_attention(0, base_keys, base_values)
    tree = SpeculativeTree(
        token_ids=(1, 2, 3, 4),
        depths=(0, 1, 1, 2),
        parents=(-1, 0, 0, 1),
        children=({2: 1, 3: 2}, {4: 3}, {}, {}),
    )
    proxy = QwenTreeKVProxy(kv, tree, 1)
    expected_keys = [base_keys]
    expected_values = [base_values]
    for node in range(4):
        proxy.current_node = node
        key = mx.full((1, 2, 1, 4), 10 + node, dtype=mx.float32)
        value = mx.full((1, 2, 1, 4), 20 + node, dtype=mx.float32)
        proxy.append_for_online_attention(0, key, value)
        if node in (0, 1, 3):
            expected_keys.append(key)
            expected_values.append(value)

    destination = PagedKVCache(
        1, max_bytes=1, spill_dir=tmp_path / "destination",
        page_positions=4, resident_pages=0)
    destination.append_for_online_attention(0, base_keys, base_values)
    proxy.commit_attention_path((0, 1, 3), destination)
    actual_keys, actual_values = destination.materialize_layer(0)
    expected_keys_array = mx.concatenate(expected_keys, axis=2)
    expected_values_array = mx.concatenate(expected_values, axis=2)
    mx.eval(actual_keys, actual_values)
    np.testing.assert_array_equal(
        np.asarray(actual_keys), np.asarray(expected_keys_array))
    np.testing.assert_array_equal(
        np.asarray(actual_values), np.asarray(expected_values_array))
    assert destination.layer_positions(0) == 11


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
