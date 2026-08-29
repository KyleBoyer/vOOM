"""Bounded numeric and validation gates for lossy fused GLM-5.3 DSA."""

from __future__ import annotations

import mlx.core as mx
import pytest

from runtime.glm5_next import (
    _glm5_next_quantize_expanded_rows_int8,
    _glm5_next_sparse_expanded_attention,
)
from runtime.glm5_next_sparse_fused import (
    glm5_next_sparse_fused_attention,
    glm5_next_sparse_fused_attention_int8,
)


def test_glm53_fused_sparse_matches_reference_numerically_without_claiming_exact():
    mx.random.seed(17)
    batch, heads, queries, keys = 1, 2, 3, 71
    key_dim, value_dim, topk = 64, 32, 64
    query = mx.random.normal(
        (batch, heads, queries, key_dim)).astype(mx.bfloat16)
    stored_keys = mx.random.normal(
        (batch, heads, keys, key_dim)).astype(mx.bfloat16)
    stored_values = mx.random.normal(
        (batch, heads, keys, value_dim)).astype(mx.bfloat16)
    selection = mx.stack([
        mx.stack([
            (mx.arange(topk, dtype=mx.int32) + row * 3) % keys
            for row in range(queries)
        ])
    ])
    selection[:, 1, -3:] = -1

    reference = _glm5_next_sparse_expanded_attention(
        query, stored_keys, stored_values, selection,
        key_dim=key_dim, query_tile_size=2)
    candidate = glm5_next_sparse_fused_attention(
        query, stored_keys, stored_values, selection, key_dim=key_dim)
    mx.eval(reference, candidate)

    difference = mx.max(mx.abs(
        reference.astype(mx.float32) - candidate.astype(mx.float32))).item()
    cosine = mx.sum(
        reference.astype(mx.float32) * candidate.astype(mx.float32)
    ) / mx.sqrt(
        mx.sum(reference.astype(mx.float32) ** 2)
        * mx.sum(candidate.astype(mx.float32) ** 2)
    )
    assert tuple(candidate.shape) == (batch, heads, queries, value_dim)
    assert difference < 0.02
    assert cosine.item() > 0.999


def test_glm53_fused_sparse_rejects_bad_geometry():
    query = mx.zeros((1, 2, 1, 64), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 4, 64), dtype=mx.bfloat16)
    values = mx.zeros((1, 2, 4, 32), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="selection geometry"):
        glm5_next_sparse_fused_attention(
            query, keys, values, mx.zeros((1, 2), dtype=mx.int32),
            key_dim=64)


def test_glm53_fused_int8_sparse_tracks_dequantized_fused_reference():
    mx.random.seed(29)
    batch, heads, queries, keys = 1, 2, 3, 97
    key_dim, value_dim, topk = 64, 32, 64
    query = mx.random.normal(
        (batch, heads, queries, key_dim)).astype(mx.bfloat16)
    stored_keys = mx.random.normal(
        (batch, heads, keys, key_dim)).astype(mx.bfloat16)
    stored_values = mx.random.normal(
        (batch, heads, keys, value_dim)).astype(mx.bfloat16)
    qkeys, key_scales = _glm5_next_quantize_expanded_rows_int8(stored_keys)
    qvalues, value_scales = _glm5_next_quantize_expanded_rows_int8(
        stored_values)
    selection = mx.stack([mx.stack([
        (mx.arange(topk, dtype=mx.int32) + row * 5) % keys
        for row in range(queries)
    ])])
    dequantized = glm5_next_sparse_fused_attention(
        query,
        (qkeys.astype(mx.float32) * key_scales).astype(mx.bfloat16),
        (qvalues.astype(mx.float32) * value_scales).astype(mx.bfloat16),
        selection, key_dim=key_dim)
    candidate = glm5_next_sparse_fused_attention_int8(
        query, qkeys, qvalues, key_scales, value_scales, selection,
        key_dim=key_dim)
    mx.eval(dequantized, candidate)

    difference = mx.max(mx.abs(
        dequantized.astype(mx.float32)
        - candidate.astype(mx.float32))).item()
    cosine = mx.sum(
        dequantized.astype(mx.float32) * candidate.astype(mx.float32)
    ) / mx.sqrt(
        mx.sum(dequantized.astype(mx.float32) ** 2)
        * mx.sum(candidate.astype(mx.float32) ** 2)
    )
    assert tuple(candidate.shape) == (batch, heads, queries, value_dim)
    assert difference < 0.02
    assert cosine.item() > 0.999
