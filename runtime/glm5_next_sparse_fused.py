"""Bounded fused sparse MLA attention for GLM-5.3 on Apple Metal.

The released exact path remains the MLX gather/FP32-score/softmax expression.
This kernel removes the enormous row-specific K/V gather by streaming selected
rows directly, but its online softmax and SIMD reductions reassociate floating
operations. It is therefore an explicit lossy candidate, never the lossless
default, even when greedy output happens to match a test prompt.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_SOURCE = r"""
    constexpr int LANES = 32;
    constexpr int GROUPS = 32;
    constexpr int KEY_EPT = DK / LANES;
    constexpr int VALUE_EPT = DV / LANES;

    const uint batch_head = threadgroup_position_in_grid.x;
    const uint query_index = threadgroup_position_in_grid.y;
    const uint batch = batch_head / H;
    const uint head = batch_head - batch * H;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;

    thread float qv[KEY_EPT];
    thread float ov[VALUE_EPT];
    threadgroup float partial_outputs[GROUPS * LANES];
    threadgroup float partial_max[GROUPS];
    threadgroup float partial_sum[GROUPS];

    const size_t q_base =
        (((size_t)batch * H + head) * Q + query_index) * DK;
    for (int j = 0; j < KEY_EPT; ++j) {
        const int dim = lane * KEY_EPT + j;
        qv[j] = static_cast<float>(query[q_base + dim])
            * static_cast<float>(scale[0]);
    }
    for (int j = 0; j < VALUE_EPT; ++j) {
        ov[j] = 0.0f;
    }

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    const size_t selection_base =
        ((size_t)batch * Q + query_index) * TOPK;
    for (int logical = group; logical < TOPK; logical += GROUPS) {
        const int selected = selection[selection_base + logical];
        if (selected >= 0 && selected < K) {
            const size_t key_base =
                (((size_t)batch * H + head) * K + selected) * DK;
            float score = 0.0f;
            for (int j = 0; j < KEY_EPT; ++j) {
                const int dim = lane * KEY_EPT + j;
                score += qv[j] * static_cast<float>(
                    keys[key_base + dim]);
            }
            score = simd_sum(score);
            const float next_max = max(running_max, score);
            const float old_factor = fast::exp(running_max - next_max);
            const float score_factor = fast::exp(score - next_max);
            running_max = next_max;
            running_sum = running_sum * old_factor + score_factor;

            const size_t value_base =
                (((size_t)batch * H + head) * K + selected) * DV;
            for (int j = 0; j < VALUE_EPT; ++j) {
                const int dim = lane * VALUE_EPT + j;
                const float value = static_cast<float>(
                    values[value_base + dim]);
                ov[j] = ov[j] * old_factor + score_factor * value;
            }
        }
    }

    if (lane == 0) {
        partial_max[group] = running_max;
        partial_sum[group] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float local_max = partial_max[lane];
    const float global_max = simd_max(local_max);
    const float correction = fast::exp(local_max - global_max);
    const float global_sum = simd_sum(partial_sum[lane] * correction);

    for (int j = 0; j < VALUE_EPT; ++j) {
        partial_outputs[lane * LANES + group] = ov[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value = simd_sum(
            partial_outputs[group * LANES + lane] * correction);
        value = global_sum == 0.0f ? 0.0f : value / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const size_t output_base =
                (((size_t)batch * H + head) * Q + query_index) * DV;
            output[output_base + group * VALUE_EPT + j] =
                static_cast<T>(value);
        }
    }
"""


_INT8_SOURCE = r"""
    constexpr int LANES = 32;
    constexpr int GROUPS = 32;
    constexpr int KEY_EPT = DK / LANES;
    constexpr int VALUE_EPT = DV / LANES;

    const uint batch_head = threadgroup_position_in_grid.x;
    const uint query_index = threadgroup_position_in_grid.y;
    const uint batch = batch_head / H;
    const uint head = batch_head - batch * H;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;

    thread float qv[KEY_EPT];
    thread float ov[VALUE_EPT];
    threadgroup float partial_outputs[GROUPS * LANES];
    threadgroup float partial_max[GROUPS];
    threadgroup float partial_sum[GROUPS];

    const size_t q_base =
        (((size_t)batch * H + head) * Q + query_index) * DK;
    for (int j = 0; j < KEY_EPT; ++j) {
        const int dim = lane * KEY_EPT + j;
        qv[j] = static_cast<float>(query[q_base + dim])
            * static_cast<float>(attention_scale[0]);
    }
    for (int j = 0; j < VALUE_EPT; ++j) {
        ov[j] = 0.0f;
    }

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    const size_t selection_base =
        ((size_t)batch * Q + query_index) * TOPK;
    for (int logical = group; logical < TOPK; logical += GROUPS) {
        const int selected = selection[selection_base + logical];
        if (selected >= 0 && selected < K) {
            const size_t row =
                ((size_t)batch * H + head) * K + selected;
            const size_t key_base = row * DK;
            const float key_scale = key_scales[row];
            float score = 0.0f;
            for (int j = 0; j < KEY_EPT; ++j) {
                const int dim = lane * KEY_EPT + j;
                score += qv[j]
                    * static_cast<float>(keys[key_base + dim])
                    * key_scale;
            }
            score = simd_sum(score);
            const float next_max = max(running_max, score);
            const float old_factor = fast::exp(running_max - next_max);
            const float score_factor = fast::exp(score - next_max);
            running_max = next_max;
            running_sum = running_sum * old_factor + score_factor;

            const size_t value_base = row * DV;
            const float value_scale = value_scales[row];
            for (int j = 0; j < VALUE_EPT; ++j) {
                const int dim = lane * VALUE_EPT + j;
                const float value =
                    static_cast<float>(values[value_base + dim])
                    * value_scale;
                ov[j] = ov[j] * old_factor + score_factor * value;
            }
        }
    }

    if (lane == 0) {
        partial_max[group] = running_max;
        partial_sum[group] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float local_max = partial_max[lane];
    const float global_max = simd_max(local_max);
    const float correction = fast::exp(local_max - global_max);
    const float global_sum = simd_sum(partial_sum[lane] * correction);

    for (int j = 0; j < VALUE_EPT; ++j) {
        partial_outputs[lane * LANES + group] = ov[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value = simd_sum(
            partial_outputs[group * LANES + lane] * correction);
        value = global_sum == 0.0f ? 0.0f : value / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const size_t output_base =
                (((size_t)batch * H + head) * Q + query_index) * DV;
            output[output_base + group * VALUE_EPT + j] =
                static_cast<T>(value);
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="voom_glm53_sparse_expanded_attention",
        input_names=[
            "query", "keys", "values", "selection", "scale"],
        output_names=["output"],
        source=_SOURCE,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=1)
def _int8_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="voom_glm53_sparse_expanded_attention_int8",
        input_names=[
            "query", "keys", "values", "key_scales", "value_scales",
            "selection", "attention_scale"],
        output_names=["output"],
        source=_INT8_SOURCE,
        compile_options={"math_mode": "fast"},
    )


def glm5_next_sparse_fused_attention(
    query: mx.array,
    keys: mx.array,
    values: mx.array,
    selection: mx.array,
    *,
    key_dim: int,
) -> mx.array:
    """Stream selected expanded K/V rows without materializing a gather."""
    if _kernel() is None:
        raise RuntimeError("GLM-5.3 fused sparse attention requires Metal")
    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("GLM-5.3 fused sparse attention expects rank-4 Q/K/V")
    batch, heads, queries, query_dim = map(int, query.shape)
    key_batch, key_heads, key_count, stored_key_dim = map(int, keys.shape)
    value_batch, value_heads, value_count, value_dim = map(int, values.shape)
    if (
        batch != key_batch
        or batch != value_batch
        or heads != key_heads
        or heads != value_heads
        or key_count != value_count
        or query_dim != int(key_dim)
        or stored_key_dim != int(key_dim)
        or int(key_dim) % 32
        or value_dim % 32
        or query.dtype != keys.dtype
        or query.dtype != values.dtype
    ):
        raise ValueError("GLM-5.3 fused sparse attention geometry mismatch")
    if (
        selection.ndim != 3
        or tuple(map(int, selection.shape[:2])) != (batch, queries)
        or int(selection.shape[2]) <= 0
    ):
        raise ValueError("GLM-5.3 fused sparse selection geometry mismatch")
    topk = int(selection.shape[2])
    return _kernel()(
        inputs=[
            query,
            keys,
            values,
            selection.astype(mx.int32),
            mx.array([int(key_dim) ** -0.5], dtype=mx.float32),
        ],
        template=[
            ("T", query.dtype),
            ("H", heads),
            ("Q", queries),
            ("K", key_count),
            ("TOPK", topk),
            ("DK", int(key_dim)),
            ("DV", value_dim),
        ],
        grid=(batch * heads * 1024, queries, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, heads, queries, value_dim)],
        output_dtypes=[query.dtype],
    )[0]


def glm5_next_sparse_fused_attention_int8(
    query: mx.array,
    keys: mx.array,
    values: mx.array,
    key_scales: mx.array,
    value_scales: mx.array,
    selection: mx.array,
    *,
    key_dim: int,
) -> mx.array:
    """Fused selected-row attention over scaled int8 prefill K/V."""
    if _int8_kernel() is None:
        raise RuntimeError("GLM-5.3 fused int8 sparse attention requires Metal")
    if query.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("GLM-5.3 fused int8 attention expects rank-4 Q/K/V")
    batch, heads, queries, query_dim = map(int, query.shape)
    key_batch, key_heads, key_count, stored_key_dim = map(int, keys.shape)
    value_batch, value_heads, value_count, value_dim = map(int, values.shape)
    expected_scale_shape = (batch, heads, key_count, 1)
    if (
        batch != key_batch
        or batch != value_batch
        or heads != key_heads
        or heads != value_heads
        or key_count != value_count
        or query_dim != int(key_dim)
        or stored_key_dim != int(key_dim)
        or int(key_dim) % 32
        or value_dim % 32
        or keys.dtype != mx.int8
        or values.dtype != mx.int8
        or tuple(map(int, key_scales.shape)) != expected_scale_shape
        or tuple(map(int, value_scales.shape)) != expected_scale_shape
        or key_scales.dtype != mx.float32
        or value_scales.dtype != mx.float32
    ):
        raise ValueError("GLM-5.3 fused int8 attention geometry mismatch")
    if (
        selection.ndim != 3
        or tuple(map(int, selection.shape[:2])) != (batch, queries)
        or int(selection.shape[2]) <= 0
    ):
        raise ValueError(
            "GLM-5.3 fused int8 selection geometry mismatch")
    topk = int(selection.shape[2])
    return _int8_kernel()(
        inputs=[
            query, keys, values, key_scales, value_scales,
            selection.astype(mx.int32),
            mx.array([int(key_dim) ** -0.5], dtype=mx.float32),
        ],
        template=[
            ("T", query.dtype),
            ("H", heads),
            ("Q", queries),
            ("K", key_count),
            ("TOPK", topk),
            ("DK", int(key_dim)),
            ("DV", value_dim),
        ],
        grid=(batch * heads * 1024, queries, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(batch, heads, queries, value_dim)],
        output_dtypes=[query.dtype],
    )[0]
