"""Explicit lossy tiled attention for one-token Qwen paged decode.

The released/lossless path continues to materialize the full BF16 K/V history
and call MLX SDPA.  This module is used only by an explicit fast-profile flag:
it consumes bounded contiguous page tiles in a fused Metal online-softmax
kernel, then combines their FP32 sufficient statistics.  The changed softmax
reduction order is numerically close but not bitwise identical to SDPA.
"""

from __future__ import annotations

import math

import mlx.core as mx


_SOURCE = r"""
    constexpr int LANES = 32;
    constexpr int GROUPS = 32;
    constexpr int EPT = D / LANES;
    constexpr int GQA = NQ / NKV;

    const int TK = counts[0];
    const uint batch_head = threadgroup_position_in_grid.x;
    const uint batch = batch_head / NQ;
    const uint query_head = batch_head - batch * NQ;
    const uint kv_head = query_head / GQA;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;

    thread float qv[EPT];
    thread float ov[EPT];
    threadgroup float partial_outputs[GROUPS * LANES];
    threadgroup float partial_max[GROUPS];
    threadgroup float partial_sum[GROUPS];

    const size_t q_base =
        (((size_t)batch * NQ + query_head) * D);
    for (int j = 0; j < EPT; ++j) {
        const int dim = lane * EPT + j;
        qv[j] = static_cast<float>(q[q_base + dim])
            * static_cast<float>(scale[0]);
        ov[j] = 0.0f;
    }

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (int logical = group; logical < TK; logical += GROUPS) {
        const size_t key_base =
            (((size_t)batch * NKV + kv_head) * TK + logical) * D;
        float score = 0.0f;
        for (int j = 0; j < EPT; ++j) {
            const int dim = lane * EPT + j;
            score += qv[j] * static_cast<float>(keys[key_base + dim]);
        }
        score = simd_sum(score);
        const float next_max = max(running_max, score);
        const float old_factor = fast::exp(running_max - next_max);
        const float score_factor = fast::exp(score - next_max);
        running_max = next_max;
        running_sum = running_sum * old_factor + score_factor;
        for (int j = 0; j < EPT; ++j) {
            const int dim = lane * EPT + j;
            const float value = static_cast<float>(
                values[key_base + dim]);
            ov[j] = ov[j] * old_factor + score_factor * value;
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

    for (int j = 0; j < EPT; ++j) {
        partial_outputs[lane * LANES + group] = ov[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float value = simd_sum(
            partial_outputs[group * LANES + lane] * correction);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const size_t out_base =
                (((size_t)batch * NQ + query_head) * D);
            numerator[out_base + group * EPT + j] = value;
        }
    }
    if (lane == 0 && group == 0) {
        const size_t stats_base =
            (((size_t)batch * NQ + query_head) * 2);
        softmax_stats[stats_base] = global_max;
        softmax_stats[stats_base + 1] = global_sum;
    }
"""


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        if not mx.metal.is_available():
            raise RuntimeError("Qwen tiled paged attention requires Metal")
        _KERNEL = mx.fast.metal_kernel(
            name="voom_qwen35_tiled_paged_attention",
            input_names=["q", "keys", "values", "scale", "counts"],
            output_names=["numerator", "softmax_stats"],
            source=_SOURCE,
            compile_options={"math_mode": "fast"},
        )
    return _KERNEL


def _tile_statistics(
    q: mx.array, keys: mx.array, values: mx.array,
) -> tuple[mx.array, mx.array]:
    if q.ndim != 4 or int(q.shape[2]) != 1:
        raise ValueError("tiled paged attention requires one query")
    if keys.ndim != 4 or keys.shape != values.shape:
        raise ValueError("tiled paged attention K/V geometry mismatch")
    batch, query_heads, _one, head_dim = map(int, q.shape)
    key_batch, kv_heads, key_count, key_dim = map(int, keys.shape)
    if (batch != key_batch or key_count <= 0 or key_dim != head_dim
            or query_heads % kv_heads):
        raise ValueError("tiled paged attention head geometry mismatch")
    outputs = _kernel()(
        inputs=[
            q, keys, values,
            mx.array([head_dim ** -0.5], dtype=mx.float32),
            mx.array([key_count], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype), ("D", head_dim),
            ("NQ", query_heads), ("NKV", kv_heads),
        ],
        grid=(batch * query_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (batch, query_heads, 1, head_dim),
            (batch, query_heads, 1, 2),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs[0], outputs[1]


def tiled_paged_attention(
    q: mx.array, kv, layer: int, *, tile_positions: int = 2048,
) -> mx.array:
    """Attend over bounded exact page tiles with reassociated softmax math."""
    if (isinstance(tile_positions, bool) or not isinstance(tile_positions, int)
            or tile_positions <= 0):
        raise ValueError("paged-attention tile positions must be positive")
    chunks = kv.iter_materialized_layer_chunks(
        int(layer), max_positions=int(tile_positions))
    running_max = None
    running_sum = None
    running_numerator = None
    observed = 0
    for keys, values in chunks:
        key_count = int(keys.shape[2])
        if key_count <= 0:
            continue
        numerator, stats = _tile_statistics(q, keys, values)
        mx.eval(numerator, stats)
        chunk_max = stats[..., :1]
        chunk_sum = stats[..., 1:2]
        if running_max is None:
            running_max = chunk_max
            running_sum = chunk_sum
            running_numerator = numerator
        else:
            next_max = mx.maximum(running_max, chunk_max)
            old_factor = mx.exp(running_max - next_max)
            chunk_factor = mx.exp(chunk_max - next_max)
            running_numerator = (
                running_numerator * old_factor
                + numerator * chunk_factor)
            running_sum = (
                running_sum * old_factor + chunk_sum * chunk_factor)
            running_max = next_max
            mx.eval(running_numerator, running_sum, running_max)
        observed += key_count
    if running_numerator is None or running_sum is None:
        raise RuntimeError("paged attention cache has no keys")
    expected = int(kv.layer_positions(int(layer)))
    if observed != expected:
        raise RuntimeError(
            f"paged attention observed {observed}/{expected} positions")
    return (running_numerator / running_sum).astype(q.dtype)


def theoretical_tile_count(positions: int, tile_positions: int) -> int:
    """Pure helper used by configuration/telemetry tests."""
    if positions <= 0 or tile_positions <= 0:
        raise ValueError("positions and tile width must be positive")
    return math.ceil(int(positions) / int(tile_positions))
