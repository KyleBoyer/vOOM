"""Explicit lossy tiled attention for one-token Qwen paged decode.

The released/lossless path continues to materialize the full BF16 K/V history
and call MLX SDPA.  This module is used only by an explicit fast-profile flag:
it consumes bounded contiguous page tiles in a fused Metal online-softmax
kernel, then combines their FP32 sufficient statistics.  The changed softmax
reduction order is numerically close but not bitwise identical to SDPA.
"""

from __future__ import annotations

import math
import time

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
_PAGE_KERNEL = None


_PAGE_SOURCE = r"""
    constexpr int LANES = 32;
    constexpr int GROUPS = 32;
    constexpr int EPT = D / LANES;
    constexpr int GQA = NQ / NKV;
    constexpr int PAGE_SLOTS = 8;

    int total_keys = 0;
    for (int page = 0; page < PAGE_SLOTS; ++page) {
        total_keys += counts[page];
    }
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

    const size_t q_base = (((size_t)batch * NQ + query_head) * D);
    for (int j = 0; j < EPT; ++j) {
        const int dim = lane * EPT + j;
        qv[j] = static_cast<float>(q[q_base + dim])
            * static_cast<float>(scale[0]);
        ov[j] = 0.0f;
    }

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (int logical = group; logical < total_keys; logical += GROUPS) {
        int page_index = 0;
        int local = logical;
        for (int page = 0; page < PAGE_SLOTS; ++page) {
            const int width = counts[page];
            if (local < width) {
                page_index = page;
                break;
            }
            local -= width;
        }

        const device T* page_keys = keys0;
        const device T* page_values = values0;
        if (page_index == 1) {
            page_keys = keys1; page_values = values1;
        } else if (page_index == 2) {
            page_keys = keys2; page_values = values2;
        } else if (page_index == 3) {
            page_keys = keys3; page_values = values3;
        } else if (page_index == 4) {
            page_keys = keys4; page_values = values4;
        } else if (page_index == 5) {
            page_keys = keys5; page_values = values5;
        } else if (page_index == 6) {
            page_keys = keys6; page_values = values6;
        } else if (page_index == 7) {
            page_keys = keys7; page_values = values7;
        }
        const int page_width = counts[page_index];
        const size_t key_base =
            (((size_t)batch * NKV + kv_head) * page_width + local) * D;
        float score = 0.0f;
        for (int j = 0; j < EPT; ++j) {
            const int dim = lane * EPT + j;
            score += qv[j] * static_cast<float>(page_keys[key_base + dim]);
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
                page_values[key_base + dim]);
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
            const size_t out_base = (((size_t)batch * NQ + query_head) * D);
            numerator[out_base + group * EPT + j] = value;
        }
    }
    if (lane == 0 && group == 0) {
        const size_t stats_base = (((size_t)batch * NQ + query_head) * 2);
        softmax_stats[stats_base] = global_max;
        softmax_stats[stats_base + 1] = global_sum;
    }
"""


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


def _page_kernel():
    global _PAGE_KERNEL
    if _PAGE_KERNEL is None:
        if not mx.metal.is_available():
            raise RuntimeError("Qwen page-native attention requires Metal")
        names = ["q"]
        for index in range(8):
            names.extend((f"keys{index}", f"values{index}"))
        names.extend(("scale", "counts"))
        _PAGE_KERNEL = mx.fast.metal_kernel(
            name="voom_qwen35_page_native_attention8",
            input_names=names,
            output_names=["numerator", "softmax_stats"],
            source=_PAGE_SOURCE,
            compile_options={"math_mode": "fast"},
        )
    return _PAGE_KERNEL


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


def _page_group_statistics(
    q: mx.array, pages: list[tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array, int]:
    if q.ndim != 4 or int(q.shape[2]) != 1:
        raise ValueError("page-native attention requires one query")
    if not 1 <= len(pages) <= 8:
        raise ValueError("page-native attention requires one to eight pages")
    batch, query_heads, _one, head_dim = map(int, q.shape)
    if head_dim % 32:
        raise ValueError(
            "page-native attention head dimension must be divisible by 32")
    counts = []
    inputs = [q]
    first_keys, first_values = pages[0]
    if first_keys.ndim != 4 or first_values.ndim != 4:
        raise ValueError("page-native attention K/V geometry mismatch")
    key_batch, kv_heads, _count, key_dim = map(int, first_keys.shape)
    if (first_keys.shape != first_values.shape
            or batch != key_batch or key_dim != head_dim
            or query_heads % kv_heads
            or q.dtype != first_keys.dtype
            or q.dtype != first_values.dtype):
        raise ValueError("page-native attention K/V geometry mismatch")
    for keys, values in pages:
        if (keys.ndim != 4 or keys.shape != values.shape
                or int(keys.shape[0]) != batch
                or int(keys.shape[1]) != kv_heads
                or int(keys.shape[2]) <= 0
                or int(keys.shape[3]) != head_dim
                or keys.dtype != first_keys.dtype
                or values.dtype != first_values.dtype):
            raise ValueError("page-native attention page geometry changed")
        inputs.extend((keys, values))
        counts.append(int(keys.shape[2]))
    while len(counts) < 8:
        inputs.extend((first_keys, first_values))
        counts.append(0)
    inputs.extend((
        mx.array([head_dim ** -0.5], dtype=mx.float32),
        mx.array(counts, dtype=mx.int32),
    ))
    outputs = _page_kernel()(
        inputs=inputs,
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
    return outputs[0], outputs[1], sum(counts)


def page_native_paged_attention(
    q: mx.array, kv, layer: int, *, pages_per_tile: int = 8,
) -> mx.array:
    """Attend over page groups without concatenating their exact BF16 K/V."""
    if (isinstance(pages_per_tile, bool)
            or not isinstance(pages_per_tile, int)
            or not 1 <= pages_per_tile <= 8):
        raise ValueError("page-native attention pages per tile must be in [1, 8]")
    iterator = getattr(kv, "iter_materialized_layer_pages", None)
    if not callable(iterator):
        raise TypeError("page-native attention requires a paged K/V cache")
    running_max = None
    running_sum = None
    running_numerator = None
    observed = 0
    groups = 0
    group = []
    started = time.perf_counter()

    def consume(pages):
        numerator, stats, positions = _page_group_statistics(q, pages)
        mx.eval(numerator, stats)
        return numerator, stats, positions

    for keys, values in iterator(int(layer)):
        group.append((keys, values))
        if len(group) < pages_per_tile:
            continue
        numerator, stats, positions = consume(group)
        groups += 1
        chunk_max = stats[..., :1]
        chunk_sum = stats[..., 1:2]
        if running_max is None:
            running_max, running_sum = chunk_max, chunk_sum
            running_numerator = numerator
        else:
            next_max = mx.maximum(running_max, chunk_max)
            old_factor = mx.exp(running_max - next_max)
            chunk_factor = mx.exp(chunk_max - next_max)
            running_numerator = (
                running_numerator * old_factor + numerator * chunk_factor)
            running_sum = running_sum * old_factor + chunk_sum * chunk_factor
            running_max = next_max
            mx.eval(running_numerator, running_sum, running_max)
        observed += positions
        group = []
    if group:
        numerator, stats, positions = consume(group)
        groups += 1
        chunk_max = stats[..., :1]
        chunk_sum = stats[..., 1:2]
        if running_max is None:
            running_max, running_sum = chunk_max, chunk_sum
            running_numerator = numerator
        else:
            next_max = mx.maximum(running_max, chunk_max)
            old_factor = mx.exp(running_max - next_max)
            chunk_factor = mx.exp(chunk_max - next_max)
            running_numerator = (
                running_numerator * old_factor + numerator * chunk_factor)
            running_sum = running_sum * old_factor + chunk_sum * chunk_factor
            running_max = next_max
            mx.eval(running_numerator, running_sum, running_max)
        observed += positions
    if running_numerator is None or running_sum is None:
        raise RuntimeError("page-native attention cache has no keys")
    expected = int(kv.layer_positions(int(layer)))
    if observed != expected:
        raise RuntimeError(
            f"page-native attention observed {observed}/{expected} positions")
    page_stats = getattr(kv, "stats", None)
    if page_stats is None:
        page_stats = getattr(getattr(kv, "base", None), "stats", None)
    if page_stats is not None:
        page_stats.page_native_calls += 1
        page_stats.page_native_groups += groups
        page_stats.page_native_positions += observed
        page_stats.page_native_s += time.perf_counter() - started
    return (running_numerator / running_sum).astype(q.dtype)


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
