"""Experimental position-free paged attention for Apple Metal.

Keys are stored *before* RoPE in physical pages. A per-request block table maps
the logical sequence to those pages, and the kernel rotates each key at its
logical position while computing attention. Consequently one immutable physical
page can be referenced by multiple prompts at different positions without a
relocation copy or a position-specific mutation.

This is intentionally a narrow inference kernel: dense RoPE, row-contiguous
Q/K/V, head dimensions divisible by 32, and an explicit causal logical-position
mask. It is the serving primitive needed by shared/paged PIC; policy and page
lifetimes live above it.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_SOURCE = r"""
    constexpr int LANES = 32;
    constexpr int GROUPS = 32;
    constexpr int EPT = D / LANES;
    constexpr int GQA = NQ / NKV;

    const int TQ = counts[0];
    const int TK = counts[1];
    const uint batch_head = threadgroup_position_in_grid.x;
    const uint query_index = threadgroup_position_in_grid.y;
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
        (((size_t)batch * NQ + query_head) * TQ + query_index) * D;
    const int query_position = q_positions[query_index];
    for (int j = 0; j < EPT; ++j) {
        const int dim = lane * EPT + j;
        qv[j] = static_cast<float>(q[q_base + dim])
            * static_cast<float>(scale[0]);
        ov[j] = 0.0f;
    }

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;

    for (int logical = group; logical < TK; logical += GROUPS) {
        const int key_position = k_positions[logical];
        if (key_position <= query_position) {
            const int logical_block = logical / BLOCK;
            const int block_offset = logical - logical_block * BLOCK;
            const int physical_block = block_table[logical_block];
            const size_t page_base =
                (((size_t)physical_block * NKV + kv_head) * BLOCK
                 + block_offset) * D;

            float score = 0.0f;
            for (int j = 0; j < EPT; ++j) {
                const int dim = lane * EPT + j;
                const int rotary_dim = dim < HALF ? dim : dim - HALF;
                const size_t rope_offset =
                    (size_t)key_position * HALF + rotary_dim;
                const float c = static_cast<float>(cos_cache[rope_offset]);
                const float s = static_cast<float>(sin_cache[rope_offset]);
                const float own = static_cast<float>(k_pages[page_base + dim]);
                const int paired_dim = dim < HALF ? dim + HALF : dim - HALF;
                const float paired = static_cast<float>(
                    k_pages[page_base + paired_dim]);
                const float rotated_full = dim < HALF
                    ? own * c - paired * s
                    : own * c + paired * s;
                // Canonical MLX RoPE returns the key in its storage dtype before
                // SDPA consumes it.  Keep FP32 trigonometric tables (they match
                // MLX's angle calculation), then round the rotated key to T at
                // this same boundary instead of silently scoring a higher-
                // precision key in the fused path.
                const T rotated_stored = static_cast<T>(rotated_full);
                score += qv[j] * static_cast<float>(rotated_stored);
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
                    v_pages[page_base + dim]);
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

    for (int j = 0; j < EPT; ++j) {
        partial_outputs[lane * LANES + group] = ov[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float value = simd_sum(
            partial_outputs[group * LANES + lane] * correction);
        value = global_sum == 0.0f ? 0.0f : value / global_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const size_t out_base =
                (((size_t)batch * NQ + query_head) * TQ + query_index) * D;
            out[out_base + group * EPT + j] = static_cast<T>(value);
        }
    }
"""


@lru_cache(maxsize=1)
def _metal_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="voom_position_free_paged_attention",
        input_names=[
            "q", "k_pages", "v_pages", "block_table",
            "k_positions", "q_positions", "cos_cache", "sin_cache",
            "scale", "counts",
        ],
        output_names=["out"],
        source=_SOURCE,
        compile_options={"math_mode": "fast"},
    )


def rope_cache(max_position: int, head_dim: int, theta: float, *,
               dtype=mx.bfloat16, denominators=None) -> tuple[mx.array, mx.array]:
    if max_position <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError("RoPE cache dimensions must be positive and even")
    half = head_dim // 2
    if denominators is None:
        denominators = theta ** (
            mx.arange(half, dtype=mx.float32) / half)
    else:
        if denominators.shape != (half,):
            raise ValueError("RoPE denominators do not match head_dim")
        denominators = denominators.astype(mx.float32)
    positions = mx.arange(max_position, dtype=mx.float32)[:, None]
    angles = positions / denominators[None, :]
    cos = mx.cos(angles).astype(dtype)
    sin = mx.sin(angles).astype(dtype)
    mx.eval(cos, sin)
    return cos, sin


def apply_rope(value: mx.array, positions: mx.array,
               cos: mx.array, sin: mx.array) -> mx.array:
    """Reference NeoX/rotate-half RoPE using the same cache as the kernel."""
    head_dim = value.shape[-1]
    half = head_dim // 2
    selected_cos = cos[positions]
    selected_sin = sin[positions]
    first, second = value[..., :half], value[..., half:]
    return mx.concatenate((
        first * selected_cos[None, None] - second * selected_sin[None, None],
        second * selected_cos[None, None] + first * selected_sin[None, None],
    ), axis=-1)


def position_free_paged_attention(
        q: mx.array, k_pages: mx.array, v_pages: mx.array,
        block_table: mx.array, key_positions: mx.array,
        query_positions: mx.array, cos: mx.array, sin: mx.array, *,
        scale: float) -> mx.array:
    """Attend with rotated Q and physical, unrotated K/V pages.

    Shapes are ``q=[B,Nq,Tq,D]``, pages ``[P,Nkv,BLOCK,D]``, block table
    ``[ceil(Tk/BLOCK)]``, and position vectors ``key=[Tk]``, ``query=[Tq]``.
    Repeating a physical block id is legal and is the sharing primitive.
    """
    if _metal_kernel() is None:
        raise RuntimeError("position-free paged attention requires Apple Metal")
    if q.ndim != 4 or k_pages.ndim != 4 or v_pages.shape != k_pages.shape:
        raise ValueError("invalid Q/K/V page ranks or shapes")
    batch, num_q_heads, query_count, head_dim = q.shape
    pages, num_kv_heads, block_size, page_head_dim = k_pages.shape
    key_count = key_positions.size
    if (page_head_dim != head_dim or head_dim % 32
            or num_q_heads % num_kv_heads or query_positions.size != query_count):
        raise ValueError("unsupported head or position geometry")
    expected_blocks = (key_count + block_size - 1) // block_size
    if block_table.ndim != 1 or block_table.size != expected_blocks:
        raise ValueError("block table does not cover logical key positions")
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[1] * 2 != head_dim:
        raise ValueError("RoPE cache does not match head dimension")
    if not (q.dtype == k_pages.dtype == v_pages.dtype):
        raise ValueError("Q/K/V page dtypes must match")
    if cos.dtype != sin.dtype or cos.dtype not in (mx.float32, q.dtype):
        raise ValueError("RoPE cache must be FP32 or match the Q/K/V dtype")
    scale_value = mx.array([float(scale)], dtype=mx.float32)
    counts = mx.array([query_count, key_count], dtype=mx.int32)
    output = _metal_kernel()(
        inputs=[
            q, k_pages, v_pages,
            block_table.astype(mx.int32),
            key_positions.astype(mx.int32), query_positions.astype(mx.int32),
            cos, sin, scale_value, counts,
        ],
        template=[
            ("T", q.dtype), ("D", head_dim), ("HALF", head_dim // 2),
            ("NQ", num_q_heads), ("NKV", num_kv_heads),
            ("BLOCK", block_size),
        ],
        grid=(batch * num_q_heads * 1024, query_count, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    return output
