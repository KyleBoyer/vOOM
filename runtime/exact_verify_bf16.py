"""Singleton-equivalent BF16 GEMV for exact speculative verification.

The target verifier has several independent activation rows but must preserve
the reduction order of ordinary one-token decode.  A conventional batched GEMM
can select a different reduction kernel and move released-model logits.  This
Metal kernel issues all rows together while retaining the one-token GEMV
accumulation order for every row.

Adapted from ``mlx-vlm``'s MIT-licensed exact speculative verifier at commit
8adff01e837b3e0f2304fc1f1e49f1c5c89d4c32.  Copyright (c) 2025 Prince
Canuma; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import mlx.core as mx


_EXACT_VERIFY_BF16_GEMV = (
    mx.fast.metal_kernel(
        name="voom_exact_verify_bf16_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=r"""
        uint lane = thread_position_in_grid.x;
        uint out_block = thread_position_in_grid.y;
        uint row = thread_position_in_grid.z;

        constexpr int TM = 4;
        constexpr int TN = 4;
        constexpr int SN = 32;
        constexpr int blockN = SN * TN;

        if (row >= R) {
            return;
        }
        int out_row = int(out_block * TM);
        if (out_row >= O) {
            return;
        }

        const device T* in_vec = x + row * K;
        const device T* mat = weight + out_row * K;
        float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
        int col = int(lane * TN);
        int n_iter = K / blockN;
        int leftover = K - blockN * n_iter;

        for (int iter = 0; iter < n_iter; ++iter) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] = static_cast<float>(in_vec[col + tn]);
            }
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    result[tm] +=
                        static_cast<float>(mat[tm * K + col + tn]) * v[tn];
                }
            }
            col += blockN;
        }

        if (leftover > 0) {
            float v[TN];
            for (int tn = 0; tn < TN; ++tn) {
                v[tn] =
                    (col + tn < K)
                    ? static_cast<float>(in_vec[col + tn]) : 0.0f;
            }
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    T m =
                        (col + tn < K) ? mat[tm * K + col + tn] : T(0);
                    result[tm] += static_cast<float>(m) * v[tn];
                }
            }
        }

        for (int tm = 0; tm < TM; ++tm) {
            for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
                result[tm] += simd_shuffle_down(result[tm], sn);
            }
        }
        if (lane == 0) {
            for (int tm = 0; tm < TM; ++tm) {
                out[row * O + out_row + tm] = static_cast<T>(result[tm]);
            }
        }
        """,
    )
    if mx.metal.is_available()
    else None
)


def exact_verify_bf16_available() -> bool:
    return _EXACT_VERIFY_BF16_GEMV is not None


def exact_verify_bf16_rejection_reason(
    x: mx.array,
    weight: mx.array,
) -> str | None:
    """Return the stable contract reason that prevents fused verification."""

    if _EXACT_VERIFY_BF16_GEMV is None:
        return "unavailable"
    # Packed lossless carriers deliberately implement only their own matmul
    # contract; they are not MLX arrays and have no rank/dtype attributes.
    # Reject them here so the verifier's existing singleton fallback can
    # dispatch each row through that carrier's exact QMV instead of crashing
    # while probing dense-kernel geometry.
    if not hasattr(weight, "ndim") or not hasattr(weight, "dtype"):
        return "weight_representation"
    if x.ndim != 3 or weight.ndim != 2:
        return "rank"
    if x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        return "dtype"
    if int(x.shape[-1]) != int(weight.shape[-1]):
        return "inner_dimension"
    batch, length, dimensions = (int(value) for value in x.shape)
    outputs = int(weight.shape[0])
    if batch <= 0:
        return "empty_batch"
    if length <= 1:
        return "singleton_window"
    if length > 8:
        return "window_too_wide"
    if outputs < 4 or outputs % 4:
        return "output_geometry"
    if dimensions >= 16 * outputs:
        return "skinny_output"
    return None


def exact_verify_bf16_matmul(
    x: mx.array,
    weight: mx.array,
) -> mx.array | None:
    """Return ``x @ weight.T`` with singleton GEMV reduction order.

    ``None`` means the geometry is outside the proven/performance-bounded
    kernel contract; callers must fall back to ordinary singleton calls.
    """

    if exact_verify_bf16_rejection_reason(x, weight) is not None:
        return None
    batch, length, dimensions = (int(value) for value in x.shape)
    outputs = int(weight.shape[0])

    rows = batch * length
    padded_rows = ((rows + 7) // 8) * 8
    source = mx.contiguous(x).reshape(rows, dimensions)
    out = _EXACT_VERIFY_BF16_GEMV(
        inputs=[source, weight],
        template=[
            ("T", x.dtype),
            ("K", dimensions),
            ("O", outputs),
            ("R", rows),
        ],
        grid=(32, outputs // 4, padded_rows),
        threadgroup=(32, 1, 8),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.reshape(batch, length, outputs)


__all__ = [
    "exact_verify_bf16_available",
    "exact_verify_bf16_matmul",
    "exact_verify_bf16_rejection_reason",
]
