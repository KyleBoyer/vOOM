"""GGUF/ggml K-quant dequantization (Q4_K, Q6_K), needed to load GGUF-only
model releases (e.g. VibeThinker-3B's tool-calling fine-tune, only published
as GGUF) without lossy re-quantization.

runtime.quant.dequantize_gguf_q4_k / dequantize_gguf_q6_k implement the
unpacking; this test validates them two ways, mirroring
test_f128_k3_mxfp4_dequant.py's own established structure:

1. Against verbatim NumPy transcriptions of the REAL ggml-org/ggml C
   reference (`dequantize_row_q4_K`/`get_scale_min_k4`/`dequantize_row_q6_K`,
   `src/ggml-quants.c`, fetched 2026-07-28) on synthetic random packed data.
2. Against the same real reference functions applied to REAL packed tensor
   bytes from a downloaded GGUF checkpoint
   (models/VibeThinker-3B-GGUF-base/VibeThinker-3B.Q4_K_M.gguf), read via
   the `gguf` package (pip-installed for this verification only -- never
   imported into the runtime).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime.quant import dequantize_gguf_q4_k, dequantize_gguf_q6_k

ROOT = Path(__file__).resolve().parent.parent
GGUF_PATH = ROOT / "models" / "VibeThinker-3B-GGUF-base" / "VibeThinker-3B.Q4_K_M.gguf"
_GGUF_AVAILABLE = GGUF_PATH.exists()
_gguf_skip = pytest.mark.skipif(
    not _GGUF_AVAILABLE,
    reason="VibeThinker-3B GGUF file not available locally",
)

try:
    import gguf as gguf_pkg
    _GGUF_PKG_AVAILABLE = True
except ImportError:
    _GGUF_PKG_AVAILABLE = False
_gguf_pkg_skip = pytest.mark.skipif(not _GGUF_PKG_AVAILABLE, reason="gguf package not installed")

QK_K = 256


def _assert_close(ref: np.ndarray, mine: np.ndarray, label: str) -> None:
    # The reference runs the real ggml formula in float64; the runtime
    # (runtime/quant.py) runs the same formula in float32 -- with random
    # synthetic scale/min bytes (unlike real, well-behaved model weights)
    # intermediate magnitudes can be large enough that plain float32 vs
    # float64 rounding alone produces ~1e-3 absolute differences, so an
    # exact-zero check is too strict here (unlike the MXFP4/INT4 oracles,
    # whose formulas are exactly representable in both precisions).
    if not np.allclose(ref, mine, rtol=1e-4, atol=1e-4):
        max_diff = np.max(np.abs(ref - mine))
        raise AssertionError(f"{label} mismatch vs real ggml reference: max_diff={max_diff}")


# ---------------------------------------------------------------------------
# Verbatim NumPy transcriptions of ggml-org/ggml's real src/ggml-quants.c
# (fetched 2026-07-28). Used here only as a numerical oracle, never imported
# into the runtime, which uses vectorized MLX ops instead (runtime/quant.py).
# ---------------------------------------------------------------------------

def _get_scale_min_k4(j: int, q: np.ndarray) -> tuple[int, int]:
    """Verbatim from get_scale_min_k4 (ggml-quants.c:880)."""
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)
    return int(d), int(m)


def _real_dequantize_row_q4_k(d: float, dmin: float, scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Verbatim from dequantize_row_q4_K (ggml-quants.c:1529), one super-block."""
    y = np.zeros(QK_K, dtype=np.float64)
    q = qs
    is_ = 0
    yi = 0
    for j in range(0, QK_K, 64):
        sc, m = _get_scale_min_k4(is_ + 0, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _get_scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        qchunk = q[j // 64 * 32:j // 64 * 32 + 32]
        for l in range(32):
            y[yi] = d1 * (qchunk[l] & 0xF) - m1
            yi += 1
        for l in range(32):
            y[yi] = d2 * (qchunk[l] >> 4) - m2
            yi += 1
        is_ += 2
    return y


def _real_dequantize_row_q6_k(d: float, ql: np.ndarray, qh: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Verbatim from dequantize_row_q6_K (ggml-quants.c:1939), one super-block."""
    y = np.zeros(QK_K, dtype=np.float64)
    # C's `int8_t q1 = (int8_t)(...) - 32` computes in a signed int before the
    # cast; numpy uint8 arithmetic instead wraps (underflows) on `- 32`, so
    # widen to int64 first.
    ql = ql.astype(np.int64)
    qh = qh.astype(np.int64)
    for n in range(0, QK_K, 128):
        ql_o = ql[n // 128 * 64: n // 128 * 64 + 64]
        qh_o = qh[n // 128 * 32: n // 128 * 32 + 32]
        sc_o = sc[n // 128 * 8: n // 128 * 8 + 8]
        for l in range(32):
            is_ = l // 16
            q1 = ((ql_o[l + 0] & 0xF) | (((qh_o[l] >> 0) & 3) << 4)) - 32
            q2 = ((ql_o[l + 32] & 0xF) | (((qh_o[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql_o[l + 0] >> 4) | (((qh_o[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql_o[l + 32] >> 4) | (((qh_o[l] >> 6) & 3) << 4)) - 32
            y[n + l + 0] = d * sc_o[is_ + 0] * q1
            y[n + l + 32] = d * sc_o[is_ + 2] * q2
            y[n + l + 64] = d * sc_o[is_ + 4] * q3
            y[n + l + 96] = d * sc_o[is_ + 6] * q4
    return y


def _f16_to_f32(raw: np.ndarray) -> np.ndarray:
    # raw: uint8 bytes, last axis size 2 -- reinterpret the byte PAIR as one
    # float16 (a plain .astype() would convert each byte's VALUE separately,
    # not merge the two bytes -- must .view(), not .astype(), to reinterpret).
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    return raw.view("<u2").view("<f2").astype(np.float64)


def _ref_q4_k_rows(packed: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    n_super = in_features // QK_K
    blocks = packed.reshape(out_features, n_super, 144)
    out = np.zeros((out_features, in_features), dtype=np.float64)
    for r in range(out_features):
        for s in range(n_super):
            b = blocks[r, s]
            d = float(_f16_to_f32(b[0:2])[0])
            dmin = float(_f16_to_f32(b[2:4])[0])
            scales = b[4:16]
            qs = b[16:16 + 128]
            out[r, s * QK_K:(s + 1) * QK_K] = _real_dequantize_row_q4_k(d, dmin, scales, qs)
    return out


def _ref_q6_k_rows(packed: np.ndarray, out_features: int, in_features: int) -> np.ndarray:
    n_super = in_features // QK_K
    blocks = packed.reshape(out_features, n_super, 210)
    out = np.zeros((out_features, in_features), dtype=np.float64)
    for r in range(out_features):
        for s in range(n_super):
            b = blocks[r, s]
            ql = b[0:128]
            qh = b[128:128 + 64]
            sc_u8 = b[192:208].astype(np.int64)
            sc = np.where(sc_u8 >= 128, sc_u8 - 256, sc_u8)
            d = float(_f16_to_f32(b[208:210])[0])
            out[r, s * QK_K:(s + 1) * QK_K] = _real_dequantize_row_q6_k(d, ql, qh, sc)
    return out


def test_q4_k_matches_real_reference_on_synthetic_data():
    rng = np.random.default_rng(0)
    out_features, n_super = 3, 2
    in_features = n_super * QK_K
    packed_np = rng.integers(0, 256, size=(out_features, n_super * 144), endpoint=False).astype(np.uint8)

    ref = _ref_q4_k_rows(packed_np, out_features, in_features)
    mine = dequantize_gguf_q4_k(mx.array(packed_np), (out_features, in_features), out_dtype=mx.float32)
    mx.eval(mine)

    assert not bool(mx.any(mx.isnan(mine)).item())
    _assert_close(ref, np.array(mine), "Q4_K synthetic")


def test_q6_k_matches_real_reference_on_synthetic_data():
    rng = np.random.default_rng(1)
    out_features, n_super = 3, 2
    in_features = n_super * QK_K
    packed_np = rng.integers(0, 256, size=(out_features, n_super * 210), endpoint=False).astype(np.uint8)

    ref = _ref_q6_k_rows(packed_np, out_features, in_features)
    mine = dequantize_gguf_q6_k(mx.array(packed_np), (out_features, in_features), out_dtype=mx.float32)
    mx.eval(mine)

    assert not bool(mx.any(mx.isnan(mine)).item())
    _assert_close(ref, np.array(mine), "Q6_K synthetic")


@_gguf_skip
@_gguf_pkg_skip
def test_q4_k_matches_real_reference_on_real_gguf_tensor():
    reader = gguf_pkg.GGUFReader(str(GGUF_PATH))
    tensor = next(t for t in reader.tensors if t.name == "blk.0.attn_k.weight")
    assert tensor.tensor_type == gguf_pkg.GGMLQuantizationType.Q4_K
    # GGUF's own shape is reversed (in_features, out_features) vs PyTorch.
    in_features, out_features = (int(d) for d in tensor.shape)
    packed_np = np.asarray(tensor.data).reshape(out_features, -1)

    ref = _ref_q4_k_rows(packed_np, out_features, in_features)
    mine = dequantize_gguf_q4_k(mx.array(packed_np), (out_features, in_features), out_dtype=mx.float32)
    mx.eval(mine)
    assert mine.shape == (out_features, in_features)

    _assert_close(ref, np.array(mine), "real-weight Q4_K")


@_gguf_skip
@_gguf_pkg_skip
def test_q6_k_matches_real_reference_on_real_gguf_tensor():
    reader = gguf_pkg.GGUFReader(str(GGUF_PATH))
    tensor = next(t for t in reader.tensors if t.tensor_type == gguf_pkg.GGMLQuantizationType.Q6_K)
    in_features, out_features = (int(d) for d in tensor.shape)
    packed_np = np.asarray(tensor.data).reshape(out_features, -1)

    ref = _ref_q6_k_rows(packed_np, out_features, in_features)
    mine = dequantize_gguf_q6_k(mx.array(packed_np), (out_features, in_features), out_dtype=mx.float32)
    mx.eval(mine)
    assert mine.shape == (out_features, in_features)

    _assert_close(ref, np.array(mine), "real-weight Q6_K")
