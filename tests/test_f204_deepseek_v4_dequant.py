"""F204: bit-exactness oracle for DeepSeek V4's two block-scaled formats.

CORRECTED. The first version of this file concluded that the 35,328 routed
expert tensors are INT8, because the safetensors header says ``I8``. The header
describes the storage CONTAINER: each byte holds two E2M1 FP4 codes, so the
logical width is twice the stored one, and ``config.json``'s
``expert_dtype: "fp4"`` was right all along. Three checks agree -- the unpacked
shape is the only one whose in/out features chain with hidden_size and
moe_intermediate_size, the stored byte count is exactly half the logical value
count, and the scale granularity comes to 32 logical values, matching the
released ``fp4_block_size``.

That error survived this file because the INT8 test compared against a numpy
reimplementation of the same assumption -- self-referential on exactly the
packing question, unlike the FP8/E8M0 tests which reference ml_dtypes. It was
caught only by a real forward pass, as a dimension mismatch. The superseded
tests are kept below under ``_superseded_`` names rather than deleted.

``ml_dtypes.float8_e4m3fn`` and ``float8_e8m0fnu`` remain the reference for the
FP8 path precisely because they are not MLX: agreeing with ourselves would
prove nothing.

``ml_dtypes.float8_e4m3fn`` and ``float8_e8m0fnu`` are the reference here
precisely because they are not MLX: agreeing with ourselves would prove
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "DeepSeek-V4-Flash-0731"


def test_e8m0_scale_matches_ml_dtypes_for_every_byte():
    """All 256 encodings, not a sample: 0xFF is NaN and must not be clamped."""
    import mlx.core as mx
    import ml_dtypes
    import numpy as np

    from runtime.quant import decode_e8m0_scale

    raw = np.arange(256, dtype=np.uint8)
    reference = raw.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    got = np.array(decode_e8m0_scale(mx.array(raw)))

    finite = np.isfinite(reference)
    assert finite.sum() == 255, "expected exactly one non-finite encoding"
    assert np.array_equal(got[finite], reference[finite]), (
        "E8M0 decode disagrees with ml_dtypes on a finite encoding")
    assert not np.isfinite(got[~finite]).any()


def test_fp8_e4m3_decode_matches_ml_dtypes_for_every_byte():
    import mlx.core as mx
    import ml_dtypes
    import numpy as np

    raw = np.arange(256, dtype=np.uint8)
    reference = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    got = np.array(mx.from_fp8(mx.array(raw), mx.float32))

    finite = np.isfinite(reference)
    assert np.array_equal(got[finite], reference[finite]), (
        "mx.from_fp8 disagrees with ml_dtypes.float8_e4m3fn")


def test_block_scale_broadcast_rejects_a_non_dividing_shape():
    """A mismatched scale must raise, not dequantize to plausible noise."""
    import mlx.core as mx

    from runtime.quant import _broadcast_block_scale

    with pytest.raises(ValueError, match="exact multiple"):
        _broadcast_block_scale(mx.ones((3, 4)), (8, 8))
    with pytest.raises(ValueError, match="rank"):
        _broadcast_block_scale(mx.ones((4,)), (8, 8))


def _superseded_int8_symmetry():
    import mlx.core as mx
    import numpy as np

    from runtime.quant import dequantize_deepseek_v4_int8

    weight = mx.array(np.array([[-128, -1, 0, 1, 127]] * 2, dtype=np.int8))
    # exponent 127 -> scale 1.0, so the result must be the raw integers
    scale = mx.array(np.full((2, 1), 127, dtype=np.uint8))
    got = np.array(dequantize_deepseek_v4_int8(weight, scale))
    assert np.array_equal(got, np.array(weight, dtype=np.float32))

    # exponent 128 -> scale 2.0 applied per row-block
    scale2 = mx.array(np.full((2, 1), 128, dtype=np.uint8))
    got2 = np.array(dequantize_deepseek_v4_int8(weight, scale2))
    assert np.array_equal(got2, np.array(weight, dtype=np.float32) * 2.0)


def _superseded_int8_dtype_guard():
    import mlx.core as mx

    from runtime.quant import (dequantize_deepseek_v4_fp8,
                               dequantize_deepseek_v4_int8)

    with pytest.raises(ValueError, match="int8"):
        dequantize_deepseek_v4_int8(mx.zeros((4, 4), mx.uint8),
                                    mx.zeros((4, 1), mx.uint8))
    with pytest.raises(ValueError, match="uint8"):
        dequantize_deepseek_v4_fp8(mx.zeros((4, 4), mx.int8),
                                   mx.zeros((1, 1), mx.uint8))


# ---- against the real released checkpoint ---------------------------------

realmodel = pytest.mark.skipif(
    not (MODEL / "model.safetensors.index.json").is_file(),
    reason="DeepSeek-V4-Flash-0731 not present")


@pytest.fixture(scope="module")
def index():
    import json

    return json.loads(
        (MODEL / "model.safetensors.index.json").read_text())["weight_map"]


def _load(index, name):
    import mlx.core as mx

    return mx.load(str(MODEL / index[name]))[name]


@realmodel
def _superseded_real_routed_int8(index):
    """End-to-end on a real routed expert, referenced to ml_dtypes."""
    import ml_dtypes
    import numpy as np

    from runtime.quant import dequantize_deepseek_v4_int8

    weight = _load(index, "layers.5.ffn.experts.0.w1.weight")
    scale = _load(index, "layers.5.ffn.experts.0.w1.scale")
    assert str(weight.dtype).endswith("int8"), (
        f"released routed expert should be int8, got {weight.dtype} -- "
        "config.json's 'fp8' and the HF API's 'FP4' both misdescribe it")

    got = np.array(dequantize_deepseek_v4_int8(weight, scale))

    raw_w = np.array(weight).astype(np.float32)
    raw_s = np.array(scale).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    columns = raw_w.shape[1] // raw_s.shape[1]
    expected = raw_w * np.repeat(raw_s, columns, axis=1)
    assert np.array_equal(got, expected)
    assert np.isfinite(got).all()


@realmodel
def test_real_shared_expert_uses_128x128_fp8_blocks(index):
    import ml_dtypes
    import mlx.core as mx
    import numpy as np

    from runtime.quant import dequantize_deepseek_v4_fp8

    weight = _load(index, "layers.5.ffn.shared_experts.w1.weight")
    scale = _load(index, "layers.5.ffn.shared_experts.w1.scale")
    assert weight.shape[0] // scale.shape[0] == 128
    assert weight.shape[1] // scale.shape[1] == 128

    got = np.array(dequantize_deepseek_v4_fp8(weight, scale).astype(mx.float32))
    raw_w = np.array(weight).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    raw_s = np.array(scale).view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    expected = raw_w * np.repeat(
        np.repeat(raw_s, 128, axis=0), 128, axis=1)
    # The dequant now emits bfloat16 -- routed experts are the resident bulk
    # and float32 quadrupled them against the packed source. Compare at
    # bfloat16 resolution rather than requiring bit equality with float32.
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)


@realmodel
def _superseded_blocking_differs(index):
    """The two schemes are not interchangeable; pin that they differ."""
    routed_w = _load(index, "layers.5.ffn.experts.0.w1.weight")
    routed_s = _load(index, "layers.5.ffn.experts.0.w1.scale")
    dense_w = _load(index, "layers.5.ffn.shared_experts.w1.weight")
    dense_s = _load(index, "layers.5.ffn.shared_experts.w1.scale")

    routed_block = (routed_w.shape[0] // routed_s.shape[0],
                    routed_w.shape[1] // routed_s.shape[1])
    dense_block = (dense_w.shape[0] // dense_s.shape[0],
                   dense_w.shape[1] // dense_s.shape[1])
    assert routed_block == (1, 16), routed_block
    assert dense_block == (128, 128), dense_block
    assert routed_block != dense_block


# ---- CORRECTION: routed experts are packed FP4, not INT8 -------------------
#
# The safetensors header says I8, which describes the storage container. Each
# byte holds TWO E2M1 codes, so the logical width is twice the stored one. The
# earlier INT8 tests are retained above under _superseded_ names rather than
# deleted, so the record shows what was believed and why it was wrong: they
# compared against a numpy reimplementation of the same assumption, which is
# self-referential on exactly the packing question. Only a real forward pass
# surfaced it, via a dimension mismatch.


@realmodel
def test_routed_expert_unpacks_to_a_chaining_shape(index):
    import json

    import mlx.core as mx

    from runtime.quant import dequantize_deepseek_v4_fp4

    config = json.loads((MODEL / "config.json").read_text())
    shards = {}
    weight = _load(index, f"layers.{5}.ffn.experts.0.w1.weight")
    scale = _load(index, f"layers.{5}.ffn.experts.0.w1.scale")
    out = dequantize_deepseek_v4_fp4(weight, scale)

    # Stored (2048, 2048) chains with nothing; unpacked (2048, 4096) has
    # in-features == hidden_size and out-features == moe_intermediate_size.
    assert out.shape == (config["moe_intermediate_size"],
                         config["hidden_size"])
    assert weight.shape[1] * 2 == out.shape[1], "expected two codes per byte"
    # 32 logical values per scale, matching the released fp4_block_size.
    assert out.shape[1] // scale.shape[1] == 32


@realmodel
def test_routed_expert_values_lie_on_the_e2m1_grid(index):
    """Every magnitude must be a representable E2M1 code times its scale."""
    import mlx.core as mx
    import numpy as np

    from runtime.quant import dequantize_deepseek_v4_fp4

    weight = _load(index, "layers.5.ffn.experts.0.w1.weight")
    scale = _load(index, "layers.5.ffn.experts.0.w1.scale")
    out = np.array(dequantize_deepseek_v4_fp4(weight, scale).astype(mx.float32))
    assert np.isfinite(out).all()

    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    block = out[:4, :32]
    nonzero = np.abs(block[block != 0])
    ratios = nonzero / nonzero.min()
    # Each block shares one power-of-two scale, so ratios must be grid ratios.
    allowed = set(np.round(grid[1:] / grid[1], 4).tolist())
    assert set(np.round(ratios, 4).tolist()) <= allowed, (
        "values do not lie on the E2M1 grid; the packing is wrong")


def test_fp4_rejects_a_float_container():
    import mlx.core as mx

    from runtime.quant import dequantize_deepseek_v4_fp4

    with pytest.raises(ValueError, match="int8/uint8"):
        dequantize_deepseek_v4_fp4(mx.zeros((4, 4), mx.float32),
                                   mx.zeros((4, 1), mx.uint8))
