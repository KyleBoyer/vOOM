"""F128: Kimi K3's compressed-tensors MXFP4 expert-weight dequantization.

Kimi K3's real, as-released checkpoint (`moonshotai/Kimi-K3`, checked
directly from its real `config.json` on 2026-07-27, hours after its
weights actually landed) stores MoE expert FFN weights (only those --
`config.json`'s own `quantization_config.ignore` list is
`["re:.*self_attn.*", ...]`, so attention stays ordinary BF16, matching
K2.5's own "experts only" pattern) as vllm-project/compressed-tensors
"mxfp4-pack-quantized": `.weight_packed`/`.weight_scale` pairs instead of a
plain `.weight` tensor -- no `.weight_shape` tensor is present (unlike
K2.5's INT4 format, which does ship one), so the logical shape must be
inferred as `packed.shape[1] * 2` (safe: MXFP4 always packs exactly 2 FP4
values per uint8 byte, unlike INT4's variable-width int32 packing that
genuinely needs a shape hint for its truncation/padding edge case).
runtime.quant.dequantize_compressed_tensors_mxfp4 implements the
unpacking; this test validates it two ways, mirroring
test_f93_k25_int4_dequant.py's own established structure:

1. Against verbatim copies of the REAL compressed-tensors functions
   (installed locally as `compressed-tensors==0.17.1` via pip for this
   verification only -- never imported into the runtime -- source copied
   here so the test has no new hard dependency) on synthetic random packed
   data.
2. Against a REAL packed weight tensor from the downloaded checkpoint
   (models/Kimi-K3), cross-checked against the same real reference
   functions applied to that same real tensor -- not just synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime import quant
from runtime.model_loader import WeightStore
from runtime.quant import dequantize_compressed_tensors_mxfp4

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
_SHARD = MODEL_DIR / "model-00005-of-000096.safetensors"
_MODEL_AVAILABLE = _SHARD.exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 shard 5 is not available locally (a real ~1.56TB model, "
           "downloading incrementally, not fetched in CI)",
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")


# Verbatim from the real, installed compressed-tensors==0.17.1 library
# (compressed_tensors/compressors/nvfp4/helpers.py -- MXFP4PackedCompressor
# inherits this unpack function unchanged from NVFP4, only scale
# compression differs) and compressed_tensors/compressors/mx_utils.py.
# Used here only as a numerical oracle, never imported into the runtime.
_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _real_unpack_fp4_from_uint8(a: torch.Tensor, m: int, n: int) -> torch.Tensor:
    assert a.dtype == torch.uint8
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    values = _E2M1_LUT[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n).to(torch.float32)


def _real_decompress_mx_scale(scale: torch.Tensor) -> torch.Tensor:
    scale_exp = scale.to(torch.int32) - 127
    return 2.0 ** scale_exp.to(torch.float32)


@_torch_skip
def test_mxfp4_unpack_matches_real_compressed_tensors_on_synthetic_data():
    rng = np.random.default_rng(0)
    rows, packed_cols = 5, 19
    logical_cols = packed_cols * 2
    group_size = logical_cols  # one scale column: exercises group_size==cols
    packed_np = rng.integers(0, 256, size=(rows, packed_cols), endpoint=False).astype(np.uint8)
    scale_np = rng.integers(120, 135, size=(rows, logical_cols // group_size), endpoint=False).astype(np.uint8)

    ref_unpacked = _real_unpack_fp4_from_uint8(
        torch.from_numpy(packed_np), rows, logical_cols)
    ref_scale = _real_decompress_mx_scale(torch.from_numpy(scale_np))
    ref = (ref_unpacked * ref_scale.repeat_interleave(group_size, dim=1)).numpy()

    mine = dequantize_compressed_tensors_mxfp4(
        mx.array(packed_np), mx.array(scale_np), (rows, logical_cols),
        out_dtype=mx.float32)
    mx.eval(mine)

    assert not bool(mx.any(mx.isnan(mine)).item())
    # MLX's Metal exponentiation differs from Torch's CPU implementation by
    # at most one or two float32 ULPs on GitHub's older M1 runner image even
    # though both implement the same E2M1 * 2**E8M0 arithmetic. This oracle is
    # a numerical equivalence gate, not a cross-device instruction-bit gate.
    np.testing.assert_allclose(
        np.array(mine), ref, rtol=0.0, atol=5e-7,
        err_msg="MXFP4 unpack mismatch vs real compressed-tensors oracle")


def _write_ct_mxfp4_fixture(path: Path, *, declared_group_size: int = 32):
    rows, cols = 8, 64
    rng = np.random.default_rng(11)
    packed = mx.array(rng.integers(
        0, 256, size=(rows, cols // 2), dtype=np.uint8))
    scales = mx.array(rng.integers(
        120, 132, size=(rows, cols // 32), dtype=np.uint8))
    prefix = "model.layers.0.block_sparse_moe.experts.0.w1"
    mx.save_safetensors(str(path / "model.safetensors"), {
        f"{prefix}.weight_packed": packed,
        f"{prefix}.weight_scale": scales,
    })
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "group_size": declared_group_size,
                        "scale_dtype": "torch.uint8",
                        "symmetric": True,
                        "type": "float",
                    },
                },
            },
        },
    }))
    return f"{prefix}.weight"


def test_weightstore_native_ct_mxfp4_is_exact_and_compact(tmp_path):
    name = _write_ct_mxfp4_fixture(tmp_path)
    dense_store = WeightStore(tmp_path)
    native_store = WeightStore(tmp_path, native_ct_mxfp4=True)

    dense, _dense_seconds, dense_read = dense_store.fetch([name])
    native, _native_seconds, native_read = native_store.fetch([name])
    dense_weight = dense[name]
    packed_weight = native[name]
    assert isinstance(packed_weight, quant.QTensor)
    assert (packed_weight.mode, packed_weight.bits, packed_weight.group_size) == (
        "mxfp4", 4, 32)
    assert dense_read == native_read == packed_weight.nbytes
    assert packed_weight.nbytes < dense_weight.nbytes

    native_dense = mx.dequantize(
        packed_weight.wq, scales=packed_weight.scales,
        group_size=32, bits=4, mode="mxfp4", dtype=mx.bfloat16)
    x = (mx.arange(64).reshape(1, 64) - 32).astype(mx.bfloat16) / 32
    dense_out = x @ dense_weight.T
    packed_out = quant.matmul(x, packed_weight)
    mx.eval(native_dense, dense_out, packed_out)
    assert mx.array_equal(native_dense, dense_weight)
    assert mx.array_equal(packed_out, dense_out)

    transform_ns, calls, input_bytes, resident_bytes = (
        native_store.stage_snapshot())
    assert transform_ns > 0
    assert calls == 1
    assert input_bytes == packed_weight.nbytes
    assert resident_bytes == packed_weight.nbytes
    assert native_store.expert_resident_bytes_per_weight == pytest.approx(
        4 / 8 + 1 / 32)


def test_weightstore_native_ct_mxfp4_rejects_non_native_descriptor(tmp_path):
    _write_ct_mxfp4_fixture(tmp_path, declared_group_size=64)
    with pytest.raises(NotImplementedError, match="bits=4 and group_size=32"):
        WeightStore(tmp_path, native_ct_mxfp4=True)


@_model_skip
@_torch_skip
def test_mxfp4_dequant_matches_oracle_on_real_k3_expert_weight():
    shard = mx.load(str(_SHARD))
    prefix = "language_model.model.layers.4.block_sparse_moe.experts.0.w1"
    packed = shard[f"{prefix}.weight_packed"]
    scale = shard[f"{prefix}.weight_scale"]
    mx.eval(packed, scale)
    rows, packed_cols = packed.shape
    shape = (rows, packed_cols * 2)

    mine = dequantize_compressed_tensors_mxfp4(packed, scale, shape, out_dtype=mx.float32)
    mx.eval(mine)
    assert mine.shape == shape
    assert not bool(mx.any(mx.isnan(mine)).item())

    packed_torch = torch.from_numpy(np.array(packed))
    scale_torch = torch.from_numpy(np.array(scale))
    ref_unpacked = _real_unpack_fp4_from_uint8(packed_torch, shape[0], shape[1])
    ref_scale = _real_decompress_mx_scale(scale_torch)
    group_size = shape[1] // scale.shape[1]
    ref = (ref_unpacked * ref_scale.repeat_interleave(group_size, dim=1)).numpy()

    max_diff = np.max(np.abs(ref - np.array(mine)))
    assert max_diff == 0, f"real-weight MXFP4 dequant mismatch vs oracle: {max_diff}"


@_model_skip
@pytest.mark.parametrize("projection", ["w1", "w2", "w3"])
def test_real_k3_expert_is_mlx_native_mxfp4_without_repacking(projection):
    shard = mx.load(str(_SHARD))
    prefix = (
        "language_model.model.layers.4.block_sparse_moe.experts.0."
        f"{projection}")
    packed = shard[f"{prefix}.weight_packed"]
    scales = shard[f"{prefix}.weight_scale"]
    mx.eval(packed, scales)
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    wq = packed.reshape(rows, -1).view(mx.uint32)

    reference = dequantize_compressed_tensors_mxfp4(
        packed, scales, (rows, cols), out_dtype=mx.bfloat16)
    native = mx.dequantize(
        wq, scales=scales, group_size=32, bits=4, mode="mxfp4",
        dtype=mx.bfloat16)
    x = mx.arange(cols, dtype=mx.float32).reshape(1, cols)
    x = (((x % 257) - 128) / 128).astype(mx.bfloat16)
    dense_out = x @ reference.T
    native_out = mx.quantized_matmul(
        x, wq, scales=scales, transpose=True,
        group_size=32, bits=4, mode="mxfp4")
    mx.eval(reference, native, dense_out, native_out)

    assert mx.array_equal(native, reference)
    assert mx.array_equal(native_out, dense_out)
