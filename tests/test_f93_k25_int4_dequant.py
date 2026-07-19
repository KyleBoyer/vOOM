"""F93: Kimi K2.5's compressed-tensors INT4 expert-weight dequantization.

Kimi K2.5's released checkpoint stores MoE expert FFN weights (only those --
attention/router weights are ordinary bf16 `.weight` tensors) as
vllm-project/compressed-tensors "pack-quantized" INT4:
`.weight_packed`/`.weight_scale`/`.weight_shape` triples instead of a plain
`.weight` tensor. runtime.quant.dequantize_compressed_tensors_int4
implements the unpacking; this test validates it two ways:

1. Against a verbatim copy of the REAL compressed-tensors unpack function
   (fetched via `gh api` from vllm-project/compressed-tensors on 2026-07-18,
   not reconstructed from memory) on synthetic random packed data, including
   a non-multiple-of-8 column count to exercise the truncation edge case.
2. Against a REAL packed weight tensor from the downloaded checkpoint
   (models/Kimi-K2.5), cross-checked against the same real unpack function
   applied to that same real tensor -- not just synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime.quant import dequantize_compressed_tensors_int4

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_MODEL_AVAILABLE = (MODEL_DIR / "config.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K2.5 is not available locally (a real ~554GB model, not fetched in CI)",
)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")


def _real_unpack_from_int32(value, num_bits, shape, packed_dim=1):
    """Verbatim from vllm-project/compressed-tensors
    src/compressed_tensors/compressors/pack_quantized/helpers.py (fetched via
    `gh api repos/vllm-project/compressed-tensors/contents/...` 2026-07-18).
    Used here only as a numerical oracle, never imported into the runtime."""
    if value.dtype is not torch.int32:
        raise ValueError(f"Expected {torch.int32} but got {value.dtype}, Aborting unpack.")
    if not 1 <= num_bits <= 8:
        raise ValueError(f"Unpacking is only supported for num_bits in [1, 8], got {num_bits}")
    if value.ndim > 2:
        return torch.stack([_real_unpack_from_int32(value[i], num_bits, shape[1:], packed_dim)
                             for i in range(value.shape[0])])
    if packed_dim == 0:
        value = value.transpose(0, 1)
    rows, num_words = value.shape
    cols = int(shape[packed_dim])
    if num_words % num_bits != 0:
        pad_words = num_bits - (num_words % num_bits)
        value = torch.nn.functional.pad(value, (0, pad_words))
        num_words += pad_words
    num_groups = num_words // num_bits
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, num_bits)
    elem_i = torch.arange(32, device=value.device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32
    lo_bits = torch.clamp(32 - bit_offset, max=num_bits)
    output_g = (value_g[:, word_idx] >> bit_offset.unsqueeze(0)) & ((1 << lo_bits) - 1).unsqueeze(0)
    ov_mask = lo_bits < num_bits
    hi_bits = num_bits - lo_bits[ov_mask]
    right = (value_g[:, word_idx[ov_mask] + 1] & ((1 << hi_bits) - 1).unsqueeze(0)) << lo_bits[ov_mask].unsqueeze(0)
    output_g[:, ov_mask] |= right
    output = output_g.view(rows, num_groups * 32)[:, :cols]
    if packed_dim == 0:
        output = output.transpose(0, 1)
    offset = 1 << (num_bits - 1)
    return (output - offset).to(torch.int8)


@_torch_skip
def test_int4_unpack_matches_real_compressed_tensors_on_synthetic_data():
    rng = np.random.default_rng(0)
    rows, cols = 5, 37  # not a multiple of 8 -- exercises the truncation path
    num_words = -(-cols // 8)
    packed_np = rng.integers(0, 2**31, size=(rows, num_words), endpoint=False).astype(np.int32)

    ref = _real_unpack_from_int32(torch.from_numpy(packed_np), num_bits=4,
                                   shape=(rows, cols), packed_dim=1)

    # Exercise the same nibble-extraction path dequantize_compressed_tensors_int4
    # uses internally, via scale=1 (pure unpack, no scale distortion) so we can
    # compare directly against the oracle's raw signed int4 output.
    scale = mx.ones((rows, 1), dtype=mx.bfloat16)  # group_size == cols
    dequant = dequantize_compressed_tensors_int4(mx.array(packed_np), scale, (rows, cols))
    mx.eval(dequant)

    assert not bool(mx.any(mx.isnan(dequant)).item())
    max_diff = np.max(np.abs(ref.numpy().astype(np.float32) - np.array(dequant.astype(mx.float32))))
    assert max_diff == 0, f"INT4 unpack mismatch vs real compressed-tensors oracle: {max_diff}"


@_model_skip
@_torch_skip
def test_int4_dequant_matches_oracle_on_real_k25_expert_weight():
    shard = mx.load(str(MODEL_DIR / "model-00005-of-000064.safetensors"))
    prefix = "language_model.model.layers.4.mlp.experts.0.down_proj"
    packed = shard[f"{prefix}.weight_packed"]
    scale = shard[f"{prefix}.weight_scale"]
    wshape_t = shard[f"{prefix}.weight_shape"]
    mx.eval(packed, scale, wshape_t)
    shape = tuple(int(v) for v in np.array(wshape_t))

    mine = dequantize_compressed_tensors_int4(packed, scale, shape)
    mx.eval(mine)
    assert mine.shape == shape
    assert not bool(mx.any(mx.isnan(mine)).item())

    packed_torch = torch.from_numpy(np.array(packed).astype(np.int32))
    scale_torch = torch.from_numpy(np.array(scale.astype(mx.float32)))
    unpacked = _real_unpack_from_int32(packed_torch, num_bits=4, shape=shape, packed_dim=1)
    group_size = shape[1] // scale_torch.shape[1]
    ref = unpacked.to(torch.float32) * scale_torch.repeat_interleave(group_size, dim=1)

    max_diff = np.max(np.abs(ref.numpy() - np.array(mine.astype(mx.float32))))
    # bf16 round-trip noise only (scale/output both pass through bf16); the
    # integer unpack itself is exact, verified separately above.
    assert max_diff < 1e-3, f"real-weight dequant mismatch vs oracle: {max_diff}"


@_model_skip
def test_weightstore_fetch_dequantizes_int4_experts_transparently():
    """2026-07-19: the two tests above validate the dequant MATH in
    isolation. This validates the actual WeightStore.fetch() integration --
    language_model.model.* prefix canonicalization, .weight_packed/
    .weight_scale/.weight_shape triplet detection, and the on-disk-
    quantization guard's compressed-tensors exemption -- all through the
    real production code path a live request actually uses, not a
    hand-rolled shortcut."""
    from runtime.model_loader import WeightStore

    store = WeightStore(str(MODEL_DIR))
    logical_name = "model.layers.4.mlp.experts.0.down_proj.weight"
    assert store.has(logical_name)
    out, _secs, nbytes = store.fetch([logical_name])
    w = out[logical_name]
    assert nbytes > 0
    assert w.shape == (7168, 2048)
    assert w.dtype == mx.bfloat16
    assert not bool(mx.any(mx.isnan(w)).item())

    # Cross-check against the standalone-verified dequant path (previous
    # test in this file) on the exact same tensor -- must match exactly,
    # not just "look reasonable".
    packed = mx.load(str(MODEL_DIR / "model-00005-of-000064.safetensors"))
    prefix = "language_model.model.layers.4.mlp.experts.0.down_proj"
    direct = dequantize_compressed_tensors_int4(
        packed[f"{prefix}.weight_packed"], packed[f"{prefix}.weight_scale"], (7168, 2048))
    mx.eval(direct)
    assert bool(mx.array_equal(w, direct).item())
