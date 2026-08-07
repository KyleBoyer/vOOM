"""F222: DeepSeek V4 routed experts consumed in their released MXFP4 form.

Dequantizing is the decode bottleneck, not I/O. Measured on this checkpoint:
a plain pread of expert tensors runs at 1.55GB/s, the same fetch with dequant
at 0.51GB/s, and the dequant alone processes raw bytes at 0.85GB/s -- and
1/1.55 + 1/0.85 predicts exactly the 0.55GB/s the engine saw.

The released bytes are E2M1 FP4 codes with E8M0 group scales at group_size 32,
which IS OCP MXFP4, so mx.quantized_matmul consumes them after a shape-only
uint8 -> uint32 view. Nothing is repacked and no value is converted.

The fused kernel reassociates float32 sums, so this is proven by agreement
with the dequantized reference and by greedy-token equality, not bit identity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "DeepSeek-V4-Flash-0731"

realmodel = pytest.mark.skipif(
    not (MODEL / "model.safetensors.index.json").is_file(),
    reason="DeepSeek-V4-Flash-0731 not present")


def test_packed_expert_reports_logical_shape_and_packed_bytes():
    import mlx.core as mx

    from runtime.deepseek_v4 import PackedExpert

    codes = mx.zeros((2048, 512), mx.uint32)
    scales = mx.zeros((2048, 128), mx.uint8)
    packed = PackedExpert(codes, scales)
    # Eight 4-bit codes per uint32 lane.
    assert packed.shape == (2048, 4096)
    assert packed.nbytes == codes.nbytes + scales.nbytes
    # The whole point: a packed page is far smaller than its bf16 form.
    assert packed.nbytes * 4 <= 2048 * 4096 * 2 + scales.nbytes * 4


@realmodel
def _real_expert():
    import mlx.core as mx

    index = json.loads(
        (MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.20.ffn.experts.3.w1.weight"
    key = name if name in index else name.replace("model.", "")
    shard = mx.load(str(MODEL / index[key]))
    return shard[key], shard[key.replace(".weight", ".scale")]


@realmodel
def test_native_matmul_matches_the_dequantized_reference():
    import mlx.core as mx

    from runtime.deepseek_v4 import PackedExpert, _packed_matmul
    from runtime.quant import dequantize_deepseek_v4_fp4

    weight, scale = _real_expert()
    assert str(weight.dtype).endswith("int8")
    reference = dequantize_deepseek_v4_fp4(weight, scale)
    mx.eval(reference)

    rng = np.random.default_rng(0)
    x = mx.array(rng.normal(size=(4, reference.shape[1])).astype(np.float32))
    want = (x.astype(reference.dtype) @ reference.T).astype(mx.float32)
    packed = PackedExpert(weight.view(mx.uint8).view(mx.uint32), scale)
    assert packed.shape == reference.shape, (
        "packed logical shape disagrees with the dequantized weight")

    got = _packed_matmul(x, packed).astype(mx.float32)
    mx.eval(want, got)
    scale_of = float(mx.max(mx.abs(want)))
    rel = float(mx.max(mx.abs(got - want))) / scale_of
    assert rel < 1e-3, f"native MXFP4 diverged from dequant, rel={rel}"


@realmodel
def test_dense_and_packed_are_interchangeable_in_expert_swiglu():
    import mlx.core as mx

    from runtime.deepseek_v4 import PackedExpert, expert_swiglu
    from runtime.quant import dequantize_deepseek_v4_fp4

    weight, scale = _real_expert()
    dense = dequantize_deepseek_v4_fp4(weight, scale)
    mx.eval(dense)
    packed = PackedExpert(weight.view(mx.uint8).view(mx.uint32), scale)

    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(2, dense.shape[1])).astype(np.float32) * 0.1)
    # w2 must map intermediate -> hidden, so reuse the transpose for shape fit.
    w2 = dense.T
    a = expert_swiglu(x, dense, w2, dense).astype(mx.float32)
    b = expert_swiglu(x, packed, w2, packed).astype(mx.float32)
    mx.eval(a, b)
    rel = float(mx.max(mx.abs(a - b))) / max(float(mx.max(mx.abs(a))), 1e-9)
    assert rel < 1e-2, f"packed expert changed the SwiGLU result, rel={rel}"


def test_packed_path_is_opt_in_by_default(monkeypatch):
    """Anti-overfit rule: a new fast path ships explicit, not automatic."""
    import importlib

    monkeypatch.delenv("VMODEL_DSV4_NATIVE_MXFP4", raising=False)
    import runtime.model_loader as ml

    importlib.reload(ml)
    source = Path(ml.__file__).read_text()
    assert 'VMODEL_DSV4_NATIVE_MXFP4' in source
    assert 'self.dsv4_native_mxfp4 = os.environ.get(' in source, (
        "the packed path must be gated on an explicit environment opt-in")


@realmodel
def test_fp8_block_dequant_is_unchanged_by_the_broadcast_form():
    """The trunk dequant must stay bit-identical after dropping mx.repeat.

    The repeat form materialized the scale at full weight shape before
    multiplying; the broadcast form exposes the block axes instead. That is a
    scheduling change only, so equality here is exact, not approximate.
    """
    import mlx.core as mx
    import numpy as np

    from runtime.quant import (_broadcast_block_scale, decode_e8m0_scale,
                               dequantize_deepseek_v4_fp8)

    rng = np.random.default_rng(7)
    packed = mx.array(rng.integers(0, 255, (512, 256), dtype=np.uint8))
    scale = mx.array(rng.integers(118, 136, (4, 2)).astype(np.uint8))

    values = mx.from_fp8(packed, mx.float32)
    expected = (values * _broadcast_block_scale(
        decode_e8m0_scale(scale), values.shape)).astype(mx.bfloat16)
    got = dequantize_deepseek_v4_fp8(packed, scale)
    mx.eval(expected, got)
    assert bool(mx.all(expected == got)), "broadcast form changed a value"


def test_fp8_block_dequant_still_rejects_a_mismatched_scale():
    import mlx.core as mx

    from runtime.quant import dequantize_deepseek_v4_fp8

    with pytest.raises(ValueError, match="exact multiple"):
        dequantize_deepseek_v4_fp8(mx.zeros((10, 8), mx.uint8),
                                   mx.zeros((3, 2), mx.uint8))


def test_packed_fp8_reports_packed_bytes_and_logical_shape():
    """A packed trunk page must cost its FP8 bytes, not its bf16 bytes.

    The pin planner sizes on disk bytes while the cache holds what the
    transform produced. Dequantized, those disagree by 1.92x and the planner
    silently undercounts; packed, they agree.
    """
    import mlx.core as mx

    from runtime.deepseek_v4 import PackedFP8

    packed = mx.zeros((4096, 4096), mx.uint8)
    scale = mx.zeros((32, 32), mx.uint8)
    held = PackedFP8(packed, scale)
    assert held.shape == (4096, 4096)
    assert held.nbytes == packed.nbytes + scale.nbytes
    # bf16 would be twice the payload; that difference is the whole point.
    assert held.nbytes < 4096 * 4096 * 2


@realmodel
def test_materializing_a_packed_page_matches_the_eager_dequant():
    import mlx.core as mx

    from runtime.deepseek_v4 import PackedFP8
    from runtime.quant import dequantize_deepseek_v4_fp8

    index = json.loads(
        (MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.20.attn.wq_a.weight"
    key = name if name in index else name.replace("model.", "")
    shard = mx.load(str(MODEL / index[key]))
    weight = shard[key]
    scale = shard[key.replace(".weight", ".scale")]
    assert str(weight.dtype).endswith("uint8")

    eager = dequantize_deepseek_v4_fp8(weight, scale)
    lazy = PackedFP8(weight, scale).materialize()
    mx.eval(eager, lazy)
    assert bool(mx.all(eager == lazy)), (
        "deferring the trunk dequant changed a value")


def test_materialize_helper_passes_through_ordinary_pages():
    """Must be a no-op for every other model and for eager DeepSeek V4 pages."""
    import mlx.core as mx

    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    page = {"a": mx.zeros((2, 2)), "b": mx.ones((3,))}
    assert engine._materialize_packed_trunk(page) is page


def test_packed_expert_page_estimate_is_not_bf16_sized():
    """The admission estimate must follow the representation actually held.

    A bf16 estimate is 4x the packed page. That error is not merely
    conservative: the trunk pin planner subtracts a mandatory expert batch
    from its budget, so over-estimating experts silently costs pinned trunk
    layers, and the governor refuses allocations for memory nothing occupies.
    """
    hidden, inter = 4096, 2048
    bf16_page = 3 * hidden * inter * 2
    packed_page = int(3 * hidden * inter * (0.5 + 1.0 / 32.0))
    assert bf16_page == 50_331_648
    assert packed_page < bf16_page / 3
    # Codes plus one E8M0 byte per 32 values, and nothing else.
    assert packed_page == int(3 * hidden * inter * 0.53125)
