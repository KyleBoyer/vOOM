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
