"""formats.gguf_reader.GGUFFile: hand-rolled GGUF container parser.

Validates the parser's header/metadata/tensor-info reading and its
`.load()` dequantization end-to-end against:
1. The `gguf` PyPI package's own reader (pip-installed for verification
   only, never imported by the runtime) applied to the same real
   downloaded VibeThinker-3B GGUF file -- cross-checks tensor names,
   shapes, ggml types, and metadata values independently of this parser's
   own byte-offset arithmetic.
2. `runtime.quant`'s already-oracle-verified dequant functions (see
   tests/test_gguf_quant_oracle.py), by construction (`.load()` calls them
   directly) -- this test's job is to prove the RIGHT bytes reach them.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from formats.gguf_reader import GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGUFFile

ROOT = Path(__file__).resolve().parent.parent
GGUF_PATH = ROOT / "models" / "VibeThinker-3B-GGUF-base" / "VibeThinker-3B.Q4_K_M.gguf"
_GGUF_AVAILABLE = GGUF_PATH.exists()
_gguf_skip = pytest.mark.skipif(not _GGUF_AVAILABLE, reason="VibeThinker-3B GGUF file not available locally")

try:
    import gguf as gguf_pkg
    _GGUF_PKG_AVAILABLE = True
except ImportError:
    _GGUF_PKG_AVAILABLE = False
_gguf_pkg_skip = pytest.mark.skipif(not _GGUF_PKG_AVAILABLE, reason="gguf package not installed")


@_gguf_skip
@_gguf_pkg_skip
def test_header_and_tensor_info_matches_real_gguf_reader():
    mine = GGUFFile(GGUF_PATH)
    ref = gguf_pkg.GGUFReader(str(GGUF_PATH))

    ref_tensors = {t.name: t for t in ref.tensors}
    assert set(mine.tensors.keys()) == set(ref_tensors.keys())
    assert len(mine.tensors) > 0

    for name, ref_t in ref_tensors.items():
        mine_t = mine.tensors[name]
        ref_shape = tuple(int(d) for d in reversed(ref_t.shape))
        assert mine_t.shape == ref_shape, f"{name}: shape {mine_t.shape} != ref {ref_shape}"
        assert mine_t.ggml_type == int(ref_t.tensor_type), f"{name}: type mismatch"

    ref_field_val = ref.fields["general.architecture"].parts[-1].tobytes().decode("utf-8")
    assert mine.metadata["general.architecture"] == ref_field_val


@_gguf_skip
@_gguf_pkg_skip
def test_raw_bytes_match_real_gguf_reader_for_each_dtype():
    mine = GGUFFile(GGUF_PATH)
    ref = gguf_pkg.GGUFReader(str(GGUF_PATH))
    ref_tensors = {t.name: t for t in ref.tensors}

    by_type = {}
    for name, info in mine.tensors.items():
        by_type.setdefault(info.ggml_type, name)
    assert GGML_TYPE_Q4_K in by_type and GGML_TYPE_Q6_K in by_type and GGML_TYPE_F32 in by_type

    for ggml_type, name in by_type.items():
        mine_bytes = mine.raw_bytes(name)
        ref_bytes = ref_tensors[name].data.tobytes()
        assert mine_bytes == ref_bytes, f"{name} (type {ggml_type}): raw byte mismatch"


@_gguf_skip
def test_load_f32_tensor_is_finite_and_correct_shape():
    mine = GGUFFile(GGUF_PATH)
    name = next(n for n, info in mine.tensors.items() if info.ggml_type == GGML_TYPE_F32)
    arr = mine.load(name, out_dtype=mx.float32)
    mx.eval(arr)
    assert arr.shape == mine.tensors[name].shape
    assert not bool(mx.any(mx.isnan(arr)).item())


@_gguf_skip
def test_load_q4_k_and_q6_k_tensors_are_finite_and_correct_shape():
    mine = GGUFFile(GGUF_PATH)
    for wanted_type in (GGML_TYPE_Q4_K, GGML_TYPE_Q6_K):
        name = next(n for n, info in mine.tensors.items() if info.ggml_type == wanted_type)
        arr = mine.load(name, out_dtype=mx.float32)
        mx.eval(arr)
        assert arr.shape == mine.tensors[name].shape
        assert not bool(mx.any(mx.isnan(arr)).item())


@_gguf_skip
@_gguf_pkg_skip
def test_load_matches_gguf_package_dequantize_on_a_real_tensor():
    mine = GGUFFile(GGUF_PATH)
    name = next(n for n, info in mine.tensors.items() if info.ggml_type == GGML_TYPE_Q4_K)
    mine_arr = np.array(mine.load(name, out_dtype=mx.float32))

    ref = gguf_pkg.GGUFReader(str(GGUF_PATH))
    ref_t = next(t for t in ref.tensors if t.name == name)
    ref_arr = gguf_pkg.dequantize(ref_t.data, ref_t.tensor_type)

    assert mine_arr.shape == ref_arr.shape
    assert np.allclose(mine_arr, ref_arr, rtol=1e-4, atol=1e-4)
