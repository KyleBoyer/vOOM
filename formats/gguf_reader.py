"""Hand-rolled GGUF container parser: header/metadata/tensor-info + on-demand
mmap'd raw tensor bytes. No new runtime dependency -- the `gguf` PyPI package
was pip-installed only as a one-time research/oracle tool
(tests/test_gguf_quant_oracle.py), matching this project's established
convention of hand-rolling binary-format parsers rather than depending on a
library for the runtime path (see formats/kimi_k3_fast_tier.py's own
safetensors header parsing).

`mx.load()` also claims GGUF support, but it eagerly materializes the whole
file into dequantized arrays on open (observed directly: >120s just to open
a 1.9GB file) -- the opposite of WeightStore's lazy, per-tensor, mmap-backed
access model that the rest of this runtime relies on for checkpoints far
larger than resident memory. This parser instead mmaps the file once and
reads only the bytes a given tensor's fetch actually needs.

Real GGUF binary layout (ggml-org/gguf spec, cross-checked against the real
ggml-org/llama.cpp gguf-py `constants.py` fetched 2026-07-28):
    magic              : 4 bytes, b"GGUF"
    version            : uint32
    tensor_count       : uint64
    metadata_kv_count  : uint64
    metadata_kv * metadata_kv_count
    tensor_info * tensor_count
    <padding to `general.alignment` (default 32)>
    tensor data section (each tensor's raw bytes at its declared offset,
    relative to the start of this section)

GGUF stores each tensor's dimensions in ggml's `ne` order, which is the
REVERSE of PyTorch's `(out_features, ..., in_features)` convention --
verified 2026-07-27 against Qwen2's real `ffn_gate.weight`/`ffn_down.weight`
shapes. `GGUFTensorInfo.shape` below is already un-reversed to PyTorch
convention.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import mlx.core as mx

from runtime.quant import dequantize_gguf_q4_k, dequantize_gguf_q6_k

# GGUFValueType (ggml-org/gguf spec).
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_STRUCT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<?",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}

# GGMLQuantizationType values this project's target GGUF checkpoints
# actually use (VibeThinker-3B's Q4_K_M file uses exactly these three) --
# not the full ggml type table, which this parser doesn't need.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

# (block_size, block_nbytes) -- from the real ggml-org/gguf GGML_QUANT_SIZES
# table (constants.py, fetched 2026-07-28), restricted to the types above.
_QUANT_SIZES = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_Q6_K: (256, 210),
}


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    shape: tuple[int, ...]  # PyTorch convention (already un-reversed)
    ggml_type: int
    offset: int  # bytes, relative to the start of the tensor-data section


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError(f"GGUF file truncated: wanted {n} bytes, got {len(buf)}")
    return buf


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, length).decode("utf-8")


def _read_value(f: BinaryIO, value_type: int) -> Any:
    if value_type == _T_STRING:
        return _read_string(f)
    if value_type == _T_ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        return [_read_value(f, elem_type) for _ in range(count)]
    fmt = _SCALAR_STRUCT[value_type]
    (val,) = struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))
    return val


def _tensor_nbytes(info: GGUFTensorInfo) -> int:
    if info.ggml_type not in _QUANT_SIZES:
        raise ValueError(
            f"tensor {info.name!r} uses ggml type {info.ggml_type}, which this "
            f"hand-rolled parser does not implement (only F32/F16/Q4_K/Q6_K -- "
            f"the types VibeThinker-3B's Q4_K_M GGUF actually uses)")
    block_size, block_nbytes = _QUANT_SIZES[info.ggml_type]
    n_elements = 1
    for d in info.shape:
        n_elements *= d
    if n_elements % block_size:
        raise ValueError(
            f"tensor {info.name!r} has {n_elements} elements, not divisible by "
            f"block size {block_size} for ggml type {info.ggml_type}")
    return (n_elements // block_size) * block_nbytes


def _parse_header(f: BinaryIO) -> tuple[dict[str, Any], dict[str, GGUFTensorInfo], int]:
    magic = _read_exact(f, 4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic {magic!r})")
    (version,) = struct.unpack("<I", _read_exact(f, 4))
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version} (only 2, 3 verified)")
    (tensor_count,) = struct.unpack("<Q", _read_exact(f, 8))
    (metadata_kv_count,) = struct.unpack("<Q", _read_exact(f, 8))

    metadata: dict[str, Any] = {}
    for _ in range(metadata_kv_count):
        key = _read_string(f)
        (value_type,) = struct.unpack("<I", _read_exact(f, 4))
        metadata[key] = _read_value(f, value_type)

    tensors: dict[str, GGUFTensorInfo] = {}
    for _ in range(tensor_count):
        name = _read_string(f)
        (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
        ne = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
        (ggml_type,) = struct.unpack("<I", _read_exact(f, 4))
        (offset,) = struct.unpack("<Q", _read_exact(f, 8))
        # GGUF's `ne` is ggml's reversed dimension order; un-reverse to the
        # PyTorch (out_features, ..., in_features) convention used elsewhere
        # in this codebase.
        shape = tuple(reversed(ne))
        tensors[name] = GGUFTensorInfo(name=name, shape=shape, ggml_type=ggml_type, offset=offset)

    alignment = int(metadata.get("general.alignment", 32))
    data_start = f.tell()
    if data_start % alignment:
        data_start += alignment - (data_start % alignment)
    return metadata, tensors, data_start


class GGUFFile:
    """Parses a GGUF file's header once, then serves individual tensors'
    raw bytes (and, via `.load()`, dequantized mx.arrays) from an mmap --
    the file is never read into memory in full."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            self.metadata, self.tensors, self._data_start = _parse_header(f)
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None

    def _ensure_open(self) -> mmap.mmap:
        if self._mmap is None:
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def raw_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        nbytes = _tensor_nbytes(info)
        mm = self._ensure_open()
        start = self._data_start + info.offset
        return mm[start:start + nbytes]

    def load(self, name: str, out_dtype=mx.bfloat16) -> mx.array:
        """Read one tensor's raw bytes and return it dequantized to
        `out_dtype`, already in PyTorch-convention shape."""
        info = self.tensors[name]
        raw = self.raw_bytes(name)
        if info.ggml_type == GGML_TYPE_F32:
            arr = mx.array(memoryview(raw)).view(mx.float32).reshape(info.shape)
            return arr.astype(out_dtype)
        if info.ggml_type == GGML_TYPE_F16:
            arr = mx.array(memoryview(raw)).view(mx.float16).reshape(info.shape)
            return arr.astype(out_dtype)
        packed = mx.array(memoryview(raw)).astype(mx.uint8).reshape(
            info.shape[0], _tensor_nbytes(info) // info.shape[0])
        if info.ggml_type == GGML_TYPE_Q4_K:
            return dequantize_gguf_q4_k(packed, info.shape, out_dtype=out_dtype)
        if info.ggml_type == GGML_TYPE_Q6_K:
            return dequantize_gguf_q6_k(packed, info.shape, out_dtype=out_dtype)
        raise ValueError(f"tensor {name!r} uses unimplemented ggml type {info.ggml_type}")


# llama.cpp's own tensor-naming convention (ggml-org/llama.cpp gguf-py
# `tensor_mapping.py`, cross-checked against the real names present in the
# downloaded VibeThinker-3B GGUF: exactly 434 = 36*12 + 2 tensors, matching
# this table one-to-one with no leftovers) -- general across llama.cpp's
# plain-transformer exports, not Qwen2-specific. Maps to this codebase's
# HF-style canonical names (model.layers.N.self_attn.q_proj.weight, ...) so
# the ordinary dense runner (runtime/layer_runner.py) reads them unchanged.
_TOP_LEVEL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",  # absent when tie_word_embeddings
}
_PER_LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def canonicalize_llama_cpp_tensor_name(name: str) -> str | None:
    """llama.cpp GGUF tensor name -> this codebase's HF-style canonical
    name, or None if `name` isn't a recognized weight/bias tensor (e.g. a
    rope_freqs helper some exports carry, which this parser doesn't need)."""
    if name in _TOP_LEVEL_NAMES:
        return _TOP_LEVEL_NAMES[name]
    if not name.startswith("blk."):
        return None
    rest = name[len("blk."):]
    layer_str, _, tail = rest.partition(".")
    if not layer_str.isdigit():
        return None
    stem, _, suffix = tail.partition(".")  # suffix: "weight" or "bias"
    mapped = _PER_LAYER_NAMES.get(stem)
    if mapped is None or suffix not in ("weight", "bias"):
        return None
    return f"model.layers.{layer_str}.{mapped}.{suffix}"
