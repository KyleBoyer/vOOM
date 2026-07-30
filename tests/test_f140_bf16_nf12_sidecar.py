from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from formats.bf16_nf12_sidecar import (
    CURRENT,
    BF16NF12Sidecar,
    build_nf12_sidecar,
    pack_tensor,
    unpack_tensor,
)


def _write_bf16_safetensors(
    path: Path, tensors: dict[str, tuple[np.ndarray, tuple[int, ...]]]
) -> None:
    header = {}
    body = bytearray()
    for name, (bits, shape) in tensors.items():
        raw = np.ascontiguousarray(bits, dtype="<u2").tobytes()
        start = len(body)
        body.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [start, len(body)],
        }
    header_raw = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode()
    header_raw += b" " * ((-len(header_raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header_raw)) + header_raw + body
    )


def _fixture(model_dir: Path) -> tuple[bytes, bytes]:
    rng = np.random.default_rng(140)
    # High exponent nibble 7 is modal, with exact non-modal exceptions and a
    # non-block-aligned tail. Sign, low exponent, and mantissa remain arbitrary.
    compressible = rng.integers(0, 1 << 11, size=700, dtype=np.uint16)
    compressible |= np.uint16(7 << 11)
    compressible[17] = np.uint16(0x8001)
    compressible[511] = np.uint16(0x3F80)
    incompressible = np.arange(16, dtype=np.uint16).repeat(64) << 11
    incompressible |= rng.integers(
        0, 1 << 11, size=incompressible.size, dtype=np.uint16
    )
    tensors = {
        "model.layers.0.input_layernorm.weight": (
            compressible,
            (compressible.size,),
        ),
        "model.layers.0.self_attn.q_proj.weight": (
            incompressible,
            (32, 32),
        ),
    }
    _write_bf16_safetensors(model_dir / "model.safetensors", tensors)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return compressible.astype("<u2").tobytes(), incompressible.astype(
        "<u2"
    ).tobytes()


def test_nf12_fixed_stream_and_exception_patches_round_trip_exactly():
    rng = np.random.default_rng(140)
    bits = rng.integers(0, 1 << 11, size=700, dtype=np.uint16)
    bits |= np.uint16(7 << 11)
    bits[[0, 255, 256, 699]] = np.array(
        [0x0000, 0x2F80, 0x8001, 0xFFFF], dtype=np.uint16
    )
    raw = bits.astype("<u2").tobytes()

    headers, codes, patches, report = pack_tensor(raw)

    assert report["selected"]
    assert report["exception_count"] == 4
    assert report["encoded_bytes"] < report["raw_bytes"]
    assert unpack_tensor(
        headers,
        codes,
        patches,
        value_count=bits.size,
    ) == raw


def test_nf12_omits_a_tensor_that_would_expand():
    rng = np.random.default_rng(141)
    high = np.arange(16, dtype=np.uint16).repeat(64)
    bits = (high << 11) | rng.integers(
        0, 1 << 11, size=high.size, dtype=np.uint16
    )

    headers, codes, patches, report = pack_tensor(
        bits.astype("<u2").tobytes()
    )

    assert not report["selected"]
    assert headers == codes == patches == b""
    assert report["encoded_bytes"] == report["raw_bytes"]


def test_partial_generation_is_checkpoint_bound_and_omits_raw_fallback(
    tmp_path,
):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    compressible, _incompressible = _fixture(model_dir)

    report = build_nf12_sidecar(
        model_dir,
        sidecar_root,
        layers=[0],
        enforce_external=False,
        min_free_after_bytes=0,
    )

    assert report["ratio_raw_over_encoded"] > 1
    assert (sidecar_root / CURRENT).read_text().strip() == report[
        "generation"
    ]
    sidecar = BF16NF12Sidecar(model_dir, sidecar_root)
    assert sidecar.has_layer(0)
    assert sidecar.encoded_names(0) == {
        "model.layers.0.input_layernorm.weight"
    }
    tensor = sidecar.layer_entry(0)["tensors"][0]
    assert tensor["raw_bytes"] == len(compressible)
    assert tensor["encoded_bytes"] < tensor["raw_bytes"]

    config = json.loads((model_dir / "config.json").read_text())
    config["hidden_size"] = 64
    (model_dir / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fingerprint"):
        BF16NF12Sidecar(model_dir, sidecar_root)
