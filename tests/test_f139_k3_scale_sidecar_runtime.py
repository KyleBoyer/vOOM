from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from formats.kimi_k3_scale_sidecar import (
    PROJECTIONS,
    ScaleRecord,
    build_scale_sidecar,
    pack_scale,
)
from runtime.kimi_k3_scale_sidecar import decode_records
from runtime.model_loader import WeightStore
from runtime.quant import QTensor


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    body = bytearray()
    for name, tensor in tensors.items():
        raw = np.ascontiguousarray(tensor, dtype=np.uint8).tobytes()
        start = len(body)
        body.extend(raw)
        header[name] = {
            "dtype": "U8",
            "shape": list(tensor.shape),
            "data_offsets": [start, len(body)],
        }
    header_raw = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode()
    header_raw += b" " * ((-len(header_raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + body)


def _write_fixture(model_dir: Path) -> list[str]:
    rng = np.random.default_rng(139)
    tensors = {}
    logical_names = []
    for expert in range(2):
        prefix = f"model.layers.1.block_sparse_moe.experts.{expert}"
        for projection_index, projection in enumerate(("w1", "w2", "w3")):
            stem = f"{prefix}.{projection}"
            tensors[f"{stem}.weight_packed"] = rng.integers(
                0, 256, size=(8, 32), dtype=np.uint8
            )
            base = 116 + expert + projection_index
            tensors[f"{stem}.weight_scale"] = (
                base + np.arange(16, dtype=np.uint8).reshape(8, 2) % 4
            )
            logical_names.append(f"{stem}.weight")
    _write_safetensors(model_dir / "model.safetensors", tensors)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "routed_expert_hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_experts": 2,
                "num_experts_per_token": 2,
                "vocab_size": 8,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "mxfp4-pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "num_bits": 4,
                                "group_size": 32,
                                "scale_dtype": "torch.uint8",
                                "symmetric": True,
                                "type": "float",
                            }
                        }
                    },
                },
            }
        )
    )
    return logical_names


def test_weightstore_sidecar_is_byte_exact_and_reads_fewer_source_bytes(
    tmp_path,
):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    names = _write_fixture(model_dir)
    build_scale_sidecar(
        model_dir,
        sidecar_root,
        layers=[1],
        enforce_external=False,
        min_free_after_bytes=0,
    )

    control = WeightStore(model_dir, native_ct_mxfp4=True)
    candidate = WeightStore(
        model_dir,
        native_ct_mxfp4=True,
        kimi_k3_scale_sidecar_dir=sidecar_root,
    )
    control_values, _, control_bytes = control.fetch(names)
    candidate_values, _, candidate_bytes = candidate.fetch(names)

    for name in names:
        reference = control_values[name]
        value = candidate_values[name]
        assert isinstance(reference, QTensor)
        assert isinstance(value, QTensor)
        assert mx.array_equal(value.wq, reference.wq)
        assert mx.array_equal(value.scales, reference.scales)
    assert candidate_bytes < control_bytes

    read_bytes, output_bytes, decode_ns, calls = (
        candidate.k3_scale_sidecar_snapshot()
    )
    assert read_bytes > 0
    assert output_bytes == sum(
        value.scales.nbytes for value in candidate_values.values()
    )
    assert decode_ns > 0
    assert calls == 1


def test_weightstore_sidecar_requires_native_mxfp4(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    _write_fixture(model_dir)
    build_scale_sidecar(
        model_dir,
        sidecar_root,
        layers=[1],
        enforce_external=False,
        min_free_after_bytes=0,
    )
    with pytest.raises(ValueError, match="native_ct_mxfp4"):
        WeightStore(
            model_dir, kimi_k3_scale_sidecar_dir=sidecar_root
        )


def test_vectorized_decoder_covers_every_fixed_width_exactly():
    values = {
        "w1": bytes([120] * 32),
        "w3": bytes([120, 121] * 16),
        "w2": bytes([120, 123] * 16),
    }
    packed = [pack_scale(values[projection]) for projection in PROJECTIONS]
    record = ScaleRecord(
        expert=7,
        bases=tuple(item[0] for item in packed),
        widths=tuple(item[1] for item in packed),
        payload=b"".join(item[2] for item in packed),
    )
    decoded = decode_records(
        [record], {projection: (4, 8) for projection in PROJECTIONS}
    )
    for projection in PROJECTIONS:
        assert np.array(decoded[(7, projection)]).tobytes() == values[
            projection
        ]

    wide = {
        "w1": bytes(range(32)),
        "w3": bytes([0, 255] * 16),
        "w2": bytes([100] * 32),
    }
    packed = [pack_scale(wide[projection]) for projection in PROJECTIONS]
    record = ScaleRecord(
        expert=8,
        bases=tuple(item[0] for item in packed),
        widths=tuple(item[1] for item in packed),
        payload=b"".join(item[2] for item in packed),
    )
    decoded = decode_records(
        [record], {projection: (32,) for projection in PROJECTIONS}
    )
    for projection in PROJECTIONS:
        assert np.array(decoded[(8, projection)]).tobytes() == wide[
            projection
        ]
