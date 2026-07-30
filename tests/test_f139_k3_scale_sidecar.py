from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from formats.kimi_k3_scale_sidecar import (
    CURRENT,
    HEADER,
    KimiK3ScaleSidecar,
    PROJECTIONS,
    assemble_decode_batch,
    build_scale_sidecar,
    pack_scale,
    unpack_scale,
)


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


def _fixture(model_dir: Path, *, experts: int = 3) -> dict[tuple, bytes]:
    tensors = {}
    expected = {}
    for expert in range(experts):
        prefix = (
            "language_model.model.layers.1.block_sparse_moe.experts."
            f"{expert}"
        )
        for projection_index, projection in enumerate(PROJECTIONS):
            base = 108 + expert + projection_index
            span = (2, 3, 7)[projection_index]
            values = (
                base + np.arange(16, dtype=np.uint8) % (span + 1)
            ).reshape(4, 4)
            name = f"{prefix}.{projection}.weight_scale"
            tensors[name] = values
            expected[(1, expert, projection)] = values.tobytes()
    _write_safetensors(model_dir / "model.safetensors", tensors)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "hidden_size": 8,
                "num_hidden_layers": 2,
            }
        )
    )
    return expected


@pytest.mark.parametrize(
    ("values", "expected_width"),
    [
        ([120] * 17, 0),
        ([120, 121] * 9, 1),
        ([120, 123] * 9, 2),
        ([110, 125] * 9, 4),
        ([0, 255] * 9, 8),
    ],
)
def test_scale_fixed_width_round_trip(values, expected_width):
    raw = bytes(values)
    base, width, packed = pack_scale(raw)
    assert width == expected_width
    assert unpack_scale(
        packed, base=base, bits=width, count=len(raw)
    ) == raw


def test_sidecar_generation_reads_arbitrary_expert_order_exactly(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    expected = _fixture(model_dir)

    report = build_scale_sidecar(
        model_dir,
        sidecar_root,
        layers=[1],
        enforce_external=False,
        min_free_after_bytes=0,
    )
    assert report["ratio_raw_over_encoded"] > 1
    generation = (sidecar_root / CURRENT).read_text().strip()
    assert generation == report["generation"]

    reader = KimiK3ScaleSidecar(model_dir, sidecar_root)
    assert reader.has_layer(1)
    assert not reader.has_layer(0)
    records, physical_bytes = reader.read_records(1, [2, 0])
    assert [record.expert for record in records] == [2, 0]
    assert physical_bytes > sum(len(record.payload) for record in records)

    for record in records:
        cursor = 0
        for projection, base, width in zip(
            PROJECTIONS, record.bases, record.widths, strict=True
        ):
            count = 16
            encoded_bytes = (count * width + 7) // 8
            payload = record.payload[cursor : cursor + encoded_bytes]
            cursor += encoded_bytes
            assert (
                unpack_scale(
                    payload, base=base, bits=width, count=count
                )
                == expected[(1, record.expert, projection)]
            )
        assert cursor == len(record.payload)

    batch = assemble_decode_batch(records)
    first = HEADER.unpack_from(batch, 0)
    second = HEADER.unpack_from(batch, HEADER.size)
    assert first[0] == len(records) * HEADER.size
    assert second[0] == first[0] + len(records[0].payload)


def test_sidecar_record_crc_rejects_corruption(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    _fixture(model_dir)
    build_scale_sidecar(
        model_dir,
        sidecar_root,
        layers=[1],
        enforce_external=False,
        min_free_after_bytes=0,
    )
    reader = KimiK3ScaleSidecar(model_dir, sidecar_root)
    manifest = reader.manifest["layers"]["1"]
    path = reader.generation_dir / manifest["file"]
    with path.open("r+b") as handle:
        handle.seek(int(manifest["header_bytes"]))
        byte = handle.read(1)
        handle.seek(int(manifest["header_bytes"]))
        handle.write(bytes([byte[0] ^ 1]))

    with pytest.raises(IOError, match="CRC mismatch"):
        reader.read_records(1, [0])


def test_sidecar_rejects_different_checkpoint_fingerprint(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    _fixture(model_dir)
    build_scale_sidecar(
        model_dir,
        sidecar_root,
        layers=[1],
        enforce_external=False,
        min_free_after_bytes=0,
    )
    config = json.loads((model_dir / "config.json").read_text())
    config["hidden_size"] = 16
    (model_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="fingerprint"):
        KimiK3ScaleSidecar(model_dir, sidecar_root)


def test_incomplete_expert_is_never_published(tmp_path):
    model_dir = tmp_path / "model"
    sidecar_root = tmp_path / "sidecar"
    model_dir.mkdir()
    _fixture(model_dir)
    path = model_dir / "model.safetensors"
    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    missing = next(name for name in header if ".experts.2.w2." in name)
    del header[missing]
    header_raw = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode()
    header_raw += b" " * ((-len(header_raw)) % 8)
    # Offsets remain valid because the body is unchanged.
    body = raw[8 + header_length :]
    path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + body)

    with pytest.raises(ValueError, match="expected"):
        build_scale_sidecar(
            model_dir,
            sidecar_root,
            layers=[1],
            enforce_external=False,
            min_free_after_bytes=0,
        )
    assert not (sidecar_root / CURRENT).exists()
