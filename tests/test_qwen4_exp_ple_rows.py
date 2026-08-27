from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from runtime.qwen4_exp_ple_rows import Qwen4ExpPLERowStore


def _write_safetensor(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    payload = bytearray()
    for name, value in tensors.items():
        raw = np.asarray(value, dtype=np.uint16).tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _fixture(tmp_path: Path) -> tuple[Path, int]:
    root = tmp_path / "qwen4-exp"
    root.mkdir()
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": 64,
            "hidden_size": 8,
            "eos_token_id": 63,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 11,
            "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 4,
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding")
    rows_per_part = 15
    row_width = 2
    tensors = {}
    weight_map = {}
    for part in range(4):
        name = f"{prefix}.shard_{part}.weight"
        start = part * rows_per_part * row_width
        tensors[name] = np.arange(
            start, start + rows_per_part * row_width,
            dtype=np.uint16,
        ).reshape(rows_per_part, row_width)
        weight_map[name] = "model.safetensors"
    shard = root / "model.safetensors"
    _write_safetensor(shard, tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": weight_map,
    }))

    revision = "a" * 40
    sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    cache = root / ".cache" / "huggingface"
    (cache / "trees").mkdir(parents=True)
    (cache / "download").mkdir()
    (cache / "trees" / f"{revision}.json").write_text(json.dumps({
        "format_version": 1,
        "files": {
            shard.name: {
                "lfs_sha256": sha,
                "lfs_size": shard.stat().st_size,
            },
        },
    }))
    (cache / "download" / f"{shard.name}.metadata").write_text(
        f"{revision}\n{sha}\n")
    return root, rows_per_part


def test_direct_rows_cross_parts_coalesce_and_cache(tmp_path):
    root, rows_per_part = _fixture(tmp_path)
    with Qwen4ExpPLERowStore(root, row_cache=8) as store:
        requested = np.array([[0, 1, 14, 15], [16, 30, 31, 59]])
        rows = store.read_rows(requested)
        assert rows.shape == (2, 4, 2)
        np.testing.assert_array_equal(
            rows,
            np.stack([
                np.array([row * 2, row * 2 + 1], dtype=np.uint16)
                for row in requested.reshape(-1)
            ]).reshape(2, 4, 2),
        )
        stats = store.telemetry()
        assert stats["source_verified_release_hash"] == 1
        assert stats["split_parts"] == 4
        assert stats["unique_shards"] == 1
        assert stats["unique_rows_read"] == 8
        assert stats["bytes_read"] == 8 * 4
        assert stats["read_extents"] == 4

        again = store.read_rows([[0, 1, rows_per_part, 59]])
        assert again.shape == (1, 4, 2)
        assert store.telemetry()["cache_hits"] == 4
        assert store.telemetry()["bytes_read"] == 8 * 4


def test_missing_release_witness_and_bad_rows_fail_closed(tmp_path):
    root, _ = _fixture(tmp_path)
    tree = next((root / ".cache" / "huggingface" / "trees").iterdir())
    tree.unlink()
    with pytest.raises(ValueError, match="release witness"):
        Qwen4ExpPLERowStore(root)
    with Qwen4ExpPLERowStore(root, require_release_hash=False) as store:
        with pytest.raises(IndexError, match="outside"):
            store.read_rows([[store.layout.padded_vocab_size]])


REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "Qwen3.8-Flash-Next"


@pytest.mark.skipif(
    not (REAL_MODEL / "model.safetensors.index.json").is_file(),
    reason="Qwen3.8-Flash-Next is not downloaded locally",
)
def test_real_flash_next_reads_only_requested_bf16_rows():
    with Qwen4ExpPLERowStore(REAL_MODEL, row_cache=64) as store:
        ids = store.layout.row_ids([17, 23, 29])
        rows = store.read_rows(ids)
        assert rows.shape == (3, 16, 160)
        assert rows.dtype == np.uint16
        assert store.identity.revision == (
            "f5d08274bafd880402bd16f5e3e6c514136ec06c")
        assert store.identity.verified_release_hash
        assert store.identity.split_parts == 128
        assert store.identity.unique_shards == 33
        stats = store.telemetry()
        assert stats["rows_requested"] == 48
        assert stats["unique_rows_read"] == 48
        assert stats["bytes_read"] == 48 * 320
