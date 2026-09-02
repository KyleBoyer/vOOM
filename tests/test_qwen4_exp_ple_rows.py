from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
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


def _overlay_fixture(base: Path, root: Path, *, download_shard: bool) -> Path:
    root.mkdir()
    shard_name = "model.safetensors"
    linked_names = ["config.json", "model.safetensors.index.json"]
    downloaded_names = [shard_name] if download_shard else []
    if not download_shard:
        linked_names.append(shard_name)
    for name in linked_names:
        (root / name).symlink_to(base / name)
    for name in downloaded_names:
        shutil.copyfile(base / name, root / name)

    def record(name: str) -> dict:
        source = base / name
        return {
            "path": name,
            "size": source.stat().st_size,
            "hash_kind": "sha256",
            "hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    base_info = {
        "repo": "Qwen/base",
        "revision": "a" * 40,
        "directory": str(base.resolve()),
    }
    candidate_info = {
        "repo": "Example/abliterated",
        "revision": "b" * 40,
    }
    downloads = [record(name) for name in downloaded_names]
    links = [record(name) for name in linked_names]
    plan = {
        "schema": "voom.hf-checkpoint-overlay-plan.v1",
        "base": base_info,
        "candidate": candidate_info,
        "destination": str(root.resolve()),
        "files": {
            "download": downloads,
            "link": links,
            "safetensor_shards": 1,
            "changed_safetensor_shards": int(download_shard),
        },
    }
    plan_path = root / ".voom-overlay-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    receipt = {
        "schema": "voom.hf-checkpoint-overlay-receipt.v1",
        "status": "verified",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "base": base_info,
        "candidate": candidate_info,
        "destination": str(root.resolve()),
        "downloaded_files": len(downloads),
        "linked_files": len(links),
        "verified_download_bytes": sum(item["size"] for item in downloads),
        "verified_link_bytes": sum(item["size"] for item in links),
        "config_equal": True,
        "tensor_to_shard_map_equal": True,
    }
    (root / "voom.overlay.receipt.json").write_text(json.dumps(receipt))
    return root


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


def test_parallel_exact_rows_preserve_bits_order_and_extent_accounting(tmp_path):
    root, _ = _fixture(tmp_path)
    requested = np.array([[59, 0, 30, 16], [1, 31, 15, 14]])
    with Qwen4ExpPLERowStore(root, row_cache=0) as serial:
        expected = serial.read_rows(requested)
        serial_stats = serial.telemetry()
    with Qwen4ExpPLERowStore(
            root, row_cache=0, read_workers=4) as parallel:
        actual = parallel.read_rows(requested)
        parallel_stats = parallel.telemetry()
    np.testing.assert_array_equal(actual, expected)
    assert parallel_stats["bytes_read"] == serial_stats["bytes_read"]
    assert parallel_stats["read_extents"] == serial_stats["read_extents"]
    assert parallel_stats["read_workers"] == 4
    assert parallel_stats["parallel_read_calls"] == 1
    assert parallel_stats["read_microseconds"] >= 0


def test_missing_release_witness_and_bad_rows_fail_closed(tmp_path):
    root, _ = _fixture(tmp_path)
    tree = next((root / ".cache" / "huggingface" / "trees").iterdir())
    tree.unlink()
    with pytest.raises(ValueError, match="release witness"):
        Qwen4ExpPLERowStore(root)
    with Qwen4ExpPLERowStore(root, require_release_hash=False) as store:
        with pytest.raises(IndexError, match="outside"):
            store.read_rows([[store.layout.padded_vocab_size]])


@pytest.mark.parametrize("download_shard", [False, True])
def test_finalized_overlay_inherits_pinned_candidate_witness(
        tmp_path, download_shard):
    base, _ = _fixture(tmp_path)
    overlay = _overlay_fixture(
        base, tmp_path / f"overlay-{int(download_shard)}",
        download_shard=download_shard)
    with Qwen4ExpPLERowStore(overlay, row_cache=0) as store:
        rows = store.read_rows([[0, 15, 59]])
        assert rows.shape == (1, 3, 2)
        assert store.identity.verified_release_hash
        assert store.identity.revision == "b" * 40

    # A receipt cannot bless a subsequently edited plan.
    plan_path = overlay / ".voom-overlay-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["candidate"]["repo"] = "tampered/repo"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="release witness"):
        Qwen4ExpPLERowStore(overlay)


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
