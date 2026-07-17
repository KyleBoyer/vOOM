from __future__ import annotations

import json
import os

import pytest

from runtime.weight_integrity import (
    MANIFEST_NAME, build_manifest, verify_manifest,
)


def _checkpoint(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first-shard-body")
    second.write_bytes(b"second-shard-body")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": first.name, "b": second.name},
    }))
    return first, second


def test_raw_safetensor_manifest_attests_every_indexed_shard(tmp_path):
    _checkpoint(tmp_path)

    manifest = build_manifest(tmp_path)
    digest = verify_manifest(tmp_path)

    assert set(manifest["files"]) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert len(digest) == 64
    assert (tmp_path / MANIFEST_NAME).is_file()


def test_same_size_same_mtime_body_replacement_is_rejected(tmp_path):
    first, _second = _checkpoint(tmp_path)
    build_manifest(tmp_path)
    stat = first.stat()
    first.write_bytes(b"FIRST-SHARD-BODY")
    assert first.stat().st_size == stat.st_size
    os.utime(first, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest(tmp_path)


def test_missing_raw_manifest_has_actionable_builder_command(tmp_path):
    _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="formats.hash_safetensors"):
        verify_manifest(tmp_path)
