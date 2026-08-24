import json
import struct

import pytest

from runtime.qwen_fast_tier_layerpack import (
    ACTIVE_MANIFEST,
    BACKUP_MANIFEST,
    CANDIDATE_MANIFEST,
    PROOF_FILE,
    _header,
    build,
)


def _write_safetensors(path, tensors):
    header = {}
    offset = 0
    for name, payload in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for payload in tensors.values():
            output.write(payload)
    return header


def _payload(path, name):
    header, data_start = _header(path)
    start, end = header[name]["data_offsets"]
    with path.open("rb") as source:
        source.seek(data_start + start)
        return source.read(end - start)


def test_layerpack_groups_exact_payloads_and_publishes_last(tmp_path):
    fast_dir = tmp_path / "fast" / "model"
    fast_dir.mkdir(parents=True)
    tensors_a = {
        "model.layers.0.a.weight": b"a" * 17,
        "model.layers.1.a.weight": b"b" * 11,
    }
    tensors_b = {
        "model.layers.0.b.weight": b"c" * 23,
        "model.layers.1.b.weight": b"d" * 13,
    }
    headers = {
        "a.safetensors": _write_safetensors(
            fast_dir / "a.safetensors", tensors_a),
        "b.safetensors": _write_safetensors(
            fast_dir / "b.safetensors", tensors_b),
    }
    manifest = {}
    for filename, tensors in (
            ("a.safetensors", tensors_a), ("b.safetensors", tensors_b)):
        for name, payload in tensors.items():
            entry = headers[filename][name]
            manifest[name] = {
                "file": filename,
                "offset": entry["data_offsets"][0],
                "nbytes": len(payload),
                "dtype": "U8",
                "shape": [len(payload)],
            }
    active_raw = json.dumps(manifest).encode()
    (fast_dir / ACTIVE_MANIFEST).write_bytes(active_raw)

    result = build(
        fast_dir, publish=True,
        global_fast_limit=1_000_000_000, min_internal_free=0)

    assert result["published"] is True
    assert result["container_files"] == 2
    assert (fast_dir / BACKUP_MANIFEST).read_bytes() == active_raw
    assert (fast_dir / CANDIDATE_MANIFEST).is_file()
    assert (fast_dir / PROOF_FILE).is_file()
    active = json.loads((fast_dir / ACTIVE_MANIFEST).read_text())
    assert len({entry["file"] for entry in active.values()}) == 2
    assert active["model.layers.0.a.weight"]["file"] == (
        active["model.layers.0.b.weight"]["file"])
    assert active["model.layers.1.a.weight"]["file"] == (
        active["model.layers.1.b.weight"]["file"])
    expected = {**tensors_a, **tensors_b}
    for name, entry in active.items():
        assert _payload(fast_dir / entry["file"], name) == expected[name]


def test_layerpack_rejects_manifest_source_mismatch(tmp_path):
    fast_dir = tmp_path / "fast" / "model"
    fast_dir.mkdir(parents=True)
    name = "model.layers.0.a.weight"
    _write_safetensors(fast_dir / "a.safetensors", {name: b"abc"})
    (fast_dir / ACTIVE_MANIFEST).write_text(json.dumps({
        name: {
            "file": "a.safetensors", "offset": 0, "nbytes": 4,
            "dtype": "U8", "shape": [3],
        },
    }))

    with pytest.raises(ValueError, match="metadata mismatch"):
        build(
            fast_dir, global_fast_limit=1_000_000_000,
            min_internal_free=0)
