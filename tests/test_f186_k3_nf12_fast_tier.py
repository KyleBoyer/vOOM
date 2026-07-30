from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _fake_generation(root: Path, payloads: list[bytes]) -> None:
    from formats.bf16_nf12_sidecar import CURRENT, SCHEMA

    generation = "gen-1-deadbeef"
    generation_dir = root / generation
    generation_dir.mkdir(parents=True)
    layers = {}
    for layer, payload in enumerate(payloads):
        path = generation_dir / f"layer-{layer:03d}.safetensors"
        path.write_bytes(payload)
        layers[str(layer)] = {
            "file": path.name,
            "file_bytes": len(payload),
            "storage_file_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema": SCHEMA,
        "layers": layers,
        "total_selected_raw_bytes": sum(map(len, payloads)),
        "total_encoded_bytes": sum(map(len, payloads)),
    }
    (generation_dir / "manifest.json").write_text(json.dumps(manifest))
    (root / CURRENT).write_text(generation + "\n")


def test_nf12_fast_tier_dry_run_is_write_free(tmp_path, monkeypatch):
    import formats.kimi_k3_nf12_fast_tier as module

    source = tmp_path / "source"
    _fake_generation(source, [b"a" * 17, b"b" * 23])
    fast = tmp_path / "fast"
    monkeypatch.setattr(module, "_is_internal_root", lambda _path: True)

    report = module.stage_nf12_fast_tier(
        source,
        fast,
        dry_run=True,
        max_bytes=1_000_000,
        min_free_bytes=0,
    )

    assert report["layers"] == 2
    assert report["encoded_payload_bytes"] == 40
    assert not fast.exists()


def test_nf12_fast_tier_copies_and_verifies_generation(tmp_path, monkeypatch):
    import formats.kimi_k3_nf12_fast_tier as module

    source = tmp_path / "source"
    _fake_generation(source, [b"a" * 17, b"b" * 23])
    fast = tmp_path / "fast"
    monkeypatch.setattr(module, "_is_internal_root", lambda _path: True)

    report = module.stage_nf12_fast_tier(
        source,
        fast,
        max_bytes=1_000_000,
        min_free_bytes=0,
    )

    target = fast / "Kimi-K3-NF12"
    generation = (target / "CURRENT").read_text().strip()
    assert (target / generation / "layer-000.safetensors").read_bytes() == (
        b"a" * 17
    )
    assert report["actual_global_fast_tier_bytes"] == (
        report["actual_storage_bytes"]
    )


def test_nf12_fast_tier_rejects_corrupt_source_before_publication(
    tmp_path, monkeypatch
):
    import formats.kimi_k3_nf12_fast_tier as module

    source = tmp_path / "source"
    _fake_generation(source, [b"valid"])
    generation = (source / "CURRENT").read_text().strip()
    (source / generation / "layer-000.safetensors").write_bytes(b"wrong")
    fast = tmp_path / "fast"
    monkeypatch.setattr(module, "_is_internal_root", lambda _path: True)

    with pytest.raises(ValueError, match="copied SHA-256"):
        module.stage_nf12_fast_tier(
            source,
            fast,
            max_bytes=1_000_000,
            min_free_bytes=0,
        )
    assert not (fast / "Kimi-K3-NF12").exists()
