"""Cryptographic identity for raw safetensors checkpoints.

Raw HF indexes map tensors to shards but do not attest shard bodies. The
default runtime therefore uses size/mtime as a cheap local invalidation guard.
Proof deployments can create this manifest once, then require a complete SHA-256
verification before any weights or persisted KV are accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


MANIFEST_NAME = "voom.safetensors.sha256.json"
FORMAT = "voom-safetensors-sha256-v1"


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def safetensor_files(model_dir: str | Path) -> tuple[Path, ...]:
    root = Path(model_dir).resolve()
    index = root / "model.safetensors.index.json"
    if index.exists():
        value = json.loads(index.read_text())
        names = sorted(set(value.get("weight_map", {}).values()))
        if not names:
            raise ValueError("safetensors index has no weight shards")
        paths = []
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"unsafe safetensors shard name: {name!r}")
            paths.append(root / name)
    else:
        paths = sorted(root.glob("*.safetensors"))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("raw checkpoint has missing safetensors shards")
    return tuple(paths)


def build_manifest(model_dir: str | Path) -> dict:
    root = Path(model_dir).resolve()
    files = {}
    for path in safetensor_files(root):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {"format": FORMAT, "files": files}
    payload = _canonical(manifest)
    target = root / MANIFEST_NAME
    tmp = root / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return manifest


def verify_manifest(model_dir: str | Path) -> str:
    root = Path(model_dir).resolve()
    path = root / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(
            f"missing {MANIFEST_NAME}; build it with "
            f"`python -m formats.hash_safetensors {root}`") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {MANIFEST_NAME}: {error}") from error
    if (not isinstance(manifest, dict)
            or manifest.get("format") != FORMAT
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError(f"invalid {MANIFEST_NAME} format")
    expected_paths = safetensor_files(root)
    expected_names = {path.name for path in expected_paths}
    if set(manifest["files"]) != expected_names:
        raise ValueError("safetensors SHA manifest shard set does not match checkpoint")
    for shard in expected_paths:
        entry = manifest["files"].get(shard.name)
        if (not isinstance(entry, dict)
                or not isinstance(entry.get("bytes"), int)
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64):
            raise ValueError(f"invalid SHA manifest entry for {shard.name}")
        if shard.stat().st_size != entry["bytes"]:
            raise ValueError(f"safetensors size mismatch: {shard.name}")
        if _sha256(shard) != entry["sha256"]:
            raise ValueError(f"safetensors SHA-256 mismatch: {shard.name}")
    return hashlib.sha256(_canonical(manifest)).hexdigest()
