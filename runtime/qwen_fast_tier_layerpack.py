#!/usr/bin/env python3
"""Repack an exact raw Qwen fast tier into one consume-order file per layer.

The existing fast tier is balanced by layer but physically grouped by source
checkpoint shard.  Decode consequently reopens and reparses large multi-layer
safetensors containers for every layer.  FreeToken's FTW layout suggests the
opposite physical organization: bytes that one layer consumes together should
be contiguous.  This tool copies the already-selected tensor payloads
byte-for-byte into per-layer safetensors, writes a candidate manifest, and can
publish that manifest atomically.  It never deletes the source containers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
ACTIVE_MANIFEST = "fast_tier_manifest.json"
CANDIDATE_MANIFEST = "fast_tier_manifest.layerpacked.json"
BACKUP_MANIFEST = "fast_tier_manifest.pre-layerpack.json"
PROOF_FILE = "fast_tier_layerpack.proof.json"
GLOBAL_FAST_LIMIT = 90_000_000_000
MIN_INTERNAL_FREE = 10_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise IOError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > 64 * 1024 * 1024:
            raise IOError(f"unsafe safetensors header length: {length}")
        payload = source.read(length)
        if len(payload) != length:
            raise IOError(f"truncated safetensors JSON header: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("safetensors header must be an object")
    return value, 8 + length


def _prefix(entries: dict[str, dict]) -> bytes:
    payload = json.dumps(
        entries, separators=(",", ":"), sort_keys=True).encode()
    payload += b" " * (-len(payload) % 8)
    return struct.pack("<Q", len(payload)) + payload


def _tree_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size for item in path.rglob("*")
        if item.is_file() and not item.is_symlink())


def _group(name: str) -> str:
    match = LAYER_RE.match(name)
    return f"layer-{int(match.group(1)):03d}" if match else "top"


def build(
    fast_dir: Path, *, publish: bool = False,
    global_fast_limit: int = GLOBAL_FAST_LIMIT,
    min_internal_free: int = MIN_INTERNAL_FREE,
) -> dict:
    fast_dir = fast_dir.expanduser().resolve()
    active_path = fast_dir / ACTIVE_MANIFEST
    active_raw = active_path.read_bytes()
    active = json.loads(active_raw)
    if not isinstance(active, dict) or not active:
        raise ValueError("active raw fast-tier manifest must be nonempty")

    headers: dict[Path, tuple[dict, int]] = {}
    groups: dict[str, list[tuple[str, dict, Path, int]]] = defaultdict(list)
    payload_bytes = 0
    for name, entry in active.items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid fast-tier entry: {name}")
        source = fast_dir / str(entry.get("file", ""))
        if source.suffix != ".safetensors" or not source.is_file():
            raise ValueError(
                "layer packing currently requires safetensors sources: "
                f"{name}")
        if source not in headers:
            headers[source] = _header(source)
        header, data_start = headers[source]
        source_entry = header.get(name)
        if not isinstance(source_entry, dict):
            raise ValueError(f"source container is missing {name}")
        start, end = (int(value) for value in source_entry["data_offsets"])
        nbytes = end - start
        if (
            int(entry.get("offset", -1)) != start
            or int(entry.get("nbytes", -1)) != nbytes
            or str(entry.get("dtype", "")) != str(source_entry.get("dtype"))
            or tuple(entry.get("shape", ()))
            != tuple(source_entry.get("shape", ()))
        ):
            raise ValueError(f"source/manifest metadata mismatch: {name}")
        groups[_group(name)].append((name, entry, source, data_start))
        payload_bytes += nbytes

    fast_root = fast_dir.parent
    projected_tree = _tree_bytes(fast_root) + payload_bytes + 2_000_000
    free_before = shutil.disk_usage(fast_dir).free
    if projected_tree > int(global_fast_limit):
        raise RuntimeError("layer pack would exceed the 90 GB fast-tier cap")
    if free_before - payload_bytes < int(min_internal_free):
        raise RuntimeError("layer pack would leave less than 10 GB internal free")

    active_sha = hashlib.sha256(active_raw).hexdigest()
    generation = active_sha[:12]
    staging = Path(tempfile.mkdtemp(prefix="layerpack.", dir=fast_dir))
    candidate: dict[str, dict] = {}
    file_hashes: dict[str, str] = {}
    group_bytes: dict[str, int] = {}
    try:
        for group, values in sorted(groups.items()):
            values = sorted(values, key=lambda value: value[0])
            filename = f"layerpack-{generation}-{group}.safetensors"
            output = staging / filename
            output_header: dict[str, dict] = {}
            offset = 0
            for name, entry, _source, _data_start in values:
                nbytes = int(entry["nbytes"])
                output_header[name] = {
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "data_offsets": [offset, offset + nbytes],
                }
                candidate[name] = {
                    **entry,
                    "file": filename,
                    "offset": offset,
                }
                offset += nbytes
            with output.open("wb") as destination:
                destination.write(_prefix(output_header))
                for name, entry, source, data_start in values:
                    source_offset = int(entry["offset"])
                    remaining = int(entry["nbytes"])
                    with source.open("rb") as source_file:
                        source_file.seek(data_start + source_offset)
                        while remaining:
                            chunk = source_file.read(min(remaining, 8 * 1024 * 1024))
                            if not chunk:
                                raise IOError(f"truncated tensor payload: {name}")
                            destination.write(chunk)
                            remaining -= len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            file_hashes[filename] = _sha256(output)
            group_bytes[group] = offset

        candidate_raw = json.dumps(
            candidate, separators=(",", ":"), sort_keys=True).encode()
        candidate_path = staging / CANDIDATE_MANIFEST
        candidate_path.write_bytes(candidate_raw)
        proof = {
            "schema": "voom.qwen-fast-tier-layerpack.v1",
            "source_manifest_sha256": active_sha,
            "candidate_manifest_sha256": hashlib.sha256(
                candidate_raw).hexdigest(),
            "payload_bytes": payload_bytes,
            "container_files": len(file_hashes),
            "file_sha256": file_hashes,
            "group_bytes": group_bytes,
        }
        proof_path = staging / PROOF_FILE
        proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
        with proof_path.open("rb") as value:
            os.fsync(value.fileno())

        for filename in file_hashes:
            destination = fast_dir / filename
            if destination.exists():
                if _sha256(destination) != file_hashes[filename]:
                    raise RuntimeError(
                        f"existing layer-pack container differs: {destination}")
                (staging / filename).unlink()
            else:
                os.replace(staging / filename, destination)
        os.replace(candidate_path, fast_dir / CANDIDATE_MANIFEST)
        os.replace(proof_path, fast_dir / PROOF_FILE)
        if publish:
            backup = fast_dir / BACKUP_MANIFEST
            if backup.exists() and backup.read_bytes() != active_raw:
                raise RuntimeError("existing layer-pack backup differs from active")
            if not backup.exists():
                backup.write_bytes(active_raw)
            temporary = active_path.with_suffix(active_path.suffix + ".tmp")
            temporary.write_bytes(candidate_raw)
            os.replace(temporary, active_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        **proof,
        "fast_dir": str(fast_dir),
        "published": bool(publish),
        "projected_global_fast_tier_bytes": projected_tree,
        "internal_free_after_bytes": shutil.disk_usage(fast_dir).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fast_dir", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = build(args.fast_dir, publish=args.publish)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
