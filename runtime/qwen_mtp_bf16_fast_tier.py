#!/usr/bin/env python3
"""Build an exact split-disk fast tier for a released BF16 Qwen MTP sidecar.

The target checkpoint remains untouched. A subset of complete safetensors is
copied byte-for-byte into one internal-SSD container; WeightStore reads that
subset concurrently with the complementary tensors from the external source.
The manifest is committed last and binds both files to SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


SCHEMA = "voom.qwen-mtp-bf16-fast-tier.v1"
MANIFEST = "mtp-bf16-fast.manifest.json"
FAST_FILE = "mtp-bf16-fast.safetensors"
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


def _tree_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size for item in path.rglob("*")
        if item.is_file())


def _select_nearest(entries: list[dict], target_bytes: int) -> list[dict]:
    # Real Qwen MTP has only 15 tensors, so exhaustive subset selection gives
    # a better device-time balance than a fragile name heuristic.
    if len(entries) > 24:
        raise ValueError("refusing exponential selection for >24 tensors")
    best_mask = 0
    best_error = None
    sizes = [int(entry["nbytes"]) for entry in entries]
    for mask in range(1, 1 << len(entries)):
        total = sum(
            size for index, size in enumerate(sizes)
            if mask & (1 << index))
        error = abs(total - target_bytes)
        if best_error is None or error < best_error:
            best_mask, best_error = mask, error
    return [
        entry for index, entry in enumerate(entries)
        if best_mask & (1 << index)]


def build(
    model_dir: Path, fast_root: Path, *, fast_fraction: float = 0.56,
    global_fast_limit: int = GLOBAL_FAST_LIMIT,
    min_internal_free: int = MIN_INTERNAL_FREE,
) -> dict:
    model_dir = model_dir.expanduser().resolve()
    fast_root = fast_root.expanduser().resolve()
    if not 0 < fast_fraction < 1:
        raise ValueError("fast_fraction must be strictly between 0 and 1")
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())
    metadata = index.get("metadata", {})
    sidecar_name = metadata.get("mtplx_mtp_sidecar")
    expected_source_sha = metadata.get("mtplx_mtp_sidecar_sha256")
    if not (
        isinstance(sidecar_name, str)
        and Path(sidecar_name).name == sidecar_name
        and isinstance(expected_source_sha, str)
        and len(expected_source_sha) == 64
    ):
        raise ValueError("checkpoint omits a safe, hashed MTPLX MTP sidecar")
    source = model_dir / sidecar_name
    source_sha = _sha256(source)
    if source_sha != expected_source_sha:
        raise ValueError("released BF16 MTP sidecar SHA-256 mismatch")

    header, data_start = _header(source)
    entries = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not name.startswith("mtp.") or value.get("dtype") != "BF16":
            raise ValueError(f"non-released-BF16 tensor in MTP sidecar: {name}")
        start, end = (int(item) for item in value["data_offsets"])
        entries.append({
            "name": name,
            "dtype": value["dtype"],
            "shape": [int(item) for item in value["shape"]],
            "start": start,
            "end": end,
            "nbytes": end - start,
        })
    total_bytes = sum(entry["nbytes"] for entry in entries)
    selected = _select_nearest(entries, round(total_bytes * fast_fraction))
    selected_bytes = sum(entry["nbytes"] for entry in selected)

    fast_root.mkdir(parents=True, exist_ok=True)
    existing_tree_bytes = _tree_bytes(fast_root.parent)
    old_bytes = sum(
        (fast_root / name).stat().st_size
        for name in (FAST_FILE, MANIFEST)
        if (fast_root / name).is_file())
    projected_tree = existing_tree_bytes - old_bytes + selected_bytes + 1_000_000
    free_before = shutil.disk_usage(fast_root).free
    if projected_tree > global_fast_limit:
        raise RuntimeError("split MTP tier would exceed the 90 GB fast-tier cap")
    if free_before - selected_bytes < min_internal_free:
        raise RuntimeError("split MTP tier would leave less than 10 GB internal free")

    staging = Path(tempfile.mkdtemp(prefix="mtp-bf16-fast.", dir=fast_root))
    try:
        fast_path = staging / FAST_FILE
        out_header = {}
        offset = 0
        for entry in selected:
            out_header[entry["name"]] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [offset, offset + entry["nbytes"]],
            }
            offset += entry["nbytes"]
        raw_header = json.dumps(
            out_header, separators=(",", ":"), sort_keys=True).encode()
        raw_header += b" " * (-len(raw_header) % 8)
        with source.open("rb") as src, fast_path.open("wb") as dst:
            dst.write(struct.pack("<Q", len(raw_header)))
            dst.write(raw_header)
            for entry in selected:
                src.seek(data_start + entry["start"])
                remaining = entry["nbytes"]
                while remaining:
                    chunk = src.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise IOError(f"truncated source tensor {entry['name']}")
                    dst.write(chunk)
                    remaining -= len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        fast_sha = _sha256(fast_path)
        tensor_manifest = {
            entry["name"]: {
                "file": FAST_FILE,
                "offset": int(out_header[entry["name"]]["data_offsets"][0]),
                "nbytes": entry["nbytes"],
                "dtype": entry["dtype"],
                "shape": entry["shape"],
            }
            for entry in selected
        }
        manifest = {
            "schema": SCHEMA,
            "source_sidecar": sidecar_name,
            "source_size": source.stat().st_size,
            "source_sha256": source_sha,
            "fast_file": FAST_FILE,
            "fast_file_sha256": fast_sha,
            "fast_fraction": fast_fraction,
            "selected_bytes": selected_bytes,
            "remaining_bytes": total_bytes - selected_bytes,
            "tensors": tensor_manifest,
        }
        manifest_path = staging / MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with manifest_path.open("rb") as value:
            os.fsync(value.fileno())
        os.replace(fast_path, fast_root / FAST_FILE)
        os.replace(manifest_path, fast_root / MANIFEST)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        **manifest,
        "fast_root": str(fast_root),
        "projected_global_fast_tier_bytes": projected_tree,
        "internal_free_after_bytes": shutil.disk_usage(fast_root).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("fast_root", type=Path)
    parser.add_argument("--fast-fraction", type=float, default=0.56)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = build(
        args.model_dir, args.fast_root, fast_fraction=args.fast_fraction)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
