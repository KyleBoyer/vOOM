"""Transactionally stage an exact Kimi K3 NF12 generation on the fast tier.

The NF12 builder intentionally writes generations to the large external
workspace. This module promotes an already verified immutable generation to
the internal device without rebuilding it, while enforcing the repository's
global fast-tier byte ceiling and free-space reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

from .bf16_nf12_sidecar import CURRENT, SCHEMA
from .kimi_k3_fast_tier import (
    MAX_INTERNAL_FAST_TIER_BYTES,
    MIN_INTERNAL_FREE_BYTES,
    _existing_parent,
    _is_internal_root,
    _tree_file_bytes,
)

_GENERATION_RE = re.compile(r"gen-[A-Za-z0-9-]+")


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_and_hash(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(8 * 1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest()


def _source_plan(source_root: Path) -> tuple[str, bytes, dict, list[dict]]:
    generation = (source_root / CURRENT).read_text().strip()
    if _GENERATION_RE.fullmatch(generation) is None:
        raise ValueError(f"invalid NF12 generation {generation!r}")
    generation_dir = source_root / generation
    manifest_path = generation_dir / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"unsupported NF12 schema {manifest.get('schema')!r}"
        )

    files: list[dict] = []
    seen: set[str] = set()
    encoded_sum = 0
    for layer, entry in sorted(
        manifest.get("layers", {}).items(), key=lambda item: int(item[0])
    ):
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in seen
        ):
            raise ValueError(f"layer {layer}: unsafe or duplicate NF12 file")
        seen.add(filename)
        source = generation_dir / filename
        expected_storage = int(entry["storage_file_bytes"])
        actual_storage = source.stat().st_size
        if actual_storage != expected_storage:
            raise ValueError(
                f"layer {layer}: NF12 file size {actual_storage} != "
                f"{expected_storage}"
            )
        expected_hash = entry.get("file_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"layer {layer}: invalid NF12 SHA-256")
        files.append(
            {
                "layer": int(layer),
                "name": filename,
                "source": source,
                "storage_bytes": expected_storage,
                "encoded_bytes": int(entry["file_bytes"]),
                "sha256": expected_hash,
            }
        )
        encoded_sum += int(entry["file_bytes"])
    if not files:
        raise ValueError("NF12 generation contains no layer files")
    if encoded_sum != int(manifest["total_encoded_bytes"]):
        raise ValueError(
            f"NF12 encoded-byte sum {encoded_sum} != manifest total "
            f"{manifest['total_encoded_bytes']}"
        )
    return generation, manifest_raw, manifest, files


def stage_nf12_fast_tier(
    source_root: str | Path,
    fast_root: str | Path,
    *,
    target_name: str = "Kimi-K3-NF12",
    dry_run: bool = False,
    max_bytes: int = MAX_INTERNAL_FAST_TIER_BYTES,
    min_free_bytes: int = MIN_INTERNAL_FREE_BYTES,
) -> dict:
    """Copy one immutable NF12 generation and publish it atomically."""
    source_root = Path(source_root).expanduser().resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    target_component = Path(target_name)
    if (
        not target_name
        or target_component.name != target_name
        or target_name in (".", "..")
    ):
        raise ValueError("target_name must be one safe path component")
    if not _is_internal_root(fast_root):
        raise ValueError("NF12 fast-tier target must be on the internal device")
    max_bytes = int(max_bytes)
    min_free_bytes = max(0, int(min_free_bytes))
    if max_bytes <= 0 or max_bytes > MAX_INTERNAL_FAST_TIER_BYTES:
        raise ValueError(
            "max_bytes must be positive and no greater than "
            f"{MAX_INTERNAL_FAST_TIER_BYTES}"
        )

    generation, manifest_raw, manifest, files = _source_plan(source_root)
    target = fast_root / target_name
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; refusing a non-transactional overwrite"
        )
    planned_bytes = (
        sum(item["storage_bytes"] for item in files)
        + len(manifest_raw)
        + len(generation.encode())
        + 1
    )
    other_fast_tier_bytes = _tree_file_bytes(fast_root)
    projected_global = other_fast_tier_bytes + planned_bytes
    if projected_global > max_bytes:
        raise ValueError(
            f"NF12 plan projects {projected_global} global fast-tier bytes, "
            f"above max_bytes={max_bytes}; no files were written"
        )
    free_before = shutil.disk_usage(_existing_parent(fast_root)).free
    projected_free = free_before - planned_bytes
    if projected_free < min_free_bytes:
        raise ValueError(
            f"NF12 plan would leave {projected_free} free bytes, below "
            f"min_free_bytes={min_free_bytes}; no files were written"
        )

    report = {
        "schema": "voom.kimi-k3-nf12-fast-tier-plan.v1",
        "source_root": str(source_root),
        "source_generation": generation,
        "target": str(target),
        "layers": len(files),
        "selected_raw_bytes": int(manifest["total_selected_raw_bytes"]),
        "encoded_payload_bytes": int(manifest["total_encoded_bytes"]),
        "planned_storage_bytes": planned_bytes,
        "other_fast_tier_bytes": other_fast_tier_bytes,
        "projected_global_fast_tier_bytes": projected_global,
        "max_bytes": max_bytes,
        "min_free_bytes": min_free_bytes,
        "projected_free_bytes": projected_free,
        "fits_budget": True,
    }
    if dry_run:
        return report

    fast_root.mkdir(parents=True, exist_ok=True)
    staging = fast_root / f".{target_name}.building-{uuid.uuid4().hex}"
    staging_generation = staging / generation
    staging_generation.mkdir(parents=True)
    try:
        copied = 0
        total_storage = sum(item["storage_bytes"] for item in files)
        for index, item in enumerate(files, start=1):
            destination = staging_generation / item["name"]
            actual_hash = _copy_and_hash(item["source"], destination)
            if actual_hash != item["sha256"]:
                raise ValueError(
                    f"layer {item['layer']}: copied SHA-256 {actual_hash} != "
                    f"{item['sha256']}"
                )
            copied += item["storage_bytes"]
            print(
                f"[{index}/{len(files)}] layer {item['layer']:03d}: "
                f"verified {copied / 1e9:.2f} / "
                f"{total_storage / 1e9:.2f} GB",
                file=sys.stderr,
                flush=True,
            )
        manifest_path = staging_generation / "manifest.json"
        with manifest_path.open("xb") as output:
            output.write(manifest_raw)
            output.flush()
            os.fsync(output.fileno())
        current_path = staging / CURRENT
        with current_path.open("xb") as output:
            output.write((generation + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        _fsync_dir(staging_generation)
        _fsync_dir(staging)

        staged_bytes = _tree_file_bytes(staging)
        global_bytes = other_fast_tier_bytes + staged_bytes
        actual_free = shutil.disk_usage(fast_root).free
        if global_bytes > max_bytes:
            raise RuntimeError(
                f"completed NF12 stage uses {global_bytes} global bytes, "
                f"above max_bytes={max_bytes}; refusing publication"
            )
        if actual_free < min_free_bytes:
            raise RuntimeError(
                f"completed NF12 stage leaves {actual_free} free bytes, below "
                f"min_free_bytes={min_free_bytes}; refusing publication"
            )
        os.replace(staging, target)
        _fsync_dir(fast_root)
        report["actual_storage_bytes"] = staged_bytes
        report["actual_global_fast_tier_bytes"] = global_bytes
        report["actual_free_bytes"] = actual_free
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--fast-root", type=Path, default="~/vmodel_fast_tier")
    parser.add_argument("--target-name", default="Kimi-K3-NF12")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-bytes", type=int, default=MAX_INTERNAL_FAST_TIER_BYTES
    )
    parser.add_argument(
        "--min-free-bytes", type=int, default=MIN_INTERNAL_FREE_BYTES
    )
    args = parser.parse_args()
    report = stage_nf12_fast_tier(
        args.source_root,
        args.fast_root,
        target_name=args.target_name,
        dry_run=args.dry_run,
        max_bytes=args.max_bytes,
        min_free_bytes=args.min_free_bytes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
