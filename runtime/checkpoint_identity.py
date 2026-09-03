"""Fail-closed identities for mutable checkpoints and exact fast tiers.

Large Hub checkpoints cannot always be replaced by constructing a second full
tree.  A replacement therefore carries a durable in-progress marker and a
final receipt.  Runtime loading refuses the marker, and exact raw fast tiers
can bind themselves to the resulting checkpoint identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPLACEMENT_MARKER_NAME = ".voom-checkpoint-replacement-in-progress.json"
REPLACEMENT_RECEIPT_NAME = "voom.checkpoint.receipt.json"
OVERLAY_RECEIPT_NAME = "voom.overlay.receipt.json"
RAW_FAST_TIER_BINDING_NAME = "raw_fast_tier_binding.json"
RAW_FAST_TIER_BINDING_SCHEMA = "voom.raw-fast-tier-binding.v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def refuse_incomplete_checkpoint(model_dir: str | Path) -> None:
    """Refuse a tree whose atomic per-file replacement has not committed."""
    marker = Path(model_dir) / REPLACEMENT_MARKER_NAME
    if marker.exists():
        raise RuntimeError(
            "checkpoint replacement is incomplete; resume or audit it before "
            f"loading: {marker}"
        )


def checkpoint_release_revision(model_dir: str | Path) -> str:
    """Return the strongest locally attested Hub revision, when available."""
    directory = Path(model_dir)
    for name in (REPLACEMENT_RECEIPT_NAME, OVERLAY_RECEIPT_NAME):
        path = directory / name
        if not path.is_file():
            continue
        receipt = _json_object(path)
        candidate = receipt.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"checkpoint receipt lacks candidate identity: {path}")
        revision = str(candidate.get("revision", ""))
        if len(revision) != 40 or any(
                char not in "0123456789abcdef" for char in revision.lower()):
            raise ValueError(f"checkpoint receipt has invalid revision: {path}")
        return revision.lower()
    return ""


def has_checkpoint_receipt(model_dir: str | Path) -> bool:
    directory = Path(model_dir)
    return any(
        (directory / name).is_file()
        for name in (REPLACEMENT_RECEIPT_NAME, OVERLAY_RECEIPT_NAME)
    )


def checkpoint_identity(model_dir: str | Path) -> dict[str, Any]:
    """Cheap local identity that changes when any indexed shard is replaced.

    This is deliberately a startup-time reuse guard, not a full 700 GB content
    attestation.  Published Hub hashes in the replacement/overlay receipt are
    the cryptographic proof; shard size+mtime detects later local mutation.
    """
    directory = Path(model_dir).resolve()
    refuse_incomplete_checkpoint(directory)
    config_path = directory / "config.json"
    index_path = directory / "model.safetensors.index.json"
    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")
    shard_names = sorted(set(str(value) for value in weight_map.values()))
    witness = hashlib.sha256()
    total_bytes = 0
    for name in shard_names:
        path = Path(name)
        if path.name != name or path.is_absolute():
            raise ValueError(f"unsafe indexed shard path: {name!r}")
        stat = (directory / name).stat()
        total_bytes += int(stat.st_size)
        witness.update(name.encode())
        witness.update(str(int(stat.st_size)).encode())
        witness.update(str(int(stat.st_mtime_ns)).encode())
    receipt_hashes = {}
    for name in (REPLACEMENT_RECEIPT_NAME, OVERLAY_RECEIPT_NAME):
        path = directory / name
        if path.is_file():
            receipt_hashes[name] = _sha256_bytes(path.read_bytes())
    return {
        "model_name": directory.name,
        "config_sha256": _sha256_bytes(config_bytes),
        "index_sha256": _sha256_bytes(index_bytes),
        "release_revision": checkpoint_release_revision(directory),
        "shard_count": len(shard_names),
        "shard_bytes": total_bytes,
        "shard_stat_sha256": witness.hexdigest(),
        "receipt_sha256": receipt_hashes,
    }


def raw_fast_tier_binding(
    model_dir: str | Path, manifest_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema": RAW_FAST_TIER_BINDING_SCHEMA,
        "checkpoint": checkpoint_identity(model_dir),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


def validate_raw_fast_tier_binding(
    model_dir: str | Path, tier_dir: str | Path, manifest_bytes: bytes,
) -> dict[str, Any]:
    tier = Path(tier_dir)
    path = tier / RAW_FAST_TIER_BINDING_NAME
    try:
        binding = _json_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "attested checkpoint requires a readable raw fast-tier binding: "
            f"{path}"
        ) from error
    expected = raw_fast_tier_binding(model_dir, manifest_bytes)
    if binding != expected:
        raise ValueError(
            "raw fast-tier source identity mismatch: "
            f"{tier} is not bound to {Path(model_dir).resolve()}"
        )
    return binding
