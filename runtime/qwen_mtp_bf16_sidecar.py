#!/usr/bin/env python3
"""Build or validate a released-BF16 Qwen MTP sidecar.

This is deliberately a sidecar operation, not another whole-model conversion:
only top-level ``mtp.*`` tensors are copied from the pinned released source.
The quantized target's existing MTP entries are removed from its weight map and
an explicit ``mtplx_mtp_sidecar`` pointer is committed last.  Non-MTP mappings
are fingerprinted before and after, so the operation fails if it changes the
body contract.

The default paths/revision are the installed Huihui Qwen3.8 27B artifacts. Run
``plan`` first; ``build`` performs full pinned-shard SHA-256 verification and a
tensor-by-tensor source/sidecar equality check before atomically updating the
target index. ``validate`` independently repeats those checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "models" / "Huihui-Qwen3.8-27B-abliterated"
DEFAULT_TARGET = (
    ROOT / "models" / "Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4")
DEFAULT_REPOSITORY = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
DEFAULT_REVISION = "d42ca8978c5a66e92c3446d46e8adfe03ef692ff"
DEFAULT_SIDECAR = "mtp-bf16.safetensors"
DEFAULT_MANIFEST = "mtp-bf16.manifest.json"
INDEX_NAME = "model.safetensors.index.json"
SIDECAR_SCHEMA = "voom.qwen-mtp-bf16-sidecar.v1"
MANIFEST_SCHEMA = "voom.qwen-mtp-bf16-sidecar-manifest.v1"


def _canonical_json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_sibling_name(value: str, label: str, suffix: str) -> str:
    if not (
        isinstance(value, str)
        and value
        and Path(value).name == value
        and not Path(value).is_absolute()
        and value.endswith(suffix)
    ):
        raise ValueError(f"unsafe {label} filename: {value!r}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    """Durably replace one small metadata file in its own directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        # Some filesystems do not expose directory fsync. The same-dir
        # os.replace remains atomic even when that durability hint fails.
        pass


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(
                f"invalid safetensors header length {header_length}: {path}")
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    return header


def _safetensors_data_start(path: Path) -> int:
    """Return the first payload byte after a validated safetensors header."""
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
    if len(length_bytes) != 8:
        raise ValueError(f"truncated safetensors header: {path}")
    header_length = struct.unpack("<Q", length_bytes)[0]
    if header_length <= 0 or header_length > path.stat().st_size - 8:
        raise ValueError(
            f"invalid safetensors header length {header_length}: {path}")
    return 8 + header_length


def _tensor_specs(path: Path) -> dict[str, dict]:
    specs = {}
    for name, value in _safetensors_header(path).items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid safetensors tensor header {name}: {path}")
        offsets = value.get("data_offsets")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if not (
            isinstance(offsets, list) and len(offsets) == 2
            and all(isinstance(item, int) for item in offsets)
            and offsets[0] >= 0 and offsets[1] >= offsets[0]
            and isinstance(shape, list)
            and all(isinstance(item, int) and item >= 0 for item in shape)
            and isinstance(dtype, str)
        ):
            raise ValueError(f"invalid safetensors tensor header {name}: {path}")
        specs[name] = {
            "dtype": dtype,
            "shape": shape,
            "bytes": offsets[1] - offsets[0],
        }
    return specs


def _sidecar_metadata(path: Path) -> dict[str, str]:
    metadata = _safetensors_header(path).get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError(f"invalid safetensors metadata: {path}")
    return metadata


def _mapping_fingerprint(mapping: dict[str, str]) -> str:
    return _sha256_bytes(_canonical_json(dict(sorted(mapping.items()))))


def _require_revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source revision must be a 40-character lowercase SHA")
    return value


def inspect_sidecar_inputs(
    source: str | Path,
    target: str | Path,
    *,
    repository: str,
    revision: str,
    sidecar_name: str = DEFAULT_SIDECAR,
    manifest_name: str = DEFAULT_MANIFEST,
    expected_tensors: int | None = 15,
) -> dict:
    """Return a no-write plan based on indexes and safetensors headers."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    revision = _require_revision(revision)
    sidecar_name = _safe_sibling_name(
        sidecar_name, "sidecar", ".safetensors")
    manifest_name = _safe_sibling_name(
        manifest_name, "manifest", ".json")
    if not repository or not isinstance(repository, str):
        raise ValueError("source repository identity must be non-empty")
    if source == target:
        raise ValueError("source and quantized target must be different directories")

    source_index_path = source / INDEX_NAME
    target_index_path = target / INDEX_NAME
    source_index_bytes = source_index_path.read_bytes()
    target_index_bytes = target_index_path.read_bytes()
    source_index = json.loads(source_index_bytes)
    target_index = json.loads(target_index_bytes)
    for label, index in (("source", source_index), ("target", target_index)):
        if not isinstance(index, dict) or not isinstance(
            index.get("weight_map"), dict
        ):
            raise ValueError(f"{label} index has no weight_map object")

    tree_path = (
        source / ".cache" / "huggingface" / "trees" / f"{revision}.json")
    tree = _read_json(tree_path)
    tree_files = tree.get("files")
    if not isinstance(tree_files, dict):
        raise ValueError(f"pinned Hugging Face tree has no files map: {tree_path}")

    source_map = source_index["weight_map"]
    source_names = sorted(
        name for name in source_map if name.startswith("mtp."))
    if not source_names:
        raise ValueError("released source index contains no top-level mtp.* tensors")
    if expected_tensors is not None and len(source_names) != expected_tensors:
        raise ValueError(
            f"released source has {len(source_names)} mtp.* tensors, "
            f"expected {expected_tensors}")

    by_shard: dict[str, list[str]] = {}
    for name in source_names:
        shard = _safe_sibling_name(
            source_map[name], f"source shard for {name}", ".safetensors")
        by_shard.setdefault(shard, []).append(name)

    source_specs = {}
    shard_records = []
    for shard, names in sorted(by_shard.items()):
        path = source / shard
        tree_entry = tree_files.get(shard)
        if not isinstance(tree_entry, dict):
            raise ValueError(
                f"pinned Hugging Face tree does not contain source shard {shard}")
        expected_size = tree_entry.get("lfs_size", tree_entry.get("size"))
        expected_hash = tree_entry.get("lfs_sha256")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError(f"pinned tree has no valid size for {shard}")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise ValueError(f"pinned tree has no LFS SHA-256 for {shard}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"pinned source shard size mismatch for {shard}: "
                f"expected {expected_size}, got {actual_size}")
        header_specs = _tensor_specs(path)
        for name in names:
            spec = header_specs.get(name)
            if spec is None:
                raise ValueError(f"source index tensor {name} is absent from {shard}")
            if spec["dtype"] != "BF16":
                raise ValueError(
                    f"released MTP tensor {name} must be BF16, got {spec['dtype']}")
            source_specs[name] = dict(spec, shard=shard)
        shard_records.append({
            "path": shard,
            "size": actual_size,
            "pinned_lfs_sha256": expected_hash,
            "tensor_names": list(names),
        })

    target_config_path = target / "config.json"
    target_config = _read_json(target_config_path)
    quantization = (
        target_config.get("quantization")
        or target_config.get("quantization_config"))
    if not isinstance(quantization, dict):
        raise ValueError("target config does not declare quantization")
    try:
        quant_bits = int(quantization["bits"])
        quant_group_size = int(quantization["group_size"])
        quant_mode = str(quantization["mode"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "target quantization needs integer bits/group_size and mode") from error
    if quant_bits <= 0 or quant_group_size <= 0 or not quant_mode:
        raise ValueError("target quantization descriptor is invalid")

    target_map = target_index["weight_map"]
    target_mtp = {
        name: target_map[name]
        for name in sorted(target_map) if name.startswith("mtp.")
    }
    target_non_mtp = {
        name: target_map[name]
        for name in sorted(target_map) if not name.startswith("mtp.")
    }
    metadata = target_index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("target index metadata must be an object")

    return {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "directory": str(source),
            "repository": repository,
            "revision": revision,
            "tree_path": str(tree_path),
            "tree_sha256": _sha256_file(tree_path),
            "index_path": str(source_index_path),
            "index_sha256": _sha256_bytes(source_index_bytes),
            "shards": shard_records,
            "mtp_tensor_count": len(source_names),
            "mtp_tensors": {
                name: source_specs[name] for name in source_names},
            "mtp_tensor_bytes": sum(
                source_specs[name]["bytes"] for name in source_names),
        },
        "target": {
            "directory": str(target),
            "index_path": str(target_index_path),
            "index_before_sha256": _sha256_bytes(target_index_bytes),
            "config_sha256": _sha256_file(target_config_path),
            "quantization": {
                "bits": quant_bits,
                "group_size": quant_group_size,
                "mode": quant_mode,
            },
            "existing_sidecar_pointer": metadata.get("mtplx_mtp_sidecar"),
            "removed_mtp_entries": target_mtp,
            "non_mtp_weight_map_sha256": _mapping_fingerprint(target_non_mtp),
            "non_mtp_tensor_count": len(target_non_mtp),
        },
        "sidecar": {
            "path": str(target / sidecar_name),
            "file": sidecar_name,
            "manifest_path": str(target / manifest_name),
            "manifest_file": manifest_name,
            "estimated_payload_bytes": sum(
                source_specs[name]["bytes"] for name in source_names),
        },
    }


def _verify_source_shards(plan: dict) -> list[dict]:
    source = Path(plan["source"]["directory"])
    verified = []
    for record in plan["source"]["shards"]:
        actual = _sha256_file(source / record["path"])
        expected = record["pinned_lfs_sha256"]
        if actual != expected:
            raise ValueError(
                f"pinned source SHA-256 mismatch for {record['path']}: "
                f"expected {expected}, got {actual}")
        verified.append(dict(record, actual_sha256=actual))
    return verified


def _compare_sidecar_to_source(plan: dict, sidecar_path: Path) -> None:
    """Require bit-for-bit equality of every released MTP tensor payload."""
    sidecar_header = _safetensors_header(sidecar_path)
    sidecar_data_start = _safetensors_data_start(sidecar_path)
    source_dir = Path(plan["source"]["directory"])
    by_shard: dict[str, list[str]] = {}
    for name, spec in plan["source"]["mtp_tensors"].items():
        by_shard.setdefault(spec["shard"], []).append(name)
    for shard, names in sorted(by_shard.items()):
        source_path = source_dir / shard
        source_header = _safetensors_header(source_path)
        source_data_start = _safetensors_data_start(source_path)
        with source_path.open("rb") as source, sidecar_path.open("rb") as sidecar:
            for name in sorted(names):
                source_offsets = source_header[name]["data_offsets"]
                sidecar_offsets = sidecar_header[name]["data_offsets"]
                source_bytes = source_offsets[1] - source_offsets[0]
                sidecar_bytes = sidecar_offsets[1] - sidecar_offsets[0]
                if source_bytes != sidecar_bytes:
                    raise ValueError(
                        f"sidecar tensor byte length differs from released "
                        f"source: {name}")
                source.seek(source_data_start + source_offsets[0])
                sidecar.seek(sidecar_data_start + sidecar_offsets[0])
                remaining = source_bytes
                while remaining:
                    chunk_bytes = min(8 * 1024 * 1024, remaining)
                    if source.read(chunk_bytes) != sidecar.read(chunk_bytes):
                        raise ValueError(
                            "sidecar tensor payload differs from released "
                            f"source: {name}")
                    remaining -= chunk_bytes


def _idempotent_artifact_present(plan: dict) -> bool:
    sidecar = Path(plan["sidecar"]["path"])
    manifest = Path(plan["sidecar"]["manifest_path"])
    pointer = plan["target"]["existing_sidecar_pointer"]
    present = (sidecar.exists(), manifest.exists(), pointer is not None)
    if any(present) and not all(present):
        raise ValueError(
            "partial BF16 MTP sidecar artifact exists; refuse to overwrite it")
    if all(present) and pointer != sidecar.name:
        raise ValueError(
            f"target index points to {pointer!r}, not {sidecar.name!r}")
    return all(present)


def build_sidecar(
    source: str | Path,
    target: str | Path,
    *,
    repository: str,
    revision: str,
    sidecar_name: str = DEFAULT_SIDECAR,
    manifest_name: str = DEFAULT_MANIFEST,
    expected_tensors: int | None = 15,
    min_free_bytes: int = 10 * 1024**3,
) -> dict:
    """Build, bind, and independently validate the BF16 MTP sidecar."""
    plan = inspect_sidecar_inputs(
        source,
        target,
        repository=repository,
        revision=revision,
        sidecar_name=sidecar_name,
        manifest_name=manifest_name,
        expected_tensors=expected_tensors,
    )
    if _idempotent_artifact_present(plan):
        return validate_sidecar(
            source,
            target,
            repository=repository,
            revision=revision,
            sidecar_name=sidecar_name,
            manifest_name=manifest_name,
            expected_tensors=expected_tensors,
        )
    if min_free_bytes < 0:
        raise ValueError("minimum free byte reserve must be non-negative")
    target_dir = Path(plan["target"]["directory"])
    available = shutil.disk_usage(target_dir).free
    estimated = int(plan["sidecar"]["estimated_payload_bytes"]) + 1024**2
    if available - estimated < min_free_bytes:
        raise OSError(
            "insufficient free space for BF16 MTP sidecar plus reserve: "
            f"available={available}, estimated={estimated}, "
            f"required_post_build={min_free_bytes}")

    verified_shards = _verify_source_shards(plan)
    source_dir = Path(plan["source"]["directory"])
    arrays = {}
    by_shard: dict[str, list[str]] = {}
    for name, spec in plan["source"]["mtp_tensors"].items():
        by_shard.setdefault(spec["shard"], []).append(name)
    for shard, names in sorted(by_shard.items()):
        loaded = mx.load(str(source_dir / shard))
        for name in sorted(names):
            value = loaded[name]
            if value.dtype != mx.bfloat16:
                raise ValueError(
                    f"released MTP tensor {name} changed dtype to {value.dtype}")
            arrays[name] = value

    sidecar_path = Path(plan["sidecar"]["path"])
    temporary_sidecar = sidecar_path.with_name(
        f".{sidecar_path.stem}.tmp-{os.getpid()}.safetensors")
    if temporary_sidecar.exists():
        raise FileExistsError(f"stale sidecar temporary exists: {temporary_sidecar}")
    sidecar_header_metadata = {
        "schema": SIDECAR_SCHEMA,
        "source_repository": repository,
        "source_revision": revision,
        "source_index_sha256": plan["source"]["index_sha256"],
        "source_tree_sha256": plan["source"]["tree_sha256"],
        "tensor_count": str(len(arrays)),
    }
    try:
        mx.save_safetensors(
            str(temporary_sidecar),
            dict(sorted(arrays.items())),
            metadata=sidecar_header_metadata,
        )
        specs = _tensor_specs(temporary_sidecar)
        expected_names = set(plan["source"]["mtp_tensors"])
        if set(specs) != expected_names:
            raise ValueError(
                "built sidecar tensor set differs from released MTP tensor set")
        if any(not name.startswith("mtp.") for name in specs):
            raise ValueError("built sidecar contains a non-MTP tensor")
        if any(spec["dtype"] != "BF16" for spec in specs.values()):
            raise ValueError("built sidecar contains a non-BF16 tensor")
        _compare_sidecar_to_source(plan, temporary_sidecar)
        sidecar_sha256 = _sha256_file(temporary_sidecar)

        target_index_path = Path(plan["target"]["index_path"])
        target_index = _read_json(target_index_path)
        if _sha256_file(target_index_path) != plan["target"][
            "index_before_sha256"
        ]:
            raise RuntimeError("target index changed while sidecar was building")
        old_map = target_index["weight_map"]
        non_mtp = {
            name: value for name, value in old_map.items()
            if not name.startswith("mtp.")
        }
        if _mapping_fingerprint(non_mtp) != plan["target"][
            "non_mtp_weight_map_sha256"
        ]:
            raise RuntimeError("target non-MTP mapping changed during build")
        target_index["weight_map"] = non_mtp
        metadata = target_index.setdefault("metadata", {})
        metadata.update({
            "mtplx_mtp_sidecar": sidecar_path.name,
            "mtplx_mtp_sidecar_manifest": manifest_name,
            "mtplx_mtp_sidecar_sha256": sidecar_sha256,
            "mtplx_mtp_source_repository": repository,
            "mtplx_mtp_source_revision": revision,
        })
        index_after_bytes = _canonical_json(target_index)
        index_after_sha256 = _sha256_bytes(index_after_bytes)
        if _mapping_fingerprint(target_index["weight_map"]) != plan[
            "target"
        ]["non_mtp_weight_map_sha256"]:
            raise RuntimeError("sidecar index rewrite changed non-MTP mappings")

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source": dict(
                plan["source"],
                shards=verified_shards,
            ),
            "target": dict(
                plan["target"],
                index_after_sha256=index_after_sha256,
            ),
            "sidecar": dict(
                plan["sidecar"],
                size=temporary_sidecar.stat().st_size,
                sha256=sidecar_sha256,
                header_metadata=sidecar_header_metadata,
                tensor_count=len(specs),
                tensors={name: specs[name] for name in sorted(specs)},
            ),
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "proof": {
                "source_shards_match_pinned_lfs_sha256": True,
                "sidecar_tensors_byte_equal_released_source": True,
                "sidecar_contains_only_top_level_mtp_tensors": True,
                "sidecar_tensors_all_bf16": True,
                "target_non_mtp_weight_map_unchanged": True,
                "target_index_committed_last": True,
            },
        }

        with temporary_sidecar.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_sidecar, sidecar_path)
        _fsync_directory(sidecar_path.parent)
        _atomic_write(
            Path(plan["sidecar"]["manifest_path"]),
            _canonical_json(manifest),
        )
        # Commit point: the loader cannot observe the sidecar until this final
        # same-directory atomic replacement publishes its explicit pointer.
        _atomic_write(target_index_path, index_after_bytes)
    finally:
        if temporary_sidecar.exists():
            temporary_sidecar.unlink()
        del arrays
        mx.clear_cache()

    return validate_sidecar(
        source,
        target,
        repository=repository,
        revision=revision,
        sidecar_name=sidecar_name,
        manifest_name=manifest_name,
        expected_tensors=expected_tensors,
    )


def validate_sidecar(
    source: str | Path,
    target: str | Path,
    *,
    repository: str,
    revision: str,
    sidecar_name: str = DEFAULT_SIDECAR,
    manifest_name: str = DEFAULT_MANIFEST,
    expected_tensors: int | None = 15,
) -> dict:
    """Validate pin, hashes, tensor identity, and atomic index binding."""
    plan = inspect_sidecar_inputs(
        source,
        target,
        repository=repository,
        revision=revision,
        sidecar_name=sidecar_name,
        manifest_name=manifest_name,
        expected_tensors=expected_tensors,
    )
    sidecar_path = Path(plan["sidecar"]["path"])
    manifest_path = Path(plan["sidecar"]["manifest_path"])
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unrecognized BF16 MTP sidecar manifest schema")
    recorded_source = manifest.get("source", {})
    if not isinstance(recorded_source, dict) or (
        recorded_source.get("repository") != repository
        or recorded_source.get("revision") != revision
        or recorded_source.get("index_sha256")
        != plan["source"]["index_sha256"]
        or recorded_source.get("tree_sha256")
        != plan["source"]["tree_sha256"]
    ):
        raise ValueError("sidecar manifest source fingerprint mismatch")
    verified_shards = _verify_source_shards(plan)
    recorded_shards = recorded_source.get("shards")
    if recorded_shards != verified_shards:
        raise ValueError("sidecar manifest pinned shard evidence mismatch")

    sidecar_sha256 = _sha256_file(sidecar_path)
    recorded_sidecar = manifest.get("sidecar", {})
    if not isinstance(recorded_sidecar, dict) or (
        recorded_sidecar.get("file") != sidecar_path.name
        or recorded_sidecar.get("sha256") != sidecar_sha256
        or recorded_sidecar.get("size") != sidecar_path.stat().st_size
    ):
        raise ValueError("sidecar file fingerprint does not match manifest")
    specs = _tensor_specs(sidecar_path)
    expected_specs = {
        name: {key: value for key, value in source_spec.items()
               if key != "shard"}
        for name, source_spec in plan["source"]["mtp_tensors"].items()
    }
    if specs != expected_specs:
        raise ValueError("sidecar tensor metadata differs from released source")
    if recorded_sidecar.get("tensors") != {
        name: specs[name] for name in sorted(specs)
    }:
        raise ValueError("sidecar tensor manifest differs from file header")
    header_metadata = _sidecar_metadata(sidecar_path)
    expected_header_metadata = {
        "schema": SIDECAR_SCHEMA,
        "source_repository": repository,
        "source_revision": revision,
        "source_index_sha256": plan["source"]["index_sha256"],
        "source_tree_sha256": plan["source"]["tree_sha256"],
        "tensor_count": str(len(specs)),
    }
    if header_metadata != expected_header_metadata:
        raise ValueError("sidecar embedded source fingerprint mismatch")

    target_index_path = Path(plan["target"]["index_path"])
    target_index = _read_json(target_index_path)
    target_map = target_index["weight_map"]
    if any(name.startswith("mtp.") for name in target_map):
        raise ValueError("target index still contains colliding MTP body entries")
    recorded_target = manifest.get("target", {})
    if not isinstance(recorded_target, dict):
        raise ValueError("sidecar manifest target evidence is absent")
    if _sha256_file(target_index_path) != recorded_target.get(
        "index_after_sha256"
    ):
        raise ValueError("target index fingerprint differs from sidecar manifest")
    non_mtp_fingerprint = _mapping_fingerprint(target_map)
    if non_mtp_fingerprint != recorded_target.get(
        "non_mtp_weight_map_sha256"
    ):
        raise ValueError("target non-MTP mapping fingerprint changed")
    metadata = target_index.get("metadata", {})
    expected_binding = {
        "mtplx_mtp_sidecar": sidecar_path.name,
        "mtplx_mtp_sidecar_manifest": manifest_path.name,
        "mtplx_mtp_sidecar_sha256": sidecar_sha256,
        "mtplx_mtp_source_repository": repository,
        "mtplx_mtp_source_revision": revision,
    }
    if any(metadata.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("target index BF16 MTP sidecar binding is incomplete")

    _compare_sidecar_to_source(plan, sidecar_path)
    return {
        "schema": MANIFEST_SCHEMA,
        "passed": True,
        "source_repository": repository,
        "source_revision": revision,
        "source_shards": verified_shards,
        "sidecar_file": sidecar_path.name,
        "sidecar_sha256": sidecar_sha256,
        "sidecar_bytes": sidecar_path.stat().st_size,
        "tensor_count": len(specs),
        "target_index_sha256": _sha256_file(target_index_path),
        "non_mtp_weight_map_sha256": non_mtp_fingerprint,
        "tensor_payloads_byte_equal_released_source": True,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--sidecar-name", default=DEFAULT_SIDECAR)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-tensors", type=int, default=15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="inspect without writing")
    _common_arguments(plan)
    build = subparsers.add_parser("build", help="build and atomically bind")
    _common_arguments(build)
    build.add_argument("--min-free-gib", type=float, default=10.0)
    validate = subparsers.add_parser("validate", help="verify an existing sidecar")
    _common_arguments(validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_tensors <= 0:
        raise SystemExit("--expected-tensors must be positive")
    common = {
        "repository": args.repository,
        "revision": args.revision,
        "sidecar_name": args.sidecar_name,
        "manifest_name": args.manifest_name,
        "expected_tensors": args.expected_tensors,
    }
    try:
        if args.command == "plan":
            report = inspect_sidecar_inputs(
                args.source, args.target, **common)
            available = shutil.disk_usage(args.target).free
            report["disk"] = {
                "available_bytes": available,
                "estimated_post_build_bytes": (
                    available
                    - report["sidecar"]["estimated_payload_bytes"]
                    - 1024**2),
            }
        elif args.command == "build":
            if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
                raise ValueError("--min-free-gib must be finite and non-negative")
            report = build_sidecar(
                args.source,
                args.target,
                min_free_bytes=int(args.min_free_gib * 1024**3),
                **common,
            )
        else:
            report = validate_sidecar(
                args.source, args.target, **common)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
