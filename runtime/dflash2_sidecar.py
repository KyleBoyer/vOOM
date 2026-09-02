#!/usr/bin/env python3
"""Plan, build, and validate pinned Qwen3.8/GLM-5.3 DFlash 2 sidecars.

Phase A is intentionally default-off.  Nothing in this module wires DFlash 2
into ``runtime.server`` or claims that recurrent target rollback is proven.
``plan`` is pure metadata work and can run before the 3.85 GB BF16 shard is
downloaded.  ``build`` requires the complete hash-pinned shard, verifies it
before creating output, quantizes only the *draft*, and atomically publishes a
standalone artifact.  A quantized drafter can change acceptance and speed; it
cannot preserve target semantics without the later exact target verifier.

The default affine-4/group-64 conversion matches z-lab/dflash's documented MLX
``--draft-bits 4`` path. Explicit affine-2/3 builds are supported only as
target-verified memory/acceptance experiments. Its README recommends a block
size no
larger than five when either target or draft is quantized, so the plan records
four proposals per round even though the checkpoint's architectural maximum is
seven.  Runtime code must not silently rewrite the checkpoint's own
``dflash_config.block_size=8``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .dflash2_schema import (
    DFlash2Config,
    OFFICIAL_CONFIG,
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OFFICIAL_UPSTREAM_REVISION,
    OFFICIAL_WEIGHTS_BYTES,
    OFFICIAL_WEIGHTS_SHA256,
    _read_safetensors_header,
    canonical_json,
    inspect_source_file,
    release_for_repository,
    release_for_variant,
    require_revision,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "dflash2-sidecar.manifest.json"
SIDECAR_SCHEMA = "voom.qwen38-dflash2-sidecar.v1"
MANIFEST_SCHEMA = "voom.qwen38-dflash2-sidecar-manifest.v1"
UPSTREAM_LICENSE_NAME = "UPSTREAM_DFLASH_MIT.txt"
DEFAULT_MIN_FREE_BYTES = 10_000_000_000

UPSTREAM_MIT_LICENSE = """MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _source_tree(
    source: Path,
    *,
    revision: str,
    expected_weights_sha256: str,
    expected_weights_bytes: int,
) -> dict[str, Any]:
    tree_path = source / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    tree = _read_json(tree_path)
    files = tree.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"pinned Hugging Face tree has no files map: {tree_path}")
    entry = files.get(WEIGHTS_NAME)
    if not isinstance(entry, dict):
        raise ValueError(f"pinned Hugging Face tree omits {WEIGHTS_NAME}")
    if entry.get("lfs_sha256") != expected_weights_sha256:
        raise ValueError(
            "pinned Hugging Face tree weight SHA-256 mismatch: "
            f"expected {expected_weights_sha256}, got {entry.get('lfs_sha256')}")
    if entry.get("lfs_size") != expected_weights_bytes:
        raise ValueError(
            "pinned Hugging Face tree weight size mismatch: "
            f"expected {expected_weights_bytes}, got {entry.get('lfs_size')}")
    return {
        "tree_path": str(tree_path),
        "lfs_sha256": entry["lfs_sha256"],
        "lfs_bytes": entry["lfs_size"],
        "blob_id": entry.get("blob_id", ""),
        "xet_hash": entry.get("xet_hash", ""),
    }


def inspect_pinned_source(
    source: str | Path,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    revision: str = OFFICIAL_REVISION,
    expected_config_sha256: str = OFFICIAL_CONFIG_SHA256,
    expected_weights_sha256: str = OFFICIAL_WEIGHTS_SHA256,
    expected_weights_bytes: int = OFFICIAL_WEIGHTS_BYTES,
    require_official_geometry: bool = True,
) -> dict[str, Any]:
    """Inspect config, Hub receipt, and tensor header without reading payload."""
    source = Path(source).resolve()
    revision = require_revision(revision)
    expected_config_sha256 = require_sha256(
        expected_config_sha256, "expected config SHA-256")
    expected_weights_sha256 = require_sha256(
        expected_weights_sha256, "expected weights SHA-256")
    if not isinstance(repository, str) or not repository:
        raise ValueError("source repository identity must be non-empty")
    if (isinstance(expected_weights_bytes, bool)
            or not isinstance(expected_weights_bytes, int)
            or expected_weights_bytes <= 0):
        raise ValueError("expected weights size must be positive")

    config_path = source / CONFIG_NAME
    weights_path = source / WEIGHTS_NAME
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete DFlash2 source checkpoint: {source}")
    config_bytes = config_path.read_bytes()
    config_sha256 = sha256_bytes(config_bytes)
    if config_sha256 != expected_config_sha256:
        raise ValueError(
            "DFlash2 source config SHA-256 mismatch: "
            f"expected {expected_config_sha256}, got {config_sha256}")
    raw_config = json.loads(config_bytes)
    config = DFlash2Config.from_mapping(raw_config)
    release = None
    if require_official_geometry:
        release = release_for_repository(repository)
        expected_identity = (
            revision,
            expected_config_sha256,
            expected_weights_sha256,
            expected_weights_bytes,
        )
        pinned_identity = (
            release.revision,
            release.config_sha256,
            release.weights_sha256,
            release.weights_bytes,
        )
        if expected_identity != pinned_identity:
            raise ValueError(
                f"DFlash2 {release.variant} source identity does not match "
                "the pinned release")
        config.validate_official_release(release)
    tree = _source_tree(
        source,
        revision=revision,
        expected_weights_sha256=expected_weights_sha256,
        expected_weights_bytes=expected_weights_bytes,
    )
    if weights_path.stat().st_size != expected_weights_bytes:
        raise ValueError(
            "DFlash2 local shard size mismatch: "
            f"expected {expected_weights_bytes}, got {weights_path.stat().st_size}")
    header = inspect_source_file(config, weights_path)
    if require_official_geometry:
        assert release is not None
        if header["parameter_count"] != release.parameter_count:
            raise ValueError(
                "DFlash2 official parameter count mismatch: "
                f"expected {release.parameter_count}, "
                f"got {header['parameter_count']}")
        if header["file_bytes"] != release.weights_bytes:
            raise ValueError("DFlash2 official file byte count mismatch")
    return {
        "repository": repository,
        "revision": revision,
        "source": str(source),
        "config_sha256": config_sha256,
        "weights_sha256": expected_weights_sha256,
        "weights_sha256_verified": False,
        "weights_bytes": expected_weights_bytes,
        "tree": tree,
        "header": header,
        "config": config,
    }


def _canonical_output_name(name: str) -> str:
    if name in {
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
    }:
        return f"{name}.weight"
    return name


def _quantizable(config: DFlash2Config, name: str, group_size: int) -> bool:
    spec = config.expected_tensor_specs()[name]
    return len(spec.shape) == 2 and spec.shape[-1] % group_size == 0


def _estimated_quantized_bytes(
    config: DFlash2Config,
    *,
    bits: int,
    group_size: int,
    mode: str,
) -> tuple[int, int, int]:
    quantized = 0
    retained = 0
    count = 0
    for name, spec in config.expected_tensor_specs().items():
        if not _quantizable(config, name, group_size):
            retained += spec.nbytes
            continue
        count += 1
        rows = math.prod(spec.shape[:-1])
        width = spec.shape[-1]
        # MLX packs floor(32 / bits) values per uint32. Non-power-of-two bit
        # widths therefore have row-tail padding that an ideal bits/8 estimate
        # would miss (notably 3-bit: 10 values per uint32).
        packed_values = 32 // bits
        quantized += rows * math.ceil(width / packed_values) * 4
        groups = rows * width // group_size
        # MLX affine stores one 16-bit scale and bias per group.  MX formats
        # use a single byte exponent scale and no bias.
        quantized += groups * (1 if mode in {"mxfp4", "mxfp8"} else 4)
    return quantized + retained, count, retained


def plan_sidecar(
    source: str | Path | None = None,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    revision: str = OFFICIAL_REVISION,
    expected_config_sha256: str = OFFICIAL_CONFIG_SHA256,
    expected_weights_sha256: str = OFFICIAL_WEIGHTS_SHA256,
    expected_weights_bytes: int = OFFICIAL_WEIGHTS_BYTES,
    require_official_geometry: bool = True,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> dict[str, Any]:
    if bits not in (2, 3, 4, 8):
        raise ValueError("DFlash2 draft bits must be 2, 3, 4, or 8")
    if group_size <= 0:
        raise ValueError("DFlash2 draft group_size must be positive")
    if mode not in ("affine", "mxfp4", "mxfp8"):
        raise ValueError("unsupported DFlash2 draft quantization mode")
    if mode == "mxfp4" and (bits, group_size) != (4, 32):
        raise ValueError("mxfp4 requires bits=4 and group_size=32")
    if mode == "mxfp8" and (bits, group_size) != (8, 32):
        raise ValueError("mxfp8 requires bits=8 and group_size=32")

    inspected = None
    if source is None:
        release = (
            release_for_repository(repository)
            if require_official_geometry else None)
        config = DFlash2Config.from_mapping(
            release.config if release is not None else OFFICIAL_CONFIG)
        if release is not None:
            config.validate_official_release(release)
    else:
        inspected = inspect_pinned_source(
            source,
            repository=repository,
            revision=revision,
            expected_config_sha256=expected_config_sha256,
            expected_weights_sha256=expected_weights_sha256,
            expected_weights_bytes=expected_weights_bytes,
            require_official_geometry=require_official_geometry,
        )
        config = inspected["config"]

    estimate, quantized_tensors, retained_bytes = _estimated_quantized_bytes(
        config, bits=bits, group_size=group_size, mode=mode)
    tensor_schema = {
        name: spec.as_dict()
        for name, spec in config.expected_tensor_specs().items()}
    source_report = {
        "repository": repository,
        "revision": revision,
        "config_sha256": expected_config_sha256,
        "weights_sha256": expected_weights_sha256,
        "weights_bytes": expected_weights_bytes,
        "local_header_validated": inspected is not None,
    }
    if inspected is not None:
        source_report.update({
            "path": inspected["source"],
            "header": inspected["header"],
            "tree": inspected["tree"],
        })
    return {
        "schema": "voom.qwen38-dflash2-build-plan.v1",
        "source": source_report,
        "architecture": {
            "architecture": "DFlash2DraftModel",
            "tensor_count": len(tensor_schema),
            "parameter_count": config.source_parameter_count,
            "tensor_schema_sha256": sha256_bytes(canonical_json(tensor_schema)),
            "target_layer_ids": list(config.target_layer_ids),
            "checkpoint_block_size": config.block_size,
            "checkpoint_proposal_count": config.proposal_count,
            "selector_top_k": config.selector_top_k,
            "selector_rank": config.selector_rank,
            "conv_kernel_size": config.conv_kernel_size,
            "conv_group_size": config.conv_group_size,
            "is_causal": False,
        },
        "conversion": {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
            "quantized_tensors": quantized_tensors,
            "retained_bf16_bytes": retained_bytes,
            "estimated_output_tensor_bytes": estimate,
            "estimate_excludes_safetensors_header": True,
        },
        "serving": {
            "runtime_supported": False,
            "enabled_by_default": False,
            "upstream_mlx_recommended_block_size": min(config.block_size, 5),
            "planned_proposal_count": min(config.block_size, 5) - 1,
            "target_verifier_required": True,
            "recurrent_rollback_oracle_required": True,
        },
        "upstream": {
            "repository": OFFICIAL_UPSTREAM_REPOSITORY,
            "revision": OFFICIAL_UPSTREAM_REVISION,
            "license": "MIT",
        },
    }


def _validate_output_header(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    header, header_bytes, file_bytes = _read_safetensors_header(path)
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("invalid DFlash2 sidecar safetensors metadata")
    required = {
        "format": "mlx",
        "vmodel_kind": "target-verified-dflash2-draft-sidecar",
        "runtime_supported": "false",
        "source_sha256": manifest["source"]["weights_sha256"],
        "source_revision": manifest["source"]["revision"],
    }
    for name, value in required.items():
        if metadata.get(name) != value:
            raise ValueError(f"DFlash2 sidecar metadata mismatch for {name}")
    if len(header) != manifest["output"]["tensor_count"]:
        raise ValueError("DFlash2 sidecar output tensor count mismatch")
    if any(not isinstance(value, dict) for value in header.values()):
        raise ValueError("invalid DFlash2 sidecar tensor header")
    return {
        "tensor_count": len(header),
        "header_bytes": header_bytes,
        "file_bytes": file_bytes,
    }


def build_sidecar(
    source: str | Path,
    output: str | Path,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    revision: str = OFFICIAL_REVISION,
    expected_config_sha256: str = OFFICIAL_CONFIG_SHA256,
    expected_weights_sha256: str = OFFICIAL_WEIGHTS_SHA256,
    expected_weights_bytes: int = OFFICIAL_WEIGHTS_BYTES,
    require_official_geometry: bool = True,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, Any]:
    """Build a quantized draft-only artifact; never mutate the target model."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("DFlash2 source and output must be different directories")
    if output.exists():
        existing = validate_sidecar(
            source, output,
            repository=repository,
            revision=revision,
            expected_config_sha256=expected_config_sha256,
            expected_weights_sha256=expected_weights_sha256,
            expected_weights_bytes=expected_weights_bytes,
            require_official_geometry=require_official_geometry,
        )
        conversion = existing.get("conversion", {})
        requested = {"bits": bits, "group_size": group_size, "mode": mode}
        if any(conversion.get(name) != value
               for name, value in requested.items()):
            raise ValueError(
                "existing DFlash2 sidecar conversion differs from request")
        return existing
    plan = plan_sidecar(
        source,
        repository=repository,
        revision=revision,
        expected_config_sha256=expected_config_sha256,
        expected_weights_sha256=expected_weights_sha256,
        expected_weights_bytes=expected_weights_bytes,
        require_official_geometry=require_official_geometry,
        bits=bits,
        group_size=group_size,
        mode=mode,
    )
    if isinstance(min_free_bytes, bool) or min_free_bytes < 0:
        raise ValueError("minimum free bytes must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    estimated = plan["conversion"]["estimated_output_tensor_bytes"]
    free = shutil.disk_usage(output.parent).free
    if free - estimated < min_free_bytes:
        raise OSError(
            "DFlash2 sidecar build would violate the free-space floor: "
            f"free={free}, estimate={estimated}, floor={min_free_bytes}")

    # This is deliberately before the MLX import and before temp output.  A
    # mismatched multi-gigabyte source fails without allocating Metal memory.
    weights_path = source / WEIGHTS_NAME
    actual_source_sha256 = sha256_file(weights_path)
    if actual_source_sha256 != expected_weights_sha256:
        raise ValueError(
            "DFlash2 source shard SHA-256 mismatch: "
            f"expected {expected_weights_sha256}, got {actual_source_sha256}")

    import mlx.core as mx  # Lazy: ``plan`` and source validation remain pure.

    config = DFlash2Config.from_json(source / CONFIG_NAME)
    lazy = dict(mx.load(str(weights_path)))
    expected_names = set(config.expected_tensor_specs())
    if set(lazy) != expected_names:
        raise ValueError("DFlash2 source tensor keys changed after header validation")

    output_tensors: dict[str, Any] = {}
    quantized_count = 0
    for source_name in sorted(tuple(lazy)):
        value = lazy.pop(source_name)
        output_name = _canonical_output_name(source_name)
        if not _quantizable(config, source_name, group_size):
            output_tensors[output_name] = value
            continue
        packed = mx.quantize(
            value, group_size=group_size, bits=bits, mode=mode)
        mx.eval(packed)
        stem = output_name[:-len(".weight")]
        output_tensors[output_name] = packed[0]
        output_tensors[f"{stem}.scales"] = packed[1]
        if len(packed) > 2:
            output_tensors[f"{stem}.biases"] = packed[2]
        quantized_count += 1

    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.building-", dir=output.parent))
    try:
        sidecar_path = tmp_dir / WEIGHTS_NAME
        mx.save_safetensors(str(sidecar_path), output_tensors, metadata={
            "format": "mlx",
            "vmodel_kind": "target-verified-dflash2-draft-sidecar",
            "runtime_supported": "false",
            "source_sha256": actual_source_sha256,
            "source_revision": revision,
        })
        output_sha256 = sha256_file(sidecar_path)
        raw_config = _read_json(source / CONFIG_NAME)
        raw_config["quantization"] = {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
        }
        raw_config["quantization_config"] = dict(raw_config["quantization"])
        raw_config["vmodel_sidecar"] = {
            "schema": SIDECAR_SCHEMA,
            "source_repository": repository,
            "source_revision": revision,
            "source_sha256": actual_source_sha256,
            "source_tensor_schema_sha256": plan["architecture"][
                "tensor_schema_sha256"],
            "draft_only_quantization": True,
            "target_verifier_required": True,
            "recurrent_rollback_oracle_required": True,
            "runtime_supported": False,
            "enabled_by_default": False,
            "upstream_revision": OFFICIAL_UPSTREAM_REVISION,
        }
        (tmp_dir / CONFIG_NAME).write_bytes(canonical_json(raw_config))
        (tmp_dir / UPSTREAM_LICENSE_NAME).write_text(UPSTREAM_MIT_LICENSE)
        for name in ("README.md", "LICENSE", "LICENSE.txt"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, tmp_dir / name)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source": {
                "path": str(source),
                "repository": repository,
                "revision": revision,
                "config_sha256": expected_config_sha256,
                "weights_sha256": actual_source_sha256,
                "weights_bytes": expected_weights_bytes,
                "tensor_schema_sha256": plan["architecture"][
                    "tensor_schema_sha256"],
            },
            "conversion": plan["conversion"],
            "output": {
                "weights_sha256": output_sha256,
                "weights_bytes": sidecar_path.stat().st_size,
                "tensor_count": len(output_tensors),
                "quantized_modules": quantized_count,
            },
            "serving": plan["serving"],
            "upstream": plan["upstream"],
            "proof": {
                "source_full_sha256_verified": True,
                "source_header_validated": True,
                "target_weights_copied": False,
                "runtime_supported": False,
                "recurrent_rollback_proven": False,
            },
        }
        (tmp_dir / MANIFEST_NAME).write_bytes(canonical_json(manifest))
        _validate_output_header(sidecar_path, manifest)
        os.replace(tmp_dir, output)
        _fsync_directory(output.parent)
        return {**manifest, "passed": True}
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def validate_sidecar(
    source: str | Path,
    output: str | Path,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    revision: str = OFFICIAL_REVISION,
    expected_config_sha256: str = OFFICIAL_CONFIG_SHA256,
    expected_weights_sha256: str = OFFICIAL_WEIGHTS_SHA256,
    expected_weights_bytes: int = OFFICIAL_WEIGHTS_BYTES,
    require_official_geometry: bool = True,
) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    inspected = inspect_pinned_source(
        source,
        repository=repository,
        revision=revision,
        expected_config_sha256=expected_config_sha256,
        expected_weights_sha256=expected_weights_sha256,
        expected_weights_bytes=expected_weights_bytes,
        require_official_geometry=require_official_geometry,
    )
    actual_source_sha256 = sha256_file(source / WEIGHTS_NAME)
    if actual_source_sha256 != expected_weights_sha256:
        raise ValueError("DFlash2 source shard SHA-256 mismatch during validation")
    manifest = _read_json(output / MANIFEST_NAME)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("DFlash2 sidecar manifest schema mismatch")
    expected_source = {
        "path": str(source),
        "repository": repository,
        "revision": revision,
        "config_sha256": expected_config_sha256,
        "weights_sha256": actual_source_sha256,
        "weights_bytes": expected_weights_bytes,
        "tensor_schema_sha256": inspected["header"]["tensor_schema_sha256"],
    }
    if manifest.get("source") != expected_source:
        raise ValueError("DFlash2 sidecar source identity mismatch")
    sidecar_path = output / WEIGHTS_NAME
    if sha256_file(sidecar_path) != manifest["output"].get("weights_sha256"):
        raise ValueError("DFlash2 sidecar output SHA-256 mismatch")
    config = _read_json(output / CONFIG_NAME)
    metadata = config.get("vmodel_sidecar")
    if not isinstance(metadata, dict):
        raise ValueError("DFlash2 sidecar config has no vmodel_sidecar metadata")
    required = {
        "schema": SIDECAR_SCHEMA,
        "source_repository": repository,
        "source_revision": revision,
        "source_sha256": actual_source_sha256,
        "runtime_supported": False,
        "enabled_by_default": False,
    }
    if any(metadata.get(name) != value for name, value in required.items()):
        raise ValueError("DFlash2 sidecar config identity mismatch")
    conversion = manifest.get("conversion")
    if not isinstance(conversion, dict):
        raise ValueError("DFlash2 sidecar manifest has no conversion object")
    expected_quantization = {
        name: conversion.get(name) for name in ("bits", "group_size", "mode")}
    if config.get("quantization") != expected_quantization:
        raise ValueError("DFlash2 sidecar quantization metadata mismatch")
    if config.get("quantization_config") != expected_quantization:
        raise ValueError("DFlash2 sidecar quantization_config mismatch")
    if (output / UPSTREAM_LICENSE_NAME).read_text() != UPSTREAM_MIT_LICENSE:
        raise ValueError("DFlash2 upstream MIT notice is missing or changed")
    _validate_output_header(sidecar_path, manifest)
    return {**manifest, "passed": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "build", "validate"):
        sub = commands.add_parser(command)
        if command != "plan":
            sub.add_argument("--source", required=True)
            sub.add_argument("--output", required=True)
        else:
            sub.add_argument("--source")
        sub.add_argument(
            "--variant", choices=("qwen38", "glm53-flash"),
            default="qwen38")
        sub.add_argument("--repository")
        sub.add_argument("--revision")
        if command in ("plan", "build"):
            sub.add_argument("--bits", type=int, default=4)
            sub.add_argument("--group-size", type=int, default=64)
            sub.add_argument("--mode", default="affine")
        if command == "build":
            sub.add_argument(
                "--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    release = release_for_variant(args.variant)
    repository = args.repository or release.repository
    revision = args.revision or release.revision
    if repository != release.repository or revision != release.revision:
        raise ValueError(
            "--repository/--revision must match the selected pinned variant")
    common = {
        "repository": repository,
        "revision": revision,
        "expected_config_sha256": release.config_sha256,
        "expected_weights_sha256": release.weights_sha256,
        "expected_weights_bytes": release.weights_bytes,
    }
    if args.command == "plan":
        report = plan_sidecar(
            args.source,
            bits=args.bits,
            group_size=args.group_size,
            mode=args.mode,
            **common,
        )
    elif args.command == "build":
        report = build_sidecar(
            args.source,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
            mode=args.mode,
            min_free_bytes=args.min_free_bytes,
            **common,
        )
    else:
        report = validate_sidecar(args.source, args.output, **common)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
