#!/usr/bin/env python3
"""Create a metadata-only target clone that restores proposal-only quantized MTP.

The Huihui all-MXFP4 conversion still contains its original packed ``mtp.*``
tensors in one shard, while the serving index intentionally replaces them with
the released-BF16 MTP sidecar.  This tool creates a sibling experiment directory
whose large files are symlinks and whose index re-exposes those packed tensors.
The served target body is unchanged; only the target-verified draft source is
different.  The source directory is never modified.
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


INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "mtp-quant-clone.manifest.json"
MANIFEST_SCHEMA = "voom.qwen-mtp-quant-clone.v1"
FAST_MANIFEST_NAME = "fast_tier_manifest.json"
FAST_ALIAS_MANIFEST_NAME = "mtp-quant-fast-alias.manifest.json"
FAST_ALIAS_SCHEMA = "voom.qwen-mtp-quant-fast-alias.v1"
EXPECTED_MTP_LOGICAL = 15
EXPECTED_MTP_PHYSICAL = 23

# Exact Qwen3.8-27B MTP proposal topology.  The clone is deliberately not a
# generic "anything named mtp" escape hatch: a different checkpoint must earn
# its own shape/header proof before packed draft weights can be exposed.
EXPECTED_MTP_SPECS = {
    "mtp.fc.scales": ("U8", (5120, 320), 1_638_400),
    "mtp.fc.weight": ("U32", (5120, 1280), 26_214_400),
    "mtp.layers.0.input_layernorm.weight": ("BF16", (5120,), 10_240),
    "mtp.layers.0.mlp.down_proj.scales": ("U8", (5120, 544), 2_785_280),
    "mtp.layers.0.mlp.down_proj.weight": ("U32", (5120, 2176), 44_564_480),
    "mtp.layers.0.mlp.gate_proj.scales": ("U8", (17408, 160), 2_785_280),
    "mtp.layers.0.mlp.gate_proj.weight": ("U32", (17408, 640), 44_564_480),
    "mtp.layers.0.mlp.up_proj.scales": ("U8", (17408, 160), 2_785_280),
    "mtp.layers.0.mlp.up_proj.weight": ("U32", (17408, 640), 44_564_480),
    "mtp.layers.0.post_attention_layernorm.weight": ("BF16", (5120,), 10_240),
    "mtp.layers.0.self_attn.k_norm.weight": ("BF16", (256,), 512),
    "mtp.layers.0.self_attn.k_proj.scales": ("U8", (1024, 160), 163_840),
    "mtp.layers.0.self_attn.k_proj.weight": ("U32", (1024, 640), 2_621_440),
    "mtp.layers.0.self_attn.o_proj.scales": ("U8", (5120, 192), 983_040),
    "mtp.layers.0.self_attn.o_proj.weight": ("U32", (5120, 768), 15_728_640),
    "mtp.layers.0.self_attn.q_norm.weight": ("BF16", (256,), 512),
    "mtp.layers.0.self_attn.q_proj.scales": ("U8", (12288, 160), 1_966_080),
    "mtp.layers.0.self_attn.q_proj.weight": ("U32", (12288, 640), 31_457_280),
    "mtp.layers.0.self_attn.v_proj.scales": ("U8", (1024, 160), 163_840),
    "mtp.layers.0.self_attn.v_proj.weight": ("U32", (1024, 640), 2_621_440),
    "mtp.norm.weight": ("BF16", (5120,), 10_240),
    "mtp.pre_fc_norm_embedding.weight": ("BF16", (5120,), 10_240),
    "mtp.pre_fc_norm_hidden.weight": ("BF16", (5120,), 10_240),
}


def _canonical_json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        value = json.loads(handle.read(length))
    if not isinstance(value, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    payload_bytes = path.stat().st_size - 8 - length
    for name, entry in value.items():
        if name == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_bytes
        ):
            raise ValueError(f"tensor extent exceeds safetensors shard: {name}")
    return value


def _mtp_physical(source: Path) -> tuple[dict[str, str], dict[str, dict]]:
    mapping: dict[str, str] = {}
    specs: dict[str, dict] = {}
    for shard in sorted(source.glob("*.safetensors")):
        if shard.name.startswith("mtp-bf16"):
            continue
        for name, value in _header(shard).items():
            if name == "__metadata__" or not name.startswith("mtp."):
                continue
            if name in mapping:
                raise ValueError(f"duplicate MTP tensor {name}")
            if not isinstance(value, dict):
                raise ValueError(f"invalid tensor metadata for {name}")
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
                raise ValueError(f"invalid tensor header for {name}")
            mapping[name] = shard.name
            specs[name] = {
                "dtype": dtype,
                "shape": shape,
                "bytes": offsets[1] - offsets[0],
            }
    return mapping, specs


def plan(source: str | Path, output: str | Path) -> dict:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("output must be a separate sibling directory")
    index_path = source / INDEX_NAME
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("source index has no weight_map")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("mtplx_mtp_sidecar"):
        raise ValueError("source is not an MTPLX BF16-sidecar target")
    config = json.loads((source / "config.json").read_text())
    provenance = config.get("voom_quantization")
    if not (
        isinstance(provenance, dict)
        and provenance.get("profile") == "all"
        and isinstance(provenance.get("source"), str)
        and provenance["source"]
    ):
        raise ValueError(
            "source must carry the all-weight vOOM lossy provenance marker")
    quant = config.get("quantization") or config.get("quantization_config")
    expected_quant = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    if not isinstance(quant, dict) or any(
        quant.get(key) != value for key, value in expected_quant.items()
    ):
        raise ValueError("source must use standard MLX MXFP4/group32")
    physical, specs = _mtp_physical(source)
    if len(physical) != EXPECTED_MTP_PHYSICAL:
        raise ValueError(
            f"expected {EXPECTED_MTP_PHYSICAL} packed MTP tensors, "
            f"found {len(physical)}")
    logical = sorted(name for name in physical if name.endswith(".weight"))
    if len(logical) != EXPECTED_MTP_LOGICAL:
        raise ValueError(
            f"expected {EXPECTED_MTP_LOGICAL} logical MTP weights, "
            f"found {len(logical)}")
    quantized = sorted(
        name for name in logical
        if name[:-len(".weight")] + ".scales" in physical)
    if len(quantized) != 8:
        raise ValueError(f"expected 8 quantized MTP matrices, found {len(quantized)}")
    actual_specs = {
        name: (spec["dtype"], tuple(spec["shape"]), spec["bytes"])
        for name, spec in specs.items()
    }
    if actual_specs != EXPECTED_MTP_SPECS:
        missing = sorted(set(EXPECTED_MTP_SPECS) - set(actual_specs))
        extra = sorted(set(actual_specs) - set(EXPECTED_MTP_SPECS))
        mismatched = sorted(
            name for name in set(actual_specs) & set(EXPECTED_MTP_SPECS)
            if actual_specs[name] != EXPECTED_MTP_SPECS[name])
        raise ValueError(
            "packed MTP schema mismatch: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}")
    return {
        "schema": MANIFEST_SCHEMA,
        "source": str(source),
        "output": str(output),
        "source_index_sha256": _sha256_bytes(index_bytes),
        "quantization": expected_quant,
        "voom_quantization": {
            "profile": provenance["profile"],
            "source": provenance["source"],
        },
        "mtp_physical_tensors": len(physical),
        "mtp_logical_tensors": len(logical),
        "mtp_quantized_matrices": len(quantized),
        "mtp_payload_bytes": sum(spec["bytes"] for spec in specs.values()),
        "mtp_mapping": dict(sorted(physical.items())),
        "mtp_specs": dict(sorted(specs.items())),
        "target_body_mapping_sha256": _sha256_bytes(_canonical_json({
            name: shard for name, shard in sorted(index["weight_map"].items())
            if not name.startswith("mtp.")
        })),
    }


def build(source: str | Path, output: str | Path) -> dict:
    record = plan(source, output)
    source = Path(record["source"])
    output = Path(record["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp-"))
    try:
        for path in source.iterdir():
            if path.name in {INDEX_NAME, "mtp-bf16.safetensors",
                             "mtp-bf16.manifest.json"}:
                continue
            if path.is_file():
                (temporary / path.name).symlink_to(path.resolve())
        index = json.loads((source / INDEX_NAME).read_text())
        index["weight_map"].update(record["mtp_mapping"])
        metadata = dict(index.get("metadata") or {})
        for name in tuple(metadata):
            if name.startswith("mtplx_mtp_"):
                metadata.pop(name)
        metadata["vmodel_mtp_proposal_representation"] = "mxfp4-q4-g32"
        metadata["vmodel_mtp_quant_clone_schema"] = MANIFEST_SCHEMA
        index["metadata"] = metadata
        index_bytes = _canonical_json(index)
        (temporary / INDEX_NAME).write_bytes(index_bytes)
        record["output_index_sha256"] = _sha256_bytes(index_bytes)
        record["enabled_by_default"] = False
        record["target_authoritative_required"] = True
        (temporary / MANIFEST_NAME).write_bytes(_canonical_json(record))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def plan_fast_alias(
    source: str | Path, output: str | Path, target: str | Path,
) -> dict:
    """Validate a zero-copy alias of the existing two-device target overlay."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("fast output must be a separate sibling directory")
    manifest_path = source / FAST_MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("fast-tier manifest must be a nonempty tensor map")
    clone_manifest_path = target / MANIFEST_NAME
    clone_manifest_bytes = clone_manifest_path.read_bytes()
    clone_manifest = json.loads(clone_manifest_bytes)
    if not (
        isinstance(clone_manifest, dict)
        and clone_manifest.get("schema") == MANIFEST_SCHEMA
        and Path(clone_manifest.get("output", "")).expanduser().resolve()
        == target
        and Path(clone_manifest.get("source", "")).name == source.name
    ):
        raise ValueError("fast-tier source is not bound to the target clone")
    target_index_bytes = (target / INDEX_NAME).read_bytes()
    if _sha256_bytes(target_index_bytes) != clone_manifest.get(
            "output_index_sha256"):
        raise ValueError("target clone index does not match its manifest")
    files: set[str] = set()
    mtp_names: list[str] = []
    total_bytes = 0
    for name, entry in manifest.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("invalid fast-tier tensor entry")
        filename = entry.get("file")
        if not (
            isinstance(filename, str)
            and Path(filename).name == filename
            and isinstance(entry.get("offset"), int)
            and isinstance(entry.get("nbytes"), int)
            and entry["offset"] >= 0
            and entry["nbytes"] >= 0
            and isinstance(entry.get("dtype"), str)
            and isinstance(entry.get("shape"), list)
        ):
            raise ValueError(f"invalid fast-tier metadata: {name}")
        path = source / filename
        if not path.is_file() or entry["offset"] + entry["nbytes"] > path.stat().st_size:
            raise ValueError(f"fast-tier tensor extent is unavailable: {name}")
        files.add(filename)
        total_bytes += entry["nbytes"]
        if name.startswith("mtp."):
            mtp_names.append(name)
            expected = EXPECTED_MTP_SPECS.get(name)
            actual = (
                entry["dtype"], tuple(entry["shape"]), entry["nbytes"])
            if expected is None or actual != expected:
                raise ValueError(f"fast-tier MTP schema mismatch: {name}")
    if not mtp_names:
        raise ValueError("fast-tier alias has no packed MTP tensors")
    return {
        "schema": FAST_ALIAS_SCHEMA,
        "source": str(source),
        "output": str(output),
        "target": str(target),
        "target_model": target.name,
        "target_index_sha256": _sha256_bytes(target_index_bytes),
        "target_clone_manifest_sha256": _sha256_bytes(clone_manifest_bytes),
        "target_body_mapping_sha256": clone_manifest[
            "target_body_mapping_sha256"],
        "source_manifest_sha256": _sha256_bytes(manifest_bytes),
        "mapped_tensors": len(manifest),
        "mapped_logical_bytes": total_bytes,
        "files": sorted(files),
        "mtp_tensors": sorted(mtp_names),
        "enabled_by_default": False,
    }


def build_fast_alias(
    source: str | Path, output: str | Path, target: str | Path,
) -> dict:
    record = plan_fast_alias(source, output, target)
    source = Path(record["source"])
    output = Path(record["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp-"))
    try:
        for filename in record["files"]:
            (temporary / filename).symlink_to((source / filename).resolve())
        manifest_bytes = (source / FAST_MANIFEST_NAME).read_bytes()
        (temporary / FAST_MANIFEST_NAME).write_bytes(manifest_bytes)
        (temporary / FAST_ALIAS_MANIFEST_NAME).write_bytes(
            _canonical_json(record))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "build", "plan-fast", "build-fast"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    operation = {
        "plan": plan,
        "build": build,
        "plan-fast": plan_fast_alias,
        "build-fast": build_fast_alias,
    }[args.command]
    if args.command.endswith("fast"):
        if args.target is None:
            parser.error("--target is required for fast-tier alias commands")
        result = operation(args.source, args.output, args.target)
    else:
        if args.target is not None:
            parser.error("--target applies only to fast-tier alias commands")
        result = operation(args.source, args.output)
    payload = _canonical_json(result)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
