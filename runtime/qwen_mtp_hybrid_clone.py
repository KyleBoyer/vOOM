#!/usr/bin/env python3
"""Create a zero-copy Qwen MTP clone with exact attention and packed MLP.

The served target body is unchanged.  The proposal-only MTP block reads its
recurrent input projection and attention matrices from the released BF16
sidecar, while the three dense SwiGLU matrices reuse the all-MXFP4 artifact's
existing packed tensors.  Every proposal is still verified by the unchanged
target before commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .qwen_mtp_quant_clone import (
    HYBRID_MANIFEST_SCHEMA,
    INDEX_NAME,
    MANIFEST_NAME,
    _canonical_json,
    _header,
    _mtp_physical,
    _sha256_bytes,
)


REPRESENTATION = "hybrid-bf16-attn-mxfp4-mlp"
MLP_MATRICES = frozenset({
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
})
EXPECTED_BF16_SPECS = {
    "mtp.fc.weight": ((5120, 10240), 104_857_600),
    "mtp.layers.0.input_layernorm.weight": ((5120,), 10_240),
    "mtp.layers.0.mlp.down_proj.weight": ((5120, 17408), 178_257_920),
    "mtp.layers.0.mlp.gate_proj.weight": ((17408, 5120), 178_257_920),
    "mtp.layers.0.mlp.up_proj.weight": ((17408, 5120), 178_257_920),
    "mtp.layers.0.post_attention_layernorm.weight": ((5120,), 10_240),
    "mtp.layers.0.self_attn.k_norm.weight": ((256,), 512),
    "mtp.layers.0.self_attn.k_proj.weight": ((1024, 5120), 10_485_760),
    "mtp.layers.0.self_attn.o_proj.weight": ((5120, 6144), 62_914_560),
    "mtp.layers.0.self_attn.q_norm.weight": ((256,), 512),
    "mtp.layers.0.self_attn.q_proj.weight": ((12288, 5120), 125_829_120),
    "mtp.layers.0.self_attn.v_proj.weight": ((1024, 5120), 10_485_760),
    "mtp.norm.weight": ((5120,), 10_240),
    "mtp.pre_fc_norm_embedding.weight": ((5120,), 10_240),
    "mtp.pre_fc_norm_hidden.weight": ((5120,), 10_240),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_specs(path: Path) -> dict[str, dict]:
    specs = {}
    for name, entry in _header(path).items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"invalid released-BF16 MTP tensor: {name}")
        offsets = entry.get("data_offsets")
        shape = entry.get("shape")
        if not (
            entry.get("dtype") == "BF16"
            and isinstance(shape, list)
            and isinstance(offsets, list)
            and len(offsets) == 2
        ):
            raise ValueError(f"invalid released-BF16 MTP layout: {name}")
        specs[name] = {
            "dtype": "BF16",
            "shape": [int(value) for value in shape],
            "bytes": int(offsets[1]) - int(offsets[0]),
        }
    actual = {
        name: (tuple(spec["shape"]), int(spec["bytes"]))
        for name, spec in specs.items()
    }
    if actual != EXPECTED_BF16_SPECS:
        missing = sorted(set(EXPECTED_BF16_SPECS) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_BF16_SPECS))
        mismatched = sorted(
            name for name in set(actual) & set(EXPECTED_BF16_SPECS)
            if actual[name] != EXPECTED_BF16_SPECS[name]
        )
        raise ValueError(
            "released-BF16 MTP schema mismatch: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}")
    return specs


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
    if not isinstance(metadata, dict):
        raise ValueError("source index metadata must be an object")
    sidecar_name = metadata.get("mtplx_mtp_sidecar")
    sidecar_sha = metadata.get("mtplx_mtp_sidecar_sha256")
    if not (
        isinstance(sidecar_name, str)
        and Path(sidecar_name).name == sidecar_name
        and sidecar_name.endswith(".safetensors")
        and isinstance(sidecar_sha, str)
        and len(sidecar_sha) == 64
    ):
        raise ValueError("source is not a hashed released-BF16 MTP target")
    sidecar_path = source / sidecar_name
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"released-BF16 MTP sidecar is missing: {sidecar_path}")
    bf16_specs = _bf16_specs(sidecar_path)
    if _sha256_file(sidecar_path) != sidecar_sha:
        raise ValueError("released-BF16 MTP sidecar SHA-256 mismatch")

    physical, quant_specs = _mtp_physical(source)
    quantized_mapping = {}
    for matrix in sorted(MLP_MATRICES):
        scale = matrix.removesuffix(".weight") + ".scales"
        if matrix not in physical or scale not in physical:
            raise ValueError(f"packed MTP MLP pair is missing: {matrix}")
        quantized_mapping[matrix] = physical[matrix]
        quantized_mapping[scale] = physical[scale]
    plain_names = sorted(set(EXPECTED_BF16_SPECS) - MLP_MATRICES)
    hybrid_mapping = dict(quantized_mapping)
    hybrid_mapping.update({name: sidecar_name for name in plain_names})
    target_body = {
        name: shard for name, shard in index["weight_map"].items()
        if not name.startswith("mtp.")
    }
    packed_bytes = sum(
        int(quant_specs[name]["bytes"]) for name in quantized_mapping)
    plain_bytes = sum(int(bf16_specs[name]["bytes"]) for name in plain_names)
    return {
        "schema": HYBRID_MANIFEST_SCHEMA,
        "source": str(source),
        "output": str(output),
        "source_index_sha256": _sha256_bytes(index_bytes),
        "source_sidecar": sidecar_name,
        "source_sidecar_sha256": sidecar_sha,
        "proposal_representation": REPRESENTATION,
        "mtp_quantized_matrices": sorted(MLP_MATRICES),
        "mtp_plain_names": plain_names,
        "mtp_mapping": dict(sorted(hybrid_mapping.items())),
        "mtp_packed_bytes": packed_bytes,
        "mtp_plain_bytes": plain_bytes,
        "mtp_payload_bytes": packed_bytes + plain_bytes,
        "mtp_released_bf16_bytes": sum(
            int(spec["bytes"]) for spec in bf16_specs.values()),
        "target_body_mapping_sha256": _sha256_bytes(_canonical_json(
            dict(sorted(target_body.items())))),
        "enabled_by_default": False,
        "target_authoritative_required": True,
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
            if path.name in {INDEX_NAME, MANIFEST_NAME}:
                continue
            if path.is_file():
                (temporary / path.name).symlink_to(path.resolve())
        index = json.loads((source / INDEX_NAME).read_text())
        index["weight_map"] = {
            name: shard for name, shard in index["weight_map"].items()
            if not name.startswith("mtp.")
        }
        index["weight_map"].update(record["mtp_mapping"])
        metadata = dict(index.get("metadata") or {})
        for name in tuple(metadata):
            if name.startswith("mtplx_mtp_"):
                metadata.pop(name)
        metadata.update({
            "vmodel_mtp_proposal_representation": REPRESENTATION,
            "vmodel_mtp_proposal_plain_sidecar": record["source_sidecar"],
            "vmodel_mtp_proposal_plain_sidecar_sha256": (
                record["source_sidecar_sha256"]),
            "vmodel_mtp_proposal_plain_names": record["mtp_plain_names"],
            "vmodel_mtp_hybrid_clone_schema": HYBRID_MANIFEST_SCHEMA,
        })
        index["metadata"] = metadata
        index_bytes = _canonical_json(index)
        (temporary / INDEX_NAME).write_bytes(index_bytes)
        record["output_index_sha256"] = _sha256_bytes(index_bytes)
        (temporary / MANIFEST_NAME).write_bytes(_canonical_json(record))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "build"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = (plan if args.command == "plan" else build)(
        args.source, args.output)
    payload = _canonical_json(result)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
