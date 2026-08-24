"""Build a compact, deterministic MLX DSpark sidecar checkpoint.

The target still verifies every proposal. With a correct target verifier,
quantizing this draft-only model can change acceptance and speed, never the
target distribution. The builder is
deliberately local-only: Hub acquisition/revision pinning stays an explicit
operator step, while this module hashes the pinned payload before conversion
and commits the output directory only after every tensor is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .dspark import DSparkConfig, DSparkDrafter


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _plain_parameters(model) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def build_sidecar(
    source: str | Path,
    output: str | Path,
    *,
    expected_source_sha256: str,
    source_repo: str = "",
    source_revision: str = "",
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> dict:
    source = Path(source).resolve()
    output = Path(output).resolve()
    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"incomplete DSpark source checkpoint: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DSpark sidecar: {output}")
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError("DSpark affine bits must be one of 2, 3, 4, 6, 8")
    if group_size <= 0:
        raise ValueError("DSpark group_size must be positive")
    expected = expected_source_sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("expected_source_sha256 must be 64 lowercase hex digits")
    actual_source_sha256 = _sha256(weights_path)
    if actual_source_sha256 != expected:
        raise ValueError(
            "DSpark source hash mismatch: "
            f"expected {expected}, got {actual_source_sha256}")

    cfg = DSparkConfig.from_json(config_path)
    model = DSparkDrafter(cfg)
    source_weights = dict(mx.load(str(weights_path)))
    model_names = set(_plain_parameters(model))
    source_names = set(source_weights)
    if model_names != source_names:
        raise ValueError(
            "DSpark source/model tensor mismatch: "
            f"missing={sorted(model_names - source_names)[:8]} "
            f"unexpected={sorted(source_names - model_names)[:8]}")
    model.load_weights(list(source_weights.items()))
    del source_weights

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode)
    quantized = _plain_parameters(model)
    mx.eval(*quantized.values())

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.building-", dir=output.parent))
    try:
        quant_path = tmp_dir / "model.safetensors"
        mx.save_safetensors(str(quant_path), quantized, metadata={
            "format": "mlx",
            "vmodel_kind": "target-verified-dspark-sidecar",
            "source_sha256": actual_source_sha256,
        })
        output_sha256 = _sha256(quant_path)
        raw_config = json.loads(config_path.read_text())
        raw_config["quantization"] = {
            "group_size": group_size,
            "bits": bits,
            "mode": mode,
        }
        raw_config["vmodel_sidecar"] = {
            "schema": "voom.dspark-sidecar.v1",
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_sha256": actual_source_sha256,
            "target_verified": True,
        }
        (tmp_dir / "config.json").write_text(
            json.dumps(raw_config, indent=2, sort_keys=True) + "\n")
        for name in ("README.md", "LICENSE", "LICENSE.txt"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, tmp_dir / name)
        manifest = {
            "schema": "voom.dspark-sidecar.v1",
            "source": str(source),
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_sha256": actual_source_sha256,
            "source_bytes": weights_path.stat().st_size,
            "output_sha256": output_sha256,
            "output_bytes": quant_path.stat().st_size,
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
            "tensor_count": len(quantized),
            "target_model_type": cfg.target_model_type,
            "target_verified": True,
        }
        (tmp_dir / "sidecar_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_dir, output)
        return manifest
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-repo", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    args = parser.parse_args()
    result = build_sidecar(
        args.source,
        args.output,
        expected_source_sha256=args.expected_source_sha256,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
