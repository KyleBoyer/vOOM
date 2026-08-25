"""Fail-closed refusal-direction artifacts for the DFlash2 draft.

The target verifier remains authoritative: projecting the draft can change
proposal quality and acceptance, never the served target distribution.  This
module deliberately accepts only a measured rank-1 output-space direction,
bound to the exact target config and DFlash2 revision.  It does not manufacture
a direction from token embeddings or refusal keywords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "voom.dflash2-ablation-direction.v1"
MANIFEST_NAME = "manifest.json"
DIRECTION_NAME = "direction.npy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_direction(value: np.ndarray, hidden_size: int) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float32)
    if direction.shape != (hidden_size,):
        raise ValueError(
            f"DFlash2 ablation direction must have shape ({hidden_size},)")
    if not np.isfinite(direction).all():
        raise ValueError("DFlash2 ablation direction contains non-finite values")
    norm = float(np.linalg.norm(direction.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("DFlash2 ablation direction has zero/invalid norm")
    direction = np.ascontiguousarray(direction / np.float32(norm))
    # The projection is sign-invariant.  Canonicalize it so builds are stable.
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return np.ascontiguousarray(direction, dtype=np.float32)


def contrastive_direction(
    harmful: np.ndarray,
    harmless: np.ndarray,
    *,
    project_harmless: bool = False,
) -> np.ndarray:
    """Build the standard normalized harmful-minus-harmless mean direction."""
    harmful = np.asarray(harmful, dtype=np.float32)
    harmless = np.asarray(harmless, dtype=np.float32)
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise ValueError("contrastive activations must both be rank 2")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("harmful/harmless activation widths differ")
    if min(harmful.shape[0], harmless.shape[0]) < 2:
        raise ValueError("contrastive direction needs at least two rows per class")
    if not np.isfinite(harmful).all() or not np.isfinite(harmless).all():
        raise ValueError("contrastive activations contain non-finite values")
    harmless_mean = harmless.astype(np.float64).mean(axis=0)
    direction = harmful.astype(np.float64).mean(axis=0) - harmless_mean
    if project_harmless:
        harmless_norm = float(np.linalg.norm(harmless_mean))
        if harmless_norm > 1e-8:
            harmless_axis = harmless_mean / harmless_norm
            direction -= float(direction @ harmless_axis) * harmless_axis
    return _canonical_direction(direction, harmful.shape[1])


def _read_f32_safetensors_vectors(path: Path) -> dict[str, np.ndarray]:
    """Read only one-dimensional F32 tensors from a small direction artifact."""
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("direction safetensors header is truncated")
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 2 <= header_length <= 16 * 1024 * 1024:
            raise ValueError("direction safetensors header length is invalid")
        header = json.loads(stream.read(header_length))
        data_start = 8 + header_length
        vectors = {}
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if meta.get("dtype") != "F32" or len(meta.get("shape", ())) != 1:
                continue
            start, end = map(int, meta.get("data_offsets", ()))
            width = int(meta["shape"][0])
            if end - start != width * 4 or start < 0 or end < start:
                raise ValueError(f"direction tensor {name} has invalid offsets")
            stream.seek(data_start + start)
            payload = stream.read(end - start)
            if len(payload) != end - start:
                raise ValueError(f"direction tensor {name} is truncated")
            vectors[name] = np.frombuffer(payload, dtype=np.float32).copy()
    if not vectors:
        raise ValueError("direction safetensors contains no F32 vectors")
    return vectors


def coherent_mean_direction(
    vectors: dict[str, np.ndarray],
    *,
    hidden_size: int,
    minimum_cosine: float = 0.9,
) -> tuple[np.ndarray, float, int]:
    """Sign-align and average a globally coherent set of rank-1 axes."""
    usable = [
        _canonical_direction(value, hidden_size)
        for name, value in sorted(vectors.items())
        if not name.startswith("__") and value.shape == (hidden_size,)
    ]
    if not usable:
        raise ValueError("direction artifact has no matching hidden-width vectors")
    reference = usable[0]
    aligned = [
        value if float(reference @ value) >= 0 else -value
        for value in usable
    ]
    dots = [
        abs(float(aligned[left] @ aligned[right]))
        for left in range(len(aligned))
        for right in range(left + 1, len(aligned))
    ]
    minimum = min(dots) if dots else 1.0
    if minimum < minimum_cosine:
        raise ValueError(
            f"rank-1 directions are not one global axis: min cosine {minimum:.6f}")
    mean = np.mean(np.stack(aligned).astype(np.float64), axis=0)
    return _canonical_direction(mean, hidden_size), minimum, len(aligned)


@dataclass(frozen=True)
class DirectionArtifact:
    direction: np.ndarray
    fingerprint: str
    manifest: dict[str, Any]


def write_artifact(
    output_dir: str | Path,
    direction: np.ndarray,
    *,
    target_config: str | Path,
    draft_revision: str,
    source: dict[str, Any],
    method: str,
) -> DirectionArtifact:
    """Atomically write one unit vector plus its target/source binding."""
    output = Path(output_dir)
    target_config = Path(target_config)
    if not target_config.is_file():
        raise ValueError("DFlash2 ablation target config does not exist")
    direction = _canonical_direction(direction, int(direction.shape[0]))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        np.save(temporary / DIRECTION_NAME, direction, allow_pickle=False)
        direction_hash = _sha256(temporary / DIRECTION_NAME)
        manifest = {
            "schema": SCHEMA,
            "enabled_by_default": False,
            "method": method,
            "hidden_size": int(direction.shape[0]),
            "dtype": "float32",
            "norm": float(np.linalg.norm(direction.astype(np.float64))),
            "direction_sha256": direction_hash,
            "target_config_sha256": _sha256(target_config),
            "draft_revision": draft_revision,
            "source": source,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if output.exists():
            raise FileExistsError(f"refusing to replace direction artifact {output}")
        os.replace(temporary, output)
        return DirectionArtifact(direction, direction_hash, manifest)
    except Exception:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise


def load_artifact(
    artifact_dir: str | Path,
    *,
    target_config: str | Path,
    draft_revision: str,
    hidden_size: int,
) -> DirectionArtifact:
    """Validate and load an exact-target-bound direction, or fail closed."""
    root = Path(artifact_dir)
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("DFlash2 ablation direction schema mismatch")
    if manifest.get("enabled_by_default") is not False:
        raise ValueError("DFlash2 ablation artifact must remain default-off")
    if manifest.get("hidden_size") != hidden_size:
        raise ValueError("DFlash2 ablation hidden width mismatch")
    if manifest.get("draft_revision") != draft_revision:
        raise ValueError("DFlash2 ablation draft revision mismatch")
    target_config = Path(target_config)
    if manifest.get("target_config_sha256") != _sha256(target_config):
        raise ValueError("DFlash2 ablation target config fingerprint mismatch")
    direction_path = root / DIRECTION_NAME
    fingerprint = _sha256(direction_path)
    if manifest.get("direction_sha256") != fingerprint:
        raise ValueError("DFlash2 ablation direction hash mismatch")
    direction = np.load(direction_path, allow_pickle=False)
    direction = _canonical_direction(direction, hidden_size)
    if not math.isclose(
        float(np.linalg.norm(direction.astype(np.float64))),
        float(manifest.get("norm", -1)),
        rel_tol=0,
        abs_tol=2e-6,
    ):
        raise ValueError("DFlash2 ablation direction norm mismatch")
    return DirectionArtifact(direction, fingerprint, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("from-safetensors")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--target-config", type=Path, required=True)
    convert.add_argument("--draft-revision", required=True)
    convert.add_argument("--source-repository", required=True)
    convert.add_argument("--source-revision", required=True)
    convert.add_argument("--hidden-size", type=int, default=5120)
    args = parser.parse_args()
    vectors = _read_f32_safetensors_vectors(args.input)
    direction, minimum, count = coherent_mean_direction(
        vectors, hidden_size=args.hidden_size)
    artifact = write_artifact(
        args.output,
        direction,
        target_config=args.target_config,
        draft_revision=args.draft_revision,
        source={
            "kind": "rank1-weight-delta-directions",
            "repository": args.source_repository,
            "revision": args.source_revision,
            "artifact_sha256": _sha256(args.input),
            "vectors": count,
            "minimum_pairwise_abs_cosine": minimum,
        },
        method="coherent-mean-rank1-output-direction",
    )
    print(json.dumps(artifact.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
