"""Bounded BF16 rank-one edit probe for a local Qwen3.8 checkpoint.

The clean checkpoint stays remote: only explicitly named residual-writer
tensors are fetched with HTTP byte ranges from a pinned Hub revision.  The
ablated checkpoint is read locally.  No complete shard or checkpoint is
downloaded, and every fetched tensor is hashed into the output provenance.

The output direction is intended for proposal-model experiments.  It does not
assert that a different architecture has the same ideal per-module strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from .dflash2_ablation import coherent_mean_direction, write_artifact


MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_TENSOR_BYTES = 256 * 1024 * 1024
DEFAULT_MODULES = (
    "model.language_model.layers.16.linear_attn.out_proj",
    "model.language_model.layers.19.self_attn.o_proj",
    "model.language_model.layers.20.linear_attn.out_proj",
    "model.language_model.layers.63.self_attn.o_proj",
)
DEFAULT_UNEDITED_MODULES = (
    "model.language_model.layers.14.linear_attn.out_proj",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bf16_to_f32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    expected = math.prod(shape) * 2
    if len(payload) != expected:
        raise ValueError("BF16 tensor payload length does not match shape")
    values = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return np.ascontiguousarray((values << 16).view(np.float32).reshape(shape))


def _validate_tensor_meta(name: str, meta: Any) -> tuple[tuple[int, ...], int, int]:
    if not isinstance(meta, dict) or meta.get("dtype") != "BF16":
        raise ValueError(f"{name} is not a BF16 safetensors tensor")
    shape = tuple(int(value) for value in meta.get("shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{name} is not a positive rank-2 tensor")
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{name} has invalid safetensors offsets")
    start, end = map(int, offsets)
    size = end - start
    if start < 0 or size != math.prod(shape) * 2:
        raise ValueError(f"{name} safetensors extent does not match BF16 shape")
    if size > MAX_TENSOR_BYTES:
        raise ValueError(f"{name} exceeds the bounded tensor byte limit")
    return shape, start, end


def _read_local_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return header, 8 + length


def _http_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start or end - start + 1 > MAX_TENSOR_BYTES:
        raise ValueError("remote range exceeds the bounded request contract")
    request = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "vOOM-qwen38-rank1-probe/1",
        },
    )
    with urlopen(request, timeout=180) as response:
        payload = response.read(end - start + 2)
        if response.status != 206:
            raise ValueError(f"remote server ignored byte range for {url}")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise ValueError(f"remote server returned the wrong byte range for {url}")
    if len(payload) != end - start + 1:
        raise ValueError(f"remote byte range was truncated for {url}")
    return payload


class LocalCheckpoint:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}

    def tensor(self, name: str) -> tuple[np.ndarray, str]:
        shard = self.weight_map.get(name)
        if not shard:
            raise ValueError(f"local checkpoint omits {name}")
        if shard not in self._headers:
            self._headers[shard] = _read_local_header(self.root / shard)
        header, data_start = self._headers[shard]
        shape, start, end = _validate_tensor_meta(name, header.get(name))
        with (self.root / shard).open("rb") as stream:
            stream.seek(data_start + start)
            payload = stream.read(end - start)
        if len(payload) != end - start:
            raise ValueError(f"local tensor {name} is truncated")
        return _bf16_to_f32(payload, shape), _sha256_bytes(payload)


class RemoteCheckpoint:
    def __init__(self, repository: str, revision: str):
        if not repository or not revision:
            raise ValueError("remote repository and revision must be explicit")
        self.repository = repository
        self.revision = revision
        index_url = self._url("model.safetensors.index.json")
        with urlopen(Request(index_url, headers={
            "User-Agent": "vOOM-qwen38-rank1-probe/1",
        }), timeout=60) as response:
            index_payload = response.read(16 * 1024 * 1024 + 1)
        if len(index_payload) > 16 * 1024 * 1024:
            raise ValueError("remote checkpoint index exceeds 16 MiB")
        self.index_sha256 = _sha256_bytes(index_payload)
        self.weight_map = json.loads(index_payload)["weight_map"]
        self._headers: dict[str, tuple[dict[str, Any], int]] = {}

    def _url(self, filename: str) -> str:
        return (
            f"https://huggingface.co/{quote(self.repository, safe='/')}/resolve/"
            f"{quote(self.revision, safe='')}/{quote(filename, safe='')}"
        )

    def _header(self, shard: str) -> tuple[dict[str, Any], int]:
        if shard in self._headers:
            return self._headers[shard]
        url = self._url(shard)
        raw = _http_range(url, 0, 7)
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f"remote shard {shard} has invalid header length")
        header = json.loads(_http_range(url, 8, 8 + length - 1))
        if not isinstance(header, dict):
            raise ValueError(f"remote shard {shard} header is not an object")
        value = (header, 8 + length)
        self._headers[shard] = value
        return value

    def tensor(self, name: str) -> tuple[np.ndarray, str]:
        shard = self.weight_map.get(name)
        if not shard:
            raise ValueError(f"remote checkpoint omits {name}")
        header, data_start = self._header(shard)
        shape, start, end = _validate_tensor_meta(name, header.get(name))
        payload = _http_range(
            self._url(shard), data_start + start, data_start + end - 1)
        return _bf16_to_f32(payload, shape), _sha256_bytes(payload)


def _top_triplet(
    delta: np.ndarray,
    *,
    iterations: int = 100,
    tolerance: float = 1e-7,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    right = rng.standard_normal(delta.shape[1], dtype=np.float32)
    right /= np.linalg.norm(right)
    previous = 0.0
    for iteration in range(1, iterations + 1):
        left = delta @ right
        left_norm = float(np.linalg.norm(left))
        if not left_norm:
            raise ValueError("rank-one probe received a zero weight delta")
        left /= left_norm
        right = delta.T @ left
        singular = float(np.linalg.norm(right))
        right /= singular
        if abs(singular - previous) <= tolerance * max(singular, 1.0):
            return left, singular, right, iteration
        previous = singular
    return left, singular, right, iterations


@dataclass(frozen=True)
class ModuleProbe:
    module: str
    edited: bool
    direction: np.ndarray | None
    metrics: dict[str, Any]


def analyze_pair(
    module: str,
    base: np.ndarray,
    ablated: np.ndarray,
    *,
    base_sha256: str,
    ablated_sha256: str,
) -> ModuleProbe:
    if base.shape != ablated.shape:
        raise ValueError(f"{module} base/ablated shapes differ")
    delta = ablated - base
    delta_norm = float(np.linalg.norm(delta))
    base_norm = float(np.linalg.norm(base))
    common = {
        "module": module,
        "shape": list(base.shape),
        "base_tensor_sha256": base_sha256,
        "ablated_tensor_sha256": ablated_sha256,
    }
    if delta_norm == 0.0:
        return ModuleProbe(module, False, None, {**common, "edited": False})
    left, singular, right, iterations = _top_triplet(delta)
    rank1_energy = float(singular * singular / (delta_norm * delta_norm))
    base_row = left @ base
    base_row_norm = float(np.linalg.norm(base_row))
    if base_row_norm <= 1e-12:
        raise ValueError(f"{module} has an invalid projected base row")
    signed_inner = float(np.dot(singular * right, base_row))
    cosine = float(abs(np.dot(right, base_row)) / base_row_norm)
    coefficient = float(singular / base_row_norm)
    direction = left.astype(np.float32)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return ModuleProbe(module, True, direction, {
        **common,
        "edited": True,
        "rank1_energy": rank1_energy,
        "relative_delta_frobenius": delta_norm / base_norm,
        "lambda_effective": coefficient,
        "subtracts": signed_inner < 0,
        "cosine_projection_form": cosine,
        "power_iterations": iterations,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repository", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--ablated", type=Path, required=True)
    parser.add_argument("--ablated-revision", required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument("--module", action="append", dest="modules")
    parser.add_argument("--expect-unedited", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    modules = tuple(args.modules or DEFAULT_MODULES)
    unedited = tuple(args.expect_unedited or DEFAULT_UNEDITED_MODULES)
    if len(set(modules + unedited)) != len(modules) + len(unedited):
        raise ValueError("rank-one probe module lists overlap or contain duplicates")
    remote = RemoteCheckpoint(args.base_repository, args.base_revision)
    local = LocalCheckpoint(args.ablated)
    probes = []
    for module in modules + unedited:
        name = f"{module}.weight"
        base, base_hash = remote.tensor(name)
        ablated, ablated_hash = local.tensor(name)
        probe = analyze_pair(
            module, base, ablated,
            base_sha256=base_hash,
            ablated_sha256=ablated_hash,
        )
        probes.append(probe)
        print(json.dumps(probe.metrics, sort_keys=True), flush=True)

    edited = [probe for probe in probes if probe.module in modules and probe.edited]
    unexpectedly_clean = [
        probe.module for probe in probes
        if probe.module in modules and not probe.edited
    ]
    unexpectedly_edited = [
        probe.module for probe in probes
        if probe.module in unedited and probe.edited
    ]
    if unexpectedly_clean or unexpectedly_edited:
        raise ValueError(
            "Huihui retained/edited layer contract failed: "
            f"clean={unexpectedly_clean} edited={unexpectedly_edited}")
    if len(edited) < 2:
        raise ValueError("rank-one probe requires at least two edited modules")
    if any(
        probe.metrics["rank1_energy"] < 0.9
        or probe.metrics["cosine_projection_form"] < 0.9
        or not probe.metrics["subtracts"]
        for probe in edited
    ):
        raise ValueError("Huihui weight deltas failed rank-one projection gates")
    direction, minimum_cosine, count = coherent_mean_direction(
        {probe.module: probe.direction for probe in edited},
        hidden_size=int(edited[0].direction.shape[0]),
        minimum_cosine=0.9,
    )
    coefficients = [probe.metrics["lambda_effective"] for probe in edited]
    report = {
        "schema": "voom.qwen38-huihui-rank1-probe.v1",
        "base_repository": args.base_repository,
        "base_revision": args.base_revision,
        "base_index_sha256": remote.index_sha256,
        "ablated": str(args.ablated.resolve()),
        "ablated_revision": args.ablated_revision,
        "minimum_pairwise_abs_cosine": minimum_cosine,
        "edited_modules": count,
        "mean_lambda_effective": float(np.mean(coefficients)),
        "modules": [probe.metrics for probe in probes],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact = write_artifact(
        args.output,
        direction,
        target_config=args.target_config,
        draft_revision=args.draft_revision,
        source={
            "kind": "huihui-vs-official-bf16-rank1-weight-delta",
            "base_repository": args.base_repository,
            "base_revision": args.base_revision,
            "base_index_sha256": remote.index_sha256,
            "ablated_revision": args.ablated_revision,
            "modules": [probe.metrics for probe in edited],
            "minimum_pairwise_abs_cosine": minimum_cosine,
            "recommended_strength_mean_lambda_effective": float(
                np.mean(coefficients)),
        },
        method="huihui-bf16-weight-delta-rank1-residual-direction",
    )
    print(json.dumps(artifact.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
