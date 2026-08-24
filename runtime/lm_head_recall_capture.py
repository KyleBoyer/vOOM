"""Privacy-safe authoritative LM-head shortlist recall evidence.

The live path never persists prompts, hidden states, token IDs, text, or full
logits.  It reduces each paired approximate/exact projection at the same
hidden state to three values: the exact winner's stable approximate rank,
whether the shortlist actually contained that winner, and whether the two
full-vocabulary winners agreed.  Request metadata is restricted to coarse
shape buckets so heterogeneous serving coverage can be proved without
recording request content.

Promotion is intentionally stricter than ordinary diagnostics: K must be 64,
at least 1,000 authoritative target positions must have 100% winner inclusion,
the exact and approximate artifacts must be explicitly fingerprint-bound, and
the corpus must cover the real request-shape dimensions called out by the
project's anti-overfit policy.  Synthetic fixtures can validate this module,
but their manifest kind is permanently ineligible for promotion.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Iterable, Iterator, Mapping, Sequence


CAPTURE_SCHEMA = "voom.huihui-authoritative-head-ranks.v1"
REPORT_SCHEMA = "voom.huihui-authoritative-head-rank-gate.v1"
LIVE_CAPTURE_KIND = "live-server-authoritative-target"
SYNTHETIC_CAPTURE_KIND = "synthetic-fixture"
PRIVACY_PROFILE = "coarse-request-shape-plus-ranks-no-content-v1"

PROMOTION_K = 64
PROMOTION_MIN_POSITIONS = 1000
PROMOTION_MIN_REQUESTS = 8
PROMOTION_MIN_SHAPES = 6
CAPTURE_MAX_POSITIONS_LIMIT = 1200
CAPTURE_MAX_PER_REQUEST_LIMIT = 128
CAPTURE_MAX_PER_SHAPE = 200
CAPTURE_MAX_FILE_BYTES = 1 << 20

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_SHAPE_KEYS = frozenset({
    "prompt_tokens_bucket",
    "system_chars_bucket",
    "tool_count_bucket",
    "message_count_bucket",
    "developer",
    "streaming",
    "temperature_class",
    "constrained",
})
_MANIFEST_KEYS = frozenset({
    "schema", "record", "capture_kind", "privacy_profile",
    "exact_source_fingerprint", "approximate_artifact_fingerprint",
    "approximate_artifact_bytes", "candidate_k", "vocab",
    "max_positions", "max_positions_per_request",
    "max_positions_per_shape",
})
_OBSERVATION_KEYS = frozenset({
    "schema", "record", "request_nonce", "shape", "shape_id", "positions",
})


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket(value: int, boundaries: Sequence[tuple[int, str]], overflow: str) -> str:
    value = max(0, int(value))
    for ceiling, label in boundaries:
        if value <= ceiling:
            return label
    return overflow


def privacy_safe_request_shape(
    *,
    prompt_tokens: int,
    system_chars: int,
    tool_count: int,
    message_count: int,
    developer: bool,
    streaming: bool,
    temperature_class: str,
    constrained: bool,
) -> dict:
    """Reduce serving metadata to the only fields permitted on disk."""

    if temperature_class not in {"greedy", "stochastic"}:
        raise ValueError("temperature_class must be greedy or stochastic")
    return {
        "prompt_tokens_bucket": _bucket(
            prompt_tokens,
            ((512, "0-512"), (2048, "513-2048"),
             (8192, "2049-8192"), (32768, "8193-32768")),
            "32769+",
        ),
        "system_chars_bucket": _bucket(
            system_chars,
            ((0, "0"), (1024, "1-1024"), (8192, "1025-8192"),
             (32768, "8193-32768")),
            "32769+",
        ),
        "tool_count_bucket": _bucket(
            tool_count,
            ((0, "0"), (8, "1-8"), (32, "9-32"), (128, "33-128")),
            "129+",
        ),
        "message_count_bucket": _bucket(
            message_count,
            ((1, "0-1"), (4, "2-4"), (16, "5-16")),
            "17+",
        ),
        "developer": bool(developer),
        "streaming": bool(streaming),
        "temperature_class": temperature_class,
        "constrained": bool(constrained),
    }


def _validate_shape(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != _SHAPE_KEYS:
        raise ValueError(
            "rank capture shape must contain only the privacy-safe schema fields")
    if not isinstance(value["developer"], bool) \
            or not isinstance(value["streaming"], bool) \
            or not isinstance(value["constrained"], bool):
        raise ValueError("rank capture shape boolean fields must be booleans")
    if value["temperature_class"] not in {"greedy", "stochastic"}:
        raise ValueError("invalid rank capture temperature class")
    for key in _SHAPE_KEYS - {
            "developer", "streaming", "constrained", "temperature_class"}:
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"rank capture shape {key} must be a non-empty bucket")
    return dict(value)


def request_shape_id(shape: Mapping) -> str:
    safe = _validate_shape(dict(shape))
    return hashlib.sha256(_canonical_json(safe).encode()).hexdigest()[:16]


def quantized_lm_head_artifact_identity(
    model_dir: str | Path,
) -> dict:
    """Content-hash the on-disk approximate LM-head tensors.

    This is intentionally paid only in the explicit evidence-capture mode.
    Hashing the packed weight plus scales binds a corpus to the actual MXFP4
    projection instead of a mutable path or a header-only descriptor.
    """

    directory = Path(model_dir).expanduser().resolve()
    index_path = directory / "model.safetensors.index.json"
    descriptor_files = {}
    for descriptor_name in ("config.json", "model.safetensors.index.json"):
        descriptor_path = directory / descriptor_name
        descriptor_files[descriptor_name] = (
            _sha256_file(descriptor_path) if descriptor_path.is_file() else "")
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("LM-head artifact index has no weight_map")
    else:
        shards = sorted(directory.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError(
                "LM-head artifact requires an indexed or single-shard checkpoint")
        with shards[0].open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
        weight_map = {
            name: shards[0].name for name in header if name != "__metadata__"
        }

    required = ("lm_head.weight", "lm_head.scales")
    optional = ("lm_head.biases",)
    physical_names = [*required]
    physical_names.extend(name for name in optional if name in weight_map)
    missing = [name for name in required if name not in weight_map]
    if missing:
        raise ValueError(
            f"approximate LM head is not a standard quantized triplet: {missing}")

    headers: dict[str, tuple[int, dict]] = {}
    digest = hashlib.sha256()
    digest.update(b"voom.quantized-lm-head-content.v1\0")
    digest.update(_canonical_json(descriptor_files).encode() + b"\0")
    total_bytes = 0
    tensor_descriptors = []
    for name in physical_names:
        shard = weight_map[name]
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"unsafe LM-head shard path {shard!r}")
        if shard not in headers:
            path = directory / shard
            with path.open("rb") as handle:
                raw_size = handle.read(8)
                if len(raw_size) != 8:
                    raise ValueError(f"invalid safetensors header in {path}")
                header_size = int.from_bytes(raw_size, "little")
                header = json.loads(handle.read(header_size))
            headers[shard] = (8 + header_size, header)
        data_start, header = headers[shard]
        meta = header.get(name)
        if not isinstance(meta, dict):
            raise ValueError(f"LM-head shard {shard} has no tensor {name!r}")
        try:
            start, end = (int(item) for item in meta["data_offsets"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor extent for {name!r}") from error
        if start < 0 or end <= start:
            raise ValueError(f"invalid tensor extent for {name!r}")
        descriptor = {
            "name": name,
            "dtype": str(meta.get("dtype", "")),
            "shape": [int(item) for item in meta.get("shape", ())],
            "bytes": end - start,
        }
        digest.update(_canonical_json(descriptor).encode() + b"\0")
        fd = os.open(directory / shard, os.O_RDONLY)
        try:
            offset = data_start + start
            remaining = end - start
            while remaining:
                chunk = os.pread(fd, min(8 << 20, remaining), offset)
                if not chunk:
                    raise IOError(
                        f"short LM-head artifact read for {name!r} at {offset}")
                digest.update(chunk)
                offset += len(chunk)
                remaining -= len(chunk)
        finally:
            os.close(fd)
        total_bytes += end - start
        tensor_descriptors.append(descriptor)
    return {
        "fingerprint": digest.hexdigest(),
        "bytes": total_bytes,
        "tensors": tensor_descriptors,
        "descriptor_sha256": descriptor_files,
    }


class AuthoritativeRankCapture:
    """Bounded append-only JSONL writer for ranks-only live evidence."""

    def __init__(
        self,
        path: str | Path,
        *,
        exact_source_fingerprint: str,
        approximate_artifact_fingerprint: str,
        approximate_artifact_bytes: int,
        candidates: int,
        vocab: int,
        max_positions: int = CAPTURE_MAX_POSITIONS_LIMIT,
        max_positions_per_request: int = CAPTURE_MAX_PER_REQUEST_LIMIT,
        capture_kind: str = LIVE_CAPTURE_KIND,
    ):
        if not _HEX64.fullmatch(exact_source_fingerprint):
            raise ValueError("rank capture requires an exact-source fingerprint")
        if not _HEX64.fullmatch(approximate_artifact_fingerprint):
            raise ValueError("rank capture requires an approximate-artifact fingerprint")
        if isinstance(approximate_artifact_bytes, bool) \
                or int(approximate_artifact_bytes) <= 0:
            raise ValueError("rank capture requires positive approximate bytes")
        if candidates != PROMOTION_K:
            raise ValueError(f"promotion capture requires K={PROMOTION_K}")
        if vocab <= candidates:
            raise ValueError("rank capture vocabulary must exceed K")
        if not PROMOTION_MIN_POSITIONS <= max_positions \
                <= CAPTURE_MAX_POSITIONS_LIMIT:
            raise ValueError(
                f"rank capture max_positions must be in "
                f"[{PROMOTION_MIN_POSITIONS}, {CAPTURE_MAX_POSITIONS_LIMIT}]")
        if not 1 <= max_positions_per_request \
                <= CAPTURE_MAX_PER_REQUEST_LIMIT:
            raise ValueError(
                "rank capture max_positions_per_request must be in "
                f"[1, {CAPTURE_MAX_PER_REQUEST_LIMIT}]")
        if capture_kind not in {LIVE_CAPTURE_KIND, SYNTHETIC_CAPTURE_KIND}:
            raise ValueError("unsupported rank capture kind")

        self.path = Path(path).expanduser().resolve()
        self.manifest = {
            "schema": CAPTURE_SCHEMA,
            "record": "manifest",
            "capture_kind": capture_kind,
            "privacy_profile": PRIVACY_PROFILE,
            "exact_source_fingerprint": exact_source_fingerprint,
            "approximate_artifact_fingerprint": approximate_artifact_fingerprint,
            "approximate_artifact_bytes": int(approximate_artifact_bytes),
            "candidate_k": int(candidates),
            "vocab": int(vocab),
            "max_positions": int(max_positions),
            "max_positions_per_request": int(max_positions_per_request),
            "max_positions_per_shape": CAPTURE_MAX_PER_SHAPE,
        }
        self._total_positions = 0
        self._shape_positions: dict[str, int] = {}
        self._request_nonce: str | None = None
        self._request_shape: dict | None = None
        self._request_positions = 0
        self._open_or_validate()

    def _open_or_validate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            payload = (_canonical_json(self.manifest) + "\n").encode()
            fd = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            return
        records = _read_capture_file(self.path)
        if not records or records[0] != self.manifest:
            raise ValueError(
                "existing rank capture manifest does not match this runtime")
        self._total_positions = sum(
            len(record["positions"])
            for record in records[1:]
        )
        for record in records[1:]:
            shape_id = record["shape_id"]
            self._shape_positions[shape_id] = (
                self._shape_positions.get(shape_id, 0)
                + len(record["positions"]))

    @contextmanager
    def request(self, shape: Mapping) -> Iterator["AuthoritativeRankCapture"]:
        if self._request_nonce is not None:
            raise RuntimeError("rank capture request contexts cannot be nested")
        self._request_nonce = secrets.token_hex(16)
        self._request_shape = _validate_shape(dict(shape))
        self._request_positions = 0
        try:
            yield self
        finally:
            self._request_nonce = None
            self._request_shape = None
            self._request_positions = 0

    @property
    def active(self) -> bool:
        return self._request_nonce is not None

    @property
    def remaining(self) -> int:
        if not self.active:
            return 0
        shape_id = request_shape_id(self._request_shape or {})
        return max(0, min(
            self.manifest["max_positions"] - self._total_positions,
            self.manifest["max_positions_per_request"] - self._request_positions,
            self.manifest["max_positions_per_shape"]
            - self._shape_positions.get(shape_id, 0),
        ))

    def record(
        self,
        approximate_ranks: Iterable[int],
        candidate_hits: Iterable[bool],
        top1_agreements: Iterable[bool],
    ) -> int:
        if not self.active or self.remaining <= 0:
            return 0
        ranks = [int(value) for value in approximate_ranks]
        hits = [bool(value) for value in candidate_hits]
        agreements = [bool(value) for value in top1_agreements]
        if not (len(ranks) == len(hits) == len(agreements)):
            raise ValueError("rank capture observation lengths differ")
        take = min(len(ranks), self.remaining)
        if take <= 0:
            return 0
        if any(rank <= 0 or rank > self.manifest["vocab"] for rank in ranks[:take]):
            raise ValueError("approximate winner rank is outside the vocabulary")
        shape = dict(self._request_shape or {})
        observation = {
            "schema": CAPTURE_SCHEMA,
            "record": "observation",
            "request_nonce": self._request_nonce,
            "shape": shape,
            "shape_id": request_shape_id(shape),
            # Compact positional tuples: [stable approximate rank,
            # actual shortlist inclusion, full-vocabulary top-1 agreement].
            "positions": [
                [ranks[index], int(hits[index]), int(agreements[index])]
                for index in range(take)
            ],
        }
        payload = (_canonical_json(observation) + "\n").encode()
        if self.path.stat().st_size + len(payload) > CAPTURE_MAX_FILE_BYTES:
            raise RuntimeError("rank capture exceeded its privacy/storage bound")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise IOError("short append to rank capture")
            os.fsync(fd)
        finally:
            os.close(fd)
        self._total_positions += take
        self._request_positions += take
        shape_id = observation["shape_id"]
        self._shape_positions[shape_id] = (
            self._shape_positions.get(shape_id, 0) + take)
        return take

    def telemetry_snapshot(self) -> dict[str, int]:
        return {
            "positions": int(self._total_positions),
            "remaining": max(
                0, int(self.manifest["max_positions"]) - self._total_positions),
        }


def _read_capture_file(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("rank capture permissions must be exactly 0600")
    if path.stat().st_size > CAPTURE_MAX_FILE_BYTES:
        raise ValueError(f"rank capture exceeds {CAPTURE_MAX_FILE_BYTES} bytes")
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"rank capture {path}:{line_number} is truncated")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid rank capture JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"rank capture {path}:{line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"empty rank capture {path}")
    manifest = records[0]
    if set(manifest) != _MANIFEST_KEYS \
            or manifest.get("schema") != CAPTURE_SCHEMA \
            or manifest.get("record") != "manifest":
        raise ValueError(f"invalid rank capture manifest in {path}")
    if manifest.get("privacy_profile") != PRIVACY_PROFILE:
        raise ValueError("rank capture privacy profile is not recognized")
    if manifest.get("capture_kind") not in {
            LIVE_CAPTURE_KIND, SYNTHETIC_CAPTURE_KIND}:
        raise ValueError("rank capture kind is not recognized")
    for key in ("exact_source_fingerprint", "approximate_artifact_fingerprint"):
        if not _HEX64.fullmatch(str(manifest.get(key, ""))):
            raise ValueError(f"rank capture manifest has invalid {key}")
    candidate_k = manifest.get("candidate_k")
    vocab = manifest.get("vocab")
    maximum = manifest.get("max_positions")
    per_request = manifest.get("max_positions_per_request")
    per_shape = manifest.get("max_positions_per_shape")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) \
            or candidate_k <= 0:
        raise ValueError("invalid rank capture candidate K")
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab <= candidate_k:
        raise ValueError("invalid rank capture vocabulary")
    if isinstance(maximum, bool) or not isinstance(maximum, int) \
            or not 1 <= maximum <= CAPTURE_MAX_POSITIONS_LIMIT:
        raise ValueError("invalid rank capture position bound")
    if isinstance(per_request, bool) or not isinstance(per_request, int) \
            or not 1 <= per_request <= CAPTURE_MAX_PER_REQUEST_LIMIT:
        raise ValueError("invalid rank capture per-request bound")
    if per_shape != CAPTURE_MAX_PER_SHAPE:
        raise ValueError("invalid rank capture per-shape bound")
    if isinstance(manifest.get("approximate_artifact_bytes"), bool) \
            or not isinstance(manifest.get("approximate_artifact_bytes"), int) \
            or manifest["approximate_artifact_bytes"] <= 0:
        raise ValueError("invalid approximate LM-head artifact byte count")

    total = 0
    by_request: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for line_number, record in enumerate(records[1:], 2):
        if set(record) != _OBSERVATION_KEYS \
                or record.get("schema") != CAPTURE_SCHEMA \
                or record.get("record") != "observation":
            raise ValueError(f"invalid rank observation at {path}:{line_number}")
        nonce = record.get("request_nonce")
        if not isinstance(nonce, str) or not _HEX32.fullmatch(nonce):
            raise ValueError(f"invalid request nonce at {path}:{line_number}")
        shape = _validate_shape(record.get("shape"))
        if record.get("shape_id") != request_shape_id(shape):
            raise ValueError(f"rank capture shape digest mismatch at {path}:{line_number}")
        positions = record.get("positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"empty rank observation at {path}:{line_number}")
        for value in positions:
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"invalid rank tuple at {path}:{line_number}")
            rank, hit, agreement = value
            if isinstance(rank, bool) or not isinstance(rank, int) \
                    or not 1 <= rank <= vocab \
                    or isinstance(hit, bool) or not isinstance(hit, int) \
                    or hit not in (0, 1) \
                    or isinstance(agreement, bool) \
                    or not isinstance(agreement, int) \
                    or agreement not in (0, 1):
                raise ValueError(f"invalid rank tuple at {path}:{line_number}")
        total += len(positions)
        by_request[nonce] = by_request.get(nonce, 0) + len(positions)
        shape_id = record["shape_id"]
        by_shape[shape_id] = by_shape.get(shape_id, 0) + len(positions)
        if by_request[nonce] > per_request:
            raise ValueError(f"rank capture request exceeds its bound in {path}")
        if by_shape[shape_id] > per_shape:
            raise ValueError(f"rank capture shape exceeds its bound in {path}")
    if total > maximum:
        raise ValueError(f"rank capture exceeds its manifest bound in {path}")
    return records


def evaluate_rank_captures(
    paths: Iterable[str | Path],
    *,
    expected_exact_fingerprint: str = "",
    expected_approximate_fingerprint: str = "",
) -> dict:
    """Evaluate one or more bounded artifacts under the fixed promotion gate."""

    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not resolved:
        raise ValueError("provide at least one rank capture")
    loaded = [(path, _read_capture_file(path)) for path in resolved]
    manifests = [records[0] for _path, records in loaded]
    identity_fields = (
        "exact_source_fingerprint", "approximate_artifact_fingerprint",
        "candidate_k", "vocab", "capture_kind", "privacy_profile",
    )
    for field in identity_fields:
        if len({manifest[field] for manifest in manifests}) != 1:
            raise ValueError(f"rank capture manifests disagree on {field}")
    manifest = manifests[0]

    ranks: list[int] = []
    hits: list[int] = []
    agreements: list[int] = []
    requests: set[str] = set()
    shapes: dict[str, dict] = {}
    for _path, records in loaded:
        for record in records[1:]:
            requests.add(record["request_nonce"])
            shapes[record["shape_id"]] = record["shape"]
            for rank, hit, agreement in record["positions"]:
                ranks.append(rank)
                hits.append(hit)
                agreements.append(agreement)

    positions = len(ranks)
    recall_at_k = {
        str(k): (sum(rank <= k for rank in ranks) / positions if positions else 0.0)
        for k in (1, 8, 16, 32, 64)
    }
    actual_recall = sum(hits) / positions if positions else 0.0
    boundary_mismatches = sum(
        bool(hit) != (rank <= PROMOTION_K)
        for rank, hit in zip(ranks, hits, strict=True)
    )
    dimension_values = {
        "streaming": sorted({shape["streaming"] for shape in shapes.values()}),
        "temperature_class": sorted({
            shape["temperature_class"] for shape in shapes.values()}),
        "developer": sorted({shape["developer"] for shape in shapes.values()}),
        "tool_count_bucket": sorted({
            shape["tool_count_bucket"] for shape in shapes.values()}),
        "system_chars_bucket": sorted({
            shape["system_chars_bucket"] for shape in shapes.values()}),
    }
    heterogeneous = bool(
        len(shapes) >= PROMOTION_MIN_SHAPES
        and len(requests) >= PROMOTION_MIN_REQUESTS
        and dimension_values["streaming"] == [False, True]
        and dimension_values["temperature_class"] == ["greedy", "stochastic"]
        and dimension_values["developer"] == [False, True]
        and len(dimension_values["tool_count_bucket"]) >= 2
        and len(dimension_values["system_chars_bucket"]) >= 2
    )
    bound = bool(
        _HEX64.fullmatch(expected_exact_fingerprint)
        and _HEX64.fullmatch(expected_approximate_fingerprint)
        and expected_exact_fingerprint == manifest["exact_source_fingerprint"]
        and expected_approximate_fingerprint
        == manifest["approximate_artifact_fingerprint"]
    )
    gates = {
        "live_authoritative_capture": (
            manifest["capture_kind"] == LIVE_CAPTURE_KIND),
        "explicit_source_binding": bound,
        "candidate_k_64": manifest["candidate_k"] == PROMOTION_K,
        "minimum_1000_positions": positions >= PROMOTION_MIN_POSITIONS,
        "actual_candidate_recall_100_percent": bool(
            positions and sum(hits) == positions),
        "stable_rank_recall_100_percent": bool(
            positions and all(rank <= PROMOTION_K for rank in ranks)),
        "no_rank_shortlist_boundary_mismatch": boundary_mismatches == 0,
        "heterogeneous_real_request_shapes": heterogeneous,
        "bounded_privacy_safe_artifacts": all(
            path.stat().st_size <= CAPTURE_MAX_FILE_BYTES
            for path in resolved),
    }
    promotion_ready = all(gates.values())

    if ranks:
        ordered = sorted(ranks)

        def percentile(fraction: float) -> int:
            index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            return ordered[max(0, index)]
        rank_summary = {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": ordered[-1],
        }
    else:
        rank_summary = {"p50": None, "p95": None, "p99": None, "max": None}
    return {
        "schema": REPORT_SCHEMA,
        "candidate_k": int(manifest["candidate_k"]),
        "vocab": int(manifest["vocab"]),
        "positions": positions,
        "requests": len(requests),
        "distinct_shapes": len(shapes),
        "actual_candidate_recall": actual_recall,
        "stable_rank_recall_at_k": recall_at_k,
        "exact_approx_top1_agreement": (
            sum(agreements) / positions if positions else 0.0),
        "exact_winner_approximate_rank": rank_summary,
        "rank_shortlist_boundary_mismatches": boundary_mismatches,
        "shape_dimension_values": dimension_values,
        "source": {
            "exact_source_fingerprint": manifest["exact_source_fingerprint"],
            "approximate_artifact_fingerprint": (
                manifest["approximate_artifact_fingerprint"]),
            "approximate_artifact_bytes": manifest["approximate_artifact_bytes"],
        },
        "artifacts": [
            {"name": path.name, "bytes": path.stat().st_size,
             "sha256": _sha256_file(path)}
            for path in resolved
        ],
        "gate": {
            **gates,
            "required_positions": PROMOTION_MIN_POSITIONS,
            "required_recall": 1.0,
            "required_min_requests": PROMOTION_MIN_REQUESTS,
            "required_min_shapes": PROMOTION_MIN_SHAPES,
            "promotion_ready": promotion_ready,
        },
    }
