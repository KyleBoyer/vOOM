"""Exact fixed-width sidecar storage for Kimi K3 MXFP4 E8M0 scales.

Kimi K3's released expert matrices use packed FP4 values plus one uint8 E8M0
scale per group.  The FP4 stream is already close to its entropy bound, while
each scale tensor occupies a narrow exponent interval.  This format stores
``scale - min(scale)`` at the smallest width in ``{0, 1, 2, 4, 8}``.

Properties relevant to the out-of-core runtime:

* every expert is independently readable; no routing-history dependency;
* one 16-byte header describes all three projection payloads;
* CRC32 protects every encoded expert record before Metal sees it;
* immutable generation directories are published through an fsync'd atomic
  ``CURRENT`` pointer, so a crash cannot replace the last usable generation;
* the source config/index fingerprint prevents attaching a sidecar to a
  different checkpoint.

This module is intentionally MLX-free.  Runtime Metal decoding lives in
``runtime.kimi_k3_scale_sidecar``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "voom.kimi-k3-scale-sidecar.v1"
CURRENT = "CURRENT"
PROJECTIONS = ("w1", "w3", "w2")
WIDTHS = (0, 1, 2, 4, 8)
HEADER = struct.Struct("<I6BIH")
HEADER_BYTES_PER_EXPERT = HEADER.size
_SCALE_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.block_sparse_moe\.experts\."
    r"(\d+)\.(w1|w2|w3)\.weight_scale$"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_fsync(path: Path, value: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(value)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _source_fingerprint(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    config_raw = config_path.read_bytes()
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_raw = index_path.read_bytes()
        index_sha256 = _sha256_bytes(index_raw)
        shard_names = sorted(set(json.loads(index_raw)["weight_map"].values()))
    else:
        index_sha256 = None
        shard_names = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"no safetensors shards under {model_dir}")
    shard_sizes = {
        name: (model_dir / name).stat().st_size for name in shard_names
    }
    identity = {
        "config_sha256": _sha256_bytes(config_raw),
        "index_sha256": index_sha256,
        "shard_sizes": shard_sizes,
    }
    identity["fingerprint"] = _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return identity


class _SafeTensorReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            self.weight_map: dict[str, str] = json.loads(
                index_path.read_text()
            )["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(model_dir.glob("*.safetensors")):
                header, _ = self._read_header(path)
                for name in header:
                    if name != "__metadata__":
                        if name in self.weight_map:
                            raise ValueError(
                                f"duplicate tensor {name!r} in safetensors"
                            )
                        self.weight_map[name] = path.name
        self._headers: dict[str, tuple[dict, int]] = {}
        self._fds: dict[str, int] = {}

    @staticmethod
    def _read_header(path: Path) -> tuple[dict, int]:
        fd = os.open(path, os.O_RDONLY)
        try:
            length_raw = os.pread(fd, 8, 0)
            if len(length_raw) != 8:
                raise EOFError(f"truncated safetensors header length: {path}")
            length = struct.unpack("<Q", length_raw)[0]
            raw = os.pread(fd, length, 8)
            if len(raw) != length:
                raise EOFError(f"truncated safetensors header: {path}")
        finally:
            os.close(fd)
        return json.loads(raw), 8 + length

    def metadata(self, name: str) -> dict:
        shard = self.weight_map[name]
        cached = self._headers.get(shard)
        if cached is None:
            cached = self._read_header(self.model_dir / shard)
            self._headers[shard] = cached
        return cached[0][name]

    def read(self, name: str) -> bytes:
        shard = self.weight_map[name]
        cached = self._headers.get(shard)
        if cached is None:
            cached = self._read_header(self.model_dir / shard)
            self._headers[shard] = cached
        metadata = cached[0][name]
        start, end = metadata["data_offsets"]
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(self.model_dir / shard, os.O_RDONLY)
            self._fds[shard] = fd
        value = os.pread(fd, end - start, cached[1] + start)
        if len(value) != end - start:
            raise EOFError(
                f"{name}: expected {end - start} bytes, read {len(value)}"
            )
        return value

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def pack_scale(raw: bytes) -> tuple[int, int, bytes]:
    values = np.frombuffer(raw, dtype=np.uint8)
    if values.size == 0:
        raise ValueError("cannot pack an empty scale tensor")
    base = int(values.min())
    span = int(values.max()) - base
    bits = next(width for width in WIDTHS if span < (1 << width))
    if bits == 0:
        return base, bits, b""
    per_byte = 8 // bits
    deltas = values.astype(np.uint16) - base
    padded = (-deltas.size) % per_byte
    if padded:
        deltas = np.pad(deltas, (0, padded))
    rows = deltas.reshape(-1, per_byte)
    packed = np.zeros(rows.shape[0], dtype=np.uint8)
    for slot in range(per_byte):
        packed |= (rows[:, slot] << (slot * bits)).astype(np.uint8)
    return base, bits, packed.tobytes()


def unpack_scale(
    packed: bytes, *, base: int, bits: int, count: int
) -> bytes:
    if bits not in WIDTHS:
        raise ValueError(f"unsupported scale width {bits}")
    if bits == 0:
        return bytes([base]) * count
    encoded = np.frombuffer(packed, dtype=np.uint8)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    decoded = np.empty(encoded.size * per_byte, dtype=np.uint8)
    for slot in range(per_byte):
        decoded[slot::per_byte] = (
            base + ((encoded >> (slot * bits)) & mask)
        ).astype(np.uint8)
    return decoded[:count].tobytes()


def _discover_scales(
    reader: _SafeTensorReader,
) -> dict[int, dict[int, dict[str, str]]]:
    layers: dict[int, dict[int, dict[str, str]]] = {}
    for name in reader.weight_map:
        match = _SCALE_RE.search(name)
        if match is None:
            continue
        layer, expert, projection = match.groups()
        layers.setdefault(int(layer), {}).setdefault(int(expert), {})[
            projection
        ] = name
    return layers


def _validate_layer(
    reader: _SafeTensorReader,
    layer: int,
    experts: dict[int, dict[str, str]],
) -> tuple[list[int], dict[str, list[int]], int, int]:
    expert_ids = sorted(experts)
    if not expert_ids or expert_ids != list(range(expert_ids[-1] + 1)):
        raise ValueError(f"layer {layer}: expert IDs must be dense from zero")
    shapes: dict[str, list[int]] = {}
    raw_bytes = 0
    for expert in expert_ids:
        names = experts[expert]
        if set(names) != set(PROJECTIONS):
            raise ValueError(
                f"layer {layer} expert {expert}: expected {PROJECTIONS}, "
                f"got {sorted(names)}"
            )
        counts = set()
        for projection in PROJECTIONS:
            metadata = reader.metadata(names[projection])
            if metadata["dtype"] != "U8":
                raise ValueError(
                    f"{names[projection]}: expected U8, got "
                    f"{metadata['dtype']}"
                )
            shape = [int(value) for value in metadata["shape"]]
            count = int(np.prod(shape, dtype=np.int64))
            counts.add(count)
            raw_bytes += count
            prior = shapes.setdefault(projection, shape)
            if prior != shape:
                raise ValueError(
                    f"layer {layer}: projection {projection} shape differs "
                    f"across experts: {prior} vs {shape}"
                )
        if len(counts) != 1:
            raise ValueError(
                f"layer {layer} expert {expert}: three projection scale "
                f"counts differ: {sorted(counts)}"
            )
    scale_count = int(np.prod(shapes[PROJECTIONS[0]], dtype=np.int64))
    return expert_ids, shapes, scale_count, raw_bytes


def _external_root_guard(output_root: Path) -> None:
    existing = output_root
    while not existing.exists():
        existing = existing.parent
    if os.stat(existing).st_dev == os.stat("/").st_dev:
        raise ValueError(
            "K3 scale sidecars must live on an external volume; refusing "
            f"internal-disk output root {output_root}"
        )


def build_scale_sidecar(
    model_dir: str | Path,
    output_root: str | Path,
    *,
    layers: list[int] | tuple[int, ...] | None = None,
    enforce_external: bool = True,
    min_free_after_bytes: int = 1_000_000_000,
) -> dict:
    """Build and atomically publish one immutable sidecar generation."""
    model_dir = Path(model_dir).resolve()
    output_root = Path(output_root).expanduser().resolve()
    if enforce_external:
        _external_root_guard(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = _source_fingerprint(model_dir)

    with _SafeTensorReader(model_dir) as reader:
        discovered = _discover_scales(reader)
        selected_layers = (
            sorted(discovered)
            if layers is None
            else sorted(set(int(layer) for layer in layers))
        )
        if not selected_layers:
            raise ValueError("no layers selected")
        missing = [layer for layer in selected_layers if layer not in discovered]
        if missing:
            raise ValueError(f"no complete K3 scale tensors for layers {missing}")

        plans = {}
        estimated_max = 0
        for layer in selected_layers:
            expert_ids, shapes, scale_count, raw_bytes = _validate_layer(
                reader, layer, discovered[layer]
            )
            plans[layer] = (expert_ids, shapes, scale_count, raw_bytes)
            estimated_max += (
                raw_bytes + len(expert_ids) * HEADER_BYTES_PER_EXPERT
            )
        free = shutil.disk_usage(output_root).free
        if free - estimated_max < int(min_free_after_bytes):
            raise OSError(
                f"sidecar worst-case {estimated_max} bytes would leave "
                f"{free - estimated_max} bytes free, below required "
                f"{min_free_after_bytes}"
            )

        generation = (
            f"gen-{int(time.time())}-"
            f"{source['fingerprint'][:12]}-{uuid.uuid4().hex[:8]}"
        )
        staging = output_root / f".{generation}.tmp"
        committed = output_root / generation
        staging.mkdir()
        layer_manifest: dict[str, dict] = {}
        total_raw = 0
        total_encoded = 0
        try:
            for layer in selected_layers:
                expert_ids, shapes, scale_count, raw_bytes = plans[layer]
                path = staging / f"layer-{layer:03d}.scales"
                fd = os.open(
                    path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644
                )
                width_histogram = {str(width): 0 for width in WIDTHS}
                headers: list[bytes] = []
                try:
                    header_bytes = len(expert_ids) * HEADER_BYTES_PER_EXPERT
                    os.ftruncate(fd, header_bytes)
                    cursor = header_bytes
                    for expert in expert_ids:
                        metadata: list[int] = []
                        payloads: list[bytes] = []
                        for projection in PROJECTIONS:
                            raw = reader.read(discovered[layer][expert][projection])
                            base, bits, packed = pack_scale(raw)
                            if (
                                unpack_scale(
                                    packed,
                                    base=base,
                                    bits=bits,
                                    count=len(raw),
                                )
                                != raw
                            ):
                                raise AssertionError(
                                    f"layer {layer} expert {expert} "
                                    f"{projection}: pack round trip changed bytes"
                                )
                            metadata.extend((base, bits))
                            payloads.append(packed)
                            width_histogram[str(bits)] += 1
                        payload = b"".join(payloads)
                        crc = zlib.crc32(payload) & 0xFFFFFFFF
                        headers.append(
                            HEADER.pack(cursor, *metadata, crc, 0)
                        )
                        view = memoryview(payload)
                        while view:
                            written = os.pwrite(fd, view, cursor)
                            if written <= 0:
                                raise OSError("short sidecar payload write")
                            cursor += written
                            view = view[written:]
                    header_blob = b"".join(headers)
                    written = os.pwrite(fd, header_blob, 0)
                    if written != len(header_blob):
                        raise OSError("short sidecar header write")
                    os.ftruncate(fd, cursor)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                encoded_bytes = path.stat().st_size
                layer_manifest[str(layer)] = {
                    "file": path.name,
                    "file_bytes": encoded_bytes,
                    "file_sha256": _sha256_file(path),
                    "expert_count": len(expert_ids),
                    "header_bytes": (
                        len(expert_ids) * HEADER_BYTES_PER_EXPERT
                    ),
                    "scale_count_per_tensor": scale_count,
                    "projection_shapes": shapes,
                    "raw_scale_bytes": raw_bytes,
                    "encoded_bytes": encoded_bytes,
                    "ratio_raw_over_encoded": raw_bytes / encoded_bytes,
                    "width_histogram": width_histogram,
                }
                total_raw += raw_bytes
                total_encoded += encoded_bytes

            manifest = {
                "schema": SCHEMA,
                "source": source,
                "projection_order": list(PROJECTIONS),
                "header_bytes_per_expert": HEADER_BYTES_PER_EXPERT,
                "layers": layer_manifest,
                "total_raw_scale_bytes": total_raw,
                "total_encoded_bytes": total_encoded,
                "ratio_raw_over_encoded": total_raw / total_encoded,
            }
            _write_fsync(
                staging / "manifest.json",
                json.dumps(
                    manifest, indent=2, sort_keys=True
                ).encode()
                + b"\n",
            )
            _fsync_dir(staging)
            os.replace(staging, committed)
            _fsync_dir(output_root)

            pointer_tmp = output_root / f".{CURRENT}.tmp-{uuid.uuid4().hex}"
            _write_fsync(pointer_tmp, (generation + "\n").encode())
            os.replace(pointer_tmp, output_root / CURRENT)
            _fsync_dir(output_root)
        except BaseException:
            # Leave an uncommitted staging generation for forensic inspection.
            # CURRENT is never changed until every file and manifest are durable.
            raise

    return {
        "generation": generation,
        "root": str(output_root),
        "layers": selected_layers,
        "total_raw_scale_bytes": total_raw,
        "total_encoded_bytes": total_encoded,
        "ratio_raw_over_encoded": total_raw / total_encoded,
        "source_fingerprint": source["fingerprint"],
    }


@dataclass(frozen=True)
class ScaleRecord:
    expert: int
    bases: tuple[int, int, int]
    widths: tuple[int, int, int]
    payload: bytes


class KimiK3ScaleSidecar:
    """Validated immutable-generation reader used by WeightStore."""

    def __init__(self, model_dir: str | Path, root: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self.root = Path(root).expanduser().resolve()
        generation = (self.root / CURRENT).read_text().strip()
        if not re.fullmatch(r"gen-[A-Za-z0-9-]+", generation):
            raise ValueError(f"invalid K3 scale-sidecar generation {generation!r}")
        self.generation_dir = self.root / generation
        self.manifest = json.loads(
            (self.generation_dir / "manifest.json").read_text()
        )
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(
                f"unsupported K3 scale-sidecar schema "
                f"{self.manifest.get('schema')!r}"
            )
        if self.manifest.get("projection_order") != list(PROJECTIONS):
            raise ValueError("K3 scale-sidecar projection order mismatch")
        if self.manifest.get("header_bytes_per_expert") != HEADER.size:
            raise ValueError("K3 scale-sidecar header size mismatch")
        source = _source_fingerprint(self.model_dir)
        if self.manifest.get("source", {}).get("fingerprint") != source[
            "fingerprint"
        ]:
            raise ValueError(
                "K3 scale-sidecar checkpoint fingerprint does not match "
                "the requested model"
            )
        self._headers: dict[int, tuple[list[tuple], int]] = {}

    def has_layer(self, layer: int) -> bool:
        return str(int(layer)) in self.manifest["layers"]

    def projection_shapes(self, layer: int) -> dict[str, tuple[int, ...]]:
        entry = self.manifest["layers"][str(int(layer))]
        return {
            name: tuple(int(value) for value in shape)
            for name, shape in entry["projection_shapes"].items()
        }

    def _layer_headers(self, layer: int) -> tuple[list[tuple], int]:
        layer = int(layer)
        cached = self._headers.get(layer)
        if cached is not None:
            return cached[0], 0
        entry = self.manifest["layers"][str(layer)]
        path = self.generation_dir / entry["file"]
        header_bytes = int(entry["header_bytes"])
        fd = os.open(path, os.O_RDONLY)
        try:
            raw = os.pread(fd, header_bytes, 0)
        finally:
            os.close(fd)
        if len(raw) != header_bytes:
            raise EOFError(f"layer {layer}: truncated sidecar headers")
        headers = [
            HEADER.unpack_from(raw, offset)
            for offset in range(0, len(raw), HEADER.size)
        ]
        file_bytes = int(entry["file_bytes"])
        offsets = [int(header[0]) for header in headers]
        if (
            len(headers) != int(entry["expert_count"])
            or not offsets
            or offsets[0] != header_bytes
            or offsets != sorted(offsets)
            or offsets[-1] > file_bytes
        ):
            raise ValueError(f"layer {layer}: invalid sidecar extent table")
        for header in headers:
            widths = (header[2], header[4], header[6])
            if any(width not in WIDTHS for width in widths):
                raise ValueError(
                    f"layer {layer}: invalid encoded width {widths}"
                )
        cached = (headers, header_bytes)
        self._headers[layer] = cached
        return cached

    def read_records(
        self, layer: int, expert_ids: list[int]
    ) -> tuple[list[ScaleRecord], int]:
        layer = int(layer)
        headers, header_read_bytes = self._layer_headers(layer)
        entry = self.manifest["layers"][str(layer)]
        file_bytes = int(entry["file_bytes"])
        path = self.generation_dir / entry["file"]
        records = []
        physical_bytes = header_read_bytes
        fd = os.open(path, os.O_RDONLY)
        try:
            for expert in expert_ids:
                if expert < 0 or expert >= len(headers):
                    raise IndexError(
                        f"layer {layer}: expert {expert} outside sidecar"
                    )
                header = headers[expert]
                start = int(header[0])
                end = (
                    int(headers[expert + 1][0])
                    if expert + 1 < len(headers)
                    else file_bytes
                )
                payload = os.pread(fd, end - start, start)
                if len(payload) != end - start:
                    raise EOFError(
                        f"layer {layer} expert {expert}: truncated payload"
                    )
                expected_crc = int(header[7])
                actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    raise IOError(
                        f"layer {layer} expert {expert}: sidecar CRC mismatch"
                    )
                records.append(
                    ScaleRecord(
                        expert=expert,
                        bases=(header[1], header[3], header[5]),
                        widths=(header[2], header[4], header[6]),
                        payload=payload,
                    )
                )
                physical_bytes += len(payload)
        finally:
            os.close(fd)
        return records, physical_bytes


def assemble_decode_batch(records: list[ScaleRecord]) -> bytes:
    """Rebase independently read records into one fused-decoder input."""
    headers = bytearray(len(records) * HEADER.size)
    payload = bytearray()
    for local, record in enumerate(records):
        offset = len(headers) + len(payload)
        metadata = [
            record.bases[0],
            record.widths[0],
            record.bases[1],
            record.widths[1],
            record.bases[2],
            record.widths[2],
        ]
        HEADER.pack_into(headers, local * HEADER.size, offset, *metadata, 0, 0)
        payload.extend(record.payload)
    return bytes(headers + payload)


def _parse_layers(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        help="comma/range list such as 1-12,46; default builds every MoE layer",
    )
    args = parser.parse_args()
    report = build_scale_sidecar(
        args.model_dir, args.output_root, layers=args.layers
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
