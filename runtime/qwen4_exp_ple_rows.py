"""Verified direct-row storage for Qwen4-Exp PLE embeddings.

The released Flash-Next checkpoint keeps its roughly 95GiB PLE matrix as
numbered row shards. This provider maps global PLE row IDs directly onto those
safetensor extents, coalesces adjacent physical rows, and returns exact BF16
storage bits without materializing or copying the full table. Published FP8
derivatives may instead store the same rows as E4M3 plus one global BF16
``weight_scale``; those rows are reconstructed after the selective read, so
the 51.2GB packed table is never widened in full.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
from typing import Sequence

import numpy as np

from .qwen4_exp_ple import Qwen4ExpPLELayout


_PART_PATTERN = re.compile(
    r"^(?P<prefix>.+\.ple_embedding\.ngram_embedding)"
    r"\.shard_(?P<part>\d+)\.weight$")
_FP8_DTYPE = "F8_E4M3"
_BF16_DTYPE = "BF16"
_OVERLAY_PLAN_SCHEMA = "voom.hf-checkpoint-overlay-plan.v1"
_OVERLAY_RECEIPT_SCHEMA = "voom.hf-checkpoint-overlay-receipt.v1"
_OVERLAY_PLAN_NAME = ".voom-overlay-plan.json"
_OVERLAY_RECEIPT_NAME = "voom.overlay.receipt.json"
_REPLACEMENT_MARKER_NAME = ".voom-checkpoint-replacement-in-progress.json"
_REPLACEMENT_RECEIPT_NAME = "voom.checkpoint.receipt.json"
_REPLACEMENT_RECEIPT_SCHEMA = "voom.hf-checkpoint-replacement-receipt.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    chunks = []
    done = 0
    while done < size:
        chunk = os.pread(fd, size - done, offset + done)
        if not chunk:
            raise IOError(f"short PLE read at {offset}: {done}/{size} bytes")
        chunks.append(chunk)
        done += len(chunk)
    return b"".join(chunks)


def _safetensor_header(path: Path) -> tuple[int, dict]:
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = _pread_exact(fd, 8, 0)
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 0 or header_size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header size in {path}")
        header = json.loads(_pread_exact(fd, header_size, 8))
    finally:
        os.close(fd)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header in {path}")
    return header_size, header


def _bf16_scalar(bits: int) -> np.float32:
    encoded = np.array([np.uint32(bits) << np.uint32(16)], dtype=np.uint32)
    return encoded.view(np.float32)[0]


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 storage bits with round-to-nearest-even."""
    floats = np.asarray(values, dtype=np.float32)
    encoded = floats.view(np.uint32)
    rounded = encoded + np.uint32(0x7FFF) + ((encoded >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _fp8_bf16_lookup(scale_bits: int) -> np.ndarray:
    """Build the exact E4M3 -> BF16 table for one published BF16 scale."""
    packed = np.arange(256, dtype=np.uint16)
    magnitude = packed & 127
    sign = ((packed >> 7) & 1) << 15
    # This mirrors MLX's fp8_e4m3 conversion, including its finite +/-480
    # interpretation of the two all-ones payloads. Every decoded FP8 value is
    # exactly representable in float32 before the BF16-scale multiply.
    half_bits = ((magnitude << 7)
                 | (((magnitude + 1) >> 7) << 14)
                 | sign).astype(np.uint16)
    values = half_bits.view(np.float16).astype(np.float32) * np.float32(256.0)
    values[magnitude == 127] = np.where(
        (packed[magnitude == 127] & 128) != 0,
        np.float32(-480.0), np.float32(480.0))
    scale = _bf16_scalar(scale_bits)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"PLE FP8 weight_scale must be positive finite, got {scale}")
    return _float32_to_bf16_bits(values * scale)


def _native_release_witness(
    model_dir: Path, shards: tuple[str, ...],
) -> tuple[list[tuple[str, str, int]], str]:
    tree_dir = model_dir / ".cache" / "huggingface" / "trees"
    for tree_path in sorted(tree_dir.glob("*.json")):
        try:
            tree = json.loads(tree_path.read_text())
            files = tree.get("files", {})
        except (OSError, ValueError):
            continue
        candidate = []
        for shard in shards:
            meta = files.get(shard, {})
            sha = str(meta.get("lfs_sha256") or "")
            size = int(meta.get("lfs_size") or 0)
            if (not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or size != (model_dir / shard).stat().st_size):
                candidate = []
                break
            metadata_path = (
                model_dir / ".cache" / "huggingface" / "download"
                / f"{shard}.metadata")
            lines = (
                metadata_path.read_text().splitlines()
                if metadata_path.is_file() else [])
            if (len(lines) < 2 or lines[0].strip() != tree_path.stem
                    or lines[1].strip().strip('"') != sha):
                candidate = []
                break
            candidate.append((shard, sha, size))
        if candidate:
            return candidate, tree_path.stem
    return [], ""


def _overlay_release_witness(
    model_dir: Path, shards: tuple[str, ...],
) -> tuple[list[tuple[str, str, int]], str]:
    """Validate the finalized hash-attested overlay provenance chain.

    Finalization already hashes every downloaded candidate file against its
    pinned Hub object and every linked file against the candidate's identical
    published object identity. Runtime revalidates the immutable plan/receipt,
    file kinds, sizes, destinations, and exact topology links; it does not
    trust a partially downloaded candidate or a name-only symlink farm.
    """
    plan_path = model_dir / _OVERLAY_PLAN_NAME
    receipt_path = model_dir / _OVERLAY_RECEIPT_NAME
    if not plan_path.is_file() or not receipt_path.is_file():
        return [], ""
    try:
        plan_bytes = plan_path.read_bytes()
        plan = json.loads(plan_bytes)
        receipt = json.loads(receipt_path.read_text())
        if (plan.get("schema") != _OVERLAY_PLAN_SCHEMA
                or receipt.get("schema") != _OVERLAY_RECEIPT_SCHEMA
                or receipt.get("status") != "verified"):
            return [], ""
        if hashlib.sha256(plan_bytes).hexdigest() != receipt.get("plan_sha256"):
            return [], ""
        if (Path(plan.get("destination", "")).resolve() != model_dir
                or Path(receipt.get("destination", "")).resolve() != model_dir):
            return [], ""
        if (plan.get("base") != receipt.get("base")
                or plan.get("candidate") != receipt.get("candidate")
                or receipt.get("config_equal") is not True
                or receipt.get("tensor_to_shard_map_equal") is not True):
            return [], ""
        files = plan.get("files", {})
        downloads = files.get("download", [])
        links = files.get("link", [])
        if (len(downloads) != int(receipt.get("downloaded_files", -1))
                or len(links) != int(receipt.get("linked_files", -1))
                or sum(int(item["size"]) for item in downloads)
                != int(receipt.get("verified_download_bytes", -1))
                or sum(int(item["size"]) for item in links)
                != int(receipt.get("verified_link_bytes", -1))):
            return [], ""
        download_map = {str(item["path"]): item for item in downloads}
        link_map = {str(item["path"]): item for item in links}
        if (len(download_map) != len(downloads)
                or len(link_map) != len(links)
                or download_map.keys() & link_map.keys()):
            return [], ""
        base = Path(plan["base"]["directory"]).resolve()
        # Exact config and tensor placement are mandatory links to the pinned
        # base. Candidate-edited topology cannot inherit the PLE address proof.
        for name in ("config.json", "model.safetensors.index.json"):
            path = model_dir / name
            target = base / name
            if (name not in link_map or not path.is_symlink()
                    or path.resolve() != target.resolve()):
                return [], ""
        revision = str(plan["candidate"].get("revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            return [], ""
        witnesses = []
        for shard in shards:
            path = model_dir / shard
            record = link_map.get(shard)
            if record is not None:
                target = base / shard
                if (not path.is_symlink()
                        or path.resolve() != target.resolve()):
                    return [], ""
            else:
                record = download_map.get(shard)
                if record is None or path.is_symlink():
                    return [], ""
            size = int(record["size"])
            sha = str(record.get("hash", ""))
            if (record.get("hash_kind") != "sha256"
                    or not re.fullmatch(r"[0-9a-f]{64}", sha)
                    or not path.is_file() or path.stat().st_size != size):
                return [], ""
            witnesses.append((shard, sha, size))
        return witnesses, revision
    except (KeyError, OSError, TypeError, ValueError):
        return [], ""


def _replacement_release_witness(
    model_dir: Path, shards: tuple[str, ...], index_path: Path,
) -> tuple[list[tuple[str, str, int]], str]:
    """Bind rows to a finalized in-place checkpoint replacement.

    Finalization hashed every candidate object before publishing its receipt.
    The receipt commits to that replacement plan; current size and modification
    time checks reject ordinary post-finalization edits without re-reading the
    full 185GB checkpoint during every server startup. This is the same cheap
    mutable-tree guard used by ``checkpoint_identity``.
    """
    receipt_path = model_dir / _REPLACEMENT_RECEIPT_NAME
    if (not receipt_path.is_file()
            or (model_dir / _REPLACEMENT_MARKER_NAME).exists()):
        return [], ""
    try:
        receipt = json.loads(receipt_path.read_text())
        candidate = receipt.get("candidate", {})
        revision = str(candidate.get("revision", ""))
        plan_sha = str(receipt.get("plan_sha256", ""))
        completed_ns = int(float(receipt.get("completed_unix", 0)) * 1e9)
        if (receipt.get("schema") != _REPLACEMENT_RECEIPT_SCHEMA
                or receipt.get("status") != "verified"
                or Path(receipt.get("model_dir", "")).resolve() != model_dir
                or not re.fullmatch(r"[0-9a-f]{40}", revision)
                or not re.fullmatch(r"[0-9a-f]{64}", plan_sha)
                or completed_ns <= 0):
            return [], ""
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return [], ""
        indexed_shards = tuple(sorted(set(map(str, weight_map.values()))))
        if (len(indexed_shards) != int(receipt.get("safetensor_shards", -1))
                or int(receipt.get("changed_safetensor_shards", -1))
                != len(indexed_shards)
                or int(receipt.get("candidate_files", -1))
                < len(indexed_shards) + 2):
            return [], ""
        shard_stats = {}
        indexed_bytes = 0
        for shard in indexed_shards:
            relative = Path(shard)
            path = model_dir / relative
            if (relative.name != shard or relative.is_absolute()
                    or not path.is_file() or path.is_symlink()):
                return [], ""
            stat = path.stat()
            if stat.st_mtime_ns > completed_ns:
                return [], ""
            indexed_bytes += int(stat.st_size)
            shard_stats[shard] = stat
        if indexed_bytes > int(receipt.get("verified_candidate_bytes", -1)):
            return [], ""
        for metadata in (model_dir / "config.json", index_path):
            if (not metadata.is_file() or metadata.is_symlink()
                    or metadata.stat().st_mtime_ns > completed_ns):
                return [], ""
        if any(shard not in shard_stats for shard in shards):
            return [], ""
        witnesses = []
        for shard in shards:
            stat = shard_stats[shard]
            stat_commitment = hashlib.sha256(
                f"{plan_sha}:{shard}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            witnesses.append((shard, stat_commitment, int(stat.st_size)))
        return witnesses, revision
    except (KeyError, OSError, TypeError, ValueError):
        return [], ""


@dataclass(frozen=True)
class PLESourceIdentity:
    fingerprint: str
    verified_release_hash: bool
    revision: str
    unique_shards: int
    split_parts: int
    padded_vocab_size: int
    row_width: int
    storage_dtype: str
    storage_row_bytes: int
    scale_bf16_bits: int | None


@dataclass(frozen=True)
class _Part:
    index: int
    name: str
    shard: str
    rows: int
    row_start: int
    data_start: int


class Qwen4ExpPLERowStore:
    """Read exact released PLE rows by global row ID."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        row_cache: int = 8192,
        require_release_hash: bool = True,
        read_workers: int = 1,
    ):
        if isinstance(row_cache, bool) or row_cache < 0:
            raise ValueError("row_cache must be non-negative")
        if (isinstance(read_workers, bool) or not isinstance(read_workers, int)
                or not 1 <= read_workers <= 16):
            raise ValueError("read_workers must be an integer in [1, 16]")
        self.model_dir = Path(model_dir).expanduser().resolve()
        config = json.loads((self.model_dir / "config.json").read_text())
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Qwen4-Exp checkpoint has no text_config")
        self.layout = Qwen4ExpPLELayout.from_text_config(text_config)
        split_parts = int(text_config.get("split_ngram_parts", 0))
        if split_parts <= 0:
            raise ValueError("Qwen4-Exp split_ngram_parts must be positive")
        index_path = self.model_dir / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Qwen4-Exp checkpoint has no safetensor weight map")

        names: dict[int, tuple[str, str]] = {}
        prefixes = set()
        for name, shard in weight_map.items():
            match = _PART_PATTERN.match(name)
            if match is None:
                continue
            part_index = int(match.group("part"))
            if part_index in names:
                raise ValueError(f"duplicate PLE shard index {part_index}")
            if not isinstance(shard, str) or not shard:
                raise ValueError(f"PLE part {part_index} has no shard file")
            names[part_index] = (name, shard)
            prefixes.add(match.group("prefix"))
        if sorted(names) != list(range(split_parts)) or len(prefixes) != 1:
            raise ValueError(
                f"PLE split is incomplete or ambiguous: found {len(names)} "
                f"parts and {len(prefixes)} prefixes, expected {split_parts}")

        headers = {}
        parts = []
        row_start = 0
        storage_dtype = ""
        for part_index in range(split_parts):
            name, shard = names[part_index]
            shard_path = self.model_dir / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"missing PLE source shard {shard_path}")
            if shard not in headers:
                headers[shard] = _safetensor_header(shard_path)
            header_size, header = headers[shard]
            meta = header.get(name)
            if not isinstance(meta, dict):
                raise ValueError(f"PLE tensor {name!r} is missing from {shard}")
            shape = tuple(int(value) for value in meta.get("shape", ()))
            dtype = str(meta.get("dtype", ""))
            if (dtype not in (_BF16_DTYPE, _FP8_DTYPE) or len(shape) != 2
                    or shape[1] != self.layout.row_width):
                raise ValueError(f"unexpected PLE tensor metadata for {name}: {meta}")
            if storage_dtype and dtype != storage_dtype:
                raise ValueError(
                    "PLE split mixes storage dtypes: "
                    f"{storage_dtype} and {dtype} at {name}")
            storage_dtype = dtype
            storage_row_bytes = self.layout.row_width * (
                2 if dtype == _BF16_DTYPE else 1)
            offsets = meta.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or offsets[1] - offsets[0]
                    != shape[0] * storage_row_bytes):
                raise ValueError(f"invalid PLE tensor extent for {name}")
            parts.append(_Part(
                index=part_index,
                name=name,
                shard=shard,
                rows=shape[0],
                row_start=row_start,
                data_start=8 + header_size + int(offsets[0]),
            ))
            row_start += shape[0]
        if row_start != self.layout.padded_vocab_size:
            raise ValueError(
                f"PLE row count {row_start} != released layout "
                f"{self.layout.padded_vocab_size}")
        self.storage_dtype = storage_dtype
        self.storage_row_bytes = self.layout.row_width * (
            2 if storage_dtype == _BF16_DTYPE else 1)
        # ``row_bytes`` historically described source I/O and remains an
        # alias for callers that inspect it. The returned rows are always
        # ``output_row_bytes`` BF16 bytes.
        self.row_bytes = self.storage_row_bytes
        self.output_row_bytes = self.layout.row_bytes_bf16
        self.scale_name = ""
        self.scale_bf16_bits: int | None = None
        self._fp8_lookup: np.ndarray | None = None
        self.scale_bytes_read = 0
        if storage_dtype == _FP8_DTYPE:
            prefix = next(iter(prefixes))
            self.scale_name = f"{prefix}.weight_scale"
            scale_shard = weight_map.get(self.scale_name)
            if not isinstance(scale_shard, str) or not scale_shard:
                raise ValueError(
                    "FP8 PLE split has no global BF16 weight_scale")
            scale_path = self.model_dir / scale_shard
            if not scale_path.is_file():
                raise FileNotFoundError(
                    f"missing PLE scale source shard {scale_path}")
            if scale_shard not in headers:
                headers[scale_shard] = _safetensor_header(scale_path)
            scale_header_size, scale_header = headers[scale_shard]
            scale_meta = scale_header.get(self.scale_name)
            if (not isinstance(scale_meta, dict)
                    or scale_meta.get("dtype") != _BF16_DTYPE
                    or tuple(scale_meta.get("shape", ())) != (1,)):
                raise ValueError(
                    f"unexpected PLE weight_scale metadata: {scale_meta}")
            scale_offsets = scale_meta.get("data_offsets")
            if (not isinstance(scale_offsets, list) or len(scale_offsets) != 2
                    or scale_offsets[1] - scale_offsets[0] != 2):
                raise ValueError("invalid PLE weight_scale extent")
            scale_fd = os.open(scale_path, os.O_RDONLY)
            try:
                scale_raw = _pread_exact(
                    scale_fd, 2,
                    8 + scale_header_size + int(scale_offsets[0]))
            finally:
                os.close(scale_fd)
            self.scale_bf16_bits = int(
                np.frombuffer(scale_raw, dtype="<u2")[0])
            self._fp8_lookup = _fp8_bf16_lookup(self.scale_bf16_bits)
            self.scale_bytes_read = len(scale_raw)
        elif f"{next(iter(prefixes))}.weight_scale" in weight_map:
            raise ValueError("BF16 PLE split unexpectedly has an FP8 weight_scale")
        self.parts = tuple(parts)
        self._part_ends = tuple(part.row_start + part.rows for part in parts)
        self.identity = self._source_identity(
            tuple(sorted(headers)), split_parts, index_path)
        if require_release_hash and not self.identity.verified_release_hash:
            raise ValueError("PLE source has no complete verified release witness")

        self._fds = {
            shard: os.open(self.model_dir / shard, os.O_RDONLY)
            for shard in headers
        }
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_cap = row_cache
        self.read_workers = read_workers
        self._executor = (
            ThreadPoolExecutor(
                max_workers=read_workers,
                thread_name_prefix="qwen4-ple-row")
            if read_workers > 1 else None)
        self.read_calls = 0
        self.read_extents = 0
        self.rows_requested = 0
        self.unique_rows_read = 0
        self.bytes_read = 0
        self.cache_hits = 0
        self.read_seconds = 0.0
        self.parallel_read_calls = 0

    def _source_identity(
        self, shards: tuple[str, ...], split_parts: int, index_path: Path,
    ) -> PLESourceIdentity:
        witnesses, revision = _native_release_witness(
            self.model_dir, shards)
        if not witnesses:
            witnesses, revision = _overlay_release_witness(
                self.model_dir, shards)
        if not witnesses:
            witnesses, revision = _replacement_release_witness(
                self.model_dir, shards, index_path)

        descriptor = {
            "revision": revision,
            "shards": witnesses,
            "split_parts": split_parts,
            "padded_vocab_size": self.layout.padded_vocab_size,
            "row_width": self.layout.row_width,
            "storage_dtype": self.storage_dtype,
            "storage_row_bytes": self.storage_row_bytes,
            "scale_name": self.scale_name,
            "scale_bf16_bits": self.scale_bf16_bits,
            "config_sha256": _sha256_file(self.model_dir / "config.json"),
            "index_sha256": _sha256_file(index_path),
            "parts": [
                {
                    "index": part.index,
                    "name": part.name,
                    "shard": part.shard,
                    "rows": part.rows,
                    "row_start": part.row_start,
                    "data_start": part.data_start,
                }
                for part in self.parts
            ],
        }
        fingerprint = hashlib.sha256(json.dumps(
            descriptor, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        return PLESourceIdentity(
            fingerprint=fingerprint,
            verified_release_hash=len(witnesses) == len(shards),
            revision=revision,
            unique_shards=len(shards),
            split_parts=split_parts,
            padded_vocab_size=self.layout.padded_vocab_size,
            row_width=self.layout.row_width,
            storage_dtype=self.storage_dtype,
            storage_row_bytes=self.storage_row_bytes,
            scale_bf16_bits=self.scale_bf16_bits,
        )

    def _location(self, row_id: int) -> tuple[_Part, int]:
        if row_id < 0 or row_id >= self.layout.padded_vocab_size:
            raise IndexError(
                f"PLE row {row_id} outside [0, {self.layout.padded_vocab_size})")
        part_index = bisect.bisect_right(self._part_ends, row_id)
        part = self.parts[part_index]
        return part, row_id - part.row_start

    def read_rows(self, row_ids: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        """Return uint16 BF16 storage with shape ``row_ids + [row_width]``."""
        requested = np.asarray(row_ids, dtype=np.int64)
        if requested.ndim == 0 or requested.size == 0:
            raise ValueError("PLE row IDs must be a non-empty array")
        if np.any(requested < 0) or np.any(
                requested >= self.layout.padded_vocab_size):
            raise IndexError("PLE row ID is outside the released table")
        flat = requested.reshape(-1)
        rows: dict[int, np.ndarray] = {}
        missing = []
        for value in flat:
            row_id = int(value)
            cached = self._cache.get(row_id)
            if cached is None:
                missing.append(row_id)
            else:
                self._cache.move_to_end(row_id)
                self.cache_hits += 1
                rows[row_id] = cached

        locations = []
        for row_id in sorted(set(missing)):
            part, local_row = self._location(row_id)
            locations.append((
                part.shard,
                part.data_start + local_row * self.storage_row_bytes,
                row_id,
            ))
        locations.sort(key=lambda value: (value[0], value[1]))
        cursor = 0
        extents = []
        while cursor < len(locations):
            shard, offset, _ = locations[cursor]
            end = cursor + 1
            while (end < len(locations)
                   and locations[end][0] == shard
                   and locations[end][1]
                   == locations[end - 1][1] + self.storage_row_bytes):
                end += 1
            extents.append((
                shard, offset, cursor, end,
                (end - cursor) * self.storage_row_bytes))
            cursor = end

        started = time.perf_counter()
        if self._executor is None or len(extents) <= 1:
            raw_extents = [
                _pread_exact(self._fds[shard], size, offset)
                for shard, offset, _start, _end, size in extents
            ]
        else:
            # Contiguous chunks keep every worker moving forward through a
            # small shard subset while the device services several exact
            # random-read streams. Results are consumed in the original
            # sorted physical order, so cache order and returned rows remain
            # deterministic regardless of completion order.
            workers = min(self.read_workers, len(extents))
            chunk = (len(extents) + workers - 1) // workers

            def read_group(group):
                return [
                    _pread_exact(self._fds[shard], size, offset)
                    for shard, offset, _start, _end, size in group
                ]

            futures = [
                self._executor.submit(read_group, extents[start:start + chunk])
                for start in range(0, len(extents), chunk)
            ]
            raw_extents = []
            for future in futures:
                raw_extents.extend(future.result())
            self.parallel_read_calls += 1
        self.read_seconds += time.perf_counter() - started

        for extent, raw in zip(extents, raw_extents):
            _shard, _offset, cursor, end, _size = extent
            if self.storage_dtype == _BF16_DTYPE:
                storage = np.frombuffer(raw, dtype=np.uint16).reshape(
                    end - cursor, self.layout.row_width)
            else:
                if self._fp8_lookup is None:
                    raise RuntimeError("FP8 PLE lookup was not initialized")
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(
                    end - cursor, self.layout.row_width)
                storage = self._fp8_lookup[packed]
            for local_index, (_, _, row_id) in enumerate(
                    locations[cursor:end]):
                row = storage[local_index].copy()
                rows[row_id] = row
                if self._cache_cap:
                    self._cache[row_id] = row
                    self._cache.move_to_end(row_id)
                    if len(self._cache) > self._cache_cap:
                        self._cache.popitem(last=False)
        self.read_calls += 1
        self.read_extents += len(extents)
        self.rows_requested += int(flat.size)
        self.unique_rows_read += len(locations)
        self.bytes_read += len(locations) * self.storage_row_bytes
        result = np.stack([rows[int(row_id)] for row_id in flat])
        return result.reshape(*requested.shape, self.layout.row_width)

    def telemetry(self) -> dict[str, int | str]:
        return {
            "source_fingerprint": self.identity.fingerprint,
            "source_revision": self.identity.revision,
            "source_verified_release_hash": int(
                self.identity.verified_release_hash),
            "split_parts": self.identity.split_parts,
            "unique_shards": self.identity.unique_shards,
            "storage_dtype": self.storage_dtype,
            "storage_row_bytes": self.storage_row_bytes,
            "output_row_bytes": self.output_row_bytes,
            "scale_bytes_read": self.scale_bytes_read,
            "read_calls": self.read_calls,
            "read_extents": self.read_extents,
            "rows_requested": self.rows_requested,
            "unique_rows_read": self.unique_rows_read,
            "bytes_read": self.bytes_read,
            "cache_hits": self.cache_hits,
            "cache_rows": len(self._cache),
            "read_workers": self.read_workers,
            "parallel_read_calls": self.parallel_read_calls,
            "read_microseconds": int(self.read_seconds * 1_000_000),
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "Qwen4ExpPLERowStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
