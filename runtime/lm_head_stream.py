"""F02 (second half): block-streamed LM-head argmax/logits.

The LM head (GLM-5.2: 154,880 x 6144 bf16 ~= 1.9 GB) is only ever used as
`normed_hidden @ lm_head.T` — a matvec (decode) or thin matmul (verify) whose
CONTRACTION dimension is hidden, not vocab. Splitting the OUTPUT (vocab)
dimension into row blocks leaves each output's mathematical dot product
unchanged. Backend kernel selection can still depend on matrix geometry, so
bit identity to a whole-tensor projection requires an oracle for the actual
dtype/shape/backend. Residency experiments use the same vocabulary block
width and compare all logits, not just top-1.

`mx.load(...)[name]` is lazy per-TENSOR but not per-SLICE: evaluating any
slice of a lazy tensor forces the whole tensor to be read (measured directly
on Qwen2.5-0.5B's embed_tokens.weight: evaluating a 1000-row slice of a
272.3 MB tensor still peaked at 272.27 MB). So this bypasses mx.load entirely
for lm_head.weight and reads raw bytes straight from the safetensors shard via
seek/pread, mirroring the row-paged technique in embed_rows.py but swept in
order across the whole vocab each call instead of cached by row index.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
_MX_DTYPE = {"BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
_NP_STORAGE_DTYPE = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32}


@dataclass(frozen=True)
class LMHeadSourceIdentity:
    fingerprint: str
    verified_release_hash: bool
    revision: str
    shard: str
    shard_sha256: str
    shard_size: int
    dtype: str
    shape: tuple[int, int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_header(model_dir: Path, name: str) -> tuple[dict, str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard = index.get("weight_map", {}).get(name)
        if not isinstance(shard, str) or not shard:
            raise ValueError(f"exact source index has no {name!r}")
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError("exact source requires an indexed or single-shard checkpoint")
        shard = shards[0].name
    shard_path = model_dir / shard
    if not shard_path.is_file():
        raise FileNotFoundError(f"missing exact source shard {shard_path}")
    with shard_path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid safetensors header in {shard_path}")
        header_size = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(header_size))
    meta = header.get(name)
    if not isinstance(meta, dict):
        raise ValueError(f"exact source shard has no tensor {name!r}")
    return meta, shard, str(shard_path)


def lm_head_source_identity(
    model_dir: str | Path, name: str = "lm_head.weight",
) -> LMHeadSourceIdentity:
    """Bind an exact LM head to Hugging Face's recorded release hash.

    Hashing a 2.5 GB head at every server start would erase much of the
    latency win. A complete ``hf download`` already records the immutable LFS
    SHA-256 and revision in its cache tree. We bind that release witness to the
    actual shard size, tensor header, config, tokenizer, and index. Sources
    without the LFS witness still receive a descriptor fingerprint for
    planning, but serving validation rejects ``verified_release_hash=False``.
    """

    directory = Path(model_dir).expanduser().resolve()
    meta, shard, shard_path_text = _tensor_header(directory, name)
    shape = tuple(int(value) for value in meta.get("shape", ()))
    dtype = str(meta.get("dtype", ""))
    if len(shape) != 2:
        raise ValueError(f"exact LM head must be rank 2, got {shape}")
    shard_path = Path(shard_path_text)
    shard_size = shard_path.stat().st_size

    revision = ""
    metadata_etag = ""
    shard_sha256 = ""
    metadata_path = (
        directory / ".cache" / "huggingface" / "download"
        / f"{shard}.metadata")
    if metadata_path.is_file():
        lines = metadata_path.read_text().splitlines()
        if lines:
            revision = lines[0].strip()
        if len(lines) > 1:
            metadata_etag = lines[1].strip().strip('"')
    tree_dir = directory / ".cache" / "huggingface" / "trees"
    revision_tree = tree_dir / f"{revision}.json"
    tree_paths = ([revision_tree] if revision and revision_tree.is_file()
                  else sorted(tree_dir.glob("*.json")))
    for tree_path in tree_paths:
        try:
            tree = json.loads(tree_path.read_text())
            file_meta = tree.get("files", {}).get(shard, {})
        except (OSError, ValueError):
            continue
        candidate = str(file_meta.get("lfs_sha256") or "")
        candidate_size = int(file_meta.get("lfs_size") or 0)
        if (re.fullmatch(r"[0-9a-f]{64}", candidate)
                and candidate_size == shard_size
                and (not re.fullmatch(r"[0-9a-f]{64}", metadata_etag)
                     or metadata_etag == candidate)):
            shard_sha256 = candidate
            if not revision:
                revision = tree_path.stem
            break

    descriptor = {
        "revision": revision,
        "shard": shard,
        "shard_sha256": shard_sha256,
        "shard_size": shard_size,
        "tensor": {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "data_offsets": meta.get("data_offsets"),
        },
        "config_sha256": (
            _sha256_file(directory / "config.json")
            if (directory / "config.json").is_file() else ""),
        "tokenizer_sha256": (
            _sha256_file(directory / "tokenizer.json")
            if (directory / "tokenizer.json").is_file() else ""),
        "index_sha256": (
            _sha256_file(directory / "model.safetensors.index.json")
            if (directory / "model.safetensors.index.json").is_file() else ""),
    }
    fingerprint = hashlib.sha256(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return LMHeadSourceIdentity(
        fingerprint=fingerprint,
        verified_release_hash=bool(shard_sha256 and revision),
        revision=revision,
        shard=shard,
        shard_sha256=shard_sha256,
        shard_size=shard_size,
        dtype=dtype,
        shape=(shape[0], shape[1]),
    )


def open_verified_exact_lm_head(
    target_dir: str | Path,
    source_dir: str | Path,
    expected_fingerprint: str,
    *,
    block_rows: int = 16384,
) -> "StreamedLMHead":
    """Open a row-paged BF16 head only when provenance is fully bound."""

    target = Path(target_dir).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_fingerprint)):
        raise ValueError("exact LM-head source fingerprint is unavailable or invalid")
    target_config_path = target / "config.json"
    if not target_config_path.is_file():
        raise FileNotFoundError(f"missing target config {target_config_path}")
    target_config = json.loads(target_config_path.read_text())
    provenance = target_config.get("voom_quantization")
    if not isinstance(provenance, dict):
        raise ValueError("target has no vOOM quantization provenance")
    recorded_source = provenance.get("source")
    if not isinstance(recorded_source, str) or Path(
            recorded_source).expanduser().resolve() != source:
        raise ValueError("exact LM-head source does not match target provenance")

    identity = lm_head_source_identity(source)
    if not identity.verified_release_hash:
        raise ValueError("exact LM-head source has no verified release hash")
    if identity.fingerprint != expected_fingerprint:
        raise ValueError("exact LM-head source fingerprint mismatch")
    recorded_fingerprint = provenance.get("source_lm_head_fingerprint")
    if (recorded_fingerprint is not None
            and recorded_fingerprint != expected_fingerprint):
        raise ValueError("target provenance fingerprint disagrees with exact source")
    if identity.dtype != "BF16":
        raise ValueError(f"exact LM-head source must be BF16, got {identity.dtype}")
    for tokenizer_name in ("tokenizer.json", "tokenizer_config.json"):
        target_tokenizer = target / tokenizer_name
        source_tokenizer = source / tokenizer_name
        if (not target_tokenizer.is_file() or not source_tokenizer.is_file()
                or _sha256_file(target_tokenizer) != _sha256_file(source_tokenizer)):
            raise ValueError(
                f"exact LM-head source {tokenizer_name} does not match target")

    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        shards = sorted(source.glob("*.safetensors"))
        if len(shards) != 1:
            raise ValueError(
                "exact source requires an indexed or single-shard checkpoint")
        weight_map = {"lm_head.weight": shards[0].name}
    return StreamedLMHead(
        source, weight_map, block_rows=block_rows)


def _pread_exact(fd: int, size: int, offset: int) -> bytes:
    """Read exactly one tensor extent or fail instead of reshaping short data."""
    parts = []
    done = 0
    while done < size:
        chunk = os.pread(fd, size - done, offset + done)
        if not chunk:
            raise IOError(f"short LM-head read at {offset}: {done}/{size} bytes")
        parts.append(chunk)
        done += len(chunk)
    return b"".join(parts)


class StreamedLMHead:
    """Drop-in replacement for a materialized lm_head.weight mx.array. Pass an
    instance of this where `layer_runner.final_logits`/`all_logits` expect the
    weight tensor; they special-case it to call `.logits(h)` in row blocks
    instead of a single `quant.matmul`."""

    def __init__(self, model_dir, weight_map: dict, name: str = "lm_head.weight",
                 block_rows: int = 16384, real_name: str | None = None):
        self.model_dir = Path(model_dir)
        self.weight_map = weight_map
        self.name = name
        # 2026-07-19: `name` is the CANONICAL key (used to find the shard
        # FILE via weight_map, which is keyed by canonical names -- see
        # WeightStore's language_model.* prefix remap in model_loader.py).
        # The tensor's actual key WITHIN that shard's own safetensors
        # header can differ (e.g. Kimi K2.5's real on-disk name is
        # "language_model.lm_head.weight", not "lm_head.weight") -- that
        # remap lives in WeightStore._real_name, not in this store-agnostic
        # utility, so the caller must pass it through explicitly.
        self.real_name = real_name or name
        self.block_rows = block_rows
        self.candidate_read_calls = 0
        self.candidate_read_extents = 0
        self.candidate_rows_requested = 0
        self.candidate_unique_rows_read = 0
        self.candidate_bytes_read = 0
        self.candidate_recall_full_scan_calls = 0
        self.candidate_recall_full_scan_bytes = 0
        # Full-vocabulary scans used by ordinary target and draft logits.
        # These reads bypass WeightStore, so keep explicit cumulative counters
        # for request-local delta attribution in the serving wrappers.
        self.full_scan_calls = 0
        self.full_read_extents = 0
        self.full_bytes_read = 0
        self.full_read_ns = 0
        self.full_scan_ns = 0
        self._open()

    def _open(self):
        path = self.model_dir / self.weight_map[self.name]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        meta = header[self.real_name]
        self.vocab, self.hidden = meta["shape"]
        self.dtype = meta["dtype"]
        self.row_bytes = self.hidden * _DTYPE_BYTES[self.dtype]
        self.data_start = 8 + n + meta["data_offsets"][0]
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)

    def close(self):
        os.close(self._fd)

    def logits(self, h: mx.array) -> mx.array:
        """h: (..., hidden) already rms-normed. Returns (..., vocab). Peak
        Metal cost per block is O(block_rows * hidden), not O(vocab * hidden)."""
        scan_started_ns = time.perf_counter_ns()
        read_ns = 0
        read_extents = 0
        read_bytes = 0
        mx.eval(h)
        chunks = []
        for start in range(0, self.vocab, self.block_rows):
            n_rows = min(self.block_rows, self.vocab - start)
            read_started_ns = time.perf_counter_ns()
            raw = _pread_exact(self._fd, n_rows * self.row_bytes,
                               self.data_start + start * self.row_bytes)
            read_ns += time.perf_counter_ns() - read_started_ns
            read_extents += 1
            read_bytes += len(raw)
            block = np.frombuffer(
                raw, dtype=_NP_STORAGE_DTYPE[self.dtype]
            ).reshape(n_rows, self.hidden)
            w_block = mx.array(block).view(_MX_DTYPE[self.dtype])
            c = h @ w_block.T
            mx.eval(c)
            chunks.append(c)
        self._record_full_scan(
            read_extents, read_bytes, read_ns,
            time.perf_counter_ns() - scan_started_ns)
        return mx.concatenate(chunks, axis=-1)

    def logits_serial_rows(self, h: mx.array) -> mx.array:
        """Stream each vocab block once while preserving one-row matmul shapes.

        Speculative verification deliberately evaluates target positions with
        the same one-position GEMM shapes as ordinary greedy decode. Calling
        :meth:`logits` once per position preserves that arithmetic but rereads
        the complete LM head for every position. A single batched call reads
        the head once, but can select a different reduction kernel.

        This method keeps those concerns separate: each position remains an
        independent ``(1, hidden) @ (hidden, block_rows)`` operation, while all
        positions share the one physical read of each vocab-row block.
        """
        if h.shape[-1] != self.hidden:
            raise ValueError(
                f"LM-head hidden width {h.shape[-1]} != {self.hidden}"
            )
        mx.eval(h)
        leading_shape = tuple(int(value) for value in h.shape[:-1])
        flat = h.reshape(-1, self.hidden)
        rows = int(flat.shape[0])
        if rows == 0:
            return mx.zeros((*leading_shape, self.vocab), dtype=h.dtype)

        scan_started_ns = time.perf_counter_ns()
        read_ns = 0
        read_extents = 0
        read_bytes = 0
        row_chunks: list[list[mx.array]] = [[] for _ in range(rows)]
        for start in range(0, self.vocab, self.block_rows):
            n_rows = min(self.block_rows, self.vocab - start)
            read_started_ns = time.perf_counter_ns()
            raw = _pread_exact(
                self._fd,
                n_rows * self.row_bytes,
                self.data_start + start * self.row_bytes,
            )
            read_ns += time.perf_counter_ns() - read_started_ns
            read_extents += 1
            read_bytes += len(raw)
            block = np.frombuffer(
                raw, dtype=_NP_STORAGE_DTYPE[self.dtype]
            ).reshape(n_rows, self.hidden)
            w_block = mx.array(block).view(_MX_DTYPE[self.dtype])
            values = [
                flat[row : row + 1] @ w_block.T
                for row in range(rows)
            ]
            mx.eval(*values)
            for chunks, value in zip(row_chunks, values, strict=True):
                chunks.append(value)

        result_rows = [
            mx.concatenate(chunks, axis=-1)
            for chunks in row_chunks
        ]
        result = mx.concatenate(result_rows, axis=0)
        self._record_full_scan(
            read_extents, read_bytes, read_ns,
            time.perf_counter_ns() - scan_started_ns)
        return result.reshape(*leading_shape, self.vocab)

    def _record_full_scan(
            self, read_extents: int, read_bytes: int, read_ns: int,
            scan_ns: int) -> None:
        self.full_scan_calls += 1
        self.full_read_extents += int(read_extents)
        self.full_bytes_read += int(read_bytes)
        self.full_read_ns += int(read_ns)
        self.full_scan_ns += int(scan_ns)

    def full_scan_telemetry(self) -> dict[str, int]:
        return {
            "full_scan_calls": int(self.full_scan_calls),
            "full_read_extents": int(self.full_read_extents),
            "full_bytes_read": int(self.full_bytes_read),
            "full_read_ns": int(self.full_read_ns),
            "full_scan_ns": int(self.full_scan_ns),
        }

    def candidate_logits(
        self, h: mx.array, indices: mx.array,
    ) -> mx.array:
        """Score only candidate rows, coalescing sorted adjacent disk reads.

        ``indices`` has the same leading dimensions as ``h`` plus a final
        candidate axis. The physical reads are the union across every leading
        row: IDs are sorted once, adjacent IDs become one ``pread``, and the
        loaded BF16 rows are remapped to each caller's original candidate
        order before ``gather_mm``. No complete head tensor is materialized.
        """

        if h.shape[-1] != self.hidden:
            raise ValueError(
                f"LM-head hidden width {h.shape[-1]} != {self.hidden}")
        if tuple(h.shape[:-1]) != tuple(indices.shape[:-1]):
            raise ValueError(
                f"candidate index shape {indices.shape} does not match "
                f"hidden leading shape {h.shape[:-1]}")
        mx.eval(h, indices)
        requested = np.asarray(indices).astype(np.int64, copy=False)
        if requested.ndim == 0 or requested.shape[-1] <= 0:
            raise ValueError("candidate indices must have a non-empty final axis")
        if np.any(requested < 0) or np.any(requested >= self.vocab):
            raise ValueError("candidate row index is outside the LM-head vocabulary")
        unique = np.unique(requested.reshape(-1))

        blocks: list[np.ndarray] = []
        extents = 0
        start = 0
        while start < len(unique):
            end = start + 1
            while end < len(unique) and unique[end] == unique[end - 1] + 1:
                end += 1
            first = int(unique[start])
            n_rows = end - start
            raw = _pread_exact(
                self._fd, n_rows * self.row_bytes,
                self.data_start + first * self.row_bytes)
            blocks.append(np.frombuffer(
                raw, dtype=_NP_STORAGE_DTYPE[self.dtype]
            ).reshape(n_rows, self.hidden))
            extents += 1
            start = end
        storage = np.concatenate(blocks, axis=0)
        exact_rows = mx.array(storage).view(_MX_DTYPE[self.dtype])
        mapped = np.searchsorted(unique, requested).astype(np.int32)
        flat = h.reshape(-1, self.hidden)
        flat_indices = mx.array(mapped.reshape(-1, requested.shape[-1]))
        lhs = mx.expand_dims(flat, (-2, -3))
        rhs = mx.expand_dims(exact_rows, -2).swapaxes(-1, -2)
        scores = mx.gather_mm(
            lhs, rhs, rhs_indices=flat_indices
        ).squeeze((-1, -2)).reshape(indices.shape)
        mx.eval(scores)

        self.candidate_read_calls += 1
        self.candidate_read_extents += extents
        self.candidate_rows_requested += int(requested.size)
        self.candidate_unique_rows_read += int(len(unique))
        self.candidate_bytes_read += int(len(unique) * self.row_bytes)
        return scores

    def candidate_telemetry(self) -> dict[str, int]:
        return {
            "candidate_read_calls": int(self.candidate_read_calls),
            "candidate_read_extents": int(self.candidate_read_extents),
            "candidate_rows_requested": int(self.candidate_rows_requested),
            "candidate_unique_rows_read": int(self.candidate_unique_rows_read),
            "candidate_bytes_read": int(self.candidate_bytes_read),
            "candidate_recall_full_scan_calls": int(
                self.candidate_recall_full_scan_calls),
            "candidate_recall_full_scan_bytes": int(
                self.candidate_recall_full_scan_bytes),
        }

    def candidate_recall_logits(self, h: mx.array) -> mx.array:
        """Full exact oracle used only by an explicitly sampled recall probe.

        This preserves one-position LM-head arithmetic while scanning the
        source once for all positions. It never retains the full BF16 matrix,
        but it intentionally reads every source row and is therefore disabled
        by default in serving profiles.
        """

        logits = self.logits_serial_rows(h)
        self.candidate_recall_full_scan_calls += 1
        self.candidate_recall_full_scan_bytes += int(
            self.vocab * self.row_bytes)
        return logits
