"""F02: virtualized embedding lookup — row-paged input embeddings.

The input embedding matrix (gpt-oss 1.16 GB, GLM-5.2 1.9 GB) is only ever READ one
row per token, yet the engine pinned the whole tensor in Metal memory. This class
materializes a raw row-major bf16 sidecar file once (bit-exact bytes), then serves
lookups with per-row preads + a small LRU row cache. Frees the full tensor's RAM
for the expert cache, at the cost of ~12 KB of disk reads per new token.

Only valid for untied-embedding models (tied models reuse the tensor as lm_head).
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import struct
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .local_config import get_storage_config


class EmbedRows:
    def __init__(self, model_dir: str | Path, store, hidden: int, row_cache: int = 8192):
        model_dir = Path(model_dir)
        if get_storage_config().is_configured_path(model_dir):
            # Externally-hosted model: keep the sidecar on LOCAL disk — faster
            # row reads and immune to a dropped/cycled mountpoint (an open fd
            # on a dropped share goes stale and kills the run).
            local = Path(__file__).resolve().parent.parent / "models" / f"{model_dir.name}.embed_rows"
            local.mkdir(parents=True, exist_ok=True)
            self.path = local / "embed_rows.bin"
        else:
            self.path = model_dir / "embed_rows.bin"
        self.meta_path = self.path.with_suffix(".meta.json")
        self.hash_path = self.path.with_suffix(".rowsha256")
        self.hidden = hidden
        self.row_bytes = hidden * 2  # bf16
        self.rows = store.config.vocab_size
        self.expected_bytes = self.rows * self.row_bytes
        self._source = self._source_descriptor(store)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cap = row_cache
        self.reads = 0
        self.hits = 0
        self._fd = -1
        self._row_hashes = b""
        if not self._is_current():
            self._materialize(store)
        self._fd = os.open(self.path, os.O_RDONLY)

    @staticmethod
    def _source_descriptor(store) -> dict:
        """Cheap operational identity for ordinary stale-sidecar detection.

        It binds the canonical tensor name, real on-disk name, container name,
        size/mtime, and config stat. Materialization also records the exact tensor
        SHA-256. Size/mtime is not a proof-grade checkpoint attestation: a
        same-stat replacement would require an immutable manifest/canary (or a
        1.9 GB source reread) to bind that recorded digest at engine-up.
        """
        name = "model.embed_tokens.weight"
        shard = store.weight_map.get(name, "")
        real_name = store._real_name.get(name, name)
        candidates = []
        if store.vpack2 is not None:
            candidates.append(store.vpack2.archive)
        candidates.extend([store.dir / shard, store.vpack / shard])
        path = next((p for p in candidates if p.exists()), None)
        stat = path.stat() if path is not None else None
        config_path = store.dir / "config.json"
        config_stat = config_path.stat()
        return {
            "tensor": name,
            "real_tensor": real_name,
            "container": shard,
            "resolved_container": path.name if path is not None else None,
            "container_size": stat.st_size if stat else None,
            "container_mtime_ns": stat.st_mtime_ns if stat else None,
            "config_size": config_stat.st_size,
            "config_mtime_ns": config_stat.st_mtime_ns,
        }

    def _is_current(self) -> bool:
        try:
            meta = json.loads(self.meta_path.read_text())
            legacy_source = dict(self._source)
            legacy_source.pop("resolved_container", None)
            source_matches = meta.get("source") in (self._source, legacy_source)
            common = (
                self.path.stat().st_size == self.expected_bytes
                and meta.get("hidden") == self.hidden
                and meta.get("rows") == self.rows
                and meta.get("bytes") == self.expected_bytes
                and source_matches
                and isinstance(meta.get("sha256"), str)
                and len(meta["sha256"]) == 64
            )
            if not common:
                return False
            if meta.get("version") == 2:
                # One-time sequential upgrade: authenticate the old whole-file
                # digest and derive per-row hashes without rewriting 1.9 GB.
                return self._upgrade_v2(meta)
            if meta.get("version") != 3:
                return False
            try:
                hashes = self.hash_path.read_bytes()
            except OSError:
                return self._upgrade_v2(meta)
            if (len(hashes) != self.rows * 32
                    or hashlib.sha256(hashes).hexdigest()
                    != meta.get("row_hashes_sha256")):
                return self._upgrade_v2(meta)
            self._row_hashes = hashes
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _publish_meta(self, meta: dict) -> None:
        self._atomic_write(
            self.meta_path, json.dumps(meta, sort_keys=True).encode() + b"\n"
        )
        dir_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _upgrade_v2(self, meta: dict) -> bool:
        """Authenticate a v2 payload once and add exact per-row hashes."""
        print(f"[embed_rows] upgrading v2 integrity metadata for {self.path}", flush=True)
        digest = hashlib.sha256()
        row_hashes = bytearray()
        fd = os.open(self.path, os.O_RDONLY)
        try:
            rows_per_block = max(1, (64 * 1024 * 1024) // self.row_bytes)
            byte_block = rows_per_block * self.row_bytes
            for offset in range(0, self.expected_bytes, byte_block):
                raw = self._pread_exact(
                    fd, min(byte_block, self.expected_bytes - offset), offset
                )
                digest.update(raw)
                for start in range(0, len(raw), self.row_bytes):
                    row_hashes.extend(
                        hashlib.sha256(raw[start : start + self.row_bytes]).digest()
                    )
        finally:
            os.close(fd)
        if digest.hexdigest() != meta["sha256"] or len(row_hashes) != self.rows * 32:
            return False
        hashes = bytes(row_hashes)
        self._atomic_write(self.hash_path, hashes)
        meta = dict(meta)
        meta.update({
            "version": 3,
            "row_hashes_file": self.hash_path.name,
            "row_hashes_sha256": hashlib.sha256(hashes).hexdigest(),
            "source": self._source,
        })
        self._publish_meta(meta)
        self._row_hashes = hashes
        return True

    @staticmethod
    def _pread_exact(fd: int, size: int, offset: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            part = os.pread(fd, remaining, offset + size - remaining)
            if not part:
                raise OSError(
                    f"short pread at offset {offset}: wanted {size}, got {size - remaining}"
                )
            chunks.append(part)
            remaining -= len(part)
        return b"".join(chunks)

    def _raw_safetensor_extent(self, store):
        """Return (path, absolute offset, bytes) for a plain BF16 embedding."""
        if store.packed:
            return None
        name = "model.embed_tokens.weight"
        path = store.dir / store.weight_map[name]
        fd = os.open(path, os.O_RDONLY)
        try:
            header_len = struct.unpack("<Q", self._pread_exact(fd, 8, 0))[0]
            header = json.loads(self._pread_exact(fd, header_len, 8))
        finally:
            os.close(fd)
        entry = header[store._real_name.get(name, name)]
        if entry["dtype"] != "BF16" or entry["shape"] != [self.rows, self.hidden]:
            raise ValueError(f"unexpected embedding metadata: {entry}")
        begin, end = entry["data_offsets"]
        if end - begin != self.expected_bytes:
            raise ValueError("embedding safetensor extent size does not match config")
        return path, 8 + header_len + begin, end - begin

    def _materialize(self, store):
        # Plain safetensors copy the exact tensor extent directly; do not call
        # WeightStore.fetch(), which materializes GLM's entire ~1.9 GB embedding
        # before the old block loop even begins.
        free = shutil.disk_usage(self.path.parent).free
        if free < self.expected_bytes + 256 * 1024 * 1024:
            raise OSError(
                f"not enough free space for embedding sidecar temp: need "
                f"{self.expected_bytes + 256 * 1024 * 1024}, have {free}"
            )
        block = max(1, (64 * 1024 * 1024) // self.row_bytes)
        tmp = self.path.with_name(self.path.name + ".tmp")
        digest = hashlib.sha256()
        row_hashes = bytearray()

        def write_exact_rows(f, raw: bytes) -> None:
            if len(raw) % self.row_bytes:
                raise ValueError("embedding materialization block splits a row")
            f.write(raw)
            digest.update(raw)
            for start in range(0, len(raw), self.row_bytes):
                row_hashes.extend(
                    hashlib.sha256(raw[start : start + self.row_bytes]).digest()
                )

        with open(tmp, "wb") as f:
            extent = self._raw_safetensor_extent(store)
            if extent is not None:
                source, source_offset, total = extent
                src_fd = os.open(source, os.O_RDONLY)
                try:
                    byte_block = block * self.row_bytes
                    for offset in range(0, total, byte_block):
                        raw = self._pread_exact(
                            src_fd, min(byte_block, total - offset), source_offset + offset
                        )
                        write_exact_rows(f, raw)
                finally:
                    os.close(src_fd)
            else:
                # Packed fallback is correct but can materialize the whole tensor;
                # packed GLM should gain a row-addressable sidecar extent before
                # this path is considered memory-safe.
                w, _, _ = store.fetch(["model.embed_tokens.weight"])
                arr = w["model.embed_tokens.weight"].view(mx.uint16)
                if list(arr.shape) != [self.rows, self.hidden]:
                    raise ValueError(f"unexpected embedding shape {arr.shape}")
                for i in range(0, self.rows, block):
                    raw = np.array(arr[i:i + block], copy=False).tobytes()
                    write_exact_rows(f, raw)
                del w, arr
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        if len(row_hashes) != self.rows * 32:
            raise ValueError(
                f"row hash count mismatch: {len(row_hashes) // 32} != {self.rows}"
            )
        hashes = bytes(row_hashes)
        self._atomic_write(self.hash_path, hashes)
        meta = {
            "version": 3,
            "hidden": self.hidden,
            "rows": self.rows,
            "bytes": self.expected_bytes,
            "sha256": digest.hexdigest(),
            "row_hashes_file": self.hash_path.name,
            "row_hashes_sha256": hashlib.sha256(hashes).hexdigest(),
            "source": self._source,
        }
        self._publish_meta(meta)
        self._row_hashes = hashes
        mx.clear_cache()
        print(f"[embed_rows] materialized {self.path.stat().st_size / 1e9:.2f}GB sidecar")

    def lookup(self, tokens: list[int]) -> mx.array:
        """tokens -> (1, L, hidden) bf16, bit-exact rows."""
        rows = []
        for t in tokens:
            cached = self._cache.get(t)
            if cached is not None:
                self._cache.move_to_end(t)
                self.hits += 1
            else:
                if t < 0 or t >= self.rows:
                    raise IndexError(f"embedding token id {t} outside [0, {self.rows})")
                raw = self._read_verified_row(t)
                cached = np.frombuffer(raw, dtype=np.uint16)
                self.reads += 1
                self._cache[t] = cached
                if len(self._cache) > self._cap:
                    self._cache.popitem(last=False)
            rows.append(cached)
        out = mx.array(np.stack(rows)).view(mx.bfloat16).reshape(1, len(tokens), self.hidden)
        mx.eval(out)
        return out

    def _read_verified_row(self, token_id: int) -> bytes:
        """Read one exact row and fail before MLX sees corrupt bytes."""
        raw = self._pread_exact(
            self._fd, self.row_bytes, token_id * self.row_bytes
        )
        expected = self._row_hashes[token_id * 32 : (token_id + 1) * 32]
        actual = hashlib.sha256(raw).digest()
        if len(expected) != 32 or actual != expected:
            raise IOError(f"embedding sidecar row hash mismatch at token {token_id}")
        return raw

    def close(self):
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
