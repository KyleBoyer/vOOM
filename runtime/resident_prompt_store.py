"""Exact, content-addressed prompt endpoints for the resident MLX-LM backend."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import mlx.core as mx


_FORMAT = "resident-mlx-prompt-v2"


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ResidentPromptStore:
    """Persist exact MLX-LM cache arrays plus the raw prompt distribution.

    Entries are immutable and become visible only when their checksummed JSON
    commit record is atomically published.  The key includes every prompt token
    and a model/runtime/arithmetic fingerprint; a different subject, tool set,
    template, checkpoint, or kernel path therefore cannot collide.
    """

    def __init__(
        self, directory: str | Path, fingerprint: str, *,
        max_bytes: int = 4_000_000_000,
    ):
        if max_bytes <= 0:
            raise ValueError("resident prompt-store max_bytes must be positive")
        self.root = Path(directory).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = self.root / fingerprint[:24]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.max_bytes = int(max_bytes)
        # One root lock makes the byte budget global across runtime/model
        # fingerprints instead of leaking one full budget after every upgrade.
        self.lock_path = self.root / ".lock"
        self.lock_path.touch(exist_ok=True)

    def key(self, prompt_ids) -> str:
        digest = hashlib.sha256()
        digest.update(_FORMAT.encode())
        digest.update(b"\0")
        digest.update(self.fingerprint.encode())
        digest.update(b"\0")
        for token in prompt_ids:
            digest.update(int(token).to_bytes(4, "little", signed=False))
        return digest.hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        return (
            self.directory / f"{key}.cache.safetensors",
            self.directory / f"{key}.logits.safetensors",
            self.directory / f"{key}.json",
        )

    def _read_manifest(self, key: str) -> dict | None:
        _cache_path, _logits_path, manifest_path = self._paths(key)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, TypeError):
            return None
        if not (
            isinstance(manifest, dict)
            and manifest.get("format") == _FORMAT
            and manifest.get("key") == key
            and manifest.get("fingerprint") == self.fingerprint
        ):
            return None
        return manifest

    @staticmethod
    def _valid_payload(path: Path, record: dict) -> bool:
        try:
            return (
                path.is_file()
                and path.stat().st_size == int(record["bytes"])
                and _sha256_file(path) == record["sha256"])
        except (OSError, KeyError, TypeError, ValueError):
            return False

    def load(self, prompt_ids):
        """Return the exact prompt endpoint and optional decode chain."""
        # mlx-lm 0.31.3 registers a tokenizer with a string key that
        # Transformers 5.13.0 rejects. ResidentMLXLMEngine normally applies
        # the repository's narrow compatibility shim before this store is
        # reached, but the store is also a public, independently tested
        # component. Make that ordering guarantee explicit here too.
        from .resident_mlx_lm import import_mlx_lm
        import_mlx_lm()
        from mlx_lm.models.cache import load_prompt_cache

        key = self.key(prompt_ids)
        cache_path, logits_path, manifest_path = self._paths(key)
        started = time.perf_counter()
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            manifest = self._read_manifest(key)
            if manifest is None:
                return None
            if int(manifest.get("prompt_tokens", -1)) != len(prompt_ids):
                return None
            if not (
                self._valid_payload(cache_path, manifest.get("cache", {}))
                and self._valid_payload(
                    logits_path, manifest.get("logits", {}))
            ):
                return None
            try:
                cache, metadata = load_prompt_cache(
                    str(cache_path), return_metadata=True)
                logits_payload = mx.load(str(logits_path))
                logits = logits_payload["prompt_logits"]
                generation_count = int(
                    manifest.get("generation_logits", 0))
                generation_tokens = (
                    logits_payload["generation_tokens"]
                    if generation_count else None)
                generation_logits = (
                    (logits,) + tuple(
                        logits_payload[f"generation_logits_{index:04d}"]
                        for index in range(1, generation_count))
                    if generation_count else ())
                if (
                    metadata.get("format") != _FORMAT
                    or metadata.get("key") != key
                    or metadata.get("fingerprint") != self.fingerprint
                    or len(cache) != int(manifest["cache_layers"])
                    or logits.ndim != 1
                    or generation_count < 0
                    or generation_count > 512
                    or (
                        generation_count
                        and (
                            generation_tokens.ndim != 1
                            or generation_tokens.size != generation_count))
                    or any(
                        item.ndim != 1 or item.size != logits.size
                        for item in generation_logits)
                ):
                    return None
                mx.eval(
                    logits,
                    generation_tokens if generation_tokens is not None else (),
                    generation_logits,
                    [entry.state for entry in cache],
                )
            except (OSError, KeyError, RuntimeError, TypeError, ValueError):
                return None
            now = time.time()
            os.utime(manifest_path, (now, now))
        return (
            cache,
            logits,
            (
                tuple(int(token) for token in generation_tokens.tolist())
                if generation_tokens is not None else ()),
            generation_logits,
            {
                "hit": 1,
                "key": key,
                "load_s": time.perf_counter() - started,
                "cache_bytes": int(manifest["cache"]["bytes"]),
                "logits_bytes": int(manifest["logits"]["bytes"]),
                "generation_logits": generation_count,
            },
        )

    def save(
        self,
        prompt_ids,
        cache,
        raw_logits,
        generation_tokens=(),
        generation_step_logits=(),
    ) -> dict:
        """Publish one exact endpoint if it is not already committed."""
        from .resident_mlx_lm import import_mlx_lm
        import_mlx_lm()
        from mlx_lm.models.cache import save_prompt_cache

        key = self.key(prompt_ids)
        cache_path, logits_path, manifest_path = self._paths(key)
        started = time.perf_counter()
        with self.lock_path.open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = self._read_manifest(key)
            existing_valid = (
                existing is not None
                and int(existing.get("prompt_tokens", -1)) == len(prompt_ids)
                and self._valid_payload(
                    cache_path, existing.get("cache", {}))
                and self._valid_payload(
                    logits_path, existing.get("logits", {}))
            )
            if existing_valid:
                return {
                    "saved": 0,
                    "key": key,
                    "save_s": time.perf_counter() - started,
                    "cache_bytes": int(
                        existing.get("cache", {}).get("bytes", 0)),
                    "logits_bytes": int(
                        existing.get("logits", {}).get("bytes", 0)),
                    "generation_logits": int(
                        existing.get("generation_logits", 0)),
                }

            # A commit record is the visibility boundary.  Remove any invalid
            # record first so a torn/corrupt entry can be repaired by the
            # normal atomic publication below; payload paths are key-scoped
            # and protected by this process-shared lock.
            if (
                manifest_path.exists()
                or cache_path.exists()
                or logits_path.exists()
            ):
                manifest_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                logits_path.unlink(missing_ok=True)
                _fsync_dir(self.directory)

            unique = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
            cache_tmp = self.directory / f"{unique}.cache.safetensors"
            logits_tmp = self.directory / f"{unique}.logits.safetensors"
            manifest_tmp = None
            try:
                generation_count = min(
                    len(generation_tokens), len(generation_step_logits), 512)
                retained_generation_tokens = mx.array(
                    generation_tokens[:generation_count], dtype=mx.int32)
                retained_generation_logits = tuple(
                    generation_step_logits[:generation_count])
                mx.eval(
                    raw_logits,
                    retained_generation_tokens,
                    retained_generation_logits,
                    [entry.state for entry in cache],
                )
                metadata = {
                    "format": _FORMAT,
                    "fingerprint": self.fingerprint,
                    "key": key,
                    "prompt_tokens": str(len(prompt_ids)),
                }
                save_prompt_cache(str(cache_tmp), cache, metadata)
                logits_payload = {
                    "prompt_logits": raw_logits,
                }
                if generation_count:
                    logits_payload["generation_tokens"] = (
                        retained_generation_tokens)
                logits_payload.update({
                    f"generation_logits_{index:04d}": values
                    for index, values in enumerate(
                        retained_generation_logits[1:], start=1)
                })
                mx.save_safetensors(
                    str(logits_tmp), logits_payload)
                _fsync_file(cache_tmp)
                _fsync_file(logits_tmp)
                cache_record = {
                    "bytes": cache_tmp.stat().st_size,
                    "sha256": _sha256_file(cache_tmp),
                }
                logits_record = {
                    "bytes": logits_tmp.stat().st_size,
                    "sha256": _sha256_file(logits_tmp),
                }
                manifest = {
                    "format": _FORMAT,
                    "fingerprint": self.fingerprint,
                    "key": key,
                    "prompt_tokens": len(prompt_ids),
                    "cache_layers": len(cache),
                    "generation_logits": generation_count,
                    "cache": cache_record,
                    "logits": logits_record,
                    "created_ns": time.time_ns(),
                }
                os.replace(cache_tmp, cache_path)
                os.replace(logits_tmp, logits_path)
                manifest_tmp = self.directory / f"{unique}.json"
                with manifest_tmp.open("xb") as stream:
                    stream.write(_canonical_json(manifest))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest_tmp, manifest_path)
                _fsync_dir(self.directory)
                self._gc_locked()
            finally:
                cache_tmp.unlink(missing_ok=True)
                logits_tmp.unlink(missing_ok=True)
                if manifest_tmp is not None:
                    manifest_tmp.unlink(missing_ok=True)
        return {
            "saved": 1,
            "key": key,
            "save_s": time.perf_counter() - started,
            "cache_bytes": int(cache_record["bytes"]),
            "logits_bytes": int(logits_record["bytes"]),
            "generation_logits": generation_count,
        }

    def _gc_locked(self) -> None:
        entries = []
        total = 0
        for manifest_path in self.root.glob("*/*.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
                key = manifest["key"]
                cache_path = manifest_path.with_name(
                    f"{key}.cache.safetensors")
                logits_path = manifest_path.with_name(
                    f"{key}.logits.safetensors")
                size = (
                    manifest_path.stat().st_size
                    + cache_path.stat().st_size
                    + logits_path.stat().st_size)
                entries.append((
                    manifest_path.stat().st_mtime_ns,
                    size,
                    manifest_path,
                    cache_path,
                    logits_path,
                ))
                total += size
            except (OSError, KeyError, TypeError, ValueError):
                continue
        touched_directories = set()
        for _mtime, size, manifest, cache, logits in sorted(entries):
            if total <= self.max_bytes:
                break
            touched_directories.add(manifest.parent)
            manifest.unlink(missing_ok=True)
            cache.unlink(missing_ok=True)
            logits.unlink(missing_ok=True)
            total -= size
        for directory in touched_directories:
            _fsync_dir(directory)
        _fsync_dir(self.root)
