"""Content-addressed semantic embeddings for large tool catalogs.

The serving model must not share a long-lived process with a second encoder on
the 16-GB target host.  This module therefore separates the two lifetimes:

* ``build`` verifies the pinned local BGE artifact and embeds tool capsules
  offline, before the serving model is loaded.
* request-time retrieval reads those immutable tool vectors, embeds only a
  short model-authored query in a disposable CPU subprocess, then releases all
  encoder memory when that subprocess exits.

Raw tool schemas and queries are never written to the cache. Cache objects are
keyed by a salted content hash and contain only normalized float32 vectors plus
an integrity sidecar. Corruption or incomplete catalog coverage fails back to
the deterministic lexical/alias ranker; it never silently ranks a half-embedded
catalog.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "tool-embed-bge-small-en-v1.5"
DEFAULT_CACHE_DIR = _REPO_ROOT / ".tool_embeddings"

ENCODER_REPO = "BAAI/bge-small-en-v1.5"
ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ENCODER_WEIGHTS_SHA256 = (
    "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad")
ENCODER_WEIGHTS_BYTES = 133_466_304
ENCODER_DIMENSION = 384
ENCODER_MAX_TOKENS = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_SCHEMA = "vmodel-tool-embedding-object-v1"
PROFILE = "bge-small-en-v1.5-cls-l2-capsule-v1"
_VERIFY_STAMP = ".vmodel_embedding_verified.json"


class ToolEmbeddingError(RuntimeError):
    """A semantic retrieval problem that should normally trigger fallback."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model_dir: Path = DEFAULT_MODEL_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    semantic_weight: float = 0.60
    timeout_seconds: float = 30.0
    min_available_mb: int = 4800
    query_cache_max: int = 2048

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        try:
            weight = float(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_WEIGHT", "0.60"))
            timeout = float(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_TIMEOUT", "30"))
            min_available = int(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_MIN_AVAILABLE_MB", "4800"))
            query_cache_max = int(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_QUERY_CACHE_MAX", "2048"))
        except ValueError as error:
            raise ToolEmbeddingError(
                "tool embedding weight/timeout must be numeric") from error
        if not 0.0 <= weight <= 1.0:
            raise ToolEmbeddingError(
                "VMODEL_TOOL_EMBEDDING_WEIGHT must be in [0, 1]")
        if not 1.0 <= timeout <= 120.0:
            raise ToolEmbeddingError(
                "VMODEL_TOOL_EMBEDDING_TIMEOUT must be in [1, 120]")
        if not 4000 <= min_available <= 12000:
            raise ToolEmbeddingError(
                "VMODEL_TOOL_EMBEDDING_MIN_AVAILABLE_MB must be in [4000, 12000]")
        if not 0 <= query_cache_max <= 100_000:
            raise ToolEmbeddingError(
                "VMODEL_TOOL_EMBEDDING_QUERY_CACHE_MAX must be in [0, 100000]")
        return cls(
            model_dir=Path(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_MODEL", str(DEFAULT_MODEL_DIR))),
            cache_dir=Path(os.environ.get(
                "VMODEL_TOOL_EMBEDDING_CACHE", str(DEFAULT_CACHE_DIR))),
            semantic_weight=weight,
            timeout_seconds=timeout,
            min_available_mb=min_available,
            query_cache_max=query_cache_max,
        )


def embeddings_enabled() -> bool:
    """Whether the hybrid scorer is requested for fast tool retrieval.

    ``auto`` is the safe default: use an already-built/verified local cache,
    otherwise retain lexical behavior. ``1`` still falls back on an individual
    request unless ``VMODEL_TOOL_EMBEDDINGS_REQUIRED=1`` is also set.
    """
    value = os.environ.get("VMODEL_TOOL_EMBEDDINGS", "auto").lower()
    if value not in ("0", "1", "auto"):
        raise ToolEmbeddingError(
            "VMODEL_TOOL_EMBEDDINGS must be 0, 1, or auto")
    if value == "0":
        return False
    config = EmbeddingConfig.from_env()
    if value == "1":
        return True
    return (config.model_dir / _VERIFY_STAMP).is_file() and config.cache_dir.is_dir()


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def verify_encoder(model_dir: Path = DEFAULT_MODEL_DIR, *, full_hash: bool) -> dict:
    """Validate the pinned local artifact; full hashing is offline-build only."""
    model_dir = Path(model_dir)
    weights = model_dir / "model.safetensors"
    required = (
        weights, model_dir / "config.json", model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ToolEmbeddingError(
            "pinned tool embedding model is incomplete: " + ", ".join(missing))
    stat = weights.stat()
    if stat.st_size != ENCODER_WEIGHTS_BYTES:
        raise ToolEmbeddingError(
            f"unexpected encoder size {stat.st_size}; expected {ENCODER_WEIGHTS_BYTES}")

    support_paths = [path for path in required if path != weights]
    support_sha256 = {
        path.name: _sha256_file(path) for path in support_paths
    }
    stamp_path = model_dir / _VERIFY_STAMP
    if not full_hash and stamp_path.is_file():
        try:
            stamp = json.loads(stamp_path.read_text())
        except (OSError, ValueError):
            stamp = {}
        if (stamp.get("schema") == CACHE_SCHEMA
                and stamp.get("revision") == ENCODER_REVISION
                and stamp.get("weights_sha256") == ENCODER_WEIGHTS_SHA256
                and stamp.get("weights_bytes") == stat.st_size
                and stamp.get("weights_mtime_ns") == stat.st_mtime_ns
                and stamp.get("weights_ctime_ns") == stat.st_ctime_ns
                and stamp.get("support_sha256") == support_sha256):
            return stamp
    if not full_hash:
        raise ToolEmbeddingError(
            "encoder is not offline-verified; run `python -m "
            "runtime.tool_embeddings build ...` before serving")

    actual = _sha256_file(weights)
    if actual != ENCODER_WEIGHTS_SHA256:
        raise ToolEmbeddingError(
            f"encoder SHA-256 mismatch: expected {ENCODER_WEIGHTS_SHA256}, got {actual}")
    stamp = {
        "schema": CACHE_SCHEMA,
        "profile": PROFILE,
        "repo": ENCODER_REPO,
        "revision": ENCODER_REVISION,
        "weights_sha256": actual,
        "weights_bytes": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
        "weights_ctime_ns": stat.st_ctime_ns,
        "support_sha256": support_sha256,
    }
    _atomic_write(stamp_path, _canonical_json(stamp) + b"\n")
    return stamp


def _object_id(kind: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(PROFILE.encode("ascii"))
    digest.update(b"\0")
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _normalize_vector(vector) -> "object":
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (ENCODER_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ToolEmbeddingError(
            f"invalid encoder vector shape/content: {array.shape}")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ToolEmbeddingError("encoder returned a zero/non-finite vector")
    return array / norm


class ToolEmbeddingCache:
    """Small integrity-checked, content-addressed vector object store."""

    def __init__(self, root: Path = DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.lock_path = self.root / ".lock"

    @contextlib.contextmanager
    def lock(self, *, exclusive: bool):
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _paths(self, object_id: str) -> tuple[Path, Path]:
        base = self.objects / object_id[:2] / object_id
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def load(self, kind: str, text: str):
        import numpy as np

        object_id = _object_id(kind, text)
        payload_path, meta_path = self._paths(object_id)
        try:
            payload = payload_path.read_bytes()
            meta = json.loads(meta_path.read_text())
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not (
            meta.get("schema") == CACHE_SCHEMA
            and meta.get("profile") == PROFILE
            and meta.get("object_id") == object_id
            and meta.get("kind") == kind
            and meta.get("dimension") == ENCODER_DIMENSION
            and meta.get("payload_sha256") == hashlib.sha256(payload).hexdigest()
        ):
            return None
        try:
            vector = np.load(io.BytesIO(payload), allow_pickle=False)
            vector = _normalize_vector(vector)
        except (ValueError, OSError, ToolEmbeddingError):
            return None
        return vector

    def store(self, kind: str, text: str, vector) -> str:
        import numpy as np

        vector = _normalize_vector(vector)
        object_id = _object_id(kind, text)
        payload_path, meta_path = self._paths(object_id)
        buffer = io.BytesIO()
        np.save(buffer, vector, allow_pickle=False)
        payload = buffer.getvalue()
        meta = {
            "schema": CACHE_SCHEMA,
            "profile": PROFILE,
            "object_id": object_id,
            "kind": kind,
            "dimension": ENCODER_DIMENSION,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        # Publish payload first and the validating sidecar last. A reader can
        # observe a miss during a concurrent build, never a mismatched object.
        _atomic_write(payload_path, payload)
        _atomic_write(meta_path, _canonical_json(meta) + b"\n")
        return object_id

    def get_many(self, kind: str, texts: Iterable[str]):
        vectors = []
        hits = 0
        with self.lock(exclusive=False):
            for text in texts:
                vector = self.load(kind, text)
                vectors.append(vector)
                hits += int(vector is not None)
        return vectors, hits

    def _prune_kind(self, kind: str, keep: int) -> int:
        candidates = []
        for meta_path in self.objects.glob("*/*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                modified = meta_path.stat().st_mtime_ns
            except (OSError, ValueError):
                continue
            if meta.get("schema") == CACHE_SCHEMA and meta.get("kind") == kind:
                candidates.append((modified, meta_path))
        candidates.sort(reverse=True)
        removed = 0
        for _modified, meta_path in candidates[max(0, keep):]:
            with contextlib.suppress(FileNotFoundError):
                meta_path.with_suffix(".npy").unlink()
            with contextlib.suppress(FileNotFoundError):
                meta_path.unlink()
            removed += 1
        return removed

    def store_many(
            self, kind: str, texts: list[str], vectors, *,
            max_objects: int | None = None) -> int:
        if len(texts) != len(vectors):
            raise ToolEmbeddingError("encoder returned the wrong vector count")
        with self.lock(exclusive=True):
            for text, vector in zip(texts, vectors):
                self.store(kind, text, vector)
            return (self._prune_kind(kind, max_objects)
                    if max_objects is not None else 0)


def _decode_vectors(payload: dict):
    import numpy as np

    if payload.get("schema") != "vmodel-tool-encoder-result-v1":
        raise ToolEmbeddingError("encoder subprocess returned an unknown schema")
    try:
        shape = tuple(int(value) for value in payload["shape"])
        raw = base64.b64decode(payload["float32_base64"], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise ToolEmbeddingError("encoder subprocess returned malformed data") from error
    if len(shape) != 2 or shape[1] != ENCODER_DIMENSION:
        raise ToolEmbeddingError(f"encoder subprocess returned shape {shape}")
    expected = shape[0] * shape[1] * 4
    if len(raw) != expected:
        raise ToolEmbeddingError(
            f"encoder byte count {len(raw)} does not match shape {shape}")
    matrix = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    return [_normalize_vector(row) for row in matrix]


def encode_texts_subprocess(
        texts: list[str], config: EmbeddingConfig,
        *, timeout_seconds: float | None = None):
    """Encode in a disposable CPU child so PyTorch memory cannot linger."""
    import psutil

    verify_encoder(config.model_dir, full_hash=False)
    available = int(psutil.virtual_memory().available)
    minimum = config.min_available_mb * 1_000_000
    if available < minimum:
        raise ToolEmbeddingError(
            "query_encoder_memory_guard: available "
            f"{available // 1_000_000} MB < {config.min_available_mb} MB")
    request = {
        "schema": "vmodel-tool-encoder-request-v1",
        "texts": texts,
        "model_dir": str(config.model_dir),
    }
    env = os.environ.copy()
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONHASHSEED": "0",
    })
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "runtime.tool_embeddings", "encode-json"],
            input=_canonical_json(request), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=_REPO_ROOT, env=env,
            timeout=(timeout_seconds or config.timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ToolEmbeddingError("encoder subprocess timed out") from error
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:].strip()
        raise ToolEmbeddingError(
            f"encoder subprocess failed ({completed.returncode}): {detail}")
    try:
        response = json.loads(completed.stdout)
    except ValueError as error:
        raise ToolEmbeddingError("encoder subprocess returned non-JSON output") from error
    return _decode_vectors(response)


def _encode_in_process(texts: list[str], model_dir: Path):
    """Pinned BGE CLS pooling. Imported only inside the disposable process."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    if not texts:
        return np.empty((0, ENCODER_DIMENSION), dtype=np.float32)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32).to("cpu")
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start:start + 16], padding=True, truncation=True,
                max_length=ENCODER_MAX_TOKENS, return_tensors="pt")
            output = model(**batch).last_hidden_state[:, 0]
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            rows.append(output.cpu().numpy().astype("<f4", copy=False))
    matrix = np.concatenate(rows, axis=0)
    if matrix.shape != (len(texts), ENCODER_DIMENSION):
        raise ToolEmbeddingError(f"unexpected BGE output shape {matrix.shape}")
    return matrix


def _encode_json_main() -> int:
    try:
        request = json.load(sys.stdin.buffer)
        if request.get("schema") != "vmodel-tool-encoder-request-v1":
            raise ToolEmbeddingError("unknown encoder request schema")
        texts = request.get("texts")
        if (not isinstance(texts, list)
                or not all(isinstance(text, str) for text in texts)
                or len(texts) > 512
                or any(len(text.encode("utf-8")) > 32_000 for text in texts)):
            raise ToolEmbeddingError("invalid encoder text batch")
        model_dir = Path(request.get("model_dir", ""))
        verify_encoder(model_dir, full_hash=False)
        matrix = _encode_in_process(texts, model_dir)
        response = {
            "schema": "vmodel-tool-encoder-result-v1",
            "shape": list(matrix.shape),
            "float32_base64": base64.b64encode(matrix.tobytes()).decode("ascii"),
        }
        sys.stdout.buffer.write(_canonical_json(response))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolEmbeddingError("non-finite hybrid ranking score")
    if high - low <= 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def hybrid_scores(
        capsules: list[str], query: str, lexical_scores: list[float], *,
        config: EmbeddingConfig | None = None,
        encoder: Callable[[list[str], EmbeddingConfig], list] | None = None,
        allow_tool_build: bool = False):
    """Return combined scores and cache telemetry, or lexical fallback.

    ``allow_tool_build`` is reserved for the offline builder/tests. The serving
    path requires 100% tool-vector coverage and never starts a large catalog
    build while Qwen is resident.
    """
    config = config or EmbeddingConfig.from_env()
    encoder = encoder or encode_texts_subprocess
    started = time.perf_counter()
    meta = {
        "tool_embedding_profile": PROFILE,
        "tool_embedding_status": "fallback",
        "tool_embedding_tool_cache_hits": 0,
        "tool_embedding_tool_cache_misses": len(capsules),
        "tool_embedding_query_cache_hit": 0,
        "tool_embedding_semantic_weight": config.semantic_weight,
    }
    if len(capsules) != len(lexical_scores) or not query.strip():
        meta["tool_embedding_fallback"] = "invalid_input"
        return list(lexical_scores), meta

    cache = ToolEmbeddingCache(config.cache_dir)
    try:
        tool_vectors, hits = cache.get_many("tool", capsules)
        meta["tool_embedding_tool_cache_hits"] = hits
        meta["tool_embedding_tool_cache_misses"] = len(capsules) - hits
        if hits != len(capsules):
            if not allow_tool_build:
                raise ToolEmbeddingError("offline_tool_cache_incomplete")
            missing_indices = [
                index for index, vector in enumerate(tool_vectors)
                if vector is None
            ]
            missing_texts = [capsules[index] for index in missing_indices]
            missing_vectors = encoder(missing_texts, config)
            cache.store_many("tool", missing_texts, missing_vectors)
            for index, vector in zip(missing_indices, missing_vectors):
                tool_vectors[index] = _normalize_vector(vector)

        query_text = QUERY_PREFIX + query.strip()
        query_vectors, query_hits = cache.get_many("query", [query_text])
        meta["tool_embedding_query_cache_hit"] = query_hits
        query_vector = query_vectors[0]
        if query_vector is None:
            query_vector = encoder([query_text], config)[0]
            cache.store_many(
                "query", [query_text], [query_vector],
                max_objects=config.query_cache_max)
        query_vector = _normalize_vector(query_vector)

        semantic_scores = [
            float(query_vector @ _normalize_vector(vector))
            for vector in tool_vectors
        ]
        lexical_normalized = _minmax(list(lexical_scores))
        semantic_normalized = _minmax(semantic_scores)
        weight = config.semantic_weight
        combined = [
            (1.0 - weight) * lexical + weight * semantic
            for lexical, semantic in zip(lexical_normalized, semantic_normalized)
        ]
        catalog_digest = hashlib.sha256()
        catalog_digest.update(PROFILE.encode("ascii"))
        for object_id in sorted(_object_id("tool", capsule) for capsule in capsules):
            catalog_digest.update(bytes.fromhex(object_id))
        meta.update({
            "tool_embedding_status": "hybrid",
            "tool_embedding_catalog_id": catalog_digest.hexdigest()[:16],
            "tool_embedding_score_min": round(min(semantic_scores), 6),
            "tool_embedding_score_max": round(max(semantic_scores), 6),
        })
        return combined, meta
    except (OSError, ValueError, ToolEmbeddingError, subprocess.SubprocessError) as error:
        meta["tool_embedding_fallback"] = str(error)[:160]
        if os.environ.get("VMODEL_TOOL_EMBEDDINGS_REQUIRED", "0") == "1":
            raise ToolEmbeddingError(str(error)) from error
        return list(lexical_scores), meta
    finally:
        meta["tool_embedding_seconds"] = round(time.perf_counter() - started, 4)


def build_tool_cache(
        capsules: list[str], *, config: EmbeddingConfig | None = None,
        encoder: Callable[[list[str], EmbeddingConfig], list] | None = None,
        verify_model: bool = True) -> dict:
    """Populate missing tool objects. Safe to run only without serving weights."""
    config = config or EmbeddingConfig.from_env()
    if verify_model:
        verify_encoder(config.model_dir, full_hash=True)
    encoder = encoder or encode_texts_subprocess
    cache = ToolEmbeddingCache(config.cache_dir)
    vectors, hits = cache.get_many("tool", capsules)
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        texts = [capsules[index] for index in missing]
        encoded = encoder(texts, config)
        cache.store_many("tool", texts, encoded)
    catalog_digest = hashlib.sha256()
    catalog_digest.update(PROFILE.encode("ascii"))
    for object_id in sorted(_object_id("tool", capsule) for capsule in capsules):
        catalog_digest.update(bytes.fromhex(object_id))
    return {
        "schema": "vmodel-tool-embedding-build-v1",
        "profile": PROFILE,
        "catalog_id": catalog_digest.hexdigest()[:16],
        "tools": len(capsules),
        "cache_hits": hits,
        "encoded": len(missing),
    }


def _build_main(args) -> int:
    capture = json.loads(Path(args.capture).read_text())
    tools = capture.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ToolEmbeddingError("capture has no tool catalog")
    from .toolcalls import tool_search_capsule

    capsules = [tool_search_capsule(tool) for tool in tools]
    config = EmbeddingConfig(
        model_dir=Path(args.model_dir), cache_dir=Path(args.cache_dir),
        semantic_weight=args.weight, timeout_seconds=args.timeout,
        min_available_mb=args.min_available_mb)
    result = build_tool_cache(capsules, config=config)
    result["capture_sha256"] = _sha256_file(Path(args.capture))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline cache builder / private BGE encoder helper")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("encode-json", help=argparse.SUPPRESS)
    build = sub.add_parser("build", help="verify encoder and cache a captured catalog")
    build.add_argument("--capture", required=True)
    build.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    build.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    build.add_argument("--weight", type=float, default=0.60)
    build.add_argument("--timeout", type=float, default=30.0)
    build.add_argument("--min-available-mb", type=int, default=4800)
    args = parser.parse_args(argv)
    if args.command == "encode-json":
        return _encode_json_main()
    try:
        return _build_main(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
