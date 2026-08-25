"""Durable, fail-closed prompt state for mixed-depth Qwen prefill.

The ordinary hot-KV journal stores token-aligned per-layer deltas.  A lossy
mixed-depth Qwen schedule intentionally violates that layout: lower attention
layers retain the complete prompt while upper layers retain only a packed
suffix (and, optionally, a prefix anchor).  This module therefore persists a
complete *stable-boundary* snapshot with the actual local length of every
attention layer instead of pretending those arrays are token-aligned deltas.

Stable-boundary snapshots are eligible only for strict extensions and carry no
endpoint logits.  A separately typed prompt-endpoint snapshot may serve an
identical prompt (with its exact stored logits) or a strict extension, but
never a rewind or branch.  Model, runtime, RoPE, quantization, and mixed-depth
schedule identity are inherited from StreamingEngine's KV fingerprint;
manifests and safetensors payloads are content-addressed and SHA-256 verified
before any tensor is trusted.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading

import mlx.core as mx

from .hot_kv_persist import (
    _canonical_json,
    _content_id,
    _fsync_dir,
    _normalize_cache_namespace,
    _publish_json_immutable,
    _publish_temp_immutable,
    _sha256_file,
    _write_safetensors_temp,
)
from .kda_state import KDAStateCache
from .kv_cache import KVCache


_FORMAT = "qwen-mixed-depth-stable-prefix-v1"
_ENDPOINT_STATE_FORMAT = "logits-hidden-v1"


class QwenMixedDepthPromptPersistence:
    """Checksummed full snapshots for Qwen mixed-depth stable prefixes."""

    # Short prompts can stay on the ordinary exact path when the configured
    # suffix window already covers the whole request.  Their state is not a
    # mixed-depth snapshot and must be retained only by the in-memory hot slot;
    # the engine consults this capability before calling ``save``.
    requires_approximate_stable_prefix = True
    supports_prompt_endpoint_snapshot = True

    def __init__(
        self,
        directory: str | Path,
        fingerprint: str,
        chunk_size: int,
        *,
        config,
        max_checkpoints: int = 8,
        max_bytes: int = 0,
        allow_prompt_endpoint: bool = False,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp = fingerprint + "|qwen-mixed-depth-stable-prefix-v1"
        self.chunk_size = int(chunk_size)
        self.config = config
        self.max_checkpoints = max(0, int(max_checkpoints))
        self.max_bytes = max(0, int(max_bytes))
        self.allow_prompt_endpoint = bool(allow_prompt_endpoint)
        self._thread_lock = threading.RLock()
        self._lock_path = self.dir / ".mixed-depth.lock"
        self._lock_path.touch(exist_ok=True)
        if getattr(config, "model_type", "") not in ("qwen3_5", "qwen3_5_moe"):
            raise ValueError("mixed-depth prompt persistence requires Qwen3.5-family state")
        if self.chunk_size <= 0:
            raise ValueError("mixed-depth prompt persistence requires a fixed chunk size")

    @contextmanager
    def _locked(self, *, exclusive: bool):
        with self._thread_lock:
            descriptor = os.open(self._lock_path, os.O_RDWR)
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
                )
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _meta_path(self, snapshot_id: str) -> Path:
        return self.dir / f"{snapshot_id}.mixed.json"

    def _payload_path(self, snapshot_id: str) -> Path:
        return self.dir / f"{snapshot_id}.mixed.safetensors"

    def _full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(tuple(self.config.layer_types))
            if kind == "full_attention"
        )

    def _recurrent_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(tuple(self.config.layer_types))
            if kind == "linear_attention"
        )

    def _read_meta(self, snapshot_id: str, *, verify_payload: bool = False):
        path = self._meta_path(snapshot_id)
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(meta, dict)
            or meta.get("format") != _FORMAT
            or meta.get("id") != snapshot_id
            or meta.get("fp") != self.fp
            or meta.get("checkpoint_kind") not in (
                "stable_prefix", "prompt_endpoint")
        ):
            return None
        if (meta.get("checkpoint_kind") == "prompt_endpoint"
                and not self.allow_prompt_endpoint):
            return None
        if (meta.get("checkpoint_kind") == "prompt_endpoint"
                and meta.get("endpoint_state_format")
                != _ENDPOINT_STATE_FORMAT):
            return None
        core = {key: value for key, value in meta.items() if key not in ("format", "id")}
        if _content_id(_FORMAT, core) != snapshot_id:
            return None
        try:
            tokens = meta["tokens"]
            lengths = meta["layer_lengths"]
            starts = meta["layer_starts"]
            namespace = _normalize_cache_namespace(
                meta.get("cache_namespace", "default"), strict=False)
            if (
                namespace is None
                or not isinstance(tokens, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in tokens)
                or len(lengths) != int(self.config.num_hidden_layers)
                or len(starts) != int(self.config.num_hidden_layers)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in (*lengths, *starts))
                or int(meta["prompt_length"]) != len(tokens)
                or not bool(meta.get("approximate"))
            ):
                return None
            payload = self._payload_path(snapshot_id)
            if payload.stat().st_size != int(meta["payload_bytes"]):
                return None
            if verify_payload and _sha256_file(payload) != meta.get("payload_sha256"):
                return None
        except (KeyError, OSError, TypeError, ValueError):
            return None
        return meta

    def _existing_snapshot(
        self, tokens, cache_namespace: str, checkpoint_kind: str,
    ) -> str | None:
        wanted = list(map(int, tokens))
        for path in self.dir.glob("*.mixed.json"):
            snapshot_id = path.name[:-len(".mixed.json")]
            meta = self._read_meta(snapshot_id, verify_payload=True)
            if (
                meta is not None
                and meta["tokens"] == wanted
                and meta.get("cache_namespace", "default") == cache_namespace
                and meta.get("checkpoint_kind") == checkpoint_kind
            ):
                os.utime(path, None)
                os.utime(self._payload_path(snapshot_id), None)
                return snapshot_id
        return None

    def _snapshot_arrays(
        self, kv: KVCache, logits: mx.array | None = None,
        hidden: mx.array | None = None,
    ) -> dict[str, mx.array]:
        if type(kv) is not KVCache or kv.compressed_mla:
            raise ValueError("mixed-depth persistence requires a plain KVCache")
        arrays: dict[str, mx.array] = {}
        expected_attention = set(self._full_attention_layers())
        for layer in range(len(kv.keys)):
            key, value = kv.keys[layer], kv.values[layer]
            if layer in expected_attention:
                if key is None or value is None:
                    raise ValueError(f"mixed-depth snapshot is missing attention layer {layer}")
                arrays[f"k{layer}"] = key
                arrays[f"v{layer}"] = value
            elif key is not None or value is not None:
                raise ValueError(f"unexpected KV tensors for recurrent layer {layer}")
        recurrent = getattr(kv, "kda_cache", None)
        if recurrent is None:
            raise ValueError("mixed-depth snapshot is missing recurrent state")
        arrays.update(recurrent.export_arrays())
        if logits is not None:
            if (not isinstance(logits, mx.array)
                    or not logits.shape
                    or int(logits.shape[-1]) != int(self.config.vocab_size)):
                raise ValueError(
                    "mixed-depth endpoint logits do not match the vocabulary")
            arrays["logits"] = logits
        if hidden is not None:
            if (not isinstance(hidden, mx.array)
                    or len(hidden.shape) != 3
                    or tuple(map(int, hidden.shape[:2])) != (1, 1)
                    or int(hidden.shape[-1])
                    != int(self.config.hidden_size)):
                raise ValueError(
                    "mixed-depth endpoint hidden state does not match "
                    "the model")
            arrays["hidden"] = hidden
        return arrays

    def save(
        self,
        parent_chain,
        parent_covered: int,
        tokens,
        kv: KVCache,
        logits,
        prompt_logits,
        prompt_length: int,
        reusable_prefix: int,
        approximate: bool = False,
        tool_capsules=(),
        cache_namespace: str = "default",
        checkpoint_kind: str = "endpoint",
    ) -> tuple[str, ...]:
        """Persist stable state only; later generated endpoints are irrelevant."""
        del parent_covered, logits, prompt_logits, reusable_prefix, tool_capsules
        if checkpoint_kind != "stable_prefix":
            return tuple(parent_chain)
        tokens = tuple(map(int, tokens))
        cache_namespace = _normalize_cache_namespace(cache_namespace, strict=True)
        if not approximate:
            raise ValueError("mixed-depth persistence requires approximate prompt state")
        if int(prompt_length) != len(tokens) or kv.offset != len(tokens):
            raise ValueError("mixed-depth stable prefix length mismatch")

        with self._locked(exclusive=True):
            existing = self._existing_snapshot(
                tokens, cache_namespace, "stable_prefix")
            if existing is not None:
                return (existing,)
            arrays = self._snapshot_arrays(kv)
            mx.eval(list(arrays.values()))
            tmp, payload_sha256, payload_bytes = _write_safetensors_temp(self.dir, arrays)
            layer_lengths = list(kv.layer_lengths())
            layer_starts = list(map(int, kv._starts))
            core = {
                "fp": self.fp,
                "checkpoint_kind": "stable_prefix",
                "tokens": list(tokens),
                "prompt_length": len(tokens),
                "approximate": True,
                "cache_namespace": cache_namespace,
                "layer_lengths": layer_lengths,
                "layer_starts": layer_starts,
                "payload_sha256": payload_sha256,
                "payload_bytes": payload_bytes,
            }
            snapshot_id = _content_id(_FORMAT, core)
            try:
                _publish_temp_immutable(tmp, self._payload_path(snapshot_id), payload_sha256)
                _publish_json_immutable(
                    self._meta_path(snapshot_id),
                    {"format": _FORMAT, "id": snapshot_id, **core},
                )
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
            _fsync_dir(self.dir)
            return (snapshot_id,)

    def save_prompt_endpoint(
        self,
        tokens,
        kv: KVCache,
        logits: mx.array,
        hidden: mx.array,
        *,
        approximate: bool,
        cache_namespace: str = "default",
    ) -> tuple[str, ...]:
        """Persist prompt state before decode mutates it.

        The synchronous call site owns the exact prompt endpoint at this
        instant, so no recurrent rewind or KV trimming is involved.  Restores
        remain limited to an identical prompt or a strict extension.
        """
        if not self.allow_prompt_endpoint:
            raise ValueError(
                "mixed-depth prompt endpoint persistence is disabled")
        tokens = tuple(map(int, tokens))
        cache_namespace = _normalize_cache_namespace(
            cache_namespace, strict=True)
        if not approximate:
            raise ValueError(
                "mixed-depth endpoint persistence requires approximate state")
        if kv.offset != len(tokens):
            raise ValueError("mixed-depth prompt endpoint length mismatch")

        with self._locked(exclusive=True):
            existing = self._existing_snapshot(
                tokens, cache_namespace, "prompt_endpoint")
            if existing is not None:
                return (existing,)
            arrays = self._snapshot_arrays(kv, logits, hidden)
            mx.eval(list(arrays.values()))
            tmp, payload_sha256, payload_bytes = _write_safetensors_temp(
                self.dir, arrays)
            core = {
                "fp": self.fp,
                "checkpoint_kind": "prompt_endpoint",
                "endpoint_state_format": _ENDPOINT_STATE_FORMAT,
                "tokens": list(tokens),
                "prompt_length": len(tokens),
                "approximate": True,
                "cache_namespace": cache_namespace,
                "layer_lengths": list(kv.layer_lengths()),
                "layer_starts": list(map(int, kv._starts)),
                "payload_sha256": payload_sha256,
                "payload_bytes": payload_bytes,
            }
            snapshot_id = _content_id(_FORMAT, core)
            try:
                _publish_temp_immutable(
                    tmp, self._payload_path(snapshot_id), payload_sha256)
                _publish_json_immutable(
                    self._meta_path(snapshot_id),
                    {"format": _FORMAT, "id": snapshot_id, **core},
                )
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
            _fsync_dir(self.dir)
            return (snapshot_id,)

    def find_best_match(
        self,
        tokens,
        chunk_size: int,
        *,
        cache_namespace: str = "default",
        min_matched_exclusive: int = -1,
    ):
        del chunk_size
        requested = tuple(map(int, tokens))
        namespace = _normalize_cache_namespace(cache_namespace, strict=True)
        candidates = []
        with self._locked(exclusive=False):
            paths = list(self.dir.glob("*.mixed.json"))
        for path in paths:
            snapshot_id = path.name[:-len(".mixed.json")]
            meta = self._read_meta(snapshot_id)
            if meta is None or meta.get("cache_namespace", "default") != namespace:
                continue
            candidate = tuple(meta["tokens"])
            checkpoint_kind = meta.get("checkpoint_kind")
            exact_endpoint = bool(
                checkpoint_kind == "prompt_endpoint"
                and len(requested) == len(candidate)
                and requested == candidate)
            extension = bool(
                len(requested) > len(candidate)
                and requested[:len(candidate)] == candidate)
            if (len(candidate) <= int(min_matched_exclusive)
                    or not (exact_endpoint or extension)):
                continue
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                mtime = 0
            candidates.append((
                len(candidate), int(exact_endpoint), mtime, snapshot_id, meta))
        candidates.sort(reverse=True)
        for matched, exact_endpoint, _mtime, snapshot_id, meta in candidates:
            if self._read_meta(snapshot_id, verify_payload=True) is None:
                continue
            return {
                "case": "endpoint" if exact_endpoint else "extension",
                "matched": matched,
                "watermark": 0,
                "n_segments": 1,
                "lcp": matched,
                "leaf": snapshot_id,
                "chain": (snapshot_id,),
                "checkpoint_id": snapshot_id,
                "approximate": True,
                "checkpoint_kind": meta.get("checkpoint_kind"),
                "cache_namespace": namespace,
            }
        return None

    def load_matched_chain(self, match: dict, num_layers: int):
        snapshot_id = str(match.get("checkpoint_id", ""))
        with self._locked(exclusive=False):
            meta = self._read_meta(snapshot_id, verify_payload=True)
            if meta is None:
                return None
            try:
                arrays = mx.load(str(self._payload_path(snapshot_id)))
                mx.eval(list(arrays.values()))
            except Exception as error:
                print(
                    "[qwen-mixed-kv] snapshot load failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                return None

        if int(num_layers) != int(self.config.num_hidden_layers):
            return None
        attention_layers = set(self._full_attention_layers())
        recurrent_names = {
            name: value
            for name, value in arrays.items()
            if name.startswith("kda_state_") or name.startswith("kda_conv_")
        }
        expected_names = set(recurrent_names)
        for layer in attention_layers:
            expected_names.update((f"k{layer}", f"v{layer}"))
        checkpoint_kind = meta.get("checkpoint_kind")
        if checkpoint_kind == "prompt_endpoint":
            expected_names.update(("logits", "hidden"))
        if set(arrays) != expected_names:
            return None

        kv = KVCache(num_layers)
        lengths = tuple(map(int, meta["layer_lengths"]))
        starts = tuple(map(int, meta["layer_starts"]))
        try:
            for layer in attention_layers:
                key, value = arrays[f"k{layer}"], arrays[f"v{layer}"]
                if int(key.shape[2]) != lengths[layer] or int(value.shape[2]) != lengths[layer]:
                    raise ValueError("attention layer length mismatch")
                kv.keys[layer], kv.values[layer] = key, value
                kv._starts[layer] = starts[layer]
            kv.kda_cache = KDAStateCache.from_arrays(
                num_layers,
                recurrent_names,
                expected_layers=self._recurrent_layers(),
            )
            if kv.offset != len(meta["tokens"]):
                raise ValueError("restored mixed-depth global offset mismatch")
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as error:
            print(
                "[qwen-mixed-kv] snapshot rejected: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return None
        exact_logits = None
        exact_hidden = None
        if match.get("case") == "endpoint":
            if checkpoint_kind != "prompt_endpoint":
                return None
            exact_logits = arrays["logits"]
            if (not exact_logits.shape
                    or int(exact_logits.shape[-1])
                    != int(self.config.vocab_size)):
                return None
            exact_hidden = arrays["hidden"]
            if (len(exact_hidden.shape) != 3
                    or tuple(map(int, exact_hidden.shape[:2])) != (1, 1)
                    or int(exact_hidden.shape[-1])
                    != int(self.config.hidden_size)):
                return None
        return tuple(meta["tokens"]), kv, exact_logits, exact_hidden

    def load_all(self, num_layers: int, limit: int) -> list[tuple]:
        del num_layers, limit
        # Disk-lazy by design: loading a large branch before request admission
        # would retain the wrong conversation and compete with weight bootstrap.
        return []

    def gc(self) -> int:
        with self._locked(exclusive=True):
            entries = []
            for path in self.dir.glob("*.mixed.json"):
                snapshot_id = path.name[:-len(".mixed.json")]
                meta = self._read_meta(snapshot_id)
                if meta is None:
                    continue
                payload = self._payload_path(snapshot_id)
                try:
                    entries.append((path.stat().st_mtime_ns, snapshot_id,
                                    path.stat().st_size + payload.stat().st_size))
                except OSError:
                    continue
            entries.sort(reverse=True)
            keep: set[str] = set()
            total = 0
            for _mtime, snapshot_id, size in entries:
                if self.max_checkpoints and len(keep) >= self.max_checkpoints:
                    continue
                if self.max_bytes and keep and total + size > self.max_bytes:
                    continue
                keep.add(snapshot_id)
                total += size
            removed = 0
            for _mtime, snapshot_id, _size in entries:
                if snapshot_id in keep:
                    continue
                self._meta_path(snapshot_id).unlink(missing_ok=True)
                self._payload_path(snapshot_id).unlink(missing_ok=True)
                removed += 1
            if removed:
                _fsync_dir(self.dir)
            return removed
