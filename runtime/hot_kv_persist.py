"""Disk backing for the in-memory hot-prompt-KV LRU (runtime/engine.py's
`_hot_prompt_slots`), so a conversation's retained KV can survive a server
restart instead of paying a full cold prefill again.

v3 is the shared durable substrate for both the hot LRU and F37's
longest-prefix store: a parent-hashed segment DAG with immutable checksummed
generations, DSA deltas, reader leases, and count/byte-bounded GC. Extending a
known endpoint writes only the new tensor positions plus one small checkpoint
payload, avoiding the former full-snapshot O(n^2) cumulative write pattern.

- A **segment** is an immutable, content-addressed KV delta: the newly
  computed keys/values for one span of NEW tokens on top of a specific
  parent segment (or no parent, for a conversation's first chunk).
  The canonical segment identity covers the fingerprint, parent, delta token
  IDs, KV layout, payload byte count, and payload SHA-256.
  Segments are chunk-aligned (`hot_prompt_kv_chunk_size`) up to
  `reusable_prefix`, then (2026-07-15, later revision) split into up to TWO
  more nodes rather than one merged tail: a **prompt-tail** segment for
  whatever non-chunk-aligned prompt remainder is left, and a separate
  **generation** segment for this turn's own generated continuation. This
  split exists specifically so a LATER request that repeats this exact
  prompt (before any of ITS OWN generation) has a real, addressable parent
  to fork from -- the prompt-tail node -- instead of the prompt/generation
  boundary being buried in the middle of one merged segment.
- A **checkpoint** is a small pointer record -- the extra per-turn facts
  (logits, prompt_logits, prompt_length, reusable_prefix) needed to treat a
  specific leaf segment as a directly loadable LRU slot. It is its own
  immutable checksummed generation, so multiple endpoint generations may
  safely reference the same leaf.
- Because segments are content-addressed and immutable, "append" is simply
  "write a new segment whose parent is the old leaf" -- no existing file is
  ever rewritten or truncated. Two divergent continuations from the same
  parent naturally produce two sibling child segments, both valid and
  loadable: this is exact-DAG conversation forking for free, and it is also
  why "rollback" needs no special operation -- a stale branch is simply a
  segment chain nothing chooses to walk into again, not something to delete
  for correctness. Content-addressing also gives free byte sharing: if two
  DIFFERENT conversations (or a restarted server with no in-memory history)
  happen to produce the same delta tokens on top of the same parent (e.g. a
  shared system prompt + tool schema prefix), `_write_segment_if_missing`
  finds the existing file and never rewrites it. This is what lets N
  independent agentic/cron tasks sharing one identical preamble each fork
  their own generation segment off the SAME prompt-tail parent, at the
  "repeat" case, without any of them rewriting the shared preamble's bytes.
- A total in-memory miss scans checkpoint/segment metadata for the best exact
  prefix before recomputing cold. Only the winning chain's tensor payloads are
  loaded. This lets more concurrent tasks share a persisted preamble than fit
  in `hot_prompt_kv_slots`; `find_best_match()` and `load_matched_chain()` are
  regression-covered and the real-Qwen gate exercises the same path.
- Checkpoint lifetime on disk is DECOUPLED from the in-memory LRU
  (`hot_prompt_kv_slots`): evicting a slot from memory frees RAM only, it
  does not delete that slot's disk checkpoint, and consuming a slot via a
  match does not delete the CONSUMED slot's checkpoint either. Both would
  defeat forking -- a later, different continuation from an earlier point
  (a "regenerate," an edited earlier message) needs that earlier point's
  checkpoint to still be directly loadable, not just reachable as someone
  else's ancestor segments. Instead `gc()` owns disk retention with its own,
  larger, recency-based budget (`max_checkpoints`): past that many
  checkpoints, the OLDEST-by-mtime are dropped, then any segment no longer
  reachable from a surviving checkpoint's chain is swept. Saving a
  checkpoint always refreshes its mtime, so an actively-continued
  conversation naturally stays young regardless of how many total
  checkpoints exist.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

import mlx.core as mx

from .kv_cache import KVCache


_SEGMENT_FORMAT = "hot-kv-segment-v3"
_CHECKPOINT_FORMAT = "hot-kv-checkpoint-v3"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()


def _content_id(kind: str, core: dict) -> str:
    return hashlib.sha256(kind.encode() + b"\0" + _canonical_json(core)).hexdigest()


def _normalize_tool_capsules(value, prompt_length: int, *, strict: bool):
    """Return durable ``(identity, start, end)`` prompt-span metadata.

    Capsule data is small enough to live in the checksummed checkpoint
    manifest.  Treat malformed on-disk metadata as an unusable checkpoint;
    malformed caller data is a programming error.  Missing metadata remains
    valid for checkpoint generations written before capsule persistence was
    added.
    """
    try:
        spans = []
        for identity, start, end in (value or ()):
            if (not isinstance(identity, str)
                    or isinstance(start, bool) or not isinstance(start, int)
                    or isinstance(end, bool) or not isinstance(end, int)):
                raise ValueError("invalid tool capsule span metadata")
            spans.append((identity, start, end))
        spans = tuple(spans)
        previous = 0
        identities = set()
        for identity, start, end in spans:
            if (not identity or identity in identities
                    or not 0 <= start < end <= prompt_length
                    or start < previous):
                raise ValueError("invalid tool capsule span metadata")
            identities.add(identity)
            previous = end
        return spans
    except (TypeError, ValueError):
        if strict:
            raise ValueError("invalid tool capsule span metadata") from None
        return None


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Identity used to carry a completed hash proof between leased reads."""
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _file_signatures(paths) -> dict[str, tuple[int, int, int, int, int]]:
    return {Path(path).name: _file_signature(Path(path)) for path in paths}


def _signatures_match(paths, verified) -> bool:
    """True only when every immutable file is unchanged since SHA validation."""
    if not verified:
        return False
    try:
        return all(
            verified.get(Path(path).name) == _file_signature(Path(path))
            for path in paths
        )
    except OSError:
        return False


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unique_tmp(path: Path, suffix: str) -> Path:
    return path / f".{os.getpid()}.{uuid.uuid4().hex}.tmp.{suffix}"


def _write_safetensors_temp(directory: Path, arrays: dict) -> tuple[Path, str, int]:
    tmp = _unique_tmp(directory, "safetensors")
    mx.save_safetensors(str(tmp), arrays)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return tmp, _sha256_file(tmp), tmp.stat().st_size


def _publish_temp_immutable(tmp: Path, final: Path, expected_sha256: str) -> None:
    """Publish without replacement; concurrent identical writers converge."""
    try:
        os.link(tmp, final)
    except FileExistsError:
        if _sha256_file(final) != expected_sha256:
            raise RuntimeError(
                f"immutable object collision at {final.name}")
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(final.parent)


def _publish_json_immutable(path: Path, value: dict) -> None:
    payload = _canonical_json(value)
    tmp = _unique_tmp(path.parent, "json")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    expected = hashlib.sha256(payload).hexdigest()
    _publish_temp_immutable(tmp, path, expected)


def _atomic_json(path: Path, value: dict) -> None:
    payload = _canonical_json(value)
    tmp = _unique_tmp(path.parent, "json")
    with tmp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _score_match(new_tokens, cand_tokens, cand_prompt_length: int,
                  cand_reusable_prefix: int, cand_chain_len: int, chunk_size: int):
    """The SAME scoring as engine.py's in-memory lookup (repeat/endpoint/
    strict-extension/branch), against a reconstructed candidate token list
    instead of a resident _HotPromptSlot -- used only for the disk-side
    search on a total in-memory miss. Returns (case, matched, watermark,
    n_segments, lcp), or None if nothing is usable (no overlap, or the
    floored reusable length is 0)."""
    lcp = 0
    for old, new in zip(cand_tokens, new_tokens):
        if old != new:
            break
        lcp += 1
    if len(new_tokens) == cand_prompt_length and lcp >= len(new_tokens):
        n_full = cand_reusable_prefix // chunk_size
        n_segments = n_full + (1 if cand_prompt_length > cand_reusable_prefix else 0)
        watermark = min(cand_reusable_prefix, len(new_tokens))
        return ("repeat", len(new_tokens), watermark, n_segments, lcp)
    if len(new_tokens) == len(cand_tokens) and lcp == len(new_tokens):
        watermark = min(cand_reusable_prefix, len(new_tokens))
        return ("endpoint", len(new_tokens), watermark, cand_chain_len, lcp)
    if len(new_tokens) > len(cand_tokens) and lcp == len(cand_tokens):
        watermark = min(cand_reusable_prefix, len(cand_tokens))
        return ("extension", len(cand_tokens), watermark, cand_chain_len, lcp)
    reusable = min(lcp, max(0, len(new_tokens) - 1))
    reusable = (reusable // chunk_size) * chunk_size
    reusable = min(reusable, cand_reusable_prefix)
    if reusable <= 0:
        return None
    return ("branch", reusable, reusable, reusable // chunk_size, lcp)


def _slice_kv(kv: KVCache, start: int, end: int) -> dict[str, mx.array]:
    arrays = {}
    for i in range(len(kv.keys)):
        if kv.keys[i] is None:
            continue
        if kv.compressed_mla:
            arrays[f"k{i}"] = kv.keys[i][:, start:end, :]
        else:
            arrays[f"k{i}"] = kv.keys[i][:, :, start:end, :]
            if kv.values[i] is not None:
                arrays[f"v{i}"] = kv.values[i][:, :, start:end, :]
    dsa = getattr(kv, "dsa", None)
    if dsa is not None:
        for layer, keys in dsa.k_idx.items():
            arrays[f"dsa_k{layer}"] = keys[:, start:end, :]
    return arrays


class HotPromptKVPersistence:
    def __init__(self, dir: str | Path, fingerprint: str, chunk_size: int,
                 max_checkpoints: int = 64, *, config=None,
                 require_dsa: bool = False, max_bytes: int = 0):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # v3 makes every tensor payload and manifest an immutable, checksummed
        # generation. The identity bump prevents a v2 entry (whose semantic id
        # did not cover tensor bytes) from entering the verified journal.
        self.fp = fingerprint + "|hot-kv-v3-durable-dsa"
        self.chunk_size = chunk_size
        self.max_checkpoints = max_checkpoints
        self.max_bytes = max(0, int(max_bytes))
        self.config = config
        self.require_dsa = require_dsa
        self._thread_lock = threading.RLock()
        self._lock_path = self.dir / ".journal.lock"
        self._lock_path.touch(exist_ok=True)
        self._leases_dir = self.dir / ".leases"
        self._leases_dir.mkdir(exist_ok=True)
        self._segment_index: dict[tuple, list[str]] = {}
        self._rebuild_segment_index()

    @contextmanager
    def _locked(self, *, exclusive: bool):
        """Cross-process journal lock; GC is exclusive, readers/writers shared."""
        with self._thread_lock:
            fd = os.open(self._lock_path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @contextmanager
    def _lease(self, paths):
        """Pin immutable files against GC while a reader validates/evaluates."""
        names = sorted({Path(path).name for path in paths})
        lease = self._leases_dir / f"{os.getpid()}.{uuid.uuid4().hex}.lease.json"
        with self._locked(exclusive=False):
            _atomic_json(lease, {
                "format": "hot-kv-reader-lease-v1",
                "pid": os.getpid(),
                "files": names,
            })
        try:
            yield
        finally:
            with self._locked(exclusive=False):
                lease.unlink(missing_ok=True)
                _fsync_dir(self._leases_dir)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _leased_names_locked(self) -> set[str]:
        """Return live leased basenames and clean leases left by dead readers."""
        names: set[str] = set()
        changed = False
        for path in self._leases_dir.glob("*.lease.json"):
            try:
                value = json.loads(path.read_text())
                pid = int(value["pid"])
                files = value["files"]
                if (value.get("format") != "hot-kv-reader-lease-v1"
                        or not isinstance(files, list)):
                    raise ValueError("invalid lease")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                changed = True
                continue
            if not self._pid_alive(pid):
                path.unlink(missing_ok=True)
                changed = True
                continue
            names.update(str(name) for name in files)
        if changed:
            _fsync_dir(self._leases_dir)
        return names

    # ---- segments ----

    def _segment_meta_path(self, seg_id: str) -> Path:
        return self.dir / f"{seg_id}.seg.json"

    def _segment_payload_path(self, seg_id: str) -> Path:
        return self.dir / f"{seg_id}.seg.safetensors"

    @staticmethod
    def _segment_index_key(parent, tokens, compressed_mla: bool) -> tuple:
        return parent, tuple(tokens), bool(compressed_mla)

    def _read_segment_meta(self, seg_id: str, *, verify_payload: bool = False
                           ) -> dict | None:
        try:
            meta = json.loads(self._segment_meta_path(seg_id).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (not isinstance(meta, dict)
                or meta.get("format") != _SEGMENT_FORMAT
                or meta.get("id") != seg_id
                or meta.get("fp") != self.fp):
            return None
        core = {key: value for key, value in meta.items()
                if key not in ("format", "id")}
        if _content_id(_SEGMENT_FORMAT, core) != seg_id:
            return None
        payload = self._segment_payload_path(seg_id)
        try:
            if payload.stat().st_size != int(meta["payload_bytes"]):
                return None
            if verify_payload and _sha256_file(payload) != meta.get("payload_sha256"):
                return None
        except (OSError, TypeError, ValueError):
            return None
        return meta

    def _rebuild_segment_index(self) -> None:
        segment_metas = {}
        for path in self.dir.glob("*.seg.json"):
            seg_id = path.name[:-len(".seg.json")]
            meta = self._read_segment_meta(seg_id)
            if meta is None:
                continue
            segment_metas[seg_id] = meta
        self._replace_segment_index(segment_metas)

    def _replace_segment_index(self, segment_metas: dict[str, dict]) -> None:
        """Refresh the process-local dedup index from a validated snapshot."""
        self._segment_index.clear()
        for seg_id, meta in segment_metas.items():
            key = self._segment_index_key(
                meta.get("parent"), meta.get("tokens", ()),
                meta.get("compressed_mla", False))
            self._segment_index.setdefault(key, []).append(seg_id)

    def _write_segment_if_missing(self, parent: str | None, delta_tokens,
                                   kv: KVCache, start: int, end: int) -> str:
        delta_tokens = tuple(delta_tokens)
        index_key = self._segment_index_key(
            parent, delta_tokens, kv.compressed_mla)
        for candidate in list(self._segment_index.get(index_key, ())):
            paths = (self._segment_meta_path(candidate),
                     self._segment_payload_path(candidate))
            with self._lease(paths):
                if self._read_segment_meta(candidate, verify_payload=True) is not None:
                    return candidate

        arrays = _slice_kv(kv, start, end)
        tmp_payload, payload_sha256, payload_bytes = _write_safetensors_temp(
            self.dir, arrays)
        core = {
            "fp": self.fp,
            "parent": parent,
            "tokens": list(delta_tokens),
            "compressed_mla": bool(kv.compressed_mla),
            "payload_sha256": payload_sha256,
            "payload_bytes": payload_bytes,
        }
        seg_id = _content_id(_SEGMENT_FORMAT, core)
        try:
            manifest = {"format": _SEGMENT_FORMAT, "id": seg_id, **core}
            with self._locked(exclusive=False):
                _publish_temp_immutable(
                    tmp_payload, self._segment_payload_path(seg_id), payload_sha256)
                _publish_json_immutable(self._segment_meta_path(seg_id), manifest)
        except RuntimeError:
            # A checksum-addressed name occupied by different bytes is a corrupt
            # generation, not permission to overwrite an immutable object.
            # Publish an independently-addressed repair generation and let GC
            # quarantine/sweep the unreachable corrupt one.
            core["recovery_nonce"] = uuid.uuid4().hex
            core["recovery_of"] = seg_id
            seg_id = _content_id(_SEGMENT_FORMAT, core)
            manifest = {"format": _SEGMENT_FORMAT, "id": seg_id, **core}
            tmp_payload, retry_sha256, retry_bytes = _write_safetensors_temp(
                self.dir, arrays)
            if (retry_sha256 != payload_sha256
                    or retry_bytes != payload_bytes):
                tmp_payload.unlink(missing_ok=True)
                raise RuntimeError("non-deterministic segment serialization")
            with self._locked(exclusive=False):
                _publish_temp_immutable(
                    tmp_payload, self._segment_payload_path(seg_id), payload_sha256)
                _publish_json_immutable(self._segment_meta_path(seg_id), manifest)
        self._segment_index.setdefault(index_key, []).append(seg_id)
        return seg_id

    def _walk_chain(self, leaf: str,
                    segment_metas: dict[str, dict] | None = None
                    ) -> list[str] | None:
        """Root-to-leaf segment ids, or None if the chain is broken (a
        missing/corrupt/foreign-fingerprint ancestor) -- never trust a
        partial reconstruction."""
        chain = []
        cur: str | None = leaf
        seen = set()
        while cur is not None:
            if cur in seen:
                return None  # cycle: corrupt, refuse rather than loop forever
            seen.add(cur)
            chain.append(cur)
            meta = (segment_metas.get(cur) if segment_metas is not None
                    else self._read_segment_meta(cur))
            if meta is None or meta.get("fp") != self.fp:
                return None
            cur = meta.get("parent")
        chain.reverse()
        return chain

    # ---- checkpoints ----

    def _checkpoint_meta_path(self, checkpoint_id: str) -> Path:
        return self.dir / f"{checkpoint_id}.ckpt.json"

    def _checkpoint_payload_path(self, checkpoint_id: str) -> Path:
        return self.dir / f"{checkpoint_id}.ckpt.safetensors"

    def _read_checkpoint_meta(
            self, checkpoint_id: str, *, verify_payload: bool = False
    ) -> dict | None:
        try:
            meta = json.loads(self._checkpoint_meta_path(checkpoint_id).read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (not isinstance(meta, dict)
                or meta.get("format") != _CHECKPOINT_FORMAT
                or meta.get("id") != checkpoint_id
                or meta.get("fp") != self.fp):
            return None
        core = {key: value for key, value in meta.items()
                if key not in ("format", "id")}
        if _content_id(_CHECKPOINT_FORMAT, core) != checkpoint_id:
            return None
        payload = self._checkpoint_payload_path(checkpoint_id)
        try:
            if payload.stat().st_size != int(meta["payload_bytes"]):
                return None
            if verify_payload and _sha256_file(payload) != meta.get("payload_sha256"):
                return None
        except (OSError, TypeError, ValueError):
            return None
        return meta

    def save(self, parent_chain: tuple[str, ...], parent_covered: int, tokens,
             kv: KVCache, logits: mx.array, prompt_logits: mx.array,
             prompt_length: int, reusable_prefix: int,
             approximate: bool = False,
             tool_capsules=()) -> tuple[str, ...]:
        # Hold a shared journal lock from parent validation through checkpoint
        # publication. GC's exclusive lock cannot retire a just-written segment
        # in the small window before its checkpoint makes it reachable.
        with self._locked(exclusive=False):
            return self._save_locked(
                parent_chain, parent_covered, tokens, kv, logits,
                prompt_logits, prompt_length, reusable_prefix, approximate,
                tool_capsules)

    def _save_locked(self, parent_chain: tuple[str, ...], parent_covered: int,
                     tokens, kv: KVCache, logits: mx.array,
                     prompt_logits: mx.array, prompt_length: int,
                     reusable_prefix: int,
                     approximate: bool = False,
                     tool_capsules=()) -> tuple[str, ...]:
        """Persist a slot as new segments on top of `parent_chain` (which the
        caller has already validated as a true prefix of `tokens`, covering
        exactly the first `parent_covered` tokens) plus a checkpoint for the
        resulting leaf. Returns the full new root-to-leaf chain. Only ever
        writes the DELTA past `parent_covered` -- true O(delta) once a
        parent exists.

        `parent_covered` MUST be passed explicitly rather than derived as
        `len(parent_chain) * chunk_size`: that arithmetic only holds while
        every segment in the chain is exactly one full chunk, which is true
        for a "branch" parent chain but false once a "repeat" parent chain
        includes a prompt-tail segment (shorter than a full chunk by
        construction). The caller already knows the true covered length
        (it's the same `matched` value the in-memory lookup computed)."""
        chunk = self.chunk_size
        tool_capsules = _normalize_tool_capsules(
            tool_capsules, int(prompt_length), strict=True)
        covered = parent_covered
        parent_id = parent_chain[-1] if parent_chain else None
        chain = list(parent_chain)
        pos = covered
        while pos + chunk <= reusable_prefix:
            end = pos + chunk
            seg_id = self._write_segment_if_missing(parent_id, tokens[pos:end], kv, pos, end)
            chain.append(seg_id)
            parent_id = seg_id
            pos = end
        # Prompt-tail: whatever's left of the ACTUAL PROMPT past the last
        # full chunk -- kept as its OWN segment, separate from generation
        # below, specifically so the "repeat" match case (a later request
        # whose tokens equal exactly this prompt, before any of ITS OWN
        # generation) has a real node to address as a parent. Bundling this
        # with generation (as v2's first cut did) buried that boundary
        # inside one segment with no way to stop there.
        if pos < prompt_length:
            seg_id = self._write_segment_if_missing(parent_id, tokens[pos:prompt_length], kv, pos, prompt_length)
            chain.append(seg_id)
            parent_id = seg_id
            pos = prompt_length
        # Generation: the model's own continuation past the prompt. A
        # DIFFERENT request repeating this same prompt will never include
        # this segment in ITS OWN parent chain -- see the "repeat" case in
        # engine.py's persist_parent_chain derivation -- so N independent
        # continuations of one shared prompt each get their own sibling
        # generation segment instead of being forced through this one.
        if pos < len(tokens):
            seg_id = self._write_segment_if_missing(parent_id, tokens[pos:], kv, pos, len(tokens))
            chain.append(seg_id)
            parent_id = seg_id
        leaf = parent_id
        if leaf is None:
            # Nothing to persist at all (empty tokens) -- should not happen
            # in practice (hot_prompt_kv_min_tokens gates tiny prompts), but
            # there is no leaf to checkpoint.
            return tuple(chain)
        tmp_payload, payload_sha256, payload_bytes = _write_safetensors_temp(
            self.dir, {"logits": logits, "prompt_logits": prompt_logits})
        core = {
            "fp": self.fp,
            "leaf": leaf,
            "prompt_length": int(prompt_length),
            "reusable_prefix": int(reusable_prefix),
            "approximate": bool(approximate),
            "tool_capsules": [list(span) for span in tool_capsules],
            "payload_sha256": payload_sha256,
            "payload_bytes": payload_bytes,
        }
        checkpoint_id = _content_id(_CHECKPOINT_FORMAT, core)
        try:
            manifest = {
                "format": _CHECKPOINT_FORMAT,
                "id": checkpoint_id,
                **core,
            }
            _publish_temp_immutable(
                tmp_payload, self._checkpoint_payload_path(checkpoint_id),
                payload_sha256)
            _publish_json_immutable(
                self._checkpoint_meta_path(checkpoint_id), manifest)
        except RuntimeError:
            # The semantic generation name is occupied by corrupt bytes or a
            # corrupt manifest.  Preserve append-only immutability: publish a
            # separately addressed repair generation instead of replacing the
            # committed file.  This mirrors segment collision recovery.
            core["recovery_nonce"] = uuid.uuid4().hex
            core["recovery_of"] = checkpoint_id
            checkpoint_id = _content_id(_CHECKPOINT_FORMAT, core)
            manifest = {
                "format": _CHECKPOINT_FORMAT,
                "id": checkpoint_id,
                **core,
            }
            tmp_payload, retry_sha256, retry_bytes = _write_safetensors_temp(
                self.dir, {"logits": logits, "prompt_logits": prompt_logits})
            if (retry_sha256 != payload_sha256
                    or retry_bytes != payload_bytes):
                tmp_payload.unlink(missing_ok=True)
                raise RuntimeError("non-deterministic checkpoint serialization")
            _publish_temp_immutable(
                tmp_payload, self._checkpoint_payload_path(checkpoint_id),
                payload_sha256)
            _publish_json_immutable(
                self._checkpoint_meta_path(checkpoint_id), manifest)
        # Recency is separate from generation contents; touching does not
        # replace either immutable file.
        os.utime(self._checkpoint_meta_path(checkpoint_id), None)
        os.utime(self._checkpoint_payload_path(checkpoint_id), None)
        _fsync_dir(self.dir)
        return tuple(chain)

    def _reconstruct_tokens(
            self, chain: list[str],
            segment_metas: dict[str, dict] | None = None
    ) -> list[int] | None:
        """Cheap, metadata-only reconstruction of a chain's full token list
        -- reads only small .seg.json files, never the .safetensors
        payloads. Used to SCORE a candidate before paying to load any KV
        tensors for it."""
        tokens: list[int] = []
        for seg_id in chain:
            meta = (segment_metas.get(seg_id) if segment_metas is not None
                    else self._read_segment_meta(seg_id))
            if meta is None:
                return None
            tokens.extend(meta["tokens"])
        return tokens

    def find_best_match(self, tokens, chunk_size: int) -> dict | None:
        """Cheap (metadata-only, no tensors loaded) scan of every
        checkpoint for the best repeat/endpoint/extension/branch candidate against
        `tokens`. Intended to be called ONLY on a total in-memory miss --
        it does not compete with an in-memory hit, it only fills the gap
        when memory has nothing useful (e.g. more concurrent agentic/cron
        tasks sharing a preamble than fit in `hot_prompt_kv_slots`, but the
        shared prefix is still sitting on disk from an earlier task).
        Returns a dict describing the winner, with NO KV tensors loaded
        yet -- pass it to load_matched_chain() to actually load."""
        candidates = []
        with self._locked(exclusive=False):
            checkpoint_paths = list(self.dir.glob("*.ckpt.json"))
            # Checkpoints share long ancestor chains. Validate every small
            # segment manifest once for this lookup instead of reparsing the
            # same root-to-leaf prefix for each checkpoint.
            segment_metas = {}
            for path in self.dir.glob("*.seg.json"):
                seg_id = path.name[:-len(".seg.json")]
                meta = self._read_segment_meta(seg_id)
                if meta is not None:
                    segment_metas[seg_id] = meta
        for j in checkpoint_paths:
            checkpoint_id = j.name[:-len(".ckpt.json")]
            meta = self._read_checkpoint_meta(checkpoint_id)
            if meta is None:
                continue
            leaf = meta.get("leaf")
            if not leaf:
                continue
            ckpt_payload = self._checkpoint_payload_path(checkpoint_id)
            if not ckpt_payload.exists():
                continue
            chain = self._walk_chain(leaf, segment_metas)
            if not chain:
                continue
            cand_tokens = self._reconstruct_tokens(chain, segment_metas)
            if cand_tokens is None:
                continue
            scored = _score_match(
                tokens, cand_tokens, int(meta["prompt_length"]),
                int(meta["reusable_prefix"]), len(chain), chunk_size)
            if scored is None:
                continue
            case, matched, watermark, n_segments, lcp = scored
            try:
                mtime = j.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append({
                "case": case, "matched": matched, "watermark": watermark,
                "n_segments": n_segments, "lcp": lcp, "leaf": leaf,
                "chain": chain, "checkpoint_id": checkpoint_id,
                "ckpt_payload": ckpt_payload, "mtime": mtime,
                "approximate": bool(meta.get("approximate", False)),
                "tool_capsules": _normalize_tool_capsules(
                    meta.get("tool_capsules", ()),
                    int(meta["prompt_length"]), strict=False),
            })

        # Prefer the longest/newest generation, but prove every payload hash
        # before returning it. If the newest generation is torn or corrupted,
        # an older immutable generation remains eligible instead of forcing a
        # cold miss (or, worse, loading unverified bytes).
        candidates.sort(
            key=lambda value: (
                value["matched"], value["lcp"], value["mtime"]),
            reverse=True,
        )
        for candidate in candidates:
            if candidate["tool_capsules"] is None:
                continue
            checkpoint_id = candidate["checkpoint_id"]
            verify_paths = [
                self._checkpoint_meta_path(checkpoint_id),
                self._checkpoint_payload_path(checkpoint_id),
            ]
            verify_paths.extend(
                path for seg_id in candidate["chain"] for path in (
                    self._segment_meta_path(seg_id),
                    self._segment_payload_path(seg_id)))
            with self._lease(verify_paths):
                try:
                    before = _file_signatures(verify_paths)
                except OSError:
                    continue
                if self._read_checkpoint_meta(
                        checkpoint_id, verify_payload=True) is None:
                    continue
                if any(self._read_segment_meta(
                        seg_id, verify_payload=True) is None
                       for seg_id in candidate["chain"]):
                    continue
                try:
                    after = _file_signatures(verify_paths)
                except OSError:
                    continue
                # A concurrent external mutation is not covered by the journal
                # lease. Do not carry a proof unless the complete validation
                # observed one stable generation of every immutable file.
                if before != after:
                    continue
            candidate["_verified_files"] = after
            return candidate
        return None

    def _load_chain_prefix(self, chain: list[str], num_layers: int,
                           verified_files=None):
        """Load exactly `chain`'s KV tensors (the caller has already sliced
        it to whatever prefix length it wants -- e.g. only `n_segments` of
        a longer chain for a branch/repeat disk match) and concatenate them
        into one KVCache. Returns (tokens, kv) or None on a corrupt/missing
        payload (checked again here, not just at metadata-scan time -- a
        .safetensors file can vanish or truncate independently of its
        .json sibling)."""
        paths = [path for seg_id in chain for path in (
            self._segment_meta_path(seg_id),
            self._segment_payload_path(seg_id))]
        try:
            with self._lease(paths):
                seg_metas, seg_arrays = [], []
                for seg_id in chain:
                    segment_paths = (
                        self._segment_meta_path(seg_id),
                        self._segment_payload_path(seg_id),
                    )
                    seg_meta = self._read_segment_meta(
                        seg_id,
                        verify_payload=not _signatures_match(
                            segment_paths, verified_files),
                    )
                    if seg_meta is None:
                        raise ValueError(
                            f"segment checksum/manifest validation failed: {seg_id}")
                    seg_metas.append(seg_meta)
                    seg_arrays.append(mx.load(
                        str(self._segment_payload_path(seg_id))))
                mx.eval([a for arr in seg_arrays for a in arr.values()])
        except Exception as e:
            print(f"[hot-kv-persist] chain load failed: {type(e).__name__}: {e}",
                  flush=True)
            return None

        compressed_mla = bool(
            seg_metas[0].get("compressed_mla")) if seg_metas else False
        if any(bool(meta.get("compressed_mla")) != compressed_mla
               for meta in seg_metas):
            print("[hot-kv-persist] mixed KV layouts in one chain; refusing to use",
                  flush=True)
            return None
        axis = 1 if compressed_mla else 2
        kv = KVCache(num_layers)
        kv.compressed_mla = compressed_mla
        pieces: dict[str, list[mx.array]] = {}
        tokens: list[int] = []
        for seg_meta, seg_arr in zip(seg_metas, seg_arrays):
            tokens.extend(seg_meta["tokens"])
            for name, arr in seg_arr.items():
                pieces.setdefault(name, []).append(arr)

        # One concatenate per tensor makes reconstruction O(total bytes).
        # The v2 loop concatenated after every segment and became quadratic at
        # million-token journal depths.
        dsa_keys: dict[int, mx.array] = {}
        for name, arrays in pieces.items():
            concat_axis = 1 if name.startswith("dsa_k") else axis
            value = arrays[0] if len(arrays) == 1 else mx.concatenate(
                arrays, axis=concat_axis)
            if name.startswith("k") and name[1:].isdigit():
                kv.keys[int(name[1:])] = value
            elif name.startswith("v") and name[1:].isdigit():
                kv.values[int(name[1:])] = value
            elif name.startswith("dsa_k") and name[len("dsa_k"):].isdigit():
                dsa_keys[int(name[len("dsa_k"):])] = value
        mx.eval([
            value for value in (*kv.keys, *kv.values, *dsa_keys.values())
            if value is not None
        ])
        if kv.offset != len(tokens):
            print(f"[hot-kv-persist] chain load: kv.offset={kv.offset} != "
                  f"len(tokens)={len(tokens)}, refusing to use", flush=True)
            return None
        if dsa_keys:
            if self.config is None:
                print("[hot-kv-persist] DSA state has no model config; refusing to use",
                      flush=True)
                return None
            from .glm_dsa import DSAState

            dsa = DSAState(self.config)
            dsa.k_idx = dsa_keys
            kv.dsa = dsa
        elif self.require_dsa:
            print("[hot-kv-persist] checkpoint is missing required DSA state; "
                  "refusing to use", flush=True)
            return None
        return tokens, kv

    def load_matched_chain(self, match: dict, num_layers: int) -> tuple | None:
        """Actually load the KV tensors for a find_best_match() winner --
        exactly `match['n_segments']` segments from the root, never more,
        so a branch/repeat match never pays to load tokens past the
        matched point. Returns (tokens, kv, exact_logits_or_None) or None
        on a load failure discovered only now (metadata was fine but a
        payload vanished/truncated)."""
        chain = match["chain"][: match["n_segments"]]
        verified_files = match.get("_verified_files")
        loaded = self._load_chain_prefix(
            chain, num_layers, verified_files=verified_files)
        if loaded is None:
            return None
        tokens, kv = loaded
        if len(tokens) != match["matched"]:
            print(f"[hot-kv-persist] disk match leaf={match['leaf']}: "
                  f"reconstructed length {len(tokens)} != expected "
                  f"{match['matched']}, refusing to use", flush=True)
            return None
        exact_logits = None
        if match["case"] in ("repeat", "endpoint"):
            checkpoint_id = match["checkpoint_id"]
            checkpoint_paths = (
                self._checkpoint_meta_path(checkpoint_id),
                self._checkpoint_payload_path(checkpoint_id),
            )
            try:
                with self._lease(checkpoint_paths):
                    meta = self._read_checkpoint_meta(
                        checkpoint_id,
                        verify_payload=not _signatures_match(
                            checkpoint_paths, verified_files),
                    )
                    if meta is None or meta.get("leaf") != match["leaf"]:
                        raise ValueError(
                            "checkpoint checksum/manifest validation failed")
                    ck = mx.load(str(match["ckpt_payload"]))
                    if "logits" not in ck or "prompt_logits" not in ck:
                        raise KeyError("missing logits/prompt_logits")
                    mx.eval([ck["logits"], ck["prompt_logits"]])
            except Exception as e:
                print(f"[hot-kv-persist] disk match leaf={match['leaf']}: "
                      f"checkpoint payload load failed: {type(e).__name__}: "
                      f"{e}", flush=True)
                return None
            exact_logits = ck["prompt_logits"] if match["case"] == "repeat" else ck["logits"]
        checkpoint_id = match["checkpoint_id"]
        with self._locked(exclusive=False):
            for path in (
                    self._checkpoint_meta_path(checkpoint_id),
                    self._checkpoint_payload_path(checkpoint_id)):
                try:
                    os.utime(path, None)
                except OSError:
                    pass
        return tuple(tokens), kv, exact_logits

    def load_all(self, num_layers: int, limit: int) -> list[tuple]:
        """Return up to `limit` (tokens, kv, logits, prompt_length,
        prompt_logits, reusable_prefix, approximate, tool_capsules,
        segment_chain) tuples, oldest-mtime
        first -- ready to append straight into `_hot_prompt_slots` in LRU
        order. Corrupt, fingerprint-mismatched, or broken-chain checkpoints
        are skipped, never fatal (same posture as F37)."""
        entries = []
        for j in self.dir.glob("*.ckpt.json"):
            checkpoint_id = j.name[:-len(".ckpt.json")]
            meta = self._read_checkpoint_meta(checkpoint_id)
            if meta is None:
                continue
            st = self._checkpoint_payload_path(checkpoint_id)
            if not meta.get("leaf") or not st.exists():
                continue
            entries.append((st.stat().st_mtime, checkpoint_id, meta, st))
        entries.sort(key=lambda e: e[0])  # oldest first
        if limit and len(entries) > limit:
            entries = entries[-limit:]  # keep the most-recently-used `limit`

        out = []
        for _, checkpoint_id, meta, ckpt_payload in entries:
            leaf = meta["leaf"]
            chain = self._walk_chain(leaf)
            if not chain:
                print(f"[hot-kv-persist] skip checkpoint with broken/missing "
                      f"segment chain (leaf={leaf})", flush=True)
                continue
            try:
                with self._lease((
                        self._checkpoint_meta_path(checkpoint_id),
                        ckpt_payload)):
                    verified = self._read_checkpoint_meta(
                        checkpoint_id, verify_payload=True)
                    if verified is None:
                        raise ValueError(
                            "checkpoint checksum/manifest validation failed")
                    ck = mx.load(str(ckpt_payload))
                    if "logits" not in ck or "prompt_logits" not in ck:
                        raise KeyError("missing logits/prompt_logits")
                    mx.eval([ck["logits"], ck["prompt_logits"]])
            except Exception as e:
                print(f"[hot-kv-persist] skip corrupt/unreadable checkpoint "
                      f"payload (leaf={leaf}): {type(e).__name__}: {e}", flush=True)
                continue
            loaded = self._load_chain_prefix(chain, num_layers)
            if loaded is None:
                print(f"[hot-kv-persist] skip checkpoint with unloadable "
                      f"chain (leaf={leaf})", flush=True)
                continue
            tokens, kv = loaded
            tool_capsules = _normalize_tool_capsules(
                meta.get("tool_capsules", ()), int(meta["prompt_length"]),
                strict=False)
            if tool_capsules is None:
                print(f"[hot-kv-persist] skip checkpoint with invalid tool "
                      f"capsule metadata (leaf={leaf})", flush=True)
                continue
            out.append((
                tuple(tokens), kv, ck["logits"], int(meta["prompt_length"]),
                ck["prompt_logits"], int(meta["reusable_prefix"]),
                bool(meta.get("approximate", False)), tool_capsules,
                tuple(chain),
            ))
        return out

    def gc(self) -> int:
        """Lease-aware mark/sweep over immutable checkpoint generations.

        Retention is bounded first by checkpoint count and then by reachable
        bytes. Readers publish leases before validating/loading; GC holds the
        exclusive journal lock and never unlinks a leased manifest or payload.
        """
        with self._locked(exclusive=True):
            leased = self._leased_names_locked()
            entries = []  # (mtime, checkpoint_id, meta, chain)
            removed = 0
            changed = False

            # Many checkpoints share the same ancestors. GC needs only one
            # validated metadata snapshot while it holds the exclusive lock;
            # walking each chain from disk made a no-op collection quadratic in
            # checkpoint depth.
            segment_metas = {}
            for path in self.dir.glob("*.seg.json"):
                seg_id = path.name[:-len(".seg.json")]
                meta = self._read_segment_meta(seg_id)
                if meta is not None:
                    segment_metas[seg_id] = meta

            for path in list(self.dir.glob("*.ckpt.json")):
                checkpoint_id = path.name[:-len(".ckpt.json")]
                meta = self._read_checkpoint_meta(checkpoint_id)
                chain = (self._walk_chain(meta.get("leaf"), segment_metas)
                         if meta else None)
                if meta is None or not chain:
                    pair = (path, self._checkpoint_payload_path(checkpoint_id))
                    if not any(item.name in leased for item in pair):
                        for item in pair:
                            item.unlink(missing_ok=True)
                        removed += 1
                        changed = True
                    continue
                try:
                    mtime = max(
                        path.stat().st_mtime,
                        self._checkpoint_payload_path(checkpoint_id).stat().st_mtime)
                except OSError:
                    continue
                entries.append((mtime, checkpoint_id, meta, chain))
            entries.sort(key=lambda item: item[0])

            def checkpoint_leased(entry) -> bool:
                _mtime, checkpoint_id, _meta, _chain = entry
                return bool({
                    self._checkpoint_meta_path(checkpoint_id).name,
                    self._checkpoint_payload_path(checkpoint_id).name,
                } & leased)

            def live_segments(values) -> set[str]:
                return {seg_id for _m, _c, _meta, chain in values
                        for seg_id in chain}

            def paths_bytes(paths) -> int:
                total = 0
                for path in paths:
                    try:
                        total += path.stat().st_size
                    except OSError:
                        pass
                return total

            def checkpoint_bytes(entry) -> int:
                _mtime, checkpoint_id, _meta, _chain = entry
                return paths_bytes((
                    self._checkpoint_meta_path(checkpoint_id),
                    self._checkpoint_payload_path(checkpoint_id),
                ))

            segment_bytes = {
                seg_id: paths_bytes((
                    self._segment_meta_path(seg_id),
                    self._segment_payload_path(seg_id),
                ))
                for seg_id in segment_metas
            }

            stale = []
            if self.max_checkpoints and len(entries) > self.max_checkpoints:
                excess = len(entries) - self.max_checkpoints
                kept = []
                for entry in entries:
                    if excess and not checkpoint_leased(entry):
                        stale.append(entry)
                        excess -= 1
                    else:
                        kept.append(entry)
                entries = kept

            if self.max_bytes:
                segment_refs = Counter(
                    seg_id for _m, _c, _meta, chain in entries
                    for seg_id in chain)
                reachable = sum(checkpoint_bytes(entry) for entry in entries)
                reachable += sum(
                    segment_bytes.get(seg_id, 0) for seg_id in segment_refs)
                byte_stale_ids = set()
                for entry in entries:
                    if reachable <= self.max_bytes:
                        break
                    if checkpoint_leased(entry):
                        continue
                    _mtime, checkpoint_id, _meta, chain = entry
                    stale.append(entry)
                    byte_stale_ids.add(checkpoint_id)
                    reachable -= checkpoint_bytes(entry)
                    for seg_id in chain:
                        segment_refs[seg_id] -= 1
                        if segment_refs[seg_id] == 0:
                            reachable -= segment_bytes.get(seg_id, 0)
                if byte_stale_ids:
                    entries = [
                        entry for entry in entries
                        if entry[1] not in byte_stale_ids
                    ]

            for _mtime, checkpoint_id, _meta, _chain in stale:
                for path in (
                        self._checkpoint_meta_path(checkpoint_id),
                        self._checkpoint_payload_path(checkpoint_id)):
                    if path.name not in leased:
                        path.unlink(missing_ok=True)
                removed += 1
                changed = True

            live = live_segments(entries)
            for path in list(self.dir.glob("*.seg.json")):
                seg_id = path.name[:-len(".seg.json")]
                pair = (path, self._segment_payload_path(seg_id))
                if seg_id not in live and not any(
                        item.name in leased for item in pair):
                    for item in pair:
                        item.unlink(missing_ok=True)
                    removed += 1
                    changed = True

            # Payload-only remnants are uncommitted generations. Shared writer
            # locks prevent an in-flight publication from appearing here.
            for pattern in ("*.seg.safetensors", "*.ckpt.safetensors"):
                for payload in self.dir.glob(pattern):
                    if payload.name in leased:
                        continue
                    stem = payload.name[:-len(".safetensors")]
                    if not (self.dir / f"{stem}.json").exists():
                        payload.unlink(missing_ok=True)
                        removed += 1
                        changed = True
            for tmp in self.dir.glob(".*.tmp.*"):
                if tmp.name not in leased:
                    tmp.unlink(missing_ok=True)
                    changed = True

            # No-op collections do not need a durability barrier and the
            # in-memory index is already current. When anything was unlinked,
            # retain the original fsync-before-return guarantee and rebuild.
            if changed:
                _fsync_dir(self.dir)
                self._rebuild_segment_index()
            else:
                # Preserve the old cross-process refresh semantics without a
                # second metadata read: the exclusive-lock snapshot is current.
                self._replace_segment_index(segment_metas)
            return removed
