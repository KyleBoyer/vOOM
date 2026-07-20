"""vpack2: single-file archive with tensors laid out in ACCESS ORDER, so every
fetch becomes a few large sequential reads instead of many small random ones.

Why: per-tensor files (vpack) made MoE expert paging random-read bound — 12.6 MB
scattered reads collapse this USB drive from 315 to ~23 MB/s. vpack2 lays tensors
out embed → per layer (attn/norms/router, then experts 0..E in id order) → final
norm → lm_head, and the reader merges adjacent requests into runs (gap tolerance),
so a layer page or a batch of routed experts is a handful of ordered sequential
reads.

Build is pure concatenation of existing vpack .vt bodies (no recompression) —
`build_from_vpack` streams, so RAM stays flat even for 100 GB models.

Files: weights.vpack2 (bodies only, back to back) + weights.vpack2.index.json:
  {name: {"off": int, "len": int, "head": {...same head dict as vpack...}}}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
import warnings
from pathlib import Path

from .packed import to_mx  # decode is shared with vpack

try:
    from compression import zstd
except ImportError:  # pragma: no cover
    import zstd  # type: ignore

import numpy as np

_COPY_CHUNK = 64 * 1024 * 1024
RUN_GAP = 2 * 1024 * 1024  # merge reads separated by less than this


def _pread_exact(fd: int, length: int, offset: int) -> bytes:
    """Read one immutable extent completely or fail closed on truncation.

    POSIX permits ``pread`` to return fewer bytes than requested without EOF.
    Treating one call as complete could feed a truncated compressed body to the
    decoder; retry short reads and raise if the archive actually ends early.
    """
    parts: list[bytes] = []
    done = 0
    while done < length:
        chunk = os.pread(fd, length - done, offset + done)
        if not chunk:
            raise EOFError(
                f"unexpected EOF at archive offset {offset + done}; "
                f"wanted {length - done} more bytes"
            )
        parts.append(chunk)
        done += len(chunk)
    return b"".join(parts)


def access_order(names: list[str], expert_rank: dict | None = None) -> list[str]:
    """embed → layers ascending (non-expert tensors first, then experts) →
    everything else (final norm, lm_head) in name order.

    expert_rank: optional {(layer, expert): rank} — F20 heat ordering. Experts are
    laid out by rank (hot first) instead of id, so co-activating routed sets fall
    into fewer, longer disk runs. Unranked experts sort after ranked ones by id."""

    def key(n: str):
        if "embed_tokens" in n:
            return (0, 0, 0, n)
        # Multimodal wrappers use either model.language_model.layers.*
        # (Qwen3-VL/Qwen3.5/3.6) or language_model.model.layers.* (Kimi
        # K2.5).  Pack manifests retain the released physical names; recognize
        # all three forms here so the final archive is truly layer/expert
        # ordered before WeightStore later canonicalizes them to model.layers.*.
        m = re.match(
            r"(?:model\.layers|model\.language_model\.layers|"
            r"language_model\.model\.layers)\."
            r"(\d+)\.(?:mlp\.experts\.(\d+)\.)?",
            n,
        )
        if m:
            layer = int(m.group(1))
            if m.group(2) is None:
                return (1, layer, 0, 0, n)
            e = int(m.group(2))
            rank = expert_rank.get((layer, e), 10**9) if expert_rank else e
            return (1, layer, 1, rank, n)
        return (2, 0, 0, 0, n)

    # normalize non-expert tuples to same arity
    def key5(n: str):
        k = key(n)
        return k if len(k) == 5 else (k[0], k[1], k[2], 0, k[-1])

    return sorted(names, key=key5)


def heat_rank_from_transitions(path: str | Path) -> dict:
    """F20: {(layer, expert): rank} by descending usage mass from a persisted
    expert_transitions.json (both endpoints of each transition contribute)."""
    import json as _json
    from collections import defaultdict

    heat: dict = defaultdict(int)
    for key, c in _json.loads(Path(path).read_text()).items():
        l, e, f = (int(v) for v in key.split(","))
        heat[(l, e)] += c
        heat[(l + 1, f)] += c
    rank: dict = {}
    by_layer: dict = defaultdict(list)
    for (l, e), h in heat.items():
        by_layer[l].append((h, e))
    for l, items in by_layer.items():
        for r, (_, e) in enumerate(sorted(items, reverse=True)):
            rank[(l, e)] = r
    return rank


def build_from_vpack(model_dir: str | Path, consume_source: bool = False,
                     expert_rank: dict | None = None, progress=None) -> Path:
    """consume_source=True deletes each .vt after it is appended, so conversion
    needs only ~one tensor of free disk. A partial index is checkpointed every
    200 tensors (weights.vpack2.index.partial.json) so an interrupted consuming
    build loses nothing: entries in the partial index are already in the archive;
    resume = finish remaining .vt files (not yet implemented — restore from the
    partial index manually if this ever bites)."""
    model_dir = Path(model_dir)
    vpack = model_dir / "weights.vpack"
    manifest = json.loads((vpack / "manifest.json").read_text())
    archive = model_dir / "weights.vpack2"
    partial = model_dir / "weights.vpack2.index.partial.json"
    index: dict[str, dict] = {}

    t0 = time.perf_counter()
    total = len(manifest)
    with open(archive, "wb") as out:
        for count, name in enumerate(access_order(list(manifest), expert_rank)):
            if progress and count % 50 == 0:
                progress(count, total)
            src_path = vpack / manifest[name]
            with open(src_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
                off = out.tell()
                h = hashlib.sha256()  # F31: content hash rides in the index
                while True:
                    chunk = f.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                    out.write(chunk)
                index[name] = {"off": off, "len": out.tell() - off, "head": head,
                               "sha": h.hexdigest()[:32]}
            if consume_source:
                src_path.unlink()
                if count % 200 == 0:
                    out.flush()
                    partial.write_text(json.dumps(index))
    (model_dir / "weights.vpack2.index.json").write_text(json.dumps(index))
    partial.unlink(missing_ok=True)
    if consume_source:
        (vpack / "manifest.json").unlink(missing_ok=True)
        try:
            vpack.rmdir()
        except OSError:
            pass  # leftover files; harmless
    if progress:
        progress(total, total)
    print(f"vpack2: {archive.stat().st_size / 1e9:.1f}GB, {len(index)} tensors, "
          f"{time.perf_counter() - t0:.0f}s")
    return archive


def decode_body(head: dict, body: bytes):
    """vpack body bytes -> original raw bytes/np array (mirrors read_tensor_bytes)."""
    if "raw" in head:
        return body
    if head.get("streamed"):
        lo = body[: head["lo"]]
        hi = zstd.decompress(body[head["lo"]:])
    else:
        hi = zstd.decompress(body[: head["hi_z"]])
        lo = body[head["hi_z"]:]
    raw = np.empty(len(hi) * 2, dtype=np.uint8)
    raw[0::2] = np.frombuffer(lo, dtype=np.uint8)
    raw[1::2] = np.frombuffer(hi, dtype=np.uint8)
    return raw


def verify_generation(model_dir: str | Path, decode: bool = True,
                      paths: "tuple[Path, Path] | None" = None) -> dict:
    """F31: full extent + content-hash verification of the ACTIVE generation.

    Checks that the index parses, every extent is in-bounds and non-overlapping,
    the archive has no gaps/tail the index doesn't know about, and every body
    matches its recorded sha (entries from pre-hash archives count as
    'unhashed') and — when decode=True — decodes without error.
    Returns a report dict with an 'errors' list (empty = PASS)."""
    model_dir = Path(model_dir)
    report = {"tensors": 0, "hashed": 0, "decoded": 0, "errors": []}
    try:
        archive, index_path = paths if paths is not None else resolve_generation(model_dir)
        index = json.loads(index_path.read_text())
    except Exception as exc:  # missing/corrupt index is itself the finding
        report["errors"].append(f"index unreadable: {exc}")
        return report
    if not archive.exists():
        report["errors"].append(f"archive missing: {archive.name}")
        return report
    size = archive.stat().st_size
    entries = sorted(index.items(), key=lambda kv: kv[1]["off"])
    report["tensors"] = len(entries)
    pos = 0
    for name, e in entries:
        if e["off"] != pos:
            report["errors"].append(f"extent gap/overlap at {name}: off {e['off']} != {pos}")
            pos = e["off"]
        pos += e["len"]
    if pos != size:
        report["errors"].append(f"archive size {size} != indexed extent end {pos}")
    with open(archive, "rb") as f:
        for name, e in entries:
            f.seek(e["off"])
            body = f.read(e["len"])
            if len(body) != e["len"]:
                report["errors"].append(f"short read at {name}")
                continue
            if e.get("sha"):
                report["hashed"] += 1
                if hashlib.sha256(body).hexdigest()[:32] != e["sha"]:
                    report["errors"].append(f"hash mismatch at {name}")
                    continue
            if decode:
                try:
                    decode_body(e["head"], body)
                    report["decoded"] += 1
                except Exception as exc:
                    report["errors"].append(f"decode failure at {name}: {exc}")
    return report


def resolve_generation(model_dir: Path) -> tuple[Path, Path]:
    """F31: a CURRENT pointer names the active immutable generation; legacy
    unversioned names remain valid when no pointer exists."""
    cur = model_dir / "vpack2.CURRENT"
    if cur.exists():
        gen = cur.read_text().strip()
        if not gen or not re.fullmatch(r"[A-Za-z0-9._-]+", gen):
            raise ValueError(f"invalid vpack2 generation name: {gen!r}")
        return model_dir / f"weights.{gen}.vpack2", model_dir / f"weights.{gen}.index.json"
    return model_dir / "weights.vpack2", model_dir / "weights.vpack2.index.json"


class Vpack2Reader:
    def __init__(self, model_dir: str | Path, *, require_hashes: bool = False):
        """Open an archive.

        ``require_hashes=True`` is mandatory for proof runs.  False preserves
        readability of pre-F31 local archives, but marks them as legacy and can
        detect only structural changes—not pre-existing body corruption.
        """
        model_dir = Path(model_dir)
        self.archive, index_path = resolve_generation(model_dir)
        self.index_path = index_path
        self.index = json.loads(index_path.read_text())
        self.require_hashes = require_hashes
        self._validate_index()
        self.unhashed_entries = sum(not e.get("sha") for e in self.index.values())
        self.integrity_mode = "sha-per-body" if not self.unhashed_entries else "legacy-unhashed"
        if self.unhashed_entries:
            warnings.warn(
                f"vpack2 archive has {self.unhashed_entries}/{len(self.index)} unhashed "
                "bodies; readable for compatibility but not a proof artifact",
                RuntimeWarning,
                stacklevel=2,
            )
        st = self.archive.stat()
        self._archive_identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        self.runs_read = 0
        self.tensors_read = 0

    def _validate_index(self) -> None:
        """Fail closed before serving from a malformed or unhashed archive.

        This is a cheap structural check, not the expensive full decode audit in
        ``verify_generation``.  Every body is independently SHA-checked when it
        is fetched, so bit flips cannot silently become model tensors.
        """
        size = self.archive.stat().st_size
        pos = 0
        for name, entry in sorted(self.index.items(), key=lambda item: item[1].get("off", -1)):
            off, length = entry.get("off"), entry.get("len")
            if not isinstance(off, int) or not isinstance(length, int) or length < 0:
                raise ValueError(f"invalid vpack2 extent for {name}")
            if off != pos:
                raise ValueError(
                    f"vpack2 extent gap/overlap at {name}: offset {off}, expected {pos}"
                )
            if not isinstance(entry.get("head"), dict):
                raise ValueError(f"missing vpack2 tensor header for {name}")
            digest = entry.get("sha")
            if digest is not None and (
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest)
            ):
                raise ValueError(f"invalid vpack2 body hash for {name}")
            if self.require_hashes and digest is None:
                raise ValueError(f"missing/invalid vpack2 body hash for {name}")
            pos += length
        if pos != size:
            raise ValueError(f"vpack2 archive size {size} != indexed extent end {pos}")

    def _assert_archive_immutable(self) -> None:
        st = self.archive.stat()
        identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if identity != self._archive_identity:
            raise IOError("vpack2 archive changed after reader initialization")

    def _checked_body(self, name: str, entry: dict, body: bytes) -> bytes:
        expected = entry.get("sha")
        if expected is None:
            if self.require_hashes:
                raise IOError(f"unhashed vpack2 body refused: {name}")
            return body
        actual = hashlib.sha256(body).hexdigest()[:32]
        if actual != expected:
            raise IOError(f"vpack2 body hash mismatch for {name}: {actual} != {expected}")
        return body

    def has(self, name: str) -> bool:
        return name in self.index

    def read_body(self, name: str) -> tuple[dict, bytes]:
        """Return one authenticated compressed body without decoding it.

        Fast-tier staging needs the original vpack payload, including its
        compressed representation, rather than an MLX tensor.  Keeping this on
        the reader centralizes immutable-archive and per-body hash checks.
        """
        self._assert_archive_immutable()
        entry = self.index[name]
        fd = os.open(self.archive, os.O_RDONLY)
        try:
            body = _pread_exact(fd, int(entry["len"]), int(entry["off"]))
        finally:
            os.close(fd)
        return entry["head"], self._checked_body(name, entry, body)

    def fetch(self, names: list[str], parallel: int = 4) -> tuple[dict, float, int]:
        """Coalesce requested tensors into sequential runs; runs are read+decoded on
        a small thread pool (scattered expert reads benefit from queue depth, and
        zstd decode overlaps I/O). Returns (name -> mx.array, seconds, bytes read)."""
        import concurrent.futures as cf
        import os

        t0 = time.perf_counter()
        self._assert_archive_immutable()
        entries = sorted(((self.index[n], n) for n in names), key=lambda e: e[0]["off"])
        runs: list[tuple[int, int, list]] = []
        i = 0
        while i < len(entries):
            start = entries[i][0]["off"]
            end = start + entries[i][0]["len"]
            j = i + 1
            while j < len(entries) and entries[j][0]["off"] - end <= RUN_GAP:
                end = max(end, entries[j][0]["off"] + entries[j][0]["len"])
                j += 1
            runs.append((start, end, entries[i:j]))
            i = j

        fd = os.open(self.archive, os.O_RDONLY)
        out = {}
        nbytes = 0
        try:
            def read_run(run):
                # I/O + zstd/numpy decode only — mx.array creation must happen on
                # the calling thread (MLX streams are not valid in nested threads)
                start, end, ents = run
                blob = _pread_exact(fd, end - start, start)
                tensors = []
                for e, name in ents:
                    rel = e["off"] - start
                    body = self._checked_body(name, e, blob[rel : rel + e["len"]])
                    tensors.append((name, e["head"], decode_body(e["head"], body)))
                return tensors, len(blob)

            if len(runs) == 1 or parallel <= 1:
                results = [read_run(r) for r in runs]
            else:
                with cf.ThreadPoolExecutor(max_workers=min(parallel, len(runs))) as pool:
                    results = list(pool.map(read_run, runs))
            for tensors, blob_len in results:
                nbytes += blob_len
                self.runs_read += 1
                for name, head, raw in tensors:
                    out[name] = to_mx(head, raw)
                    self.tensors_read += 1
        finally:
            os.close(fd)
        import mlx.core as mx

        mx.eval(list(out.values()))
        return out, time.perf_counter() - t0, nbytes


def reorder_vpack2(model_dir: str | Path, expert_rank: dict, staging_dir: str | Path) -> None:
    """F20 for consumed models: rewrite an existing vpack2 archive in heat order
    via a staging directory on another disk (local free space < 2x archive).
    Sequence: stream old->staging in new order, delete local, copy staging back."""
    import shutil

    model_dir = Path(model_dir)
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    old_index = json.loads(resolve_generation(model_dir)[1].read_text())
    new_archive = staging / "weights.vpack2"
    new_index: dict[str, dict] = {}
    t0 = time.perf_counter()
    src_archive, _ = resolve_generation(model_dir)
    with open(src_archive, "rb") as src, open(new_archive, "wb") as out:
        for name in access_order(list(old_index), expert_rank):
            e = old_index[name]
            src.seek(e["off"])
            off = out.tell()
            remaining = e["len"]
            h = hashlib.sha256()
            while remaining:
                chunk = src.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(
                        f"unexpected EOF while reordering {name}: "
                        f"{remaining} bytes still expected"
                    )
                h.update(chunk)
                out.write(chunk)
                remaining -= len(chunk)
            sha = h.hexdigest()[:32]
            # F31: when the source index carries a hash, a mismatch here means the
            # bytes were corrupted in transit — abort BEFORE any commit machinery
            if e.get("sha") and e["sha"] != sha:
                raise IOError(f"reorder read corruption on {name}: {e['sha']} != {sha}")
            new_index[name] = {"off": off, "len": e["len"], "head": e["head"], "sha": sha}
    (staging / "weights.vpack2.index.json").write_text(json.dumps(new_index))
    print(f"staged reorder in {time.perf_counter() - t0:.0f}s; committing new generation", flush=True)
    # F31 transactional commit: copy the staged pair in as an IMMUTABLE new
    # generation, verify (size + a sample-tensor decode smoke test), fsync,
    # then atomically flip the CURRENT pointer. The old generation survives
    # until after the flip. No unlink ever precedes a verified replacement.
    import os
    import uuid as _uuid

    gen = f"g{_uuid.uuid4().hex[:8]}"
    garch = model_dir / f"weights.{gen}.vpack2"
    gidx = model_dir / f"weights.{gen}.index.json"
    shutil.copyfile(str(new_archive), str(garch))
    assert garch.stat().st_size == new_archive.stat().st_size, "generation copy size mismatch"
    gidx.write_text(json.dumps(new_index))
    # F31-v2: FULL verification of the new generation BEFORE the flip —
    # every extent, every body hash, every decode — against the explicit new
    # files, never through the pointer (readers keep the old generation until
    # the verified flip). A generation that fails here is deleted.
    report = verify_generation(model_dir, decode=True, paths=(garch, gidx))
    if report["errors"]:
        garch.unlink(missing_ok=True)
        gidx.unlink(missing_ok=True)
        raise IOError(f"new generation failed verification, flip aborted: "
                      f"{report['errors'][:3]}")
    for path in (garch, gidx):
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    old_arch, old_idx = resolve_generation(model_dir)
    tmp_ptr = model_dir / "vpack2.CURRENT.tmp"
    tmp_ptr.write_text(gen)
    # F31-v3: fsync the tmp pointer's OWN bytes before the rename. Without
    # this, `os.replace` only guarantees the directory-entry rename is atomic
    # — the file's content ("gen") can still be sitting in the page cache and
    # lost on a crash between write_text and replace, leaving CURRENT.tmp (or
    # a renamed-but-empty/truncated CURRENT) instead of the committed pointer.
    tfd = os.open(tmp_ptr, os.O_RDONLY)
    os.fsync(tfd)
    os.close(tfd)
    os.replace(tmp_ptr, model_dir / "vpack2.CURRENT")
    dfd = os.open(model_dir, os.O_RDONLY)  # F31-v2: persist the rename itself
    os.fsync(dfd)
    os.close(dfd)
    # retire the previous generation only after the pointer flip
    for p_old in (old_arch, old_idx):
        if p_old.exists() and p_old != garch:
            p_old.unlink()
    new_archive.unlink(missing_ok=True)
    (staging / "weights.vpack2.index.json").unlink(missing_ok=True)
    print(f"reorder committed as generation {gen}", flush=True)
