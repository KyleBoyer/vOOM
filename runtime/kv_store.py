"""Durable prompt-KV prefix persistence.

F37 v6 uses the same v3 immutable, checksummed parent-hashed delta journal as the
hot prompt cache. Extending a known endpoint writes only new KV positions plus a
small checkpoint payload; readers verify SHA-256 before MLX loads, hold file
leases against concurrent GC, and fall back to an older valid generation after
corruption. Model, tokenizer, runtime, arithmetic, RoPE, compressed-MLA, and DSA
state are fingerprinted. The v5 full-snapshot class remains below only as format
history; ``PromptKVStore`` is the journal-backed implementation used at runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx

from .kv_cache import KVCache


# F37 durability fix (2026-07-13): the fingerprint previously covered
# checkpoint/tokenizer/quant identity but NOT the runtime code that computes
# the cached KV values — a bug fix to attention/MoE/DSA math between a save
# and a later load would silently serve stale-code-computed KV as if it were
# still correct. Mirrors experiments/speculative_decode.py's
# source_fingerprint() for the same reason: any change to these files changes
# what a cached entry MEANS, so it must invalidate all prior entries.
def _runtime_fingerprint(root: Path | None = None) -> bytes:
    """Hash every runtime source module; optional root enables a no-mutation test."""
    root = root or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.digest()


def model_fingerprint(model_dir: str | Path, compressed_mla: bool,
                      dsa_elided: bool = False, quant: str = "bf16",
                      arithmetic: str = "") -> str:
    """Identity of everything that can change what a cached prefix MEANS:
    checkpoint (config + weight index = shard layout/sizes as a revision
    proxy), tokenizer, ARITHMETIC identity (quantization mode — a fast-mode
    q4 engine must never share entries with a lossless one), state flags,
    the runtime code that computes the cached values, and a runtime
    state-format version bumped on layout changes."""
    d = Path(model_dir)
    h = hashlib.sha256()
    h.update(b"kvstore-v6")  # v6: immutable checksummed delta-journal generations
    h.update(d.name.encode())
    h.update((d / "config.json").read_bytes())
    for extra in (
        "model.safetensors.index.json", "tokenizer.json",
        "weights.vpack2.index.json", "vpack2.CURRENT",
    ):
        p = d / extra
        if p.exists():  # checkpoint-revision + tokenizer identity
            h.update(hashlib.sha256(p.read_bytes()).digest())
    # Avoid a 1.49-TB content scan at engine-up, but distinguish single-file and
    # packed checkpoints that previously shared config/index identity. Artifact
    # SHA manifests remain the stronger proof; size+mtime is a fail-closed local
    # reuse guard, not a cryptographic weight attestation.
    for p in sorted(d.iterdir()):
        if p.is_file() and (
            p.suffix in {".safetensors", ".vpack", ".vpack2", ".vt"}
            or ".vpack2" in p.name
        ):
            st = p.stat()
            h.update(p.name.encode())
            h.update(str(st.st_size).encode())
            h.update(str(st.st_mtime_ns).encode())
    h.update(quant.encode())
    h.update(arithmetic.encode())
    h.update(b"cmla1" if compressed_mla else b"cmla0")
    if dsa_elided:  # F43: bounded-mode saves carry no indexer k-cache — an
        h.update(b"dsae1")  # unbounded run must never restore them
    h.update(_runtime_fingerprint())
    return h.hexdigest()


def _key(fingerprint: str, tokens: list[int]) -> str:
    return hashlib.sha256((fingerprint + json.dumps(tokens)).encode()).hexdigest()


class _LegacyPromptKVStore:
    def __init__(self, dir: str | Path, fingerprint: str,
                 max_bytes: int = 2_000_000_000):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp = fingerprint
        self.max_bytes = max_bytes  # LRU eviction budget for the whole store

    def _evict(self):
        """Keep the store under max_bytes: drop least-recently-USED entries
        (loads touch mtimes, so recency = usage). Also sweeps orphaned
        .safetensors older than an hour (torn saves — JSON is the commit
        record, so an orphan is never referenced), and F37-durability-fix
        *.tmp.safetensors / *.json.tmp leftovers from a crash between
        writing a tmp payload and its rename (neither glob pattern below
        matches a .tmp suffix, so these would otherwise accumulate forever)."""
        import os
        import time as _time

        for pattern in ("*.tmp.safetensors", "*.json.tmp"):
            for tmp in self.dir.glob(pattern):
                if _time.time() - tmp.stat().st_mtime > 3600:
                    tmp.unlink(missing_ok=True)

        entries, total = [], 0
        for j in self.dir.glob("*.json"):
            st = j.with_suffix(".safetensors")
            size = j.stat().st_size + (st.stat().st_size if st.exists() else 0)
            mtime = max(j.stat().st_mtime,
                        st.stat().st_mtime if st.exists() else 0)
            entries.append((mtime, size, j, st))
            total += size
        for st in self.dir.glob("*.safetensors"):
            if not st.with_suffix(".json").exists() and \
                    _time.time() - st.stat().st_mtime > 3600:
                st.unlink(missing_ok=True)
        entries.sort()  # oldest access first
        for _, size, j, st in entries:
            if total <= self.max_bytes:
                break
            j.unlink(missing_ok=True)
            st.unlink(missing_ok=True)
            total -= size

    def save(self, tokens: list[int], kv: KVCache, logits: mx.array, dsa=None):
        """Snapshot prefill state: KV (naive or compressed-MLA), last-position
        logits, and DSA indexer k-caches when present."""
        key = _key(self.fp, tokens)
        arrays = {"logits": logits}
        end = kv.offset
        for i in range(len(kv.keys)):
            if kv.keys[i] is not None:
                if kv.compressed_mla:
                    arrays[f"k{i}"] = kv.keys[i][:, :end, :]
                else:
                    arrays[f"k{i}"] = kv.keys[i][:, :, :end, :]
                if not kv.compressed_mla and kv.values[i] is not None:
                    arrays[f"v{i}"] = kv.values[i][:, :, :end, :]
        if dsa is not None:
            for layer, karr in dsa.k_idx.items():
                arrays[f"dsa{layer}"] = karr
        # Never write a payload that cannot survive this store's own budget.
        # Previously a >2GB long-context snapshot was fully written and then
        # immediately deleted by _evict(), wasting I/O and providing no resume
        # point. Safetensors metadata is small; add a conservative 1 MiB margin.
        estimated_bytes = sum(int(a.nbytes) for a in arrays.values()) + 1_048_576
        if estimated_bytes > self.max_bytes:
            print(
                f"[prompt-kv] skip {estimated_bytes / 1e9:.2f}GB snapshot: "
                f"exceeds {self.max_bytes / 1e9:.2f}GB store budget",
                flush=True,
            )
            return False
        # Atomic commit: the JSON is the commit record. Write tensors first,
        # then publish the JSON via tmp+rename — a crash at any point leaves
        # either a complete entry or an unreferenced orphan, never a torn one.
        #
        # F37 durability fix (2026-07-13): the .safetensors payload used to be
        # written DIRECTLY to its final name. Re-saving an existing key (a
        # repeated/overlapping prompt, or — vanishingly rarely — a truncated-
        # hash collision) then overwrote a valid entry in place; a crash
        # mid-write left a torn .safetensors file with an intact, now-lying
        # JSON commit record still pointing at it, and the next load would
        # either error opening a corrupt file or worse. Now goes through the
        # same tmp+fsync+rename discipline as the JSON (and as F31's
        # vpack2.CURRENT.tmp fix): a crash at any point during the payload
        # write leaves the OLD entry (if any) fully intact.
        import os

        # NOTE: mx.save_safetensors force-appends ".safetensors" to any path
        # that doesn't already end with it (confirmed: saving to
        # "x.safetensors.tmp" actually creates "x.safetensors.tmp.safetensors")
        # — so the tmp name must ALREADY end in ".safetensors", or the
        # subsequent os.open/os.replace calls target a file that was never
        # written (caught by a fixture test immediately after this fix).
        tmp_st = self.dir / f"{key}.tmp.safetensors"
        mx.save_safetensors(str(tmp_st), arrays)
        stfd = os.open(tmp_st, os.O_RDONLY)
        os.fsync(stfd)
        os.close(stfd)
        os.replace(tmp_st, self.dir / f"{key}.safetensors")

        tmp = self.dir / f"{key}.json.tmp"
        tmp.write_text(json.dumps(
            {"tokens": tokens, "fp": self.fp, "compressed_mla": kv.compressed_mla}))
        jfd = os.open(tmp, os.O_RDONLY)
        os.fsync(jfd)
        os.close(jfd)
        os.replace(tmp, self.dir / f"{key}.json")

        dfd = os.open(self.dir, os.O_RDONLY)  # persist both renames
        os.fsync(dfd)
        os.close(dfd)
        self._evict()
        return True

    def load_longest_prefix(self, tokens: list[int], num_layers: int, dsa=None
                            ) -> tuple[KVCache | None, int, mx.array | None]:
        """Return (kv, matched_len, logits_if_exact). Only entries with this
        store's fingerprint match. DSA k-caches are restored into `dsa`."""
        candidates = []
        for meta_path in self.dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # torn entry: never take the whole scan down
            if not isinstance(meta, dict) or meta.get("fp") != self.fp:
                continue  # legacy/foreign formats are skipped, not fatal
            stored = meta.get("tokens")
            if not isinstance(stored, list):
                continue
            n = len(stored)
            if n <= len(tokens) and stored == tokens[:n]:
                candidates.append((n, meta))
        for n, best in sorted(candidates, key=lambda item: item[0], reverse=True):
            key = _key(self.fp, best["tokens"])
            payload = self.dir / f"{key}.safetensors"
            if not payload.exists():
                continue
            try:
                lazy = mx.load(str(payload))
                if "logits" not in lazy:
                    raise KeyError("missing logits")
                mx.eval([a for a in lazy.values()])
            except Exception as e:
                print(f"[prompt-kv] skip corrupt/unreadable {payload.name}: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue

            kv = KVCache(num_layers)
            kv.compressed_mla = bool(best.get("compressed_mla"))
            dsa_arrays = {}
            for name, arr in lazy.items():
                if name.startswith("k") and name[1:].isdigit():
                    kv.keys[int(name[1:])] = arr
                elif name.startswith("v") and name[1:].isdigit():
                    kv.values[int(name[1:])] = arr
                elif name.startswith("dsa") and dsa is not None:
                    dsa_arrays[int(name[3:])] = arr
            if dsa is not None:
                dsa.k_idx.update(dsa_arrays)

            import os as _os
            for suffix in (".json", ".safetensors"):
                try:
                    _os.utime(self.dir / f"{key}{suffix}")
                except OSError:
                    pass
            logits = lazy["logits"] if n == len(tokens) else None
            return kv, n, logits
        return None, 0, None


class PromptKVStore:
    """F37 v6: durable immutable generations over an append-only KV journal.

    Unlike the legacy full-snapshot implementation above, extending a known
    prefix writes only newly appended KV positions. Tensor payload hashes are
    part of immutable segment/checkpoint identities and are verified before
    MLX loads them. Reader leases make concurrent GC safe.
    """

    def __init__(self, dir: str | Path, fingerprint: str,
                 max_bytes: int = 2_000_000_000, *, chunk_size: int = 512,
                 config=None, require_dsa: bool = False):
        from .hot_kv_persist import HotPromptKVPersistence

        self.dir = Path(dir)
        self.fp = fingerprint
        self.max_bytes = int(max_bytes)
        self.chunk_size = max(1, int(chunk_size))
        self.journal = HotPromptKVPersistence(
            self.dir,
            fingerprint + "|f37-delta-v1",
            self.chunk_size,
            max_checkpoints=0,
            max_bytes=self.max_bytes,
            config=config,
            require_dsa=require_dsa,
        )
        self._sweep_legacy_v5_once()

    def _sweep_legacy_v5_once(self) -> None:
        """Remove unusable full-snapshot generations after the v6 format bump.

        Legacy entries use bare ``<64hex>.json/.safetensors`` names. Their v5
        fingerprint cannot match this store, and leaving them outside the v6
        journal's byte accounting permanently consumes the old budget (1.4 GB
        in the measured local store). A durable marker makes the directory scan
        and fsync a one-time upgrade operation.
        """
        from .hot_kv_persist import _atomic_json, _fsync_dir

        marker = self.dir / ".f37-v6-legacy-swept.json"
        if marker.exists():
            return
        removed = 0
        with self.journal._locked(exclusive=True):
            if marker.exists():
                return
            legacy_ids = set()
            for pattern, suffix in (("*.json", ".json"),
                                    ("*.safetensors", ".safetensors")):
                for path in self.dir.glob(pattern):
                    stem = path.name[:-len(suffix)]
                    if (len(stem) == 64
                            and all(character in "0123456789abcdef"
                                    for character in stem)):
                        legacy_ids.add(stem)
            for legacy_id in legacy_ids:
                for suffix in (".json", ".safetensors"):
                    path = self.dir / f"{legacy_id}{suffix}"
                    if path.exists():
                        path.unlink()
                        removed += 1
            _atomic_json(marker, {
                "format": "f37-v6-legacy-sweep-v1",
                "removed_files": removed,
            })
            _fsync_dir(self.dir)
        if removed:
            print(f"[prompt-kv] removed {removed} obsolete v5 cache files",
                  flush=True)

    @staticmethod
    def _state_bytes(kv: KVCache, logits: mx.array, dsa=None) -> int:
        total = int(logits.nbytes)
        for values in (kv.keys, kv.values):
            total += sum(int(value.nbytes) for value in values
                         if value is not None)
        if dsa is not None:
            total += sum(int(value.nbytes) for value in dsa.k_idx.values())
        return total

    def save(self, tokens: list[int], kv: KVCache, logits: mx.array, dsa=None):
        if not tokens or kv.offset != len(tokens):
            if tokens:
                print(
                    f"[prompt-kv] skip inconsistent journal endpoint: "
                    f"tokens={len(tokens)} kv.offset={kv.offset}",
                    flush=True,
                )
            return False
        # A loadable checkpoint must fit as one reachable chain. Delta writes
        # remove O(n^2) cumulative I/O, but they cannot make an undersized store
        # retain a state larger than its own hard budget.
        estimated = self._state_bytes(kv, logits, dsa) + 1_048_576
        if self.max_bytes and estimated > self.max_bytes:
            print(
                f"[prompt-kv] skip {estimated / 1e9:.2f}GB journal endpoint: "
                f"exceeds {self.max_bytes / 1e9:.2f}GB store budget",
                flush=True,
            )
            return False

        match = self.journal.find_best_match(tokens, self.chunk_size)
        parent_chain: tuple[str, ...] = ()
        parent_covered = 0
        if match is not None:
            parent_chain = tuple(match["chain"][:match["n_segments"]])
            parent_covered = int(match["matched"])

        previous_dsa = getattr(kv, "dsa", None)
        if dsa is not None:
            kv.dsa = dsa
        try:
            chain = self.journal.save(
                parent_chain=parent_chain,
                parent_covered=parent_covered,
                tokens=tokens,
                kv=kv,
                logits=logits,
                prompt_logits=logits,
                prompt_length=len(tokens),
                reusable_prefix=len(tokens),
            )
        finally:
            kv.dsa = previous_dsa
        self.journal.gc()
        return bool(chain)

    def load_longest_prefix(self, tokens: list[int], num_layers: int, dsa=None
                            ) -> tuple[KVCache | None, int, mx.array | None]:
        match = self.journal.find_best_match(tokens, self.chunk_size)
        if match is None:
            return None, 0, None
        loaded = self.journal.load_matched_chain(match, num_layers)
        if loaded is None:
            return None, 0, None
        loaded_tokens, kv, exact_logits = loaded
        if dsa is not None:
            loaded_dsa = getattr(kv, "dsa", None)
            if loaded_dsa is None:
                return None, 0, None
            dsa.k_idx.update(loaded_dsa.k_idx)
            kv.dsa = dsa
        return kv, len(loaded_tokens), exact_logits
