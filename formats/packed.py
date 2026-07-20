"""vpack: lossless byte-plane compressed weight store.

bf16/fp16 tensors are split into two byte planes; the high plane (sign + exponent
+ top mantissa bits) zstd-compresses ~2x because trained-weight exponents cluster,
while the low plane is incompressible and is stored raw. Measured on Qwen2.5-32B:
1.34x smaller reads, decode 1.5 GB/s (~5x this disk's throughput), bit-exact.

Layout: <model>.vpack/ directory, one file per tensor (name with '/' -> '__'):
    8-byte little-endian header length, JSON header
    {"dtype": "BF16", "shape": [...], "hi_z": <zstd bytes>, "lo": <raw bytes>}
    followed by hi_z then lo as raw bytes.
Plus manifest.json mapping tensor name -> file, copied config/tokenizer files.
"""

from __future__ import annotations

import json
import struct
import time
from compression import zstd
from pathlib import Path

import numpy as np

import re as _re

PLANE_DTYPES = {"BF16", "F16"}
STREAM_THRESHOLD = 256 * 1024 * 1024  # tensors above this are packed in chunks

# Fused MoE expert tensors (gpt-oss style: [n_experts, ...]) are split into
# per-expert tensors at pack time so expert paging can fetch one expert's slice.
_FUSED_EXPERT_RE = _re.compile(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)_(blocks|scales|bias)$")
_FUSED_BF16_EXPERT_RE = _re.compile(
    r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$")
_SHORT = {"gate_up_proj": "gate_up", "down_proj": "down"}
CHUNK = 64 * 1024 * 1024  # even, so byte-plane parity is stable across chunks


def _write_small_tensor(out_path: Path, dtype: str, shape: list[int],
                        raw: bytes, level: int) -> int:
    """Write one materialized tensor in ordinary vpack form.

    Qwen3.5/3.6 stores every layer's 256 experts as two large BF16 tensors.
    Splitting them produces ~2-4 MB tensors, safely below STREAM_THRESHOLD;
    use the same byte-plane representation as the ordinary pack path and
    return the exact stored byte count for accounting.
    """
    if dtype in PLANE_DTYPES:
        hi_z = zstd.compress(raw[1::2], level=level)
        lo = raw[0::2]
        head = {"dtype": dtype, "shape": shape,
                "hi_z": len(hi_z), "lo": len(lo)}
        body = hi_z + lo
    else:
        head = {"dtype": dtype, "shape": shape, "raw": len(raw)}
        body = raw
    hj = json.dumps(head).encode()
    with open(out_path, "wb") as of:
        of.write(struct.pack("<Q", len(hj)) + hj + body)
    return len(hj) + 8 + len(body)


def _pack_tensor_streamed(src, abs_offset: int, nbytes: int, meta: dict, out_path: Path, level: int):
    """Chunked pack for large tensors: peak RAM ~2 chunks instead of ~2.5x tensor.
    Layout variant (head has "streamed"): lo plane raw first, then the zstd stream
    of the hi plane running to EOF (so its length needs no header field)."""
    head = {"dtype": meta["dtype"], "shape": meta["shape"], "lo": nbytes // 2, "streamed": True}
    hj = json.dumps(head).encode()
    with open(out_path, "wb") as of:
        of.write(struct.pack("<Q", len(hj)) + hj)
        for plane in (0, 1):  # 0 = lo (raw), 1 = hi (compressed stream)
            comp = zstd.ZstdCompressor(level=level) if plane else None
            src.seek(abs_offset)
            remaining = nbytes
            while remaining:
                chunk = src.read(min(CHUNK, remaining))
                remaining -= len(chunk)
                part = np.frombuffer(chunk, dtype=np.uint8)[plane::2].tobytes()
                of.write(comp.compress(part) if comp else part)
            if comp:
                of.write(comp.flush())


def _verify_tensor_streamed(src, abs_offset: int, nbytes: int, out_path: Path) -> bool:
    """Chunked bit-exact check of a streamed-format tensor against the original."""
    with open(out_path, "rb") as pf:
        n = struct.unpack("<Q", pf.read(8))[0]
        head = json.loads(pf.read(n))
        lo = pf.read(head["lo"])
        decomp = zstd.ZstdDecompressor()
        hi = bytearray()
        while True:
            block = pf.read(CHUNK)
            if not block:
                break
            hi += decomp.decompress(block)
    src.seek(abs_offset)
    pos = 0
    while pos < nbytes:
        chunk = src.read(min(CHUNK, nbytes - pos))
        arr = np.frombuffer(chunk, dtype=np.uint8)
        half = len(chunk) // 2
        if arr[0::2].tobytes() != lo[pos // 2 : pos // 2 + half]:
            return False
        if arr[1::2].tobytes() != bytes(hi[pos // 2 : pos // 2 + half]):
            return False
        pos += len(chunk)
    return True


def _iter_safetensors(path: Path):
    """Yield (name, meta, raw_bytes) without any framework."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            yield name, meta, f.read(b - a)


def pack_model(
    model_dir: str | Path,
    out_dir: str | Path | None = None,
    level: int = 1,  # F06 sweep: level 1 beats 3/6 on BOTH ratio and decode speed
    # for every bf16 tensor class measured on real GLM-5.2 weights (1.44-1.46x at
    # 2.2-2.6 GB/s vs 1.38-1.41x at ~1.6 GB/s) — see docs/benchmark_results.md
    delete_shards: bool = False,
    verify_shards: bool = False,
    progress=None,  # optional callable(shards_done: int, shards_total: int); additive,
    # coarse (per-shard, not per-tensor) so it's low-risk to thread through this
    # existing verified pipeline. Used by the HTTP server's auto-pack-on-repeat-
    # request feature (2026-07-13) for a percent/ETA estimate; None elsewhere.
) -> Path:
    """verify_shards=True round-trips every packed tensor against its source
    shard while retaining that shard. delete_shards=True implies the same full
    verification and removes a shard only after it passes.

    delete_shards=True: after each shard is fully packed AND every one of its
    tensors round-trips bit-exact, the raw shard is removed. Lets a model be packed
    in-place with only ~one shard of extra disk (needed when free space < 0.75x
    model size). The packed store is a complete lossless replacement."""
    model_dir = Path(model_dir)
    out = Path(out_dir) if out_dir else model_dir / "weights.vpack"
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    t0 = time.perf_counter()
    total_in = total_out = 0
    shards = sorted(model_dir.glob("*.safetensors"))

    # Resume support for delete_shards runs interrupted mid-model: tensors whose
    # shard is already gone must have verified .vt files — recover their manifest
    # entries from the HF index (partial .vt files of the current shard are simply
    # re-packed below).
    index_path = model_dir / "model.safetensors.index.json"
    if delete_shards and index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        existing = {s.name for s in shards}
        for name, shard_name in weight_map.items():
            if shard_name not in existing:
                fused_bf16 = _FUSED_BF16_EXPERT_RE.match(name)
                if fused_bf16:
                    prefix = f"{fused_bf16.group(1)}.".replace("/", "__")
                    projections = (
                        ("gate_proj.weight", "up_proj.weight")
                        if fused_bf16.group(2) == "gate_up_proj"
                        else ("down_proj.weight",)
                    )
                    recovered = []
                    for projection in projections:
                        recovered.extend(sorted(out.glob(
                            f"{prefix}*.{projection}.vt")))
                    assert recovered, (
                        f"missing unfused BF16 tensors for {name} "
                        f"(shard {shard_name} deleted)")
                    for p in recovered:
                        manifest[p.stem.replace("__", "/")] = p.name
                    continue
                fused = _FUSED_EXPERT_RE.match(name)
                if fused:
                    prefix = f"{fused.group(1)}.".replace("/", "__")
                    short = f"{_SHORT[fused.group(2)]}_{fused.group(3)}"
                    subs = sorted(out.glob(f"{prefix}*.{short}.vt"))
                    assert subs, f"missing unfused tensors for {name} (shard {shard_name} deleted)"
                    for p in subs:
                        manifest[p.stem.replace("__", "/")] = p.name
                    continue
                fname = name.replace("/", "__") + ".vt"
                assert (out / fname).exists(), f"missing packed tensor {name} (shard {shard_name} deleted)"
                manifest[name] = fname
        if manifest:
            print(f"  resuming: {len(manifest)} tensors already packed from deleted shards", flush=True)
    for shard in shards:
        shard_names: list[str] = []
        with open(shard, "rb") as sf:
            hn = struct.unpack("<Q", sf.read(8))[0]
            shard_header = json.loads(sf.read(hn))
        base = 8 + hn
        for name, meta in shard_header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            fused_bf16 = _FUSED_BF16_EXPERT_RE.match(name)
            if (fused_bf16 and len(meta["shape"]) == 3
                    and meta["shape"][0] > 1):
                if meta["dtype"] not in PLANE_DTYPES:
                    raise ValueError(
                        f"unsupported fused expert dtype {meta['dtype']} for {name}")
                n_e = meta["shape"][0]
                row = (b - a) // n_e
                with open(shard, "rb") as sf:
                    for e in range(n_e):
                        sf.seek(base + a + e * row)
                        raw_e = sf.read(row)
                        if fused_bf16.group(2) == "gate_up_proj":
                            if meta["shape"][1] % 2 or row % 2:
                                raise ValueError(
                                    f"odd fused gate/up split for {name}: "
                                    f"shape={meta['shape']}, bytes/expert={row}")
                            half = row // 2
                            inter = meta["shape"][1] // 2
                            parts = (
                                ("gate_proj.weight", [inter, meta["shape"][2]],
                                 raw_e[:half]),
                                ("up_proj.weight", [inter, meta["shape"][2]],
                                 raw_e[half:]),
                            )
                        else:
                            parts = ((
                                "down_proj.weight", list(meta["shape"][1:]), raw_e),)
                        for projection, sub_shape, sub_raw in parts:
                            sub = f"{fused_bf16.group(1)}.{e}.{projection}"
                            sub_fname = sub.replace("/", "__") + ".vt"
                            total_out += _write_small_tensor(
                                out / sub_fname, meta["dtype"], sub_shape,
                                sub_raw, level)
                            manifest[sub] = sub_fname
                shard_names.append(name)
                total_in += b - a
                continue
            fused = _FUSED_EXPERT_RE.match(name)
            if fused and len(meta["shape"]) >= 2 and meta["shape"][0] > 1:
                n_e = meta["shape"][0]
                row = (b - a) // n_e
                sub_shape = meta["shape"][1:]
                with open(shard, "rb") as sf:
                    for e in range(n_e):
                        sub = f"{fused.group(1)}.{e}.{_SHORT[fused.group(2)]}_{fused.group(3)}"
                        sf.seek(base + a + e * row)
                        raw_e = sf.read(row)
                        head = {"dtype": meta["dtype"], "shape": sub_shape, "raw": row}
                        hj = json.dumps(head).encode()
                        sub_fname = sub.replace("/", "__") + ".vt"
                        with open(out / sub_fname, "wb") as of:
                            of.write(struct.pack("<Q", len(hj)) + hj + raw_e)
                        manifest[sub] = sub_fname
                shard_names.append(name)
                total_in += b - a
                total_out += b - a
                continue
            fname = name.replace("/", "__") + ".vt"
            if meta["dtype"] in PLANE_DTYPES and b - a > STREAM_THRESHOLD:
                with open(shard, "rb") as sf:
                    _pack_tensor_streamed(sf, base + a, b - a, meta, out / fname, level)
                manifest[name] = fname
                shard_names.append(name)
                total_in += b - a
                total_out += (out / fname).stat().st_size
                continue
            with open(shard, "rb") as sf:
                sf.seek(base + a)
                raw = sf.read(b - a)
            if meta["dtype"] in PLANE_DTYPES:
                hi_z = zstd.compress(raw[1::2], level=level)
                lo = raw[0::2]
                head = {"dtype": meta["dtype"], "shape": meta["shape"],
                        "hi_z": len(hi_z), "lo": len(lo)}
                body = hi_z + lo
            else:
                head = {"dtype": meta["dtype"], "shape": meta["shape"], "raw": len(raw)}
                body = raw
            hj = json.dumps(head).encode()
            with open(out / fname, "wb") as f:
                f.write(struct.pack("<Q", len(hj)) + hj + body)
            manifest[name] = fname
            shard_names.append(name)
            total_in += len(raw)
            total_out += len(body) + len(hj) + 8
        if delete_shards or verify_shards:
            # Verify every tensor of this shard round-trips. Source retention is
            # independent: large checkpoints with enough disk can now produce a
            # proof-grade pack without making verification destructive.
            with open(shard, "rb") as sf:
                for name in shard_names:
                    meta = shard_header[name]
                    a, b = meta["data_offsets"]
                    fused_bf16 = _FUSED_BF16_EXPERT_RE.match(name)
                    if (fused_bf16 and len(meta["shape"]) == 3
                            and meta["shape"][0] > 1):
                        n_e = meta["shape"][0]
                        row = (b - a) // n_e
                        for e in range(n_e):
                            sf.seek(base + a + e * row)
                            expected = sf.read(row)
                            if fused_bf16.group(2) == "gate_up_proj":
                                half = row // 2
                                projections = (
                                    ("gate_proj.weight", expected[:half]),
                                    ("up_proj.weight", expected[half:]),
                                )
                            else:
                                projections = (("down_proj.weight", expected),)
                            for projection, expected_part in projections:
                                sub = (f"{fused_bf16.group(1)}.{e}."
                                       f"{projection}")
                                _head, raw2 = read_tensor_bytes(
                                    out, manifest[sub])
                                got = (raw2.tobytes()
                                       if isinstance(raw2, np.ndarray) else raw2)
                                assert got == expected_part, (
                                    f"round-trip mismatch: {sub}")
                        continue
                    fused = _FUSED_EXPERT_RE.match(name)
                    if fused and len(meta["shape"]) >= 2 and meta["shape"][0] > 1:
                        n_e = meta["shape"][0]
                        row = (b - a) // n_e
                        for e in range(n_e):
                            sub = f"{fused.group(1)}.{e}.{_SHORT[fused.group(2)]}_{fused.group(3)}"
                            sf.seek(base + a + e * row)
                            head, raw2 = read_tensor_bytes(out, manifest[sub])
                            got = raw2.tobytes() if isinstance(raw2, np.ndarray) else raw2
                            assert got == sf.read(row), f"round-trip mismatch: {sub}"
                        continue
                    if meta["dtype"] in PLANE_DTYPES and b - a > STREAM_THRESHOLD:
                        assert _verify_tensor_streamed(sf, base + a, b - a, out / manifest[name]), \
                            f"round-trip mismatch (streamed): {name} in {shard.name}"
                        continue
                    sf.seek(base + a)
                    raw = sf.read(b - a)
                    head, raw2 = read_tensor_bytes(out, manifest[name])
                    got = raw2.tobytes() if isinstance(raw2, np.ndarray) else raw2
                    assert got == raw, f"round-trip mismatch: {name} in {shard.name}"
            if delete_shards:
                shard.unlink()
                print(f"  packed+verified+deleted {shard.name}", flush=True)
            else:
                print(f"  packed+verified {shard.name}", flush=True)
        if progress:
            progress(shards.index(shard) + 1, len(shards))
    (out / "manifest.json").write_text(json.dumps(manifest))
    dt = time.perf_counter() - t0
    print(f"packed {total_in / 1e9:.1f}GB -> {total_out / 1e9:.1f}GB "
          f"({total_in / total_out:.3f}x) in {dt / 60:.1f}min")
    return out


def read_tensor_bytes(vpack_dir: Path, fname: str) -> tuple[dict, bytes | np.ndarray]:
    """Read + decode one tensor to its original raw bytes (bit-exact). Returns a
    numpy uint8 array on the plane-decode path to avoid a full extra copy —
    np.frombuffer/to_mx accept both."""
    with open(vpack_dir / fname, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        if "raw" in head:
            return head, f.read(head["raw"])
        if head.get("streamed"):  # lo plane first, hi zstd stream to EOF
            lo = f.read(head["lo"])
            hi = zstd.decompress(f.read())
        else:
            hi = zstd.decompress(f.read(head["hi_z"]))
            lo = f.read(head["lo"])
    raw = np.empty(len(hi) * 2, dtype=np.uint8)
    raw[0::2] = np.frombuffer(lo, dtype=np.uint8)
    raw[1::2] = np.frombuffer(hi, dtype=np.uint8)
    return head, raw


_MX_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "I64": "int64"}


def to_mx(head: dict, raw: "bytes | np.ndarray"):
    """Materialize decoded bytes as an mx.array (single copy into unified memory)."""
    import mlx.core as mx

    buf = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.uint8)
    dt = head["dtype"]
    if dt in ("BF16", "F16"):
        u16 = buf.view(np.uint16).reshape(head["shape"])
        target = mx.bfloat16 if dt == "BF16" else mx.float16
        return mx.array(u16).view(target)
    npdt = {"F32": np.float32, "I64": np.int64, "U8": np.uint8, "I32": np.int32,
            "U32": np.uint32, "I8": np.int8}[dt]
    return mx.array(buf.view(npdt).reshape(head["shape"]))
