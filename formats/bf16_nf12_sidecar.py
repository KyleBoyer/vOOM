"""Exact block-decodable 12-bit storage for BF16 transformer weights.

BF16 uses one sign bit, eight exponent bits, and seven mantissa bits. Trained
weights typically use a narrow exponent range. For each tensor this format
stores:

* sign + mantissa + the low exponent nibble in a fixed 12-bit stream;
* the modal high exponent nibble once per 256-value block header;
* rare non-modal values as ``(uint8 local_index, uint16 exact_bits)`` patches.

The decoder reconstructs the original uint16 bit pattern; this is compression,
not quantization. Tensors that would not shrink are omitted and retain their
ordinary checkpoint path.

This module is MLX-free. Metal decoding lives in
``runtime.bf16_nf12_sidecar``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import shutil
import struct
import time
import uuid
from pathlib import Path

import numpy as np

from .kimi_k3_scale_sidecar import (
    CURRENT,
    _SafeTensorReader,
    _external_root_guard,
    _fsync_dir,
    _sha256_file,
    _source_fingerprint,
    _write_fsync,
)

SCHEMA = "voom.bf16-nf12-sidecar.v2"
BLOCK_VALUES = 256
CODE_BYTES_PER_BLOCK = BLOCK_VALUES * 12 // 8
PATCH_BYTES = 3
BLOCK_HEADER = struct.Struct("<IIHHBBH")
BLOCK_HEADER_BYTES = BLOCK_HEADER.size
_LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")
TENSOR_ORDERS = ("lexical", "kimi_k3_attention_mlp")


def _canonical_name(name: str) -> str:
    return name.removeprefix("language_model.")


def _safetensors_prefix(payload_bytes: int) -> bytes:
    header = json.dumps(
        {
            "encoded": {
                "dtype": "U8",
                "shape": [int(payload_bytes)],
                "data_offsets": [0, int(payload_bytes)],
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


def _pack_codes12(values: np.ndarray, blocks: int) -> bytes:
    padded_count = blocks * BLOCK_VALUES
    if values.size < padded_count:
        padded = np.zeros(padded_count, dtype=np.uint16)
        padded[: values.size] = values
    else:
        padded = values
    codes = (
        (padded & np.uint16(0x07FF))
        | ((padded >> np.uint16(4)) & np.uint16(0x0800))
    )
    pairs = codes.reshape(-1, 2)
    packed = np.empty((pairs.shape[0], 3), dtype=np.uint8)
    packed[:, 0] = (pairs[:, 0] & 0xFF).astype(np.uint8)
    packed[:, 1] = (
        ((pairs[:, 0] >> 8) & 0x0F)
        | ((pairs[:, 1] & 0x0F) << 4)
    ).astype(np.uint8)
    packed[:, 2] = (pairs[:, 1] >> 4).astype(np.uint8)
    return packed.tobytes()


def pack_tensor(
    raw: bytes,
    *,
    output_offset: int = 0,
    patch_base: int = 0,
) -> tuple[bytes, bytes, bytes, dict]:
    """Return preliminary headers, fixed codes, patches, and exact metadata.

    Header patch offsets are relative to the layer patch stream plus
    ``patch_base``. The builder rebases them once final file section offsets
    are known.
    """
    if len(raw) % 2:
        raise ValueError("BF16 tensor byte count must be even")
    values = np.frombuffer(raw, dtype="<u2")
    if not values.size:
        raise ValueError("cannot encode an empty BF16 tensor")
    high = ((values >> 11) & 0x0F).astype(np.uint8)
    mode = int(np.bincount(high, minlength=16).argmax())
    exception_indices = np.flatnonzero(high != mode)
    blocks = math.ceil(int(values.size) / BLOCK_VALUES)
    charged_bytes = (
        blocks * (BLOCK_HEADER_BYTES + CODE_BYTES_PER_BLOCK)
        + PATCH_BYTES * int(exception_indices.size)
    )
    if charged_bytes >= len(raw):
        return b"", b"", b"", {
            "selected": False,
            "raw_bytes": len(raw),
            "encoded_bytes": len(raw),
            "ratio": 1.0,
            "mode": mode,
            "exception_count": int(exception_indices.size),
            "block_count": blocks,
        }

    block_ids = exception_indices // BLOCK_VALUES
    counts = np.bincount(block_ids, minlength=blocks).astype(np.uint16)
    patch_starts = np.empty(blocks, dtype=np.uint64)
    patch_starts[0] = int(patch_base)
    if blocks > 1:
        patch_starts[1:] = (
            int(patch_base)
            + np.cumsum(counts[:-1], dtype=np.uint64) * PATCH_BYTES
        )
    valid = np.full(blocks, BLOCK_VALUES, dtype=np.uint16)
    tail = int(values.size) % BLOCK_VALUES
    if tail:
        valid[-1] = tail
    output = (
        int(output_offset)
        + np.arange(blocks, dtype=np.uint64) * BLOCK_VALUES
    )
    if (
        patch_starts.max(initial=0) > np.iinfo(np.uint32).max
        or output.max(initial=0) > np.iinfo(np.uint32).max
    ):
        raise OverflowError("NF12 layer offsets exceed uint32 format")

    headers = bytearray(blocks * BLOCK_HEADER_BYTES)
    for block in range(blocks):
        BLOCK_HEADER.pack_into(
            headers,
            block * BLOCK_HEADER_BYTES,
            int(patch_starts[block]),
            int(output[block]),
            int(valid[block]),
            int(counts[block]),
            mode,
            0,
            0,
        )

    patches = np.empty((exception_indices.size, PATCH_BYTES), dtype=np.uint8)
    if exception_indices.size:
        exact = values[exception_indices]
        patches[:, 0] = (exception_indices % BLOCK_VALUES).astype(np.uint8)
        patches[:, 1] = (exact & 0xFF).astype(np.uint8)
        patches[:, 2] = (exact >> 8).astype(np.uint8)
    codes = _pack_codes12(values, blocks)
    return bytes(headers), codes, patches.tobytes(), {
        "selected": True,
        "raw_bytes": len(raw),
        "encoded_bytes": charged_bytes,
        "ratio": len(raw) / charged_bytes,
        "mode": mode,
        "exception_count": int(exception_indices.size),
        "block_count": blocks,
        "value_count": int(values.size),
    }


def unpack_tensor(
    headers: bytes, codes: bytes, patches: bytes, *, value_count: int
) -> bytes:
    """Pure-NumPy format oracle for tests and builder verification."""
    if len(headers) % BLOCK_HEADER_BYTES:
        raise ValueError("misaligned NF12 header bytes")
    blocks = len(headers) // BLOCK_HEADER_BYTES
    if len(codes) != blocks * CODE_BYTES_PER_BLOCK:
        raise ValueError("wrong NF12 code-stream length")
    packed = np.frombuffer(codes, dtype=np.uint8).reshape(-1, 3)
    decoded_codes = np.empty(packed.shape[0] * 2, dtype=np.uint16)
    decoded_codes[0::2] = (
        packed[:, 0].astype(np.uint16)
        | ((packed[:, 1].astype(np.uint16) & 0x0F) << 8)
    )
    decoded_codes[1::2] = (
        (packed[:, 1].astype(np.uint16) >> 4)
        | (packed[:, 2].astype(np.uint16) << 4)
    )
    output = np.empty(blocks * BLOCK_VALUES, dtype=np.uint16)
    patch_view = memoryview(patches)
    for block in range(blocks):
        patch_offset, out, valid, count, mode, flags, reserved = (
            BLOCK_HEADER.unpack_from(headers, block * BLOCK_HEADER_BYTES)
        )
        if flags or reserved or out != block * BLOCK_VALUES:
            raise ValueError("invalid standalone NF12 block header")
        start = block * BLOCK_VALUES
        code = decoded_codes[start : start + valid]
        restored = (
            (code & 0x07FF)
            | ((code & 0x0800) << 4)
            | (np.uint16(mode) << np.uint16(11))
        )
        output[start : start + valid] = restored
        for patch in range(count):
            at = patch_offset + patch * PATCH_BYTES
            local = patch_view[at]
            exact = patch_view[at + 1] | (patch_view[at + 2] << 8)
            output[start + local] = exact
    return output[:value_count].astype("<u2", copy=False).tobytes()


def _discover_layers(reader: _SafeTensorReader) -> dict[int, list[str]]:
    layers: dict[int, list[str]] = {}
    for name in reader.weight_map:
        match = _LAYER_RE.search(name)
        if match is None or ".experts." in name:
            continue
        if reader.metadata(name)["dtype"] == "BF16":
            layers.setdefault(int(match.group(1)), []).append(name)
    return layers


def _tensor_order_key(name: str, tensor_order: str) -> tuple[int, str]:
    canonical = _canonical_name(name)
    if tensor_order == "lexical":
        return 0, canonical
    match = _LAYER_RE.search(canonical)
    if match is None:
        raise ValueError(f"cannot lifetime-order non-layer tensor {name}")
    suffix = canonical[match.end() :]
    attention = (
        suffix.startswith("self_attn.")
        or suffix == "input_layernorm.weight"
        or suffix.startswith("self_attention_res_")
    )
    return (0 if attention else 1), canonical


def build_nf12_sidecar(
    model_dir: str | Path,
    output_root: str | Path,
    *,
    layers: list[int] | tuple[int, ...] | None = None,
    tensor_order: str = "lexical",
    enforce_external: bool = True,
    min_free_after_bytes: int = 1_000_000_000,
) -> dict:
    """Build and atomically publish an immutable partial/full generation."""
    model_dir = Path(model_dir).resolve()
    output_root = Path(output_root).expanduser().resolve()
    if enforce_external:
        _external_root_guard(output_root)
    if tensor_order not in TENSOR_ORDERS:
        raise ValueError(
            f"tensor_order must be one of {TENSOR_ORDERS}, got "
            f"{tensor_order!r}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    source = _source_fingerprint(model_dir)

    with _SafeTensorReader(model_dir) as reader:
        discovered = _discover_layers(reader)
        selected_layers = (
            sorted(discovered)
            if layers is None
            else sorted(set(int(layer) for layer in layers))
        )
        if not selected_layers:
            raise ValueError("no BF16 layers selected")
        missing = [layer for layer in selected_layers if layer not in discovered]
        if missing:
            raise ValueError(f"no BF16 tensors for layers {missing}")
        worst_case = sum(
            int(np.prod(reader.metadata(name)["shape"], dtype=np.int64)) * 2
            for layer in selected_layers
            for name in discovered[layer]
        )
        free = shutil.disk_usage(output_root).free
        if free - worst_case < int(min_free_after_bytes):
            raise OSError(
                f"NF12 worst-case {worst_case} bytes would leave "
                f"{free - worst_case} bytes free, below required "
                f"{min_free_after_bytes}"
            )

        generation = (
            f"gen-{int(time.time())}-"
            f"{source['fingerprint'][:12]}-{uuid.uuid4().hex[:8]}"
        )
        staging = output_root / f".{generation}.tmp"
        committed = output_root / generation
        staging.mkdir()
        layer_manifest: dict[str, dict] = {}
        total_raw = total_encoded = 0

        for layer in selected_layers:
            header_tmp = staging / f".layer-{layer:03d}.headers"
            code_tmp = staging / f".layer-{layer:03d}.codes"
            patch_tmp = staging / f".layer-{layer:03d}.patches"
            tensor_specs = []
            output_count = block_count = patch_bytes = 0
            raw_layer_bytes = selected_raw_bytes = 0

            with (
                header_tmp.open("wb") as header_out,
                code_tmp.open("wb") as code_out,
                patch_tmp.open("wb") as patch_out,
            ):
                for name in sorted(
                    discovered[layer],
                    key=lambda value: _tensor_order_key(
                        value, tensor_order
                    ),
                ):
                    metadata = reader.metadata(name)
                    raw = reader.read(name)
                    expected = (
                        int(np.prod(metadata["shape"], dtype=np.int64)) * 2
                    )
                    if len(raw) != expected:
                        raise ValueError(
                            f"{name}: BF16 byte count {len(raw)} != {expected}"
                        )
                    raw_layer_bytes += len(raw)
                    headers, codes, patches, report = pack_tensor(
                        raw,
                        output_offset=output_count,
                        patch_base=patch_bytes,
                    )
                    if not report["selected"]:
                        continue
                    # Verify exactness before the source tensor can be released.
                    local_headers = bytearray(headers)
                    for offset in range(0, len(local_headers), BLOCK_HEADER_BYTES):
                        fields = list(BLOCK_HEADER.unpack_from(local_headers, offset))
                        fields[0] -= patch_bytes
                        fields[1] -= output_count
                        BLOCK_HEADER.pack_into(local_headers, offset, *fields)
                    if unpack_tensor(
                        bytes(local_headers),
                        codes,
                        patches,
                        value_count=report["value_count"],
                    ) != raw:
                        raise AssertionError(f"{name}: NF12 changed BF16 bits")
                    header_out.write(headers)
                    code_out.write(codes)
                    patch_out.write(patches)
                    tensor_specs.append(
                        {
                            "physical_name": name,
                            "name": _canonical_name(name),
                            "shape": [int(value) for value in metadata["shape"]],
                            "output_offset": output_count,
                            "value_count": report["value_count"],
                            "block_start": block_count,
                            "block_count": report["block_count"],
                            "raw_bytes": report["raw_bytes"],
                            "encoded_bytes": report["encoded_bytes"],
                            "ratio": report["ratio"],
                            "mode": report["mode"],
                            "exception_count": report["exception_count"],
                        }
                    )
                    output_count += report["value_count"]
                    block_count += report["block_count"]
                    patch_bytes += len(patches)
                    selected_raw_bytes += len(raw)

            if not tensor_specs:
                raise ValueError(f"layer {layer}: no BF16 tensor shrank")
            header_bytes = header_tmp.stat().st_size
            code_bytes = code_tmp.stat().st_size
            patches_base = header_bytes + code_bytes
            payload_bytes = header_bytes + code_bytes + patch_bytes
            path = staging / f"layer-{layer:03d}.safetensors"
            with path.open("xb") as final:
                final.write(_safetensors_prefix(payload_bytes))
                with header_tmp.open("rb") as source_headers:
                    while raw_headers := source_headers.read(
                        BLOCK_HEADER_BYTES * 65_536
                    ):
                        adjusted = bytearray(raw_headers)
                        for offset in range(
                            0, len(adjusted), BLOCK_HEADER_BYTES
                        ):
                            fields = list(
                                BLOCK_HEADER.unpack_from(adjusted, offset)
                            )
                            fields[0] += patches_base
                            BLOCK_HEADER.pack_into(adjusted, offset, *fields)
                        final.write(adjusted)
                for temporary in (code_tmp, patch_tmp):
                    with temporary.open("rb") as source_part:
                        shutil.copyfileobj(source_part, final, 8 * 1024 * 1024)
                final.flush()
                os.fsync(final.fileno())
            for temporary in (header_tmp, code_tmp, patch_tmp):
                temporary.unlink()

            encoded_bytes = payload_bytes
            storage_file_bytes = path.stat().st_size
            layer_manifest[str(layer)] = {
                "file": path.name,
                "file_bytes": encoded_bytes,
                "storage_file_bytes": storage_file_bytes,
                "file_sha256": _sha256_file(path),
                "block_values": BLOCK_VALUES,
                "block_header_bytes": BLOCK_HEADER_BYTES,
                "code_bytes_per_block": CODE_BYTES_PER_BLOCK,
                "patch_bytes": PATCH_BYTES,
                "block_count": block_count,
                "header_bytes": header_bytes,
                "code_bytes": code_bytes,
                "patch_stream_bytes": patch_bytes,
                "output_value_count": output_count,
                "selected_raw_bytes": selected_raw_bytes,
                "all_bf16_raw_bytes": raw_layer_bytes,
                "encoded_bytes": encoded_bytes,
                "ratio_selected_raw_over_encoded": (
                    selected_raw_bytes / encoded_bytes
                ),
                "saved_bytes": selected_raw_bytes - encoded_bytes,
                "tensors": tensor_specs,
            }
            total_raw += selected_raw_bytes
            total_encoded += encoded_bytes

        manifest = {
            "schema": SCHEMA,
            "source": source,
            "block_values": BLOCK_VALUES,
            "block_header_bytes": BLOCK_HEADER_BYTES,
            "code_bytes_per_block": CODE_BYTES_PER_BLOCK,
            "patch_bytes": PATCH_BYTES,
            "tensor_order": tensor_order,
            "layers": layer_manifest,
            "total_selected_raw_bytes": total_raw,
            "total_encoded_bytes": total_encoded,
            "saved_bytes": total_raw - total_encoded,
            "ratio_raw_over_encoded": total_raw / total_encoded,
        }
        _write_fsync(
            staging / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
        )
        _fsync_dir(staging)
        os.replace(staging, committed)
        _fsync_dir(output_root)
        pointer_tmp = output_root / f".{CURRENT}.tmp-{uuid.uuid4().hex}"
        _write_fsync(pointer_tmp, (generation + "\n").encode())
        os.replace(pointer_tmp, output_root / CURRENT)
        _fsync_dir(output_root)

    return {
        "generation": generation,
        "root": str(output_root),
        "layers": selected_layers,
        "total_selected_raw_bytes": total_raw,
        "total_encoded_bytes": total_encoded,
        "saved_bytes": total_raw - total_encoded,
        "ratio_raw_over_encoded": total_raw / total_encoded,
        "source_fingerprint": source["fingerprint"],
        "tensor_order": tensor_order,
    }


class BF16NF12Sidecar:
    """Checkpoint-bound immutable-generation reader."""

    def __init__(self, model_dir: str | Path, root: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self.root = Path(root).expanduser().resolve()
        generation = (self.root / CURRENT).read_text().strip()
        if not re.fullmatch(r"gen-[A-Za-z0-9-]+", generation):
            raise ValueError(f"invalid BF16 NF12 generation {generation!r}")
        self.generation_dir = self.root / generation
        self.manifest = json.loads(
            (self.generation_dir / "manifest.json").read_text()
        )
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(
                f"unsupported BF16 NF12 schema "
                f"{self.manifest.get('schema')!r}"
            )
        source = _source_fingerprint(self.model_dir)
        if self.manifest.get("source", {}).get("fingerprint") != source[
            "fingerprint"
        ]:
            raise ValueError(
                "BF16 NF12 checkpoint fingerprint does not match model"
            )

    def has_layer(self, layer: int) -> bool:
        return str(int(layer)) in self.manifest["layers"]

    def layer_entry(self, layer: int) -> dict:
        return self.manifest["layers"][str(int(layer))]

    def encoded_names(self, layer: int) -> set[str]:
        return {
            tensor["name"] for tensor in self.layer_entry(layer)["tensors"]
        }

    def layer_path(self, layer: int) -> Path:
        return self.generation_dir / self.layer_entry(layer)["file"]

    def invalidate_layer_cache(self, layer: int) -> bool:
        """Best-effort Darwin UBC invalidation after a mapped one-shot read.

        Chromium's macOS test utility uses a shared read-only mapping followed
        by ``msync(MS_INVALIDATE)`` because Darwin exposes no direct per-file
        cache eviction API. Failure is non-fatal: the live memory governor
        remains authoritative.
        """
        if os.uname().sysname != "Darwin":
            return False
        path = self.layer_path(layer)
        length = path.stat().st_size
        if length <= 0:
            return True
        fd = os.open(path, os.O_RDONLY)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        libc.msync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        pointer = libc.mmap(None, length, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        os.close(fd)
        failed = ctypes.c_void_p(-1).value
        if pointer == failed:
            return False
        try:
            return libc.msync(pointer, length, 2) == 0  # MS_INVALIDATE
        finally:
            libc.munmap(pointer, length)

    def read_layer(
        self, layer: int, *, uncached: bool = False
    ) -> tuple[bytes, int]:
        entry = self.layer_entry(layer)
        path = self.layer_path(layer)
        fd = os.open(path, os.O_RDONLY)
        try:
            if uncached:
                try:
                    import fcntl

                    command = getattr(fcntl, "F_NOCACHE", None)
                    if command is not None:
                        fcntl.fcntl(fd, command, 1)
                except (ImportError, OSError):
                    pass
            length_raw = os.pread(fd, 8, 0)
            if len(length_raw) != 8:
                raise EOFError(f"layer {layer}: truncated safetensors length")
            header_length = struct.unpack("<Q", length_raw)[0]
            header_raw = os.pread(fd, header_length, 8)
            if len(header_raw) != header_length:
                raise EOFError(f"layer {layer}: truncated safetensors header")
            header = json.loads(header_raw)
            metadata = header["encoded"]
            start, end = (int(value) for value in metadata["data_offsets"])
            if metadata["dtype"] != "U8" or metadata["shape"] != [end - start]:
                raise ValueError(f"layer {layer}: invalid encoded tensor header")
            raw = os.pread(fd, end - start, 8 + header_length + start)
        finally:
            os.close(fd)
        expected = int(entry["file_bytes"])
        if len(raw) != expected:
            raise EOFError(
                f"layer {layer}: NF12 bytes {len(raw)} != {expected}"
            )
        return raw, len(raw)

    def read_compact_tensors(
        self, layer: int, names: list[str] | tuple[str, ...]
    ) -> dict[str, tuple[bytes, dict, dict]]:
        """Read independently consumable exact NF12 tensor records.

        Each record retains the same block headers/codes/patches but rebases
        patch and output offsets to zero. This lets a prefetch worker perform
        the physical reads while a later fused linear kernel consumes the
        compressed operand without materializing a dense BF16 matrix.
        """
        entry = self.layer_entry(layer)
        specs = {
            tensor["name"]: tensor for tensor in entry["tensors"]
        }
        path = self.layer_path(layer)
        fd = os.open(path, os.O_RDONLY)
        try:
            length_raw = os.pread(fd, 8, 0)
            if len(length_raw) != 8:
                raise EOFError(
                    f"layer {layer}: truncated safetensors length"
                )
            header_length = struct.unpack("<Q", length_raw)[0]
            payload_base = 8 + header_length
            output = {}
            for name in dict.fromkeys(names):
                spec = specs.get(name)
                if spec is None:
                    raise KeyError(
                        f"{name}: tensor is not encoded in NF12 layer"
                    )
                blocks = int(spec["block_count"])
                block_start = int(spec["block_start"])
                header_bytes = blocks * BLOCK_HEADER_BYTES
                code_bytes = blocks * CODE_BYTES_PER_BLOCK
                patch_bytes = (
                    int(spec["encoded_bytes"])
                    - header_bytes
                    - code_bytes
                )
                headers = bytearray(
                    os.pread(
                        fd,
                        header_bytes,
                        payload_base
                        + block_start * BLOCK_HEADER_BYTES,
                    )
                )
                if len(headers) != header_bytes:
                    raise EOFError(f"{name}: truncated NF12 headers")
                first_patch = BLOCK_HEADER.unpack_from(headers, 0)[0]
                codes = os.pread(
                    fd,
                    code_bytes,
                    payload_base
                    + int(entry["header_bytes"])
                    + block_start * CODE_BYTES_PER_BLOCK,
                )
                if len(codes) != code_bytes:
                    raise EOFError(f"{name}: truncated NF12 codes")
                patches = os.pread(
                    fd,
                    patch_bytes,
                    payload_base + first_patch,
                )
                if len(patches) != patch_bytes:
                    raise EOFError(f"{name}: truncated NF12 patches")

                header_view = np.frombuffer(
                    headers,
                    dtype=np.dtype(
                        [
                            ("patch", "<u4"),
                            ("output", "<u4"),
                            ("valid", "<u2"),
                            ("count", "<u2"),
                            ("mode", "u1"),
                            ("flags", "u1"),
                            ("reserved", "<u2"),
                        ]
                    ),
                )
                header_view["patch"] = (
                    header_view["patch"]
                    - np.uint32(first_patch)
                    + np.uint32(header_bytes + code_bytes)
                )
                header_view["output"] -= np.uint32(
                    int(spec["output_offset"])
                )
                payload = bytes(headers) + codes + patches
                if len(payload) != int(spec["encoded_bytes"]):
                    raise AssertionError(
                        f"{name}: compact NF12 byte accounting changed"
                    )
                local_spec = dict(spec)
                local_spec["block_start"] = 0
                local_spec["output_offset"] = 0
                local_entry = {
                    "file_bytes": len(payload),
                    "block_count": blocks,
                    "header_bytes": header_bytes,
                    "output_value_count": int(spec["value_count"]),
                    "tensors": [local_spec],
                }
                output[name] = (payload, local_entry, local_spec)
        finally:
            os.close(fd)
        return output

    def warm_tensor_ranges(
        self,
        layer: int,
        names: list[str] | tuple[str, ...],
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
        merge_gap_bytes: int = 64 * 1024,
    ) -> int:
        """Synchronously populate Darwin's file cache for selected tensors.

        The NF12 file stores block headers, codes, and rare patches in three
        global streams. A fused consumer needs only the spans belonging to its
        matrices, but letting Metal take their first page faults serializes
        storage with compute. The ordinary weight-prefetch worker calls this
        method first; bounded ``pread`` buffers are discarded immediately while
        the immutable pages remain in UBC for a later main-thread mmap.

        Returns physical bytes submitted to ``pread`` after nearby ranges are
        merged. It performs no representation transform and retains no payload.
        """
        chunk_bytes = int(chunk_bytes)
        merge_gap_bytes = int(merge_gap_bytes)
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if merge_gap_bytes < 0:
            raise ValueError("merge_gap_bytes must be non-negative")

        entry = self.layer_entry(layer)
        specs = {
            tensor["name"]: tensor for tensor in entry["tensors"]
        }
        path = self.layer_path(layer)
        fd = os.open(path, os.O_RDONLY)
        try:
            length_raw = os.pread(fd, 8, 0)
            if len(length_raw) != 8:
                raise EOFError(
                    f"layer {layer}: truncated safetensors length"
                )
            payload_base = 8 + struct.unpack("<Q", length_raw)[0]
            ranges: list[tuple[int, int]] = []
            for name in dict.fromkeys(names):
                spec = specs.get(name)
                if spec is None:
                    raise KeyError(
                        f"{name}: tensor is not encoded in NF12 layer"
                    )
                blocks = int(spec["block_count"])
                block_start = int(spec["block_start"])
                header_bytes = blocks * BLOCK_HEADER_BYTES
                code_bytes = blocks * CODE_BYTES_PER_BLOCK
                header_start = (
                    payload_base + block_start * BLOCK_HEADER_BYTES
                )
                code_start = (
                    payload_base
                    + int(entry["header_bytes"])
                    + block_start * CODE_BYTES_PER_BLOCK
                )
                first_patch_raw = os.pread(fd, 4, header_start)
                if len(first_patch_raw) != 4:
                    raise EOFError(f"{name}: truncated NF12 header")
                first_patch = struct.unpack("<I", first_patch_raw)[0]
                patch_bytes = (
                    int(spec["encoded_bytes"])
                    - header_bytes
                    - code_bytes
                )
                ranges.extend(
                    (
                        (header_start, header_start + header_bytes),
                        (code_start, code_start + code_bytes),
                    )
                )
                if patch_bytes:
                    ranges.append(
                        (
                            payload_base + first_patch,
                            payload_base + first_patch + patch_bytes,
                        )
                    )

            merged: list[list[int]] = []
            for start, end in sorted(ranges):
                if end <= start:
                    continue
                if (
                    merged
                    and start <= merged[-1][1] + merge_gap_bytes
                ):
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])

            physical_bytes = 0
            for start, end in merged:
                offset = start
                while offset < end:
                    wanted = min(chunk_bytes, end - offset)
                    payload = os.pread(fd, wanted, offset)
                    if len(payload) != wanted:
                        raise EOFError(
                            f"layer {layer}: truncated NF12 warm span"
                        )
                    physical_bytes += len(payload)
                    offset += len(payload)
            return physical_bytes
        finally:
            os.close(fd)


def _parse_layers(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--layers", type=_parse_layers)
    parser.add_argument(
        "--tensor-order", choices=TENSOR_ORDERS, default="lexical"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_nf12_sidecar(
                args.model_dir,
                args.output_root,
                layers=args.layers,
                tensor_order=args.tensor_order,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
