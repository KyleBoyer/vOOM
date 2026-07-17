#!/usr/bin/env python3
"""Pure EmbedRows v2->v3/hash tests with a stub MLX module."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

# runtime.embed_rows only needs MLX inside materialization/lookup. Stub it before
# import so these filesystem/integrity gates cannot initialize Metal.
fake_mlx = types.ModuleType("mlx")
fake_core = types.ModuleType("mlx.core")
fake_mlx.core = fake_core
sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_core)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.embed_rows import EmbedRows


def _v2_fixture() -> tuple[EmbedRows, tempfile.TemporaryDirectory, bytes]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    rows = [b"abcd", b"efgh", b"ijkl"]
    payload = b"".join(rows)
    obj = EmbedRows.__new__(EmbedRows)
    obj.path = root / "embed_rows.bin"
    obj.meta_path = root / "embed_rows.meta.json"
    obj.hash_path = root / "embed_rows.rowsha256"
    obj.hidden = 2
    obj.row_bytes = 4
    obj.rows = len(rows)
    obj.expected_bytes = len(payload)
    obj._source = {"fixture": 1}
    obj._row_hashes = b""
    obj._fd = -1
    obj.path.write_bytes(payload)
    obj.meta_path.write_text(json.dumps({
        "version": 2,
        "hidden": obj.hidden,
        "rows": obj.rows,
        "bytes": obj.expected_bytes,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": obj._source,
    }))
    return obj, tmp, payload


def test_v2_upgrade_publishes_per_row_hashes() -> None:
    obj, tmp, payload = _v2_fixture()
    try:
        assert obj._is_current()
        meta = json.loads(obj.meta_path.read_text())
        assert meta["version"] == 3
        hashes = obj.hash_path.read_bytes()
        assert len(hashes) == obj.rows * 32
        expected = b"".join(
            hashlib.sha256(payload[i:i + obj.row_bytes]).digest()
            for i in range(0, len(payload), obj.row_bytes)
        )
        assert hashes == expected == obj._row_hashes
        assert not list(obj.path.parent.glob("*.tmp"))
    finally:
        tmp.cleanup()


def test_verified_row_rejects_later_bit_flip() -> None:
    obj, tmp, _ = _v2_fixture()
    try:
        assert obj._is_current()
        obj._fd = os.open(obj.path, os.O_RDONLY)
        assert obj._read_verified_row(1) == b"efgh"
        with obj.path.open("r+b") as f:
            f.seek(obj.row_bytes)
            f.write(b"Efgh")
            f.flush()
            os.fsync(f.fileno())
        try:
            obj._read_verified_row(1)
        except IOError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("corrupt embedding row was accepted")
    finally:
        if obj._fd >= 0:
            os.close(obj._fd)
        tmp.cleanup()


def test_bad_v2_payload_refuses_attestation() -> None:
    obj, tmp, _ = _v2_fixture()
    try:
        with obj.path.open("r+b") as f:
            f.seek(0)
            f.write(b"X")
        assert not obj._is_current()
        assert not obj.hash_path.exists()
    finally:
        tmp.cleanup()


def _run_all() -> None:
    tests = [
        test_v2_upgrade_publishes_per_row_hashes,
        test_verified_row_rejects_later_bit_flip,
        test_bad_v2_payload_refuses_attestation,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert sys.modules["mlx.core"] is fake_core
    assert "torch" not in sys.modules
    print(f"PASS {len(tests)}/{len(tests)}; stub MLX/no Torch")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
