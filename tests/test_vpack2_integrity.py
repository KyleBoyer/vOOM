#!/usr/bin/env python3
"""Pure vpack2 corruption gates; imports neither MLX nor Torch."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formats.packed2 import Vpack2Reader


def _fixture(*, hashed: bool = True) -> tuple[Path, tempfile.TemporaryDirectory, bytes]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    body = b"exact-bf16-body"
    (root / "weights.vpack2").write_bytes(body)
    entry = {
        "off": 0,
        "len": len(body),
        "head": {"raw": len(body), "dtype": "U8", "shape": [len(body)]},
    }
    if hashed:
        entry["sha"] = hashlib.sha256(body).hexdigest()[:32]
    (root / "weights.vpack2.index.json").write_text(json.dumps({"w": entry}))
    return root, tmp, body


def test_valid_body_hash() -> None:
    root, tmp, body = _fixture()
    try:
        reader = Vpack2Reader(root)
        assert reader._checked_body("w", reader.index["w"], body) == body
    finally:
        tmp.cleanup()


def test_missing_hash_fails_closed() -> None:
    root, tmp, _ = _fixture(hashed=False)
    try:
        try:
            Vpack2Reader(root, require_hashes=True)
        except ValueError as exc:
            assert "hash" in str(exc)
        else:
            raise AssertionError("strict reader accepted an unhashed tensor")
    finally:
        tmp.cleanup()


def test_corrupt_body_is_rejected() -> None:
    root, tmp, _ = _fixture()
    try:
        reader = Vpack2Reader(root)
        try:
            reader._checked_body("w", reader.index["w"], b"corrupt")
        except IOError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise AssertionError("corrupt tensor body was accepted")
    finally:
        tmp.cleanup()


def test_archive_mutation_is_rejected() -> None:
    root, tmp, _ = _fixture()
    try:
        reader = Vpack2Reader(root)
        with (root / "weights.vpack2").open("ab") as f:
            f.write(b"!")
        try:
            reader._assert_archive_immutable()
        except IOError as exc:
            assert "changed" in str(exc)
        else:
            raise AssertionError("mutated archive identity was accepted")
    finally:
        tmp.cleanup()


def _run_all() -> None:
    tests = [
        test_valid_body_hash,
        test_missing_hash_fails_closed,
        test_corrupt_body_is_rejected,
        test_archive_mutation_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert "mlx" not in sys.modules
    assert "torch" not in sys.modules
    print(f"PASS {len(tests)}/{len(tests)}; no MLX/Torch import")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
