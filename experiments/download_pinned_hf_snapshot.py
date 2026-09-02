#!/usr/bin/env python3
"""Download and attest a small pinned Hugging Face model snapshot.

This is intentionally for checkpoints that fit as ordinary Hub snapshots, not
the project's asynchronous multi-hundred-GB model acquisition path.  It writes
the tree receipt consumed by fail-closed sidecar builders and publishes an
atomic child result suitable for ``experiments/run_gate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--expected-file", action="append", default=[],
                        metavar="PATH:BYTES:SHA256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive")
    expected: dict[str, tuple[int, str]] = {}
    for value in args.expected_file:
        name, raw_bytes, sha256 = value.rsplit(":", 2)
        if not name or len(sha256) != 64:
            raise ValueError(f"invalid --expected-file {value!r}")
        expected[name] = (int(raw_bytes), sha256)

    started = time.perf_counter()
    info = HfApi().model_info(
        args.repository, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError(
            f"Hub resolved {args.revision} to unexpected revision {info.sha}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repository,
        revision=args.revision,
        local_dir=output,
        allow_patterns=(args.include or None),
        max_workers=args.max_workers,
    )

    selected: dict[str, dict] = {}
    for sibling in info.siblings or ():
        name = sibling.rfilename
        if not (output / name).is_file():
            continue
        entry = {
            "size": int(sibling.size or (output / name).stat().st_size),
            "blob_id": str(sibling.blob_id or ""),
        }
        lfs = sibling.lfs
        if lfs is not None:
            entry.update({
                "lfs_sha256": str(lfs.sha256),
                "lfs_size": int(lfs.size),
            })
        xet = getattr(sibling, "xet_file_data", None)
        if xet is not None:
            entry["xet_hash"] = str(xet.file_hash)
        selected[name] = entry

    checks = {}
    for name, (expected_bytes, expected_sha256) in expected.items():
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"download omitted expected file {name}")
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256(path)
        if (actual_bytes, actual_sha256) != (expected_bytes, expected_sha256):
            raise ValueError(
                f"downloaded {name} identity mismatch: "
                f"bytes={actual_bytes}, sha256={actual_sha256}")
        checks[name] = {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }

    receipt = {
        "format_version": 1,
        "repository": args.repository,
        "revision": args.revision,
        "files": dict(sorted(selected.items())),
    }
    tree_path = (
        output / ".cache" / "huggingface" / "trees"
        / f"{args.revision}.json")
    _atomic_json(tree_path, receipt)
    result = {
        "schema": "voom.pinned-hf-snapshot-download.v1",
        "passed": True,
        "repository": args.repository,
        "revision": args.revision,
        "output": str(output),
        "downloaded_files": len(selected),
        "verified_files": checks,
        "tree_receipt": str(tree_path),
        "wall_s": time.perf_counter() - started,
    }
    _atomic_json(args.result_json.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
