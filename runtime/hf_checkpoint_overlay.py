"""Build a hash-attested Hugging Face checkpoint overlay.

The overlay downloads only files whose immutable Hub object identity differs
from a pinned local base checkpoint. Byte-identical files become symlinks only
after their local bytes pass the candidate's own published hash. This keeps a
candidate isolated for evaluation without duplicating hundreds of gigabytes or
mutating the serving checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
from typing import Any


PLAN_SCHEMA = "voom.hf-checkpoint-overlay-plan.v1"
RECEIPT_SCHEMA = "voom.hf-checkpoint-overlay-receipt.v1"
PLAN_NAME = ".voom-overlay-plan.json"
RECEIPT_NAME = "voom.overlay.receipt.json"


def _safe_relative(name: str) -> str:
    raw = str(name)
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(
            part in ("", ".", "..") for part in raw.split("/")):
        raise ValueError(f"unsafe Hub path: {name!r}")
    return path.as_posix()


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    kind = str(record["hash_kind"])
    digest = str(record["hash"])
    if kind not in ("sha256", "git-blob-sha1"):
        raise ValueError(f"unsupported Hub hash kind: {kind}")
    expected = 64 if kind == "sha256" else 40
    if len(digest) != expected or any(
            char not in "0123456789abcdef" for char in digest.lower()):
        raise ValueError(f"invalid {kind} digest for {record.get('path')}")
    return kind, digest.lower()


def _hub_record(sibling: Any) -> dict[str, Any]:
    name = _safe_relative(sibling.rfilename)
    size = int(sibling.size)
    lfs = getattr(sibling, "lfs", None)
    lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if lfs_sha:
        kind, digest = "sha256", str(lfs_sha)
    else:
        kind, digest = "git-blob-sha1", str(sibling.blob_id)
    record = {
        "path": name,
        "size": size,
        "hash_kind": kind,
        "hash": digest.lower(),
    }
    _identity(record)
    return record


def _records(info: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for sibling in info.siblings:
        record = _hub_record(sibling)
        if record["path"] in result:
            raise ValueError(f"duplicate Hub path: {record['path']}")
        result[record["path"]] = record
    return result


def build_plan(
    *, base_repo: str, base_revision: str, base_dir: Path,
    base_records: dict[str, dict[str, Any]], candidate_repo: str,
    candidate_revision: str,
    candidate_records: dict[str, dict[str, Any]], destination: Path,
) -> dict[str, Any]:
    """Create the pure, serializable overlay plan from pinned Hub records."""
    base_shards = {
        name for name in base_records if name.endswith(".safetensors")}
    candidate_shards = {
        name for name in candidate_records if name.endswith(".safetensors")}
    if not base_shards or base_shards != candidate_shards:
        missing = sorted(base_shards - candidate_shards)
        extra = sorted(candidate_shards - base_shards)
        raise ValueError(
            "candidate safetensor layout differs from base: "
            f"missing={missing[:4]}, extra={extra[:4]}")

    links = []
    downloads = []
    for name in sorted(candidate_records):
        candidate = dict(candidate_records[name])
        _safe_relative(name)
        _identity(candidate)
        base = base_records.get(name)
        if (base is not None and int(base["size"]) == int(candidate["size"])
                and _identity(base) == _identity(candidate)):
            links.append(candidate)
        else:
            downloads.append(candidate)

    changed_shards = sum(
        record["path"].endswith(".safetensors") for record in downloads)
    return {
        "schema": PLAN_SCHEMA,
        "base": {
            "repo": base_repo,
            "revision": base_revision,
            "directory": str(base_dir.resolve()),
        },
        "candidate": {
            "repo": candidate_repo,
            "revision": candidate_revision,
        },
        "destination": str(destination.resolve()),
        "files": {
            "candidate_total": len(candidate_records),
            "safetensor_shards": len(candidate_shards),
            "changed_safetensor_shards": changed_shards,
            "download": downloads,
            "link": links,
            "download_bytes": sum(int(item["size"]) for item in downloads),
            "link_bytes": sum(int(item["size"]) for item in links),
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _load_plan(destination: Path) -> dict[str, Any]:
    plan_path = destination / PLAN_NAME
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported overlay plan: {plan.get('schema')}")
    if Path(plan["destination"]).resolve() != destination.resolve():
        raise ValueError("overlay plan destination does not match CLI target")
    return plan


def _hash_file(path: Path, kind: str) -> str:
    size = path.stat().st_size
    if kind == "sha256":
        digest = hashlib.sha256()
    elif kind == "git-blob-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode())
    else:
        raise ValueError(f"unsupported hash kind: {kind}")
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, record: dict[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected downloaded regular file: {path}")
    if path.stat().st_size != int(record["size"]):
        raise ValueError(f"size mismatch: {path}")
    kind, expected = _identity(record)
    actual = _hash_file(path, kind)
    if actual != expected:
        raise ValueError(
            f"{kind} mismatch for {path}: {actual} != {expected}")


def _verify_base(path: Path, record: dict[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"missing base file: {path}")
    if path.stat().st_size != int(record["size"]):
        raise ValueError(f"base size mismatch: {path}")
    kind, expected = _identity(record)
    actual = _hash_file(path, kind)
    if actual != expected:
        raise ValueError(
            f"base {kind} mismatch for {path}: {actual} != {expected}")


def _assert_model_metadata(base: Path, candidate: Path) -> int:
    base_config = json.loads((base / "config.json").read_text())
    candidate_config = json.loads((candidate / "config.json").read_text())
    if base_config != candidate_config:
        raise ValueError("candidate config.json differs from pinned base")
    base_index = json.loads(
        (base / "model.safetensors.index.json").read_text())
    candidate_index = json.loads(
        (candidate / "model.safetensors.index.json").read_text())
    base_map = base_index.get("weight_map", {})
    candidate_map = candidate_index.get("weight_map", {})
    if base_map != candidate_map:
        raise ValueError("candidate tensor-to-shard map differs from base")
    return len(base_map)


def plan_command(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    destination = args.destination.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        raise ValueError(f"base checkpoint does not exist: {base_dir}")
    if destination == base_dir or base_dir in destination.parents:
        raise ValueError("overlay destination must be outside the base checkpoint")
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / PLAN_NAME
    if plan_path.exists() and not args.resume:
        raise FileExistsError(f"overlay plan already exists: {plan_path}")

    api = HfApi()
    base_info = api.model_info(
        args.base_repo, revision=args.base_revision, files_metadata=True)
    candidate_info = api.model_info(
        args.candidate_repo, revision=args.candidate_revision,
        files_metadata=True)
    if base_info.sha != args.base_revision:
        raise ValueError(
            f"base revision resolved to {base_info.sha}, expected "
            f"{args.base_revision}")
    if candidate_info.sha != args.candidate_revision:
        raise ValueError(
            f"candidate revision resolved to {candidate_info.sha}, expected "
            f"{args.candidate_revision}")
    plan = build_plan(
        base_repo=args.base_repo,
        base_revision=base_info.sha,
        base_dir=base_dir,
        base_records=_records(base_info),
        candidate_repo=args.candidate_repo,
        candidate_revision=candidate_info.sha,
        candidate_records=_records(candidate_info),
        destination=destination,
    )
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise ValueError("existing overlay plan differs from pinned Hub state")
    else:
        _atomic_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


def download_command(args: argparse.Namespace) -> None:
    destination = args.destination.expanduser().resolve()
    plan = _load_plan(destination)
    records = plan["files"]["download"]
    required = sum(int(item["size"]) for item in records)
    free = shutil.disk_usage(destination).free
    reserve = int(args.reserve_gb * 1_000_000_000)
    present = sum(
        (destination / item["path"]).stat().st_size
        for item in records
        if (destination / item["path"]).is_file()
        and not (destination / item["path"]).is_symlink())
    remaining = max(0, required - present)
    if free - remaining < reserve:
        raise RuntimeError(
            f"overlay download needs {remaining} bytes while preserving "
            f"{reserve}; filesystem has {free} free")
    command = [
        args.hf_binary, "download", plan["candidate"]["repo"],
        *(item["path"] for item in records),
        "--revision", plan["candidate"]["revision"],
        "--local-dir", str(destination),
        "--max-workers", str(args.max_workers),
        "--format", "agent",
    ]
    subprocess.run(command, check=True)


def finalize_command(args: argparse.Namespace) -> None:
    destination = args.destination.expanduser().resolve()
    plan = _load_plan(destination)
    base = Path(plan["base"]["directory"]).resolve()
    if (destination / RECEIPT_NAME).exists():
        raise FileExistsError("overlay receipt already exists")
    verified_download_bytes = 0
    for record in plan["files"]["download"]:
        path = destination / record["path"]
        _verify(path, record)
        verified_download_bytes += int(record["size"])

    verified_link_bytes = 0
    for record in plan["files"]["link"]:
        source = base / record["path"]
        _verify_base(source, record)
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError(f"overlay link target already differs: {target}")
        else:
            temporary = target.with_name(
                target.name + f".{os.getpid()}.link.tmp")
            os.symlink(source, temporary)
            os.replace(temporary, target)
        verified_link_bytes += int(record["size"])

    tensor_count = _assert_model_metadata(base, destination)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_unix": time.time(),
        "plan_sha256": hashlib.sha256(
            (destination / PLAN_NAME).read_bytes()).hexdigest(),
        "base": plan["base"],
        "candidate": plan["candidate"],
        "destination": str(destination),
        "downloaded_files": len(plan["files"]["download"]),
        "linked_files": len(plan["files"]["link"]),
        "verified_download_bytes": verified_download_bytes,
        "verified_link_bytes": verified_link_bytes,
        "safetensor_shards": plan["files"]["safetensor_shards"],
        "changed_safetensor_shards": plan["files"][
            "changed_safetensor_shards"],
        "tensor_map_entries": tensor_count,
        "config_equal": True,
        "tensor_to_shard_map_equal": True,
        "status": "verified",
    }
    _atomic_json(destination / RECEIPT_NAME, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base-repo", required=True)
    plan_parser.add_argument("--base-revision", required=True)
    plan_parser.add_argument("--base-dir", type=Path, required=True)
    plan_parser.add_argument("--candidate-repo", required=True)
    plan_parser.add_argument("--candidate-revision", required=True)
    plan_parser.add_argument("--destination", type=Path, required=True)
    plan_parser.add_argument("--resume", action="store_true")
    plan_parser.set_defaults(handler=plan_command)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--destination", type=Path, required=True)
    download_parser.add_argument("--hf-binary", default="hf")
    download_parser.add_argument("--max-workers", type=int, default=4)
    download_parser.add_argument("--reserve-gb", type=float, default=10.0)
    download_parser.set_defaults(handler=download_command)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--destination", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize_command)

    args = parser.parse_args()
    if getattr(args, "max_workers", 1) <= 0:
        parser.error("--max-workers must be positive")
    if getattr(args, "reserve_gb", 0.0) < 0:
        parser.error("--reserve-gb must be non-negative")
    args.handler(args)


if __name__ == "__main__":
    main()
