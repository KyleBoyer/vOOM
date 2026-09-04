"""Resumable, hash-attested in-place Hugging Face checkpoint replacement.

This is for same-shard releases that are too large to duplicate locally.  A
durable marker is published before the first changed byte; WeightStore refuses
that marker.  Each file is verified as the pinned base, downloaded and verified
as the pinned candidate, then atomically renamed on the checkpoint filesystem.
There is deliberately no hidden NAS backup.  The pinned Hub base revision is
the recovery source.

The default remains strict: serving config and tensor index must be identical.
An explicit ``--allow-layout-change`` plan may replace those files too when the
candidate deliberately changes representation while retaining the same shard
filenames.  The loader-blocking marker spans the entire mixed-layout interval,
so no partially converted checkpoint can be opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from .checkpoint_identity import (
    REPLACEMENT_MARKER_NAME,
    REPLACEMENT_RECEIPT_NAME,
)
from .hf_checkpoint_overlay import (
    _atomic_json,
    _hash_file,
    _identity,
    _records,
    _safe_relative,
    _verify,
    _verify_base,
)


PLAN_SCHEMA = "voom.hf-checkpoint-replacement-plan.v1"
RECEIPT_SCHEMA = "voom.hf-checkpoint-replacement-receipt.v1"
STAGING_NAME = ".voom-checkpoint-replacement-staging"


def _same_record(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left["size"]) == int(right["size"])
        and _identity(left) == _identity(right)
    )


def build_plan(
    *, base_repo: str, base_revision: str,
    base_records: dict[str, dict[str, Any]], candidate_repo: str,
    candidate_revision: str,
    candidate_records: dict[str, dict[str, Any]], model_dir: Path,
    allow_layout_change: bool = False,
) -> dict[str, Any]:
    """Build a same-shard replacement plan from pinned Hub records."""
    base_shards = {
        name for name in base_records if name.endswith(".safetensors")}
    candidate_shards = {
        name for name in candidate_records if name.endswith(".safetensors")}
    if not base_shards or base_shards != candidate_shards:
        raise ValueError(
            "candidate safetensor layout differs from base: "
            f"missing={sorted(base_shards - candidate_shards)[:4]}, "
            f"extra={sorted(candidate_shards - base_shards)[:4]}"
        )
    # Strict replacement is intentionally narrower than an overlay.  An
    # explicit representation migration may change both serving files, but
    # they must still exist as pinned candidate objects and every shard name
    # must remain one-for-one so each old shard can be atomically overwritten.
    for required in ("config.json", "model.safetensors.index.json"):
        base = base_records.get(required)
        candidate = candidate_records.get(required)
        if base is None or candidate is None:
            raise ValueError(
                f"replacement requires both releases to publish {required}"
            )
        if not allow_layout_change and not _same_record(base, candidate):
            raise ValueError(
                f"same-layout replacement requires identical {required}"
            )

    replace = []
    keep = []
    for name in sorted(candidate_records):
        _safe_relative(name)
        candidate = dict(candidate_records[name])
        _identity(candidate)
        base = base_records.get(name)
        if base is not None and _same_record(base, candidate):
            keep.append(candidate)
        else:
            item = {"candidate": candidate}
            if base is not None:
                item["base"] = dict(base)
            replace.append(item)
    remove = [
        dict(base_records[name])
        for name in sorted(set(base_records) - set(candidate_records))
    ]
    return {
        "schema": PLAN_SCHEMA,
        "created_unix": time.time(),
        "model_dir": str(model_dir.resolve()),
        "base": {"repo": base_repo, "revision": base_revision},
        "candidate": {
            "repo": candidate_repo, "revision": candidate_revision},
        "replacement_mode": (
            "same-shards-layout-change"
            if allow_layout_change else "same-layout"),
        "files": {
            "candidate_total": len(candidate_records),
            "safetensor_shards": len(candidate_shards),
            "changed_safetensor_shards": sum(
                item["candidate"]["path"].endswith(".safetensors")
                for item in replace
            ),
            "replace": replace,
            "keep": keep,
            "remove": remove,
            "replace_bytes": sum(
                int(item["candidate"]["size"]) for item in replace),
            "keep_bytes": sum(int(item["size"]) for item in keep),
        },
    }


def _load_plan(model_dir: Path) -> dict[str, Any]:
    marker = model_dir / REPLACEMENT_MARKER_NAME
    plan = json.loads(marker.read_text())
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported checkpoint replacement marker: {marker}")
    if Path(plan.get("model_dir", "")).resolve() != model_dir.resolve():
        raise ValueError("replacement marker belongs to a different checkpoint")
    return plan


def _matches(path: Path, record: dict[str, Any] | None) -> bool:
    if record is None or not path.is_file() or path.is_symlink():
        return False
    if path.stat().st_size != int(record["size"]):
        return False
    kind, expected = _identity(record)
    return _hash_file(path, kind) == expected


def _classify_changed_path(
    path: Path, candidate: dict[str, Any], base: dict[str, Any] | None,
) -> str:
    """Classify base/candidate bytes with one content pass when possible."""
    if not path.exists() and base is None:
        return "missing_new"
    if not path.is_file() or path.is_symlink():
        return "invalid"
    size = path.stat().st_size
    candidate_kind, candidate_hash = _identity(candidate)
    base_identity = _identity(base) if base is not None else None
    candidate_size = int(candidate["size"])
    base_size = int(base["size"]) if base is not None else -1
    possible_candidate = size == candidate_size
    possible_base = size == base_size
    if not possible_candidate and not possible_base:
        return "invalid"
    if (
        possible_candidate and possible_base and base_identity is not None
        and base_identity[0] == candidate_kind
    ):
        actual = _hash_file(path, candidate_kind)
        if actual == candidate_hash:
            return "candidate"
        if actual == base_identity[1]:
            return "base"
        return "invalid"
    if possible_candidate and _hash_file(path, candidate_kind) == candidate_hash:
        return "candidate"
    if possible_base and base_identity is not None:
        if _hash_file(path, base_identity[0]) == base_identity[1]:
            return "base"
    return "invalid"


def audit(model_dir: Path, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    plan = plan or _load_plan(model_dir)
    states = {"base": 0, "candidate": 0, "missing_new": 0, "invalid": 0}
    bytes_by_state = {key: 0 for key in states}
    invalid_paths = []
    path_states = {}
    for item in plan["files"]["replace"]:
        candidate = item["candidate"]
        base = item.get("base")
        path = model_dir / candidate["path"]
        state = _classify_changed_path(path, candidate, base)
        if state == "invalid":
            invalid_paths.append(candidate["path"])
        states[state] += 1
        bytes_by_state[state] += int(candidate["size"])
        path_states[candidate["path"]] = state
    return {
        "schema": "voom.hf-checkpoint-replacement-audit.v1",
        "model_dir": str(model_dir),
        "base": plan["base"],
        "candidate": plan["candidate"],
        "states": states,
        "bytes": bytes_by_state,
        "path_states": path_states,
        "invalid_paths": invalid_paths,
        "complete": (
            states["candidate"] == len(plan["files"]["replace"])
            and not states["invalid"]
        ),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _download_one(
    *, repo: str, revision: str, record: dict[str, Any],
    staging: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    relative = _safe_relative(record["path"])
    result = Path(hf_hub_download(
        repo_id=repo,
        filename=relative,
        revision=revision,
        local_dir=staging,
    ))
    expected = (staging / relative).resolve()
    if result.resolve() != expected or staging.resolve() not in expected.parents:
        raise ValueError(f"Hub download escaped staging directory: {result}")
    _verify(result, record)
    return result


def apply_plan(
    model_dir: Path, *, reserve_bytes: int = 10_000_000_000,
    max_files: int = 0,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    plan = _load_plan(model_dir)
    before = audit(model_dir, plan)
    if before["invalid_paths"]:
        raise ValueError(
            "replacement refuses unrecognized local bytes: "
            + ", ".join(before["invalid_paths"][:4])
        )
    staging = model_dir / STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)
    applied = 0
    applied_bytes = 0
    applied_paths = []
    for item in plan["files"]["replace"]:
        candidate = item["candidate"]
        target = model_dir / candidate["path"]
        state = before["path_states"][candidate["path"]]
        if state == "candidate":
            continue
        base = item.get("base")
        if state == "base":
            assert base is not None
        elif state == "missing_new":
            assert base is None and not target.exists()
        else:
            raise ValueError(f"new candidate path already exists: {target}")
        required = 2 * int(candidate["size"])
        free = shutil.disk_usage(model_dir).free
        if free - required < int(reserve_bytes):
            raise RuntimeError(
                f"replacing {candidate['path']} may need {required} staging "
                f"bytes while preserving {reserve_bytes}; only {free} free"
            )
        staging.mkdir(mode=0o700)
        try:
            downloaded = _download_one(
                repo=plan["candidate"]["repo"],
                revision=plan["candidate"]["revision"],
                record=candidate,
                staging=staging,
            )
            downloaded_stat = downloaded.stat()
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, target)
            _fsync_directory(target.parent)
            target_stat = target.stat()
            if (
                target_stat.st_dev != downloaded_stat.st_dev
                or target_stat.st_ino != downloaded_stat.st_ino
                or target_stat.st_size != int(candidate["size"])
            ):
                raise IOError(
                    f"atomic replacement identity changed unexpectedly: {target}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        applied += 1
        applied_bytes += int(candidate["size"])
        applied_paths.append(candidate["path"])
        print(json.dumps({
            "replaced": candidate["path"],
            "files_this_run": applied,
            "bytes_this_run": applied_bytes,
        }), flush=True)
        if max_files and applied >= max_files:
            break
    after = dict(before)
    after["states"] = dict(before["states"])
    after["bytes"] = dict(before["bytes"])
    after["path_states"] = dict(before["path_states"])
    entries = {
        item["candidate"]["path"]: item["candidate"]
        for item in plan["files"]["replace"]
    }
    for path in applied_paths:
        previous = after["path_states"][path]
        size = int(entries[path]["size"])
        after["states"][previous] -= 1
        after["bytes"][previous] -= size
        after["states"]["candidate"] += 1
        after["bytes"]["candidate"] += size
        after["path_states"][path] = "candidate"
    after["complete"] = (
        after["states"]["candidate"] == len(plan["files"]["replace"])
        and not after["states"]["invalid"]
    )
    after["applied_files"] = applied
    after["applied_bytes"] = applied_bytes
    return after


def finalize(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    plan = _load_plan(model_dir)
    report = audit(model_dir, plan)
    if not report["complete"]:
        raise RuntimeError(f"replacement is incomplete: {report['states']}")

    # Full candidate-tree attestation is intentionally expensive and occurs
    # once, before the marker is removed.
    verified_bytes = 0
    for record in plan["files"]["keep"]:
        _verify_base(model_dir / record["path"], record)
        verified_bytes += int(record["size"])
    verified_bytes += report["bytes"]["candidate"]

    removed = []
    for record in plan["files"]["remove"]:
        target = model_dir / record["path"]
        if not target.exists():
            continue
        _verify_base(target, record)
        target.unlink()
        removed.append(record["path"])

    marker = model_dir / REPLACEMENT_MARKER_NAME
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "completed_unix": time.time(),
        "plan_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        "base": plan["base"],
        "candidate": plan["candidate"],
        "model_dir": str(model_dir),
        "candidate_files": plan["files"]["candidate_total"],
        "safetensor_shards": plan["files"]["safetensor_shards"],
        "changed_safetensor_shards": plan["files"][
            "changed_safetensor_shards"],
        "verified_candidate_bytes": verified_bytes,
        "removed_base_only_files": removed,
        "status": "verified",
    }
    _atomic_json(model_dir / REPLACEMENT_RECEIPT_NAME, receipt)
    marker.unlink()
    _fsync_directory(model_dir)
    return receipt


def plan_command(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"checkpoint does not exist: {model_dir}")
    marker = model_dir / REPLACEMENT_MARKER_NAME
    if marker.exists():
        if not args.resume:
            raise FileExistsError(f"replacement marker already exists: {marker}")
        print(json.dumps(_load_plan(model_dir), indent=2, sort_keys=True))
        return
    api = HfApi()
    base_info = api.model_info(
        args.base_repo, revision=args.base_revision, files_metadata=True)
    candidate_info = api.model_info(
        args.candidate_repo,
        revision=args.candidate_revision,
        files_metadata=True,
    )
    if base_info.sha != args.base_revision:
        raise ValueError("base revision did not resolve exactly")
    if candidate_info.sha != args.candidate_revision:
        raise ValueError("candidate revision did not resolve exactly")
    base_records = _records(base_info)
    candidate_records = _records(candidate_info)
    # Before publishing the fail-closed marker, prove the two serving metadata
    # files are exactly the pinned base.
    for name in ("config.json", "model.safetensors.index.json"):
        _verify_base(model_dir / name, base_records[name])
    plan = build_plan(
        base_repo=args.base_repo,
        base_revision=base_info.sha,
        base_records=base_records,
        candidate_repo=args.candidate_repo,
        candidate_revision=candidate_info.sha,
        candidate_records=candidate_records,
        model_dir=model_dir,
        allow_layout_change=args.allow_layout_change,
    )
    _atomic_json(marker, plan)
    _fsync_directory(model_dir)
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base-repo", required=True)
    plan_parser.add_argument("--base-revision", required=True)
    plan_parser.add_argument("--candidate-repo", required=True)
    plan_parser.add_argument("--candidate-revision", required=True)
    plan_parser.add_argument("--model-dir", type=Path, required=True)
    plan_parser.add_argument("--resume", action="store_true")
    plan_parser.add_argument(
        "--allow-layout-change", action="store_true",
        help=(
            "explicitly allow candidate config/index changes while requiring "
            "the identical safetensor shard filename set"),
    )

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--model-dir", type=Path, required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--model-dir", type=Path, required=True)
    apply_parser.add_argument("--reserve-gb", type=float, default=10.0)
    apply_parser.add_argument("--max-files", type=int, default=0)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--model-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        plan_command(args)
    elif args.command == "audit":
        print(json.dumps(audit(args.model_dir), indent=2, sort_keys=True))
    elif args.command == "apply":
        if args.reserve_gb < 0 or args.max_files < 0:
            parser.error("reserve and max-files must be non-negative")
        print(json.dumps(apply_plan(
            args.model_dir,
            reserve_bytes=int(args.reserve_gb * 1_000_000_000),
            max_files=args.max_files,
        ), indent=2, sort_keys=True))
    else:
        print(json.dumps(finalize(args.model_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
