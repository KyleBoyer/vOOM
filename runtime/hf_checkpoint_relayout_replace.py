"""Resumable, space-bounded replacement across Hugging Face shard layouts.

The same-layout replacement path deliberately rejects a changed tensor-to-
shard map.  This module handles the narrower relayout case where config.json
is byte-identical and the base and candidate indexes contain exactly the same
tensor names, but the shard filenames differ.  Candidate shards are committed
under their new names before every base shard that they cover is hash-verified
and removed.  A loader-blocking marker exists for the entire transition.

There is no backup copy.  Both immutable Hub revisions are recorded as the
recovery sources, and every downloaded or removed shard is checked against its
published Hub object identity.
"""

from __future__ import annotations

import argparse
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
    _identity,
    _records,
    _safe_relative,
    _verify,
    _verify_base,
)
from .hf_checkpoint_replace import (
    _classify_changed_path,
    _download_one,
    _fsync_directory,
    _same_record,
)


PLAN_SCHEMA = "voom.hf-checkpoint-relayout-plan.v1"
AUDIT_SCHEMA = "voom.hf-checkpoint-relayout-audit.v1"
RECEIPT_SCHEMA = "voom.hf-checkpoint-relayout-receipt.v1"
STAGING_NAME = ".voom-checkpoint-relayout-staging"


def _index_map(index: dict[str, Any], label: str) -> dict[str, str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{label} index has no non-empty weight_map")
    result = {}
    for tensor, shard in weight_map.items():
        if not isinstance(tensor, str) or not tensor:
            raise ValueError(f"{label} index has an invalid tensor name")
        if not isinstance(shard, str) or not shard.endswith(".safetensors"):
            raise ValueError(
                f"{label} index has an invalid shard for {tensor!r}")
        result[tensor] = _safe_relative(shard)
    return result


def _indexed_shards(
    records: dict[str, dict[str, Any]], weight_map: dict[str, str], label: str,
) -> set[str]:
    indexed = set(weight_map.values())
    published = {
        name for name in records if name.endswith(".safetensors")}
    if indexed != published:
        raise ValueError(
            f"{label} safetensor records differ from its index: "
            f"unindexed={sorted(published - indexed)[:4]}, "
            f"missing={sorted(indexed - published)[:4]}")
    return indexed


def _download_order(
    base_to_candidate: dict[str, list[str]], candidate_shards: set[str],
) -> list[str]:
    """Produce a deterministic coverage-first order with bounded residency."""
    ready: set[str] = set()
    order = []
    for base in sorted(base_to_candidate):
        for candidate in sorted(base_to_candidate[base]):
            if candidate not in ready:
                ready.add(candidate)
                order.append(candidate)
    for candidate in sorted(candidate_shards - ready):
        order.append(candidate)
    return order


def _simulate_shard_peak(
    order: list[str], base_to_candidate: dict[str, list[str]],
    base_records: dict[str, dict[str, Any]],
    candidate_records: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Simulate retained bytes plus a conservative 2x download transient."""
    ready: set[str] = set()
    released: set[str] = set()
    candidate_bytes = 0
    released_bytes = 0
    peak_extra = 0
    for candidate in order:
        size = int(candidate_records[candidate]["size"])
        # huggingface_hub local_dir may briefly retain both its incomplete/cache
        # object and the final file on the checkpoint filesystem.
        peak_extra = max(
            peak_extra, candidate_bytes - released_bytes + 2 * size)
        candidate_bytes += size
        ready.add(candidate)
        for base, outputs in base_to_candidate.items():
            if base not in released and set(outputs) <= ready:
                released.add(base)
                released_bytes += int(base_records[base]["size"])
    if released != set(base_to_candidate):
        raise AssertionError("download schedule does not cover every base shard")
    return {
        "peak_extra_bytes": peak_extra,
        "candidate_shard_bytes": candidate_bytes,
        "released_base_shard_bytes": released_bytes,
        "final_shard_delta_bytes": candidate_bytes - released_bytes,
    }


def build_plan(
    *, base_repo: str, base_revision: str,
    base_records: dict[str, dict[str, Any]], base_index: dict[str, Any],
    candidate_repo: str, candidate_revision: str,
    candidate_records: dict[str, dict[str, Any]],
    candidate_index: dict[str, Any], model_dir: Path,
) -> dict[str, Any]:
    """Build a pure, serializable plan for an exact tensor-set relayout."""
    for records in (base_records, candidate_records):
        for name, record in records.items():
            if name != record.get("path"):
                raise ValueError(f"record key/path mismatch: {name!r}")
            _safe_relative(name)
            _identity(record)

    base_config = base_records.get("config.json")
    candidate_config = candidate_records.get("config.json")
    if (
        base_config is None or candidate_config is None
        or not _same_record(base_config, candidate_config)
    ):
        raise ValueError("relayout replacement requires identical config.json")

    base_map = _index_map(base_index, "base")
    candidate_map = _index_map(candidate_index, "candidate")
    if set(base_map) != set(candidate_map):
        raise ValueError(
            "candidate tensor names differ from base: "
            f"base_only={sorted(set(base_map) - set(candidate_map))[:4]}, "
            f"candidate_only={sorted(set(candidate_map) - set(base_map))[:4]}")

    base_shards = _indexed_shards(base_records, base_map, "base")
    candidate_shards = _indexed_shards(
        candidate_records, candidate_map, "candidate")
    overlap = base_shards & candidate_shards
    if overlap:
        raise ValueError(
            "relayout shard filenames must be disjoint: "
            + ", ".join(sorted(overlap)[:4]))

    candidate_to_base: dict[str, set[str]] = {
        name: set() for name in candidate_shards}
    base_to_candidate: dict[str, set[str]] = {
        name: set() for name in base_shards}
    for tensor, base_shard in base_map.items():
        candidate_shard = candidate_map[tensor]
        candidate_to_base[candidate_shard].add(base_shard)
        base_to_candidate[base_shard].add(candidate_shard)
    if any(not sources for sources in candidate_to_base.values()):
        raise AssertionError("candidate shard has no source tensor")
    if any(not outputs for outputs in base_to_candidate.values()):
        raise AssertionError("base shard has no candidate coverage")

    candidate_to_base_json = {
        name: sorted(values)
        for name, values in sorted(candidate_to_base.items())}
    base_to_candidate_json = {
        name: sorted(values)
        for name, values in sorted(base_to_candidate.items())}
    order = _download_order(base_to_candidate_json, candidate_shards)
    simulation = _simulate_shard_peak(
        order, base_to_candidate_json, base_records, candidate_records)

    metadata_keep = []
    metadata_replace = []
    for name in sorted(candidate_records):
        if name in candidate_shards:
            continue
        candidate = dict(candidate_records[name])
        base = base_records.get(name)
        if base is not None and _same_record(base, candidate):
            metadata_keep.append(candidate)
        else:
            item = {"candidate": candidate}
            if base is not None:
                item["base"] = dict(base)
            metadata_replace.append(item)
    preserved_base_only = [
        dict(base_records[name])
        for name in sorted(set(base_records) - set(candidate_records))
        if name not in base_shards
    ]

    return {
        "schema": PLAN_SCHEMA,
        "created_unix": time.time(),
        "model_dir": str(model_dir.resolve()),
        "base": {"repo": base_repo, "revision": base_revision},
        "candidate": {
            "repo": candidate_repo, "revision": candidate_revision},
        "tensors": {
            "count": len(base_map),
            "name_sets_equal": True,
        },
        "shards": {
            "base": {
                name: dict(base_records[name]) for name in sorted(base_shards)},
            "candidate": {
                name: dict(candidate_records[name])
                for name in sorted(candidate_shards)},
            "download_order": order,
            "candidate_to_base": candidate_to_base_json,
            "base_to_candidate": base_to_candidate_json,
            **simulation,
        },
        "metadata": {
            "keep": metadata_keep,
            "replace": metadata_replace,
            # Tokenizers and model code frequently live only in the upstream
            # repository.  Preserve rather than silently deleting them.
            "preserve_base_only": preserved_base_only,
        },
    }


def _load_plan(model_dir: Path) -> dict[str, Any]:
    marker = model_dir / REPLACEMENT_MARKER_NAME
    plan = json.loads(marker.read_text())
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported checkpoint relayout marker: {marker}")
    if Path(plan.get("model_dir", "")).resolve() != model_dir.resolve():
        raise ValueError("relayout marker belongs to a different checkpoint")
    return plan


def _regular_file_with_size(path: Path, size: int) -> bool:
    return (
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == int(size)
    )


def audit(model_dir: Path, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify committed candidates and structural safety of remaining bases."""
    model_dir = model_dir.resolve()
    plan = plan or _load_plan(model_dir)
    candidate_states = {"candidate": 0, "missing": 0, "invalid": 0}
    base_states = {"base": 0, "missing": 0, "invalid": 0}
    candidate_ready: set[str] = set()
    invalid_paths = []

    for name, record in plan["shards"]["candidate"].items():
        path = model_dir / name
        if not path.exists():
            state = "missing"
        elif _regular_file_with_size(path, int(record["size"])):
            try:
                _verify(path, record)
            except ValueError:
                state = "invalid"
            else:
                state = "candidate"
                candidate_ready.add(name)
        else:
            state = "invalid"
        candidate_states[state] += 1
        if state == "invalid":
            invalid_paths.append(name)

    base_present: set[str] = set()
    for name, record in plan["shards"]["base"].items():
        path = model_dir / name
        if not path.exists():
            state = "missing"
        elif _regular_file_with_size(path, int(record["size"])):
            # Full base hashes are checked immediately before unlinking.  Size
            # classification keeps first-run audit from reading 755 GB twice.
            state = "base"
            base_present.add(name)
        else:
            state = "invalid"
        base_states[state] += 1
        if state == "invalid":
            invalid_paths.append(name)

    coverage_holes = []
    for base, outputs in plan["shards"]["base_to_candidate"].items():
        if base not in base_present and not set(outputs) <= candidate_ready:
            coverage_holes.append(base)

    metadata_states = {
        "base": 0, "candidate": 0, "missing_new": 0, "invalid": 0}
    metadata_path_states = {}
    for item in plan["metadata"]["replace"]:
        candidate = item["candidate"]
        name = candidate["path"]
        state = _classify_changed_path(
            model_dir / name, candidate, item.get("base"))
        metadata_states[state] += 1
        metadata_path_states[name] = state
        if state == "invalid":
            invalid_paths.append(name)

    return {
        "schema": AUDIT_SCHEMA,
        "model_dir": str(model_dir),
        "base": plan["base"],
        "candidate": plan["candidate"],
        "candidate_shards": candidate_states,
        "base_shards": base_states,
        "metadata": metadata_states,
        "metadata_path_states": metadata_path_states,
        "coverage_holes": coverage_holes,
        "invalid_paths": invalid_paths,
        "complete": (
            candidate_states["candidate"]
            == len(plan["shards"]["candidate"])
            and base_states["missing"] == len(plan["shards"]["base"])
            and metadata_states["candidate"]
            == len(plan["metadata"]["replace"])
            and not invalid_paths and not coverage_holes
        ),
    }


def _release_covered_bases(
    model_dir: Path, plan: dict[str, Any], candidate_ready: set[str],
    base_present: set[str],
) -> tuple[list[str], int]:
    released = []
    released_bytes = 0
    for base in sorted(tuple(base_present)):
        outputs = set(plan["shards"]["base_to_candidate"][base])
        if not outputs <= candidate_ready:
            continue
        path = model_dir / base
        record = plan["shards"]["base"][base]
        _verify_base(path, record)
        path.unlink()
        _fsync_directory(path.parent)
        base_present.remove(base)
        released.append(base)
        released_bytes += int(record["size"])
        print(json.dumps({
            "released_base_shard": base,
            "released_bytes_this_step": int(record["size"]),
        }), flush=True)
    return released, released_bytes


def _commit_download(
    *, model_dir: Path, plan: dict[str, Any], record: dict[str, Any],
    reserve_bytes: int, replace_base: dict[str, Any] | None = None,
) -> Path:
    target = model_dir / record["path"]
    if replace_base is not None:
        _verify_base(target, replace_base)
    elif target.exists():
        raise ValueError(f"candidate target already exists: {target}")
    required = 2 * int(record["size"])
    free = shutil.disk_usage(model_dir).free
    if free - required < int(reserve_bytes):
        raise RuntimeError(
            f"downloading {record['path']} may need {required} staging bytes "
            f"while preserving {reserve_bytes}; only {free} free")
    staging = model_dir / STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        downloaded = _download_one(
            repo=plan["candidate"]["repo"],
            revision=plan["candidate"]["revision"],
            record=record,
            staging=staging,
        )
        source_stat = downloaded.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded, target)
        _fsync_directory(target.parent)
        target_stat = target.stat()
        if (
            target_stat.st_dev != source_stat.st_dev
            or target_stat.st_ino != source_stat.st_ino
            or target_stat.st_size != int(record["size"])
        ):
            raise IOError(
                f"atomic relayout identity changed unexpectedly: {target}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def apply_plan(
    model_dir: Path, *, reserve_bytes: int = 10_000_000_000,
    max_files: int = 0,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    plan = _load_plan(model_dir)
    before = audit(model_dir, plan)
    if before["invalid_paths"]:
        raise ValueError(
            "relayout refuses unrecognized local bytes: "
            + ", ".join(before["invalid_paths"][:4]))
    if before["coverage_holes"]:
        raise RuntimeError(
            "relayout has uncovered deleted base shards: "
            + ", ".join(before["coverage_holes"][:4]))

    candidate_ready = {
        name for name in plan["shards"]["candidate"]
        if (model_dir / name).exists()}
    base_present = {
        name for name in plan["shards"]["base"]
        if (model_dir / name).exists()}
    released_paths, released_bytes = _release_covered_bases(
        model_dir, plan, candidate_ready, base_present)
    applied_paths = []
    applied_bytes = 0

    for name in plan["shards"]["download_order"]:
        if name in candidate_ready:
            continue
        record = plan["shards"]["candidate"][name]
        _commit_download(
            model_dir=model_dir, plan=plan, record=record,
            reserve_bytes=reserve_bytes)
        candidate_ready.add(name)
        applied_paths.append(name)
        applied_bytes += int(record["size"])
        print(json.dumps({
            "committed_candidate_shard": name,
            "files_this_run": len(applied_paths),
            "bytes_this_run": applied_bytes,
        }), flush=True)
        released, released_now = _release_covered_bases(
            model_dir, plan, candidate_ready, base_present)
        released_paths.extend(released)
        released_bytes += released_now
        if max_files and len(applied_paths) >= max_files:
            break

    # Serving metadata is swapped only after every candidate shard is present
    # and every old shard has been released.  The marker still blocks loading.
    if not max_files and len(candidate_ready) == len(
            plan["shards"]["candidate"]) and not base_present:
        metadata_states = before["metadata_path_states"]
        for item in plan["metadata"]["replace"]:
            record = item["candidate"]
            name = record["path"]
            state = metadata_states[name]
            if state == "candidate":
                continue
            target = model_dir / name
            if state == "base":
                replace_base = item["base"]
            elif state == "missing_new":
                if target.exists():
                    raise ValueError(f"metadata target appeared: {target}")
                replace_base = None
            else:
                raise ValueError(f"invalid metadata state for {name}: {state}")
            _commit_download(
                model_dir=model_dir, plan=plan, record=record,
                reserve_bytes=reserve_bytes, replace_base=replace_base)
            print(json.dumps({"committed_candidate_metadata": name}), flush=True)

    after = audit(model_dir, plan)
    after.update({
        "applied_files": len(applied_paths),
        "applied_bytes": applied_bytes,
        "released_base_files": len(released_paths),
        "released_base_bytes": released_bytes,
    })
    return after


def finalize(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    plan = _load_plan(model_dir)
    report = audit(model_dir, plan)
    if not report["complete"]:
        raise RuntimeError(
            "relayout replacement is incomplete: "
            f"candidate={report['candidate_shards']}, "
            f"base={report['base_shards']}, metadata={report['metadata']}")

    verified_bytes = 0
    for record in plan["metadata"]["keep"]:
        _verify_base(model_dir / record["path"], record)
        verified_bytes += int(record["size"])
    verified_bytes += sum(
        int(record["size"])
        for record in plan["shards"]["candidate"].values())
    verified_bytes += sum(
        int(item["candidate"]["size"])
        for item in plan["metadata"]["replace"])

    marker = model_dir / REPLACEMENT_MARKER_NAME
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "completed_unix": time.time(),
        "base": plan["base"],
        "candidate": plan["candidate"],
        "model_dir": str(model_dir),
        "tensor_count": plan["tensors"]["count"],
        "base_safetensor_shards": len(plan["shards"]["base"]),
        "safetensor_shards": len(plan["shards"]["candidate"]),
        "verified_candidate_bytes": verified_bytes,
        "preserved_base_only_files": [
            record["path"]
            for record in plan["metadata"]["preserve_base_only"]],
        "status": "verified",
    }
    _atomic_json(model_dir / REPLACEMENT_RECEIPT_NAME, receipt)
    marker.unlink()
    _fsync_directory(model_dir)
    return receipt


def _plan_from_hub(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    from huggingface_hub import HfApi

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"checkpoint does not exist: {model_dir}")
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
    for name in ("config.json", "model.safetensors.index.json"):
        _verify_base(model_dir / name, base_records[name])
    if not _same_record(
            base_records["config.json"], candidate_records["config.json"]):
        raise ValueError("relayout replacement requires identical config.json")

    staging = model_dir / STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    try:
        candidate_index_path = _download_one(
            repo=args.candidate_repo,
            revision=args.candidate_revision,
            record=candidate_records["model.safetensors.index.json"],
            staging=staging,
        )
        base_index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text())
        candidate_index = json.loads(candidate_index_path.read_text())
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return build_plan(
        base_repo=args.base_repo,
        base_revision=base_info.sha,
        base_records=base_records,
        base_index=base_index,
        candidate_repo=args.candidate_repo,
        candidate_revision=candidate_info.sha,
        candidate_records=candidate_records,
        candidate_index=candidate_index,
        model_dir=model_dir,
    ), model_dir


def plan_command(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.expanduser().resolve()
    marker = model_dir / REPLACEMENT_MARKER_NAME
    if marker.exists():
        if not args.resume:
            raise FileExistsError(f"replacement marker already exists: {marker}")
        plan = _load_plan(model_dir)
    else:
        plan, model_dir = _plan_from_hub(args)
        reserve_bytes = int(args.reserve_gb * 1_000_000_000)
        free = shutil.disk_usage(model_dir).free
        peak = int(plan["shards"]["peak_extra_bytes"])
        if free - peak < reserve_bytes:
            raise RuntimeError(
                f"planned shard peak needs {peak} extra bytes while preserving "
                f"{reserve_bytes}; only {free} free")
        if not args.dry_run:
            _atomic_json(marker, plan)
            _fsync_directory(model_dir)
    summary = {
        "schema": plan["schema"],
        "dry_run": bool(args.dry_run),
        "model_dir": plan["model_dir"],
        "base": plan["base"],
        "candidate": plan["candidate"],
        "tensor_count": plan["tensors"]["count"],
        "base_shards": len(plan["shards"]["base"]),
        "candidate_shards": len(plan["shards"]["candidate"]),
        "candidate_shard_bytes": plan["shards"]["candidate_shard_bytes"],
        "released_base_shard_bytes": plan["shards"][
            "released_base_shard_bytes"],
        "peak_extra_bytes": plan["shards"]["peak_extra_bytes"],
        "final_shard_delta_bytes": plan["shards"][
            "final_shard_delta_bytes"],
        "metadata_replacements": len(plan["metadata"]["replace"]),
        "preserved_base_only": len(plan["metadata"][
            "preserve_base_only"]),
        "marker_published": marker.exists(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base-repo", required=True)
    plan_parser.add_argument("--base-revision", required=True)
    plan_parser.add_argument("--candidate-repo", required=True)
    plan_parser.add_argument("--candidate-revision", required=True)
    plan_parser.add_argument("--model-dir", type=Path, required=True)
    plan_parser.add_argument("--reserve-gb", type=float, default=10.0)
    plan_parser.add_argument("--dry-run", action="store_true")
    plan_parser.add_argument("--resume", action="store_true")

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
        if args.reserve_gb < 0:
            parser.error("reserve must be non-negative")
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
