"""F128/F177: bounded raw-safetensors fast-tier mirror.

Unlike `formats/fast_tier.py` (which stages *predicted* hot experts ranked
by learned routing heat -- the same weak-locality class this project's own
F126 measurement already found regresses for this model family), this
mirrors a *deterministic* subset: the non-expert tensors every real
forward pass touches on every layer regardless of MoE routing
(self_attn/KDA projections, the Stable-LatentMoE routed_expert_down_proj/
up_proj/norm wrapper, the MoE gate, layer-0's dense MLP, norms, AttnRes
projections). These are never sometimes-needed the way routed experts are,
so there is no prediction risk -- only a real question of whether a second,
comparably-fast local disk can serve them concurrently with the main
external volume during a real fetch.

The byte copier is architecture-agnostic and is also useful for dense Qwen
checkpoints: with a partial, layer-balanced mirror, each layer fetch can read
from the internal and external NVMe devices concurrently. Raw byte-exact copies
(no MLX/dtype involvement at all) come straight from each tensor's real
safetensors data_offsets -- lossless by construction, not by re-verification
after the fact.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import sys
import uuid
from collections import defaultdict
from pathlib import Path

_EXPERT_RE = re.compile(r"(?:block_sparse_moe|mlp)\.experts\.")
_LAYER_RE = re.compile(
    r"(?:^|\.)(?:model\.language_model|language_model\.model|model)"
    r"\.layers\.(\d+)\."
)
MAX_INTERNAL_FAST_TIER_BYTES = 90_000_000_000
MIN_INTERNAL_FREE_BYTES = 10_000_000_000
MANIFEST_RESERVE_BYTES = 2_000_000


def _canonical(name: str) -> str:
    """Mirrors runtime.model_loader.WeightStore.__init__'s own
    language_model.-stripping exactly (model_loader.py, ~line 216-226) --
    the manifest's keys must match whatever fetch() looks names up as,
    not the raw index.json names."""
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model.model."):
        return "model." + name[len("language_model.model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def _category(name: str) -> str | None:
    """None means "leave on the slow tier" (routed experts, embedding/head,
    vision/mm_projector -- predicted rather than deterministic, already
    streamed specially, or unused by text-only generation).

    Shared experts are deliberately eligible.  They are deterministic
    always-used weights, not routed predictions; the old 3 GB policy excluded
    them only because they were too large for that budget.
    """
    if _EXPERT_RE.search(name):
        return None
    if ("embed_tokens" in name or "lm_head" in name or ".ple." in name
            or "vision_tower" in name or "mm_projector" in name
            or ".visual." in name or name.startswith("visual.")):
        return None
    return "keep"


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def _safetensors_prefix(entries: dict[str, dict]) -> bytes:
    """Build a standards-compatible header for unchanged selected payloads."""
    header = json.dumps(
        entries,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header


def _tree_file_bytes(root: Path, *, exclude: Path | None = None) -> int:
    """Return logical file bytes without following directory symlinks."""
    if not root.exists():
        return 0
    excluded = exclude.resolve() if exclude is not None and exclude.exists() else None
    total = 0
    for directory, dirnames, filenames in os.walk(root):
        current = Path(directory)
        if excluded is not None:
            dirnames[:] = [
                name
                for name in dirnames
                if (current / name).resolve() != excluded
            ]
        for filename in filenames:
            path = current / filename
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing parent for {path}")
        current = current.parent
    return current


def _is_internal_root(path: Path) -> bool:
    """Treat a target as internal when it resolves to the home volume."""
    try:
        return _existing_parent(path).stat().st_dev == Path.home().stat().st_dev
    except OSError:
        return False


def _select_budgeted(
    candidates: list[dict], budget_bytes: int,
) -> list[dict]:
    """Select a deterministic, layer-balanced, near-full subset.

    Parallel latency is paid per layer, so a global largest-first selection can
    put nearly all of one layer on the internal disk and very little of another.
    Rank each tensor by the cumulative fraction it would complete within its
    layer.  This water-fills every layer at approximately the same byte fraction
    while preferring larger tensors within a layer.  The packed-per-shard output
    below removes the per-tensor file-open cost that would otherwise favor a
    much less balanced plan.
    """
    budget_bytes = max(0, int(budget_bytes))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        match = _LAYER_RE.search(item["name"])
        group = f"layer:{match.group(1)}" if match is not None else "top"
        grouped[group].append(item)

    ranked: list[tuple[float, int, str, dict]] = []
    for group, items in grouped.items():
        total = sum(int(item["nbytes"]) for item in items)
        cumulative = 0
        for item in sorted(
            items, key=lambda candidate: (
                -int(candidate["nbytes"]), candidate["name"]
            )
        ):
            cumulative += int(item["nbytes"])
            ranked.append(
                (
                    cumulative / total,
                    -int(item["nbytes"]),
                    item["name"],
                    item,
                )
            )

    selected: list[dict] = []
    selected_names: set[str] = set()
    used = 0
    for _fraction, _negative_size, _name, item in sorted(ranked):
        size = int(item["nbytes"])
        if used + size <= budget_bytes:
            selected.append(item)
            selected_names.add(item["name"])
            used += size
    for item in sorted(
        (item for item in candidates if item["name"] not in selected_names),
        key=lambda candidate: (int(candidate["nbytes"]), candidate["name"]),
    ):
        size = int(item["nbytes"])
        if used + size <= budget_bytes:
            selected.append(item)
            selected_names.add(item["name"])
            used += size
    return sorted(selected, key=lambda item: (item["shard"], item["offset"]))


def build_fast_tier(
    model_dir: str | Path, fast_root: str | Path, *, dry_run: bool = False,
    max_bytes: int = MAX_INTERNAL_FAST_TIER_BYTES,
    min_free_bytes: int = MIN_INTERNAL_FREE_BYTES,
    container_format: str = "raw",
) -> dict:
    model_dir = Path(model_dir).resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    max_bytes = int(max_bytes)
    if max_bytes <= 0:
        raise ValueError("fast-tier max_bytes must be positive")
    min_free_bytes = max(0, int(min_free_bytes))
    if container_format not in ("raw", "safetensors"):
        raise ValueError("container_format must be raw or safetensors")
    internal_root = _is_internal_root(fast_root)
    if internal_root:
        if max_bytes > MAX_INTERNAL_FAST_TIER_BYTES:
            raise ValueError(
                "internal fast-tier max_bytes cannot exceed the repository's "
                f"{MAX_INTERNAL_FAST_TIER_BYTES}-byte policy ceiling")
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    candidate_names = [n for n in weight_map if _category(n) == "keep"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for n in candidate_names:
        by_shard[weight_map[n]].append(n)

    # Resolve the complete candidate catalog before creating a directory or
    # writing one byte.  The original F129 script streamed as it discovered
    # tensors and silently exceeded the then-current internal policy.
    plan: dict[str, tuple[dict, int]] = {}
    candidates: list[dict] = []
    for shard, names in sorted(by_shard.items()):
        shard_path = model_dir / shard
        header, data_start = _read_header(shard_path)
        plan[shard] = (header, data_start)
        for name in names:
            start, end = header[name]["data_offsets"]
            candidates.append({
                "name": name,
                "shard": shard,
                "offset": int(start),
                "nbytes": int(end - start),
            })

    target = fast_root / model_dir.name
    other_fast_tier_bytes = (
        _tree_file_bytes(fast_root, exclude=target)
        if internal_root else 0
    )
    available_budget = (
        max(
            0,
            max_bytes - other_fast_tier_bytes - MANIFEST_RESERVE_BYTES,
        )
        if internal_root else max_bytes
    )
    selected = _select_budgeted(candidates, available_budget)
    total_candidate_bytes = sum(int(item["nbytes"]) for item in candidates)
    selected_bytes = sum(int(item["nbytes"]) for item in selected)
    if not selected:
        raise ValueError(
            "no eligible deterministic tensor fits the available fast-tier budget; "
            "no files were written")
    projected_free = (
        shutil.disk_usage(_existing_parent(fast_root)).free
        - selected_bytes
        - MANIFEST_RESERVE_BYTES
    )
    if not dry_run and internal_root and projected_free < min_free_bytes:
        raise ValueError(
            "fast-tier plan would leave only "
            f"{projected_free} free bytes, below min_free_bytes="
            f"{min_free_bytes}; no files were written")
    if not dry_run and target.exists():
        raise FileExistsError(
            f"{target} already exists; refusing a non-transactional overwrite")

    selected_by_shard: dict[str, list[dict]] = defaultdict(list)
    for item in selected:
        selected_by_shard[item["shard"]].append(item)

    layer_candidate_bytes: dict[int, int] = defaultdict(int)
    layer_selected_bytes: dict[int, int] = defaultdict(int)
    for item in candidates:
        match = _LAYER_RE.search(item["name"])
        if match is not None:
            layer_candidate_bytes[int(match.group(1))] += int(item["nbytes"])
    for item in selected:
        match = _LAYER_RE.search(item["name"])
        if match is not None:
            layer_selected_bytes[int(match.group(1))] += int(item["nbytes"])
    layer_placement = {
        str(layer): {
            "candidate_bytes": layer_candidate_bytes[layer],
            "selected_bytes": layer_selected_bytes.get(layer, 0),
            "selected_fraction": (
                layer_selected_bytes.get(layer, 0)
                / layer_candidate_bytes[layer]
            ),
        }
        for layer in sorted(layer_candidate_bytes)
    }

    manifest: dict[str, dict] = {}
    file_index = 0
    staging = fast_root / f".{model_dir.name}.building-{uuid.uuid4().hex}"
    if not dry_run:
        fast_root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()

    copied_bytes = 0
    try:
        for shard_i, (shard, items) in enumerate(
            sorted(selected_by_shard.items())
        ):
            shard_path = model_dir / shard
            header, data_start = plan[shard]
            if dry_run:
                continue
            suffix = ".bin" if container_format == "raw" else ".safetensors"
            dest = staging / f"{file_index:06d}{suffix}"
            dest_offset = 0
            container_entries: dict[str, dict] = {}
            for item in items:
                entry = header[item["name"]]
                start, end = entry["data_offsets"]
                canonical = _canonical(item["name"])
                container_entries[canonical] = {
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "data_offsets": [
                        dest_offset,
                        dest_offset + int(end - start),
                    ],
                }
                dest_offset += int(end - start)
            prefix = (
                b""
                if container_format == "raw"
                else _safetensors_prefix(container_entries)
            )
            with shard_path.open("rb") as src, dest.open("wb") as out:
                out.write(prefix)
                dest_offset = 0
                for item in items:
                    name = item["name"]
                    entry = header[name]
                    start, end = entry["data_offsets"]
                    nbytes = end - start
                    src.seek(data_start + start)
                    remaining = nbytes
                    while remaining:
                        chunk = src.read(min(remaining, 8 * 1024 * 1024))
                        if not chunk:
                            raise IOError(
                                f"truncated read for {name} in {shard_path}")
                        out.write(chunk)
                        remaining -= len(chunk)
                    copied_bytes += nbytes
                    manifest[_canonical(name)] = {
                        "file": dest.name,
                        "offset": dest_offset,
                        "nbytes": nbytes,
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                    }
                    dest_offset += nbytes
            file_index += 1
            if not dry_run:
                print(
                    f"[{shard_i + 1}/{len(selected_by_shard)}] {shard}: "
                    f"{len(items)} tensors, copied "
                    f"{copied_bytes / 1e9:.2f} / "
                    f"{selected_bytes / 1e9:.2f} GB",
                    file=sys.stderr, flush=True,
                )

        if not dry_run:
            manifest_path = staging / "fast_tier_manifest.json"
            with manifest_path.open("w") as output:
                json.dump(manifest, output)
                output.flush()
                os.fsync(output.fileno())
            if internal_root:
                staged_bytes = _tree_file_bytes(staging)
                if other_fast_tier_bytes + staged_bytes > max_bytes:
                    raise RuntimeError(
                        "completed fast-tier generation exceeds the global "
                        "internal ceiling; refusing publication"
                    )
            os.replace(staging, target)
    except BaseException:
        if not dry_run:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema": "voom.kimi-k3-fast-tier-plan.v2",
        "candidate_tensors": len(candidates),
        "candidate_bytes": total_candidate_bytes,
        "selected_tensors": len(selected),
        "selected_bytes": selected_bytes,
        "selected_shared_expert_bytes": sum(
            int(item["nbytes"])
            for item in selected
            if "shared_experts" in item["name"]
        ),
        "excluded_tensors": len(candidates) - len(selected),
        "excluded_bytes": total_candidate_bytes - selected_bytes,
        "other_fast_tier_bytes": other_fast_tier_bytes,
        "projected_global_fast_tier_bytes": (
            other_fast_tier_bytes + selected_bytes + MANIFEST_RESERVE_BYTES
        ),
        "manifest_reserve_bytes": MANIFEST_RESERVE_BYTES,
        "max_bytes": max_bytes,
        "min_free_bytes": min_free_bytes,
        "projected_free_bytes": projected_free,
        "fits_budget": (
            (
                other_fast_tier_bytes
                + selected_bytes
                + MANIFEST_RESERVE_BYTES
                <= max_bytes
            )
            if internal_root else selected_bytes <= max_bytes
        ),
        "internal_root": internal_root,
        "container_format": container_format,
        "layer_placement": layer_placement,
        "target": str(target),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--fast-root", default="~/vmodel_fast_tier")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-bytes", type=int, default=MAX_INTERNAL_FAST_TIER_BYTES)
    parser.add_argument(
        "--min-free-bytes", type=int, default=MIN_INTERNAL_FREE_BYTES)
    parser.add_argument(
        "--container-format",
        choices=("raw", "safetensors"),
        default="raw",
    )
    args = parser.parse_args()
    report = build_fast_tier(
        args.model_dir, args.fast_root, dry_run=args.dry_run,
        max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes,
        container_format=args.container_format)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
