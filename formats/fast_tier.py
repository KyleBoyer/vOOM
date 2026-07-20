"""Bounded hot-expert staging for a second, faster local disk.

The primary vpack2 archive remains authoritative and complete.  This module
copies whole, lossless `.vt` expert pages from its sibling vpack store, or
reconstructs them from the verified vpack2 archive after intermediate cleanup,
into a small cache directory ranked by learned routing heat. WeightStore
authenticates each staged body against the vpack2 index before using it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import tempfile
import time
from collections import defaultdict
from pathlib import Path


_EXPERT = re.compile(
    r"^(?:model\.language_model\.|language_model\.model\.|model\.)"
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
DEFAULT_GLOBAL_BUDGET = 3_000_000_000


def _routing_heat(path: Path) -> dict[tuple[int, int], int]:
    heat: dict[tuple[int, int], int] = defaultdict(int)
    for key, count in json.loads(path.read_text()).items():
        layer, source, target = (int(value) for value in key.split(","))
        heat[(layer, source)] += int(count)
        heat[(layer + 1, target)] += int(count)
    return dict(heat)


def _expert_files(manifest: dict[str, str]) -> dict[tuple[int, int], list[str]]:
    groups: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    for name, filename in manifest.items():
        match = _EXPERT.match(name)
        if match is not None:
            groups[(int(match.group(1)), int(match.group(2)))].append(
                (name, filename))

    complete = {}
    projections = ("gate_proj", "up_proj", "down_proj")
    for key, entries in groups.items():
        names = {name for name, _filename in entries}
        if not all(any(name.endswith(f".{projection}.weight")
                       for name in names) for projection in projections):
            continue
        # Standard MLX checkpoints add a scales tensor (and sometimes biases)
        # beside every quantized projection. A hot expert must be atomic: never
        # stage only its packed weight and force the sidecars back to USB.
        quantized = any(name.endswith(".scales") for name in names)
        if quantized and not all(any(name.endswith(f".{projection}.scales")
                                     for name in names)
                                 for projection in projections):
            continue
        complete[key] = sorted(filename for _name, filename in entries)
    return complete


def _atomic_copy(source: Path, destination: Path) -> None:
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".",
        suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, handle, length=8 * 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_extract(reader, name: str, destination: Path) -> None:
    """Recreate one ordinary vpack file from an authenticated vpack2 body."""
    header, body = reader.read_body(name)
    encoded = json.dumps(header).encode()
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".",
        suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_cache_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    home = Path.home().resolve()
    if root in (Path("/").resolve(), home) or home not in root.parents:
        raise ValueError("fast tier must be a dedicated directory below the home directory")
    return root


def stage_hot_experts(
    model_dir: str | Path,
    fast_root: str | Path = "~/vmodel_fast_tier",
    *, budget_bytes: int = DEFAULT_GLOBAL_BUDGET,
) -> dict:
    """Populate one model namespace while enforcing a global cache budget."""
    model_dir = Path(model_dir).resolve()
    fast_root = _safe_cache_root(Path(fast_root))
    budget_bytes = int(budget_bytes)
    if budget_bytes <= 0 or budget_bytes > DEFAULT_GLOBAL_BUDGET:
        raise ValueError(
            f"fast-tier budget must be in (0, {DEFAULT_GLOBAL_BUDGET}]")
    vpack = model_dir / "weights.vpack"
    manifest_path = vpack / "manifest.json"
    heat_path = model_dir / "expert_transitions.json"
    if not manifest_path.exists():
        raise FileNotFoundError("lossless vpack manifest is required for staging")
    if not heat_path.exists():
        raise FileNotFoundError("expert routing heat is not available yet")
    manifest = json.loads(manifest_path.read_text())
    names_by_file = {filename: name for name, filename in manifest.items()}
    groups = _expert_files(manifest)
    heat = _routing_heat(heat_path)
    archive_reader = None

    def archived_reader():
        nonlocal archive_reader
        if archive_reader is None:
            from .packed2 import Vpack2Reader

            archive_reader = Vpack2Reader(model_dir, require_hashes=True)
        return archive_reader

    def staged_size(filename: str) -> int:
        source = vpack / filename
        if source.is_file():
            return source.stat().st_size
        name = names_by_file[filename]
        entry = archived_reader().index[name]
        encoded = json.dumps(entry["head"]).encode()
        return 8 + len(encoded) + int(entry["len"])

    ranked = sorted(
        groups, key=lambda key: (-heat.get(key, 0), key[0], key[1]))

    target = fast_root / model_dir.name
    target.mkdir(parents=True, exist_ok=True)
    desired: set[str] = set()
    desired_bytes = 0
    selected_experts = 0
    for key in ranked:
        if heat.get(key, 0) <= 0:
            continue
        files = groups[key]
        page_bytes = sum(staged_size(filename) for filename in files)
        if desired_bytes + page_bytes > budget_bytes:
            continue
        desired.update(files)
        desired_bytes += page_bytes
        selected_experts += 1

    removed_files = 0
    removed_bytes = 0
    # First discard obsolete files from this model's namespace.
    for path in target.glob("*.vt"):
        if path.name not in desired:
            size = path.stat().st_size
            path.unlink()
            removed_files += 1
            removed_bytes += size

    existing_desired = {
        path.name: path.stat().st_size
        for path in target.glob("*.vt") if path.name in desired
    }
    missing_bytes = sum(
        staged_size(filename)
        for filename in desired if filename not in existing_desired)
    current_files = [
        path for path in fast_root.rglob("*.vt") if path.is_file()
    ]
    current_bytes = sum(path.stat().st_size for path in current_files)
    # Cache copies are recoverable. Evict oldest files outside the selected set
    # one explicit file at a time; never recurse-delete a directory.
    protected = {target / filename for filename in desired}
    candidates = sorted(
        (path for path in current_files if path not in protected),
        key=lambda path: path.stat().st_mtime_ns,
    )
    while current_bytes + missing_bytes > budget_bytes and candidates:
        victim = candidates.pop(0)
        size = victim.stat().st_size
        victim.unlink()
        current_bytes -= size
        removed_files += 1
        removed_bytes += size
    if current_bytes + missing_bytes > budget_bytes:
        raise RuntimeError("selected hot pages cannot fit the fast-tier budget")

    copied_files = 0
    copied_bytes = 0
    for filename in sorted(desired):
        destination = target / filename
        source = vpack / filename
        expected_size = staged_size(filename)
        if destination.exists() and destination.stat().st_size == expected_size:
            os.utime(destination, None)
            continue
        if source.is_file():
            _atomic_copy(source, destination)
        else:
            _atomic_extract(
                archived_reader(), names_by_file[filename], destination)
        copied_files += 1
        copied_bytes += destination.stat().st_size

    total_bytes = sum(
        path.stat().st_size for path in fast_root.rglob("*.vt") if path.is_file())
    if total_bytes > budget_bytes:
        raise RuntimeError("fast-tier budget invariant was exceeded")
    report = {
        "model": model_dir.name,
        "created_at": int(time.time()),
        "budget_bytes": budget_bytes,
        "total_bytes": total_bytes,
        "selected_experts": selected_experts,
        "selected_files": len(desired),
        "selected_bytes": desired_bytes,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
    }
    metadata = target / "stage.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, metadata)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--fast-root", default="~/vmodel_fast_tier")
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_GLOBAL_BUDGET)
    args = parser.parse_args()
    report = stage_hot_experts(
        args.model_dir, args.fast_root, budget_bytes=args.budget_bytes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
