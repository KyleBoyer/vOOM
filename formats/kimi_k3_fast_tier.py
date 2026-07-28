"""F128: bounded fast-tier mirror of Kimi K3's always-touched tensors.

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

Raw byte-exact copies (no MLX/dtype involvement at all) straight from each
tensor's real safetensors data_offsets -- lossless by construction, not by
re-verification after the fact.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

_EXPERT_RE = re.compile(r"block_sparse_moe\.experts\.")


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
    """None means "leave on the slow tier" (shared_experts, top_level/
    vision/mm_projector -- either too large for the fast-tier budget or
    already handled by embed_rows/stream_lm_head, or simply unused by
    text-only generation)."""
    if _EXPERT_RE.search(name):
        return None
    if "shared_experts" in name:
        return None
    if ("embed_tokens" in name or "lm_head" in name
            or "vision_tower" in name or "mm_projector" in name):
        return None
    return "keep"


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def build_fast_tier(
    model_dir: str | Path, fast_root: str | Path, *, dry_run: bool = False,
) -> dict:
    model_dir = Path(model_dir).resolve()
    fast_root = Path(fast_root).expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    selected_names = [n for n in weight_map if _category(n) == "keep"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for n in selected_names:
        by_shard[weight_map[n]].append(n)

    target = fast_root / model_dir.name
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    total_bytes = 0
    file_index = 0

    for shard_i, (shard, names) in enumerate(sorted(by_shard.items())):
        shard_path = model_dir / shard
        header, data_start = _read_header(shard_path)
        with shard_path.open("rb") as src:
            for n in names:
                entry = header[n]
                start, end = entry["data_offsets"]
                nbytes = end - start
                total_bytes += nbytes
                if dry_run:
                    continue
                dest = target / f"{file_index:06d}.bin"
                src.seek(data_start + start)
                remaining = nbytes
                with dest.open("wb") as out:
                    while remaining:
                        chunk = src.read(min(remaining, 8 * 1024 * 1024))
                        if not chunk:
                            raise IOError(
                                f"truncated read for {n} in {shard_path}")
                        out.write(chunk)
                        remaining -= len(chunk)
                manifest[_canonical(n)] = {
                    "file": dest.name,
                    "nbytes": nbytes,
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                }
                file_index += 1
        print(
            f"[{shard_i + 1}/{len(by_shard)}] {shard}: "
            f"{len(names)} tensors, running total {total_bytes / 1e9:.2f} GB",
            file=sys.stderr, flush=True,
        )

    if not dry_run:
        manifest_path = target / "fast_tier_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

    return {
        "selected_tensors": len(selected_names),
        "total_bytes": total_bytes,
        "target": str(target),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--fast-root", default="~/vmodel_fast_tier")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = build_fast_tier(args.model_dir, args.fast_root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
