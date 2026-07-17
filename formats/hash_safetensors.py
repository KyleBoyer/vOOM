#!/usr/bin/env python3
"""Create or verify a full-body SHA-256 manifest for raw safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.weight_integrity import build_manifest, verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if args.verify:
        digest = verify_manifest(model)
        report = {"verified": True, "manifest_sha256": digest}
    else:
        manifest = build_manifest(model)
        digest = verify_manifest(model)
        report = {
            "verified": True,
            "shards": len(manifest["files"]),
            "bytes": sum(entry["bytes"] for entry in manifest["files"].values()),
            "manifest_sha256": digest,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
