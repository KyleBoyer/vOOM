#!/usr/bin/env python3
"""Fail-closed offline K=64 gate for live authoritative Huihui head ranks.

Example after accumulating heterogeneous real traffic under the explicit
capture environment:

  .venv/bin/python tests/fixtures/huihui_qwen38_head_rank_gate.py \
    logs/huihui_rerank64_authoritative_ranks.jsonl \
    --expected-exact-fingerprint 02c9ed... \
    --expected-approximate-fingerprint <reported-by-server-start> \
    --enforce-promotion-gate \
    --output logs/huihui_rerank64_authoritative_rank_gate.json

Thresholds are intentionally not CLI-tunable: promotion always means K=64,
at least 1,000 authoritative target positions, and 100% candidate recall over
the required heterogeneous live request shapes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.lm_head_recall_capture import (
    evaluate_rank_captures,
    quantized_lm_head_artifact_identity,
)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--expected-exact-fingerprint", default="")
    parser.add_argument("--expected-approximate-fingerprint", default="")
    parser.add_argument(
        "--approximate-model-dir", type=Path,
        help="independently content-hash the current MXFP4 LM head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-promotion-gate", action="store_true")
    args = parser.parse_args(argv)
    if args.approximate_model_dir and args.expected_approximate_fingerprint:
        parser.error(
            "set --approximate-model-dir or "
            "--expected-approximate-fingerprint, not both")
    approximate_identity = (
        quantized_lm_head_artifact_identity(args.approximate_model_dir)
        if args.approximate_model_dir else None)
    report = evaluate_rank_captures(
        args.captures,
        expected_exact_fingerprint=args.expected_exact_fingerprint,
        expected_approximate_fingerprint=(
            approximate_identity["fingerprint"]
            if approximate_identity is not None
            else args.expected_approximate_fingerprint),
    )
    if approximate_identity is not None:
        report["offline_approximate_artifact_identity"] = {
            "fingerprint": approximate_identity["fingerprint"],
            "bytes": approximate_identity["bytes"],
        }
    if args.output:
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.enforce_promotion_gate
               and not report["gate"]["promotion_ready"])


if __name__ == "__main__":
    raise SystemExit(main())
