#!/usr/bin/env python3
"""Calibrate exact stochastic Qwen MTP proposal q from sparse replays.

The target distribution remains authoritative.  This tool changes only the
proposal distribution q and scores exact Leviathan overlap ``sum(min(p, q))``
on a held-out validation split.  It never loads model weights or starts Metal
work beyond importing the small runtime module.

Example:

  .venv/bin/python runtime/qwen_mtp_q_calibrate.py \
    --calibration calibration.jsonl --validation validation.jsonl \
    --target-sweep-bytes 28700000000 --draft-sweep-bytes 650000000 \
    --out proposal_q_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.qwen35_mtp import (  # noqa: E402
    ProposalQReplayRow,
    calibrate_proposal_q,
    default_proposal_q_policies,
)


def _records_from_json(value, source: Path) -> list[dict]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        if (
            "target_probabilities" in value
            and ("draft_probabilities" in value or "draft_logits" in value)
        ):
            records = [value]
        else:
            records = value.get("records")
        if records is None:
            records = value.get("qwen_mtp_proposal_q_replay")
        if records is None and isinstance(value.get("path_stats"), dict):
            records = value["path_stats"].get(
                "qwen_mtp_proposal_q_replay")
        if records is None and isinstance(value.get("timing"), dict):
            records = value["timing"].get(
                "qwen_mtp_proposal_q_replay")
        if records is None and isinstance(value.get("runs"), list):
            # The privacy-preserving real replay fixture stores serving
            # telemetry under runs[*].timing. Accept that durable archive
            # directly so calibration does not require an ad-hoc extraction
            # script (and can combine repeated/scenario runs safely).
            nested = []
            for run in value["runs"]:
                if not isinstance(run, dict):
                    continue
                for container_name in ("timing", "path_stats"):
                    container = run.get(container_name)
                    if not isinstance(container, dict):
                        continue
                    rows = container.get("qwen_mtp_proposal_q_replay")
                    if isinstance(rows, list):
                        nested.extend(rows)
            if nested:
                records = nested
        if records is None:
            raise ValueError(
                f"{source}: JSON object has no replay records field")
    else:
        raise ValueError(f"{source}: expected a JSON array or object")
    if not isinstance(records, list):
        raise ValueError(f"{source}: replay records must be a JSON array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{source}: every replay record must be an object")
    return records


def load_replay_rows(path: str | Path) -> tuple[ProposalQReplayRow, ...]:
    """Load a JSON array/result archive or one-record-per-line JSONL."""
    source = Path(path)
    raw = source.read_text()
    if not raw.strip():
        raise ValueError(f"{source}: replay file is empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{source}:{line_number}: replay row must be an object")
            records.append(record)
    else:
        records = _records_from_json(value, source)
    rows = tuple(ProposalQReplayRow.from_mapping(record) for record in records)
    if not rows:
        raise ValueError(f"{source}: replay file has no records")
    return rows


def _csv_numbers(raw: str, cast, label: str):
    try:
        values = tuple(cast(value.strip()) for value in raw.split(",")
                       if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated numeric list") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top-k", default="1,2,4,8,16")
    parser.add_argument(
        "--temperatures", default="0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--rank-powers", default="0.5,1,2")
    parser.add_argument("--target-sweep-bytes", type=int, default=0)
    parser.add_argument("--draft-sweep-bytes", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibration.resolve() == args.validation.resolve():
        raise SystemExit(
            "calibration and validation must be separate replay files")
    if args.target_sweep_bytes < 0 or args.draft_sweep_bytes < 0:
        raise SystemExit("sweep byte counts must be non-negative")
    try:
        top_ks = _csv_numbers(args.top_k, int, "top-k")
        temperatures = _csv_numbers(
            args.temperatures, float, "temperatures")
        rank_powers = _csv_numbers(
            args.rank_powers, float, "rank-powers")
        policies = default_proposal_q_policies(
            top_ks=top_ks,
            temperatures=temperatures,
            rank_powers=rank_powers,
        )
        calibration = load_replay_rows(args.calibration)
        validation = load_replay_rows(args.validation)
        report = calibrate_proposal_q(
            calibration,
            validation,
            policies=policies,
            target_sweep_bytes=args.target_sweep_bytes,
            draft_sweep_bytes=args.draft_sweep_bytes,
        )
    except (OSError, ValueError, argparse.ArgumentTypeError) as error:
        raise SystemExit(str(error)) from error
    report["sources"] = {
        "calibration": str(args.calibration),
        "calibration_rows": len(calibration),
        "validation": str(args.validation),
        "validation_rows": len(validation),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(rendered)
    else:
        temporary = args.out.with_name(
            f".{args.out.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(rendered)
            temporary.replace(args.out)
        finally:
            if temporary.exists():
                temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
