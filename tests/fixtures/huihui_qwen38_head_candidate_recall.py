#!/usr/bin/env python3
"""CPU-only exact-top1 recall gate for approximate LM-head shortlists.

Feed paired ``(samples, vocab)`` NumPy logits captured at the same hidden
states from the released BF16 and all-MXFP4 heads. This legacy driver measures whether
the exact BF16 winner is inside each approximate top-K shortlist; it does not
mistake agreement inside a shortlist for full-vocabulary recall.

Raw NumPy pairs have no live-origin, request-shape, privacy, or artifact-binding
attestation, so they are now diagnostic-only and permanently ineligible for
promotion. Use ``tests/fixtures/huihui_qwen38_head_rank_gate.py`` with the
runtime's bounded authoritative rank capture for promotion evidence. With no
inputs a deterministic synthetic rank fixture is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


DEFAULT_KS = (1, 8, 16, 32, 64, 128)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(row: np.ndarray, token: int) -> int:
    """One-based descending rank with vocabulary-ID tie breaking."""

    value = row[token]
    return 1 + int(np.count_nonzero(row > value)) + int(
        np.count_nonzero(row[:token] == value))


def evaluate_candidate_recall(
    exact_logits: np.ndarray,
    approximate_logits: np.ndarray,
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    gate_k: int = 64,
    required_recall: float = 1.0,
    min_samples: int = 1000,
    real_capture: bool = True,
) -> dict:
    """Evaluate exact-BF16 top-1 inclusion in approximate top-K.

    The stable rank avoids nondeterministic boundary behavior when logits tie.
    Real floating-point captures should almost never tie; reporting the rule
    keeps offline results reproducible across NumPy versions.
    """

    exact = np.asanyarray(exact_logits)
    approximate = np.asanyarray(approximate_logits)
    if exact.ndim != 2 or approximate.ndim != 2 or exact.shape != approximate.shape:
        raise ValueError(
            "exact and approximate logits must have the same rank-2 shape")
    samples, vocab = map(int, exact.shape)
    if samples <= 0 or vocab <= 1:
        raise ValueError("candidate-recall input must contain samples and a vocabulary")
    if not np.all(np.isfinite(exact)) or not np.all(np.isfinite(approximate)):
        raise ValueError("candidate-recall logits must be finite")
    requested_ks = tuple(sorted(set(int(k) for k in (*ks, gate_k))))
    if any(k <= 0 or k > vocab for k in requested_ks):
        raise ValueError(f"all K values must be in [1, {vocab}]")
    if not math.isfinite(required_recall) or not 0 <= required_recall <= 1:
        raise ValueError("required_recall must be finite and in [0, 1]")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    exact_ids = np.argmax(exact, axis=1)
    approximate_ids = np.argmax(approximate, axis=1)
    ranks = np.empty(samples, dtype=np.int64)
    for row_index in range(samples):
        ranks[row_index] = _stable_rank(
            approximate[row_index], int(exact_ids[row_index]))

    recalls = {
        str(k): float(np.count_nonzero(ranks <= k) / samples)
        for k in requested_ks
    }
    rank_percentiles = {
        "p50": float(np.percentile(ranks, 50)),
        "p95": float(np.percentile(ranks, 95)),
        "p99": float(np.percentile(ranks, 99)),
        "max": int(np.max(ranks)),
    }
    score_gate_passed = recalls[str(gate_k)] >= required_recall
    sample_gate_passed = samples >= min_samples
    # This legacy array evaluator has no request-shape manifest or artifact
    # bindings. It cannot establish promotion provenance even when a caller
    # labels its arrays "real"; only the authoritative rank gate can promote.
    promotion_ready = False
    return {
        "schema": "voom.huihui-lm-head-candidate-recall.v1",
        "samples": samples,
        "vocab": vocab,
        "rank_tie_break": "descending-logit-then-ascending-token-id",
        "exact_approx_top1_agreement": float(
            np.count_nonzero(exact_ids == approximate_ids) / samples),
        "recall_at_k": recalls,
        "exact_top1_approximate_rank": rank_percentiles,
        "gate": {
            "k": int(gate_k),
            "required_recall": float(required_recall),
            "min_samples": int(min_samples),
            "real_capture": bool(real_capture),
            "promotion_supported": False,
            "score_gate_passed": bool(score_gate_passed),
            "sample_gate_passed": bool(sample_gate_passed),
            "promotion_ready": promotion_ready,
        },
    }


def synthetic_rank_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Known exact winners at approximate ranks 1/2/8/16/32/64/65."""

    ranks = (1, 2, 8, 16, 32, 64, 65)
    vocab = 128
    approximate = np.tile(
        np.arange(vocab, 0, -1, dtype=np.float32), (len(ranks), 1))
    exact = np.zeros_like(approximate)
    for row, rank in enumerate(ranks):
        exact[row, rank - 1] = 1.0
    return exact, approximate


def _load_pair(exact_path: Path, approximate_path: Path):
    exact = np.load(exact_path, mmap_mode="r", allow_pickle=False)
    approximate = np.load(approximate_path, mmap_mode="r", allow_pickle=False)
    return exact, approximate


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-logits", type=Path)
    parser.add_argument("--approximate-logits", type=Path)
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--gate-k", type=int, default=64)
    parser.add_argument("--required-recall", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-promotion-gate", action="store_true",
        help="exit nonzero unless a real paired corpus clears every gate")
    args = parser.parse_args(argv)
    if bool(args.exact_logits) != bool(args.approximate_logits):
        parser.error("--exact-logits and --approximate-logits must be set together")
    ks = tuple(int(value) for value in args.ks.split(",") if value.strip())
    if args.exact_logits:
        exact, approximate = _load_pair(
            args.exact_logits, args.approximate_logits)
        # A file path is not provenance: arbitrary/synthetic arrays could be
        # supplied here. Promotion requires the live ranks-only manifest and
        # heterogeneous request-shape contract enforced by the new gate.
        real_capture = False
        sources = {
            "eligibility": "unattested-paired-logits-diagnostic-only",
            "exact": {
                "path": str(args.exact_logits.resolve()),
                "sha256": _sha256_file(args.exact_logits),
            },
            "approximate": {
                "path": str(args.approximate_logits.resolve()),
                "sha256": _sha256_file(args.approximate_logits),
            },
        }
    else:
        exact, approximate = synthetic_rank_fixture()
        real_capture = False
        sources = {"fixture": "synthetic-known-ranks-v1"}
    report = evaluate_candidate_recall(
        exact, approximate, ks=ks, gate_k=args.gate_k,
        required_recall=args.required_recall, min_samples=args.min_samples,
        real_capture=real_capture)
    report["sources"] = sources
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        _write_json_atomic(args.output, report)
    print(rendered)
    if args.enforce_promotion_gate and not report["gate"]["promotion_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
