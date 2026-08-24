from __future__ import annotations

import numpy as np

from tests.fixtures.huihui_qwen38_head_candidate_recall import (
    evaluate_candidate_recall,
    main,
    synthetic_rank_fixture,
)


def test_synthetic_candidate_recall_has_known_rank_boundary():
    exact, approximate = synthetic_rank_fixture()
    report = evaluate_candidate_recall(
        exact, approximate, min_samples=1, real_capture=False)

    assert report["recall_at_k"]["1"] == 1 / 7
    assert report["recall_at_k"]["64"] == 6 / 7
    assert report["recall_at_k"]["128"] == 1.0
    assert report["exact_top1_approximate_rank"]["max"] == 65
    assert not report["gate"]["score_gate_passed"]
    assert not report["gate"]["promotion_ready"]


def test_legacy_array_scores_never_promote_even_when_labeled_real():
    exact = np.array([[3, 1, 0], [0, 1, 3]], dtype=np.float32)
    approximate = np.array([[2, 1, 0], [0, 1, 2]], dtype=np.float32)
    passing = evaluate_candidate_recall(
        exact, approximate, ks=(1, 2), gate_k=1,
        min_samples=2, real_capture=True)
    assert passing["gate"]["score_gate_passed"]
    assert passing["gate"]["sample_gate_passed"]
    assert not passing["gate"]["promotion_supported"]
    assert not passing["gate"]["promotion_ready"]
    undersampled = evaluate_candidate_recall(
        exact, approximate, ks=(1, 2), gate_k=1,
        min_samples=3, real_capture=True)
    assert not undersampled["gate"]["promotion_ready"]


def test_candidate_recall_cli_fixture_can_enforce_fail_closed(capsys):
    assert main([]) == 0
    assert '"real_capture": false' in capsys.readouterr().out
    assert main(["--enforce-promotion-gate"]) == 1


def test_unattested_paired_numpy_logits_cannot_promote(tmp_path):
    exact = tmp_path / "exact.npy"
    approximate = tmp_path / "approximate.npy"
    np.save(exact, np.array([[3, 2, 1]], dtype=np.float32))
    np.save(approximate, np.array([[3, 2, 1]], dtype=np.float32))
    assert main([
        "--exact-logits", str(exact),
        "--approximate-logits", str(approximate),
        "--min-samples", "1",
        "--ks", "1",
        "--gate-k", "1",
        "--enforce-promotion-gate",
    ]) == 1
