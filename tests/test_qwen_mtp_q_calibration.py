import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.qwen_mtp_q_calibrate import load_replay_rows, main
from runtime.qwen35_mtp import (
    ProposalQPolicy,
    ProposalQReplayRow,
    QwenMTPDrafter,
    QwenMTPSpeculativeEngine,
    _proposal_q_probabilities,
    calibrate_proposal_q,
    default_proposal_q_policies,
    evaluate_proposal_q,
    proposal_q_distribution,
    proposal_q_overlap,
    proposal_q_replay_record,
)


def _row(draft, target, *, score_kind="probabilities"):
    return ProposalQReplayRow(
        draft_token_ids=tuple(range(len(draft))),
        draft_values=tuple(draft),
        target_probabilities=tuple(target),
        score_kind=score_kind,
    )


def test_default_proposal_q_grid_covers_required_top_k_and_calibrations():
    policies = default_proposal_q_policies()

    assert {policy.top_k for policy in policies} == {1, 2, 4, 8, 16}
    assert any(policy.kind == "temperature" for policy in policies)
    assert any(policy.kind == "rank" for policy in policies)
    assert len({policy.name for policy in policies}) == len(policies)


def test_proposal_q_flat_overlap_and_temperature_distribution_are_exact():
    row = _row([0.6, 0.3, 0.1], [0.1, 0.6, 0.3])

    flat = ProposalQPolicy("flat", 2)
    assert proposal_q_distribution(row, flat) == pytest.approx(
        (0.5, 0.5, 0.0))
    assert proposal_q_overlap(row, flat) == pytest.approx(0.6)

    softened = proposal_q_distribution(
        row, ProposalQPolicy("temperature", 2, temperature=2.0))
    assert sum(softened) == pytest.approx(1.0)
    assert softened == pytest.approx(
        (0.5857864376, 0.4142135624, 0.0))

    ranked = proposal_q_distribution(
        row, ProposalQPolicy("rank", 2, rank_power=1.0))
    assert ranked == pytest.approx((2 / 3, 1 / 3, 0.0))


@pytest.mark.parametrize("policy", [
    ProposalQPolicy("flat", 2),
    ProposalQPolicy("temperature", 2, temperature=2.0),
    ProposalQPolicy("rank", 2, rank_power=1.0),
])
def test_runtime_q_policy_matches_offline_calibration_distribution(policy):
    calibrated = mx.array([0.6, 0.3, 0.1])
    row = _row([0.6, 0.3, 0.1], [0.1, 0.6, 0.3])

    runtime = _proposal_q_probabilities(calibrated, policy)
    mx.eval(runtime)

    assert runtime.tolist() == pytest.approx(
        proposal_q_distribution(row, policy))


def test_typed_runtime_q_policy_is_part_of_adapter_identity():
    target = SimpleNamespace(
        store=SimpleNamespace(names_with_prefix=lambda _prefix: ["mtp.x"]),
        cfg=SimpleNamespace(num_experts=0),
    )
    policy = ProposalQPolicy("rank", 8, rank_power=2.0)

    engine = QwenMTPSpeculativeEngine(
        target, proposal_q_policy=policy, depth=1)

    assert engine.proposal_q_policy is policy
    assert engine.stochastic_draft_top_k == 8
    assert engine.mtp_engine_identity == "qwen-mtp-depth1-rank-k8-p2"


def test_calibration_selection_never_uses_validation_outcome():
    calibration = [_row([0.9, 0.1], [0.2, 0.8])]
    validation = [_row([0.9, 0.1], [1.0, 0.0])]
    policies = (
        ProposalQPolicy("flat", 1),
        ProposalQPolicy("flat", 2),
    )

    report = calibrate_proposal_q(
        calibration, validation, policies=policies)

    assert report["selected"]["policy"]["name"] == "flat-k2"
    assert report["selected"]["calibration"][
        "expected_acceptance"] == pytest.approx(0.7)
    # Validation prefers flat-k1 (1.0 vs 0.5), proving it did not select.
    assert report["selected"]["validation"][
        "expected_acceptance"] == pytest.approx(0.5)
    assert report["validation_used_for_selection"] is False
    assert report["exact_target_distribution"] is True


def test_proposal_q_byte_projection_includes_draft_sweep_cost():
    metrics = evaluate_proposal_q(
        [_row([0.5, 0.5], [0.25, 0.25])],
        ProposalQPolicy("flat", 2),
        target_sweep_bytes=100,
        draft_sweep_bytes=10,
    )

    assert metrics["expected_acceptance"] == pytest.approx(0.5)
    assert metrics[
        "expected_emitted_tokens_per_target_sweep"] == pytest.approx(1.5)
    assert metrics["projected_bytes_per_output_token"] == pytest.approx(
        110 / 1.5)
    assert metrics["projected_byte_speedup_vs_plain_target"] == pytest.approx(
        100 / (110 / 1.5))


def test_sparse_replay_capture_preserves_rank_alignment_and_roundtrips():
    record = proposal_q_replay_record(
        mx.array([0.1, 0.6, 0.3, 0.0]),
        mx.array([0.4, 0.2, 0.3, 0.1]),
        max_rank=2,
        round_index=7,
    )

    assert record["draft_token_ids"] == [1, 2]
    assert record["draft_probabilities"] == pytest.approx([0.6, 0.3])
    assert record["target_probabilities"] == pytest.approx([0.2, 0.3])
    assert record["round_index"] == 7
    row = ProposalQReplayRow.from_mapping(record)
    assert row.draft_token_ids == (1, 2)


def test_round_local_weights_activate_only_for_validated_sidecar():
    class Store:
        def __init__(self, sidecar):
            self.mtplx_mtp_sidecar = sidecar
            self._mtplx_mtp_sidecar_layout = {
                "mtp.norm.weight": ("BF16", (4,), 8)}

        def names_with_prefix(self, prefix):
            assert prefix == "mtp."
            return ["mtp.norm.weight"]

    class Cache:
        def __init__(self):
            self.calls = 0
            self.discards = 0
            self.prepares = []
            self.total_bytes = 24
            self.weights = {
                "mtp.norm.weight": mx.ones((4,), dtype=mx.bfloat16)}

        def get(self, key, names, *, apply_transform=True):
            self.calls += 1
            assert key == "qwen35_mtp:released-bf16"
            assert names == ["mtp.norm.weight"]
            assert not apply_transform
            return self.weights

        def prepare_for(self, incoming_bytes):
            self.prepares.append(int(incoming_bytes))
            self.total_bytes = 16

        def discard(self, key, names):
            self.discards += 1
            assert key == "qwen35_mtp:released-bf16"
            assert names == ["mtp.norm.weight"]
            return True

    cache = Cache()
    drafter = QwenMTPDrafter(SimpleNamespace(
        store=Store("mtp.safetensors"), cache=cache))
    retained = drafter.prepare_request_weights()
    assert cache.prepares == [8]
    assert drafter.last_cache_prepare_released_bytes == 8
    assert retained is cache.weights
    assert cache.calls == 1
    release = drafter.release_request_weights(retained)
    assert retained == {}
    assert cache.discards == 1
    assert release == {"resident_bytes": 8, "cache_discarded": 1}

    no_sidecar_cache = Cache()
    drafter = QwenMTPDrafter(SimpleNamespace(
        store=Store(None), cache=no_sidecar_cache))
    assert drafter.prepare_request_weights() is None
    assert no_sidecar_cache.calls == 0


def test_calibration_cli_reads_jsonl_and_result_archive(tmp_path):
    calibration = tmp_path / "calibration.jsonl"
    validation = tmp_path / "validation.json"
    output = tmp_path / "report.json"
    calibration.write_text(json.dumps({
        "draft_token_ids": [1, 2],
        "draft_probabilities": [0.8, 0.2],
        "target_probabilities": [0.3, 0.6],
    }) + "\n")
    validation.write_text(json.dumps({"path_stats": {
        "qwen_mtp_proposal_q_replay": [{
            "draft_token_ids": [3, 4],
            "draft_probabilities": [0.6, 0.4],
            "target_probabilities": [0.5, 0.2],
        }]
    }}))

    assert len(load_replay_rows(calibration)) == 1
    assert main([
        "--calibration", str(calibration),
        "--validation", str(validation),
        "--target-sweep-bytes", "100",
        "--draft-sweep-bytes", "10",
        "--out", str(output),
    ]) == 0

    report = json.loads(output.read_text())
    assert report["sources"]["calibration_rows"] == 1
    assert report["sources"]["validation_rows"] == 1
    assert {candidate["policy"]["top_k"]
            for candidate in report["candidates"]} == {1, 2, 4, 8, 16}
    assert report["selected"]["validation"][
        "projected_bytes_per_output_token"] > 0


def test_calibration_reads_private_replay_fixture_timing_runs(tmp_path):
    archive = tmp_path / "fixture-result.json"
    row = {
        "draft_token_ids": [7, 9],
        "draft_probabilities": [0.7, 0.3],
        "target_probabilities": [0.2, 0.6],
    }
    archive.write_text(json.dumps({
        "passed": True,
        "runs": [
            {"timing": {"qwen_mtp_proposal_q_replay": [row]}},
            {"timing": None},
        ],
    }))

    loaded = load_replay_rows(archive)
    assert len(loaded) == 1
    assert loaded[0].draft_token_ids == (7, 9)
