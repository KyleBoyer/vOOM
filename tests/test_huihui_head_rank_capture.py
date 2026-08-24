from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from tests.fixtures.huihui_qwen38_head_rank_gate import main as gate_main
from runtime.lm_head_recall_capture import (
    AuthoritativeRankCapture,
    LIVE_CAPTURE_KIND,
    SYNTHETIC_CAPTURE_KIND,
    evaluate_rank_captures,
    privacy_safe_request_shape,
    quantized_lm_head_artifact_identity,
)
from runtime.lm_head_stream import StreamedLMHead
from runtime.quant import (
    QTensor,
    make_row_paged_reranked_q_head,
    matmul,
    reranked_lm_head_capture_scope,
)


EXACT_FINGERPRINT = "a" * 64
APPROXIMATE_FINGERPRINT = "b" * 64


def _shape(index: int) -> dict:
    return privacy_safe_request_shape(
        prompt_tokens=(128, 700, 3000, 10_000)[index % 4],
        system_chars=(0, 500, 2_000, 10_000)[index % 4],
        tool_count=(0, 2, 16, 64, 134)[index % 5],
        message_count=(1, 3, 8, 20)[index % 4],
        developer=index >= 4,
        streaming=index % 2 == 0,
        temperature_class="greedy" if index % 2 == 0 else "stochastic",
        constrained=index % 3 == 0,
    )


def _write_thousand_position_fixture(path, *, miss: bool = False):
    writer = AuthoritativeRankCapture(
        path,
        exact_source_fingerprint=EXACT_FINGERPRINT,
        approximate_artifact_fingerprint=APPROXIMATE_FINGERPRINT,
        approximate_artifact_bytes=123_456,
        candidates=64,
        vocab=248_320,
        max_positions=1000,
        max_positions_per_request=128,
        capture_kind=SYNTHETIC_CAPTURE_KIND,
    )
    for request in range(8):
        ranks = [1 + (request % 32)] * 125
        hits = [True] * 125
        if miss and request == 7:
            ranks[-1] = 65
            hits[-1] = False
        with writer.request(_shape(request)):
            assert writer.record(ranks, hits, [True] * 125) == 125
    return writer


def test_synthetic_rank_fixture_can_clear_math_but_never_promote(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    _write_thousand_position_fixture(path)

    report = evaluate_rank_captures(
        [path],
        expected_exact_fingerprint=EXACT_FINGERPRINT,
        expected_approximate_fingerprint=APPROXIMATE_FINGERPRINT,
    )

    assert report["positions"] == 1000
    assert report["requests"] == 8
    assert report["distinct_shapes"] == 8
    assert report["actual_candidate_recall"] == 1.0
    assert report["gate"]["heterogeneous_real_request_shapes"]
    assert report["gate"]["explicit_source_binding"]
    assert not report["gate"]["live_authoritative_capture"]
    assert not report["gate"]["promotion_ready"]
    assert gate_main([
        str(path),
        "--expected-exact-fingerprint", EXACT_FINGERPRINT,
        "--expected-approximate-fingerprint", APPROXIMATE_FINGERPRINT,
        "--enforce-promotion-gate",
    ]) == 1


def test_one_rank_miss_fails_both_actual_and_stable_recall(tmp_path):
    path = tmp_path / "one-miss.jsonl"
    _write_thousand_position_fixture(path, miss=True)
    report = evaluate_rank_captures(
        [path],
        expected_exact_fingerprint=EXACT_FINGERPRINT,
        expected_approximate_fingerprint=APPROXIMATE_FINGERPRINT,
    )
    assert report["positions"] == 1000
    assert report["actual_candidate_recall"] == 0.999
    assert not report["gate"]["actual_candidate_recall_100_percent"]
    assert not report["gate"]["stable_rank_recall_100_percent"]
    assert not report["gate"]["promotion_ready"]


def test_rank_capture_rejects_schema_fields_that_could_hold_content(tmp_path):
    path = tmp_path / "strict.jsonl"
    writer = AuthoritativeRankCapture(
        path,
        exact_source_fingerprint=EXACT_FINGERPRINT,
        approximate_artifact_fingerprint=APPROXIMATE_FINGERPRINT,
        approximate_artifact_bytes=123,
        candidates=64,
        vocab=128,
        max_positions=1000,
        capture_kind=SYNTHETIC_CAPTURE_KIND,
    )
    lines = path.read_text().splitlines()
    manifest = json.loads(lines[0])
    manifest["prompt"] = "must never be accepted"
    path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="manifest"):
        evaluate_rank_captures([path])
    assert writer.telemetry_snapshot()["positions"] == 0


def test_one_shape_cannot_exhaust_the_bounded_corpus(tmp_path):
    writer = AuthoritativeRankCapture(
        tmp_path / "shape-bound.jsonl",
        exact_source_fingerprint=EXACT_FINGERPRINT,
        approximate_artifact_fingerprint=APPROXIMATE_FINGERPRINT,
        approximate_artifact_bytes=123,
        candidates=64,
        vocab=128,
        max_positions=1000,
        max_positions_per_request=128,
        capture_kind=SYNTHETIC_CAPTURE_KIND,
    )
    for expected in (125, 75, 0):
        with writer.request(_shape(0)):
            assert writer.record(
                [1] * 125, [True] * 125, [True] * 125) == expected
    assert writer.telemetry_snapshot()["positions"] == 200


def test_quantized_head_identity_hashes_weight_and_scale_content(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, delta in ((first, 0), (second, 1)):
        weight = mx.arange(64, dtype=mx.uint32).reshape(8, 8)
        if delta:
            weight = weight.at[0, 0].add(mx.array(1, dtype=mx.uint32))
        scales = mx.arange(16, dtype=mx.uint8).reshape(8, 2)
        mx.save_safetensors(str(directory / "model.safetensors"), {
            "lm_head.weight": weight,
            "lm_head.scales": scales,
        })

    first_identity = quantized_lm_head_artifact_identity(first)
    repeated = quantized_lm_head_artifact_identity(first)
    second_identity = quantized_lm_head_artifact_identity(second)
    assert first_identity == repeated
    assert first_identity["fingerprint"] != second_identity["fingerprint"]
    assert first_identity["bytes"] == weight.nbytes + scales.nbytes


def test_live_projection_capture_records_only_authoritative_target_scope(
        tmp_path):
    vocab, hidden = 65, 32
    exact_values = np.sin(
        np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden) / 11.0)
    exact = mx.array(exact_values).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {
        "lm_head.weight": exact,
    })
    provider = StreamedLMHead(
        tmp_path, {"lm_head.weight": "model.safetensors"}, block_rows=17)
    packed = mx.quantize(exact, group_size=32, bits=4, mode="mxfp4")
    approximate = QTensor(
        packed[0], packed[1], packed[2] if len(packed) > 2 else None,
        4, 32, "mxfp4")
    head = make_row_paged_reranked_q_head(
        approximate, provider, candidates=64, recall_probe_every=0)
    capture_path = tmp_path / "ranks.jsonl"
    head.recall_rank_capture = AuthoritativeRankCapture(
        capture_path,
        exact_source_fingerprint=EXACT_FINGERPRINT,
        approximate_artifact_fingerprint=APPROXIMATE_FINGERPRINT,
        approximate_artifact_bytes=approximate.nbytes,
        candidates=64,
        vocab=vocab,
        max_positions=1000,
        capture_kind=SYNTHETIC_CAPTURE_KIND,
    )
    hidden_state = mx.array(
        np.cos(np.arange(hidden, dtype=np.float32) / 7.0)
    ).reshape(1, 1, hidden).astype(mx.bfloat16)

    try:
        with head.recall_rank_capture.request(_shape(0)):
            with reranked_lm_head_capture_scope(
                    head, "constraint-provisional"):
                provisional_logits = matmul(hidden_state, head)
                mx.eval(provisional_logits)
            with reranked_lm_head_capture_scope(
                    head, "authoritative-target"):
                target_logits = matmul(hidden_state, head)
                mx.eval(target_logits)
            with reranked_lm_head_capture_scope(head, "mtp-draft"):
                draft_logits = matmul(hidden_state, head)
                mx.eval(draft_logits)
        telemetry = head.telemetry_snapshot()
        report = evaluate_rank_captures([capture_path])
        assert telemetry["calls"] == 3
        assert telemetry["positions"] == 3
        assert telemetry["candidate_rank_capture_positions"] == 1
        assert telemetry["candidate_recall_full_scan_calls"] == 1
        assert report["positions"] == 1
        assert not report["gate"]["live_authoritative_capture"]
        assert not report["gate"]["promotion_ready"]
    finally:
        provider.close()


def test_live_kind_is_a_distinct_non_synthetic_manifest_value():
    assert LIVE_CAPTURE_KIND != SYNTHETIC_CAPTURE_KIND
