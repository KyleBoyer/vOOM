"""Shadow-only observation: tiny controller/MLX gates, not model performance."""

import json
import inspect
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from runtime.qwen4_mtp import Qwen4MTPSpeculativeEngine
from runtime.sampler import SamplingParams
from tests.test_qwen4_mtp import (
    _FakeDrafter, _FakeTarget, _FakeTokenizer, _cache_io_noop,
)


OUTPUT = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 255]


class _Prompt(str):
    def __new__(cls, namespace):
        value = super().__new__(cls, "unrelated prompt and tools")
        value.cache_namespace = namespace
        return value


class _SequenceTarget(_FakeTarget):
    def forward_tokens_serial_positions(self, tokens, kv, **kwargs):
        start = kv.offset - len(self.tokenizer.ids) + 1
        self.target_rows = OUTPUT[start:start + len(tokens)]
        return super().forward_tokens_serial_positions(tokens, kv, **kwargs)


def _engine(shadow):
    target = _SequenceTarget([])
    # Even a prompt containing the entire answer must not seed output history.
    target.tokenizer = _FakeTokenizer(OUTPUT)
    drafter = _FakeDrafter([11, 12, 14, 15, 17, 18, 20, 21])
    return Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=drafter, completed_history_shadow=shadow)


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("streaming", [False, True])
def test_shadow_observes_completed_repeat_without_changing_tokens_state_or_rng(
        _cache_io_noop, temperature, streaming):
    arms = []
    for enabled in (False, True):
        engine = _engine(enabled)
        mx.random.seed(64008)
        results, streams = [], []
        for _ in range(2):
            chunks = []
            results.append(engine.generate(
                _Prompt("tenant-a"), max_tokens=32,
                sampling=SamplingParams(temperature=temperature),
                on_token=chunks.append if streaming else None))
            streams.append("".join(chunks))
        arms.append((engine, results, streams, mx.random.uniform().item()))

    control, candidate = arms
    assert control[3] == candidate[3]  # The observer consumes no RNG draws.
    assert control[0].target.verify_calls == candidate[0].target.verify_calls
    assert control[0].drafter.proposal_steps == candidate[0].drafter.proposal_steps
    assert control[0].target.last_kv.layer_lengths() == (
        candidate[0].target.last_kv.layer_lengths())
    assert control[0].target.last_kv.qwen4_cache.restores == (
        candidate[0].target.last_kv.qwen4_cache.restores)
    np.testing.assert_array_equal(
        np.array(control[0].target._h_last.astype(mx.float32)),
        np.array(candidate[0].target._h_last.astype(mx.float32)))
    for left, right in zip(control[1], candidate[1], strict=True):
        for key in ("tokens", "text", "termination_reason", "kv_positions"):
            assert left[key] == right[key]
        assert right["tokens"] == OUTPUT
        assert "qwen4_completed_history_shadow" not in left["path_stats"]
        assert left["path_stats"]["qwen4_mtp_round_widths"] == (
            right["path_stats"]["qwen4_mtp_round_widths"])
        assert right["path_stats"]["qwen4_mtp_proposal_sources"] == "M,M,M,M"
    assert control[2] == candidate[2]
    cold, repeat = [r["path_stats"]["qwen4_completed_history_shadow"]
                    for r in candidate[1]]
    assert cold["candidate_rounds"] == 0
    assert cold["completed_output_added"]
    assert repeat["candidate_rounds"] == 3
    assert repeat["observation_only"]
    assert not repeat["stochastic_acceptance_proof"]
    assert repeat["skipped_rounds"] == 0
    for row in repeat["records"]:
        assert row["matching_prefix_tokens"] == row["proposed_tokens"]
        assert row["unobserved_tokens"] == 0
        assert set(row) == {"output_offset", "match_length", "proposed_tokens",
                            "observed_tokens", "matching_prefix_tokens",
                            "unobserved_tokens"}
    # The serialized observer reports counts, never cached content or namespace.
    assert "tenant-a" not in json.dumps(repeat)


def test_shadow_history_is_engine_local_namespaced_and_cleared_on_close(
        _cache_io_noop):
    engine = _engine(True)
    first = engine.generate(_Prompt("a"), max_tokens=32)
    assert first["path_stats"]["qwen4_completed_history_shadow"][
        "completed_output_added"]
    other = engine.generate(_Prompt("b"), max_tokens=32)
    assert other["path_stats"]["qwen4_completed_history_shadow"][
        "candidate_rounds"] == 0
    separate = _engine(True).generate(_Prompt("a"), max_tokens=32)
    assert separate["path_stats"]["qwen4_completed_history_shadow"][
        "candidate_rounds"] == 0
    retained = engine._completed_output_history
    engine.close()
    assert engine._completed_output_history is None
    assert not retained.propose(
        "a", OUTPUT[:4], max_tokens=7).tokens


def test_shadow_does_not_admit_output_caps_or_early_fallbacks(_cache_io_noop):
    engine = _engine(True)
    capped = engine.generate(_Prompt("a"), max_tokens=7)
    assert capped["termination_reason"] == "length"
    assert not capped["path_stats"]["qwen4_completed_history_shadow"][
        "completed_output_added"]
    assert not engine._completed_output_history.propose(
        "a", OUTPUT[:4], max_tokens=2).tokens
    fallback = engine.generate(_Prompt("a"), max_tokens=1)
    assert "qwen4_completed_history_shadow" not in fallback["path_stats"]


@pytest.mark.parametrize("failure", ["propose", "add_output"])
def test_shadow_failure_cannot_change_completed_generation(
        monkeypatch, _cache_io_noop, failure):
    engine = _engine(True)

    def fail(*_args, **_kwargs):
        raise RuntimeError("private content must not be echoed")

    monkeypatch.setattr(engine._completed_output_history, failure, fail)
    result = engine.generate(_Prompt("a"), max_tokens=32)
    assert result["tokens"] == OUTPUT
    witness = result["path_stats"]["qwen4_completed_history_shadow"]
    assert not witness["available"]
    assert witness["error_type"] == "RuntimeError"
    assert "private" not in json.dumps(witness)


def test_shadow_record_count_is_bounded_on_long_output(_cache_io_noop):
    target = _FakeTarget([11, 12, 13])
    target.effective_max_position_embeddings = 1024
    engine = Qwen4MTPSpeculativeEngine(
        target, depth=2, drafter=_FakeDrafter([11, 12]),
        completed_history_shadow=True)
    result = engine.generate(_Prompt("a"), max_tokens=404)
    witness = result["path_stats"]["qwen4_completed_history_shadow"]
    assert len(witness["records"]) == witness["record_limit"] == 128
    assert witness["skipped_rounds"] == (
        result["path_stats"]["qwen4_mtp_target_sweeps"] - 128)
    assert witness["skipped_rounds"] > 0
    assert not witness["completed_output_added"]


def test_shadow_flag_requires_bool(_cache_io_noop):
    with pytest.raises(TypeError, match="completed_history_shadow"):
        _engine(1)


def test_shadow_telemetry_survives_protocol_boundary_without_default_payload():
    from runtime.server import _vision_protocol_timing

    witness = {"schema": "voom.completed-history-shadow.v1",
               "observation_only": True, "candidate_rounds": 3,
               "records": [{"output_offset": 4, "proposed_tokens": 7}]}
    result = {"path_stats": {"qwen4_completed_history_shadow": witness}}
    assert _vision_protocol_timing(result)["qwen4_completed_history_shadow"] == witness
    assert "qwen4_completed_history_shadow" not in _vision_protocol_timing({})


@pytest.mark.parametrize("failure", ["clock", "clear", "stats"])
def test_shadow_final_diagnostic_failures_cannot_escape(monkeypatch, failure):
    from runtime import qwen4_mtp

    engine = _engine(True)
    result = {"tokens": OUTPUT, "termination_reason": "eos", "path_stats": {}}

    def fail(*_args, **_kwargs):
        raise RuntimeError("diagnostic failure")

    if failure == "clock":
        monkeypatch.setattr(qwen4_mtp, "time", SimpleNamespace(perf_counter=fail))
    elif failure == "clear":
        monkeypatch.setattr(engine._completed_output_history, "clear", fail)
    else:
        class RefusingStats(dict):
            def __setitem__(self, _key, _value):
                fail()
        result["path_stats"] = RefusingStats()
    engine._finish_completed_history_shadow(
        _Prompt("a"), result, [], 0.0, 0,
        "LookupError" if failure == "clear" else None)
    assert result["tokens"] == OUTPUT
    if failure == "clear":
        assert engine._completed_output_history is None


def test_shadow_cleanup_failure_cannot_prevent_target_close(monkeypatch):
    engine = _engine(True)
    closed = []

    def fail():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(engine._completed_output_history, "clear", fail)
    monkeypatch.setattr(engine.target, "close", lambda: closed.append(True))
    engine.close()
    assert closed == [True]
    assert engine._completed_output_history is None


@pytest.mark.parametrize("clock_read", [1, 2])
def test_shadow_lookup_clock_failure_does_not_interrupt_generation(
        monkeypatch, _cache_io_noop, clock_read):
    from runtime import qwen4_mtp

    real_clock = qwen4_mtp.time.perf_counter
    reads = 0

    def clock():
        nonlocal reads
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "generate" and frame.f_locals.get("rounds") == 1:
            reads += 1
            if reads == clock_read:
                raise RuntimeError("shadow clock unavailable")
        return real_clock()

    monkeypatch.setattr(qwen4_mtp, "time", SimpleNamespace(perf_counter=clock))
    result = _engine(True).generate(_Prompt("a"), max_tokens=32)
    assert reads >= clock_read
    assert result["tokens"] == OUTPUT
    assert result["path_stats"]["qwen4_completed_history_shadow"]["available"] is False
