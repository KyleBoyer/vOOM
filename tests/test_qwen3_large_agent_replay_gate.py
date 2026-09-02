from tests.fixtures.qwen3_large_agent_replay_gate import (
    _parse_sse_comment_progress,
    _parse_sse_comment_retry_metadata,
)


def test_parse_privacy_safe_progress_comment():
    assert _parse_sse_comment_progress(": prefill_layer 17/45") == (
        "prefill_layer", 17, 45)
    assert _parse_sse_comment_progress(": vision 3/8") == (
        "vision", 3, 8)
    assert _parse_sse_comment_progress(": memory_retry 1/4") == (
        "memory_retry", 1, 4)


def test_parse_memory_retry_progress_comment_with_diagnostic_suffix():
    line = (
        ": memory_retry 1/5 retry_reason=hard_metal_cap "
        "retry_subphase=attention_tile retry_layer=3 "
        "retry_completed_tokens=24160 "
        "retry_observed_metal_bytes=8501319252 "
        "retry_metal_limit_bytes=8500000000 retry_chunk=8")
    assert _parse_sse_comment_progress(line) == ("memory_retry", 1, 5)
    assert _parse_sse_comment_retry_metadata(line) == {
        "retry_reason": "hard_metal_cap",
        "retry_subphase": "attention_tile",
        "retry_layer": 3,
        "retry_completed_tokens": 24160,
        "retry_observed_metal_bytes": 8501319252,
        "retry_metal_limit_bytes": 8500000000,
        "retry_chunk": 8,
    }


def test_reject_non_progress_or_invalid_comment():
    for value in (
        "data: {}",
        ": keepalive",
        ": secret_phase 1/2",
        ": prefill one/two",
        ": prefill -1/2",
        ": prefill 3/2",
        ": prefill 0/0",
    ):
        assert _parse_sse_comment_progress(value) is None
