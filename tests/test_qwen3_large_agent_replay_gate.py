from tests.fixtures.qwen3_large_agent_replay_gate import (
    _parse_sse_comment_progress,
)


def test_parse_privacy_safe_progress_comment():
    assert _parse_sse_comment_progress(": prefill_layer 17/45") == (
        "prefill_layer", 17, 45)
    assert _parse_sse_comment_progress(": vision 3/8") == (
        "vision", 3, 8)


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
