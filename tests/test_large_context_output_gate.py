"""The retrieval gate must not give the model its answers in the task suffix."""
from types import SimpleNamespace

import pytest

from tests.fixtures.qwen_large_context_output_gate import (
    CANARY_A, CANARY_B, _build_user_text, _response_text,
    _response_integrity_failures,
)


class CharacterTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(text.encode()))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


def test_retrieval_answers_occur_only_in_the_two_distant_records():
    text, tokens = _build_user_text(CharacterTokenizer(), 2100)
    assert tokens == 2100
    assert text.count(CANARY_A) == text.count(CANARY_B) == 1
    suffix = text.split("End of archive.", 1)[1]
    assert CANARY_A not in suffix and CANARY_B not in suffix
    assert "TARGET RECORD A" in suffix and "TARGET RECORD B" in suffix
    assert text.index(CANARY_A) < text.index(CANARY_B) < text.index("End of archive.")


def test_legacy_answer_revealing_body_requires_explicit_diagnostic_mode():
    text, tokens = _build_user_text(
        CharacterTokenizer(), 2100, legacy_copy_output_diagnostic=True)
    assert tokens == 2100
    assert text.count(CANARY_A) == text.count(CANARY_B) == 2
    assert f"Begin the answer with exactly: A={CANARY_A} B={CANARY_B}" in text


def test_visible_output_is_not_duplicated_or_taken_from_reasoning():
    response = {
        "output_text": "answer",
        "output": [
            {"type": "reasoning", "content": [{"type": "output_text", "text": "secret"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        ],
    }
    assert _response_text(response) == "answer"
    assert _response_text({"output_text": "fallback"}) == "fallback"


@pytest.mark.parametrize("status,reason,passed", [
    ("completed", None, True),
    ("incomplete", "max_output_tokens", True),
    ("incomplete", "content_filter", False),
    ("failed", None, False),
    (None, None, False),
])
def test_capped_diagnostic_still_requires_valid_termination(status, reason, passed):
    response = {"status": status, "incomplete_details": {"reason": reason},
                "vmodel_timing": {"true_peak_metal_bytes": 100}}
    assert (not _response_integrity_failures(response)) is passed


@pytest.mark.parametrize("peak", [None, 0, -1, True, "123", float("nan"), float("inf")])
def test_missing_or_invalid_peak_cannot_silently_pass_as_zero(peak):
    response = {"status": "completed",
                "vmodel_timing": {"true_peak_metal_bytes": peak}}
    assert "true peak Metal telemetry is missing or invalid" in _response_integrity_failures(response)


def test_protocol_error_or_unhandled_call_cannot_pass():
    response = {"status": "completed", "error": {"message": "failed"},
                "output": [{"type": "function_call", "name": "tool"}],
                "vmodel_timing": {"true_peak_metal_bytes": 100}}
    assert len(_response_integrity_failures(response)) == 2
