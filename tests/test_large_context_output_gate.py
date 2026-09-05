"""The retrieval gate must not give the model its answers in the task suffix."""
from types import SimpleNamespace

from tests.fixtures.qwen_large_context_output_gate import (
    CANARY_A, CANARY_B, _build_user_text,
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
