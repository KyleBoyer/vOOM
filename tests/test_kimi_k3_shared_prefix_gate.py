from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.kimi_k3_shared_prefix_gate import (
    _fallback_static_stem,
    _leading_static_request,
    _longest_common_prefix,
    _tokenizer_safe_static_prefix,
)


def test_leading_static_request_excludes_all_user_and_history_content():
    request = {
        "model": "Kimi-K3",
        "input": [
            {"role": "system", "content": "stable"},
            {"role": "developer", "content": "also stable"},
            {"role": "user", "content": "never seed this"},
            {"role": "assistant", "content": "or this"},
        ],
    }

    static, leading, dynamic = _leading_static_request(request)

    assert static["input"] == leading == request["input"][:2]
    assert dynamic == request["input"][2:]
    assert request["input"][2]["content"] not in str(static)


def test_tokenizer_safe_prefix_stops_before_sentinel_content():
    class CharacterTokenizer:
        @staticmethod
        def encode(text):
            return SimpleNamespace(ids=[ord(character) for character in text])

    engine = SimpleNamespace(tokenizer=CharacterTokenizer())
    stem = "system: stable\n"

    prefix = _tokenizer_safe_static_prefix(engine, stem)

    assert prefix == tuple(ord(character) for character in stem + "user: ")
    assert ord("A") not in prefix[-1:]
    assert ord("Z") not in prefix[-1:]


def test_fallback_static_stem_removes_only_generation_marker():
    assert _fallback_static_stem("system: stable\nassistant:") == (
        "system: stable\n"
    )
    with pytest.raises(ValueError, match="fallback renderer"):
        _fallback_static_stem("system: stable\n")


def test_longest_common_prefix_accepts_tokenizer_sequences():
    assert _longest_common_prefix((1, 2, 3), [1, 2, 9]) == (1, 2)
