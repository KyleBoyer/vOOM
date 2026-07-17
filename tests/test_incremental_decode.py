"""Pure stateful streaming-detokenization regressions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.incremental_decode import (
    IncrementalDetokenizer, _byte_level_inverse,
)


class _ByteFallbackTokenizer:
    def decode(self, ids):
        if ids == [129]:
            return "\ufffd"
        if ids == [129, 110]:
            return "Ų"
        return "".join(chr(value) for value in ids)


class _CharTokenizer:
    def decode(self, ids):
        return "".join(chr(value) for value in ids)


class ByteLevel:
    pass


class _AddedToken:
    special = True


class _StatefulByteTokenizer:
    decoder = ByteLevel()

    def __init__(self, tokens, special_ids=()):
        self.tokens = dict(tokens)
        self.special_ids = set(special_ids)

    def id_to_token(self, token_id):
        return self.tokens.get(token_id)

    def get_added_tokens_decoder(self):
        return {token_id: _AddedToken() for token_id in self.special_ids}

    def decode(self, ids):
        inverse = _byte_level_inverse()
        raw = b"".join(
            bytes(inverse[value] for value in self.tokens[token_id])
            for token_id in ids if token_id not in self.special_ids)
        return raw.decode("utf-8", errors="replace")


def test_byte_fallback_pair_is_streamed_as_joint_decode_not_replacements():
    decoder = IncrementalDetokenizer(_ByteFallbackTokenizer())
    assert decoder.push([129]) == ""
    assert decoder.push([129, 110]) == "Ų"
    assert decoder.finish([129, 110]) == ""


def test_partial_stop_suffix_is_never_emitted():
    decoder = IncrementalDetokenizer(_CharTokenizer(), ["STOP"])
    ids = [ord(c) for c in "answer ST"]
    assert decoder.push(ids) == "answer "
    ids += [ord("O")]
    assert decoder.push(ids) == ""
    # The engine detects the completed stop before push and finalizes the text
    # with the stop removed.
    assert decoder.finish(ids + [ord("P")], final_text="answer ") == ""


def test_bytelevel_token_stream_preserves_split_unicode_and_special_tokens():
    tokenizer = _StatefulByteTokenizer({
        1: "Å",  # first UTF-8 byte of Ų
        2: "²",  # second UTF-8 byte of Ų
        3: "!",
        99: "<special>",
    }, special_ids={99})
    decoder = IncrementalDetokenizer(tokenizer)

    assert decoder.push_token(1) == ""
    assert decoder.push_token(2) == "Ų"
    assert decoder.push_token(99) == ""
    assert decoder.push_token(3) == "!"
    delta, text = decoder.finish_token_stream()

    assert delta == ""
    assert text == tokenizer.decode([1, 2, 99, 3]) == "Ų!"


def test_bytelevel_token_stream_flushes_final_invalid_utf8_exactly():
    tokenizer = _StatefulByteTokenizer({1: "Å"})
    decoder = IncrementalDetokenizer(tokenizer)

    assert decoder.push_token(1) == ""
    delta, text = decoder.finish_token_stream()

    assert delta == "\ufffd"
    assert text == tokenizer.decode([1]) == "\ufffd"


def test_bytelevel_token_stream_matches_stop_across_tokens():
    value = "answer STOP trailing"
    forward = {
        byte: character for character, byte in _byte_level_inverse().items()
    }
    tokenizer = _StatefulByteTokenizer({
        index: forward[ord(character)]
        for index, character in enumerate(value, start=1)
    })
    decoder = IncrementalDetokenizer(tokenizer, ["STOP"])
    emitted = []
    for token_id in range(1, value.index("P") + 2):
        emitted.append(decoder.push_token(token_id))
        if decoder.matched_stop_sequence:
            break
    delta, text = decoder.finish_token_stream()
    emitted.append(delta)

    assert decoder.matched_stop_sequence == "STOP"
    assert text == "answer "
    assert "".join(emitted) == text


def test_bytelevel_stop_inside_one_token_defers_safe_prefix_to_finish():
    tokenizer = _StatefulByteTokenizer({1: "prefixSTOPtail"})
    decoder = IncrementalDetokenizer(tokenizer, ["STOP"])

    assert decoder.push_token(1) == ""
    assert decoder.stop_text == "prefix"
    delta, text = decoder.finish_token_stream()

    assert delta == "prefix"
    assert text == "prefix"


def _run_all():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
