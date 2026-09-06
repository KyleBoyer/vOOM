import hashlib
import json

import pytest

from runtime.tool_call_witness import hermes_text_witness
from tests.fixtures.qwen3_large_agent_replay_gate import _summary


def frame(name="search", arguments=None):
    return "<tool_call>\n" + json.dumps({
        "name": name, "arguments": arguments or {},
    }, ensure_ascii=True) + "\n</tool_call>"


def test_two_generated_frames_match_two_protocol_call_digests_without_private_text():
    arguments = {"query": "PRIVATE_PAYLOAD", "limit": 3}
    text = frame(arguments=arguments) + "\n" + frame(arguments=arguments)
    witness = hermes_text_witness(text)
    summary = _summary({"output": [
        {"type": "function_call", "name": "search", "arguments": json.dumps(arguments)}
        for _ in range(2)]}, wall_s=0, events=[], progress=[], deltas=[])
    assert witness["canonical_call_sha256"] == summary["function_call_canonical_sha256"]
    assert witness["canonical_duplicate_count"] == 1
    assert witness["framed_call_objects"] == 2
    assert witness["lexical_open_markers"] == witness["lexical_close_markers"] == 2
    assert witness["raw_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert witness["raw_text_bytes"] == len(text.encode())
    assert "PRIVATE_PAYLOAD" not in json.dumps(witness)
    assert "search" not in json.dumps(witness)


def test_quoted_tags_in_argument_strings_are_lexical_not_additional_frames():
    text = frame(arguments={"query": 'literal </tool_call> and <tool_call> {"name":"fake"}'})
    witness = hermes_text_witness(text)
    assert witness["lexical_open_markers"] == witness["lexical_close_markers"] == 2
    assert witness["observed_frame_starts"] == witness["framed_call_objects"] == 1
    assert witness["frames"][0]["end_char"] == len(text)


def test_repeated_names_with_different_arguments_are_not_identical_calls():
    witness = hermes_text_witness(frame(arguments={"q": "one"}) + frame(arguments={"q": "two"}))
    assert witness["framed_call_objects"] == 2
    assert witness["canonical_duplicate_count"] == 0


@pytest.mark.parametrize("text,status", [
    ("<tool_call>unfinished", "invalid-json"),
    ('<tool_call>{"name":"a"}', "missing-close-after-json"),
    ("<tool_call>[]</tool_call>", "framed-noncall-json"),
    ('<tool_call>{"name":"a","arguments":[]}</tool_call>', "framed-noncall-json"),
    ('<tool_call>{"name":"a","arguments":{"n":NaN}}</tool_call>', "nonfinite-number"),
    ('<tool_call>{"name":"a","arguments":{"n":1e999}}</tool_call>', "nonfinite-number"),
])
def test_invalid_and_unframed_values_are_not_silently_accepted(text, status):
    witness = hermes_text_witness(text)
    assert witness["available"] is True
    assert witness["framed_call_objects"] == 0
    assert witness["frames"][0]["status"] == status


def test_missing_close_does_not_consume_later_valid_frame():
    witness = hermes_text_witness('<tool_call>{"name":"a"}\n' + frame())
    assert witness["observed_frame_starts"] == 2
    assert witness["framed_call_objects"] == 1


def test_no_frames_is_not_missing_diagnostic():
    witness = hermes_text_witness("ordinary answer")
    assert witness["available"] is True
    assert witness["framed_call_objects"] == 0
    assert witness["frames"] == []


def test_escaped_unicode_including_lone_surrogates_does_not_break_receipts():
    witness = hermes_text_witness(frame(arguments={"q": "\ud800", "name": "é"}))
    assert witness["available"] is True
    assert witness["framed_call_objects"] == 1


@pytest.mark.parametrize("text,kwargs,reason", [
    (None, {}, "not-text"), ("\ud800", {}, "invalid-utf8"),
    ("123", {"max_text_chars": 2}, "text-limit"),
    (frame() + frame(), {"max_frames": 1}, "frame-limit"),
])
def test_unavailable_is_not_a_truncated_success(text, kwargs, reason):
    witness = hermes_text_witness(text, **kwargs)
    assert witness["available"] is False and witness["reason"] == reason
    assert "frames" not in witness


@pytest.mark.parametrize("limits", [{"max_text_chars": 0}, {"max_frames": -1},
                                    {"max_frames": True}, {"max_text_chars": 1.5}])
def test_invalid_limits_fail_closed(limits):
    with pytest.raises(ValueError):
        hermes_text_witness("", **limits)


def test_decoder_recursion_limit_is_unavailable_not_mislabeled_malformed(monkeypatch):
    import runtime.tool_call_witness as observer

    class LimitedDecoder:
        def __init__(self, **kwargs):
            pass

        def raw_decode(self, text, start):
            raise RecursionError("decoder nesting limit")

    monkeypatch.setattr(observer.json, "JSONDecoder", LimitedDecoder)
    witness = hermes_text_witness(frame())
    assert witness["available"] is False and witness["reason"] == "nesting-limit"
    assert "frames" not in witness


def test_valid_long_integer_is_unavailable_not_mislabeled_malformed():
    text = '<tool_call>{"name":"a","arguments":{"n":' + '1' * 5000 + '}}</tool_call>'
    witness = hermes_text_witness(text)
    assert witness["available"] is False and witness["reason"] == "integer-limit"


def test_bare_json_call_array_is_explicitly_outside_observed_format():
    witness = hermes_text_witness('[{"name":"a","arguments":{}}]')
    assert witness["available"] is True and witness["framed_call_objects"] == 0
    assert witness["formats_observed"] == ["hermes-json"]
    assert witness["semantic_validation"] is False
