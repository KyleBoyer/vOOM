import hashlib
import io
import json
import sys

import pytest

from tests.fixtures import qwen3_large_agent_replay_gate as gate
from tests.fixtures.qwen3_large_agent_replay_gate import (
    _peak_metal_failure,
    _parse_sse_comment_progress,
    _parse_sse_comment_retry_metadata,
    _request_wire_metadata,
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


def test_wire_metadata_fingerprints_actual_bytes_without_private_values():
    original = b'{"input":"private original","tools":[{"name":"private tool"}],"stream":false}'
    wire = b'{"input":"private changed","tools":[],"stream":true,"seed":8}'
    before = bytes(wire)
    result = _request_wire_metadata(wire, original)
    assert result == {
        "request_sha256": hashlib.sha256(wire).hexdigest(),
        "request_bytes": len(wire),
        "request_changed_fields": ["input", "seed", "stream", "tools"],
    }
    assert wire == before
    assert "private" not in json.dumps(result)


def test_wire_metadata_distinguishes_missing_and_null():
    result = _request_wire_metadata(b'{"added":null}', b'{"removed":null}')
    assert result["request_changed_fields"] == ["added", "removed"]


def test_wire_metadata_preserves_byte_identity_despite_equivalent_json():
    original, wire = b'{"a":1}', b'{ "a": 1 }'
    result = _request_wire_metadata(wire, original)
    assert result["request_changed_fields"] == []
    assert result["request_sha256"] != hashlib.sha256(original).hexdigest()
    assert result["request_bytes"] == len(wire)


@pytest.mark.parametrize("value", [None, False, True, 0, -1, "42", 1.5,
                                  float("nan"), float("inf"), {}, []])
def test_requested_peak_gate_rejects_missing_or_invalid_telemetry(value):
    assert _peak_metal_failure({"true_peak_metal_bytes": value}, 8.5) is not None


def test_requested_peak_gate_rejects_absent_field():
    assert _peak_metal_failure({}, 8.5) is not None


@pytest.mark.parametrize("value,passes", [(1, True), (8_499_999_999, True),
                                        (8_500_000_000, False), (8_500_000_001, False)])
def test_peak_gate_keeps_strict_original_ceiling(value, passes):
    assert (_peak_metal_failure({"true_peak_metal_bytes": value}, 8.5) is None) is passes


def test_unrequested_peak_gate_does_not_impose_new_telemetry_requirement():
    assert _peak_metal_failure({}, None) is None


@pytest.mark.parametrize("peak,output_hash,passes", [
    (None, "a" * 64, False), (True, "a" * 64, False),
    (2_000_000_000, "a" * 64, True),
    (2_000_000_000, "b" * 64, False), (2_000_000_000, None, False),
])
def test_main_records_each_actual_payload_and_fails_closed_on_peak(
        monkeypatch, tmp_path, peak, output_hash, passes):
    raw = b'{"model":"test","input":[],"tools":[],"stream":true}'
    capture = tmp_path / "capture.json"
    capture.write_bytes(raw)
    monkeypatch.setattr(gate, "KNOWN_CAPTURES", {
        "synthetic": {"sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "tools": 0},
    })
    monkeypatch.setattr(gate, "_pressure", lambda: gate.Pressure(7_000_000_000, 0, 0))
    payloads, reports = [], []

    def post(url, payload, timeout, stream, **kwargs):
        payloads.append(payload)
        assert stream is False
        return {"http_status": 200, "wall_seconds": 1,
                "output_sha256": output_hash,
                "timing": {"true_peak_metal_bytes": peak}}

    monkeypatch.setattr(gate, "_post", post)
    monkeypatch.setattr(gate, "_write", lambda path, report: reports.append(report))
    monkeypatch.setattr(sys, "argv", ["gate", str(capture), "--repeats", "2",
                                     "--max-output-tokens", "512",
                                     "--expected-max-peak-metal-gb", "8.5",
                                     "--expected-output-sha256", "a" * 64])
    assert gate.main() == (0 if passes else 1)
    report, = reports
    assert report["passed"] is passes
    assert len(payloads) == len(report["runs"]) == 2
    for payload, row in zip(payloads, report["runs"]):
        assert row["request_sha256"] == hashlib.sha256(payload).hexdigest()
        assert row["request_bytes"] == len(payload)
        assert row["request_changed_fields"] == ["max_output_tokens", "stream"]
    assert len(report["failures"]) == (0 if passes else 2)
    assert all("peak Metal" in failure or "output SHA256" in failure
               for failure in report["failures"])
    assert report["expectations"]["output_sha256"] == "a" * 64


@pytest.mark.parametrize("invalid", ["", "a" * 63, "G" * 64, "a" * 65])
def test_bad_expected_output_digest_is_rejected_before_capture_read(monkeypatch, invalid):
    monkeypatch.setattr(sys, "argv", ["gate", "unused-capture.json",
                                     "--expected-output-sha256", invalid])
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 2


def _media_tools():
    return [
        {"type": "function", "name": "plugin__plex__plex_list_library_media",
         "description": "Original library schema", "parameters": {
             "type": "object", "required": ["offset"],
             "properties": {"offset": {"anyOf": [
                 {"type": "number"}, {"type": "null"}]}},
             "additionalProperties": False}},
        {"type": "function", "name": "unrelated_tool", "parameters": {}},
        {"type": "function", "name": "plugin__plex__plex_search_media",
         "description": "Original search schema", "parameters": {
             "type": "object", "required": ["query", "type", "limit"],
             "properties": {"query": {"type": "string"}},
             "x-optional": ["type", "limit"]}},
    ]


@pytest.mark.parametrize("user_text", [None, "Search for documentaries instead."])
def test_media_scenario_preserves_original_tool_objects_order_and_schema(user_text):
    tools = _media_tools()
    before = json.dumps(tools, sort_keys=True)
    request = {"tools": tools, "input": [{"role": "developer"}],
               "tool_choice": "auto", "temperature": 0.7, "stream": True}
    turns = gate._media_search_scenario(request, user_text)
    assert request["tools"][0] is tools[0]
    assert request["tools"][1] is tools[2]
    assert json.dumps(tools, sort_keys=True) == before
    assert request["input"] is turns
    assert [turn["role"] for turn in turns] == ["system", "user"]
    assert request["temperature"] == 0.7 and request["stream"] is True
    assert request["tool_choice"] == "auto"
    if user_text is not None:
        assert turns[-1]["content"][0]["text"] == user_text
    turns[0]["content"] = "changed"
    assert gate.MEDIA_SEARCH_ACTION_INPUT[0]["content"] != "changed"


@pytest.mark.parametrize("indices", [[], [0], [2], [0, 0], [2, 2], [0, 2, 2]])
def test_media_scenario_rejects_missing_or_duplicate_tools_without_mutation(indices):
    tools = _media_tools()
    request = {"tools": [tools[index] for index in indices], "input": []}
    before = json.dumps(request)
    with pytest.raises(ValueError, match="both unique captured Plex tools"):
        gate._media_search_scenario(request, None)
    assert json.dumps(request) == before


def test_media_scenario_cli_reports_modified_nonstream_request(monkeypatch, tmp_path):
    original = {"model": "test", "input": [{"role": "user", "content": "PRIVATE_CAPTURE_SENTINEL"}],
                "tools": _media_tools(), "stream": True, "tool_choice": "auto"}
    raw = json.dumps(original).encode()
    capture = tmp_path / "capture.json"
    capture.write_bytes(raw)
    monkeypatch.setattr(gate, "KNOWN_CAPTURES", {
        "synthetic": {"sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "tools": 3}})
    monkeypatch.setattr(gate, "_pressure", lambda: gate.Pressure(7_000_000_000, 0, 0))
    payloads, reports = [], []

    def post(url, payload, timeout, stream, **kwargs):
        payloads.append(payload)
        assert stream is False
        return {"http_status": 200, "wall_seconds": 1}

    monkeypatch.setattr(gate, "_post", post)
    monkeypatch.setattr(gate, "_write", lambda path, report: reports.append(report))
    monkeypatch.setattr(sys, "argv", [
        "gate", str(capture), "--scenario", "media-search-action",
        "--scenario-user-text", "Search for Arrival movies, at most three.",
        "--repeats", "1", "--max-output-tokens", "512"])
    assert gate.main() == 0
    payload, = payloads
    request = json.loads(payload)
    assert request["tools"] == [original["tools"][0], original["tools"][2]]
    assert [turn["role"] for turn in request["input"]] == ["system", "user"]
    assert request["input"][-1]["content"][0]["text"].startswith("Search for Arrival")
    assert "vmodel_progress_events" not in request
    report, = reports
    assert report["request"]["scenario"] == "media-search-action"
    assert report["request"]["scenario_user_sha256"]
    row, = report["runs"]
    assert row["request_changed_fields"] == ["input", "max_output_tokens", "stream", "tools"]
    assert row["request_sha256"] == hashlib.sha256(payload).hexdigest()
    assert "PRIVATE_CAPTURE_SENTINEL" not in json.dumps(report)


def _call_summary(calls, **kwargs):
    return gate._summary(
        {"output": [{"type": "function_call", **call} for call in calls]},
        wall_s=1, events=[], progress=[], deltas=[], **kwargs)


@pytest.mark.parametrize("expected_name", ["search", "absent"])
def test_unrelated_call_cannot_lend_arguments_to_expected_function(expected_name):
    result = _call_summary([
        {"name": "search", "arguments": '{"query":"wrong","limit":0}'},
        {"name": "unrelated", "arguments": '{"query":"private-value","limit":3}'},
    ], expected_function_call_name=expected_name,
        expected_function_arguments={"query": "private-value", "limit": 3},
        expected_positive_function_arguments=("limit",),
        expected_nonempty_function_arguments=("query", "limit"))
    assert result["function_call_arguments_match"] is False
    assert result["function_call_positive_arguments_match"] is False
    assert result["function_call_nonempty_arguments_match"] is False
    assert result["function_call_argument_match_scope"] == "expected_name"
    assert "private-value" not in json.dumps(result)


def test_canonical_duplicate_witness_ignores_ids_and_json_spacing_but_not_arguments():
    result = _call_summary([
        {"name": "search", "call_id": "one", "arguments": '{"q":"PRIVATE_VALUE", "n":3}'},
        {"name": "search", "call_id": "two", "arguments": '{"n": 3,"q": "PRIVATE_VALUE"}'},
        {"name": "search", "arguments": '{"q":"different","n":3}'},
    ], expected_function_call_name="search",
        expected_function_arguments={"q": "PRIVATE_VALUE", "n": 3})
    digests = result["function_call_canonical_sha256"]
    assert len(digests) == 3 and digests[0] == digests[1] != digests[2]
    assert result["function_call_duplicate_count"] == 1
    assert result["function_call_argument_parse_failures"] == 0
    assert result["function_call_arguments_match"] is True
    assert "PRIVATE_VALUE" not in json.dumps(result)


@pytest.mark.parametrize("arguments", [None, "", "not-json", "[]", "null", "true", "7"])
def test_malformed_arguments_are_counted_and_cannot_match_empty_expected_object(arguments):
    result = _call_summary([{"name": "search", "arguments": arguments}],
                           expected_function_arguments={})
    assert result["function_call_names"] == ["search"]
    assert result["function_call_argument_parse_failures"] == 1
    assert result["function_call_canonical_sha256"] == []
    assert result["function_call_arguments_match"] is False


@pytest.mark.parametrize("actual_count,expected_count,passes", [
    (0, 0, True), (1, 1, True), (2, 1, False), (1, 0, False), (0, 1, False)])
def test_cli_exact_call_count_is_a_persisted_gate(
        monkeypatch, tmp_path, actual_count, expected_count, passes):
    raw = b'{"input":[],"tools":[]}'
    capture = tmp_path / "capture.json"
    capture.write_bytes(raw)
    monkeypatch.setattr(gate, "KNOWN_CAPTURES", {
        "synthetic": {"sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "tools": 0}})
    monkeypatch.setattr(gate, "_pressure", lambda: gate.Pressure(7_000_000_000, 0, 0))
    reports = []

    def post(*args, **kwargs):
        assert kwargs["expected_function_call_name"] == "search"
        return {"http_status": 200, "wall_seconds": 1,
                "function_call_names": ["search"] * actual_count}

    monkeypatch.setattr(gate, "_post", post)
    monkeypatch.setattr(gate, "_write", lambda path, report: reports.append(report))
    # The name is passed to _post even when no calls are expected. For that
    # one case keep the assertion about count separate from the name gate.
    monkeypatch.setattr(sys, "argv", [
        "gate", str(capture), "--repeats", "1", "--expected-function-call-name", "search",
        "--expected-function-call-count", str(expected_count)])
    exit_code = gate.main()
    report, = reports
    count_failures = [f for f in report["failures"] if "function call count" in f]
    assert bool(count_failures) is (not passes)
    assert report["expectations"]["function_call_count"] == expected_count
    assert exit_code == (1 if report["failures"] else 0)


def test_negative_expected_call_count_fails_before_reading_capture(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gate", "unused", "--expected-function-call-count", "-1"])
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 2


@pytest.mark.parametrize("stream", [False, True])
def test_http_post_binds_expected_name_through_nonstream_and_sse(monkeypatch, stream):
    response = {"status": "completed", "output": [
        {"type": "function_call", "name": "search", "arguments": '{"limit":0}'},
        {"type": "function_call", "name": "other", "arguments": '{"limit":3}'},
    ]}
    body = ("data: " + json.dumps({"type": "response.completed", "response": response})
            + "\n\n" if stream else json.dumps(response)).encode()
    monkeypatch.setattr(gate.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(body))
    row = gate._post(
        "http://localhost/fixture", b"{}", 1, stream,
        expected_function_call_name="search", expected_function_arguments={"limit": 3})
    assert row["function_call_arguments_match"] is False
    assert row["function_call_argument_match_scope"] == "expected_name"


def test_unscoped_argument_matching_remains_explicitly_any_name():
    row = _call_summary([{"name": "anything", "arguments": '{"limit":3}'}],
                        expected_function_arguments={"limit": 3})
    assert row["function_call_arguments_match"] is True
    assert row["function_call_argument_match_scope"] == "any_name"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_json_arguments_fail_closed_including_nested_values(value):
    for argument_text in ('{"limit":' + value + '}', '{"nested":[' + value + ']}'):
        row = _call_summary([{"name": "search", "arguments": argument_text}],
                            expected_function_call_name="search",
                            expected_function_arguments={},
                            expected_positive_function_arguments=("limit",),
                            expected_nonempty_function_arguments=("limit",))
        assert row["function_call_argument_parse_failures"] == 1
        assert row["function_call_canonical_sha256"] == []
        assert row["function_call_arguments_match"] is False
        assert row["function_call_positive_arguments_match"] is False
        assert row["function_call_nonempty_arguments_match"] is False


def test_escaped_lone_surrogate_cannot_crash_canonical_call_receipt():
    row = _call_summary([{"name": "search", "arguments": r'{"q":"\ud800"}'}])
    assert row["function_call_argument_parse_failures"] == 0
    digest, = row["function_call_canonical_sha256"]
    assert len(digest) == 64
    assert "ud800" not in json.dumps(row)


def test_canonical_call_hash_normalizes_unicode_escaping_and_accepts_finite_numbers():
    row = _call_summary([
        {"name": "search", "arguments": r'{"q":"\u00e9","limit":3.0}'},
        {"name": "search", "arguments": '{"limit":3.0,"q":"é"}'},
    ], expected_function_arguments={"limit": 3},
        expected_positive_function_arguments=("limit",))
    assert row["function_call_duplicate_count"] == 1
    assert row["function_call_argument_parse_failures"] == 0
    assert row["function_call_arguments_match"] is True
    assert row["function_call_positive_arguments_match"] is True
