"""Regression tests for runtime/toolcalls.py (server tool-calling + vision
message normalization). Pure Python, no Metal/MLX — safe to run anytime.

  .venv/bin/python -m pytest tests/test_toolcalls.py -v
  (or just: .venv/bin/python tests/test_toolcalls.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime.toolcalls import (anthropic_messages_to_canonical,
                               canonical_tool_indices, canonicalize_tool_history,
                               compact_tool_schema, expand_image_pad_tokens,
                               load_image, merge_leading_system_messages,
                               normalize_messages, parse_tool_calls, pinned_tool_indices,
                               rank_tool_indices, responses_input_to_messages, tools_preamble)


def test_hermes_single_call():
    t = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    c, calls = parse_tool_calls(t, "llama")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert c.strip() == ""


def test_hermes_multi_call():
    t = ('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
         '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>')
    c, calls = parse_tool_calls(t, "llama")
    assert len(calls) == 2
    assert [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
    assert c.strip() == ""


def test_unoffered_generated_tool_remains_plain_text():
    unknown = '<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>'
    known = '<tool_call>{"name":"weather","arguments":{"city":"Chicago"}}</tool_call>'
    content, calls = parse_tool_calls(
        unknown + known, "llama", allowed_names={"weather"})
    assert [call["function"]["name"] for call in calls] == ["weather"]
    assert unknown in content
    assert known not in content


def test_non_string_generated_tool_name_is_not_executable():
    text = '<tool_call>{"name":{"bad":true},"arguments":{}}</tool_call>'
    content, calls = parse_tool_calls(text, "llama")
    assert calls == []
    assert content == text


def test_hermes_nested_json_arguments():
    t = '<tool_call>{"name": "search", "arguments": {"filters": {"category": "news", "limit": 5}}}</tool_call>'
    _, calls = parse_tool_calls(t, "llama")
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["filters"]["category"] == "news"


def test_hermes_surrounding_text_preserved():
    t = "Let me check.\n<tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>\nDone checking."
    c, calls = parse_tool_calls(t, "llama")
    assert "Let me check." in c and "Done checking." in c
    assert len(calls) == 1


def test_hermes_missing_arguments_key():
    t = '<tool_call>{"name": "ping"}</tool_call>'
    _, calls = parse_tool_calls(t, "llama")
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == "{}"


def test_hermes_malformed_json_left_alone():
    t = '<tool_call>{"name": "x", "arguments": {bad}</tool_call>'
    c, calls = parse_tool_calls(t, "llama")
    assert calls == []


def test_non_finite_tool_arguments_are_not_executable():
    text = '<tool_call>{"name":"x","arguments":{"value":NaN}}</tool_call>'
    content, calls = parse_tool_calls(text, "llama")
    assert calls == []
    assert content == text


def test_generated_tool_arguments_must_match_supplied_json_schema():
    from runtime.server import _parse_request_tool_calls

    tools = [{"type": "function", "function": {
        "name": "weather", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }}}]
    valid = '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'
    invalid = '<tool_call>{"name":"weather","arguments":{"units":"C"}}</tool_call>'
    content, calls = _parse_request_tool_calls(valid, tools, "qwen2")
    assert content == "" and len(calls) == 1
    content, calls = _parse_request_tool_calls(invalid, tools, "qwen2")
    assert calls == [] and content == invalid


def test_harmony_single_call_with_glyphs():
    t = '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Lyon"}<|call|>'
    _, calls = parse_tool_calls(t, "gpt_oss")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lyon"}


def test_harmony_multi_call_with_call_terminators():
    t = ('commentary to=functions.a json {"x": 1}<|call|>'
         'commentary to=functions.b json {"y": 2}<|call|>')
    _, calls = parse_tool_calls(t, "gpt_oss")
    assert len(calls) == 2
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_harmony_multi_call_glyphs_stripped_regression():
    """2026-07-13 bug: without <|call|>, the old regex's '$' fallback
    backtracked across the whole remaining text and merged two calls into
    one malformed match, silently dropping the second call entirely."""
    t = ('commentary to=functions.a json {"x": 1} '
         'commentary to=functions.b json {"y": 2}')
    _, calls = parse_tool_calls(t, "gpt_oss")
    assert len(calls) == 2, f"expected 2 calls, got {calls}"
    assert calls[0]["function"]["name"] == "a"
    assert calls[1]["function"]["name"] == "b"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    assert json.loads(calls[1]["function"]["arguments"]) == {"y": 2}


def test_harmony_malformed_neighbor_remains_visible():
    """A malformed call beside a valid call is assistant text, not a span the
    parser may silently erase merely because some other call parsed."""
    t = (
        'commentary to=functions.good json {"x": 1}<|call|>'
        'VISIBLE commentary to=functions.bad json {oops}<|call|> tail'
    )
    content, calls = parse_tool_calls(t, "gpt_oss")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "good"
    assert "VISIBLE" in content
    assert "functions.bad" in content
    assert "{oops}" in content
    assert "tail" in content


def test_harmony_valid_call_removes_leading_channel_control_glyph():
    text = (
        '<|channel|>commentary to=functions.weather <|constrain|>json'
        '<|message|>{"city":"Chicago"}<|call|>'
    )
    content, calls = parse_tool_calls(text, "gpt_oss")
    assert content == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "weather"


def test_plain_text_passthrough():
    c, calls = parse_tool_calls("Paris is the capital of France.", "gpt_oss")
    assert calls == []
    assert c == "Paris is the capital of France."


def test_normalize_messages_extracts_images_and_flattens_text():
    msgs, images = normalize_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "What is in "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            {"type": "text", "text": " this image?"}]}])
    assert images == ["data:image/png;base64,AAA"]
    assert msgs[0]["content"] == "What is in <|vision_start|><|image_pad|><|vision_end|> this image?"


def test_normalize_messages_plain_string_passthrough():
    msgs, images = normalize_messages([{"role": "user", "content": "plain"}])
    assert images == []
    assert msgs[0]["content"] == "plain"


def test_refusal_history_is_preserved_instead_of_silently_dropped():
    msgs, images = normalize_messages([{"role": "assistant", "content": [
        {"type": "refusal", "refusal": "I cannot do that."},
    ]}])
    assert images == []
    assert msgs[0]["content"] == "I cannot do that."


def test_unsupported_content_and_responses_items_fail_closed():
    invalid_messages = [{"role": "user", "content": [
        {"type": "input_file", "file_id": "file_123"},
    ]}]
    try:
        normalize_messages(invalid_messages)
    except ValueError as error:
        assert "unsupported content block type" in str(error)
        assert "input_file" in str(error)
    else:
        raise AssertionError("unsupported file block was silently dropped")

    try:
        responses_input_to_messages([{"type": "reasoning", "summary": []}])
    except ValueError as error:
        assert "unsupported Responses input item type" in str(error)
    else:
        raise AssertionError("unsupported Responses item was accepted")

    try:
        responses_input_to_messages([{
            "type": "function_call_output", "call_id": "call_x",
            "output": [{"type": "input_file", "file_id": "file_123"}],
        }])
    except ValueError as error:
        assert "unsupported Responses function output block" in str(error)
        assert "input_file" in str(error)
    else:
        raise AssertionError("unsupported function-output file was dropped")


def test_responses_function_output_preserves_text_and_image_blocks():
    canonical = responses_input_to_messages([
        {"type": "function_call", "call_id": "call_x",
         "name": "inspect", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_x", "output": [
            {"type": "input_text", "text": "result: "},
            {"type": "input_image", "image_url":
             "data:image/png;base64,AAA"},
        ]},
    ])
    normalized, images = normalize_messages(canonical)
    assert images == ["data:image/png;base64,AAA"]
    assert normalized[1]["content"] == (
        "result: <|vision_start|><|image_pad|><|vision_end|>")


def test_anthropic_falsey_system_and_malformed_text_fail_closed():
    for system in ({}, 0, [{"type": "text", "text": 7}]):
        try:
            anthropic_messages_to_canonical([], system)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid Anthropic system accepted: {system!r}")


def test_anthropic_tool_result_preserves_image_and_error_semantics():
    canonical = anthropic_messages_to_canonical([
        {"role": "assistant", "content": [{
            "type": "tool_use", "id": "toolu_x", "name": "inspect",
            "input": {},
        }]},
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "toolu_x",
            "is_error": True,
            "content": [
                {"type": "text", "text": "failed: "},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "AAA"}},
            ],
        }]},
    ])
    normalized, images = normalize_messages(canonical)
    assert images == ["data:image/png;base64,AAA"]
    assert normalized[1]["content"] == (
        "[tool error] failed: <|vision_start|><|image_pad|><|vision_end|>")


def test_load_image_accepts_valid_data_uri_and_decodes_eagerly():
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (1, 2, 3)).save(buffer, format="PNG")
    source = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()).decode()
    image = load_image(source)
    assert image.size == (8, 6)
    assert image.getpixel((0, 0)) == (1, 2, 3)
    assert image.fp is None


def test_load_image_rejects_malformed_or_oversized_inputs(monkeypatch, tmp_path):
    import base64
    import io

    from PIL import Image

    for source, expected in (
        ("", "non-empty string"),
        ("data:text/plain;base64,SGVsbG8=", "image/*;base64"),
        ("data:image/png;base64,%%%", "invalid base64"),
        (str(tmp_path / "missing.png"), "does not exist"),
    ):
        try:
            load_image(source)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"invalid image source was accepted: {source!r}")

    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buffer, format="PNG")
    source = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()).decode()
    monkeypatch.setenv("VMODEL_MAX_SOURCE_IMAGE_PIXELS", "50")
    try:
        load_image(source)
    except ValueError as error:
        assert "50-pixel limit" in str(error)
    else:
        raise AssertionError("oversized source image was accepted")


def test_responses_output_text_history_is_preserved():
    """Responses returns assistant history as output_text blocks. Dropping
    those blocks made a multi-turn prompt differ from the transcript the model
    actually produced and also destroyed prompt-prefix cacheability."""
    canonical = responses_input_to_messages([
        {"role": "user", "content": [
            {"type": "input_text", "text": "What time is it?"}]},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "It is 14:22.", "annotations": []}]},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Thanks"}]},
    ])
    msgs, images = normalize_messages(canonical)
    assert images == []
    assert [m["content"] for m in msgs] == [
        "What time is it?", "It is 14:22.", "Thanks"]


def test_responses_mixed_text_and_calls_rehydrate_as_one_assistant_turn():
    canonical = responses_input_to_messages([
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "I will check."}]},
        {"type": "function_call", "call_id": "call_weather",
         "name": "weather", "arguments": '{"city":"Chicago"}'},
        {"type": "function_call_output", "call_id": "call_weather",
         "output": "Sunny"},
    ])
    assert len(canonical) == 2
    assistant, tool = canonical
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["text"] == "I will check."
    assert assistant["tool_calls"][0]["function"]["name"] == "weather"
    assert tool == {"role": "tool", "tool_call_id": "call_weather",
                    "content": "Sunny"}


def test_merge_leading_system_messages_collapses_two_in_band_system_turns():
    """Live-confirmed shape from a real Codex/Kai client: no top-level
    `instructions`, but two separate role="system" items inside `input` --
    a main system prompt and a distinct working-memory instruction. Every
    Qwen chat template rejects a non-leading system message, so these must
    collapse into exactly one before rendering."""
    canonical = responses_input_to_messages([
        {"role": "system", "content": "You are Kai."},
        {"role": "system", "content": "WORKING_MEMORY_SYSTEM_INSTRUCTION: ..."},
        {"role": "user", "content": "hi"},
    ])
    msgs, _ = normalize_messages(canonical)
    merged = merge_leading_system_messages(msgs)
    assert [m["role"] for m in merged] == ["system", "user"]
    assert merged[0]["content"] == (
        "You are Kai.\n\nWORKING_MEMORY_SYSTEM_INSTRUCTION: ...")


def test_merge_leading_system_messages_merges_instructions_with_in_band_system():
    """The other real shape: top-level `instructions` PLUS an explicit
    system item in `input` -- also must not yield two leading system
    messages."""
    canonical = responses_input_to_messages(
        [{"role": "system", "content": "Be extra concise."},
         {"role": "user", "content": "hi"}],
        instructions="You are a helpful assistant.")
    msgs, _ = normalize_messages(canonical)
    merged = merge_leading_system_messages(msgs)
    assert [m["role"] for m in merged] == ["system", "user"]
    assert merged[0]["content"] == (
        "You are a helpful assistant.\n\nBe extra concise.")


def test_merge_leading_system_messages_is_noop_for_single_system_message():
    msgs = [{"role": "system", "content": "hi"}, {"role": "user", "content": "hey"}]
    assert merge_leading_system_messages(msgs) == msgs


def test_merge_leading_system_messages_is_noop_without_leading_system():
    msgs = [{"role": "user", "content": "hey"}]
    assert merge_leading_system_messages(msgs) == msgs


def test_parallel_tool_results_follow_declared_call_order():
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_alpha", "type": "function",
         "function": {"name": "alpha", "arguments": "{}"}},
        {"id": "call_beta", "type": "function",
         "function": {"name": "beta", "arguments": "{}"}},
    ]}
    result_alpha = {"role": "tool", "tool_call_id": "call_alpha",
                    "content": "alpha-result"}
    result_beta = {"role": "tool", "tool_call_id": "call_beta",
                   "content": "beta-result"}

    normalized = canonicalize_tool_history(
        [assistant, result_beta, result_alpha])
    assert normalized == [assistant, result_alpha, result_beta]


def test_tool_results_immediately_before_matching_calls_are_repaired():
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_a", "type": "function",
         "function": {"name": "a", "arguments": "{}"}},
        {"id": "call_b", "type": "function",
         "function": {"name": "b", "arguments": "{}"}},
    ]}
    before = [
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
    ]
    normalized = canonicalize_tool_history([*before, assistant])
    assert [message.get("role") for message in normalized] == [
        "assistant", "tool", "tool"]
    assert [message.get("tool_call_id") for message in normalized[1:]] == [
        "call_a", "call_b"]


def test_split_assistant_text_and_function_items_merge_into_one_turn():
    normalized = canonicalize_tool_history([
        {"role": "assistant", "content": "Checking."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_x", "type": "function",
            "function": {"name": "x", "arguments": "{}"}}]},
    ])
    assert len(normalized) == 1
    assert normalized[0]["content"] == "Checking."
    assert normalized[0]["tool_calls"][0]["id"] == "call_x"


def test_duplicate_tool_result_ids_fail_closed():
    history = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_x", "type": "function",
            "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_x", "content": "first"},
        {"role": "tool", "tool_call_id": "call_x", "content": "second"},
    ]
    try:
        canonicalize_tool_history(history)
    except ValueError as error:
        assert str(error) == "duplicate tool result id: 'call_x'"
    else:
        raise AssertionError("duplicate tool results were accepted")


def test_duplicate_call_ids_fail_even_without_adjacent_results():
    history = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_x", "type": "function",
         "function": {"name": "alpha", "arguments": "{}"}},
        {"id": "call_x", "type": "function",
         "function": {"name": "beta", "arguments": "{}"}},
    ]}]
    try:
        canonicalize_tool_history(history)
    except ValueError as error:
        assert str(error) == "duplicate assistant tool call id: 'call_x'"
    else:
        raise AssertionError("duplicate call ids without results were accepted")


def test_orphan_tool_result_fails_closed():
    try:
        canonicalize_tool_history([
            {"role": "tool", "tool_call_id": "call_missing", "content": "x"},
            {"role": "user", "content": "continue"},
        ])
    except ValueError as error:
        assert str(error) == "orphan tool result id: 'call_missing'"
    else:
        raise AssertionError("orphan tool result was accepted")


def test_single_idless_call_and_result_are_repaired_deterministically():
    history = [
        {"role": "tool", "content": "done"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "type": "function",
            "function": {"name": "work", "arguments": {"x": 1}}}]},
    ]
    first = canonicalize_tool_history(history)
    second = canonicalize_tool_history(history)
    assert first == second
    assert [message["role"] for message in first] == ["assistant", "tool"]
    call_id = first[0]["tool_calls"][0]["id"]
    assert call_id.startswith("call_repaired_")
    assert first[1]["tool_call_id"] == call_id
    assert first[0]["tool_calls"][0]["function"]["arguments"] == '{"x":1}'


def test_parallel_idless_calls_fail_as_ambiguous():
    try:
        canonicalize_tool_history([{
            "role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {
                    "name": "alpha", "arguments": "{}"}},
                {"type": "function", "function": {
                    "name": "beta", "arguments": "{}"}},
            ]}])
    except ValueError as error:
        assert "parallel assistant tool calls require non-empty ids" in str(error)
    else:
        raise AssertionError("ambiguous id-less parallel calls were accepted")


def test_non_finite_historical_tool_arguments_fail_closed():
    try:
        canonicalize_tool_history([{
            "role": "assistant", "content": None, "tool_calls": [{
                "id": "call_x", "type": "function",
                "function": {"name": "x", "arguments": '{"value":NaN}'},
            }],
        }])
    except ValueError as error:
        assert "arguments must contain valid JSON" in str(error)
    else:
        raise AssertionError("non-finite historical arguments were accepted")


def test_fast_tool_ranking_finds_relevant_late_tool_deterministically():
    tools = [
        {"type": "function", "function": {
            "name": f"unrelated_{i}", "description": "Manage an unrelated widget",
            "parameters": {}}}
        for i in range(40)
    ]
    tools.append({"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather forecast for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}}}})
    messages = [{"role": "user", "content": "What is the weather in Chicago?"}]
    ranked1 = rank_tool_indices(tools, messages)
    ranked2 = rank_tool_indices(tools, messages)
    assert ranked1 == ranked2
    assert ranked1[0] == 40


def test_fast_tool_ranking_keeps_historical_call_available():
    tools = [
        {"type": "function", "function": {"name": "alpha", "parameters": {}}},
        {"type": "function", "function": {"name": "beta", "parameters": {}}},
    ]
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "beta", "arguments": "{}"}}]}]
    assert rank_tool_indices(tools, messages)[0] == 1
    assert pinned_tool_indices(tools, messages) == [1]


def test_fast_tool_ranking_capability_capsule_matches_shell_paraphrases():
    tools = [
        {"type": "function", "function": {
            "name": "calendar_create_event",
            "description": "Create a scheduled meeting.", "parameters": {}}},
        {"type": "function", "function": {
            "name": "workspace_execute_command",
            "description": "Execute a command in the workspace.", "parameters": {}}},
        {"type": "function", "function": {
            "name": "browser_open_page",
            "description": "Navigate to a web page.", "parameters": {}}},
    ]
    for paraphrase in ("run this with bash", "use the terminal", "invoke a CLI"):
        messages = [{"role": "user", "content": paraphrase}]
        assert rank_tool_indices(tools, messages)[0] == 1


def test_tool_order_and_ranking_ties_are_canonical_not_request_order():
    tools = [
        {"type": "function", "function": {"name": "zeta", "parameters": {}}},
        {"type": "function", "function": {"name": "alpha", "parameters": {}}},
        {"type": "function", "function": {"name": "mu", "parameters": {}}},
    ]
    assert canonical_tool_indices(tools) == [1, 2, 0]
    assert [tools[i]["function"]["name"]
            for i in rank_tool_indices(tools, [])] == ["alpha", "mu", "zeta"]


def test_canonical_tool_order_rejects_ambiguous_names_deterministically():
    duplicate = [
        {"type": "function", "function": {"name": "same", "parameters": {}}},
        {"type": "function", "function": {"name": "same", "parameters": {
            "type": "object"}}},
    ]
    for tools in (duplicate, list(reversed(duplicate))):
        try:
            canonical_tool_indices(tools)
        except ValueError as error:
            assert str(error) == "duplicate tool function name: 'same'"
        else:
            raise AssertionError("duplicate function names were accepted")


def test_fast_tool_pins_explicit_names_even_past_soft_limit():
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("alpha", "beta", "gamma")
    ]
    messages = [{"role": "user", "content": "Use alpha and gamma together."}]
    assert pinned_tool_indices(tools, messages) == [0, 2]


def test_fast_schema_compaction_preserves_constraints_not_nested_prose():
    original = {"type": "function", "function": {
        "name": "weather", "description": "Top-level selection hint",
        "parameters": {"type": "object", "description": "redundant prose",
            "properties": {"unit": {"type": "string", "enum": ["c", "f"],
                                      "description": "Temperature unit", "default": "c"}},
            "required": ["unit"], "additionalProperties": False}}}
    compact = compact_tool_schema(original)
    fn = compact["function"]
    assert fn["name"] == "weather"
    assert fn["description"] == "Top-level selection hint"
    assert fn["parameters"]["required"] == ["unit"]
    assert fn["parameters"]["properties"]["unit"]["enum"] == ["c", "f"]
    assert "description" not in fn["parameters"]
    assert "description" not in fn["parameters"]["properties"]["unit"]
    assert "default" not in fn["parameters"]["properties"]["unit"]
    assert original["function"]["parameters"]["properties"]["unit"]["default"] == "c"


def test_fast_schema_compaction_honors_explicit_x_optional_arguments():
    original = {"type": "function", "function": {
        "name": "list_files", "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": ["string", "null"], "default": "."},
                "depth": {"type": ["number", "null"], "default": 2},
            },
            "required": ["path", "depth"],
            "x-optional": ["path", "depth"],
            "additionalProperties": False,
        }}}
    compact = compact_tool_schema(original)
    schema = compact["function"]["parameters"]
    assert "required" not in schema
    assert "x-optional" not in schema
    assert "default" not in schema["properties"]["path"]
    assert original["function"]["parameters"]["required"] == ["path", "depth"]


def test_fast_schema_compaction_preserves_property_names_that_match_annotations():
    """JSON-Schema property maps may legally contain fields literally named
    ``title`` or ``description``.  Those keys are arguments, not annotations."""
    original = {"type": "function", "function": {
        "name": "create_artifact",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "title": "Artifact title"},
            "description": {"type": "string", "description": "Artifact prose"},
            "payload": {"$ref": "#/$defs/title"},
        }, "$defs": {
            "title": {"type": "object", "properties": {
                "description": {"type": "string", "description": "Nested prose"}}},
        }, "required": ["title", "description"]}}}
    compact = compact_tool_schema(original)
    schema = compact["function"]["parameters"]
    assert set(schema["properties"]) == {"title", "description", "payload"}
    assert schema["required"] == ["title", "description"]
    assert "title" not in schema["properties"]["title"]
    assert "description" not in schema["properties"]["description"]
    assert "title" in schema["$defs"]
    assert "description" in schema["$defs"]["title"]["properties"]
    assert "description" not in schema["$defs"]["title"]["properties"]["description"]


def test_fast_schema_compaction_preserves_opaque_enum_and_const_objects():
    original = {"type": "function", "function": {
        "name": "set_role", "parameters": {"type": "object", "properties": {
            "role": {"enum": [
                {"title": "admin", "description": "literal", "default": True, "value": 1}
            ]},
            "policy": {"const": {"title": "fixed", "default": "deny"}},
        }}}}
    compact = compact_tool_schema(original)
    props = compact["function"]["parameters"]["properties"]
    assert props["role"]["enum"] == original["function"]["parameters"]["properties"]["role"]["enum"]
    assert props["policy"]["const"] == {"title": "fixed", "default": "deny"}


def test_image_placeholder_expansion_and_count_validation():
    assert expand_image_pad_tokens([1, 9, 2, 9, 3], 9, [2, 3]) == [1, 9, 9, 2, 9, 9, 9, 3]
    for counts in ([2], [2, 3, 4], [0, 3]):
        try:
            expand_image_pad_tokens([1, 9, 2, 9, 3], 9, counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid image counts accepted: {counts}")


def test_tools_preamble_includes_schema():
    p = tools_preamble([{"type": "function", "function": {"name": "get_time", "parameters": {}}}])
    assert "get_time" in p
    assert "<tool_call>" in p


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
