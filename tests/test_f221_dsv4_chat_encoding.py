"""F221: the DeepSeek V4 server path must use the checkpoint's own encoder.

The checkpoint ships no chat_template, so without this the server falls
through to a generic ``user:``/``assistant:`` transcript with no learned turn
boundary. These tests check the adapter against the released module's own
constants rather than against literal strings, so a protocol change in a
future checkpoint surfaces as a failure here instead of as bad output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "DeepSeek-V4-Flash-0731"

pytestmark = pytest.mark.skipif(
    not (MODEL / "encoding" / "encoding_dsv4.py").is_file(),
    reason="DeepSeek-V4-Flash-0731 encoding/ not present")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"},
                           "days": {"type": "integer"}},
            "required": ["city"]},
    },
}]


@pytest.fixture(scope="module")
def released():
    from runtime.dsv4_chat import load_encoder

    return load_encoder(MODEL)


def test_prompt_ends_with_the_assistant_marker(released):
    from runtime.dsv4_chat import render_prompt

    prompt = render_prompt(MODEL, [{"role": "user", "content": "hi"}])
    assert prompt.startswith(released.bos_token)
    assert released.ASSISTANT_SP_TOKEN in prompt
    assert released.USER_SP_TOKEN in prompt
    # The generic fallback's giveaway; must not appear.
    assert "\nassistant:" not in prompt


def test_tools_are_rendered_into_the_system_section(released):
    from runtime.dsv4_chat import render_prompt

    prompt = render_prompt(MODEL, [{"role": "user", "content": "weather?"}],
                           tools=TOOLS)
    assert "get_weather" in prompt
    assert released.dsml_token in prompt, "DSML protocol section missing"
    # One system section, not one per tool.
    assert prompt.count("### Available Tool Schemas") == 1


def test_tools_merge_into_an_existing_system_message(released):
    from runtime.dsv4_chat import render_prompt

    messages = [{"role": "system", "content": "You are terse."},
                {"role": "user", "content": "weather?"}]
    prompt = render_prompt(MODEL, messages, tools=TOOLS)
    assert "You are terse." in prompt
    assert "get_weather" in prompt
    assert prompt.count("### Available Tool Schemas") == 1


def test_thinking_mode_selection():
    from runtime.dsv4_chat import thinking_mode

    assert thinking_mode(None, False) == "chat"
    assert thinking_mode(None, True) == "thinking"
    assert thinking_mode(True, False) == "thinking"
    assert thinking_mode(False, True) == "chat", (
        "an explicit enable_thinking=False must win over reasoning_effort")


def _round_trip_text(released, name, arguments):
    """Build a completion the way the model emits one."""
    encoded = released.encode_arguments_to_dsml({"name": name,
                                                 "arguments": arguments})
    call = released.tool_call_template.format(
        dsml_token=released.dsml_token, name=name, arguments=encoded)
    return released.tool_calls_template.format(
        dsml_token=released.dsml_token,
        tc_block_name=released.tool_calls_block_name,
        tool_calls=call)


def test_parses_a_released_format_tool_call(released):
    from runtime.dsv4_chat import parse_tool_calls

    body = _round_trip_text(released, "get_weather",
                            json.dumps({"city": "Paris", "days": 3}))
    text = "I'll check that.\n\n" + body
    content, calls = parse_tool_calls(MODEL, text)

    assert content == "I'll check that."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "get_weather"
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments == {"city": "Paris", "days": 3}, (
        "argument types did not survive the DSML round trip")
    assert calls[0]["id"]


def test_unknown_tool_names_are_dropped(released):
    from runtime.dsv4_chat import parse_tool_calls

    text = _round_trip_text(released, "rm_rf", json.dumps({"path": "/"}))
    content, calls = parse_tool_calls(MODEL, text,
                                      allowed_names={"get_weather"})
    assert calls == []
    assert content == text, "text must be preserved when nothing is extracted"


def test_a_truncated_block_is_not_an_error(released):
    """max_tokens cutting a call short is an ordinary outcome."""
    from runtime.dsv4_chat import parse_tool_calls

    body = _round_trip_text(released, "get_weather",
                            json.dumps({"city": "Paris"}))
    text = ("Checking.\n\n" + body)[:-12]
    content, calls = parse_tool_calls(MODEL, text)
    assert calls == []
    assert content == text


def test_plain_text_passes_through(released):
    from runtime.dsv4_chat import parse_tool_calls

    content, calls = parse_tool_calls(MODEL, "Paris is sunny today.")
    assert calls == []
    assert content == "Paris is sunny today."
