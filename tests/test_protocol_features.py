"""Integration tests for tool calling, vision, streaming-with-tools, and
reasoning/sampling-param handling across all three protocol surfaces
(2026-07-13, user request: "fix all of it").

Honesty note: SmolLM2-135M is a raw BASE model with no instruction tuning,
so it will not reliably CHOOSE to invoke a tool on its own — that's a model
capability question, not a plumbing question, and this project doesn't
have a small local instruction-tuned text model to exercise that
specifically. What IS tested here, deterministically, regardless of model
intelligence:
  - tool schemas are accepted and the response validates against the real
    SDK's Pydantic models (request-side schema conversion correctness)
  - multi-turn history INCLUDING a prior tool call + tool result (which
    this test constructs directly, not relying on the model to produce)
    parses without error and the server keeps generating (input-side
    adapter correctness: responses_input_to_messages /
    anthropic_messages_to_canonical)
  - streaming still completes correctly when tools are configured
    (buffer-and-withhold path)
  - reasoning/temperature/top_p are accepted and echoed, never silently
    dropped
  - vision works end-to-end for both new endpoints using an installed Qwen3-VL
    checkpoint; fast mode additionally preserves two-image ordering

  .venv/bin/python tests/test_protocol_features.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = 8097
TEXT_MODEL = "SmolLM2-135M"
_VISION_MODEL_CANDIDATES = (
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
)
VISION_MODEL = os.environ.get("VMODEL_TEST_VISION_MODEL") or next(
    (name for name in _VISION_MODEL_CANDIDATES
     if (ROOT / "models" / name / "config.json").exists()),
    _VISION_MODEL_CANDIDATES[-1],
)

# This is a real, multi-GB vision model the project may have local access to --
# not something to fetch in CI or on a fresh clone. Skip
# gracefully (both under pytest and under _run_all()'s direct execution)
# rather than trying to download a model or failing on a model that
# was never expected to be present.
_VISION_MODEL_AVAILABLE = (ROOT / "models" / VISION_MODEL / "config.json").exists()
_vision_skip = pytest.mark.skipif(
    not _VISION_MODEL_AVAILABLE,
    reason=f"{VISION_MODEL} is not available locally (a real multi-GB model, not fetched in CI)",
)

WEATHER_TOOL_OPENAI = {
    "type": "function", "function": {
        "name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}
WEATHER_TOOL_RESPONSES = {
    "type": "function", "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}
WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather", "description": "Get the weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}


def _wait_for_server(proc, timeout=30):
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited early (code {proc.returncode})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not become ready in time")


def _start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "runtime.server", "--port", str(PORT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    _wait_for_server(proc)
    return proc


def _stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# ---- tool schema acceptance (request-side conversion correctness) ----

def test_openai_responses_tools_accepted():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=16, input="What's the weather in Paris?",
            tools=[WEATHER_TOOL_RESPONSES])
        # constructing `resp` validates our JSON against the SDK's schema
        # even when the model doesn't choose to call the tool
        assert resp.tool_choice == "auto"
        assert len(resp.tools) == 1
    finally:
        _stop_server(proc)


def test_openai_responses_tool_choice_none_and_parallel_false_are_honored():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=8, input="Say hello.",
            tools=[WEATHER_TOOL_RESPONSES], tool_choice="none",
            parallel_tool_calls=False)
        assert resp.tool_choice == "none"
        assert resp.parallel_tool_calls is False
        assert len(resp.tools) == 1
        assert all(item.type != "function_call" for item in resp.output)
    finally:
        _stop_server(proc)


def test_anthropic_messages_tools_accepted():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        resp = client.messages.create(
            model=TEXT_MODEL, max_tokens=16, tools=[WEATHER_TOOL_ANTHROPIC],
            messages=[{"role": "user", "content": "What's the weather in Paris?"}])
        assert resp.type == "message"
        assert resp.stop_reason in ("end_turn", "max_tokens", "tool_use")
    finally:
        _stop_server(proc)


def test_anthropic_tool_choice_none_disables_tool_output():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        resp = client.messages.create(
            model=TEXT_MODEL, max_tokens=8, tools=[WEATHER_TOOL_ANTHROPIC],
            tool_choice={"type": "none"},
            messages=[{"role": "user", "content": "Say hello."}])
        assert resp.stop_reason != "tool_use"
        assert all(block.type != "tool_use" for block in resp.content)
    finally:
        _stop_server(proc)


# ---- multi-turn history with a PRE-CONSTRUCTED tool call + result ----
# (tests the INPUT-side adapter deterministically — we build these blocks
# ourselves rather than relying on the tiny base model to produce them)

def test_openai_responses_function_call_history_round_trip():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=16,
            tools=[WEATHER_TOOL_RESPONSES],
            input=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"type": "function_call", "call_id": "call_abc123",
                 "name": "get_weather", "arguments": '{"city": "Paris"}'},
                {"type": "function_call_output", "call_id": "call_abc123",
                 "output": "Sunny, 22C"},
            ])
        assert resp.object == "response"
        assert resp.status in ("completed", "incomplete")
    finally:
        _stop_server(proc)


def test_anthropic_messages_tool_use_history_round_trip():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        resp = client.messages.create(
            model=TEXT_MODEL, max_tokens=16, tools=[WEATHER_TOOL_ANTHROPIC],
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_abc123", "name": "get_weather",
                     "input": {"city": "Paris"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc123",
                     "content": "Sunny, 22C"}]},
            ])
        assert resp.type == "message"
        assert isinstance(resp.content[0].text, str)
    finally:
        _stop_server(proc)


# ---- streaming with tools configured (buffer-and-withhold path) ----

def test_openai_responses_streaming_with_tools():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        stream = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=8, stream=True,
            tools=[WEATHER_TOOL_RESPONSES], input="What's the weather in Paris?")
        events = list(stream)
        assert len(events) > 0
        assert any(
            e.type in ("response.completed", "response.incomplete")
            for e in events)
    finally:
        _stop_server(proc)


def test_anthropic_messages_streaming_with_tools():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        events = []
        with client.messages.stream(
            model=TEXT_MODEL, max_tokens=8, tools=[WEATHER_TOOL_ANTHROPIC],
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        ) as stream:
            for event in stream:
                events.append(event)
        assert len(events) > 0
        assert any(getattr(e, "type", None) == "message_stop" for e in events)
    finally:
        _stop_server(proc)


# ---- reasoning/sampling-param honesty ----

def test_openai_responses_sampling_is_seeded_and_functional():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        kwargs = dict(
            model=TEXT_MODEL, max_output_tokens=8, input="Hi",
            temperature=0.7, top_p=0.9,
            extra_body={"top_k": 16, "seed": 1234})
        first = client.responses.create(**kwargs)
        second = client.responses.create(**kwargs)
        assert first.temperature == 0.7
        assert first.top_p == 0.9
        assert first.output_text == second.output_text
        raw = first.model_extra or {}
        assert raw.get("vmodel_sampling") == "categorical"
        assert raw.get("vmodel_top_k") == 16
        assert raw.get("vmodel_seed") == 1234
    finally:
        _stop_server(proc)


def test_anthropic_messages_sampling_and_effort_are_applied():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        resp = client.messages.create(
            model=TEXT_MODEL, max_tokens=8, temperature=0.7, top_p=0.9,
            top_k=8, thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Hi"}])
        raw = resp.model_extra or {}
        assert raw.get("vmodel_sampling") == "categorical"
        assert raw.get("requested_temperature") == 0.7
        assert raw.get("requested_top_k") == 8
        assert raw.get("vmodel_reasoning_effort") == "low"
        assert raw.get("vmodel_thinking_enabled") is False
    finally:
        _stop_server(proc)


def test_openai_responses_json_schema_is_token_constrained():
    import json

    from openai import OpenAI
    from runtime.structured import validate_json_schema

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "confidence": {"type": "string", "enum": ["low", "high"]},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }
    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=48,
            input="Return 2+2 and confidence.",
            text={"format": {"type": "json_schema", "name": "answer",
                             "schema": schema, "strict": True}})
        value = json.loads(resp.output_text)
        validate_json_schema(value, schema)
        assert (resp.model_extra or {}).get("vmodel_constraint") == "json_schema"
    finally:
        _stop_server(proc)


def test_openai_responses_specific_tool_is_required_and_schema_valid():
    import json

    from openai import OpenAI
    from runtime.structured import validate_json_schema

    clock = {
        "type": "function", "name": "clock", "description": "Get time",
        "parameters": {"type": "object", "properties": {
            "tz": {"type": "string"}}, "required": ["tz"],
            "additionalProperties": False},
    }
    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=TEXT_MODEL, max_output_tokens=64, input="Use weather for Paris.",
            tools=[WEATHER_TOOL_RESPONSES, clock],
            tool_choice={"type": "function", "name": "get_weather"})
        calls = [item for item in resp.output if item.type == "function_call"]
        assert len(calls) == 1 and calls[0].name == "get_weather"
        args = json.loads(calls[0].arguments)
        validate_json_schema(args, WEATHER_TOOL_RESPONSES["parameters"])
        assert (resp.model_extra or {}).get("vmodel_constraint") == "required_tool"
    finally:
        _stop_server(proc)


# ---- vision for the two new endpoints ----

def _make_test_image_data_uri(color=(0, 255, 0)) -> str:
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


@_vision_skip
def test_openai_responses_vision():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        datauri = _make_test_image_data_uri()
        request = {
            "model": VISION_MODEL,
            "max_output_tokens": 20,
            "input": [{"role": "user", "content": [
                {"type": "input_text",
                 "text": "What color is this square? One word."},
                {"type": "input_image", "image_url": datauri},
            ]}],
        }
        resp = client.responses.create(**request)
        assert resp.status in ("completed", "incomplete")
        assert "green" in resp.output_text.lower()
        assert resp.vmodel_timing["cache_source"] == "cold"
        assert resp.vmodel_timing["vision_prompt_cache_exact_hit"] == 0

        repeated = client.responses.create(**request)
        assert "green" in repeated.output_text.lower()
        assert repeated.vmodel_timing["cache_source"] == "vision_memory"
        assert repeated.vmodel_timing["vision_prompt_cache_exact_hit"] == 1
        assert repeated.vmodel_timing["vision_prompt_cache_tower_skipped"] == 1
    finally:
        _stop_server(proc)


@_vision_skip
def test_openai_chat_streaming_vision_with_usage():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        stream = client.chat.completions.create(
            model=f"lossy-{VISION_MODEL}", max_tokens=20, stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What color is this square? One word."},
                {"type": "image_url", "image_url": {
                    "url": _make_test_image_data_uri()}},
            ]}],
        )
        chunks = list(stream)
        text = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks if chunk.choices)
        usage = [chunk.usage for chunk in chunks if chunk.usage is not None]
        assert "green" in text.lower()
        assert len(usage) == 1 and usage[0].completion_tokens > 0
        assert usage[0].total_tokens == usage[0].prompt_tokens + usage[0].completion_tokens
    finally:
        _stop_server(proc)


@_vision_skip
def test_openai_responses_streaming_vision():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        stream = client.responses.create(
            model=VISION_MODEL, max_output_tokens=20, stream=True,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text":
                 "What color is this square? One word."},
                {"type": "input_image", "image_url":
                 _make_test_image_data_uri()},
            ]}],
        )
        events = list(stream)
        text = "".join(
            event.delta for event in events
            if event.type == "response.output_text.delta")
        assert "green" in text.lower()
        assert any(event.type == "response.completed" for event in events)
    finally:
        _stop_server(proc)


@_vision_skip
def test_openai_responses_fast_vision_preserves_image_order():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.responses.create(
            model=f"lossy-{VISION_MODEL}", max_output_tokens=12,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text":
                 "Name each square color in image order. Two words."},
                {"type": "input_image", "image_url":
                 _make_test_image_data_uri((0, 255, 0))},
                {"type": "input_image", "image_url":
                 _make_test_image_data_uri((0, 0, 255))},
            ]}],
        )
        text = resp.output_text.lower()
        assert "green" in text and "blue" in text
        assert text.index("green") < text.index("blue")
        assert resp.vmodel_timing["resident_pipelined_decode_steps"] > 0
    finally:
        _stop_server(proc)


@_vision_skip
def test_anthropic_messages_vision():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.messages.create(
            model=VISION_MODEL, max_tokens=20,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What color is this square? One word."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            ]}])
        assert resp.content[0].type == "text"
        assert "green" in resp.content[0].text.lower()
    finally:
        _stop_server(proc)


@_vision_skip
def test_anthropic_messages_streaming_vision():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        events = []
        with client.messages.stream(
            model=VISION_MODEL, max_tokens=20,
            messages=[{"role": "user", "content": [
                {"type": "text", "text":
                 "What color is this square? One word."},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": _make_test_image_data_uri().split(",", 1)[1],
                }},
            ]}],
        ) as stream:
            events.extend(stream)
        text = "".join(
            event.delta.text for event in events
            if (event.type == "content_block_delta"
                and event.delta.type == "text_delta"))
        assert "green" in text.lower()
        assert any(event.type == "message_stop" for event in events)
    finally:
        _stop_server(proc)


_VISION_TEST_NAMES = {
    "test_openai_responses_vision",
    "test_openai_chat_streaming_vision_with_usage",
    "test_openai_responses_streaming_vision",
    "test_openai_responses_fast_vision_preserves_image_order",
    "test_anthropic_messages_vision",
    "test_anthropic_messages_streaming_vision",
}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed, skipped = 0, [], 0
    for fn in fns:
        if fn.__name__ in _VISION_TEST_NAMES and not _VISION_MODEL_AVAILABLE:
            print(f"  {fn.__name__}: SKIP ({VISION_MODEL} not available locally)")
            skipped += 1
            continue
        try:
            fn()
            print(f"  {fn.__name__}: PASS")
            passed += 1
        except Exception as e:
            print(f"  {fn.__name__}: FAIL ({type(e).__name__}: {e})")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(fns) - skipped} tests passed ({skipped} skipped)")
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
