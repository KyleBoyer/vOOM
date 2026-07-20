"""Server adapter checks that do not import MLX or start a model process."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.server import (Handler, INFER_LOCK, PreparedPrompt, RequestValidationError,
                            _TokenOffsetIndex,
                            _active_context_limit,
                            _advertised_model_ids,
                            _cache_phase_telemetry,
                            _execution_profile_fields,
                            _fast_dense_resident_kv_projection,
                            _hidden_gateway_catalogs,
                            _hidden_gateway_activation_clear,
                            _hidden_gateway_activation_get,
                            _hidden_gateway_activation_put,
                            _hidden_gateway_conversation_key,
                            _HiddenDecisionStream,
                            _hidden_gateway_decision_choice,
                            _hidden_gateway_force_reason,
                            _hidden_gateway_search_result_limit,
                            _hidden_tool_gateway_enabled,
                            _hidden_tool_abstain_pair,
                            _hidden_tool_enable_pair,
                            _hidden_tool_search_pair,
                            _hidden_gateway_virtual_pairs,
                            _load_vision_images,
                            _MarkerHoldback,
                            _chat_prompt,
                            _omitted_output_token_limit,
                            _positive_token_limit,
                            _prepare_chat_prompt, _render_template,
                            _compiled_template,
                            _openai_finish_reason, _responses_output_items, _safe_emit_len,
                            _parse_request_tool_calls, _tool_request_controls,
                            _preferred_fast_artifact,
                            _dspark_draft_for,
                            _speculative_draft_for,
                            _request_reasoning_controls, _request_sampling,
                            _registry,
                            _tool_capsule_spans,
                            _vision_protocol_timing,
                            _validate_fast_dense_resident_kv,
                            _validate_context_budget, _validate_generation_controls,
                            split_model_mode)


class _CharTokenizer:
    def encode(self, text):
        return SimpleNamespace(
            ids=list(text), offsets=[(index, index + 1)
                                     for index in range(len(text))])


class _CountingCharTokenizer(_CharTokenizer):
    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return super().encode(text)


def test_route_ignores_query_string():
    handler = object.__new__(Handler)
    handler.path = "/v1/responses?trace=true"
    assert handler._route() == "/responses"


def test_vision_protocol_timing_uses_generic_path_stats():
    result = {
        "vision_cache_hits": 99,
        "vision_tool_pic_reused_tokens": 98,
        "path_stats": {
            "vision_cache_hits": 2,
            "vision_cache_misses": 1,
            "vision_prompt_cache_tower_skipped": 1,
            "prompt_cache_prefix_tokens": 144,
            "prompt_cache_exact_hit": 1,
            "vision_prompt_cache_stored": 1,
            "prompt_cache_source": "vision_tool_pic",
            "tool_pic": 1,
            "tool_pic_reused_tokens": 80,
            "tool_pic_selected_tokens": 64,
            "tool_pic_repaired_tokens": 4,
            "tool_pic_memory_admitted": 1,
            "tool_pic_projected_bytes": 123_456,
            "prompt_state_approximate": 1,
        },
    }

    timing = _vision_protocol_timing(result)

    assert timing == {
        "vision_cache_hits": 2,
        "vision_cache_misses": 1,
        "vision_prompt_cache_tower_skipped": 1,
        "vision_prompt_cache_prefix_tokens": 144,
        "vision_prompt_cache_exact_hit": 1,
        "vision_prompt_cache_stored": 1,
        "cache_source": "vision_tool_pic",
        "tool_pic": 1,
        "tool_pic_reused_tokens": 80,
        "tool_pic_selected_tokens": 64,
        "tool_pic_repaired_tokens": 4,
        "tool_pic_memory_admitted": 1,
        "tool_pic_projected_bytes": 123_456,
        "tool_pic_system_available_bytes": 0,
        "tool_pic_system_floor_bytes": 0,
        "tool_pic_system_memory_admitted": 0,
        "prompt_state_approximate": 1,
    }


def test_response_write_timeout_releases_inference_lock(monkeypatch):
    class Connection:
        def __init__(self):
            self.timeout = None
            self.values = []

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value
            self.values.append(value)

    handler = object.__new__(Handler)
    handler.connection = Connection()
    handler._read_json_request = lambda: (b"{}", {}, 2)
    handler._preflight_nested_request = lambda req: ([], [], [])

    def timeout():
        raise TimeoutError("client stopped reading")

    handler._do_post_locked = timeout
    monkeypatch.setenv("VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS", "0.05")
    handler.do_POST()

    assert handler.close_connection
    assert handler.connection.values == [0.05, None]
    assert INFER_LOCK.acquire(blocking=False)
    INFER_LOCK.release()


def _fake_engine(*, model_limit=1_000_000, context_bound=0, model_type="qwen2"):
    return SimpleNamespace(
        tokenizer=_CharTokenizer(),
        cfg=SimpleNamespace(
            model_type=model_type, max_position_embeddings=model_limit),
        rc=SimpleNamespace(context_bound=context_bound),
        effective_max_position_embeddings=model_limit,
        rope_profile="released",
    )


def test_compact_json_is_sorted_and_minified():
    template = "{{ value | tojson }}"
    rendered = _render_template(
        template, compact_json=True, value={"z": 1, "a": {"y": 2, "b": 3}})
    assert rendered == '{"a":{"b":3,"y":2},"z":1}'


def test_compact_json_preserves_jinja_htmlsafe_escaping():
    rendered = _render_template(
        "{{ value | tojson }}", compact_json=True,
        value={"text": "</tools> & it's safe"})
    assert r"\u003c/tools\u003e" in rendered
    assert r"\u0026" in rendered
    assert r"\u0027" in rendered
    assert "</tools>" not in rendered


def test_compact_json_accepts_standard_indent_argument_but_stays_canonical():
    rendered = _render_template(
        "{{ value | tojson(indent=2) }}", compact_json=True,
        value={"z": 1, "a": 2})
    assert rendered == '{"a":2,"z":1}'


def test_template_compilation_is_cached_by_text_and_render_profile():
    _compiled_template.cache_clear()
    template = "{{ value | tojson }}"
    _render_template(template, value={"x": 1})
    _render_template(template, value={"x": 2})
    after_released = _compiled_template.cache_info()
    assert (after_released.hits, after_released.misses) == (1, 1)

    _render_template(template, compact_json=True, value={"x": 3})
    after_compact = _compiled_template.cache_info()
    assert (after_compact.hits, after_compact.misses) == (1, 2)


def test_exact_long_rendered_prompt_token_ids_are_engine_local_lru(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0].content }}")
    engine = _fake_engine()
    engine.tokenizer = _CountingCharTokenizer()
    messages = [{"role": "user", "content": "x" * 1500}]
    args = (engine, tmp_path, messages, "low", [], [], "lossless", 1)

    first = _prepare_chat_prompt(*args)
    second = _prepare_chat_prompt(*args)

    assert engine.tokenizer.calls == 1
    assert first[0].token_ids == second[0].token_ids
    assert first[4]["prompt_token_cache_hit"] == 0
    assert second[4]["prompt_token_cache_hit"] == 1


def test_native_template_history_renders_tool_arguments_as_object_not_string():
    template = (
        "{% for message in messages %}{% if message.tool_calls %}"
        "{{ message.tool_calls[0].function.arguments | tojson }}"
        "{% endif %}{% endfor %}"
    )
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_weather", "type": "function", "function": {
            "name": "weather", "arguments": '{"city":"Chicago"}'}}]}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": template}))
        released = _chat_prompt(_fake_engine(), model_dir, messages, "low")
        compact = _chat_prompt(
            _fake_engine(), model_dir, messages, "low", compact_json=True)
    assert released == '{"city": "Chicago"}'
    assert compact == '{"city":"Chicago"}'


def test_standalone_jinja_template_receives_reasoning_and_thinking_controls():
    template = (
        "{{ reasoning_effort }}|"
        "{{ enable_thinking if enable_thinking is defined else 'unset' }}|"
        "{{ messages[0].content }}"
    )
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        engine = _fake_engine(model_type="glm_moe_dsa")
        released = _chat_prompt(
            engine, model_dir, [{"role": "user", "content": "Hello"}], "high")
        fastest = _chat_prompt(
            engine, model_dir, [{"role": "user", "content": "Hello"}], "high",
            enable_thinking=False)
    assert released == "high|unset|Hello"
    assert fastest == "high|False|Hello"


def test_fast_mode_disables_template_thinking_while_lossless_keeps_default():
    template = (
        "{{ 'thinking' if enable_thinking is not defined or enable_thinking "
        "else 'no-thinking' }}"
    )
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        engine = _fake_engine(model_type="glm_moe_dsa")
        args = (engine, model_dir, [{"role": "user", "content": "x"}],
                "high", [], [])
        released, *_ = _prepare_chat_prompt(*args, "lossless", 1)
        fastest, *_ = _prepare_chat_prompt(*args, "fast", 1)
    assert released == "thinking"
    assert fastest == "no-thinking"


def test_explicit_high_effort_overrides_fast_no_thinking_default():
    template = "{{ 'thinking' if enable_thinking else 'no-thinking' }}"
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        prompt, *_ = _prepare_chat_prompt(
            _fake_engine(model_type="qwen3"), model_dir,
            [{"role": "user", "content": "x"}], "high", [], [], "fast", 1,
            enable_thinking=True, reasoning_requested=True)
    assert prompt == "thinking"


def test_explicit_effort_injects_instruction_for_non_reasoning_template():
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(
            "{% for message in messages %}{{ message.role }}={{ message.content }}|{% endfor %}")
        prompt = _chat_prompt(
            _fake_engine(), model_dir, [{"role": "user", "content": "Solve it"}],
            "high", enable_thinking=True, reasoning_requested=True)
    assert prompt.startswith("system=Reason thoroughly")
    assert prompt.endswith("user=Solve it|")


def _named_tool(name: str, marker: str | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": marker or f"Call {name}",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "description": f"Value for {name}"}}},
    }}


def test_tool_request_controls_implement_auto_none_and_parallel_policy():
    chat_tools = [_named_tool("weather")]
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"parallel_tool_calls": False}, chat_tools)
    assert effective == chat_tools
    assert choice == "auto"
    assert not parallel

    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"tool_choice": "none"}, chat_tools)
    assert effective == []
    assert choice == "none"
    assert parallel

    anthropic_tools = [{"name": "weather", "input_schema": {"type": "object"}}]
    effective, choice, parallel = _tool_request_controls(
        "/messages", {"tool_choice": {
            "type": "auto", "disable_parallel_tool_use": True}}, anthropic_tools)
    assert effective == anthropic_tools
    assert choice == "auto"
    assert not parallel


def test_tool_request_controls_reject_unenforceable_and_malformed_choices():
    chat_tools = [_named_tool("weather")]
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"tool_choice": "required"}, chat_tools)
    assert effective == chat_tools and choice == "required" and parallel
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {
            "tool_choice": {"type": "function", "function": {"name": "weather"}}},
        chat_tools)
    assert effective == chat_tools and choice == "specific:weather" and parallel
    bad_requests = [
        {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
        {"parallel_tool_calls": "false"},
    ]
    for request in bad_requests:
        try:
            _tool_request_controls("/chat/completions", request, chat_tools)
        except RequestValidationError:
            pass
        else:
            raise AssertionError(f"malformed/unenforceable controls accepted: {request}")


def test_unsupported_generation_controls_fail_instead_of_being_ignored():
    sampling = _validate_generation_controls(
        "/chat/completions", {"n": 1, "response_format": {"type": "text"}})
    assert sampling.is_greedy
    sampling = _validate_generation_controls(
        "/messages", {"temperature": 0.7, "top_p": 0.9, "top_k": 10,
                      "seed": 123})
    assert not sampling.is_greedy
    assert (sampling.temperature, sampling.top_p, sampling.top_k, sampling.seed) == (
        0.7, 0.9, 10, 123)
    assert not _validate_generation_controls(
        "/responses", {"top_p": 0.8}).is_greedy
    assert _validate_generation_controls(
        "/chat/completions", {"response_format": {"type": "json_object"}})
    bad = [
        ("/chat/completions", {"n": 2}),
        ("/chat/completions", {"logprobs": True}),
        ("/chat/completions", {"presence_penalty": 0.5}),
        ("/chat/completions", {"logit_bias": {"42": 100}}),
        ("/chat/completions", {"functions": []}),
        ("/completions", {"best_of": 2}),
        ("/completions", {"echo": True}),
        ("/responses", {"top_logprobs": 5}),
        ("/responses", {"previous_response_id": "resp_old"}),
        ("/responses", {"background": True}),
        ("/responses", {"truncation": "auto"}),
        ("/responses", {"text": {"format": {"type": "json_schema"}}}),
        ("/responses", {"text": {"verbosity": "low"}}),
        ("/responses", {"temperature": "0.7"}),
        ("/responses", {"top_p": 1.5}),
        ("/responses", {"reasoning": {"effort": 7}}),
        ("/messages", {"top_k": -1}),
        ("/responses", {"seed": -1}),
        ("/messages", {"thinking": {"type": "mystery"}}),
        ("/messages", {"thinking": {"type": "enabled", "budget_tokens": 0}}),
    ]
    for route, request in bad:
        try:
            _validate_generation_controls(route, request)
        except RequestValidationError:
            pass
        else:
            raise AssertionError(f"unsupported control was ignored: {request}")


def test_reasoning_controls_map_all_protocols():
    assert _request_reasoning_controls(
        "/chat/completions", {"reasoning_effort": "high"})[:3] == (
            "high", True, True)
    assert _request_reasoning_controls(
        "/responses", {"reasoning": {"effort": "minimal"}})[:3] == (
            "minimal", False, True)
    assert _request_reasoning_controls(
        "/messages", {"thinking": {"type": "enabled", "budget_tokens": 128}}) == (
            "high", True, True, 128)


def test_parallel_tool_calls_false_keeps_only_first_parsed_call():
    text = (
        '<tool_call>{"name":"alpha","arguments":{}}</tool_call>'
        '<tool_call>{"name":"beta","arguments":{}}</tool_call>')
    tools = [_named_tool("alpha"), _named_tool("beta")]
    content, calls = _parse_request_tool_calls(
        text, tools, "qwen2", allow_parallel=False)
    assert content == ""
    assert [call["function"]["name"] for call in calls] == ["alpha"]


def test_fast_tool_catalog_permutations_render_identically_but_wire_order_survives(
        tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}"
        "{% for message in messages %}{{ message.content }}{% endfor %}")
    engine = _fake_engine()
    tools = [_named_tool("zeta"), _named_tool("alpha"), _named_tool("mu")]
    raw = [{"type": "function", "name": t["function"]["name"],
            "parameters": t["function"]["parameters"]} for t in tools]
    messages = [{"role": "user", "content": "hello"}]

    first = _prepare_chat_prompt(
        engine, tmp_path, messages, "low", tools, raw, "fast", 1)
    permutation = [2, 0, 1]
    second = _prepare_chat_prompt(
        engine, tmp_path, messages, "low", [tools[i] for i in permutation],
        [raw[i] for i in permutation], "fast", 1)

    assert first[0] == second[0]
    assert isinstance(first[0], PreparedPrompt)
    assert first[0].token_ids == tuple(engine.tokenizer.encode(first[0]).ids)
    assert first[1] == second[1]
    assert first[4]["tool_catalog_id"] == second[4]["tool_catalog_id"]
    assert first[4]["tool_order_profile"] == "canonical-name-v1"
    # Prompt ordering is an internal cache optimization; response schemas retain
    # the exact request order expected by each protocol adapter.
    assert [t["function"]["name"] for t in second[2]] == ["mu", "zeta", "alpha"]
    assert [t["name"] for t in second[3]] == ["mu", "zeta", "alpha"]


def test_fast_qwen_style_tools_carry_token_aligned_capsule_spans(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "<tools>{% for tool in tools %}\n{{ tool | tojson }}{% endfor %}"
        "\n</tools>{{ messages[0].content }}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]

    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "fast", 1)

    assert len(prompt.tool_capsules) == 2
    bodies = ["".join(prompt.token_ids[start:end])
              for _identity, start, end in prompt.tool_capsules]
    assert '"name":"alpha"' in bodies[0]
    assert '"name":"zeta"' in bodies[1]


def test_tool_capsule_spans_preserve_duplicate_occurrences_and_boundaries():
    from jinja2.utils import htmlsafe_json_dumps

    tool = _named_tool("duplicate")
    serialized = str(htmlsafe_json_dumps(
        tool, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True))
    prompt = (
        "The literal <tools></tools> is documentation."
        f"<tools>\n{serialized}\n{serialized}\n</tools>")
    token_ids = tuple(prompt)
    offsets = tuple((index, index + 1) for index in range(len(prompt)))

    spans = _tool_capsule_spans(
        prompt, [tool, tool], token_ids, offsets)

    assert len(spans) == 2
    assert spans[0][0] == spans[1][0]
    assert spans[0][2] <= spans[1][1]
    assert prompt[spans[0][1]:spans[0][2]] == serialized
    assert prompt[spans[1][1]:spans[1][2]] == serialized


def test_tool_capsule_offset_index_uses_first_nonempty_duplicate_start():
    from jinja2.utils import htmlsafe_json_dumps

    tool = _named_tool("alpha")
    serialized = str(htmlsafe_json_dumps(
        tool, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True))
    prompt = f"<tools>{serialized}</tools>"
    char_start = prompt.index(serialized)
    offsets = [(index, index + 1) for index in range(len(prompt))]
    offsets.insert(char_start, (char_start, char_start))
    token_ids = tuple(range(len(offsets)))

    spans = _tool_capsule_spans(prompt, [tool], token_ids, tuple(offsets))

    assert len(spans) == 1
    assert spans[0][1] == char_start + 1
    assert spans[0][2] == char_start + len(serialized) + 1


def test_token_offset_index_matches_first_interval_reference_across_blocks():
    offsets = tuple(
        (index // 2, min(1_000, index // 2 + 1 + (index * 17) % 47))
        for index in range(1_024))
    index = _TokenOffsetIndex(offsets, 1_000)

    for token_start in (0, 1, 127, 255, 256, 511, 700, 1_023):
        for char_end in (1, 17, 128, 255, 400, 511):
            expected = next((
                position + 1
                for position, (start, end) in enumerate(offsets)
                if position >= token_start and start < char_end <= end
            ), None)
            assert index.token_end(char_end, token_start) == expected


def test_tool_capsule_spans_fail_closed_on_nonmonotonic_offsets():
    tool = _named_tool("alpha")
    prompt = "<tools>{}</tools>"
    offsets = tuple(
        [(0, 1), (2, 3), (1, 2)]
        + [(index, index + 1) for index in range(3, len(prompt))])

    assert _tool_capsule_spans(
        prompt, [tool], tuple(range(len(offsets))), offsets) == ()


def test_native_template_that_ignores_tools_gets_canonical_tool_preamble(
        tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role }}:"
        "{{ message.content }}|{% endfor %}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]

    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "fast", 1)

    assert prompt.startswith("system:You have access to the following tools.")
    assert "<tools>" in prompt and "</tools>" in prompt
    assert prompt.index('"name":"alpha"') < prompt.index('"name":"zeta"')
    assert len(prompt.tool_capsules) == 2


def test_lossless_tool_order_remains_request_order(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool.function.name }}|{% endfor %}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]
    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "lossless", 1)
    assert prompt == "zeta|alpha|"


def test_parallel_tool_completion_order_renders_one_canonical_prompt(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role }}:"
        "{% for call in message.tool_calls or [] %}{{ call.id }};{% endfor %}"
        "{{ message.tool_call_id or '' }}={{ message.content or '' }}|{% endfor %}")
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_alpha", "type": "function",
         "function": {"name": "alpha", "arguments": "{}"}},
        {"id": "call_beta", "type": "function",
         "function": {"name": "beta", "arguments": "{}"}},
    ]}
    alpha = {"role": "tool", "tool_call_id": "call_alpha", "content": "A"}
    beta = {"role": "tool", "tool_call_id": "call_beta", "content": "B"}
    args = (_fake_engine(), tmp_path)

    first = _prepare_chat_prompt(
        *args, [assistant, alpha, beta], "low", [], [], "lossless", 1)
    second = _prepare_chat_prompt(
        *args, [assistant, beta, alpha], "low", [], [], "lossless", 1)

    assert first[0] == second[0]
    assert first[0].token_ids == second[0].token_ids


def test_fast_added_or_removed_tool_preserves_canonical_catalog_prefix(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "HEADER|{% for tool in tools %}[{{ tool.function.name }}]"
        "{% endfor %}|MESSAGES")
    base = [_named_tool(name) for name in ("zeta", "beta", "delta")]
    added = base + [_named_tool("epsilon")]
    args = (_fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low")
    base_prompt, *_ = _prepare_chat_prompt(*args, base, base, "fast", 1)
    added_prompt, *_ = _prepare_chat_prompt(*args, added, added, "fast", 1)
    assert base_prompt == "HEADER|[beta][delta][zeta]|MESSAGES"
    assert added_prompt == "HEADER|[beta][delta][epsilon][zeta]|MESSAGES"
    common = os.path.commonprefix((base_prompt, added_prompt))
    assert common == "HEADER|[beta][delta]["


def test_fast_duplicate_tool_names_fail_before_rendering(tmp_path):
    tools = [_named_tool("same", "one"), _named_tool("same", "two")]
    try:
        _prepare_chat_prompt(
            _fake_engine(), tmp_path, [{"role": "user", "content": "x"}],
            "low", tools, tools, "fast", 1)
    except RequestValidationError as error:
        assert str(error) == "duplicate tool function name: 'same'"
    else:
        raise AssertionError("ambiguous duplicate tools were rendered")


def test_fast_shortlist_is_permutation_invariant_at_score_ties(tmp_path):
    from unittest.mock import patch

    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool.function.name }}|{% endfor %}")
    tools = [_named_tool(name) for name in ("zeta", "beta", "alpha")]
    permutation = [tools[1], tools[0], tools[2]]
    args = (_fake_engine(), tmp_path, [{"role": "user", "content": "unrelated"}],
            "low")
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_LIMIT": "2"}):
        first = _prepare_chat_prompt(*args, tools, tools, "fast", 1)
        second = _prepare_chat_prompt(*args, permutation, permutation, "fast", 1)
    assert first[0] == second[0] == "alpha|beta|"
    assert {t["function"]["name"] for t in first[2]} == {"alpha", "beta"}
    assert {t["function"]["name"] for t in second[2]} == {"alpha", "beta"}


def test_hidden_tool_gateway_starts_virtual_only_then_retrieves_real_tools():
    from unittest.mock import patch

    tools = [
        _named_tool("workspace_execute", "Execute a shell command in the workspace."),
        _named_tool("browser_open", "Open a web page in a browser."),
        _named_tool("calendar_create", "Create a calendar event."),
    ]
    raw = [{
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "parameters": tool["function"]["parameters"],
    } for tool in tools]
    messages = [{"role": "user", "content": "Tell me a NodeJS joke."}]
    initial, initial_raw, pinned, retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, limit=2)
    assert initial == [] and initial_raw == [] and pinned == 0
    assert retrieval["tool_embedding_status"] == "not_queried"

    selected, selected_raw, pinned, retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, query="open this page in the browser", limit=2)
    assert "browser_open" in {
        tool["function"]["name"] for tool in selected
    }
    assert "browser_open" in {tool["name"] for tool in selected_raw}
    assert len(selected) == 2 and pinned == 0
    assert "tool_retrieval_profile" in retrieval

    virtual, virtual_raw = _hidden_tool_search_pair()
    assert virtual["function"]["name"] == "vmodel_search_tools"
    assert virtual_raw["name"] == "vmodel_search_tools"
    enable, enable_raw = _hidden_tool_enable_pair()
    assert enable["function"]["name"] == "vmodel_enable_tools"
    assert enable_raw["name"] == "vmodel_enable_tools"
    virtuals, virtuals_raw = _hidden_gateway_virtual_pairs()
    assert [tool["function"]["name"] for tool in virtuals] == [
        "vmodel_search_tools", "vmodel_enable_tools"]
    assert [tool["name"] for tool in virtuals_raw] == [
        "vmodel_search_tools", "vmodel_enable_tools"]
    abstain, abstain_raw = _hidden_tool_abstain_pair()
    assert abstain["function"]["name"] == "vmodel_no_suitable_tool"
    assert abstain_raw["parameters"]["required"] == ["reason"]
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_GATEWAY": "1"}):
        assert _hidden_tool_gateway_enabled("fast", len(tools), "auto")
        assert not _hidden_tool_gateway_enabled("lossless", len(tools), "auto")
        assert not _hidden_tool_gateway_enabled("fast", len(tools), "specific:browser_open")


def test_plex_transcript_keeps_fixed_decision_catalog_and_pinned_execution_set():
    """Regression for the user's real 2026-07-19 Plex pagination transcript."""
    from unittest.mock import patch

    tools = [
        _named_tool("plugin__plex__plex_list_library", "List Plex media with pagination."),
        _named_tool("workspace_execute", "Execute a workspace command."),
        _named_tool("browser_open", "Open a browser page."),
        _named_tool("calendar_create", "Create a calendar event."),
    ]
    raw = [{
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "parameters": tool["function"]["parameters"],
    } for tool in tools]
    first_turn = [{
        "role": "user",
        "content": (
            "list the plex movies/tv shows that are age rating PG13 or TV-7 "
            "or less(for younger kids) and whose root folder does NOT contain "
            "\"/Kids/\"\nMake sure to paginate the plex listing\n"),
    }]
    later_turn = first_turn + [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_plex", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library",
                "arguments": '{"limit":32,"offset":0}',
            },
        }],
    }, {
        "role": "tool", "tool_call_id": "call_plex",
        "name": "plugin__plex__plex_list_library",
        "content": '{"movies":[{"title":"A Christmas Carol",'
                   '"contentRating":"PG"}],"movieHasMore":true}',
    }, {
        "role": "user", "content": "try just doing no query?",
    }]

    # The decision schemas never include transcript-pinned real functions.
    virtuals, _raw_virtuals = _hidden_gateway_virtual_pairs()
    assert [tool["function"]["name"] for tool in virtuals] == [
        "vmodel_search_tools", "vmodel_enable_tools"]

    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([0, 1, 2, 3], {"tool_embedding_status": "test"})):
        selected, _selected_raw, _pinned, _meta = _hidden_gateway_catalogs(
            tools, raw, first_turn, query="list Plex media", limit=2)
    activated = tuple(tool["function"]["name"] for tool in selected)
    assert activated == (
        "plugin__plex__plex_list_library", "workspace_execute")

    # Page/corrected-argument intent ranks an already-activated tool first:
    # preserve the exact schema set even though call history now hard-pins Plex.
    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([0, 2, 3, 1], {"tool_embedding_status": "test"})):
        stable, _stable_raw, pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, later_turn, query="continue Plex pagination without query",
            limit=2, activated_names=activated,
            expansion_limit=2, max_activated=4)
    assert tuple(tool["function"]["name"] for tool in stable) == activated
    assert pinned == 1
    assert metadata["gateway_activation_profile"] == "stable-hit"

    # A genuinely different top capability is still admitted rather than being
    # trapped behind the old tool choice.
    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([2, 3, 0, 1], {"tool_embedding_status": "test"})):
        expanded, _expanded_raw, _pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, later_turn, query="open a page in the browser",
            limit=2, activated_names=activated,
            expansion_limit=2, max_activated=4)
    assert {tool["function"]["name"] for tool in expanded} == {
        *activated, "browser_open", "calendar_create"}
    assert metadata["gateway_activation_profile"] == "expanded"


def test_hidden_gateway_search_hydrates_at_most_four_without_forcing_four():
    assert _hidden_gateway_search_result_limit(32, 4, 32) == 4
    assert _hidden_gateway_search_result_limit(32, 4, 2) == 2
    assert _hidden_gateway_search_result_limit(32, 4, 0) == 1
    assert _hidden_gateway_search_result_limit(32, 4, "many") == 4
    assert _hidden_gateway_search_result_limit(3, 4, 32) == 3


def test_gateway_activation_key_survives_appended_tool_turns_without_raw_state():
    tools = [_named_tool("plugin__plex__plex_list_library")]
    anchor = [{"role": "system", "content": "stable harness prompt"}, {
        "role": "user", "content": "list Plex media and paginate"}]
    continuation = anchor + [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library", "arguments": "{}"}}]}, {
        "role": "tool", "tool_call_id": "call_1", "content": "page one"}, {
        "role": "user", "content": "get page two"}]
    first_key = _hidden_gateway_conversation_key("lossy-Qwen3-4B", tools, anchor)
    next_key = _hidden_gateway_conversation_key(
        "lossy-Qwen3-4B", tools, continuation)
    assert first_key == next_key
    assert len(first_key) == 64

    _hidden_gateway_activation_clear()
    try:
        _hidden_gateway_activation_put(first_key, tools)
        assert _hidden_gateway_activation_get(next_key, tools) == (
            "plugin__plex__plex_list_library",)
    finally:
        _hidden_gateway_activation_clear()


def test_hidden_gateway_forces_only_high_confidence_external_intents():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Tell me a joke about Node.js."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "How does the Node.js event loop work?"},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Write one coherent paragraph about streaming."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "List three reasons streaming feels faster."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Create a short poem."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "What folder are we in?"},
    ]) == "external-state-inspection"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Whats the largest top level directory?"},
    ]) == "external-state-inspection"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Check for real."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Write this file in the workspace."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Search the web."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Use an available tool to inspect it."},
    ]) == "explicit-tool-request"


def test_hidden_gateway_forces_confirmed_deferred_action_but_not_bare_ack():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Which directory is largest?"},
        {"role": "assistant", "content": "I'll run a command to check it."},
        {"role": "user", "content": "do it"},
    ]) == "confirmed-deferred-action"
    assert _hidden_gateway_force_reason([
        {"role": "assistant", "content": "That sounds good."},
        {"role": "user", "content": "okay"},
    ]) is None


def test_hidden_gateway_does_not_force_again_after_tool_result():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "What folder are we in?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "pwd", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
    ]) is None


def test_hidden_gateway_required_client_or_intent_targets_only_search():
    assert _hidden_gateway_decision_choice(
        "auto", "external-state-inspection") == \
        "specific:vmodel_search_tools"
    assert _hidden_gateway_decision_choice(
        "required", "client-required") == "specific:vmodel_search_tools"
    assert _hidden_gateway_decision_choice("auto", None) == "auto"


def test_hidden_tool_gateway_hard_pins_transcript_tools():
    tools = [_named_tool("workspace_execute"), _named_tool("browser_open")]
    raw = [{
        "type": "function", "name": tool["function"]["name"],
        "description": "", "parameters": {},
    } for tool in tools]
    messages = [{
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "workspace_execute", "arguments": "{}"},
        }],
    }]
    selected, _raw, pinned, _retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, limit=1)
    assert pinned == 1
    assert [tool["function"]["name"] for tool in selected] == ["workspace_execute"]


def test_hidden_search_query_is_not_diluted_by_large_system_prompt():
    from unittest.mock import patch

    tools = [
        _named_tool("workspace_execute", "Execute a shell command."),
        _named_tool("calendar_create", "Create a calendar meeting."),
    ]
    raw = [{
        "type": "function", "name": tool["function"]["name"],
        "description": tool["function"]["description"], "parameters": {},
    } for tool in tools]
    messages = [{
        "role": "system",
        "content": "calendar meeting appointment event " * 5_000,
    }]
    with patch.dict(os.environ, {"VMODEL_TOOL_EMBEDDINGS": "0"}):
        selected, _raw, _pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, messages, query="run a terminal command", limit=1)
    assert [tool["function"]["name"] for tool in selected] == [
        "workspace_execute"]
    assert metadata["tool_embedding_status"] == "unavailable"


def test_glm_fast_mode_enables_quantized_cache_pages():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="glm_moe_dsa", tie_word_embeddings=False,
        index_topk=2048, vision_config=None)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-glm"), "fast")

    assert captured[0].quant_bits == 4
    assert captured[0].max_weight_cache_mb == 5000


def test_k25_lossless_uses_demand_paging_without_speculative_prefetch():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k25", tie_word_embeddings=False,
        index_topk=0, vision_config=None)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-k25"), "lossless")

    rc = captured[0]
    assert rc.max_weight_cache_mb == 1500
    assert rc.prefetch_depth == 0
    assert rc.stream_lm_head
    assert not rc.pin_lm_head
    assert rc.quant_bits == 0


def test_qwen36_profiles_bound_experts_and_use_hybrid_endpoint_cache():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config={"depth": 27})
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen36"), "lossless")
        EngineManager().get(Path("/tmp/fake-qwen36"), "fast")

    lossless, fast = captured
    for rc in captured:
        assert rc.prompt_kv_dir == ""
        assert rc.hot_prompt_kv
        assert rc.hot_prompt_kv_slots == 2
        assert rc.hot_prompt_kv_min_tokens == 16
        assert rc.prefill_chunk_size == 512
        assert rc.hot_prompt_kv_chunk_size == rc.prefill_chunk_size
        assert rc.expert_fetch_batch == 1
        assert rc.decode_expert_fetch_batch == 8
        assert rc.fast_dirs[0].endswith("vmodel_fast_tier/fake-qwen36")
    assert lossless.quant_bits == 0
    assert lossless.max_weight_cache_mb == 6000
    assert fast.quant_bits == 4
    assert fast.quant_mode == "mxfp4"
    assert not fast.quant_attention
    assert not fast.quant_router
    assert not fast.quant_lm_head
    assert fast.max_weight_cache_mb == 6000


def test_dense_fast_mode_uses_validated_mxfp4_and_pipelined_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits, rc.quant_min_dim) == (
        "mxfp4", 32, 4, 0)
    assert rc.quantize_tied_lm_head
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 512
    assert rc.fused_swiglu
    assert rc.stepped_kv_threshold == 512
    assert not rc.embed_rows
    assert rc.min_weight_cache_mb == 1500


def test_dense_fast_gateway_uses_reclaimable_floor_and_2k_kv_boundaries():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_GATEWAY": "1"}), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen-gateway"), "fast")

    rc = captured[0]
    assert rc.min_weight_cache_mb == 600
    assert rc.prefill_chunk_size == 512
    assert rc.hot_prompt_kv_chunk_size == 512
    assert rc.adaptive_kv_spill_mb == 256
    assert rc.adaptive_kv_spill_prefill_chunk_size == 512
    assert rc.hot_prompt_kv_min_available_mb == 0


def test_dense_fast_paged_kv_profile_disables_incompatible_hot_paths(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)
    env = {
        "VMODEL_FAST_KV_MAX_MB": "2200",
        "VMODEL_FAST_KV_SPILL_DIR": str(tmp_path / "spill"),
        "VMODEL_FAST_KV_SPILL_COMPRESS": "1",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3"), "fast")

    rc = captured[0]
    assert rc.max_kv_mb == 2200
    assert rc.release_paged_kv_after_generate
    assert rc.prefill_chunk_size == 512
    assert rc.mlx_cache_limit_mb == 64
    assert rc.kv_spill_dir == str(tmp_path / "spill")
    assert rc.kv_spill_compress
    assert not rc.hot_prompt_kv
    assert not rc.tool_pic
    assert not rc.tool_pic_shared_pages
    assert rc.hot_prompt_kv_persist_dir == ""


def test_qwen3_fast_resident_kv_projection_rejects_real_harness_scale():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="qwen3", vision_config=None, num_experts=0,
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128),
        rc=SimpleNamespace(max_kv_mb=0),
    )
    safe = _fast_dense_resident_kv_projection(engine, "fast", 10_774, 16)
    assert safe["bytes_per_token"] == 147_456
    assert safe["projected_bytes"] < safe["limit_bytes"]
    assert safe["positions"] == 10_774
    assert safe["declared_positions"] == 10_790
    assert safe["declared_projected_bytes"] > safe["projected_bytes"]

    try:
        _validate_fast_dense_resident_kv(engine, "fast", 28_307, 64)
    except RequestValidationError as error:
        message = str(error)
        assert "resident BF16 KV projection" in message
        assert "VMODEL_FAST_TOOL_GATEWAY=1" in message
        assert "VMODEL_FAST_TOOL_LIMIT=32" in message
        assert "quarantined" in message
    else:
        raise AssertionError("unsafe real-harness resident KV was accepted")

    engine.rc.adaptive_kv_spill_mb = 256
    adaptive = _validate_fast_dense_resident_kv(
        engine, "fast", 28_307, 64)
    assert adaptive["adaptive_spill_required"] == 1
    assert adaptive["adaptive_spill_mb"] == 256

    engine.rc.max_kv_mb = 2200
    assert _validate_fast_dense_resident_kv(
        engine, "fast", 28_307, 64) is None


def test_dense_kv_preflight_subtracts_evictable_retained_state():
    base = dict(
        active_metal_bytes=7_830_000_000,
        retained_prompt_kv_bytes=1_680_000_000,
        orphan_prompt_kv_bytes=0,
        evictable_prompt_kv_bytes=1_680_000_000,
        hot_prompt_slots=1,
        metal_ceiling_bytes=9_050_000_000,
    )
    engine = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="qwen3", vision_config=None, num_experts=0,
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128),
        rc=SimpleNamespace(max_kv_mb=0),
        prompt_cache_memory_snapshot=lambda: base,
    )
    projection = _validate_fast_dense_resident_kv(
        engine, "fast", 10_453, 4_096)
    assert projection["retained_prompt_kv_bytes"] == 1_680_000_000
    assert projection["dynamic_projected_bytes"] < 9_050_000_000

    engine.prompt_cache_memory_snapshot = lambda: {
        **base,
        "retained_prompt_kv_bytes": 0,
        "evictable_prompt_kv_bytes": 0,
    }
    try:
        _validate_fast_dense_resident_kv(engine, "fast", 10_453, 4_096)
    except RequestValidationError as error:
        assert "live dense-Qwen Metal projection" in str(error)
        assert "before generation" in str(error)
    else:
        raise AssertionError("live projection ignored retained-cache pressure")


def test_cache_phase_telemetry_keeps_hidden_phases_separate():
    decision = _cache_phase_telemetry("gateway_decision", {
        "prompt_tokens": 2_000,
        "prefill_s": 0.25,
        "path_stats": {
            "prompt_cache_namespace": "gateway_decision",
            "prompt_cache_prefix_tokens": 1_900,
            "prompt_cache_source": "hot_disk",
            "prompt_cache_exact_hit": 0,
        },
    })
    execution = _cache_phase_telemetry("gateway_execution", {
        "prompt_tokens": 10_000,
        "prefill_s": 42.0,
        "path_stats": {
            "prompt_cache_namespace": "gateway_execution",
            "prompt_cache_prefix_tokens": 0,
            "prompt_cache_source": "cold",
            "tool_pic_reused_tokens": 128,
            "tool_pic_selected_tokens": 9_872,
            "hot_prompt_admission_evicted_slots": 1,
            "hot_prompt_admission_evicted_bytes": 1_500_000_000,
        },
    })
    assert decision["cached_tokens"] == 1_900
    assert decision["cache_source"] == "hot_disk"
    assert execution["cached_tokens"] == 0
    assert execution["effective_reused_tokens"] == 128
    assert execution["admission_evicted_bytes"] == 1_500_000_000


def test_responses_stream_emits_terminal_failure_instead_of_truncated_sse():
    import io

    handler = Handler.__new__(Handler)
    handler.wfile = io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None
    handler._sampling = SimpleNamespace()
    handler._constraint = None

    class Engine:
        cfg = SimpleNamespace(model_type="qwen3")

        def __init__(self):
            self.cleaned = 0

        def discard_failed_request_state(self):
            self.cleaned += 1

    engine = Engine()

    def fail(_on_token, _on_progress):
        raise MemoryError("projected working set exceeds ceiling")

    handler._stream_responses(
        "prompt", 64, [], engine, [], lambda *_args: {},
        "resp_test", "Qwen3-4B", 1, None, None, None, [],
        "msg_test", "auto", False, generate_fn=fail)

    wire = handler.wfile.getvalue().decode()
    assert statuses == [200]
    assert '"type": "response.failed"' in wire
    assert '"code": "server_memory_error"' in wire
    assert '"status": "failed"' in wire
    assert engine.cleaned == 1


def test_failed_request_cleanup_releases_only_failed_kv():
    from runtime.engine import StreamingEngine

    class State:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    failed = State()
    survivor = State()
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.last_kv = failed
    engine._hot_prompt_slots = [
        SimpleNamespace(kv=survivor),
        SimpleNamespace(kv=failed),
    ]
    engine._h_window = object()
    engine._h_last = object()
    engine._provisional = object()

    engine.discard_failed_request_state()

    assert failed.releases == 1
    assert survivor.releases == 0
    assert [slot.kv for slot in engine._hot_prompt_slots] == [survivor]
    assert engine.last_kv is None
    assert engine._h_window is None
    assert engine._h_last is None
    assert engine._provisional is None


def test_interrupted_prefill_retains_only_complete_exact_chunk():
    from runtime.engine import KVCache, StreamingEngine

    class State(KVCache):
        def __init__(self, offset):
            self._offset = offset
            self.releases = 0

        @property
        def offset(self):
            return self._offset

        def release(self):
            self.releases += 1

    survivor = State(128)
    partial = State(4096)
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(
        hot_prompt_kv=True, max_kv_mb=0,
        hot_prompt_kv_min_tokens=2048, hot_prompt_kv_slots=1)
    engine._hot_kv_persist = None
    engine._hot_prompt_slots = [SimpleNamespace(kv=survivor)]
    engine.last_kv = partial
    engine._h_window = object()
    engine._h_last = object()
    tokens = list(range(8192))
    capsules = (("inside", 100, 200), ("crosses", 4000, 4200))

    assert engine._retain_interrupted_prefill(
        tokens, partial, 4096, capsules)

    assert survivor.releases == 1
    assert len(engine._hot_prompt_slots) == 1
    slot = engine._hot_prompt_slots[0]
    assert slot.kv is partial
    assert slot.tokens == tuple(tokens[:4096])
    assert slot.logits is None and slot.prompt_logits is None
    assert slot.reusable_prefix == 4096
    assert slot.tool_capsules == (("inside", 100, 200),)
    assert engine._h_window is None and engine._h_last is None


def test_vision_fast_mode_quantizes_only_quality_gated_text_mlp():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_vl", tie_word_embeddings=True,
        index_topk=0, vision_config={"depth": 24}, num_experts=0,
        hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False,
    )
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3-vl"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits) == (
        "mxfp4", 32, 4)
    assert rc.quant_mlp
    assert not rc.quant_attention
    assert not rc.quant_lm_head
    assert not rc.quantize_tied_lm_head
    assert rc.resident_fast_decode
    assert rc.vision_max_patches == 1024
    assert rc.tool_pic


def test_small_lossless_vision_model_uses_exact_resident_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_vl", tie_word_embeddings=True,
        index_topk=0, vision_config={"depth": 24}, num_experts=0,
        hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False,
    )
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=4_000_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3-vl"), "lossless")

    rc = captured[0]
    assert rc.quant_bits == 0
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 2048
    assert not rc.embed_rows
    assert not rc.fused_swiglu
    assert not rc.hot_prompt_kv


def test_small_dense_lossless_mode_uses_exact_resident_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12,
        num_key_value_heads=2, head_dim=128, vocab_size=151936,
        attention_bias=True)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen-1.5b"), "lossless")

    rc = captured[0]
    assert rc.quant_bits == 0
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 2048
    assert not rc.fused_swiglu
    assert not rc.embed_rows
    assert rc.stepped_kv_threshold == 2048
    assert rc.hot_prompt_kv
    assert rc.hot_prompt_kv_slots == 1
    assert rc.hot_prompt_kv_min_tokens == 2048
    assert rc.prompt_kv_min_tokens == 2048


def test_generic_moe_fast_mode_quantizes_experts_but_preserves_sensitive_trunk():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-olmoe"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits) == ("mxfp4", 32, 4)
    assert rc.quant_mlp
    assert not rc.quant_attention
    assert not rc.quant_router
    assert not rc.quant_lm_head
    assert not rc.resident_fast_decode
    assert rc.resident_moe_decode
    assert rc.fused_swiglu
    assert rc.rerank_lm_head
    assert rc.rerank_lm_head_candidates == 32
    assert (rc.rerank_lm_head_mode, rc.rerank_lm_head_bits,
            rc.rerank_lm_head_group_size) == ("affine", 2, 64)
    assert rc.resident_attention_mode == "mxfp8"
    assert rc.resident_attention_bits == 8
    assert rc.stepped_kv_threshold == 1
    assert not rc.embed_rows
    assert rc.prefill_chunk_size == 2048
    assert rc.prefill_last_token_separate
    assert rc.tool_pic
    assert rc.expert_top_k_by_layer == ()


def test_lossless_olmoe_expands_exact_cache_when_governor_admits_it():
    from unittest.mock import patch

    from runtime.server import EngineManager

    made = []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc, self.closes = rc, 0
            self.cache = SimpleNamespace(max_bytes=15_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=13_840_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        engine = EngineManager().get(Path("/tmp/fake-olmoe"), "lossless")

    assert engine is made[0]
    assert len(made) == 1
    assert made[0].rc.max_weight_cache_mb == 14_809
    assert made[0].rc.quant_bits == 0
    assert not made[0].rc.resident_moe_decode


def test_lossless_olmoe_rebuilds_streamed_cache_when_admission_fails():
    from unittest.mock import patch

    from runtime.server import EngineManager

    made = []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc, self.closes = rc, 0
            self.cache = SimpleNamespace(max_bytes=8_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=13_840_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        engine = EngineManager().get(Path("/tmp/fake-olmoe"), "lossless")

    assert engine is made[1]
    assert len(made) == 2
    assert made[0].closes == 1
    assert made[0].rc.max_weight_cache_mb == 14_809
    assert made[1].rc.max_weight_cache_mb == 6000


def test_dense_embedding_pin_estimate_keeps_large_models_row_paged():
    from runtime.server import (
        _dense_fast_resident_bytes, _dense_lossless_resident_bytes)

    qwen_14b = SimpleNamespace(
        hidden_size=5120, intermediate_size=13824,
        num_hidden_layers=48, num_attention_heads=40,
        num_key_value_heads=8, head_dim=128, vocab_size=152064,
        attention_bias=True, tie_word_embeddings=False)
    assert _dense_fast_resident_bytes(qwen_14b) > int(7_000_000_000 * 0.85)

    qwen_1_5b = SimpleNamespace(
        hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12,
        num_key_value_heads=2, head_dim=128, vocab_size=151936,
        attention_bias=True, tie_word_embeddings=True)
    qwen_7b = SimpleNamespace(
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True, tie_word_embeddings=False)
    assert _dense_lossless_resident_bytes(qwen_1_5b) < int(6_000_000_000 * 0.85)
    assert _dense_lossless_resident_bytes(qwen_7b) > int(6_000_000_000 * 0.85)


def test_lossless_qwen_discovers_only_same_family_complete_mxfp4_draft(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen2.5-7B-Instruct"
    preferred = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    wrong_variant = tmp_path / "Qwen2.5-0.5B-Base-mlx-mxfp4"
    unvalidated_size = tmp_path / "Qwen2.5-0.5B-Instruct-mlx-mxfp4"
    for path in (target, preferred, wrong_variant, unvalidated_size):
        path.mkdir()
    common = {
        "model_type": "qwen2", "vision_config": None, "num_experts": 0,
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
    }
    (preferred / "config.json").write_text(json.dumps({
        **common, "hidden_size": 1536}))
    (preferred / "model.safetensors").write_bytes(b"complete")
    (wrong_variant / "config.json").write_text(json.dumps({
        **common, "hidden_size": 896}))
    (wrong_variant / "model.safetensors").write_bytes(b"complete")
    (unvalidated_size / "config.json").write_text(json.dumps({
        **common, "hidden_size": 896}))
    (unvalidated_size / "model.safetensors").write_bytes(b"complete")

    cfg = SimpleNamespace(hidden_size=3584)
    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        assert _speculative_draft_for(target, cfg) == preferred.resolve()


def test_speculative_draft_can_be_disabled_or_explicitly_overridden(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen2.5-7B-Instruct"
    draft = tmp_path / "custom-draft"
    target.mkdir()
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").write_bytes(b"complete")
    cfg = SimpleNamespace(hidden_size=3584)

    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "off"}):
        assert _speculative_draft_for(target, cfg) is None
    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": str(draft)}):
        assert _speculative_draft_for(target, cfg) == draft.resolve()


def test_qwen3_dspark_discovers_only_shape_compatible_block7(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen3-4B"
    good = tmp_path / "dspark_qwen3_4b_block7"
    wrong = tmp_path / "dspark_qwen3_8b_block7"
    for path in (target, good, wrong):
        path.mkdir()
    common = {
        "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3",
        "vocab_size": 151936, "num_target_layers": 36, "block_size": 7,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }
    (good / "config.json").write_text(json.dumps({
        **common, "hidden_size": 2560}))
    (good / "model.safetensors").write_bytes(b"complete")
    (wrong / "config.json").write_text(json.dumps({
        **common, "hidden_size": 4096}))
    (wrong / "model.safetensors").write_bytes(b"complete")
    cfg = SimpleNamespace(hidden_size=2560, vocab_size=151936,
                          num_hidden_layers=36)

    with patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "auto"}):
        assert _dspark_draft_for(target, cfg) == good.resolve()
    with patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "off"}):
        assert _dspark_draft_for(target, cfg) is None


def test_engine_manager_wraps_streamed_lossless_qwen3_with_dspark(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen3-4B"
    draft_path = tmp_path / "dspark_qwen3_4b_block7"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            # Simulate a governor-constrained host: the fitted target cache is
            # below the exact 4B footprint, so DSpark remains useful.
            self.cache = SimpleNamespace(max_bytes=6_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeDSparkEngine:
        def __init__(self, target, draft_dir, *, max_draft_tokens,
                     max_prompt_tokens, confidence_threshold,
                     prompt_cache_min_tokens):
            self.target = target
            self.draft_dir = Path(draft_dir)
            self.max_draft_tokens = max_draft_tokens
            self.max_prompt_tokens = max_prompt_tokens
            self.confidence_threshold = confidence_threshold
            self.prompt_cache_min_tokens = prompt_cache_min_tokens
            self.closes = 0

        def close(self):
            self.closes += 1
            self.target.close()

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine", FakeDSparkEngine), \
         patch.dict("os.environ", {
             "VMODEL_DSPARK_DRAFT": "auto",
             "VMODEL_DSPARK_MAX_DRAFT_TOKENS": "4",
             "VMODEL_DSPARK_MAX_PROMPT_TOKENS": "2048",
         }):
        wrapped = manager.get(target_path, "lossless")
        assert isinstance(wrapped, FakeDSparkEngine)
        assert wrapped.draft_dir == draft_path
        assert wrapped.max_draft_tokens == 4
        assert wrapped.max_prompt_tokens == 2048
        assert wrapped.confidence_threshold == 0.0
        assert wrapped.prompt_cache_min_tokens == 2048
        assert len(made) == 1
        assert made[0].rc.max_weight_cache_mb > 9000
        assert made[0].rc.prefetch_workers == 2
        assert made[0].rc.prefetch_depth == 4
        assert made[0].rc.hot_prompt_kv
        assert made[0].rc.hot_prompt_kv_min_tokens == 2048

        manager.get(target_path, "fast")

    assert wrapped.closes == 1
    assert wrapped.target.closes == 1


def test_engine_manager_prefers_full_resident_qwen3_when_governor_admits_it(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen3-4B"
    draft_path = tmp_path / "dspark_qwen3_4b_block7"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=10_000_000_000)
            self.governor = None

        def close(self):
            self.closes += 1

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine",
               side_effect=AssertionError("resident target should win")), \
         patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "auto"}):
        engine = manager.get(target_path, "lossless")

    assert isinstance(engine, FakeEngine)
    assert engine.rc.resident_fast_decode
    assert engine.rc.resident_fast_prefill_limit == 2048
    assert engine.rc.stepped_kv_threshold == 2048
    assert engine.rc.hot_prompt_kv
    assert engine.rc.hot_prompt_kv_min_tokens == 2048


def test_engine_manager_wraps_large_lossless_qwen_and_swaps_both_owners(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            # Simulate the constrained target machine: a model-sized probe is
            # fitted below the complete exact 7B footprint.
            self.cache = SimpleNamespace(max_bytes=6_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeSpeculativeEngine:
        def __init__(self, target, draft, *, k, max_prompt_tokens,
                     prompt_cache_min_tokens):
            self.target, self.draft = target, draft
            self.k, self.max_prompt_tokens, self.closes = k, max_prompt_tokens, 0
            self.prompt_cache_min_tokens = prompt_cache_min_tokens

        def close(self):
            self.closes += 1
            self.draft.close()
            self.target.close()

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine", FakeSpeculativeEngine), \
         patch.dict("os.environ", {
             "VMODEL_SPECULATIVE_DRAFT": "auto",
             "VMODEL_SPECULATIVE_K": "6",
             "VMODEL_SPECULATIVE_MAX_PROMPT_TOKENS": "2048",
         }):
        wrapped = manager.get(target_path, "lossless")
        assert isinstance(wrapped, FakeSpeculativeEngine)
        assert wrapped.prompt_cache_min_tokens == 2048
        assert len(made) == 3
        assert made[0].rc.max_weight_cache_mb > 16_000
        assert made[0].closes == 1
        assert made[1].path == target_path
        assert made[1].rc.max_weight_cache_mb == 6000
        assert made[1].rc.prefetch_workers == 2
        assert made[1].rc.prefetch_depth == 4
        assert made[2].path == draft_path
        assert made[2].rc.resident_fast_decode
        assert made[2].rc.max_weight_cache_mb == 1200

        manager.get(target_path, "fast")

    assert wrapped.closes == 1
    assert (wrapped.target.closes, wrapped.draft.closes) == (1, 1)


def test_engine_manager_prefers_full_resident_qwen2_when_governor_admits_it(
        tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=18_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine",
               side_effect=AssertionError("resident target should win")), \
         patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        engine = manager.get(target_path, "lossless")

    assert engine is made[1]
    assert len(made) == 2
    assert made[0].closes == 1
    assert made[0].rc.max_weight_cache_mb > 16_000
    assert made[0].rc.embed_rows
    assert not made[0].rc.resident_fast_decode
    assert made[1].rc.max_weight_cache_mb > 16_000
    assert not made[1].rc.embed_rows
    assert made[1].rc.resident_fast_decode
    assert made[1].rc.resident_fast_prefill_limit == 2048
    assert made[1].rc.stepped_kv_threshold == 2048
    assert made[1].rc.hot_prompt_kv
    assert made[1].rc.hot_prompt_kv_min_tokens == 2048
    assert made[1].rc.prompt_kv_min_tokens == 2048


def test_qwen2_admission_race_falls_back_to_streamed_speculation(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeGovernor:
        def __init__(self, fitted):
            self.fitted = fitted

        def fit_cache_to_live_headroom(self):
            return self.fitted

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            index = len(made)
            fitted = (18_000_000_000 if index == 0 else
                      8_000_000_000 if index == 1 else
                      rc.max_weight_cache_mb * 1_000_000)
            self.cache = SimpleNamespace(
                max_bytes=rc.max_weight_cache_mb * 1_000_000)
            self.governor = FakeGovernor(fitted)
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeSpeculativeEngine:
        def __init__(self, target, draft, **_kwargs):
            self.target, self.draft = target, draft

        def close(self):
            self.draft.close()
            self.target.close()

    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine", FakeSpeculativeEngine), \
         patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        wrapped = EngineManager().get(target_path, "lossless")

    assert isinstance(wrapped, FakeSpeculativeEngine)
    assert len(made) == 4
    assert made[0].closes == 1
    assert made[1].closes == 1
    assert wrapped.target is made[2]
    assert wrapped.target.rc.max_weight_cache_mb == 6000
    assert not wrapped.target.rc.resident_fast_decode
    assert wrapped.draft is made[3]


def test_fast_long_prefix_is_distinct_from_fast_prefix():
    assert split_model_mode("lossy-long-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast-long")
    assert split_model_mode("lossy-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast")


def test_derived_quantized_checkpoint_is_advertised_only_as_lossy(tmp_path):
    from unittest.mock import patch

    released = tmp_path / "released"
    derived = tmp_path / "derived"
    released.mkdir()
    derived.mkdir()
    (released / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
    (derived / "config.json").write_text(json.dumps({
        "model_type": "olmoe",
        "voom_quantization": {"profile": "experts", "source": str(released)},
    }))

    with patch("runtime.server._registry", return_value={
            "released": released, "derived": derived}):
        ids = _advertised_model_ids()
    assert "released" in ids
    assert "lossy-released" in ids
    assert "derived" not in ids
    assert "lossy-derived" in ids


def test_registry_does_not_advertise_auxiliary_embedding_encoders(tmp_path):
    from unittest.mock import patch

    from runtime.local_config import StorageConfig

    models = tmp_path / "models"
    chat = models / "Qwen-test"
    embed = models / "tool-embed-bge-small-en-v1.5"
    chat.mkdir(parents=True)
    embed.mkdir()
    (chat / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (embed / "config.json").write_text(json.dumps({"model_type": "bert"}))
    with patch("runtime.server.ROOT", tmp_path), \
         patch("runtime.local_config.get_storage_config",
               return_value=StorageConfig()):
        registry = _registry()
    assert "Qwen-test" in registry
    assert "tool-embed-bge-small-en-v1.5" not in registry


def test_engine_manager_rejects_derived_checkpoint_as_lossless(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    (tmp_path / "config.json").write_text(json.dumps({
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path):
        try:
            EngineManager().get(tmp_path, "lossless")
        except RequestValidationError as error:
            assert "vOOM-derived lossy artifact" in str(error)
        else:
            raise AssertionError("derived checkpoint was accepted as lossless")


def test_base_olmoe_prefers_complete_expert_mxfp4_sibling(tmp_path):
    source = tmp_path / "OLMoE"
    q4 = tmp_path / "OLMoE-mlx-expert-mxfp4"
    q8 = tmp_path / "OLMoE-mlx-expert-mxfp8"
    source.mkdir()
    for path, mode, bits in ((q4, "mxfp4", 4), (q8, "mxfp8", 8)):
        path.mkdir()
        (path / "config.json").write_text(json.dumps({
            "model_type": "olmoe",
            "quantization": {"mode": mode, "bits": bits, "group_size": 32},
            "voom_quantization": {
                "profile": "experts", "source": str(source.resolve())},
        }))
        (path / "model.safetensors.index.json").write_text("{}")

    assert _preferred_fast_artifact(source) == q4
    assert _preferred_fast_artifact(q8) == q8


def test_execution_profile_discloses_effective_derived_artifact(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(
            quantization={"mode": "mxfp4", "bits": 4, "group_size": 32},
            on_disk_quantized=True),
        rc=SimpleNamespace(quant_bits=4, quant_mode="mxfp4", quant_group_size=32),
    )

    assert _execution_profile_fields(engine) == {
        "vmodel_checkpoint": tmp_path.name,
        "vmodel_weight_profile": "experts-mxfp4-q4-g32",
    }


def test_execution_profile_discloses_reranked_head(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(
            quantization={"mode": "mxfp4", "bits": 4, "group_size": 32},
            on_disk_quantized=True),
        rc=SimpleNamespace(
            quant_bits=4, quant_mode="mxfp4", quant_group_size=32,
            rerank_lm_head=True, rerank_lm_head_mode="mxfp4",
            rerank_lm_head_bits=4, rerank_lm_head_group_size=32,
            rerank_lm_head_candidates=32,
            resident_attention_mode="mxfp8",
            resident_attention_bits=8,
            resident_attention_group_size=32),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "experts-mxfp4-q4-g32+head-mxfp4-q4-g32-rerank32"
        "+attn-mxfp8-q8-g32")


def test_execution_profile_discloses_olmoe_top_k_schedule(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(quantization={}, on_disk_quantized=False),
        rc=SimpleNamespace(
            quant_bits=0,
            rerank_lm_head=False,
            resident_attention_mode="",
            expert_top_k_by_layer=(7, 7, 8, 8),
        ),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "released+olmoe-topk-7.7.8.8")


def test_mxfp8_olmoe_profile_gets_resident_admission_budget(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp8", "bits": 8, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(tmp_path, "fast")

    assert captured[0].resident_moe_decode
    assert captured[0].max_weight_cache_mb == 9000
    assert captured[0].resident_attention_mode == ""
    assert captured[0].stepped_kv_threshold == 1


def test_active_context_limit_uses_stricter_runtime_correctness_bound():
    engine = _fake_engine(model_limit=1_000_000, context_bound=2_048)
    assert _active_context_limit(engine) == 2_048
    assert _validate_context_budget(
        engine, 2_000, 48, prompt_label="prompt", output_label="max_tokens") == 2_048
    try:
        _validate_context_budget(
            engine, 2_000, 49, prompt_label="prompt", output_label="max_tokens")
    except RequestValidationError as error:
        assert "active context limit=2048" in str(error)
    else:
        raise AssertionError("runtime correctness-bound overflow was accepted")


def test_chat_prompt_rejects_correctness_bound_before_generation():
    with tempfile.TemporaryDirectory() as directory:
        engine = _fake_engine(model_limit=1_000_000, context_bound=40)
        args = (engine, Path(directory), [{"role": "user", "content": "x"}],
                "low", [], [], "lossless")
        _prompt, prompt_tokens, *_rest, metadata = _prepare_chat_prompt(*args, 1)
        assert metadata["context_limit"] == 40
        remaining = 40 - prompt_tokens
        _prepare_chat_prompt(*args, remaining)
        try:
            _prepare_chat_prompt(*args, remaining + 1)
        except RequestValidationError as error:
            assert "active context limit=40" in str(error)
        else:
            raise AssertionError("chat correctness-bound overflow was accepted")


def test_invalid_image_payload_is_a_request_validation_error():
    try:
        _load_vision_images(["data:image/png;base64,%%%"])
    except RequestValidationError as error:
        assert str(error) == "invalid image 1: image data URI contains invalid base64"
    else:
        raise AssertionError("invalid image payload was accepted")


def test_positive_token_limit_rejects_zero_negative_bool_fraction_and_text():
    assert _positive_token_limit(1, "max_tokens") == 1
    assert _positive_token_limit("2", "max_tokens") == 2
    for value in (0, -1, True, 1.5, "nope", None):
        try:
            _positive_token_limit(value, "max_tokens")
        except RequestValidationError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(f"invalid token limit accepted: {value!r}")


def test_omitted_output_budget_is_eos_safety_ceiling_not_legacy_64():
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VMODEL_OMITTED_MAX_OUTPUT_TOKENS", None)
        assert _omitted_output_token_limit() == 4096
    with patch.dict(os.environ, {"VMODEL_OMITTED_MAX_OUTPUT_TOKENS": "768"}):
        assert _omitted_output_token_limit() == 768
    with patch.dict(os.environ, {"VMODEL_OMITTED_MAX_OUTPUT_TOKENS": "0"}):
        try:
            _omitted_output_token_limit()
        except RequestValidationError:
            pass
        else:
            raise AssertionError("accepted a zero omitted-output safety ceiling")


def test_responses_mixed_text_and_tool_call_keeps_both_items_and_id():
    text = (
        "I will check.\n"
        '<tool_call>{"name":"weather","arguments":{"city":"Chicago"}}</tool_call>'
        "\nThen I will summarize."
    )
    content, output = _responses_output_items(
        text, [{"type": "function", "function": {
            "name": "weather", "parameters": {}}}], "qwen2", "msg_fixed")
    assert "I will check." in content
    assert "Then I will summarize." in content
    assert [item["type"] for item in output] == ["message", "function_call"]
    assert output[0]["id"] == "msg_fixed"
    assert output[0]["content"][0]["text"] == content
    assert output[1]["name"] == "weather"


def test_responses_incomplete_container_has_incomplete_message_item():
    _content, output = _responses_output_items(
        "partial", [], "qwen2", "msg_partial", message_status="incomplete")
    assert output[0]["status"] == "incomplete"


def test_release_generation_state_drops_every_previous_request_owner():
    from runtime.request_state import release_generation_state

    marker = object()
    owner = SimpleNamespace(
        _hot_prompt_slots=[marker, marker],
        last_kv=marker, _h_window=marker, _h_last=marker, _provisional=marker,
    )
    release_generation_state(owner)
    assert owner._hot_prompt_slots == []
    for name in ("last_kv", "_h_window", "_h_last", "_provisional"):
        assert getattr(owner, name) is None


def test_release_generation_state_releases_aliased_refcounted_kv_once():
    from runtime.request_state import release_generation_state

    class Releasable:
        calls = 0

        def release(self):
            self.calls += 1

    state = Releasable()
    owner = SimpleNamespace(
        _hot_prompt_slots=[SimpleNamespace(kv=state)],
        last_kv=state, _h_window=None, _h_last=None, _provisional=None,
    )
    release_generation_state(owner)
    assert state.calls == 1


def test_openai_finish_reason_distinguishes_length_eos_and_tool_calls():
    assert _openai_finish_reason({"termination_reason": "length"}) == "length"
    assert _openai_finish_reason({"termination_reason": "eos"}) == "stop"
    assert _openai_finish_reason(
        {"termination_reason": "length"}, has_tool_calls=True) == "tool_calls"


def test_stream_holdback_catches_complete_marker_with_trailing_text():
    markers = ("<tool_call>",)
    assert _safe_emit_len("<tool_call>", markers) == 0
    assert _safe_emit_len("<tool_call>{", markers) == 0
    assert _safe_emit_len("hello<tool_call>{", markers) == len("hello")
    assert _safe_emit_len("hello", markers) == len("hello")


def test_stream_holdback_survives_marker_split_across_decode_pieces():
    markers = ("<tool_call>",)
    pending = ""
    emitted = ""
    for piece in ("hello<tool_", "call>{"):
        pending += piece
        safe = _safe_emit_len(pending, markers)
        emitted += pending[:safe]
        pending = pending[safe:]
    assert emitted == "hello"
    assert pending == "<tool_call>{"
    assert _safe_emit_len(pending, markers) == 0


def test_marker_holdback_streams_safe_text_and_replays_post_call_text():
    holdback = _MarkerHoldback(("<tool_call>",))
    assert holdback.feed("hello<tool_") == "hello"
    assert holdback.feed("call>{\"name\":\"x\"}") == ""
    assert holdback.holding
    assert holdback.final_remainder("hello after") == " after"


def test_hidden_decision_streams_after_first_irreversible_prefix_mismatch():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    for piece in ("H", "ello", " from", " the model"):
        decision.feed(piece)
    decision.finish_direct("Hello from the model")
    assert decision.branch == "direct"
    assert "".join(emitted) == "Hello from the model"
    assert len(emitted) == 4


def test_hidden_decision_holds_marker_at_every_decode_split():
    marker = "<tool_call>"
    for split in range(len(marker) + 1):
        emitted = []
        decision = _HiddenDecisionStream("qwen3", emitted.append)
        decision.feed(" \n" + marker[:split])
        decision.feed(marker[split:] + '{"name":"vmodel_search_tools"}')
        assert decision.branch == "tool"
        assert emitted == []


def test_hidden_decision_releases_marker_like_direct_text():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    decision.feed("<tool_calls> is ordinary text")
    decision.finish_direct("<tool_calls> is ordinary text")
    assert decision.branch == "direct"
    assert "".join(emitted) == "<tool_calls> is ordinary text"


def test_hidden_decision_never_leaks_late_virtual_marker():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    before = "Before. "
    marker = (
        '<tool_call>{"name":"vmodel_search_tools",'
        '"arguments":{"query":"browser"}}</tool_call>')
    after = " After."
    decision.feed(before)
    decision.feed("<tool_")
    decision.feed(marker[len("<tool_"):] + after)
    assert decision.branch == "direct"
    assert decision.late_marker_detected
    assert "<tool" not in "".join(emitted)
    virtual, _raw = _hidden_tool_search_pair()
    content, calls = _parse_request_tool_calls(
        before + marker + after, [virtual], "qwen3", allow_parallel=False)
    assert [call["function"]["name"] for call in calls] == [
        "vmodel_search_tools"]
    decision.finish_direct(content)
    assert "".join(emitted) == before + after


def test_hidden_decision_handles_harmony_spacing_and_final_channel():
    emitted = []
    tool_decision = _HiddenDecisionStream("gpt_oss", emitted.append)
    tool_decision.feed("commentary   to=functions.vmodel_search_tools")
    assert tool_decision.branch == "tool"
    assert emitted == []

    direct = _HiddenDecisionStream("gpt_oss", emitted.append)
    direct.feed("<|channel|>final")
    direct.finish_direct("<|channel|>final")
    assert direct.branch == "direct"
    assert "".join(emitted) == "<|channel|>final"


def test_resident_adjusted_transient_excludes_persistent_cache_growth():
    from runtime.engine import _resident_adjusted_transient

    assert _resident_adjusted_transient(1_000, 2_500, 2_500) == 0
    assert _resident_adjusted_transient(1_000, 2_500, 2_900) == 400
    assert _resident_adjusted_transient(2_500, 1_000, 2_900) == 400


def test_cache_io_delta_reports_only_current_request():
    from types import SimpleNamespace

    from runtime.engine import _cache_io_snapshot, _record_cache_io_delta

    cache_stats = SimpleNamespace(
        hits=10, misses=20, evictions=3, bytes_read=1_000)
    engine = SimpleNamespace(
        cache=SimpleNamespace(
            stats=cache_stats, total_bytes=400, max_bytes=800),
        store=SimpleNamespace(fast_tier_bytes=100, archive_bytes=900),
        governor=SimpleNamespace(reservations=2, reservation_failures=1),
        expert_hits=4, expert_misses=5,
        _layer_transient=60, _token_transient=70,
    )
    before = _cache_io_snapshot(engine)
    cache_stats.hits += 2
    cache_stats.misses += 3
    cache_stats.evictions += 4
    cache_stats.bytes_read += 5_000
    engine.expert_hits += 6
    engine.expert_misses += 7
    engine.governor.reservations += 8
    engine.store.fast_tier_bytes += 9
    engine.store.archive_bytes += 10
    stats = {}
    _record_cache_io_delta(engine, before, stats)

    assert stats["weight_cache_hits"] == 2
    assert stats["weight_cache_misses"] == 3
    assert stats["weight_cache_evictions"] == 4
    assert stats["weight_store_bytes_read"] == 5_000
    assert stats["expert_cache_hits"] == 6
    assert stats["expert_cache_misses"] == 7
    assert stats["governor_reservations"] == 8
    assert stats["weight_fast_tier_bytes"] == 9
    assert stats["weight_archive_bytes"] == 10
    assert stats["weight_cache_resident_bytes"] == 400
    assert stats["layer_transient_bytes"] == 60


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
