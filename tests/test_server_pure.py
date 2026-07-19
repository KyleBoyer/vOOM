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
                            _execution_profile_fields,
                            _load_vision_images,
                            _MarkerHoldback,
                            _chat_prompt,
                            _positive_token_limit,
                            _prepare_chat_prompt, _render_template,
                            _compiled_template,
                            _openai_finish_reason, _responses_output_items, _safe_emit_len,
                            _parse_request_tool_calls, _tool_request_controls,
                            _preferred_fast_artifact,
                            _dspark_draft_for,
                            _speculative_draft_for,
                            _request_reasoning_controls, _request_sampling,
                            _tool_capsule_spans,
                            _vision_protocol_timing,
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


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
