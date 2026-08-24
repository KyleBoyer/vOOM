"""Pure HTTP/config gates for explicit Qwen layer-stationary serving."""

from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from runtime.qwen35_multi_request_server import (
    QwenMultiRequestServerConfig,
    QwenMultiRequestValidationError,
    parse_batch_payload,
)
from runtime.server import Handler
import runtime.qwen35_multi_request_server as batch_module
import runtime.server as server_module


_ENV_NAMES = (
    "VMODEL_QWEN_MULTI_REQUEST_BATCH",
    "VMODEL_QWEN_MULTI_REQUEST_MAX_REQUESTS",
    "VMODEL_QWEN_MULTI_REQUEST_MAX_PROMPT_TOKENS",
    "VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_PROMPT_TOKENS",
    "VMODEL_QWEN_MULTI_REQUEST_MAX_OUTPUT_TOKENS",
    "VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_OUTPUT_TOKENS",
)


def _clear_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_server_config_is_default_off_bounded_and_identified(monkeypatch):
    _clear_env(monkeypatch)
    config = QwenMultiRequestServerConfig.from_env()
    assert not config.enabled
    assert config.max_requests == 4
    assert config.max_total_prompt_tokens == 8192
    assert config.max_total_output_tokens == 512
    assert config.identity == "qwen-ls-http-v1:r4:p4096:pt8192:o256:ot512"

    monkeypatch.setenv("VMODEL_QWEN_MULTI_REQUEST_BATCH", "1")
    monkeypatch.setenv("VMODEL_QWEN_MULTI_REQUEST_MAX_REQUESTS", "3")
    monkeypatch.setenv(
        "VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_OUTPUT_TOKENS", "300")
    changed = QwenMultiRequestServerConfig.from_env()
    assert changed.enabled
    assert changed.max_requests == 3
    assert changed.max_total_output_tokens == 300
    assert changed.identity != config.identity


@pytest.mark.parametrize("name,value", [
    ("VMODEL_QWEN_MULTI_REQUEST_BATCH", "auto"),
    ("VMODEL_QWEN_MULTI_REQUEST_MAX_REQUESTS", "17"),
    ("VMODEL_QWEN_MULTI_REQUEST_MAX_PROMPT_TOKENS", "0"),
    ("VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_OUTPUT_TOKENS", "nan"),
])
def test_server_config_rejects_unbounded_or_malformed_values(
        monkeypatch, name, value):
    _clear_env(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(QwenMultiRequestValidationError):
        QwenMultiRequestServerConfig.from_env()


def test_wire_parser_accepts_heterogeneous_deterministic_requests():
    config = QwenMultiRequestServerConfig(
        enabled=True, max_requests=2, max_output_tokens=8,
        max_total_output_tokens=12)
    parsed = parse_batch_payload({
        "model": "Qwen3.5",
        "vmodel_mode": "lossless",
        "stream": False,
        "requests": [
            {
                "id": "short",
                "prompt": "a",
                "max_tokens": 4,
                "temperature": 0,
                "repetition_penalty": 1.1,
            },
            {
                "id": "long",
                "prompt": "different prompt",
                "max_tokens": 8,
                "temperature": 0.8,
                "top_k": 1,
                "stop": ["done"],
            },
        ],
    }, config)
    assert [item.request_id for item in parsed] == ["short", "long"]
    assert [item.max_tokens for item in parsed] == [4, 8]
    assert all(item.sampling.is_greedy for item in parsed)
    assert parsed[0].sampling.repetition_penalty == 1.1
    assert parsed[1].stop == ("done",)


def test_wire_parser_rejects_global_rng_sampling_and_total_budget():
    config = QwenMultiRequestServerConfig(
        enabled=True, max_requests=2, max_output_tokens=8,
        max_total_output_tokens=8)
    base = {
        "model": "Qwen3.5",
        "requests": [{
            "id": "a", "prompt": "a", "max_tokens": 4,
            "temperature": 0.5,
        }],
    }
    with pytest.raises(QwenMultiRequestValidationError, match="global MLX RNG"):
        parse_batch_payload(base, config)

    over = {
        "model": "Qwen3.5",
        "requests": [
            {"id": "a", "prompt": "a", "max_tokens": 5},
            {"id": "b", "prompt": "b", "max_tokens": 4},
        ],
    }
    with pytest.raises(QwenMultiRequestValidationError, match="total output"):
        parse_batch_payload(over, config)


class _Connection:
    def __init__(self):
        self.timeout = None

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value


def test_versioned_http_route_is_admitted_and_decoded(monkeypatch):
    payload = {
        "model": "Qwen3.5",
        "requests": [{"id": "a", "prompt": "x", "max_tokens": 1}],
    }
    raw = json.dumps(payload).encode()
    handler = Handler.__new__(Handler)
    handler.path = "/v1/qwen/layer-stationary/completions?probe=1"
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = BytesIO(raw)
    handler.connection = _Connection()
    responses = []
    handler._json = lambda code, body: responses.append((code, body))

    parsed = handler._read_json_request()
    assert responses == []
    assert parsed == (raw, payload, len(raw))
    assert handler._route() == "/qwen/layer-stationary/completions"


def test_locked_dispatch_uses_explicit_batch_handler_before_generic_protocol():
    payload = {"requests": []}
    handler = Handler.__new__(Handler)
    handler._parsed_request = (b"{}", payload, 2)
    handler._route = lambda: "/qwen/layer-stationary/completions"
    calls = []
    handler._do_qwen_layer_stationary_batch = (
        lambda req, length: calls.append((req, length)) or "served")
    assert handler._do_post_locked() == "served"
    assert calls == [(payload, 2)]


def test_endpoint_remains_unexposed_when_default_off(monkeypatch):
    _clear_env(monkeypatch)
    handler = Handler.__new__(Handler)
    handler.headers = {}
    responses = []
    handler._json = lambda code, body: responses.append((code, body))
    result = handler._do_qwen_layer_stationary_batch({}, 2)
    assert result is None
    assert responses[0][0] == 404
    assert "disabled" in responses[0][1]["error"]


def test_http_handler_prepares_private_namespaces_and_returns_telemetry(
        monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("VMODEL_QWEN_MULTI_REQUEST_BATCH", "1")

    class _Tokenizer:
        @staticmethod
        def encode(text):
            return SimpleNamespace(ids=[ord(value) for value in str(text)])

    target = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="qwen3_5", max_position_embeddings=128),
        tokenizer=_Tokenizer(),
        effective_max_position_embeddings=128,
        rope_profile="released",
    )
    monkeypatch.setattr(server_module, "_resolve", lambda _model: tmp_path)
    get_calls = []
    monkeypatch.setattr(
        server_module.MANAGER,
        "get",
        lambda *args, **kwargs: get_calls.append((args, kwargs)) or target,
    )
    seen = []

    def fake_run(engine, items, *, max_requests, bootstrap_generate):
        assert engine is target
        assert max_requests == 4
        assert callable(bootstrap_generate)
        seen.extend(items)
        return {
            "choices": [
                {
                    "id": item.request_id,
                    "text": item.request_id,
                    "tokens": [1],
                    "finish_reason": "length",
                    "termination_reason": "length",
                    "stop_sequence": None,
                    "prompt_tokens": len(item.prompt_token_ids),
                    "completion_tokens": 1,
                    "sampling": "greedy",
                }
                for item in items
            ],
            "telemetry": {
                "cache_identity_policy": "private-kv-and-kda-per-request",
                "layer_page_get_calls": 2,
            },
        }

    monkeypatch.setattr(batch_module, "run_qwen_multi_request_batch", fake_run)
    payload = {
        "model": "Qwen3.5",
        "requests": [
            {"id": "a", "prompt": "x", "max_tokens": 2},
            {"id": "b", "prompt": "yz", "max_tokens": 3},
        ],
    }
    handler = Handler.__new__(Handler)
    handler.headers = {}
    responses = []
    handler._json = lambda code, body: responses.append((code, body))
    handler._do_qwen_layer_stationary_batch(payload, 123)

    assert len(get_calls) == 1
    assert len(seen) == 2
    assert [item.prompt_token_ids for item in seen] == [(120,), (121, 122)]
    namespaces = [item.prompt.cache_namespace for item in seen]
    assert len(set(namespaces)) == 2
    assert all(value.startswith("qwen-layer-stationary:")
               for value in namespaces)
    code, response = responses[0]
    assert code == 200
    assert response["object"] == "qwen.layer_stationary.batch_completion"
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    telemetry = response["vmodel_multi_request_telemetry"]
    assert telemetry["cache_identity_policy"] == (
        "private-kv-and-kda-per-request")
    assert telemetry["config_identity"].startswith("qwen-ls-http-v1:")
