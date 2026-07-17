"""Integration test using the REAL `openai` and `anthropic` Python client
libraries against a live runtime.server instance, covering all three
protocol surfaces (OpenAI Chat Completions, OpenAI Responses, Anthropic
Messages) and confirming routing works with AND without the `/v1/` prefix
(2026-07-13, user request: "the http server should work correctly with
whatever protocol... based on which URL endpoint is being called. With or
without the /v1/ prefix too").

Starts one real server subprocess on a dedicated test port, local
SmolLM2-135M (no download).

  .venv/bin/python tests/test_multi_protocol_clients.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = 8098  # distinct from both the default 8077 and test_openai_client_integration's 8099


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


def _start_server(env=None):
    proc = subprocess.Popen(
        [sys.executable, "-m", "runtime.server", "--port", str(PORT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, **(env or {})},
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


def test_openai_responses_api_with_v1_prefix():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="not-needed")
        resp = client.responses.create(
            model="SmolLM2-135M", max_output_tokens=8, input="The capital of France is")
        # constructing `resp` validates our JSON against the SDK's Pydantic schema
        assert resp.object == "response"
        assert resp.status in ("completed", "incomplete")
        assert resp.output[0].type == "message"
        assert resp.output[0].status == resp.status
        if resp.status == "incomplete":
            assert resp.incomplete_details.reason == "max_output_tokens"
        assert resp.output[0].content[0].type == "output_text"
        assert isinstance(resp.output[0].content[0].text, str)
        assert resp.usage.output_tokens == 8
        assert resp.usage.total_tokens == resp.usage.input_tokens + resp.usage.output_tokens
    finally:
        _stop_server(proc)


def test_openai_responses_api_without_v1_prefix():
    """Same request, but the client's base_url omits /v1 — routing must
    still resolve /responses correctly."""
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}", api_key="not-needed")
        resp = client.responses.create(
            model="SmolLM2-135M", max_output_tokens=8, input="The capital of France is")
        assert resp.object == "response"
        assert resp.output[0].content[0].text
    finally:
        _stop_server(proc)


def test_anthropic_messages_api_real_client():
    """The Anthropic SDK hardcodes the `/v1/messages` path suffix onto
    `base_url` itself (confirmed directly: `Anthropic(base_url=X).base_url
    == X`, unmodified — the SDK's `.messages.create()` call appends
    "/v1/messages" internally) — there is no client-configurable way to
    make it hit a bare `/messages` path, so this test exercises the
    WITH-/v1 route (the only one a real Anthropic client can ever reach);
    see test_bare_messages_path_via_raw_http below for the bare-path route,
    tested directly since no real client can exercise it."""
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="not-needed")
        resp = client.messages.create(
            model="SmolLM2-135M", max_tokens=8,
            messages=[{"role": "user", "content": "The capital of France is"}])
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.content[0].type == "text"
        assert isinstance(resp.content[0].text, str)
        assert resp.usage.output_tokens == 8
        assert resp.stop_reason in ("end_turn", "max_tokens", "stop_sequence")
    finally:
        _stop_server(proc)


def test_bare_messages_path_via_raw_http():
    """No real Anthropic client can hit a bare /messages path (see note
    above), but our server's routing must still accept it — verified via a
    raw HTTP POST instead of the SDK."""
    import json
    import urllib.request

    proc = _start_server()
    try:
        payload = {"model": "SmolLM2-135M", "max_tokens": 8,
                   "messages": [{"role": "user", "content": "The capital of France is"}]}
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/messages", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        assert body["type"] == "message"
        assert body["content"][0]["text"]
    finally:
        _stop_server(proc)


def test_anthropic_system_prompt_and_stop_sequences():
    from anthropic import Anthropic

    proc = _start_server()
    try:
        client = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="not-needed")
        baseline = client.messages.create(
            model="SmolLM2-135M", max_tokens=16,
            messages=[{"role": "user", "content": "The capital of France is"}])
        full_text = baseline.content[0].text
        assert len(full_text) > 4
        stop_str = full_text[2:5]

        stopped = client.messages.create(
            model="SmolLM2-135M", max_tokens=16, stop_sequences=[stop_str],
            messages=[{"role": "user", "content": "The capital of France is"}])
        assert stop_str not in stopped.content[0].text
        assert stopped.stop_reason == "stop_sequence"
    finally:
        _stop_server(proc)


def test_malformed_http_requests_fail_as_4xx_before_model_lookup():
    import http.client
    import json

    proc = _start_server()
    try:
        def post(path, body):
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
            connection.request(
                "POST", path, body=body,
                headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
            return response.status, payload

        status, body = post("/v1/responses", b"{")
        assert status == 400 and "invalid JSON" in body["error"]

        status, body = post("/v1/responses", b'{"input":NaN}')
        assert status == 400 and "non-finite JSON constant" in body["error"]

        status, body = post("/v1/responses", b"[]")
        assert status == 400 and "JSON object" in body["error"]

        status, body = post(
            "/v1/responses", json.dumps({"stream": "false"}).encode())
        assert status == 400 and body["error"] == "stream must be a boolean"

        status, body = post(
            "/v1/responses", json.dumps({"stop": [123]}).encode())
        assert status == 400 and "list of strings" in body["error"]

        for path, payload, expected in (
            ("/v1/responses", {"model": 123}, "model must"),
            ("/v1/responses", {"tools": {}}, "tools must"),
            ("/v1/responses", {"input": {}}, "Responses input"),
            ("/v1/responses", {"vmodel_mode": 7}, "vmodel_mode must"),
            ("/v1/chat/completions", {
                "stream": True, "stream_options": {"include_usage": "yes"}},
             "stream_options.include_usage must"),
            ("/v1/chat/completions", {
                "stream": True, "stream_options": {"include_obfuscation": True}},
             "stream obfuscation is not supported"),
            ("/v1/responses", {"parallel_tool_calls": "false"},
             "parallel_tool_calls must"),
            ("/v1/responses", {"temperature": "0.7"},
             "temperature must be a finite number"),
            ("/v1/responses", {"top_p": 2}, "top_p must be between"),
            ("/v1/responses", {"tool_choice": "required"},
             "requires at least one tool"),
            ("/v1/responses", {"tools": [{"type": "function", "name": ""}]},
             "function name"),
            ("/v1/responses", {"tools": [{"type": "web_search_preview"}]},
             "Responses function tool"),
            ("/v1/messages", {"stop_sequences": "bad"},
             "stop_sequences must"),
            ("/v1/messages", {"tools": [{"name": "x", "input_schema": None}]},
             "input_schema must be an object"),
            ("/v1/messages", {"thinking": []}, "thinking must"),
            ("/v1/messages", {"top_k": -1}, "top_k must be"),
            ("/v1/responses", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "input": [{"type": "reasoning", "summary": []}]},
             "unsupported Responses input item"),
            ("/v1/chat/completions", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "messages": [{"role": "tool", "tool_call_id": "orphan",
                              "content": "result"}]},
             "orphan tool result"),
            ("/v1/chat/completions", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "n": 2}, "n must be 1"),
            ("/v1/chat/completions", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "response_format": {"type": "xml"}},
             "structured output type must"),
            ("/v1/chat/completions", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "logit_bias": {"42": 100}}, "logit_bias is not supported"),
            ("/v1/responses", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "previous_response_id": "resp_old"}, "stateless server"),
            ("/v1/responses", {
                "model": "not-a-real-owner/not-a-real-model-validation",
                "text": {"format": {"type": "json_schema"}}},
             "json_schema.schema must"),
            ("/v1/completions", {"prompt": ["not", "supported"]},
             "prompt must"),
        ):
            status, body = post(path, json.dumps(payload).encode())
            assert status == 400 and expected in body["error"]

        status, body = post("/v1/not-a-real-route", b"{}")
        assert status == 404 and body["error"] == "not found"

        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        connection.putrequest("POST", "/v1/responses")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(65 * 1024 * 1024))
        connection.endheaders()
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 413
        assert "VMODEL_MAX_REQUEST_BODY_MB" in payload["error"]
    finally:
        _stop_server(proc)


def test_incomplete_upload_does_not_hold_inference_lock():
    import http.client
    import json
    import socket

    proc = _start_server()
    stalled = socket.create_connection(("127.0.0.1", PORT), timeout=2)
    try:
        stalled.sendall(
            b"POST /v1/responses HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 1024\r\n\r\n")
        # Give the first handler time to block waiting for its declared body.
        time.sleep(0.1)

        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
        started = time.monotonic()
        connection.request(
            "POST", "/v1/responses", body=json.dumps({"tools": {}}),
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        elapsed = time.monotonic() - started
        connection.close()

        assert response.status == 400
        assert payload["error"] == "tools must be an array of objects"
        assert elapsed < 1.0
    finally:
        stalled.close()
        _stop_server(proc)


def test_incomplete_upload_has_configurable_read_deadline():
    import http.client
    import json

    proc = _start_server({"VMODEL_REQUEST_READ_TIMEOUT_SECONDS": "0.05"})
    try:
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
        connection.putrequest("POST", "/v1/responses")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "2")
        connection.endheaders()
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 408
        assert "VMODEL_REQUEST_READ_TIMEOUT_SECONDS" in payload["error"]
    finally:
        _stop_server(proc)


def test_stalled_remote_image_fetch_does_not_hold_inference_lock():
    import http.client
    import io
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(buffer, format="PNG")
    png = buffer.getvalue()
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    class SlowImageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            fetch_started.set()
            release_fetch.wait(timeout=5)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, *_args):
            pass

    image_server = ThreadingHTTPServer(("127.0.0.1", 0), SlowImageHandler)
    image_server.daemon_threads = True
    image_thread = threading.Thread(target=image_server.serve_forever, daemon=True)
    image_thread.start()
    image_port = image_server.server_address[1]

    proc = _start_server()
    first_result = {}

    def image_request():
        try:
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
            payload = json.dumps({
                "model": "SmolLM2-135M", "max_output_tokens": 2,
                "input": [{"role": "user", "content": [{
                    "type": "input_image",
                    "image_url": f"http://127.0.0.1:{image_port}/slow.png",
                }]}],
            })
            connection.request(
                "POST", "/v1/responses", body=payload,
                headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            first_result["status"] = response.status
            response.read()
            connection.close()
        except Exception as error:
            first_result["error"] = error

    request_thread = threading.Thread(target=image_request)
    request_thread.start()
    try:
        assert fetch_started.wait(timeout=5), "runtime never began remote image fetch"

        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1.5)
        started = time.monotonic()
        connection.request(
            "POST", "/v1/responses", body=json.dumps({"tools": {}}),
            headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        elapsed = time.monotonic() - started
        connection.close()

        assert response.status == 400
        assert body["error"] == "tools must be an array of objects"
        assert elapsed < 1.0
    finally:
        release_fetch.set()
        request_thread.join(timeout=10)
        _stop_server(proc)
        image_server.shutdown()
        image_server.server_close()
        image_thread.join(timeout=5)

    assert "error" not in first_result
    assert first_result.get("status") == 400  # text model correctly rejects image input


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
