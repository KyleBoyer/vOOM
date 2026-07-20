"""Integration test using the REAL `openai` Python client library against a
live runtime.server instance — this catches schema mismatches that our own
hand-rolled JSON assertions can't: the openai client's Pydantic response
models will raise or silently drop fields if our server's JSON doesn't
match the expected shape exactly, which is a much stronger check than
"the JSON we wrote parses as JSON."

Starts a real server subprocess on a dedicated test port, uses SmolLM2-135M
(local, fast, no download), tears the server down afterward regardless of
outcome.

  .venv/bin/python tests/test_openai_client_integration.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = 8099  # dedicated test port, distinct from the default 8077
BASE_URL = f"http://127.0.0.1:{PORT}/v1"


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


def _stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "runtime.server", "--port", str(PORT)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # 2026-07-20: every caller does `proc = _start_server(); try: ... finally:
    # _stop_server(proc)`, but that try/finally only wraps code AFTER this
    # function returns. _wait_for_server raising (a slow model load past its
    # 30s readiness timeout, or the process exiting early) propagated out of
    # HERE, before proc ever reached the caller's try/finally -- orphaning an
    # already-Popen'd live server with nothing left holding a reference to
    # kill it. Same bug as tests/test_protocol_features.py's identical
    # helper, fixed there first; a real orphaned --port 8099 process from
    # this file is the likely cause of a later full-suite run hanging for
    # 8+ minutes with zero CPU activity, consistent with a fresh test's
    # server subprocess failing to bind an already-occupied port and
    # something downstream blocking on that instead of failing fast. Clean
    # up here on any failure, then re-raise so the test still reports the
    # original error.
    try:
        _wait_for_server(proc)
    except Exception:
        _stop_server(proc)
        raise
    return proc


def test_chat_completions_non_streaming():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed")
        resp = client.chat.completions.create(
            model="SmolLM2-135M", max_tokens=8,
            messages=[{"role": "user", "content": "The capital of France is"}],
        )
        # if the server's JSON didn't match the expected schema, constructing
        # `resp` itself (Pydantic validation inside the SDK) would have raised
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.role == "assistant"
        assert isinstance(resp.choices[0].message.content, str)
        assert resp.usage.completion_tokens == 8
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens
        assert resp.choices[0].finish_reason in ("stop", "length")

        modern = client.chat.completions.create(
            model="SmolLM2-135M", max_completion_tokens=3,
            messages=[{"role": "user", "content": "Count upward:"}],
        )
        assert modern.usage.completion_tokens == 3
    finally:
        _stop_server(proc)


def test_chat_completions_streaming():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed")
        stream = client.chat.completions.create(
            model="SmolLM2-135M", max_tokens=8, stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "user", "content": "The capital of France is"}],
        )
        chunks = list(stream)  # SDK parses each SSE chunk into a validated Pydantic model
        assert len(chunks) > 0
        assert all(c.object == "chat.completion.chunk" for c in chunks)
        assert next(c for c in chunks if c.choices).choices[0].delta.role == "assistant"
        reconstructed = "".join(
            c.choices[0].delta.content or "" for c in chunks if c.choices)
        assert len(reconstructed) > 0
        usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].choices == []
        assert usage_chunks[0].usage.completion_tokens == 8
        assert (usage_chunks[0].usage.total_tokens
                == usage_chunks[0].usage.prompt_tokens + 8)
    finally:
        _stop_server(proc)


def test_stop_sequence_via_real_client():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed")
        baseline = client.chat.completions.create(
            model="SmolLM2-135M", max_tokens=16,
            messages=[{"role": "user", "content": "The capital of France is"}],
        )
        full_text = baseline.choices[0].message.content
        assert len(full_text) > 4
        stop_str = full_text[2:5]

        stopped = client.chat.completions.create(
            model="SmolLM2-135M", max_tokens=16, stop=stop_str,
            messages=[{"role": "user", "content": "The capital of France is"}],
        )
        assert stop_str not in stopped.choices[0].message.content
    finally:
        _stop_server(proc)


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
