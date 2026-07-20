"""Regression test for the `lossy-<model>` model-ID prefix convention
(2026-07-13, user request): a natural, protocol-agnostic way to pick
GOAL/Sub-Goal (lossless, default) vs Side-Quest (fast) mode, since the
`model` field is the one thing every supported protocol (OpenAI chat/
completions and Responses, Anthropic Messages) already has as a plain
string — no non-standard header required. Every base model is advertised
BOTH ways on GET /v1/models so a client can discover the convention
without needing prior knowledge of it.

  .venv/bin/python tests/test_model_mode_prefix.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = 8096
MODEL = "SmolLM2-135M"


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
    # 2026-07-20: _wait_for_server raising (a slow model load past its 30s
    # readiness timeout, or the process exiting early) propagated out of
    # HERE, before proc ever reached the caller's `try: ... finally:
    # _stop_server(proc)` -- orphaning an already-Popen'd live server with
    # nothing left holding a reference to kill it. Same bug fixed in
    # tests/test_protocol_features.py and
    # tests/test_openai_client_integration.py's identical helpers; a real
    # orphaned server process from one of these files is the likely cause
    # of a full-suite run hanging for 8+ minutes with zero CPU activity.
    try:
        _wait_for_server(proc)
    except Exception:
        _stop_server(proc)
        raise
    return proc


def _stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def test_split_model_mode_unit():
    from runtime.server import split_model_mode

    assert split_model_mode(f"lossy-{MODEL}") == (MODEL, "fast")
    assert split_model_mode("lossy-long-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast-long")
    assert split_model_mode(MODEL) == (MODEL, None)
    assert split_model_mode("lossy-Qwen/Qwen2.5-72B") == ("Qwen/Qwen2.5-72B", "fast")
    # case-insensitive prefix match
    assert split_model_mode(f"Lossy-{MODEL}") == (MODEL, "fast")
    assert split_model_mode("Lossy-Long-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast-long")


def test_models_endpoint_advertises_both_variants():
    import json
    import urllib.request

    proc = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
        ids = {m["id"] for m in data["data"]}
        assert MODEL in ids
        assert f"lossy-{MODEL}" in ids
    finally:
        _stop_server(proc)


def test_lossy_prefix_engages_fast_mode_via_real_openai_client():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.chat.completions.create(
            model=f"lossy-{MODEL}", max_tokens=8,
            messages=[{"role": "user", "content": "The capital of France is"}])
        assert resp.model == MODEL  # prefix stripped in the echoed model field
        raw = resp.model_extra or {}
        usage_extra = resp.usage.model_extra if resp.usage else {}
        assert (usage_extra or {}).get("vmodel_mode") == "fast"
    finally:
        _stop_server(proc)


def test_bare_model_defaults_to_lossless_via_real_openai_client():
    from openai import OpenAI

    proc = _start_server()
    try:
        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=8,
            messages=[{"role": "user", "content": "The capital of France is"}])
        usage_extra = resp.usage.model_extra if resp.usage else {}
        assert (usage_extra or {}).get("vmodel_mode") == "lossless"
    finally:
        _stop_server(proc)


def test_lossy_prefix_works_via_responses_and_messages_apis():
    from anthropic import Anthropic
    from openai import OpenAI

    proc = _start_server()
    try:
        oclient = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x")
        oresp = oclient.responses.create(
            model=f"lossy-{MODEL}", max_output_tokens=8, input="Hi")
        assert oresp.model == MODEL

        aclient = Anthropic(base_url=f"http://127.0.0.1:{PORT}", api_key="x")
        aresp = aclient.messages.create(
            model=f"lossy-{MODEL}", max_tokens=8,
            messages=[{"role": "user", "content": "Hi"}])
        assert aresp.model == MODEL
        raw = aresp.model_extra or {}
        assert raw.get("vmodel_sampling") == "greedy"  # sanity: real generation happened
    finally:
        _stop_server(proc)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__}: PASS")
            passed += 1
        except Exception as e:
            print(f"  {fn.__name__}: FAIL ({type(e).__name__}: {e})")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(fns)} tests passed")
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
