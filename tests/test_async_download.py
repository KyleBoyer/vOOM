"""Regression test for the async model-download fix (2026-07-13, user
question: "Does the http server also properly handle random hugging face
model IDs that we havent downloaded, and error with model downloading/model
packing/status updated until its ready or something?").

Live-tested finding this fix addresses: `_resolve()` used to call HF's
`snapshot_download()` INLINE inside the locked request handler. Two real,
confirmed failure modes:
  1. hf-internal-testing/tiny-random-gpt2 (unsupported architecture: its
     config.json uses GPT-2's `n_head` instead of the Llama-style
     `num_attention_heads` this codebase's config parser expects) surfaced
     as a raw, unhelpful `KeyError` inside a bare 500 response.
  2. yujiepan/qwen2.5-tiny-random (a SUPPORTED architecture, used to isolate
     the first failure from a pure architecture-name mismatch) instead hit
     a genuine network stall — confirmed via `ps -p <pid> -o time,state`
     that the server process's CPU time was static (blocked on I/O, not
     computing) for 90+ seconds with zero client-visible progress and zero
     timeout.

This test exercises the SAME two repos against the fix: a request for an
unresolved model must return promptly (202 "downloading"), never block, and
the eventual failure/success must be clear and polling-visible, not a raw
traceback. Network-dependent (real HF pulls) and slow (bounded by a real
download) by nature — this is deliberately a live test, not a mock, per
this project's "test everything live, not just unit-level" discipline.

  .venv/bin/python tests/test_async_download.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PORT = 8098

UNSUPPORTED_ARCH_MODEL = "hf-internal-testing/tiny-random-gpt2"
STALL_PRONE_MODEL = "yujiepan/qwen2.5-tiny-random"


def _wait_for_server(proc, timeout=30):
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
    for d in ("tiny-random-gpt2", "qwen2.5-tiny-random"):
        import shutil

        p = ROOT / "models" / d
        if p.exists():
            shutil.rmtree(p)
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


def _post(model: str, timeout=10) -> tuple[int, dict]:
    body = json.dumps({"model": model, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_unresolved_model_returns_immediately_not_blocking():
    """The core regression: this request used to block for as long as the
    download took (or hang forever on a stall). It must now return in well
    under a second."""
    proc = _start_server()
    try:
        t0 = time.time()
        status, body = _post(UNSUPPORTED_ARCH_MODEL, timeout=15)
        elapsed = time.time() - t0
        assert elapsed < 5, f"request took {elapsed:.1f}s -- should return almost instantly"
        assert status == 202, body
        assert body.get("vmodel_download_status") == "downloading"
    finally:
        _stop_server(proc)


def test_unsupported_architecture_fails_clearly_not_a_raw_traceback():
    """hf-internal-testing/tiny-random-gpt2's config.json uses `n_head`
    (GPT-2 family), not this codebase's expected `num_attention_heads` —
    must surface as a clear 422 with an actionable message, not a bare
    KeyError inside a 500."""
    proc = _start_server()
    try:
        status, body = _post(UNSUPPORTED_ARCH_MODEL)  # kick off the download
        assert status == 202

        deadline = time.time() + 30
        while time.time() < deadline:
            status, body = _post(UNSUPPORTED_ARCH_MODEL)
            if body.get("vmodel_download_status") != "downloading":
                break
            time.sleep(1)
        else:
            raise TimeoutError("download+validation never finished")

        assert status == 422, body
        assert body.get("vmodel_download_status") == "failed"
        err = body.get("error", "")
        # This is deliberately a live HF probe.  When the checkpoint reaches
        # us, require the exact unsupported-config diagnostic.  Corporate
        # proxies and offline runners can fail before any config is available;
        # in that case the endpoint must still expose a useful download error.
        # The architecture branch itself is pinned without a network dependency
        # by test_unsupported_architecture_validation_offline below.
        if "num_attention_heads" not in err:
            network_markers = (
                "LocalEntryNotFoundError", "ConnectError", "ConnectionError",
                "CERTIFICATE_VERIFY_FAILED", "403", "internet connection",
            )
            assert any(marker in err for marker in network_markers), err
        assert "KeyError" not in err  # not a bare traceback string
    finally:
        _stop_server(proc)


def test_unsupported_architecture_validation_offline():
    """Exercise the post-download validation branch without trusting HF or a
    proxy.  The fake downloader writes a GPT-2-shaped config, then the real
    DownloadManager must report the missing Llama-style key and remove the
    invalid local directory."""
    from runtime.server import DownloadManager

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "tiny-random-gpt2"

        def fake_snapshot_download(model_id, *, local_dir, **kwargs):
            del model_id, kwargs
            model_dir = Path(local_dir)
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "gpt2",
                "n_head": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "vocab_size": 128,
            }))
            return str(model_dir)

        manager = DownloadManager()
        with patch("huggingface_hub.snapshot_download", fake_snapshot_download):
            manager.start(UNSUPPORTED_ARCH_MODEL, target.name, target)
            deadline = time.time() + 5
            while time.time() < deadline:
                status = manager.status(UNSUPPORTED_ARCH_MODEL)
                if status and status["state"] != "downloading":
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("offline validation never finished")

        assert status["state"] == "failed", status
        assert "num_attention_heads" in status["error"]
        assert "KeyError" not in status["error"]
        assert not target.exists(), "invalid model directory was not cleaned up"


def test_models_endpoint_shows_failed_download_status():
    proc = _start_server()
    try:
        _post(UNSUPPORTED_ARCH_MODEL)
        deadline = time.time() + 30
        while time.time() < deadline:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as resp:
                data = json.loads(resp.read())
            entries = [m for m in data["data"] if m["id"] == UNSUPPORTED_ARCH_MODEL]
            if entries and entries[0].get("vmodel_download_status") == "failed":
                assert "vmodel_download_error" in entries[0]
                return
            time.sleep(1)
        raise TimeoutError("GET /v1/models never showed the failed status")
    finally:
        _stop_server(proc)


def test_repeated_polls_during_download_never_block_other_requests():
    """While one model is downloading (or stalled) in the background,
    unrelated requests for an already-local model must still be served
    promptly — the shared INFER_LOCK must not be held for the download's
    duration."""
    proc = _start_server()
    try:
        status, body = _post(STALL_PRONE_MODEL)
        assert status == 202

        t0 = time.time()
        status, body = _post("SmolLM2-135M", timeout=20)
        elapsed = time.time() - t0
        assert status == 200, body
        assert elapsed < 15, f"unrelated request took {elapsed:.1f}s -- background download blocked it"
    finally:
        _stop_server(proc)
        import shutil

        p = ROOT / "models" / "qwen2.5-tiny-random"
        if p.exists():
            shutil.rmtree(p)


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
