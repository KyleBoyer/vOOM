"""Regression test for runtime/server.py's _resolve() + DownloadManager.

Originally covered only the disk-safety guard (added 2026-07-13): an
unrecognized model id (e.g. a client typo) must not silently trigger an HF
download attempt when free disk space is low on this project's perpetually
near-full external drive.

Updated 2026-07-13 (same day, later): _resolve() was rewritten to be async
(runtime/server.py's DownloadManager) after a live-confirmed bug where the
old synchronous snapshot_download() call inside the locked request handler
could hang a client connection indefinitely. The disk-safety check now runs
INSIDE the background download thread, not synchronously in _resolve()
itself, so _resolve() always either returns a local Path immediately or
raises ModelDownloading/ModelDownloadFailed right away — it never blocks.
These tests were updated to match; see tests/test_async_download.py for the
live, real-HF-network end-to-end coverage of the async behavior itself.

  .venv/bin/python tests/test_server_resolve.py
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runtime.server as server


def _wait_for_terminal_status(model_id: str, timeout=10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = server.DOWNLOADS.status(model_id)
        if st is not None and st["state"] != "downloading":
            return st
        time.sleep(0.05)
    raise TimeoutError(f"{model_id} never left the 'downloading' state")


def test_refuses_download_on_low_disk():
    """Low free space must fail the BACKGROUND download cleanly, not
    silently proceed. The check moved from synchronous-in-_resolve() to
    inside DownloadManager's thread with the async rewrite, so _resolve()
    itself now raises ModelDownloading immediately (the failure surfaces a
    moment later via DOWNLOADS.status())."""
    model_id = "totally-unknown-model-low-disk-test"
    with patch("shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=2_000_000_000)  # 2GB, below the 5GB floor
        try:
            server._resolve(model_id)
            assert False, "expected ModelDownloading (the download itself runs in the background)"
        except server.ModelDownloading:
            pass
    st = _wait_for_terminal_status(model_id)
    assert st["state"] == "failed", st
    assert "free" in st["error"] and "GB free" in st["error"], st["error"]


def test_known_local_model_resolves_without_disk_check():
    """A registry hit must never even reach the disk-space check or the
    download manager — no download should be attempted for a model that's
    already local, so a (deliberately absurd) mocked-out-of-space
    disk_usage must not matter."""
    with patch("shutil.disk_usage") as mock_du:
        mock_du.return_value = MagicMock(free=0)
        result = server._resolve("SmolLM2-135M")
        assert result.name == "SmolLM2-135M"
        mock_du.assert_not_called()


def test_healthy_disk_does_not_block_unknown_model_path():
    """With plenty of free space, an unknown model id's BACKGROUND download
    must reach snapshot_download (not be refused) — patch snapshot_download
    itself to avoid a real network call. _resolve() itself must return
    (raise ModelDownloading) near-instantly, never blocking on the mocked
    download."""
    model_id = "some-org/totally-unknown-model-healthy-disk-test"
    with patch("shutil.disk_usage") as mock_du, \
         patch("huggingface_hub.snapshot_download") as mock_dl:
        mock_du.return_value = MagicMock(free=50_000_000_000)  # 50GB, healthy
        mock_dl.return_value = None
        t0 = time.time()
        try:
            server._resolve(model_id)
            assert False, "expected ModelDownloading -- must not block on the download"
        except server.ModelDownloading:
            pass
        assert time.time() - t0 < 1, "_resolve() blocked instead of returning immediately"

        deadline = time.time() + 10
        while not mock_dl.called and time.time() < deadline:
            time.sleep(0.05)
        mock_dl.assert_called_once()


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
