"""Regression test for the quarantined auto-pack-on-repeat-request path.

The consuming daemon pipeline is disabled by default after the late durability
audit. These tests verify the safe default and exercise the legacy behavior only
against a throwaway model with the explicit in-process unsafe flag.

A model that came in through the async auto-download path (DOWNLOADS) and is
requested a SECOND time can get packed
into the zstd-compressed, heat-ordered vpack2 format (F06/F20) in the
background — no client action, no blocking, informational status fields
only (PackManager, runtime/server.py).

Uses a throwaway copy of the local SmolLM2-135M fixture (never mutates the
real one — pack_model with delete_shards=True DESTROYS the source shards)
registered under a fake model id, with DOWNLOADS pre-seeded to "ready" to
simulate a completed auto-download without any real network dependency.

  .venv/bin/python tests/test_autopack.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

import runtime.server as server

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "_test_autopack_model"
TARGET = ROOT / "models" / MODEL_ID


def _seed_fresh_copy():
    shutil.rmtree(TARGET, ignore_errors=True)
    shutil.copytree(ROOT / "models" / "SmolLM2-135M", TARGET)
    for leftover in ("weights.vpack", "weights.vpack2", "weights.vpack2.index.json"):
        p = TARGET / leftover
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    server.DOWNLOADS._status.pop(MODEL_ID, None)
    server.PACKS._status.pop(MODEL_ID, None)
    server.PACKS._request_counts.pop(MODEL_ID, None)
    server.DOWNLOADS._status[MODEL_ID] = {"state": "ready", "error": None, "started_at": time.time()}


def _wait_for_pack_terminal(timeout=30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = server.PACKS.status(MODEL_ID)
        if st is not None and st["state"] in ("packed", "failed"):
            return st
        time.sleep(0.2)
    raise TimeoutError("auto-pack never reached a terminal state")


def test_first_request_does_not_trigger_pack():
    old = server._ENABLE_UNSAFE_AUTOPACK
    server._ENABLE_UNSAFE_AUTOPACK = False
    _seed_fresh_copy()
    try:
        p = server._resolve(MODEL_ID)
        assert p == TARGET
        assert server.PACKS.status(MODEL_ID) is None
    finally:
        shutil.rmtree(TARGET, ignore_errors=True)
        server.DOWNLOADS._status.pop(MODEL_ID, None)
        server.PACKS._status.pop(MODEL_ID, None)
        server.PACKS._request_counts.pop(MODEL_ID, None)
        server._ENABLE_UNSAFE_AUTOPACK = old


def test_default_disabled_even_after_second_request():
    old = server._ENABLE_UNSAFE_AUTOPACK
    server._ENABLE_UNSAFE_AUTOPACK = False
    _seed_fresh_copy()
    try:
        server._resolve(MODEL_ID)
        server._resolve(MODEL_ID)
        assert server.PACKS.status(MODEL_ID) is None
        assert (TARGET / "model.safetensors").exists()
    finally:
        shutil.rmtree(TARGET, ignore_errors=True)
        server.DOWNLOADS._status.pop(MODEL_ID, None)
        server.PACKS._status.pop(MODEL_ID, None)
        server.PACKS._request_counts.pop(MODEL_ID, None)
        server._ENABLE_UNSAFE_AUTOPACK = old


def test_second_request_triggers_pack_and_completes_correctly():
    old = server._ENABLE_UNSAFE_AUTOPACK
    server._ENABLE_UNSAFE_AUTOPACK = True
    _seed_fresh_copy()
    try:
        server._resolve(MODEL_ID)  # 1st request: no pack
        server._resolve(MODEL_ID)  # 2nd request: triggers auto-pack
        st = server.PACKS.status(MODEL_ID)
        assert st is not None and st["state"] == "packing", st

        final = _wait_for_pack_terminal()
        assert final["state"] == "packed", final
        assert (TARGET / "weights.vpack2.index.json").exists()
        assert not (TARGET / "model.safetensors").exists()  # delete_shards reclaimed it
    finally:
        shutil.rmtree(TARGET, ignore_errors=True)
        server.DOWNLOADS._status.pop(MODEL_ID, None)
        server.PACKS._status.pop(MODEL_ID, None)
        server.PACKS._request_counts.pop(MODEL_ID, None)
        server._ENABLE_UNSAFE_AUTOPACK = old


def test_engine_auto_invalidates_and_serves_identical_tokens_after_pack():
    old = server._ENABLE_UNSAFE_AUTOPACK
    server._ENABLE_UNSAFE_AUTOPACK = True
    _seed_fresh_copy()
    try:
        prompt = "The capital of France is"
        p1 = server._resolve(MODEL_ID)
        engine = server.MANAGER.get(p1, "lossless")
        assert engine.store.vpack2 is None  # still raw
        before = engine.generate(prompt, max_tokens=16)["tokens"]

        server._resolve(MODEL_ID)  # triggers the auto-pack
        _wait_for_pack_terminal()

        engine2 = server.MANAGER.get(p1, "lossless")
        assert engine2.store.vpack2 is not None  # picked up the pack automatically
        after = engine2.generate(prompt, max_tokens=16)["tokens"]
        assert before == after  # lossless gate: packing must never move a token
        engine2.close()
        mx.clear_cache()
    finally:
        shutil.rmtree(TARGET, ignore_errors=True)
        server.DOWNLOADS._status.pop(MODEL_ID, None)
        server.PACKS._status.pop(MODEL_ID, None)
        server.PACKS._request_counts.pop(MODEL_ID, None)
        server._ENABLE_UNSAFE_AUTOPACK = old


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
