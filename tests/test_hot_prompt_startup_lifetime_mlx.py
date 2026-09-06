"""Local Metal/engine lifetime regression, not a production-model speed gate."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
import weakref

import mlx.core as mx
import numpy as np
import pytest

from runtime.kv_cache import KVCache
from tests.test_hot_prompt_startup_lifetime_pure import make_owner, startup


@pytest.mark.parametrize("retained", ["none", "alias", "fork"])
def test_real_plain_kv_active_bytes_follow_surviving_owners(startup, retained):
    old = KVCache(1)
    old.keys[0] = mx.ones((1, 1, 1024, 1024), dtype=mx.float32)
    old.values[0] = mx.full((1, 1, 1024, 1024), 2, dtype=mx.float32)
    mx.eval(old.keys, old.values)
    slots = [] if retained == "none" else [SimpleNamespace(
        kv=old if retained == "alias" else old.fork(), tokens=(1, 2))]
    owner = make_owner(startup, old, slots)
    logical = old.nbytes()
    ref = weakref.ref(old)
    del old
    # Quiesce the last consumer before measuring ownership, not after release.
    mx.synchronize()
    before = mx.get_active_memory()

    def before_scan():
        after = mx.get_active_memory()
        assert logical == 8_388_608
        assert before - after == (logical if retained == "none" else 0)
        assert (ref() is not None) is (retained == "alias")
        print("startup_lifetime_witness " + json.dumps({
            "retained": retained, "logical_bytes": logical,
            "active_before": before, "active_after": after,
            "active_drop": before - after, "orphan_alive": ref() is not None,
            "scope": "synthetic_MLX_ownership_not_system_pressure",
        }, sort_keys=True))
        if slots:
            assert owner._hot_prompt_slots[0] is slots[0]
            assert np.all(np.asarray(slots[0].kv.keys[0]) == 1)
            assert np.all(np.asarray(slots[0].kv.values[0]) == 2)

    startup[0](owner, before_scan)


@pytest.mark.parametrize("request_kind", ["repeat", "extension", "branch"])
def test_real_engine_drops_orphan_before_next_sweep_and_keeps_reuse(
        monkeypatch, request_kind):
    from runtime.engine import StreamingEngine
    from runtime.server import PreparedPrompt
    from tests.test_hot_prompt_kv import (
        FIRST, SECOND, FIXTURE, _config, _ensure_fixture,
    )

    _ensure_fixture()
    engine = StreamingEngine(str(FIXTURE), _config())
    control = StreamingEngine(str(FIXTURE), _config())
    try:
        first = engine.generate(FIRST, 4)
        assert first["tokens"] == control.generate(FIRST, 4)["tokens"]
        slots = engine._hot_prompt_slots
        assert slots
        # Keep no strong copy of the diagnostic orphan. Real Qwen aligned
        # retention creates this same ownership shape without test injection.
        engine.last_kv = KVCache(1)
        engine.last_kv.keys[0] = mx.ones((1, 1, 1024, 1024), dtype=mx.float32)
        mx.eval(engine.last_kv.keys)
        orphan = weakref.ref(engine.last_kv)
        observed = []
        control_observed = []
        sweep = engine._sweep
        control_sweep = control._sweep

        def slot_snapshot(target):
            return [(slot.tokens, slot.reusable_prefix, slot.prompt_length,
                     slot.chunk_size, _cache_digest(slot.kv))
                    for slot in target._hot_prompt_slots]

        def checked_sweep(*args, **kwargs):
            assert orphan() is None, "old request survives into next sweep"
            if not observed:
                observed.append(slot_snapshot(engine))
            return sweep(*args, **kwargs)

        def checked_control_sweep(*args, **kwargs):
            if not control_observed:
                control_observed.append(slot_snapshot(control))
            return control_sweep(*args, **kwargs)

        # This fixture disables the governor, so it has no admission callback.
        # The production sweep is the first actual state consumer on every arm.
        monkeypatch.setattr(engine, "_sweep", checked_sweep)
        monkeypatch.setattr(control, "_sweep", checked_control_sweep)
        if request_kind == "repeat":
            prompt = FIRST
        elif request_kind == "branch":
            prompt = SECOND
        else:
            tokens = list(slots[-1].tokens) + engine.tokenizer.encode(
                "\nTool result accepted. Continue the next turn.").ids
            prompt = PreparedPrompt("extension", tokens)
        result = engine.generate(prompt, 3)
        expected = control.generate(prompt, 3)
        assert observed
        assert observed == control_observed
        assert result["tokens"] == expected["tokens"]
        assert result["text"] == expected["text"]
        for key in ("prompt_cache_source", "prompt_cache_prefix_tokens",
                    "prompt_cache_exact_hit", "hot_prompt_lcp_tokens"):
            assert result["path_stats"].get(key) == expected["path_stats"].get(key)
        assert result["path_stats"]["prompt_cache_source"] == "memory"
        assert _cache_digest(engine.last_kv) == _cache_digest(control.last_kv)
        print("startup_engine_witness " + json.dumps({
            "request_kind": request_kind, "tokens": result["tokens"],
            "state_sha256": _cache_digest(engine.last_kv),
            "prefix_tokens": result["path_stats"]["prompt_cache_prefix_tokens"],
            "prefill_seconds": result.get("prefill_s"),
            "decode_seconds": result.get("decode_s"),
            "scope": "synthetic_tiny_GLM_not_production_latency_or_quality",
            "orphan_dead_before_next_sweep": True,
        }, sort_keys=True))
    finally:
        engine.close()
        control.close()


def _cache_digest(kv):
    """Exact tiny-GLM endpoint plus DSA contents, including dtype and shape."""
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, mx.array):
            mx.eval(value)
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(np.asarray(value.view(mx.uint8)).tobytes())
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                update(key)
                update(item)
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        elif isinstance(value, set):
            update(sorted(value))
        else:
            digest.update(json.dumps(value).encode())

    update((kv.offset, kv.compressed_mla, kv._starts, kv._windows,
            kv.keys, kv.values))
    dsa = getattr(kv, "dsa", None)
    if dsa is not None:
        update((dsa.k_idx, dsa.selection, dsa.sel_layer,
                dsa.selection_ranges, dsa.dense_ranges))
    return digest.hexdigest()
