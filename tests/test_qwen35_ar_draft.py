from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.qwen35_ar_draft import (
    ResidentQwenARDrafter,
    validate_qwen_ar_draft_compatibility,
)


class _CacheEntry:
    def __init__(self):
        self.cache = []

    @property
    def state(self):
        return (mx.array(self.cache, dtype=mx.int32),)

    @property
    def nbytes(self):
        return len(self.cache) * 4


class _Model:
    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, inputs, *, cache):
        tokens = [int(value) for value in inputs.reshape(-1).tolist()]
        self.calls.append(tuple(tokens))
        cache[0].cache.extend(tokens)
        logits = mx.full((1, len(tokens), self.vocab_size), -100.0)
        for position, token in enumerate(tokens):
            logits = logits.at[0, position, (token + 1) % self.vocab_size].add(
                200.0)
        return logits


class _Backend:
    def __init__(self):
        self.model = _Model()
        self.effective_max_position_embeddings = 128
        self.closed = False

    def _make_prompt_cache(self):
        return [_CacheEntry()]

    def close(self):
        self.closed = True
        self.model = None


def test_resident_ar_draft_replays_only_target_committed_inputs():
    backend = _Backend()
    drafter = ResidentQwenARDrafter(
        backend, identity="test-ar", prefill_step_size=2)
    hidden = mx.zeros((1, 1, 1))

    drafter.begin_request([1, 2, 3])
    assert backend.model.calls == [(1, 2), (3,)]
    assert drafter._committed_cache[0].cache == [1, 2, 3]

    drafter.begin_round([1, 2, 3, 4])
    for token in (4, 5, 6, 7):
        logits, hidden = drafter.draft_step(hidden, token, None, 0)
        assert int(mx.argmax(logits)) == token + 1
    # The proposal fork reached token 7, but a reject after one accepted draft
    # commits only catchup=4 and accepted input=5.  Tokens 6/7 must disappear.
    drafter.commit_target_inputs([4, 5])
    assert drafter._committed_cache[0].cache == [1, 2, 3, 4, 5]

    # Two plain target outputs occurred before the next speculative round.
    # begin_round catches up every now-fed target input, leaving 10 pending.
    drafter.begin_round([1, 2, 3, 4, 5, 8, 9, 10])
    assert drafter._committed_cache[0].cache == [1, 2, 3, 4, 5, 8, 9]
    logits, _ = drafter.draft_step(hidden, 10, None, 0)
    assert int(mx.argmax(logits)) == 11
    drafter.commit_target_inputs([10])
    assert drafter._committed_cache[0].cache == [
        1, 2, 3, 4, 5, 8, 9, 10]

    stats = drafter.telemetry_snapshot()
    assert stats["request_prompt_tokens"] == 3
    assert stats["proposal_steps"] == 5
    assert stats["commit_replay_steps"] == 3
    assert stats["round_sync_steps"] == 2
    assert stats["committed_tokens"] == 8
    drafter.end_request()
    assert drafter._committed_cache is None


def test_resident_ar_draft_rejects_nonprefix_commit_and_history_divergence():
    drafter = ResidentQwenARDrafter(
        _Backend(), identity="test-ar", prefill_step_size=4)
    hidden = mx.zeros((1, 1, 1))
    drafter.begin_request([1, 2, 3])
    drafter.begin_round([1, 2, 3, 4])
    drafter.draft_step(hidden, 4, None, 0)
    with pytest.raises(RuntimeError, match="non-prefix"):
        drafter.commit_target_inputs([9])

    drafter.begin_request([1, 2, 3])
    with pytest.raises(RuntimeError, match="diverged"):
        drafter.begin_round([1, 7, 3, 4])


def _tokenizer(vocab=None, *, added_tokens=()):
    return {
        "model": {
            "type": "BPE",
            "vocab": vocab or {"a": 0, "b": 1},
            "merges": ["a b"],
            "unk_token": None,
            "byte_fallback": False,
            "fuse_unk": False,
            "ignore_merges": False,
        },
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": list(added_tokens),
    }


def _cfg(*, hidden_size=8):
    return SimpleNamespace(
        model_type="qwen3_5",
        num_experts=0,
        vocab_size=2,
        max_position_embeddings=128,
        hidden_size=hidden_size,
    )


def test_qwen_ar_draft_compatibility_uses_id_semantics_not_registration(tmp_path):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "tokenizer.json").write_text(json.dumps(
        _tokenizer(added_tokens=[{"id": 1, "content": "<tool>"}])))
    (draft / "tokenizer.json").write_text(json.dumps(
        _tokenizer(added_tokens=[])))

    fingerprint = validate_qwen_ar_draft_compatibility(
        target, _cfg(hidden_size=16), draft, _cfg(hidden_size=8))
    assert len(fingerprint) == 64

    structured = _tokenizer(added_tokens=[])
    structured["model"]["merges"] = [["a", "b"]]
    (draft / "tokenizer.json").write_text(json.dumps(structured))
    assert validate_qwen_ar_draft_compatibility(
        target, _cfg(hidden_size=16), draft, _cfg(hidden_size=8),
    ) == fingerprint

    (draft / "tokenizer.json").write_text(json.dumps(
        _tokenizer(vocab={"a": 1, "b": 0})))
    with pytest.raises(ValueError, match="tokenizer ID semantics"):
        validate_qwen_ar_draft_compatibility(
            target, _cfg(hidden_size=16), draft, _cfg(hidden_size=8))


def test_resident_ar_draft_close_releases_auxiliary_backend():
    backend = _Backend()
    drafter = ResidentQwenARDrafter(backend, identity="test-ar")
    drafter.begin_request([1])
    drafter.close()
    assert backend.closed
    assert drafter.backend is None
    assert drafter.model is None


def test_resident_ar_factory_explicitly_allows_lossy_draft_profile(
    monkeypatch, tmp_path,
):
    captured = {}
    draft_dir = tmp_path / "draft"
    target_dir = tmp_path / "target"
    draft_dir.mkdir()
    target_dir.mkdir()
    draft_cfg = _cfg(hidden_size=8)
    target_cfg = _cfg(hidden_size=16)
    (draft_dir / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    (target_dir / "tokenizer.json").write_text(json.dumps(_tokenizer()))

    monkeypatch.setattr(
        "runtime.qwen35_ar_draft.ModelConfig.from_dir", lambda _path: draft_cfg)
    monkeypatch.setattr(
        "runtime.qwen35_ar_draft.choose_resident_backend",
        lambda *args, **kwargs: (
            captured.update(kwargs) or SimpleNamespace(
                admitted=False,
                reason="fixture-stop",
                payload_bytes=1,
                estimated_metal_bytes=2,
                available_bytes=3,
            )
        ),
    )

    with pytest.raises(MemoryError, match="fixture-stop"):
        ResidentQwenARDrafter.from_model_dir(
            draft_dir, target_dir=target_dir, target_cfg=target_cfg)
    assert captured["allow_lossy_draft"] is True


def test_lazy_factory_defers_backend_load_and_unloads_after_request(
    monkeypatch, tmp_path,
):
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    (draft / "tokenizer.json").write_text(json.dumps(_tokenizer()))
    (draft / "config.json").write_text(json.dumps({
        "quantization_config": {
            "mode": "affine", "bits": 3, "group_size": 64,
        },
    }))
    cfg = _cfg()
    decision = SimpleNamespace(
        admitted=True, payload_bytes=123, estimated_metal_bytes=456,
        available_bytes=789,
    )
    made = []

    class FakeResident(_Backend):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(
        "runtime.qwen35_ar_draft.ModelConfig.from_dir", lambda _path: cfg)
    monkeypatch.setattr(
        "runtime.qwen35_ar_draft.choose_resident_backend",
        lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(
        "runtime.qwen35_ar_draft.ResidentMLXLMEngine", FakeResident)

    drafter = ResidentQwenARDrafter.from_model_dir(
        draft, target_dir=target, target_cfg=cfg, prefill_step_size=2)
    assert drafter.staged_load
    assert drafter.backend is None
    assert made == []

    drafter.begin_request([1, 2, 3])
    assert len(made) == 1
    assert drafter.backend is made[0]
    drafter.end_request()
    assert made[0].closed
    assert drafter.backend is None
    assert drafter.telemetry_snapshot()["backend_unloads"] == 1


def test_verifier_suspension_releases_weights_but_preserves_exact_round_base():
    first = _Backend()
    restored = []

    def load_backend():
        backend = _Backend()
        restored.append(backend)
        return backend

    drafter = ResidentQwenARDrafter(
        first,
        identity="test-ar",
        backend_loader=load_backend,
        unload_between_requests=True,
    )
    hidden = mx.zeros((1, 1, 1))
    drafter.begin_request([1, 2, 3])
    drafter.begin_round([1, 2, 3, 4])
    drafter.draft_step(hidden, 4, None, 0)
    drafter.draft_step(hidden, 5, None, 1)
    assert drafter._round_base[0].cache == [1, 2, 3]
    assert drafter._working_cache[0].cache == [1, 2, 3, 4, 5]

    suspended = drafter.suspend_for_target_verification()

    assert suspended["suspended"] == 1
    assert first.closed
    assert drafter.backend is None
    assert drafter.model is None
    assert drafter._working_cache is None
    assert drafter._round_base[0].cache == [1, 2, 3]
    assert drafter._working_inputs == [4, 5]

    drafter.ensure_loaded()
    assert drafter.backend is restored[0]
    drafter.commit_target_inputs([4])
    assert drafter._committed_cache[0].cache == [1, 2, 3, 4]
    stats = drafter.telemetry_snapshot()
    assert stats["verification_suspends"] == 1
    assert stats["backend_loads"] == 1
    assert stats["backend_unloads"] == 1
    assert stats["verification_suspend_s"] >= 0.0
    assert stats["verification_released_active_bytes"] >= 0

    drafter.end_request()
    assert restored[0].closed
    assert drafter.telemetry_snapshot()["backend_unloads"] == 2


def test_bounded_draft_can_retain_weights_through_target_verification():
    backend = _Backend()
    drafter = ResidentQwenARDrafter(
        backend,
        identity="test-retained-ar",
        retain_for_target_verification=True,
        resident_payload_bytes=123_000_000,
    )
    hidden = mx.zeros((1, 1, 1))
    drafter.begin_request([1, 2, 3])
    drafter.begin_round([1, 2, 3, 4])
    drafter.draft_step(hidden, 4, None, 0)

    retained = drafter.suspend_for_target_verification()

    assert retained["suspended"] == 0
    assert retained["retained"] == 1
    assert not backend.closed
    assert drafter.backend is backend
    assert drafter.weights_loaded
    assert drafter._working_cache is None
    assert drafter._round_base[0].cache == [1, 2, 3]
    drafter.commit_target_inputs([4])
    stats = drafter.telemetry_snapshot()
    assert stats["verification_retain_enabled"] == 1
    assert stats["verification_retained_rounds"] == 1
    assert stats["verification_retained_active_bytes_peak"] >= 0
    assert stats["backend_unloads"] == 0
    drafter.close()


def test_retained_draft_rejects_payload_above_hard_ceiling():
    with pytest.raises(ValueError, match="600MB"):
        ResidentQwenARDrafter(
            _Backend(),
            identity="oversize-retained-ar",
            retain_for_target_verification=True,
            resident_payload_bytes=600_000_001,
        )


def test_staged_load_drains_prefetch_and_trims_target_before_loading():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    events = []

    class FakePrefetcher:
        paused = False

        def pause_and_wait_idle(self):
            events.append("prefetch-idle")
            self.paused = True

    class FakeCache:
        pinned_bytes = 17

        def trim_to(self, target):
            events.append(("trim", target))
            return 123

    class FakeDrafter:
        staged_load = True

        def ensure_loaded(self):
            events.append("load")

    engine = object.__new__(QwenMTPSpeculativeEngine)
    engine.target = SimpleNamespace(
        prefetcher=FakePrefetcher(), cache=FakeCache())
    engine.drafter = FakeDrafter()

    stats = engine._stage_drafter_after_target_prefill()

    assert events == ["prefetch-idle", ("trim", 17), "load"]
    assert not engine.target.prefetcher.paused
    assert stats["qwen_mtp_staged_draft_load"] == 1
    assert stats["qwen_mtp_staged_target_cache_released_bytes"] == 123
    assert stats["qwen_mtp_staged_draft_load_s"] >= 0.0


def test_staged_load_is_noop_when_retained_draft_is_already_loaded():
    from runtime.qwen35_mtp import QwenMTPSpeculativeEngine

    class FakeDrafter:
        staged_load = True
        weights_loaded = True

        def ensure_loaded(self):
            raise AssertionError("retained weights must not reload")

    engine = object.__new__(QwenMTPSpeculativeEngine)
    engine.target = SimpleNamespace(
        prefetcher=None,
        cache=SimpleNamespace(
            pinned_bytes=0,
            trim_to=lambda _target: (_ for _ in ()).throw(
                AssertionError("retained draft must not trim target cache")),
        ),
    )
    engine.drafter = FakeDrafter()

    assert engine._stage_drafter_after_target_prefill() == {
        "qwen_mtp_staged_draft_load": 0,
        "qwen_mtp_staged_target_cache_released_bytes": 0,
        "qwen_mtp_staged_draft_load_s": 0.0,
    }
