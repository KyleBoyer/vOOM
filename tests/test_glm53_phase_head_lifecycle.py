"""Request ownership, admission, and real Metal reclamation for the GLM head."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.engine import StreamingEngine
from runtime.speculative import NativeMTPEngine
from runtime.weight_cache import WeightCache


HEAD_BYTES = 1_268_776_960


class _Cache:
    def __init__(self, head, events):
        self.head = head
        self.events = events
        self.max_bytes = 2_800_000_000
        self.total_bytes = HEAD_BYTES
        self.pinned_bytes = HEAD_BYTES
        self.promote_ok = True
        self.release_ok = True

    def get(self, key, names):
        self.events.append(("get", key))
        return {"lm_head.weight": self.head}

    def release_pinned(self, key, names):
        self.events.append(("release", key))
        if not self.release_ok:
            return 0
        self.total_bytes = self.pinned_bytes = 0
        return HEAD_BYTES

    def promote_to_pin(self, source_key, target_key, *, tensors=None):
        self.events.append(("promote", target_key))
        if not self.promote_ok:
            return None
        self.total_bytes = self.pinned_bytes = HEAD_BYTES
        return tensors

    def trim_to(self, limit):
        self.events.append(("trim", limit))
        return 123

    def discard(self, key, names):
        self.events.append(("discard", key))
        self.total_bytes = 0
        return True


def _engine(*, resident=False):
    head = SimpleNamespace(nbytes=HEAD_BYTES)
    events = []
    engine = object.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(
        glm53_phase_lm_head=True,
        qwen4_serial_verify_suspend_lm_head=False,
    )
    engine.cfg = SimpleNamespace(
        tie_word_embeddings=False, model_type="glm5_next", vocab_size=16)
    engine.tokenizer = SimpleNamespace(
        encode=lambda _prompt: SimpleNamespace(ids=[1, 2]))
    engine._streamed_lm_head = None
    engine.cache = _Cache(head, events)
    engine._lm_head_w = head if resident else None
    engine._qwen35_lm_head_pin_suspended = False
    engine._qwen4_lm_head_pin_suspended = False
    engine._glm53_lm_head_pin_suspended = not resident
    engine._glm53_phase_head_bytes = HEAD_BYTES
    engine._reset_glm53_phase_head_request_stats()
    engine.governor = SimpleNamespace(
        reserve=lambda incoming, **kwargs: events.append(
            ("reserve", incoming, kwargs["reason"])))
    return engine, head, events


def _adapter(monkeypatch, target, *, fail=False):
    def generate(*args, **kwargs):
        target._lm_head_weight()
        if fail:
            raise ValueError("generation failed")
        return {
            "first_token_s": 1.0,
            "total_s": 2.0,
            "path_stats": {
                "lm_head_pinned": 1,
                "lm_head_pinned_bytes": HEAD_BYTES,
                "weight_cache_resident_bytes": HEAD_BYTES,
                "weight_cache_pinned_bytes": HEAD_BYTES,
            },
        }

    monkeypatch.setattr(
        "runtime.speculative.SpeculativeDecoder",
        lambda *args, **kwargs: SimpleNamespace(
            generate=generate,
            mtp=SimpleNamespace(configure_confidence=lambda **kwargs: None)))
    target.generate = generate
    target.generate_with_memory_retry = generate
    return NativeMTPEngine(target)


def test_restore_admits_before_fetch_and_promotes_without_copy():
    engine, head, events = _engine()
    assert engine._lm_head_weight() is head
    assert events == [
        ("trim", 2_800_000_000 - HEAD_BYTES),
        ("reserve", HEAD_BYTES, "glm53-phase-lm-head"),
        ("get", "lm_head"),
        ("promote", "glm53:lm_head:persistent"),
    ]
    assert engine._glm53_phase_head_restore_successes == 1
    assert engine._lm_head_weight() is head
    assert len(events) == 4


def test_governor_refusal_prevents_head_fetch():
    engine, _head, events = _engine()

    def refuse(*args, **kwargs):
        raise MemoryError("insufficient headroom")

    engine.governor.reserve = refuse
    with pytest.raises(MemoryError, match="insufficient"):
        engine._lm_head_weight()
    assert events == [("trim", 2_800_000_000 - HEAD_BYTES)]
    assert engine._glm53_lm_head_pin_suspended


def test_zero_release_preserves_live_reference_and_reports_no_reclamation():
    engine, head, _events = _engine(resident=True)
    engine.cache.release_ok = False
    assert engine._suspend_glm53_phase_lm_head() == 0
    assert engine._lm_head_w is head
    assert not engine._glm53_lm_head_pin_suspended
    assert engine._glm53_phase_head_suspend_bytes == 0


def test_refused_promotion_cleans_up_unpinned_head():
    engine, head, events = _engine()
    engine.cache.promote_ok = False
    assert engine._lm_head_weight() is head
    assert engine._lm_head_w is None
    assert engine._glm53_lm_head_pin_suspended
    assert engine._suspend_glm53_phase_lm_head() == 0
    assert events[-1] == ("discard", "lm_head")
    assert engine._glm53_phase_head_unpinned_cleanup_calls == 1


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_native_success_and_fallback_release_and_refresh_telemetry(
        monkeypatch, max_tokens):
    engine, _head, _events = _engine()
    adapter = _adapter(monkeypatch, engine)
    for _request in range(2):
        result = adapter.generate("prompt", max_tokens=max_tokens)
        stats = result["path_stats"]
        assert stats["glm53_mtp_used"] == int(max_tokens > 1)
        assert stats["glm53_phase_lm_head_idle_release_bytes"] == HEAD_BYTES
        assert stats["glm53_phase_lm_head_suspend_calls"] == 1
        assert stats["glm53_phase_lm_head_restore_successes"] == 1
        assert stats["lm_head_pinned"] == stats["lm_head_pinned_bytes"] == 0
        assert stats["weight_cache_resident_bytes"] == 0
        assert stats["weight_cache_pinned_bytes"] == 0
        assert result["total_s"] >= 2.0
        assert engine._lm_head_w is None
        assert engine._glm53_lm_head_pin_suspended


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_native_and_fallback_errors_still_release(monkeypatch, max_tokens):
    engine, _head, events = _engine()
    adapter = _adapter(monkeypatch, engine, fail=True)
    with pytest.raises(ValueError, match="generation failed"):
        adapter.generate("prompt", max_tokens=max_tokens)
    assert engine._lm_head_w is None
    assert engine._glm53_lm_head_pin_suspended
    assert events[-1] == ("release", "glm53:lm_head:persistent")


@pytest.mark.parametrize("generation_fails", [False, True])
def test_cleanup_failure_is_reported_without_masking_original(
        monkeypatch, generation_fails):
    engine, _head, _events = _engine()
    adapter = _adapter(monkeypatch, engine, fail=generation_fails)

    def fail_release(*args):
        raise RuntimeError("cache cleanup failed")

    engine.cache.release_pinned = fail_release
    expected = ValueError if generation_fails else RuntimeError
    with pytest.raises(expected) as error:
        adapter.generate("prompt", max_tokens=4)
    assert engine._lm_head_w is not None
    if generation_fails:
        assert any("cleanup also failed" in note
                   for note in error.value.__notes__)


def test_cleanup_error_is_not_masked_by_callers_handled_exception(monkeypatch):
    engine, _head, _events = _engine()
    adapter = _adapter(monkeypatch, engine)

    def fail_release(*args):
        raise RuntimeError("cache cleanup failed")

    engine.cache.release_pinned = fail_release
    try:
        raise ValueError("unrelated caller exception")
    except ValueError:
        with pytest.raises(RuntimeError, match="cache cleanup failed"):
            adapter.generate("prompt", max_tokens=4)


def test_disabled_phase_mode_does_not_run_cleanup(monkeypatch):
    engine, _head, events = _engine(resident=True)
    engine.rc.glm53_phase_lm_head = False
    result = _adapter(monkeypatch, engine).generate("prompt", max_tokens=4)
    assert "glm53_phase_lm_head" not in result["path_stats"]
    assert not events


def test_real_cache_release_drops_last_metal_reference():
    engine, _head, _events = _engine()
    size = 2048 * 2048 * 2

    class Store:
        def fetch(self, names):
            value = mx.ones((2048, 2048), dtype=mx.bfloat16)
            mx.eval(value)
            return {"lm_head.weight": value}, 0.0, size

    engine.cache = WeightCache(Store(), max_bytes=16_000_000)
    engine._glm53_phase_head_bytes = size
    engine.cache.register_suspended_pin("glm53:lm_head:persistent", size)
    mx.clear_cache()
    initial = mx.get_active_memory()
    engine._lm_head_weight()
    before = mx.get_active_memory()
    assert before >= initial + size
    assert engine._suspend_glm53_phase_lm_head() == size
    assert engine.cache.pinned_bytes == engine.cache.total_bytes == 0
    assert mx.get_active_memory() <= before - size
    assert engine._glm53_phase_head_suspend_active_released_bytes >= size


def test_late_real_cache_release_failure_does_not_restore_unowned_head():
    engine, _head, _events = _engine()
    size = 2048 * 2048 * 2

    class Store:
        def fetch(self, names):
            value = mx.ones((2048, 2048), dtype=mx.bfloat16)
            mx.eval(value)
            return {"lm_head.weight": value}, 0.0, size

        def release_cache_pages(self, names):
            raise RuntimeError("late mapping cleanup failed")

    engine.cache = WeightCache(Store(), max_bytes=16_000_000)
    engine._glm53_phase_head_bytes = size
    engine.cache.register_suspended_pin("glm53:lm_head:persistent", size)
    engine._lm_head_weight()
    before = mx.get_active_memory()
    with pytest.raises(RuntimeError, match="late mapping cleanup failed"):
        engine._suspend_glm53_phase_lm_head()
    assert engine._lm_head_w is None
    assert engine._glm53_lm_head_pin_suspended
    assert engine.cache.pinned_bytes == engine.cache.total_bytes == 0
    assert engine.cache.suspended_pin_bytes("glm53:lm_head:persistent") == size
    assert mx.get_active_memory() <= before - size
