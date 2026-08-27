"""Exact post-prefill Qwen4 LM-head lease gates."""

from types import SimpleNamespace

from runtime.engine import StreamingEngine


class _Cache:
    def __init__(self, head):
        self.head = head
        self.release_calls = []
        self.promote_calls = []
        self.get_calls = []

    def get(self, key, names):
        self.get_calls.append((key, tuple(names)))
        return {"lm_head.weight": self.head}

    def release_pinned(self, key, names):
        self.release_calls.append((key, tuple(names)))
        return 1_271_930_880

    def promote_to_pin(self, source_key, target_key, *, tensors=None):
        self.promote_calls.append((source_key, target_key, tensors))
        return {"lm_head.weight": self.head}


def _engine(*, resident=True):
    head = object()
    engine = object.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(qwen4_phase_lm_head=True)
    engine.cfg = SimpleNamespace(tie_word_embeddings=False)
    engine._streamed_lm_head = None
    engine.cache = _Cache(head)
    engine._lm_head_w = head if resident else None
    engine._qwen35_lm_head_pin_suspended = False
    engine._qwen4_lm_head_pin_suspended = not resident
    engine._qwen4_phase_head_suspend_calls = 0
    engine._qwen4_phase_head_suspend_bytes = 0
    engine._qwen4_phase_head_suspend_s = 0.0
    engine._qwen4_phase_head_restore_calls = 0
    engine._qwen4_phase_head_restore_successes = 0
    engine._qwen4_phase_head_restore_refusals = 0
    engine._qwen4_phase_head_restore_s = 0.0
    return engine, head


def test_prior_decode_head_is_released_before_next_prefill():
    engine, _head = _engine(resident=True)

    released = engine._suspend_qwen4_phase_lm_head()

    assert released == 1_271_930_880
    assert engine._lm_head_w is None
    assert engine._qwen4_lm_head_pin_suspended is True
    assert engine.cache.release_calls == [(
        "qwen4:lm_head:persistent", ("lm_head.weight",))]


def test_first_post_prefill_projection_restores_exact_lease_without_copy():
    engine, head = _engine(resident=False)

    assert engine._lm_head_weight() is head
    assert engine._lm_head_w is head
    assert engine._qwen4_lm_head_pin_suspended is False
    assert engine.cache.get_calls == [("lm_head", ("lm_head.weight",))]
    assert engine.cache.promote_calls == [(
        "lm_head", "qwen4:lm_head:persistent",
        {"lm_head.weight": head})]
    assert engine._qwen4_phase_head_restore_successes == 1


def test_disabled_qwen4_phase_head_is_neutral():
    engine, head = _engine(resident=True)
    engine.rc.qwen4_phase_lm_head = False

    assert engine._suspend_qwen4_phase_lm_head() == 0
    assert engine._restore_qwen4_phase_lm_head() is False
    assert engine._lm_head_w is head
    assert engine.cache.release_calls == []
    assert engine.cache.promote_calls == []
