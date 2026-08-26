"""Exact phase-scoped Qwen LM-head lifetime gates."""

from types import SimpleNamespace

from runtime.engine import (
    QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS,
    StreamingEngine,
    qwen35_phase_head_request_active,
)


class _Cache:
    def __init__(self, head):
        self.head = head
        self.release_calls = []
        self.promote_calls = []
        self.get_calls = []
        self.promote = True

    def get(self, key, names):
        self.get_calls.append((key, tuple(names)))
        return {"lm_head.weight": self.head}

    def release_pinned(self, key, names):
        self.release_calls.append((key, tuple(names)))
        return 675_000_000

    def promote_to_pin(self, source_key, target_key, *, tensors=None):
        self.promote_calls.append((source_key, target_key, tensors))
        if not self.promote:
            return None
        return {"lm_head.weight": self.head}

    def register_suspended_pin(self, key, nbytes):
        raise AssertionError("constructor-only API is not used by these gates")


def _engine():
    head = object()
    engine = object.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(
        qwen35_serial_verify_suspend_lm_head=True)
    engine.cfg = SimpleNamespace(tie_word_embeddings=False)
    engine._streamed_lm_head = None
    engine.cache = _Cache(head)
    engine._lm_head_w = head
    engine._qwen35_lm_head_pin_suspended = False
    engine._qwen35_lm_head_suspend_request_active = True
    engine._qwen35_serial_verify_head_suspend_calls = 0
    engine._qwen35_serial_verify_head_suspend_bytes = 0
    engine._qwen35_serial_verify_head_suspend_active_released_bytes = 0
    engine._qwen35_serial_verify_head_suspend_active_peak_bytes = 0
    engine._qwen35_serial_verify_head_suspend_s = 0.0
    engine._qwen35_serial_verify_head_restore_calls = 0
    engine._qwen35_serial_verify_head_restore_successes = 0
    engine._qwen35_serial_verify_head_restore_refusals = 0
    engine._qwen35_serial_verify_head_restore_s = 0.0
    return engine, head


def test_serial_verifier_releases_then_restores_same_head_without_fetch():
    engine, head = _engine()

    assert engine._suspend_qwen35_serial_verify_lm_head() == 675_000_000
    assert engine._lm_head_w is None
    assert engine.cache.release_calls == [(
        "qwen35:lm_head:persistent", ("lm_head.weight",))]

    assert engine._restore_qwen35_serial_verify_lm_head(head) is True
    assert engine._lm_head_w is head
    assert engine.cache.promote_calls == [(
        "lm_head", "qwen35:lm_head:persistent",
        {"lm_head.weight": head})]
    assert engine._qwen35_serial_verify_head_suspend_calls == 1
    assert engine._qwen35_serial_verify_head_suspend_bytes == 675_000_000
    assert engine._qwen35_serial_verify_head_restore_calls == 1
    assert engine._qwen35_serial_verify_head_restore_successes == 1
    assert engine._qwen35_serial_verify_head_restore_refusals == 0


def test_serial_verifier_restore_refusal_leaves_ordinary_demand_path():
    engine, _head = _engine()
    engine._suspend_qwen35_serial_verify_lm_head()
    engine.cache.promote = False

    assert engine._restore_qwen35_serial_verify_lm_head(_head) is False
    assert engine._lm_head_w is None
    assert engine._qwen35_serial_verify_head_restore_calls == 1
    assert engine._qwen35_serial_verify_head_restore_successes == 0
    assert engine._qwen35_serial_verify_head_restore_refusals == 1


def test_first_head_projection_restores_suspended_pin_automatically():
    engine, head = _engine()
    engine._suspend_qwen35_serial_verify_lm_head()

    assert engine._lm_head_weight() is head
    assert engine._lm_head_w is head
    assert engine._qwen35_lm_head_pin_suspended is False
    assert engine.cache.get_calls == [
        ("lm_head", ("lm_head.weight",))]
    assert engine._qwen35_serial_verify_head_restore_successes == 1


def test_disabled_serial_verifier_head_lifecycle_is_neutral():
    engine, head = _engine()
    engine.rc.qwen35_serial_verify_suspend_lm_head = False

    assert engine._suspend_qwen35_serial_verify_lm_head() == 0
    assert engine._restore_qwen35_serial_verify_lm_head() is False
    assert engine._lm_head_w is head
    assert engine.cache.release_calls == []
    assert engine.cache.promote_calls == []


def test_phase_head_request_gate_is_content_blind_and_long_context_only():
    boundary = QWEN35_PHASE_HEAD_MIN_PROMPT_TOKENS

    assert not qwen35_phase_head_request_active(False, boundary + 1)
    assert not qwen35_phase_head_request_active(True, boundary - 1)
    assert qwen35_phase_head_request_active(True, boundary)
    assert qwen35_phase_head_request_active(True, boundary + 1)


def test_phase_head_request_gate_accepts_explicit_content_blind_boundary():
    assert not qwen35_phase_head_request_active(True, 6_338, 6_339)
    assert qwen35_phase_head_request_active(True, 6_339, 6_339)
