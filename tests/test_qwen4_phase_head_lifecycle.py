"""Exact post-prefill Qwen4 LM-head lease gates."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.engine import StreamingEngine
from runtime.weight_cache import WeightCache


HEAD_BYTES = 1_271_398_400


class _Cache:
    def __init__(self, head):
        self.head = head
        self.release_calls = []
        self.promote_calls = []
        self.get_calls = []
        self.trim_calls = []
        self.events = []

    def get(self, key, names):
        self.events.append(("get", key))
        self.get_calls.append((key, tuple(names)))
        return {"lm_head.weight": self.head}

    def release_pinned(self, key, names):
        self.release_calls.append((key, tuple(names)))
        return HEAD_BYTES

    def promote_to_pin(self, source_key, target_key, *, tensors=None):
        self.events.append(("promote", target_key))
        self.promote_calls.append((source_key, target_key, tensors))
        return {"lm_head.weight": self.head}

    def trim_to(self, target_bytes):
        self.events.append(("trim", target_bytes))
        self.trim_calls.append(target_bytes)
        return 123


class _Governor:
    def __init__(self, events):
        self.events = events
        self.reservation_fast_path_calls = 0
        self.reservation_clear_cache_only_calls = 0
        self.reservation_cache_released_bytes = 0
        self.reservation_budget_reduced_bytes = 0
        self.reservation_budget_restored_bytes = 0
        self.reservation_unproductive_shrinks = 0
        self.reservation_zero_release_short_circuits = 0
        self.reservations = 0

    def reserve(self, incoming, *, reason):
        self.events.append(("reserve", incoming, reason))
        self.reservation_fast_path_calls += 1


def _engine(*, resident=True):
    head = object()
    engine = object.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(
        qwen4_phase_lm_head=True,
        qwen4_serial_verify_suspend_lm_head=False,
    )
    engine.cfg = SimpleNamespace(tie_word_embeddings=False)
    engine._streamed_lm_head = None
    engine.cache = _Cache(head)
    engine.governor = _Governor(engine.cache.events)
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
    engine._qwen4_phase_head_bytes = HEAD_BYTES
    engine._qwen4_phase_head_admission_stats = {}
    engine._qwen4_serial_verify_head_suspend_calls = 0
    engine._qwen4_serial_verify_head_suspend_bytes = 0
    engine._qwen4_serial_verify_head_restore_trim_bytes = 0
    return engine, head


def test_prior_decode_head_is_released_before_next_prefill():
    engine, _head = _engine(resident=True)

    released = engine._suspend_qwen4_phase_lm_head()

    assert released == HEAD_BYTES
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


def test_opt_in_serial_verifier_releases_and_counts_exact_head():
    engine, _head = _engine(resident=True)
    engine.rc.qwen4_serial_verify_suspend_lm_head = True

    released = engine._suspend_qwen4_serial_verify_lm_head()

    assert released == HEAD_BYTES
    assert engine._lm_head_w is None
    assert engine._qwen4_serial_verify_head_suspend_calls == 1
    assert engine._qwen4_serial_verify_head_suspend_bytes == HEAD_BYTES


def test_serial_verifier_trims_consumed_trunk_before_exact_head_restore():
    engine, head = _engine(resident=False)
    engine.rc.qwen4_serial_verify_suspend_lm_head = True

    assert engine._lm_head_weight() is head
    assert engine.cache.trim_calls == [0]
    assert engine._qwen4_serial_verify_head_restore_trim_bytes == 123


def test_disabled_serial_verifier_does_not_change_phase_lease():
    engine, head = _engine(resident=True)

    assert engine._suspend_qwen4_serial_verify_lm_head() == 0
    assert engine._lm_head_w is head
    assert engine._qwen4_serial_verify_head_suspend_calls == 0
    assert engine.cache.release_calls == []


def test_exact_head_admission_precedes_fetch_and_same_object_promotion():
    engine, head = _engine(resident=False)
    engine.rc.qwen4_serial_verify_suspend_lm_head = True
    assert engine._lm_head_weight() is head
    assert engine.cache.events == [
        ("trim", 0), ("reserve", HEAD_BYTES, "qwen4-phase-lm-head"),
        ("get", "lm_head"), ("promote", "qwen4:lm_head:persistent")]
    stats = engine._qwen4_phase_head_admission_stats
    assert stats["calls"] == stats["reservation_fast_path_calls"] == 1
    assert stats["requested_bytes"] == HEAD_BYTES
    assert stats["refusals"] == 0 and stats["seconds"] >= 0
    assert stats["reservation_cache_released_bytes"] == 0
    assert engine._lm_head_weight() is head
    assert len(engine.cache.events) == 4 and stats["calls"] == 1


def test_head_admission_refusal_keeps_dormant_lease_and_never_fetches():
    engine, _head = _engine(resident=False)
    engine.rc.qwen4_serial_verify_suspend_lm_head = True
    error = MemoryError("unsafe incoming head")

    def refuse(incoming, *, reason):
        engine.cache.events.append(("reserve", incoming, reason))
        raise error

    engine.governor.reserve = refuse
    with pytest.raises(MemoryError) as caught:
        engine._lm_head_weight()
    assert caught.value is error
    assert engine.cache.events == [
        ("trim", 0), ("reserve", HEAD_BYTES, "qwen4-phase-lm-head")]
    assert engine._lm_head_w is None and engine._qwen4_lm_head_pin_suspended
    assert not engine.cache.get_calls and not engine.cache.promote_calls
    assert engine._qwen4_phase_head_admission_stats["refusals"] == 1


@pytest.mark.parametrize("bad_bytes", [True, None, 0, -1, 1.5, "1271398400"])
def test_invalid_phase_head_metadata_refuses_before_fetch(bad_bytes):
    engine, _head = _engine(resident=False)
    engine._qwen4_phase_head_bytes = bad_bytes
    with pytest.raises(ValueError, match="exact positive byte metadata"):
        engine._lm_head_weight()
    assert not engine.cache.get_calls and not engine.cache.promote_calls
    assert engine._qwen4_lm_head_pin_suspended
    assert engine._qwen4_phase_head_admission_stats == {}


@pytest.mark.parametrize("path", ["resident", "tied", "streamed"])
def test_owned_tied_and_streamed_heads_do_not_reserve(path):
    engine, head = _engine(resident=path == "resident")
    if path == "tied":
        engine.cfg.tie_word_embeddings = True
        engine._tied_lm_head_w = head
    elif path == "streamed":
        engine._streamed_lm_head = head
    assert engine._lm_head_weight() is head
    assert engine.cache.events == []
    assert engine._qwen4_phase_head_admission_stats == {}


def test_no_governor_keeps_existing_exact_fetch_and_promotion():
    engine, head = _engine(resident=False)
    engine.governor = None
    assert engine._lm_head_weight() is head
    assert engine.cache.events == [
        ("get", "lm_head"), ("promote", "qwen4:lm_head:persistent")]
    assert engine._qwen4_phase_head_admission_stats == {}


def test_non_qwen_phase_head_does_not_gain_a_qwen_admission():
    engine, head = _engine(resident=False)
    engine.rc.qwen4_phase_lm_head = False
    engine._qwen4_lm_head_pin_suspended = False
    assert engine._lm_head_weight() is head
    assert engine.cache.events == [("get", "lm_head")]
    assert engine._qwen4_phase_head_admission_stats == {}


def test_head_admission_counters_are_deltas_not_previous_unrelated_work():
    engine, _head = _engine(resident=False)
    engine.governor.reservation_fast_path_calls = 900
    engine.governor.reservation_cache_released_bytes = 500
    engine.governor.reservations = 42
    engine._lm_head_weight()
    stats = engine._qwen4_phase_head_admission_stats
    assert stats["reservation_fast_path_calls"] == 1
    assert stats["reservation_cache_released_bytes"] == stats["reservations"] == 0
    engine._suspend_qwen4_phase_lm_head()
    engine._lm_head_weight()
    assert stats["calls"] == stats["reservation_fast_path_calls"] == 2
    assert stats["requested_bytes"] == 2 * HEAD_BYTES


def test_real_tiny_bf16_head_admits_fetches_releases_and_restores():
    engine, _head = _engine(resident=False)
    size = 2048 * 2048 * 2
    events = engine.cache.events

    class Store:
        def fetch(self, names):
            events.append(("fetch-real", tuple(names)))
            value = mx.ones((2048, 2048), dtype=mx.bfloat16)
            mx.eval(value)
            return {"lm_head.weight": value}, 0.0, size

    engine.cache = WeightCache(Store(), max_bytes=16_000_000)
    engine._qwen4_phase_head_bytes = size
    engine.cache.register_suspended_pin("qwen4:lm_head:persistent", size)
    for iteration in (1, 2):
        head = engine._lm_head_weight()
        assert head is engine._lm_head_w
        assert head.dtype == mx.bfloat16 and head[0, 0].item() == 1.0
        assert engine.cache.pinned_bytes == size
        assert engine._lm_head_weight() is head  # owns it, no second admission
        # Complete the tiny test consumer before asserting immediate ownership
        # reclamation; do not hide a failed release with post-release sync.
        mx.synchronize()
        del head
        before = mx.get_active_memory()
        assert engine._suspend_qwen4_phase_lm_head() == size
        assert engine.cache.pinned_bytes == engine.cache.total_bytes == 0
        assert engine.cache.suspended_pin_bytes("qwen4:lm_head:persistent") == size
        assert mx.get_active_memory() <= before - size
        stats = engine._qwen4_phase_head_admission_stats
        assert stats["calls"] == stats["reservation_fast_path_calls"] == iteration
    assert events == [
        ("reserve", size, "qwen4-phase-lm-head"),
        ("fetch-real", ("lm_head.weight",)),
    ] * 2
