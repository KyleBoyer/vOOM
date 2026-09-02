"""F95: per-conversation prefill_chunk_size adaptivity for qwen3_5/
qwen3_5_moe hot_prompt_kv targets.

Tests the two small, directly-callable helpers StreamingEngine.generate()
uses (StreamingEngine._hybrid_chunk_size_applies,
StreamingEngine._select_prefill_chunk_size) in isolation, using a bare
engine object (no real model/weights needed) -- the same pattern already
used by tests/test_hot_prompt_kv.py for other hot-KV internals.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_prefill_chunk_ceiling_contract_is_sparse_and_default_neutral():
    from runtime.engine import (
        QWEN35_PREFILL_CHUNK_CEILINGS, RuntimeConfig)

    assert QWEN35_PREFILL_CHUNK_CEILINGS == {0, 1, 8, 32, 128, 512}
    assert RuntimeConfig().qwen35_prefill_chunk_ceiling == 0


def _bare_engine(model_type: str, hot_kv_persist,
                 hot_prompt_kv_chunk_size: int = 128,
                 qwen35_prefill_chunk_ceiling: int = 0):
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type=model_type)
    engine._hot_kv_persist = hot_kv_persist
    engine.rc = SimpleNamespace(
        hot_prompt_kv_chunk_size=hot_prompt_kv_chunk_size,
        prefill_chunk_size=hot_prompt_kv_chunk_size,
        qwen35_prefill_chunk_ceiling=qwen35_prefill_chunk_ceiling,
        adaptive_chunk_size=False,
        layer_stationary_prefill=False,
        hot_prompt_kv=False,
        quant_bits=0)
    return engine


def test_applies_only_for_hybrid_models_with_persistence_off():
    for model_type in ("qwen3_5", "qwen3_5_moe"):
        assert _bare_engine(model_type, hot_kv_persist=None)._hybrid_chunk_size_applies()

    for model_type in ("qwen2", "qwen3", "glm_moe_dsa", "gpt_oss", "olmoe",
                        "kimi_linear"):
        assert not _bare_engine(
            model_type, hot_kv_persist=None)._hybrid_chunk_size_applies()

    # F95: durable persistence bakes ONE chunk size into its on-disk format
    # (HotPromptKVPersistence) -- adaptivity is skipped whenever it's
    # active, even for an otherwise-eligible model type.
    assert not _bare_engine(
        "qwen3_5_moe", hot_kv_persist=object())._hybrid_chunk_size_applies()


def test_select_chunk_size_reuses_matched_slot_without_sampling_memory():
    """Continuing a specific conversation lineage MUST reuse whatever
    chunk size actually built that slot's KV/recurrent state -- it must
    NOT resample live memory (a different conversation could be running
    under completely different conditions right now)."""
    from runtime.engine import _HotPromptSlot

    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None,
                          hot_prompt_kv_chunk_size=128)
    slot = _HotPromptSlot(
        tokens=(1, 2, 3), kv=None, logits=None, prompt_length=3,
        prompt_logits=None, reusable_prefix=0, chunk_size=512)

    with patch("runtime.engine.psutil.virtual_memory") as mock_vm:
        result = engine._select_prefill_chunk_size(slot)
        mock_vm.assert_not_called()
    assert result == 512


def test_select_chunk_size_falls_back_to_engine_default_if_slot_unset():
    """A slot somehow missing chunk_size (e.g. hypothetically constructed
    by older/foreign code) falls back to the engine's current
    hot_prompt_kv_chunk_size rather than crashing or silently using 0."""
    from runtime.engine import _HotPromptSlot

    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None,
                          hot_prompt_kv_chunk_size=64)
    slot = _HotPromptSlot(
        tokens=(1,), kv=None, logits=None, prompt_length=1,
        prompt_logits=None, reusable_prefix=0, chunk_size=0)

    assert engine._select_prefill_chunk_size(slot) == 64


def test_select_chunk_size_samples_fresh_memory_for_new_conversation():
    """matched_slot=None (brand-new conversation, no match at all) samples
    LIVE memory right then via the same hybrid_prefill_chunk_size ladder
    used at server-side construction -- proving this is a genuine fresh
    read, not a cached/stale value."""
    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None,
                          hot_prompt_kv_chunk_size=999)

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=10_000_000_000)):
        assert engine._select_prefill_chunk_size(None) == 512

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=500_000_000)):
        assert engine._select_prefill_chunk_size(None) == 8


def test_explicit_ceiling_clamps_live_selection_but_zero_preserves_auto():
    auto = _bare_engine("qwen3_5", hot_kv_persist=None)
    capped = _bare_engine(
        "qwen3_5", hot_kv_persist=None,
        qwen35_prefill_chunk_ceiling=128)

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=10_000_000_000)):
        assert auto._select_prefill_chunk_size(None) == 512
        assert capped._select_prefill_chunk_size(None) == 128


def test_explicit_ceiling_is_applied_after_retry_ceiling():
    engine = _bare_engine(
        "qwen3_5_moe", hot_kv_persist=None,
        qwen35_prefill_chunk_ceiling=128)
    engine._hybrid_retry_chunk_ceiling = 32

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=10_000_000_000)):
        assert engine._select_prefill_chunk_size(None) == 32

    engine.rc.qwen35_prefill_chunk_ceiling = 8
    engine._hybrid_retry_chunk_ceiling = 128
    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=10_000_000_000)):
        assert engine._select_prefill_chunk_size(None) == 8


def test_explicit_ceiling_also_clamps_fixed_durable_execution():
    engine = _bare_engine(
        "qwen3_5", hot_kv_persist=object(),
        hot_prompt_kv_chunk_size=512,
        qwen35_prefill_chunk_ceiling=32)

    assert not engine._hybrid_chunk_size_applies()
    assert engine._apply_qwen35_prefill_chunk_ceiling(
        engine.rc.prefill_chunk_size) == 32


def test_hot_kv_admission_uses_effective_qwen_ceiling_before_execution():
    engine = _bare_engine(
        "qwen3_5", hot_kv_persist=object(),
        hot_prompt_kv_chunk_size=512,
        qwen35_prefill_chunk_ceiling=32)

    assert engine._prefill_admission_positions(16_000) == 32
    assert engine._prefill_admission_positions(7) == 7

    non_qwen = _bare_engine(
        "glm_moe_dsa", hot_kv_persist=None,
        hot_prompt_kv_chunk_size=512,
        qwen35_prefill_chunk_ceiling=32)
    assert non_qwen._prefill_admission_positions(16_000) == 512


def test_two_conversations_can_use_different_chunk_sizes_independently():
    """The actual point of this feature: slot A (built under tight memory)
    and slot B (built under healthy memory) coexist with DIFFERENT
    recorded chunk sizes, and each is retrieved independently without
    disturbing the other -- proving chunk size is now a per-lineage
    property, not an engine-wide constant."""
    from runtime.engine import _HotPromptSlot

    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None)
    tight_slot = _HotPromptSlot(
        tokens=(1, 2), kv=None, logits=None, prompt_length=2,
        prompt_logits=None, reusable_prefix=0, chunk_size=8)
    healthy_slot = _HotPromptSlot(
        tokens=(9, 9, 9), kv=None, logits=None, prompt_length=3,
        prompt_logits=None, reusable_prefix=0, chunk_size=512)

    assert engine._select_prefill_chunk_size(tight_slot) == 8
    assert engine._select_prefill_chunk_size(healthy_slot) == 512
    # Retrieving one again is unaffected by having just retrieved the other.
    assert engine._select_prefill_chunk_size(tight_slot) == 8


def test_memory_retry_replays_unsampled_prefill_on_lower_rungs():
    from runtime.engine import StreamingEngine

    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None,
                          hot_prompt_kv_chunk_size=512)
    engine.rc.prefill_chunk_size = 512
    attempts = []
    discards = []
    progress = []

    def generate(*_args, **_kwargs):
        attempts.append(engine.rc.prefill_chunk_size)
        engine._generation_sampled_tokens = 0
        if len(attempts) < 3:
            raise MemoryError("synthetic governor refusal")
        return {
            "prefill_s": 1.0, "first_token_s": 1.5, "total_s": 2.0,
            "path_stats": {},
        }

    engine.generate = generate
    engine.discard_failed_request_state = lambda: discards.append(True)
    with (patch("runtime.engine.mx.clear_cache"),
          patch("runtime.engine.mx.reset_peak_memory")):
        result = StreamingEngine.generate_with_memory_retry(
            engine, "prompt", on_progress=progress.append)

    assert attempts == [512, 128, 32]
    assert len(discards) == 2
    assert result["path_stats"]["memory_prefill_retries"] == 2
    assert result["path_stats"]["memory_prefill_retry_chunks"] == [128, 32]
    assert progress == [
        {
            "phase": "memory_retry",
            "completed_retries": 1,
            "total_retries": 4,
            "retry_chunk": 128,
            "retry_coalesced_expert_max_positions": 0,
        },
        {
            "phase": "memory_retry",
            "completed_retries": 2,
            "total_retries": 4,
            "retry_chunk": 32,
            "retry_coalesced_expert_max_positions": 0,
        },
    ]
    cleanup = result["path_stats"]["memory_prefill_retry_cleanup"]
    assert [entry["chunk"] for entry in cleanup] == [128, 32]
    assert all(entry["released_bytes"] >= 0 for entry in cleanup)
    assert result["total_s"] >= 2.0
    assert engine._hybrid_retry_chunk_ceiling == 0


def test_memory_retry_never_replays_after_sampling_started():
    from runtime.engine import StreamingEngine

    engine = _bare_engine("qwen3_5_moe", hot_kv_persist=None,
                          hot_prompt_kv_chunk_size=512)
    engine.rc.prefill_chunk_size = 512
    calls = []

    def generate(*_args, **_kwargs):
        calls.append(True)
        engine._generation_sampled_tokens = 1
        raise MemoryError("decode refusal")

    engine.generate = generate
    engine.discard_failed_request_state = lambda: None
    try:
        StreamingEngine.generate_with_memory_retry(engine, "prompt")
    except MemoryError as error:
        assert "decode refusal" in str(error)
    else:
        raise AssertionError("decode MemoryError must propagate")
    assert len(calls) == 1


def test_glm53_long_prefill_retries_at_intermediate_64_tile():
    from runtime.engine import StreamingEngine

    engine = _bare_engine(
        "glm5_next", hot_kv_persist=None,
        hot_prompt_kv_chunk_size=128)
    engine.rc.layer_stationary_prefill = True
    attempts = []

    def generate(*_args, **_kwargs):
        attempts.append(engine.rc.prefill_chunk_size)
        engine._generation_sampled_tokens = 0
        if len(attempts) == 1:
            raise MemoryError("synthetic 8.5GB GLM prefill cap")
        return {
            "prefill_s": 1.0, "first_token_s": 1.5, "total_s": 2.0,
            "path_stats": {},
        }

    engine.generate = generate
    engine.discard_failed_request_state = lambda: None
    with (patch("runtime.engine.mx.clear_cache"),
          patch("runtime.engine.mx.reset_peak_memory")):
        result = StreamingEngine.generate_with_memory_retry(engine, "prompt")

    assert attempts == [128, 64]
    assert result["path_stats"]["memory_prefill_retry_chunks"] == [64]
    assert engine._hybrid_retry_chunk_ceiling == 0


def test_glm53_coalesced_prefill_retries_position_limit_before_tile_width():
    from runtime.engine import StreamingEngine

    engine = _bare_engine(
        "glm5_next", hot_kv_persist=None,
        hot_prompt_kv_chunk_size=32)
    engine.rc.layer_stationary_prefill = True
    engine.rc.glm53_coalesced_expert_positions = True
    engine.rc.glm53_coalesced_expert_max_positions = 512
    attempts = []

    def generate(*_args, **_kwargs):
        attempts.append((
            engine.rc.prefill_chunk_size,
            engine.rc.glm53_coalesced_expert_max_positions))
        engine._generation_sampled_tokens = 0
        if len(attempts) == 1:
            raise MemoryError("synthetic coalesced expert operand refusal")
        return {
            "prefill_s": 1.0, "first_token_s": 1.5, "total_s": 2.0,
            "path_stats": {},
        }

    engine.generate = generate
    engine.discard_failed_request_state = lambda: None
    with (patch("runtime.engine.mx.clear_cache"),
          patch("runtime.engine.mx.reset_peak_memory")):
        result = StreamingEngine.generate_with_memory_retry(engine, "prompt")

    assert attempts == [(32, 512), (32, 256)]
    assert result["path_stats"]["memory_prefill_retry_chunks"] == []
    assert result["path_stats"][
        "memory_prefill_retry_coalesced_limits"] == [256]
    assert result["path_stats"]["memory_prefill_retries"] == 1
    assert engine._hybrid_retry_chunk_ceiling == 0


def test_glm53_retry_requires_layer_stationary_prefill_and_no_persistence():
    engine = _bare_engine("glm5_next", hot_kv_persist=None)
    engine.rc.layer_stationary_prefill = True
    assert engine._memory_prefill_retry_applies()

    engine.rc.layer_stationary_prefill = False
    assert not engine._memory_prefill_retry_applies()
    engine.rc.layer_stationary_prefill = True
    engine._hot_kv_persist = object()
    assert not engine._memory_prefill_retry_applies()


def test_memory_retry_also_applies_to_lossy_dense_qwen_without_persistence():
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="qwen2")
    engine._hot_kv_persist = None
    engine.rc = SimpleNamespace(
        hot_prompt_kv=True, quant_bits=4, adaptive_chunk_size=False,
        prefill_chunk_size=32, hot_prompt_kv_chunk_size=32)
    assert engine._memory_prefill_retry_applies()

    engine.rc.quant_bits = 0
    assert not engine._memory_prefill_retry_applies()


def test_memory_retry_also_applies_to_any_adaptive_chunk_model():
    """F68's AdaptiveChunkController is used by gpt_oss/GLM/Kimi K3/etc, not
    just the qwen3_5 hybrid family _hybrid_chunk_size_applies covers -- an
    unstarted prefill is equally safe to discard and replay slower
    regardless of which model type enabled adaptive chunking. Live-
    confirmed 2026-07-29 (EpistemeAI/VibeCoder-20B, real gpt_oss checkpoint):
    a MemoryError from _fetch_experts' independent governor.reserve() call
    killed a whole request with zero retry coverage before this fix."""
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="gpt_oss")
    engine._hot_kv_persist = None
    engine.rc = SimpleNamespace(
        hot_prompt_kv=False, quant_bits=0, adaptive_chunk_size=True,
        prefill_chunk_size=64, hot_prompt_kv_chunk_size=64)
    assert engine._memory_prefill_retry_applies()

    engine.rc.adaptive_chunk_size = False
    assert not engine._memory_prefill_retry_applies()

    # Durable persistence still locks out retry even with adaptive chunking
    # on, same invariant as every other branch of this method.
    engine.rc.adaptive_chunk_size = True
    engine._hot_kv_persist = object()
    assert not engine._memory_prefill_retry_applies()


def test_memory_retry_pins_non_adaptive_chunk_after_expert_fetch_failure():
    """The retry ladder alone is not enough for adaptive-chunk models: a
    fresh AdaptiveChunkController would just grow back out from a smaller
    seed and can re-hit the same _fetch_experts reservation ceiling, since
    that budget is invisible to the controller's own compute-scratch check.
    Pin a hard, non-adaptive chunk once this retry fires, matching F68's
    own "never auto-restore after a shrink" stance.

    Uses a DEDICATED flag (_adaptive_chunk_pinned_after_retry), not
    rc.adaptive_chunk_size itself -- live-confirmed 2026-07-29 that
    flipping the config field directly breaks retry eligibility for any
    SUBSEQUENT failure in this same loop (see the next test), since
    _memory_prefill_retry_applies reads that same field."""
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="gpt_oss")
    engine._hot_kv_persist = None
    engine.rc = SimpleNamespace(
        hot_prompt_kv=False, quant_bits=0, adaptive_chunk_size=True,
        prefill_chunk_size=64, hot_prompt_kv_chunk_size=64)
    engine._hybrid_retry_chunk_ceiling = 0
    attempts = []

    def generate(*_args, **_kwargs):
        attempts.append(engine.rc.prefill_chunk_size)
        engine._generation_sampled_tokens = 0
        if len(attempts) < 2:
            raise MemoryError("synthetic expert-fetch reservation refusal")
        return {
            "prefill_s": 1.0, "first_token_s": 1.5, "total_s": 2.0,
            "path_stats": {},
        }

    engine.generate = generate
    engine.discard_failed_request_state = lambda: None
    with (patch("runtime.engine.mx.clear_cache"),
          patch("runtime.engine.mx.reset_peak_memory")):
        result = StreamingEngine.generate_with_memory_retry(engine, "prompt")

    assert attempts == [64, 32]
    assert engine.rc.adaptive_chunk_size is True
    assert engine._adaptive_chunk_pinned_after_retry is True
    assert result["path_stats"]["memory_prefill_retries"] == 1


def test_memory_retry_continues_ladder_after_pinned_chunk_also_fails():
    """Real bug live-confirmed 2026-07-29 (EpistemeAI/VibeCoder-20B): the
    first fix for the expert-fetch crash disabled rc.adaptive_chunk_size
    directly to pin a hard ceiling after the first retry -- but
    _memory_prefill_retry_applies() also reads that same field, so a
    SECOND real MemoryError (at the now-fixed, non-adaptive chunk size)
    silently lost retry eligibility and re-raised instead of continuing
    down the ladder (64->32->8->1). Verifies the ladder now continues
    through multiple real failures regardless of the pin."""
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="gpt_oss")
    engine._hot_kv_persist = None
    engine.rc = SimpleNamespace(
        hot_prompt_kv=False, quant_bits=0, adaptive_chunk_size=True,
        prefill_chunk_size=64, hot_prompt_kv_chunk_size=64)
    engine._hybrid_retry_chunk_ceiling = 0
    attempts = []

    def generate(*_args, **_kwargs):
        attempts.append(engine.rc.prefill_chunk_size)
        engine._generation_sampled_tokens = 0
        if len(attempts) < 3:
            raise MemoryError("synthetic expert-fetch reservation refusal")
        return {
            "prefill_s": 1.0, "first_token_s": 1.5, "total_s": 2.0,
            "path_stats": {},
        }

    engine.generate = generate
    engine.discard_failed_request_state = lambda: None
    with (patch("runtime.engine.mx.clear_cache"),
          patch("runtime.engine.mx.reset_peak_memory")):
        result = StreamingEngine.generate_with_memory_retry(engine, "prompt")

    assert attempts == [64, 32, 8], (
        "retry ladder stopped early after the pin took effect -- a second "
        f"real failure lost retry coverage: {attempts}")
    assert result["path_stats"]["memory_prefill_retries"] == 2
