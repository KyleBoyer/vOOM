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


def _bare_engine(model_type: str, hot_kv_persist, hot_prompt_kv_chunk_size: int = 128):
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type=model_type)
    engine._hot_kv_persist = hot_kv_persist
    engine.rc = SimpleNamespace(hot_prompt_kv_chunk_size=hot_prompt_kv_chunk_size)
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
    with patch("runtime.engine.mx.clear_cache"):
        result = StreamingEngine.generate_with_memory_retry(engine, "prompt")

    assert attempts == [512, 128, 32]
    assert len(discards) == 2
    assert result["path_stats"]["memory_prefill_retries"] == 2
    assert result["path_stats"]["memory_prefill_retry_chunks"] == [128, 32]
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


def test_memory_retry_also_applies_to_lossy_dense_qwen_without_persistence():
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(model_type="qwen2")
    engine._hot_kv_persist = None
    engine.rc = SimpleNamespace(
        hot_prompt_kv=True, quant_bits=4,
        prefill_chunk_size=32, hot_prompt_kv_chunk_size=32)
    assert engine._memory_prefill_retry_applies()

    engine.rc.quant_bits = 0
    assert not engine._memory_prefill_retry_applies()
