"""F96: retain a stable-boundary checkpoint (not the full post-reply
endpoint) for recurrent_exact_only (qwen3_5/qwen3_5_moe/kimi_linear) hot-KV
slots.

Live-reproduced 2026-07-22: the released Qwen3.5/3.6 chat template
re-renders any but the LATEST assistant turn WITHOUT its own generation
scaffold (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` for a
no-thinking answer) once a further turn follows it. A hot-KV slot retained
at the full post-generation endpoint (the old behavior, still correct for
ordinary attention-KV models) therefore diverges from every real second
turn a few tokens before its own end, and recurrent_exact_only's matching
loop tolerates zero divergence (DeltaNet/KDA state cannot be trimmed to an
arbitrary common prefix) -- so it fell back to a full cold re-prefill on
literally every second turn, with zero exceptions observed live.

``StreamingEngine._new_hot_prompt_slot`` is the small, directly-callable
decision this fix hinges on: whether a completed request's boundary fork
(taken during its own prefill sweep, see the fork site in generate()) or
the ordinary full endpoint gets retained. It is tested here in isolation,
using a bare engine object (no real model/weights needed) -- the same
pattern already used by tests/test_hybrid_dynamic_chunk_size.py for the
sibling F95 feature.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")


def _bare_engine(prefill_chunk_size: int = 512):
    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(prefill_chunk_size=prefill_chunk_size)
    return engine


def test_recurrent_model_with_boundary_fork_retains_boundary_not_full_endpoint():
    engine = _bare_engine(prefill_chunk_size=128)
    boundary_kv = object()
    full_kv = object()
    slot = engine._new_hot_prompt_slot(
        recurrent_exact_only=True,
        boundary_fork_kv=boundary_kv,
        boundary_fork_tokens=7,
        tokens=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        full_tokens=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        kv=full_kv,
        logits="full-endpoint-logits",
        prompt_endpoint_logits="prompt-endpoint-logits",
        reusable_watermark=4,
        prompt_state_approximate=False,
        tool_capsules=(("id", 0, 3),),
        segment_chain=("disk-segment-1",),
        boundary_segment_chain=("stable-segment-1",),
        cache_namespace="default",
    )
    assert slot.tokens == (1, 2, 3, 4, 5, 6, 7)
    assert slot.kv is boundary_kv
    assert slot.logits is None
    assert slot.prompt_logits is None
    assert slot.prompt_length == 7
    assert slot.reusable_prefix == 0
    assert slot.chunk_size == 128
    # PIC capsule bookkeeping still describes the old full endpoint and is
    # dropped. The separately persisted exact boundary lineage is retained.
    assert slot.tool_capsules == ()
    assert slot.segment_chain == ("stable-segment-1",)


def test_recurrent_model_without_boundary_fork_falls_back_to_full_endpoint():
    """No stable-boundary hint degrades to the pre-F96 behavior unchanged."""
    engine = _bare_engine(prefill_chunk_size=256)
    full_kv = object()
    slot = engine._new_hot_prompt_slot(
        recurrent_exact_only=True,
        boundary_fork_kv=None,
        boundary_fork_tokens=0,
        tokens=(1, 2, 3),
        full_tokens=(1, 2, 3, 4),
        kv=full_kv,
        logits="full-endpoint-logits",
        prompt_endpoint_logits="prompt-endpoint-logits",
        reusable_watermark=2,
        prompt_state_approximate=False,
        tool_capsules=(),
        segment_chain=("disk-segment-2",),
        cache_namespace="ns-a",
    )
    assert slot.tokens == (1, 2, 3, 4)
    assert slot.kv is full_kv
    assert slot.logits == "full-endpoint-logits"
    assert slot.prompt_logits == "prompt-endpoint-logits"
    assert slot.prompt_length == 3
    assert slot.reusable_prefix == 2
    assert slot.chunk_size == 256
    assert slot.segment_chain == ("disk-segment-2",)
    assert slot.cache_namespace == "ns-a"


def test_matched_stable_boundary_is_forked_before_suffix_mutates_source():
    from runtime.engine import _fork_matched_hybrid_stable_boundary
    from runtime.kv_cache import KVCache

    class _Recurrent:
        def __init__(self, identity):
            self.identity = identity
            self.synchronized = 0

        def fork(self):
            return _Recurrent(self.identity)

        def synchronize(self):
            self.synchronized += 1

    source = KVCache(2)
    source.kda_cache = _Recurrent("stable-128")

    retained = _fork_matched_hybrid_stable_boundary(
        source,
        matched_tokens=128,
        stable_boundary_tokens=128,
        prompt_tokens=141,
    )

    assert retained is not None and retained is not source
    assert retained.keys == source.keys
    assert retained.values == source.values
    assert retained.kda_cache is not source.kda_cache
    assert retained.kda_cache.identity == "stable-128"
    assert retained.kda_cache.synchronized == 1

    # Only exact equality at a strict proper boundary is eligible. Earlier or
    # later matches still require the ordinary prefill/fork logic.
    for matched, stable, total in (
        (127, 128, 141),
        (129, 128, 141),
        (0, 0, 141),
        (141, 141, 141),
    ):
        assert _fork_matched_hybrid_stable_boundary(
            source,
            matched_tokens=matched,
            stable_boundary_tokens=stable,
            prompt_tokens=total,
        ) is None


def test_mixed_depth_disk_boundary_is_skipped_for_exact_short_prompt():
    from runtime.engine import _stable_boundary_persistence_allowed

    mixed_store = SimpleNamespace(
        requires_approximate_stable_prefix=True)
    ordinary_store = SimpleNamespace()

    assert not _stable_boundary_persistence_allowed(
        mixed_store, approximate=False)
    assert _stable_boundary_persistence_allowed(
        mixed_store, approximate=True)
    assert _stable_boundary_persistence_allowed(
        ordinary_store, approximate=False)
    assert not _stable_boundary_persistence_allowed(
        None, approximate=True)


@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_moe", "qwen4_exp"))
def test_exact_qwen_hybrids_prefer_strictly_longer_persisted_extension(
    model_type,
):
    from runtime.engine import _prefer_longer_persisted_hybrid_prefix

    assert _prefer_longer_persisted_hybrid_prefix(
        model_type=model_type, best_case="extension")
    for best_case in ("repeat", "branch", "endpoint", ""):
        assert not _prefer_longer_persisted_hybrid_prefix(
            model_type=model_type, best_case=best_case)


@pytest.mark.parametrize(
    "model_type", ("kimi_linear", "kimi_k3", "jet_nemotron", "gpt_oss"))
def test_other_models_do_not_replace_resident_state_from_disk(model_type):
    from runtime.engine import _prefer_longer_persisted_hybrid_prefix

    assert not _prefer_longer_persisted_hybrid_prefix(
        model_type=model_type, best_case="extension")


def test_non_recurrent_model_always_uses_full_endpoint_even_with_a_fork():
    """A boundary fork should never even be produced for an ordinary
    attention-KV model (the fork site in generate() is itself gated on
    recurrent_exact_only), but this decision point stays fail-safe even if
    it somehow received one: recurrent_exact_only=False must ignore it."""
    engine = _bare_engine()
    full_kv = object()
    slot = engine._new_hot_prompt_slot(
        recurrent_exact_only=False,
        boundary_fork_kv=object(),
        boundary_fork_tokens=3,
        tokens=(1, 2, 3),
        full_tokens=(1, 2, 3, 4, 5),
        kv=full_kv,
        logits="endpoint-logits",
        prompt_endpoint_logits="prompt-logits",
        reusable_watermark=0,
        prompt_state_approximate=False,
        tool_capsules=(),
        segment_chain=(),
        cache_namespace="default",
    )
    assert slot.tokens == (1, 2, 3, 4, 5)
    assert slot.kv is full_kv
    assert slot.logits == "endpoint-logits"


@pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")
def test_hybrid_stable_boundary_tokens_against_real_chat_template():
    """The actual bug this feature fixes, proven against the REAL released
    template: a genuine growing conversation's second-turn prompt must
    share the computed boundary as a byte-identical token prefix, and that
    boundary must fall strictly before the full first-turn prompt (i.e. it
    excludes at least the generation scaffold).

    Uses a real (lazily-constructed, no weights touched) StreamingEngine
    against the real downloaded checkpoint so this exercises the actual
    shipped chat_template.jinja, not a hand-written stand-in of it.
    """
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.server import _chat_prompt, _hybrid_stable_boundary_tokens, _prepared_prompt_ids

    rc = RuntimeConfig(
        hot_prompt_kv=True, prefill_chunk_size=32, hot_prompt_kv_chunk_size=32)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        turn1_messages = [
            {"role": "user", "content": "Please respond with exactly the word: apple"},
        ]
        prompt = _chat_prompt(
            engine, _REAL_MODEL_DIR, turn1_messages, "low",
            enable_thinking=False)
        prompt_ids, _offsets, _hit = _prepared_prompt_ids(engine, prompt)

        boundary = _hybrid_stable_boundary_tokens(
            engine, _REAL_MODEL_DIR, turn1_messages, "low", None, prompt_ids,
            compact_json=False, enable_thinking=False,
            reasoning_requested=False, canonical_hermes_tools=False)

        assert 0 < boundary < len(prompt_ids), (
            "boundary must be a strict, non-trivial prefix of the full "
            "generation-ready prompt")
        assert tuple(prompt_ids[:boundary]) == tuple(prompt_ids)[:boundary]

        # The actual regression: render what a SECOND turn's prompt looks
        # like once turn 1 has a reply, and confirm the computed boundary
        # is still a byte-identical prefix of it -- this is exactly the
        # guarantee _new_hot_prompt_slot's "extension" reuse depends on.
        turn2_messages = turn1_messages + [
            {"role": "assistant", "content": "apple"},
            {"role": "user", "content": "Now respond with exactly the word: banana"},
        ]
        prompt2 = _chat_prompt(
            engine, _REAL_MODEL_DIR, turn2_messages, "low",
            enable_thinking=False)
        prompt2_ids, _offsets2, _hit2 = _prepared_prompt_ids(engine, prompt2)
        assert tuple(prompt2_ids[:boundary]) == tuple(prompt_ids[:boundary]), (
            "the computed boundary must reappear byte-identically at the "
            "start of a real second turn -- this is the exact divergence "
            "that made the old (full-endpoint) retention never match again"
        )

        # The opt-in serving boundary is earlier still: it excludes the latest
        # mutable user turn while retaining the identical system/tool prefix.
        # Two different first-turn requests must therefore extend one exact
        # recurrent checkpoint rather than matching an exact-request hash.
        shared_system = {"role": "system", "content": "Keep answers brief."}
        request_a = [shared_system, {
            "role": "user", "content": "Count the red objects."}]
        request_b = [shared_system, {
            "role": "user", "content": "Count the blue objects instead."}]
        prompt_a = _chat_prompt(
            engine, _REAL_MODEL_DIR, request_a, "low",
            enable_thinking=False)
        prompt_b = _chat_prompt(
            engine, _REAL_MODEL_DIR, request_b, "low",
            enable_thinking=False)
        ids_a, _offsets_a, _hit_a = _prepared_prompt_ids(engine, prompt_a)
        ids_b, _offsets_b, _hit_b = _prepared_prompt_ids(engine, prompt_b)
        reusable = _hybrid_stable_boundary_tokens(
            engine, _REAL_MODEL_DIR, request_a, "low", None, ids_a,
            compact_json=False, enable_thinking=False,
            reasoning_requested=False, canonical_hermes_tools=False,
            reusable_user_prefix=True)
        assert 0 < reusable < min(len(ids_a), len(ids_b))
        assert tuple(ids_a[:reusable]) == tuple(ids_b[:reusable])
    finally:
        engine.close()
