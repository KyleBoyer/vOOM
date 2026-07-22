"""One-entry in-memory prompt-KV cache regressions.

Uses the tiny local GLM fixture; no NAS or production weights are needed.

  .venv/bin/python tests/test_hot_prompt_kv.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"
COMMON = "Hi there, how are you today my friend\n"
FIRST = COMMON + "First branch."
SECOND = COMMON + "Second branch."
# On the external volume deliberately, not tempfile/tmpdir (internal disk) --
# this project's own standing rule for any scratch/checkpoint path, even a
# throwaway test artifact this tiny.
PERSIST_DIR = Path(__file__).resolve().parent / "_hot_kv_persist_scratch"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current

    if not is_current(FIXTURE):
        build(FIXTURE)


def _checkpoint_for_leaf(leaf: str) -> bool:
    """v3 checkpoint filenames identify immutable generations, not leaves."""
    for path in PERSIST_DIR.glob("*.ckpt.json"):
        try:
            if json.loads(path.read_text()).get("leaf") == leaf:
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def _config():
    from runtime.engine import RuntimeConfig

    return RuntimeConfig(
        max_weight_cache_mb=200,
        pin_lm_head=True,
        mla_compressed_kv=True,
        prefill_chunk_size=4,
        hot_prompt_kv=True,
        hot_prompt_kv_chunk_size=4,
        governor=False,
    )


def test_aligned_hot_prefix_matches_fresh_cold_engine():
    """A divergent second prompt reuses only the aligned common prefix and
    must emit exactly the tokens produced by a fresh engine from position 0."""
    _ensure_fixture()
    from runtime.engine import StreamingEngine

    warm_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        warm_engine.generate(FIRST, 3)
        warm = warm_engine.generate(SECOND, 4)
    finally:
        warm_engine.close()

    cold_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        cold = cold_engine.generate(SECOND, 4)
    finally:
        cold_engine.close()

    stats = warm["path_stats"]
    prefix = stats["prompt_cache_prefix_tokens"]
    assert stats["prompt_cache_source"] == "memory"
    assert stats["hot_prompt_lcp_tokens"] > prefix > 0
    assert prefix % 4 == 0, f"hot prefix was not chunk-aligned: {prefix}"
    assert prefix < warm["prompt_tokens"], "branch path must leave a token for endpoint logits"
    assert warm["tokens"] == cold["tokens"], (
        f"hot/cold greedy tokens diverged: {warm['tokens']} != {cold['tokens']}"
    )


def test_exact_hot_endpoint_reuses_retained_logits():
    """Prompt-end logits are retained separately from the extended postgen KV,
    so a normal multi-token request repeat is still a zero-prefill hit."""
    _ensure_fixture()
    from runtime.engine import StreamingEngine

    hot_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        hot_engine.generate(FIRST, 4)
        hot = hot_engine.generate(FIRST, 3)
    finally:
        hot_engine.close()

    cold_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        cold = cold_engine.generate(FIRST, 3)
    finally:
        cold_engine.close()

    stats = hot["path_stats"]
    assert stats["prompt_cache_source"] == "memory"
    assert stats["prompt_cache_exact_hit"] == 1
    assert stats["prompt_cache_prefix_tokens"] == hot["prompt_tokens"]
    assert hot["tokens"] == cold["tokens"]


def test_strict_next_turn_extension_reuses_full_postgeneration_endpoint():
    """A normal next turn starts with every retained post-generation token.

    That exact endpoint is safe even when it is not chunk aligned; only an
    arbitrary divergent branch must fall back to the aligned watermark.
    """
    _ensure_fixture()
    from runtime.engine import StreamingEngine
    from runtime.server import PreparedPrompt

    warm_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        warm_engine.generate(FIRST, 4)
        retained = list(warm_engine._hot_prompt_slots[-1].tokens)
        aligned_watermark = warm_engine._hot_prompt_slots[-1].reusable_prefix
        extension = retained + warm_engine.tokenizer.encode(
            "\nTool result accepted. Continue the next turn.").ids
        warm = warm_engine.generate(
            PreparedPrompt("strict extension", extension), 3)
    finally:
        warm_engine.close()

    cold_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        cold = cold_engine.generate(
            PreparedPrompt("strict extension cold", extension), 3)
    finally:
        cold_engine.close()

    stats = warm["path_stats"]
    assert stats["prompt_cache_source"] == "memory"
    assert stats["hot_prompt_lcp_tokens"] == len(retained)
    assert stats["prompt_cache_prefix_tokens"] == len(retained)
    assert len(retained) > aligned_watermark
    assert stats["hot_prompt_reusable_prefix_tokens"] == aligned_watermark
    assert warm["tokens"] == cold["tokens"]


def test_persistent_match_scores_strict_extension_at_full_endpoint():
    from runtime.hot_kv_persist import _score_match

    candidate = tuple(range(11))
    new = candidate + (99, 100)
    scored = _score_match(
        new, candidate, cand_prompt_length=8,
        cand_reusable_prefix=8, cand_chain_len=3, chunk_size=4)

    assert scored == ("extension", 11, 8, 3, 11)


def test_disk_fallback_reuses_full_strict_extension_endpoint():
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.server import PreparedPrompt

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    cfg = RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
        hot_prompt_kv_slots=1, hot_prompt_kv_persist_dir=str(PERSIST_DIR),
        hot_prompt_kv_persist_max_checkpoints=8, governor=False,
    )
    try:
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            engine.generate(FIRST, 4)
            retained = list(engine._hot_prompt_slots[-1].tokens)
            extension = retained + engine.tokenizer.encode(
                "\nPersisted tool result; continue.").ids
            engine.generate("Unrelated request evicts memory.", 2)
            warm = engine.generate(
                PreparedPrompt("persisted strict extension", extension), 3)
        finally:
            engine.close()

        cold_engine = StreamingEngine(str(FIXTURE), _config())
        try:
            cold = cold_engine.generate(
                PreparedPrompt("cold strict extension", extension), 3)
        finally:
            cold_engine.close()

        stats = warm["path_stats"]
        assert stats["prompt_cache_source"] == "hot_disk"
        assert stats["hot_prompt_kv_disk_hit"] == 1
        assert stats["hot_prompt_lcp_tokens"] == len(retained)
        assert stats["prompt_cache_prefix_tokens"] == len(retained)
        assert warm["tokens"] == cold["tokens"]
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_prepared_prompt_uses_already_validated_token_ids():
    """The server's context-validation encode is the engine's sole encode.

    Give the carrier deliberately unrelated display text: matching the normal
    FIRST result proves generation consumed its attached IDs, not a second
    tokenizer pass over the string value.
    """
    _ensure_fixture()
    from runtime.engine import StreamingEngine
    from runtime.server import PreparedPrompt

    prepared_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        ids = prepared_engine.tokenizer.encode(FIRST).ids
        prepared = prepared_engine.generate(
            PreparedPrompt("this text must not be tokenized", ids), 3)
    finally:
        prepared_engine.close()

    ordinary_engine = StreamingEngine(str(FIXTURE), _config())
    try:
        ordinary = ordinary_engine.generate(FIRST, 3)
    finally:
        ordinary_engine.close()

    assert prepared["prompt_tokens"] == ordinary["prompt_tokens"] == len(ids)
    assert prepared["tokens"] == ordinary["tokens"]


def test_decode_tokens_do_not_advance_branchable_prefill_watermark():
    """A generated token may cross a numeric chunk boundary, but it was
    computed with the one-token decode shape and is not branchable prefill."""
    _ensure_fixture()
    from runtime.engine import StreamingEngine

    engine = StreamingEngine(str(FIXTURE), _config())
    try:
        result = engine.generate(FIRST, 8)
        watermark = engine._hot_prompt_slots[-1].reusable_prefix
        assert watermark % 4 == 0
        assert watermark <= result["prompt_tokens"]
        assert watermark <= (result["prompt_tokens"] // 4) * 4
    finally:
        engine.close()


def test_lru_multiple_slots_survive_an_interleaved_unrelated_request():
    """Reproduces a real incident found live against a real external harness
    (2026-07-14): a title-generation-style request between two turns of the
    same conversation evicted the single retained slot before the SECOND
    turn could ever reuse it, so it also paid a full cold prefill (26,907
    tokens, twice). With hot_prompt_kv_slots=1 (the default, preserving
    original behavior exactly) this is confirmed here to still happen --
    that assertion documents the pre-fix bug, it is not itself a regression.
    With hot_prompt_kv_slots=2, the main conversation's slot survives the
    interleaved side request intact and the second turn is a real hit."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    def run(slots: int):
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_slots=slots, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            engine.generate(FIRST, 3)                       # main conversation, turn 1
            engine.generate("Unrelated side request.", 2)    # interleaved, e.g. title-gen
            second = engine.generate(FIRST, 3)                # main conversation, turn 2 (repeat)
        finally:
            engine.close()
        return second["path_stats"]

    miss_stats = run(slots=1)
    assert miss_stats["prompt_cache_source"] != "memory", (
        "with only 1 slot, the interleaved request should still evict the "
        "main conversation's state -- this documents the ORIGINAL bug, not "
        "a regression: hot_prompt_kv_slots=1 preserves the original "
        "single-slot behavior exactly"
    )

    hit_stats = run(slots=2)
    assert hit_stats["prompt_cache_source"] == "memory"
    assert hit_stats["prompt_cache_exact_hit"] == 1


def test_min_tokens_gate_prevents_tiny_side_requests_from_evicting():
    """Real harness traffic (2026-07-15, kai-desktop) showed a VARIABLE number
    of tiny non-conversational calls (title generation, working-memory
    updates: 89 and 885 tokens, tools=0) between real conversation turns
    (26,872-27,047 tokens, tools=131) -- one interleaved call between one
    pair of turns, two between the next. hot_prompt_kv_slots=2 covered the
    first case and missed the second: no fixed slot count is safe against a
    harness that can always send one more interleaved call than the count
    assumes. hot_prompt_kv_min_tokens refuses to ever RETAIN a slot for a
    prompt below the threshold, so tiny side requests can't evict anything
    no matter how many show up. Proven here with the smallest possible LRU
    (hot_prompt_kv_slots=1) and TWO interleaved small requests."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    probe_cfg = RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
        hot_prompt_kv_slots=1, governor=False,
    )
    probe = StreamingEngine(str(FIXTURE), probe_cfg)
    try:
        small_tokens = probe.generate("Unrelated side request.", 2)["prompt_tokens"]
        main_tokens = probe.generate(FIRST, 3)["prompt_tokens"]
    finally:
        probe.close()
    assert main_tokens > small_tokens, (
        "fixture assumption broken: the main conversation prompt must be "
        "longer than the side-request prompt for this test to mean anything"
    )

    cfg = RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
        hot_prompt_kv_slots=1, hot_prompt_kv_min_tokens=small_tokens + 1,
        governor=False,
    )
    engine = StreamingEngine(str(FIXTURE), cfg)
    try:
        engine.generate(FIRST, 3)                        # main conversation, turn 1
        engine.generate("Unrelated side request.", 2)     # interleaved side call 1
        engine.generate("Another unrelated call!", 2)     # interleaved side call 2
        second = engine.generate(FIRST, 3)                 # main conversation, turn 2
    finally:
        engine.close()

    stats = second["path_stats"]
    assert stats["prompt_cache_source"] == "memory", (
        "small interleaved requests below hot_prompt_kv_min_tokens must never "
        "occupy the single slot, so the main conversation's own turn-1 state "
        "must still be there for turn 2 to hit"
    )
    assert stats["prompt_cache_exact_hit"] == 1


def test_hot_prompt_kv_persists_across_engine_restart():
    """The in-memory LRU is pure in-memory by default and does not survive a
    restart. With hot_prompt_kv_persist_dir set, a slot written by one engine
    instance must be reloaded by a FRESH instance (simulating a server
    restart) and still produce a real exact hit -- proving the disk round
    trip preserves KV, logits, prompt_logits, prompt_length and the
    reusable-prefix watermark well enough for the existing lookup path to
    use it exactly as if it had never left memory."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_persist_dir=str(PERSIST_DIR), governor=False,
        )
        first_engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            first_engine.generate(FIRST, 4)
        finally:
            first_engine.close()

        assert list(PERSIST_DIR.glob("*.ckpt.json")), (
            "expected a persisted checkpoint after the first engine's turn"
        )

        # Fresh instance, same persist dir -- simulates a server restart.
        second_engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            assert second_engine._hot_prompt_slots, (
                "a fresh engine must reload the persisted slot before its "
                "first request, not start with an empty LRU"
            )
            hot = second_engine.generate(FIRST, 3)
        finally:
            second_engine.close()

        cold_engine = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            governor=False,
        ))
        try:
            cold = cold_engine.generate(FIRST, 3)
        finally:
            cold_engine.close()

        stats = hot["path_stats"]
        assert stats["prompt_cache_source"] == "memory", (
            "a restarted engine should hit the RELOADED slot, not recompute cold"
        )
        assert stats["prompt_cache_exact_hit"] == 1
        assert hot["tokens"] == cold["tokens"]
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_checkpoint_retention_is_recency_bounded_not_lru_bounded():
    """Disk checkpoint retention is deliberately DECOUPLED from the
    in-memory LRU capacity (hot_prompt_kv_slots): it has its own, separate
    budget (hot_prompt_kv_persist_max_checkpoints), evicting the OLDEST-by-
    mtime checkpoint once that budget is exceeded -- not "whatever just left
    memory." With the budget set to 1, a second conversation turn must still
    leave exactly 1 checkpoint (the newest), same end state as the old
    delete-on-consume design would have produced, but for a different
    reason (recency, not LRU departure)."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_slots=1, hot_prompt_kv_persist_dir=str(PERSIST_DIR),
            hot_prompt_kv_persist_max_checkpoints=1, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            engine.generate(FIRST, 3)
            assert len(list(PERSIST_DIR.glob("*.ckpt.json"))) == 1

            engine.generate("Unrelated side request.", 2)  # evicts in-memory
            # capacity=1, but disk retention is a separate budget -- this
            # alone must not be why a checkpoint count is enforced
            assert len(list(PERSIST_DIR.glob("*.ckpt.json"))) == 1, (
                "checkpoint count must stay at the configured budget (1), "
                "enforced by gc()'s own recency policy"
            )
        finally:
            engine.close()
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_forking_keeps_a_consumed_checkpoint_retrievable():
    """The whole point of the segment-DAG redesign: consuming a checkpoint
    in memory (a new continuation matches and pops it) must NOT delete that
    checkpoint from disk when there's budget headroom, or a LATER, DIFFERENT
    continuation from that same earlier point (a fork -- "regenerate," an
    edited earlier message) could never find it again. Real content-
    addressed sharing is also proven here: reusing FIRST's own shared
    ancestor segments while forking to SECOND must not duplicate their
    bytes -- checked by asserting no shared segment's mtime changes."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_slots=1, hot_prompt_kv_persist_dir=str(PERSIST_DIR),
            hot_prompt_kv_persist_max_checkpoints=8, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            engine.generate(FIRST, 3)
            leaf_a = engine._hot_prompt_slots[-1].segment_chain[-1]
            segments_after_a = {
                p.name: p.stat().st_mtime for p in PERSIST_DIR.glob("*.seg.json")
            }
            assert segments_after_a, "expected at least one persisted segment"

            # SECOND shares FIRST's COMMON prefix then diverges -- a fork
            # point, consuming FIRST's in-memory slot via a "branch" match.
            engine.generate(SECOND, 4)
            leaf_b = engine._hot_prompt_slots[-1].segment_chain[-1]
            assert leaf_b != leaf_a

            assert _checkpoint_for_leaf(leaf_a), (
                "FIRST's checkpoint must survive being consumed by SECOND's "
                "fork -- deleting it here would make FIRST's own endpoint "
                "unreachable for any future third branch from that point"
            )
            assert _checkpoint_for_leaf(leaf_b)

            segments_after_b = {
                p.name: p.stat().st_mtime for p in PERSIST_DIR.glob("*.seg.json")
            }
            for name, mtime in segments_after_a.items():
                assert segments_after_b.get(name) == mtime, (
                    f"shared ancestor segment {name} was rewritten instead of "
                    "reused -- true incremental append requires the common "
                    "prefix's bytes to be written exactly once, ever"
                )
            assert len(segments_after_b) > len(segments_after_a), (
                "SECOND's own diverging suffix must add at least one new segment"
            )
        finally:
            engine.close()
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_repeat_case_forks_independent_generations_off_shared_prompt():
    """The scenario that motivated this fix: N agentic/cron tasks that all
    start from the IDENTICAL preamble/prompt, each generating its own
    independent continuation. Each such request hits the "repeat" match
    case (same tokens as a prior request's PROMPT, before that prior
    request's own generation). Proves save() now splits the prompt-tail
    from generation into separate segments, so a second task's own
    generation forks off the SAME prompt-tail parent as the first task's,
    without rewriting any shared byte, and both tasks' own checkpoints
    remain independently resumable."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_slots=1, hot_prompt_kv_persist_dir=str(PERSIST_DIR),
            hot_prompt_kv_persist_max_checkpoints=8, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            # Task 1: the shared preamble, generating its own output.
            task1 = engine.generate(FIRST, 4)
            leaf_task1 = engine._hot_prompt_slots[-1].segment_chain[-1]
            segments_after_task1 = {
                p.name: p.stat().st_mtime for p in PERSIST_DIR.glob("*.seg.json")
            }
            assert segments_after_task1

            # Task 2: the IDENTICAL preamble/prompt, its own generation.
            # This must hit the "repeat" arm, not "branch" or "endpoint".
            task2 = engine.generate(FIRST, 3)
            stats2 = task2["path_stats"]
            assert stats2["prompt_cache_source"] == "memory"
            assert stats2["prompt_cache_exact_hit"] == 1
            assert stats2["prompt_cache_prefix_tokens"] == task1["prompt_tokens"], (
                "must match at exactly the shared prompt length -- the "
                "'repeat' case, not a shorter chunk-floored branch"
            )
            leaf_task2 = engine._hot_prompt_slots[-1].segment_chain[-1]

            assert leaf_task2 != leaf_task1, (
                "two independent continuations of the same prompt must fork "
                "to two DIFFERENT leaves, not collapse into one"
            )
            assert _checkpoint_for_leaf(leaf_task1), (
                "task 1's own checkpoint must remain independently resumable "
                "after task 2 forks off the same shared prompt"
            )
            assert _checkpoint_for_leaf(leaf_task2)

            segments_after_task2 = {
                p.name: p.stat().st_mtime for p in PERSIST_DIR.glob("*.seg.json")
            }
            # Every one of task 1's segments -- shared parents AND its own
            # generation leaf -- must be completely untouched by task 2's
            # save: task 2's parent chain never includes task 1's own
            # generation segment, so nothing about task 2 should write to it.
            for name, mtime in segments_after_task1.items():
                assert segments_after_task2.get(name) == mtime, (
                    f"shared prompt segment {name} was rewritten for task 2 "
                    "instead of being forked from -- this is exactly the "
                    "cron/agentic-fleet sharing the fix is meant to give"
                )
            assert len(segments_after_task2) > len(segments_after_task1), (
                "task 2's own generation must add at least one new segment"
            )
        finally:
            engine.close()
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_disk_fallback_recovers_a_task_evicted_from_the_in_memory_lru():
    """The gap explicitly flagged as not-yet-closed: with more concurrent
    tasks sharing one preamble than fit in hot_prompt_kv_slots, an earlier
    task's shared prefix can still be sitting on disk even after being
    evicted from memory. Proves the disk-side find_best_match()/
    load_matched_chain() fallback recovers it: task 1 runs, gets evicted
    from a slots=1 LRU by an unrelated request, and a LATER repeat of
    task 1's own prompt must still hit -- via disk, not memory -- instead
    of silently recomputing cold."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
            hot_prompt_kv_slots=1, hot_prompt_kv_persist_dir=str(PERSIST_DIR),
            hot_prompt_kv_persist_max_checkpoints=8, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            task1 = engine.generate(FIRST, 3)  # task 1: the shared preamble

            # An unrelated request evicts task 1 from the slots=1 in-memory
            # LRU -- task 1's checkpoint remains on disk (retention is
            # decoupled from in-memory capacity), but memory now has
            # nothing that matches FIRST at all.
            engine.generate("Unrelated side request.", 2)
            assert not any(
                s.tokens[: len(engine.tokenizer.encode(FIRST).ids)]
                == tuple(engine.tokenizer.encode(FIRST).ids)
                for s in engine._hot_prompt_slots
            ), "test setup assumption broken: task 1 should be gone from memory"

            # A later repeat of task 1's OWN prompt: must recover via disk,
            # not recompute cold, and not silently miss.
            task3 = engine.generate(FIRST, 4)
            stats3 = task3["path_stats"]
            assert stats3["prompt_cache_source"] == "hot_disk", (
                f"expected a disk recovery, got {stats3['prompt_cache_source']!r} "
                "-- the shared preamble should still be on disk even though "
                "task 1 was evicted from memory"
            )
            assert stats3["hot_prompt_kv_disk_hit"] == 1
            assert stats3["prompt_cache_exact_hit"] == 1
            assert stats3["prompt_cache_prefix_tokens"] == task1["prompt_tokens"]
        finally:
            engine.close()
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_phase_aware_single_slot_protects_gateway_execution_state():
    from types import SimpleNamespace

    from runtime.engine import _HotPromptSlot, StreamingEngine

    class State:
        def __init__(self, size):
            self.size = size
            self.releases = 0

        def allocated_nbytes(self):
            return self.size

        def release(self):
            self.releases += 1

    def slot(namespace, state):
        return _HotPromptSlot(
            tokens=(1,), kv=state, logits=None, prompt_length=1,
            prompt_logits=None, reusable_prefix=0, chunk_size=32,
            segment_chain=(f"{namespace}-persisted",),
            cache_namespace=namespace)

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(hot_prompt_kv_slots=1)

    execution_state = State(1_500_000_000)
    decision_state = State(300_000_000)
    execution = slot("gateway_execution", execution_state)
    decision = slot("gateway_decision", decision_state)
    engine._hot_prompt_slots = [execution]
    count, _bytes = engine._append_hot_prompt_slot(decision)
    assert count == 1
    assert engine._hot_prompt_slots == [execution]
    assert execution_state.releases == 0

    engine._hot_prompt_slots = [decision]
    count, _bytes = engine._append_hot_prompt_slot(execution)
    assert count == 1
    assert engine._hot_prompt_slots == [execution]
    assert decision_state.releases == 1


def test_identical_tokens_do_not_cross_reuse_hidden_phase_namespaces():
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.server import PreparedPrompt

    cfg = RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        prefill_chunk_size=4, hot_prompt_kv=True, hot_prompt_kv_chunk_size=4,
        hot_prompt_kv_slots=2, hot_prompt_kv_min_tokens=0, governor=False,
    )
    engine = StreamingEngine(str(FIXTURE), cfg)
    try:
        ids = engine.tokenizer.encode(FIRST).ids
        decision_prompt = PreparedPrompt(
            FIRST, ids, cache_namespace="gateway_decision")
        execution_prompt = PreparedPrompt(
            FIRST, ids, cache_namespace="gateway_execution")
        first = engine.generate(decision_prompt, 3)
        other_phase = engine.generate(execution_prompt, 3)
        same_phase = engine.generate(decision_prompt, 3)
    finally:
        engine.close()

    assert first["path_stats"]["prompt_cache_source"] == "cold"
    assert other_phase["path_stats"]["prompt_cache_source"] == "cold"
    assert same_phase["path_stats"]["prompt_cache_source"] == "memory"
    assert same_phase["path_stats"]["prompt_cache_exact_hit"] == 1


def test_single_memory_slot_keeps_both_hidden_phases_durable_on_disk():
    """The one-slot RAM policy is independent from the durable phase tier."""
    _ensure_fixture()
    import shutil

    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.server import PreparedPrompt

    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    try:
        cfg = RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prefill_chunk_size=4, hot_prompt_kv=True,
            hot_prompt_kv_chunk_size=4, hot_prompt_kv_slots=1,
            hot_prompt_kv_min_tokens=0,
            hot_prompt_kv_persist_dir=str(PERSIST_DIR),
            hot_prompt_kv_persist_max_checkpoints=8, governor=False,
        )
        engine = StreamingEngine(str(FIXTURE), cfg)
        try:
            decision_ids = engine.tokenizer.encode(FIRST).ids
            execution_ids = engine.tokenizer.encode(SECOND).ids
            decision_prompt = PreparedPrompt(
                FIRST, decision_ids, cache_namespace="gateway_decision")
            execution_prompt = PreparedPrompt(
                SECOND, execution_ids, cache_namespace="gateway_execution")
            engine.generate(decision_prompt, 3)
            engine.generate(execution_prompt, 3)

            manifests = [
                json.loads(path.read_text())
                for path in PERSIST_DIR.glob("*.ckpt.json")]
            assert {value["cache_namespace"] for value in manifests} == {
                "gateway_decision", "gateway_execution"}
            assert len(engine._hot_prompt_slots) == 1
            assert engine._hot_prompt_slots[0].cache_namespace == \
                "gateway_execution"

            # The transient decision state was evicted from RAM but remains an
            # exact, namespace-scoped disk checkpoint.
            recovered = engine.generate(decision_prompt, 3)
            assert recovered["path_stats"]["prompt_cache_source"] == "hot_disk"
            assert recovered["path_stats"]["prompt_cache_exact_hit"] == 1
        finally:
            engine.close()
    finally:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


def test_memory_admission_evicts_persisted_unmatched_phase_before_new_kv():
    from types import SimpleNamespace
    from unittest.mock import patch

    from runtime.engine import _HotPromptSlot, StreamingEngine

    active = [7_830_000_000]

    class State:
        def __init__(self, size):
            self.size = size
            self.releases = 0

        def allocated_nbytes(self):
            return self.size

        def release(self):
            self.releases += 1
            active[0] -= self.size

    retained = State(1_680_000_000)
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.governor = SimpleNamespace(
        current_ceiling=lambda: 9_050_000_000)
    engine._hot_prompt_slots = [_HotPromptSlot(
        tokens=(1,), kv=retained, logits=None, prompt_length=1,
        prompt_logits=None, reusable_prefix=0, chunk_size=32,
        segment_chain=("durable",),
        cache_namespace="gateway_execution")]

    with patch("runtime.engine.mx.get_active_memory",
               side_effect=lambda: active[0]), \
         patch("runtime.engine.mx.clear_cache"):
        stats = engine._evict_hot_slots_for_admission(
            2_145_000_000, None, "gateway_decision")

    assert stats["evicted_slots"] == 1
    assert stats["evicted_persisted_slots"] == 1
    assert stats["evicted_bytes"] == 1_680_000_000
    assert retained.releases == 1
    assert engine._hot_prompt_slots == []
    assert active[0] + 2_145_000_000 + 400_000_000 < 9_050_000_000


def test_memory_admission_preserves_live_system_available_floor():
    from types import SimpleNamespace
    from unittest.mock import patch

    from runtime.engine import _HotPromptSlot, StreamingEngine

    available = [4_800_000_000]
    reserve_calls = []

    class State:
        def allocated_nbytes(self):
            return 1_700_000_000

        def release(self):
            available[0] = 6_600_000_000

    def reserve(incoming, margin):
        reserve_calls.append((incoming, margin))

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(hot_prompt_kv_min_available_mb=4000)
    engine._resident_fast_layers = ("stale-cache-view",)
    engine._resident_fast_evictions = 7
    engine.governor = SimpleNamespace(
        current_ceiling=lambda: 12_000_000_000,
        critical=1_200_000_000, reservations=0, reserve=reserve)
    retained = State()
    engine._hot_prompt_slots = [_HotPromptSlot(
        tokens=(1,), kv=retained, logits=None, prompt_length=1,
        prompt_logits=None, reusable_prefix=0, chunk_size=32,
        segment_chain=("durable",), cache_namespace="gateway_execution")]

    with patch("runtime.engine.mx.get_active_memory", return_value=2_000_000_000), \
         patch("runtime.engine.mx.clear_cache"), \
         patch("runtime.engine.psutil.virtual_memory",
               side_effect=lambda: SimpleNamespace(available=available[0])):
        stats = engine._evict_hot_slots_for_admission(
            1_500_000_000, None, "gateway_decision")

    assert stats["evicted_slots"] == 1
    assert stats["evicted_persisted_slots"] == 1
    assert stats["system_available_floor_bytes"] == 4_000_000_000
    assert stats["system_available_bytes"] == 6_600_000_000
    assert reserve_calls == [(1_500_000_000, 2_800_000_000)]
    assert engine._resident_fast_layers is None
    assert engine._resident_fast_evictions == -1


def test_pic_duplicate_allocation_respects_same_system_floor():
    from types import SimpleNamespace
    from unittest.mock import patch

    from runtime.engine import _system_allocation_preserves_floor

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=5_500_000_000)):
        admitted, available, floor = _system_allocation_preserves_floor(
            1_730_000_000, 4000)
    assert not admitted
    assert available == 5_500_000_000
    assert floor == 4_000_000_000

    with patch("runtime.engine.psutil.virtual_memory",
               return_value=SimpleNamespace(available=6_000_000_000)):
        admitted, _available, _floor = _system_allocation_preserves_floor(
            1_730_000_000, 4000)
    assert admitted


def test_runtime_quantized_resident_qwen_defers_restart_kv_until_bootstrap():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="qwen3", vision_config=None, num_experts=0)
    engine.rc = SimpleNamespace(quant_bits=4, resident_fast_decode=True)
    engine.store = SimpleNamespace(on_disk_quantized=False)
    engine._defer_persisted_kv_until_bootstrap = (
        engine._should_defer_persisted_kv_until_bootstrap())
    engine._completed_generations = 0

    assert engine._defer_persisted_kv_until_bootstrap
    assert not engine._persisted_kv_restore_allowed()
    engine._completed_generations = 1
    assert engine._persisted_kv_restore_allowed()

    # A genuinely pre-quantized checkpoint has no lazy BF16->Q4 bootstrap
    # collision, so its restart KV remains eligible immediately.
    engine.store.on_disk_quantized = True
    assert not engine._should_defer_persisted_kv_until_bootstrap()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
