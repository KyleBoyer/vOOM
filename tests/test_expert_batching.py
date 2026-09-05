"""F74-v2: regression test for runtime/expert_batching.py's lifetime
guarantee -- consume_expert_batches() must release batch N's mapping
before requesting batch N+1 from the producer. The module's own docstring
flags a subtle Python gotcha (a naive `for x, y in batches:` loop calls
`next()` before rebinding the loop targets, so two batches' mappings
coexist during the fetch of the next one) -- this test exercises the real
exported function against that exact failure mode, not just the manual
iterator pattern in isolation.
"""
import concurrent.futures as cf
import gc
import sys
import threading
import weakref
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.expert_batching import consume_expert_batches


def test_exact_expert_batch_pipeline_remains_explicit_opt_in():
    from runtime.engine import RuntimeConfig

    assert not RuntimeConfig().expert_batch_prefetch
    assert not RuntimeConfig().glm53_expert_batch_prefetch_prefill_only
    assert not RuntimeConfig().glm53_serial_verify_coalesced_barriers
    assert not RuntimeConfig().expert_route_overlap_telemetry


def test_route_overlap_summary_separates_union_and_cross_sweep_reuse():
    from runtime.engine import expert_route_overlap_summary

    # Positions 0/1 route {1,2}/{2,3}; the previous sweep ended on {0,2}.
    summary, last = expert_route_overlap_summary(
        {
            1: [0],
            2: [0, 1],
            3: [1],
        },
        previous_route=(0, 2),
    )

    assert last == (2, 3)
    assert summary == {
        "calls": 1,
        "positions": 2,
        "selected_slots": 4,
        "union_experts": 3,
        "adjacent_pairs": 2,
        "within_call_pairs": 1,
        "cross_call_pairs": 1,
        "adjacent_intersection_experts": 2,
        "adjacent_union_experts": 6,
        "adjacent_current_experts": 4,
        "exact_adjacent_pairs": 0,
        "cross_call_intersection_experts": 1,
        "cross_call_current_experts": 2,
    }


def test_route_overlap_summary_without_prior_has_only_within_call_pairs():
    from runtime.engine import expert_route_overlap_summary

    summary, last = expert_route_overlap_summary(
        {4: [0, 1], 5: [0, 1], 6: [1]}
    )

    assert last == (4, 5, 6)
    assert summary["within_call_pairs"] == 1
    assert summary["cross_call_pairs"] == 0
    assert summary["adjacent_intersection_experts"] == 2
    assert summary["adjacent_union_experts"] == 3


def test_serial_verify_route_rows_are_offset_before_provisional_commit():
    from types import SimpleNamespace

    from runtime.engine import (
        StreamingEngine,
        offset_expert_route_positions,
    )

    first = offset_expert_route_positions(
        {4: [0], 7: [0, 0]}, 3
    )
    assert first == {
        4: [3],
        7: [3, 3],
    }
    assert offset_expert_route_positions(None, 9) is None

    engine = SimpleNamespace(
        _provisional=[
            (5, {1: [0]}),
            (5, offset_expert_route_positions({2: [0]}, 2)),
        ],
        expert_usage={},
        predictor=None,
    )
    StreamingEngine.commit_provisional(engine, accepted_positions=2)
    assert engine.expert_usage == {(5, 1): 1}


def test_compact_expert_io_batch_uses_representation_bytes_not_model_identity():
    from runtime.engine import compact_expert_io_batch_size

    # K3's released native MXFP4 expert page is ~17.5 MB: sixteen pages reach
    # the 256 MiB coalescing neighborhood without approaching a 3 GB cache.
    assert compact_expert_io_batch_size(
        17_550_000, 3_000_000_000) == 16
    # A smaller compact expert reaches the same byte target with the explicit
    # page-count ceiling; a BF16-sized expert remains conservative.
    assert compact_expert_io_batch_size(
        6_300_000, 3_000_000_000) == 16
    assert compact_expert_io_batch_size(
        75_500_000, 3_000_000_000) == 4


def test_compact_expert_io_batch_respects_tight_cache_and_validates_inputs():
    import pytest

    from runtime.engine import compact_expert_io_batch_size

    assert compact_expert_io_batch_size(
        17_550_000, 150_000_000) == 2
    assert compact_expert_io_batch_size(
        17_550_000, 3_000_000_000, max_batch=4) == 4
    with pytest.raises(ValueError, match="page_bytes"):
        compact_expert_io_batch_size(0, 3_000_000_000)
    with pytest.raises(ValueError, match="cache_bytes"):
        compact_expert_io_batch_size(1, -1)


def test_decode_only_expert_batch_can_be_larger_than_prefill_batch():
    """A one-position routed union uses the side-quest decode batch, while a
    multi-position prefill retains the conservative lifetime bound."""
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    class FakeEngine:
        rc = SimpleNamespace(expert_fetch_batch=1, decode_expert_fetch_batch=8)
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            return {expert: object() for expert in expert_ids}

    engine = FakeEngine()
    expert_ids = list(range(8))
    decode = list(StreamingEngine._iter_expert_batches(
        engine, 4, expert_ids, positions={expert: [0] for expert in expert_ids}))
    prefill = list(StreamingEngine._iter_expert_batches(
        engine, 4, expert_ids,
        positions={expert: [expert % 2] for expert in expert_ids}))

    assert [len(ids) for ids, _pages in decode] == [8]
    assert [len(ids) for ids, _pages in prefill] == [1] * 8


def test_expert_io_batch_can_exceed_compute_materialization_batch():
    """Storage coalescing must not silently change arithmetic boundaries."""
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    fetches = []

    class FakeEngine:
        rc = SimpleNamespace(
            expert_fetch_batch=8,
            expert_compute_batch=4,
            decode_expert_fetch_batch=0,
        )
        governor = None
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            fetches.append(list(expert_ids))
            return {expert: f"page-{expert}" for expert in expert_ids}

    engine = FakeEngine()
    batches = list(StreamingEngine._iter_expert_batches(
        engine, 4, list(range(10)),
        positions={expert: [expert % 2] for expert in range(10)}))

    assert fetches == [list(range(8)), [8, 9]]
    assert [ids for ids, _pages in batches] == [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert all(
        pages[expert] == f"page-{expert}"
        for ids, pages in batches for expert in ids
    )
    assert engine._expert_compute_batches == 3
    assert engine._max_experts_per_compute_batch == 4


def test_governor_clamp_preserves_expert_compute_batch_alignment():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    class FakeGovernor:
        def admissible_units(
                self, *, unit_bytes, fixed_bytes, max_units, margin):
            return min(6, max_units)

    fetches = []

    class FakeEngine:
        rc = SimpleNamespace(
            expert_fetch_batch=8,
            expert_compute_batch=4,
            decode_expert_fetch_batch=0,
        )
        governor = FakeGovernor()
        _expert_fetch_page_bytes = 100
        _layer_transient = 200
        _layer_transient_margin = 0
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            fetches.append(list(expert_ids))
            return {expert: object() for expert in expert_ids}

    engine = FakeEngine()
    batches = list(StreamingEngine._iter_expert_batches(
        engine, 4, list(range(12)),
        positions={expert: [expert % 2] for expert in range(12)}))

    assert [len(ids) for ids in fetches] == [4, 4, 4]
    assert [len(ids) for ids, _pages in batches] == [4, 4, 4]
    assert engine._adaptive_expert_batch_clamps == 2


def test_governor_clamps_validated_decode_cap_using_live_headroom():
    """Adaptive scheduling may shrink the mode's cap, never grow it."""
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    class FakeGovernor:
        def admissible_units(
                self, *, unit_bytes, fixed_bytes, max_units, margin):
            assert unit_bytes == 100
            assert fixed_bytes == 200
            assert margin == 0
            return min(3, max_units)

    class FakeEngine:
        rc = SimpleNamespace(expert_fetch_batch=1, decode_expert_fetch_batch=8)
        governor = FakeGovernor()
        _expert_page_bytes = 100
        _expert_fetch_page_bytes = 100
        _layer_transient = 200
        _layer_transient_margin = 0
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            return {expert: object() for expert in expert_ids}

    engine = FakeEngine()
    expert_ids = list(range(8))
    decode = list(StreamingEngine._iter_expert_batches(
        engine, 4, expert_ids,
        positions={expert: [0] for expert in expert_ids}))

    assert [len(ids) for ids, _pages in decode] == [3, 3, 2]
    assert engine._adaptive_expert_batch_clamps == 2
    assert engine._min_adaptive_expert_batch == 2


def test_exact_next_expert_batch_fetch_overlaps_current_batch_consumer():
    """The pipeline knows every expert from the authoritative router first.

    While batch zero's consumer is active, batch one's fetch must already be
    running. Expert order and page identity remain unchanged.
    """
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    second_started = threading.Event()
    release_second = threading.Event()
    fetched = []
    consumed = []

    class FakeEngine:
        rc = SimpleNamespace(
            expert_fetch_batch=1,
            decode_expert_fetch_batch=0,
        )
        governor = None
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0
        _expert_batch_prefetch_submitted = 0
        _expert_batch_prefetch_wait_s = 0.0
        _expert_batch_prefetch_hidden_s = 0.0
        _expert_batch_prefetch_max_futures = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            expert = expert_ids[0]
            fetched.append(expert)
            if expert == 1:
                second_started.set()
                assert release_second.wait(timeout=2)
            return {expert: f"page-{expert}"}

    engine = FakeEngine()
    engine._expert_batch_executor = cf.ThreadPoolExecutor(max_workers=1)
    try:
        batches = StreamingEngine._iter_expert_batches(
            engine, 4, [0, 1, 2],
            positions={0: [0], 1: [1], 2: [0]})

        def consume(batch_ids, pages):
            expert = batch_ids[0]
            consumed.append((expert, pages[expert]))
            if expert == 0:
                assert second_started.wait(timeout=2)
                # Model useful current-batch compute while the next fetch is
                # active, making the hidden-time witness deterministic.
                assert not release_second.wait(timeout=0.02)
                release_second.set()

        consume_expert_batches(batches, consume)
    finally:
        release_second.set()
        engine._expert_batch_executor.shutdown(
            wait=True, cancel_futures=True)

    assert fetched == [0, 1, 2]
    assert consumed == [
        (0, "page-0"), (1, "page-1"), (2, "page-2")]
    assert engine._expert_batch_prefetch_submitted == 3
    assert engine._expert_batch_prefetch_hidden_s > 0


def test_exact_expert_batch_prefetch_depth_two_queues_two_ordered_futures():
    import concurrent.futures as cf
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    consumed = []

    class FakeEngine:
        rc = SimpleNamespace(
            expert_fetch_batch=1,
            decode_expert_fetch_batch=0,
            expert_batch_prefetch_depth=2,
        )
        governor = None
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0
        _expert_batch_prefetch_submitted = 0
        _expert_batch_prefetch_wait_s = 0.0
        _expert_batch_prefetch_hidden_s = 0.0
        _expert_batch_prefetch_max_futures = 0

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            return {expert: f"page-{expert}" for expert in expert_ids}

    engine = FakeEngine()
    engine._expert_batch_executor = cf.ThreadPoolExecutor(max_workers=1)
    try:
        batches = StreamingEngine._iter_expert_batches(
            engine, 4, [0, 1, 2],
            positions={0: [0], 1: [1], 2: [2]})
        assert engine._expert_batch_prefetch_submitted == 2
        for batch_ids, pages in batches:
            consumed.extend((expert, pages[expert]) for expert in batch_ids)
    finally:
        engine._expert_batch_executor.shutdown(
            wait=True, cancel_futures=True)

    assert consumed == [(0, "page-0"), (1, "page-1"), (2, "page-2")]
    assert engine._expert_batch_prefetch_submitted == 3
    assert engine._expert_batch_prefetch_max_futures == 2


@pytest.mark.parametrize("family", ["qwen4", "glm53"])
def test_prefill_only_expert_prefetch_disables_worker_for_decode(family):
    import concurrent.futures as cf
    import threading
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    fetch_threads = []

    class FakeEngine:
        rc = SimpleNamespace(
            expert_fetch_batch=1,
            decode_expert_fetch_batch=1,
            expert_batch_prefetch_depth=1,
            glm53_expert_batch_prefetch_prefill_only=family == "glm53",
            qwen4_expert_batch_prefetch_prefill_only=family == "qwen4",
        )
        store = SimpleNamespace(
            glm53_fp8_direct_qmv=False,
            qwen4_fp8_direct_qmv=False,
        )
        governor = None
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _adaptive_expert_batch_clamps = 0
        _min_adaptive_expert_batch = 0
        _expert_batch_prefetch_submitted = 0
        _expert_batch_prefetch_wait_s = 0.0
        _expert_batch_prefetch_hidden_s = 0.0
        _expert_batch_prefetch_max_futures = 0
        _expert_batch_prefetch_phase = "prefill"
        _expert_batch_prefetch_submitted_by_phase = {
            "prefill": 0, "decode": 0,
        }
        _expert_batch_prefetch_wait_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }
        _expert_batch_prefetch_hidden_s_by_phase = {
            "prefill": 0.0, "decode": 0.0,
        }

        def _record_expert_route(self, *_args, **_kwargs):
            pass

        def _fetch_experts(self, _layer, expert_ids):
            fetch_threads.append(threading.current_thread().name)
            return {expert: f"page-{expert}" for expert in expert_ids}

    engine = FakeEngine()
    engine._expert_batch_executor = cf.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="vmodel-expert-batch")
    engine._expert_batch_prefetch_active = True
    try:
        StreamingEngine._set_finegrained_fp8_direct_phase(engine, "prefill")
        list(StreamingEngine._iter_expert_batches(
            engine, 4, [0], positions={0: [0, 1]}))
        StreamingEngine._set_finegrained_fp8_direct_phase(engine, "decode")
        list(StreamingEngine._iter_expert_batches(
            engine, 4, [1], positions={1: [0]}))
    finally:
        engine._expert_batch_executor.shutdown(
            wait=True, cancel_futures=True)

    assert fetch_threads[0].startswith("vmodel-expert-batch")
    assert fetch_threads[1] == threading.current_thread().name
    assert engine._expert_batch_prefetch_submitted_by_phase == {
        "prefill": 1, "decode": 0,
    }


def test_k25_layer_page_estimate_distinguishes_dense_and_sparse_pages():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    engine = SimpleNamespace(cfg=SimpleNamespace(
        model_type="kimi_k25",
        hidden_size=7168,
        num_attention_heads=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        mlp_layer_types=("dense", "sparse"),
        first_k_dense_replace=1,
        intermediate_size=18432,
        num_experts=384,
        moe_intermediate_size=2048,
        n_shared_experts=1,
    ))

    dense = StreamingEngine._layer_fetch_bytes_estimate(engine, 0)
    sparse = StreamingEngine._layer_fetch_bytes_estimate(engine, 1)

    assert 900_000_000 < dense < 1_200_000_000
    assert 250_000_000 < sparse < 400_000_000
    assert dense > sparse * 3


class _WeakrefableDict(dict):
    """Plain dict can't hold a weakref; a bare subclass can (gains
    __weakref__), letting the test observe when the mapping is actually
    collected without changing its dict-like behavior."""


def test_previous_batch_is_released_before_next_batch_is_produced():
    def producer():
        prev_ref = None
        for i in range(4):
            if prev_ref is not None:
                gc.collect()
                assert prev_ref() is None, (
                    f"batch {i - 1}'s mapping was still alive when batch {i} "
                    "was requested -- consume_expert_batches is not releasing "
                    "the previous batch before fetching the next one"
                )
            experts = _WeakrefableDict({f"expert.{i}": list(range(100))})
            prev_ref = weakref.ref(experts)
            yield [i], experts
            del experts  # this generator's own frame must not retain it either

    seen = []

    def consume(batch_ids, experts):
        seen.append(batch_ids[0])

    consume_expert_batches(producer(), consume)
    assert seen == [0, 1, 2, 3]


def test_naive_for_loop_would_have_kept_two_batches_alive():
    """Documents WHY consume_expert_batches can't be a for-loop: a for-loop
    calls next() before rebinding its targets, so the previous iteration's
    values are still referenced (by the loop targets) while the generator
    computes the next one."""
    live_during_next_fetch = []

    def producer():
        prev_experts = None
        for i in range(3):
            if prev_experts is not None:
                live_during_next_fetch.append(prev_experts is not None)
            experts = {f"expert.{i}": i}
            yield [i], experts
            prev_experts = experts  # simulates a for-loop's loop-target retention

    for _batch_ids, _experts in producer():
        pass

    assert live_during_next_fetch == [True, True]


def test_empty_batches_no_error():
    consume_expert_batches(iter(()), lambda *_: None)
