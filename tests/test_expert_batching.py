"""F74-v2: regression test for runtime/expert_batching.py's lifetime
guarantee -- consume_expert_batches() must release batch N's mapping
before requesting batch N+1 from the producer. The module's own docstring
flags a subtle Python gotcha (a naive `for x, y in batches:` loop calls
`next()` before rebinding the loop targets, so two batches' mappings
coexist during the fetch of the next one) -- this test exercises the real
exported function against that exact failure mode, not just the manual
iterator pattern in isolation.
"""
import gc
import sys
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.expert_batching import consume_expert_batches


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
