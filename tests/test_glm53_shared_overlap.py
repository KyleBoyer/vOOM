from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import mlx.core as mx
import numpy as np
import pytest

from runtime.engine import StreamingEngine
from runtime.expert_batching import consume_expert_batches
from runtime import glm5_next


@pytest.mark.parametrize("fail", [False, True])
def test_callback_runs_after_fetch_start_before_consume_and_unwinds(fail):
    started, release, finished = (threading.Event() for _ in range(3))
    events = []
    joins = []

    class Engine:
        rc = SimpleNamespace(expert_fetch_batch=1, decode_expert_fetch_batch=0,
                             expert_batch_prefetch_depth=1)
        governor = None
        _expert_compute_batches = 0
        _max_experts_per_compute_batch = 0
        _expert_batch_prefetch_submitted = 0
        _expert_batch_prefetch_wait_s = 0.0
        _expert_batch_prefetch_hidden_s = 0.0
        _expert_batch_prefetch_max_futures = 0

        def _record_expert_route(self, *args, **kwargs):
            pass

        def _fetch_experts(self, layer, ids):
            events.append("fetch-start")
            started.set()
            assert release.wait(2)
            finished.set()
            return {i: "page" for i in ids}

    def overlap():
        assert started.wait(2)
        events.append("shared-eval")
        if fail:
            release.set()
            raise ValueError("shared failed")
        release.set()

    engine = Engine()
    pool = ThreadPoolExecutor(max_workers=1)

    class FutureWitness:
        def __init__(self, future):
            self.future = future

        def cancel(self):
            return self.future.cancel()

        def result(self):
            joins.append(True)
            return self.future.result()

    engine._expert_batch_executor = SimpleNamespace(
        submit=lambda *args: FutureWitness(pool.submit(*args)))
    try:
        batches = StreamingEngine._iter_expert_batches(
            engine, 0, [0], {0: [0]}, on_prefetch_started=overlap)
        # With independent compute, the generator owns every submitted task;
        # an iterator never advanced submits no work at all.
        assert engine._expert_batch_prefetch_submitted == 0
        if fail:
            with pytest.raises(ValueError, match="shared failed"):
                consume_expert_batches(batches, lambda *_: events.append("consume"))
            assert "consume" not in events
            assert joins and finished.is_set()  # producer, not test cleanup, joined
        else:
            consume_expert_batches(batches, lambda *_: events.append("consume"))
            assert events == ["fetch-start", "shared-eval", "consume"]
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("tiled,widths", [
    (False, (1,)), (False, (2,)), (True, (1,)),
    (True, (1, 1, 1, 1)), (True, (2,)), (True, (1,) * 7),
])
def test_shared_overlap_preserves_outputs_and_rejects_wide_shapes(
        monkeypatch, widths, tiled):
    tiles = [mx.full((1, width, 4), 0.75, dtype=mx.bfloat16) for width in widths]
    cfg = SimpleNamespace(mlp_layer_types=("moe",), first_k_dense_replace=0)
    calls = []

    def route(h, *_):
        return mx.zeros((1, h.shape[1], 1), dtype=mx.int32), mx.ones((1, h.shape[1], 1))

    def swiglu(h, w, prefix, cfg):
        calls.append("shared" if "shared_experts" in prefix else "routed")
        return (h * (0.3 if "shared_experts" in prefix else 0.7)).astype(mx.bfloat16)

    def batches(layer, ids, *, positions, on_prefetch_started=None):
        calls.append("submit")
        if on_prefetch_started is not None:
            on_prefetch_started()
        calls.append("wait")
        yield ids, {i: {} for i in ids}

    monkeypatch.setattr(glm5_next, "_route_experts", route)
    monkeypatch.setattr(glm5_next, "glm5_next_swiglu", swiglu)
    fn = (glm5_next.glm5_next_mlp_layer_stationary_tiles if tiled
          else glm5_next.glm5_next_mlp)
    input_value = tiles if tiled else tiles[0]
    expected = fn(input_value, {}, "model.layers.0", cfg, 0, None,
                  iter_expert_batches=batches)
    calls.clear()
    stats = {}
    actual = fn(input_value, {}, "model.layers.0", cfg, 0, None,
                iter_expert_batches=batches, shared_expert_overlap=True,
                shared_overlap_stats=stats)
    expected = expected if tiled else [expected]
    actual = actual if tiled else [actual]
    mx.eval(*expected, *actual)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(left.view(mx.uint16)),
                                      np.asarray(right.view(mx.uint16)))
    if len(widths) <= 6 and all(width == 1 for width in widths):
        assert calls.index("submit") < calls.index("shared") < calls.index("wait")
        assert stats["calls"] == 1 and stats["positions"] == len(widths)
        assert stats["retained_bytes_peak"] == 8 * len(widths)
    else:
        assert calls.index("shared") > calls.index("routed")
        assert stats == {}
