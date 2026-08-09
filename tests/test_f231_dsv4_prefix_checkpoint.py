"""F231: resume a DeepSeek V4 prefill from the longest stored prefix.

An agent-harness prompt is almost entirely a stable prefix. The captured
51,220-token request is 46,941 tokens of tool schemas plus 4,233 of system
prompt against a 42-token user message -- 99.9% identical on every turn --
and prefill is 62 of its 71 minutes. Resuming from a checkpoint below the
divergence point skips nearly all of that.

The checkpoint is captured DURING the single layer-stationary pass: each
layer's ring and compressor state at position B is available the moment that
layer's tile loop crosses B, so one entry per layer is the complete state at
B without a second sweep.

Correctness rests on one rule these tests pin: a checkpoint may be used only
when its ENTIRE key is a prefix of the request. Its state covers exactly
those positions -- a request diverging earlier must not use it, and one
diverging later resumes and prefills the remainder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FakeKV:
    pass


def _engine(slots=2):
    from collections import OrderedDict

    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine._dsv4_prefix_checkpoints = OrderedDict()
    engine._dsv4_prefix_slots = slots
    engine._dsv4_checkpoint = None
    return engine


def _capture(engine, boundary, layers=2, fill=1.0):
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    per_layer = {}
    for layer in range(layers):
        per_layer[layer] = {
            "ring": mx.full((1, 8, 4), fill),
            "store": mx.full((1, 3, 4), fill),
            "cstate": CompressorState(4, 2, batch=1, dtype=mx.float32),
        }
    engine._dsv4_checkpoint = (boundary, per_layer)


def test_resumes_from_a_checkpoint_that_is_a_prefix():
    engine = _engine()
    _capture(engine, 4)
    assert engine._dsv4_prefix_store([1, 2, 3, 4, 5, 6]) == 4

    kv = _FakeKV()
    assert engine._dsv4_prefix_restore(kv, [1, 2, 3, 4, 9, 9, 9]) == 4
    assert kv.dsv4_pos == 4
    assert set(kv.dsv4_rings) == {0, 1}


def test_refuses_a_checkpoint_the_request_diverges_before():
    """The state covers positions the request does not share."""
    engine = _engine()
    _capture(engine, 4)
    engine._dsv4_prefix_store([1, 2, 3, 4, 5])

    kv = _FakeKV()
    assert engine._dsv4_prefix_restore(kv, [1, 2, 9, 4, 5]) == 0
    assert not hasattr(kv, "dsv4_pos")


def test_refuses_a_checkpoint_longer_than_the_request():
    engine = _engine()
    _capture(engine, 4)
    engine._dsv4_prefix_store([1, 2, 3, 4, 5])
    kv = _FakeKV()
    assert engine._dsv4_prefix_restore(kv, [1, 2, 3]) == 0


def test_picks_the_longest_usable_checkpoint():
    engine = _engine(slots=4)
    _capture(engine, 2, fill=1.0)
    engine._dsv4_prefix_store([1, 2, 3, 4, 5, 6])
    _capture(engine, 4, fill=2.0)
    engine._dsv4_prefix_store([1, 2, 3, 4, 5, 6])

    kv = _FakeKV()
    assert engine._dsv4_prefix_restore(kv, [1, 2, 3, 4, 7, 7]) == 4


def test_restore_gives_the_request_its_own_containers():
    """A later request must not advance the stored checkpoint's own state."""
    import mlx.core as mx

    engine = _engine()
    _capture(engine, 4, fill=3.0)
    engine._dsv4_prefix_store([1, 2, 3, 4])

    first = _FakeKV()
    engine._dsv4_prefix_restore(first, [1, 2, 3, 4, 5])
    first.dsv4_rings[0] = mx.zeros((1, 8, 4))
    first.dsv4_cstate[0].kv_state = mx.full((1, 8, 4), 9.0)
    first.dsv4_pos = 99

    second = _FakeKV()
    assert engine._dsv4_prefix_restore(second, [1, 2, 3, 4, 6]) == 4
    assert float(mx.max(second.dsv4_rings[0])) == 3.0
    assert float(mx.max(second.dsv4_cstate[0].kv_state)) == 0.0
    assert second.dsv4_pos == 4


def test_capture_is_cleared_after_storing():
    """A stale capture must not be attributed to the next request."""
    engine = _engine()
    _capture(engine, 4)
    assert engine._dsv4_prefix_store([1, 2, 3, 4, 5]) == 4
    assert engine._dsv4_prefix_store([9, 9, 9, 9, 9]) == 0


def test_slots_are_bounded():
    engine = _engine(slots=2)
    for i in range(4):
        _capture(engine, 4)
        engine._dsv4_prefix_store([i, i, i, i, i])
    assert len(engine._dsv4_prefix_checkpoints) <= 2
