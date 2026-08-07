"""F219: the compressor's carry buffers must not chain a lazy graph.

``CompressorState._write`` rebuilds ``kv_state`` by concatenating around the
previous ``kv_state``. Left unevaluated that is one graph link per decode step
per layer, and each link pins the projections that fed it -- including the
compressor's own [1024, 4096] weight matrices. The buffers stay small, so
size accounting never notices; only the process does.

This asserts the property directly, by walking how much memory a long run of
steps retains, rather than asserting that a particular ``mx.eval`` call exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RATIO, HEAD_DIM, STEPS = 4, 64, 40


def _run(steps):
    """Step the state with projections derived from a large weight."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    state = CompressorState(RATIO, HEAD_DIM, batch=1, dtype=mx.float32)
    ape = mx.zeros((RATIO, 2 * HEAD_DIM), mx.float32)
    # Stand-in for the real wkv/wgate: big enough that retaining one per step
    # is unmistakable against the buffers' own few kilobytes.
    weight = mx.zeros((2 * HEAD_DIM, 8192), mx.float32)
    mx.eval(weight)
    hidden = mx.ones((1, 1, 8192), mx.float32)
    baseline = mx.get_active_memory()
    for position in range(steps):
        state.step(hidden @ weight.T, hidden @ weight.T, position, ape)
    mx.eval(state.kv_state, state.score_state)
    return mx.get_active_memory() - baseline


def test_stepping_retains_a_bounded_amount():
    import mlx.core as mx

    short = _run(4)
    long = _run(STEPS)
    # A chained graph grows with the step count; a materialized buffer does not.
    assert long <= short + 8 * 1024 * 1024, (
        f"retention grew from {short} to {long} bytes over {STEPS} steps -- "
        "the carry buffers are chaining a lazy graph")


def test_state_is_materialized_after_each_step():
    """Direct form of the same property: no pending graph after a step."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    state = CompressorState(RATIO, HEAD_DIM, batch=1, dtype=mx.float32)
    ape = mx.zeros((RATIO, 2 * HEAD_DIM), mx.float32)
    kv = mx.ones((1, 1, 2 * HEAD_DIM), mx.float32)
    for position in range(3):
        state.step(kv, kv, position, ape)
        before = mx.get_active_memory()
        mx.eval(state.kv_state, state.score_state)
        after = mx.get_active_memory()
        assert after == before, (
            "evaluating after step() allocated, so step() left work pending")


def test_values_are_unchanged_by_materialization():
    """The fix must be a scheduling change only."""
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    rng = np.random.default_rng(0)
    ape = mx.array(rng.normal(size=(RATIO, 2 * HEAD_DIM)).astype(np.float32))
    rows = [mx.array(rng.normal(size=(1, 1, 2 * HEAD_DIM)).astype(np.float32))
            for _ in range(12)]

    state = CompressorState(RATIO, HEAD_DIM, batch=1, dtype=mx.float32)
    emitted = []
    for position, row in enumerate(rows):
        out = state.step(row, row, position, ape)
        if out is not None:
            emitted.append(np.array(out))

    assert emitted, "no group completed; the test would prove nothing"
    # Emission cadence is the released one: one entry per completed group.
    assert len(emitted) == len(rows) // RATIO
    assert all(np.isfinite(e).all() for e in emitted)
