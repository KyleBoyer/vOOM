"""F223: exact-prompt reuse of DeepSeek V4's ring/compressor state.

F37's prompt store cannot serialize this model -- its state is not a KVCache,
so load_longest_prefix always misses and every repeat pays full prefill. The
state is small (a 128-slot ring plus a compressor carry buffer per layer,
~20MB), so an exact-prompt snapshot skips the sweep entirely.

The subtle requirement is container ownership. Decode rebinds dict entries and
CompressorState attributes rather than mutating arrays, so sharing arrays is
safe while sharing containers is not: a later request would advance the
snapshot's own ring and silently serve corrupted state. These tests assert
that separation directly, because a shared container passes every shape and
type check and only shows up as wrong output on the third request.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FakeKV:
    pass


def _engine():
    """A StreamingEngine shell with only the snapshot state initialized."""
    from collections import OrderedDict

    from runtime.engine import StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine._dsv4_snapshots = OrderedDict()
    engine._dsv4_snapshot_slots = 2
    return engine


def _kv(pos=5, fill=1.0):
    import mlx.core as mx

    from runtime.deepseek_v4 import CompressorState

    kv = _FakeKV()
    kv.dsv4_rings = {0: mx.full((1, 8, 4), fill)}
    kv.dsv4_compressed = {0: mx.full((1, 2, 4), fill)}
    state = CompressorState(4, 2, batch=1, dtype=mx.float32)
    kv.dsv4_cstate = {0: state}
    kv.dsv4_pos = pos
    return kv


def test_snapshot_survives_later_mutation_of_the_live_state():
    import mlx.core as mx

    engine = _engine()
    kv = _kv(pos=5, fill=1.0)
    engine._dsv4_snapshot_store([1, 2, 3], kv, mx.zeros((1, 4)))

    # Simulate what decode does: rebind entries and advance position.
    kv.dsv4_rings[0] = mx.full((1, 8, 4), 9.0)
    kv.dsv4_compressed[1] = mx.zeros((1, 1, 4))
    kv.dsv4_cstate[0].kv_state = mx.full((1, 8, 4), 9.0)
    kv.dsv4_pos = 99

    snapshot = engine._dsv4_snapshot_lookup([1, 2, 3])
    assert snapshot is not None
    assert snapshot["pos"] == 5, "snapshot position followed the live state"
    assert float(mx.max(snapshot["rings"][0])) == 1.0, (
        "snapshot ring was overwritten by later decoding")
    assert 1 not in snapshot["compressed"], (
        "snapshot compressed store grew with the live one")
    assert float(mx.max(snapshot["cstate"][0].kv_state)) == 0.0, (
        "snapshot compressor state followed the live object")


def test_restore_gives_the_request_its_own_containers():
    import mlx.core as mx

    engine = _engine()
    source = _kv(pos=7, fill=2.0)
    engine._dsv4_snapshot_store([4, 5], source, mx.zeros((1, 4)))
    snapshot = engine._dsv4_snapshot_lookup([4, 5])

    target = _FakeKV()
    engine._dsv4_snapshot_restore(target, snapshot)
    assert target.dsv4_pos == 7
    assert float(mx.max(target.dsv4_rings[0])) == 2.0

    # Mutating the restored request must not damage the snapshot.
    target.dsv4_rings[0] = mx.zeros((1, 8, 4))
    target.dsv4_cstate[0].kv_state = mx.full((1, 8, 4), 5.0)
    target.dsv4_pos = 123
    assert snapshot["pos"] == 7
    assert float(mx.max(snapshot["rings"][0])) == 2.0
    assert float(mx.max(snapshot["cstate"][0].kv_state)) == 0.0

    # And a second restore is still clean.
    second = _FakeKV()
    engine._dsv4_snapshot_restore(second, snapshot)
    assert float(mx.max(second.dsv4_rings[0])) == 2.0
    assert second.dsv4_pos == 7


def test_lookup_is_exact_and_never_serves_a_prefix():
    """The ring holds only the LAST 128 positions, so a shorter prompt cannot
    be served from a longer snapshot."""
    import mlx.core as mx

    engine = _engine()
    engine._dsv4_snapshot_store([1, 2, 3, 4], _kv(), mx.zeros((1, 4)))
    assert engine._dsv4_snapshot_lookup([1, 2, 3, 4]) is not None
    assert engine._dsv4_snapshot_lookup([1, 2, 3]) is None
    assert engine._dsv4_snapshot_lookup([1, 2, 3, 4, 5]) is None
    assert engine._dsv4_snapshot_lookup([1, 2, 4, 3]) is None


def test_slots_are_bounded_and_evict_oldest_first():
    import mlx.core as mx

    engine = _engine()  # slots = 2
    for i in range(4):
        engine._dsv4_snapshot_store([i], _kv(pos=i), mx.zeros((1, 4)))
    assert len(engine._dsv4_snapshots) <= 2
    assert engine._dsv4_snapshot_lookup([0]) is None
    assert engine._dsv4_snapshot_lookup([3]) is not None


def test_store_ignores_a_request_with_no_dsv4_state():
    """Never snapshot a kv that never ran the DeepSeek V4 path."""
    import mlx.core as mx

    engine = _engine()
    engine._dsv4_snapshot_store([1], _FakeKV(), mx.zeros((1, 4)))
    assert engine._dsv4_snapshots == {}


def test_reuse_is_opt_in():
    source = Path(ROOT / "runtime" / "engine.py").read_text()
    assert 'VMODEL_DSV4_PROMPT_REUSE' in source
    assert '_os.environ.get("VMODEL_DSV4_PROMPT_REUSE") == "1"' in source, (
        "prompt state reuse must be gated on an explicit opt-in")
