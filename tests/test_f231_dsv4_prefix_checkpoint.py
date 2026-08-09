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

    from runtime.engine import RuntimeConfig, StreamingEngine

    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = RuntimeConfig()          # persistence off unless a test sets it
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


# ---- on-disk persistence --------------------------------------------------


def _disk_engine(tmp_path, fingerprint="fp-1", ratios=(0, 4, 128)):
    from collections import OrderedDict

    from runtime.engine import RuntimeConfig, StreamingEngine

    engine = _engine()
    engine.rc = RuntimeConfig(dsv4_prefix_cache_dir=str(tmp_path))
    engine._dsv4_prefix_fingerprint = lambda: fingerprint

    class _Cfg:
        compress_ratios = list(ratios)
        head_dim = 4
    engine.cfg = _Cfg()
    return engine


def test_checkpoint_survives_a_process_boundary(tmp_path):
    """The point: a COLD start skips the preamble it never prefilled."""
    import mlx.core as mx

    writer = _disk_engine(tmp_path)
    _capture(writer, 4, layers=2, fill=7.0)
    assert writer._dsv4_prefix_store([1, 2, 3, 4, 5, 6]) == 4

    # A fresh engine with no in-memory state at all.
    reader = _disk_engine(tmp_path)
    kv = _FakeKV()
    assert reader._dsv4_prefix_restore(kv, [1, 2, 3, 4, 9]) == 4
    assert kv.dsv4_pos == 4
    assert float(mx.max(kv.dsv4_rings[0])) == 7.0
    assert set(kv.dsv4_cstate) == {0, 1}


def test_disk_checkpoint_refuses_a_different_fingerprint(tmp_path):
    """A model or runtime change must invalidate it, not reinterpret it."""
    writer = _disk_engine(tmp_path, fingerprint="fp-A")
    _capture(writer, 4)
    writer._dsv4_prefix_store([1, 2, 3, 4, 5])

    reader = _disk_engine(tmp_path, fingerprint="fp-B")
    kv = _FakeKV()
    assert reader._dsv4_prefix_restore(kv, [1, 2, 3, 4, 5]) == 0


def test_disk_checkpoint_refuses_a_diverging_prefix(tmp_path):
    writer = _disk_engine(tmp_path)
    _capture(writer, 4)
    writer._dsv4_prefix_store([1, 2, 3, 4, 5])

    reader = _disk_engine(tmp_path)
    kv = _FakeKV()
    assert reader._dsv4_prefix_restore(kv, [1, 2, 9, 4, 5]) == 0
    assert reader._dsv4_prefix_restore(kv, [1, 2, 3]) == 0


def test_disk_checkpoint_stores_tokens_not_a_digest(tmp_path):
    """Verification must be exact; a digest collision would be unrecoverable."""
    import mlx.core as mx

    writer = _disk_engine(tmp_path)
    _capture(writer, 4)
    writer._dsv4_prefix_store([11, 22, 33, 44, 55])

    files = list(tmp_path.glob("dsv4_prefix_*.safetensors"))
    assert files, "nothing written"
    arrays, meta = mx.load(str(files[0]), return_metadata=True)
    assert "tokens" in arrays
    assert [int(v) for v in arrays["tokens"].tolist()] == [11, 22, 33, 44]
    assert meta["boundary"] == "4"


def test_missing_directory_is_not_an_error(tmp_path):
    """Persistence is best-effort; an unset directory just disables it."""
    engine = _engine()

    from runtime.engine import RuntimeConfig

    engine.rc = RuntimeConfig()
    assert engine._dsv4_prefix_dir() is None
    kv = _FakeKV()
    assert engine._dsv4_prefix_restore(kv, [1, 2, 3]) == 0
