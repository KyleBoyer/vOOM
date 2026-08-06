"""F214: oracle for DeepSeek V4's window ring buffer and gather assembly.

One buffer per layer holds a ``window_size`` ring of recent positions followed
by the compressed region, and a single gather list addresses both -- which is
why compressed indices carry an offset.

The index generators are checked against the checkpoint's own
``get_compress_topk_idxs``, imported unchanged. The ring placement is checked
by the property it exists to guarantee: slot ``p % window`` holds position
``p``. That is what every subsequent decode step depends on, and a contiguous
write satisfies every shape while breaking it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INFERENCE = ROOT / "models" / "DeepSeek-V4-Flash-0731" / "inference"
has_reference = (INFERENCE / "model.py").is_file()

WINDOW = 8


def _reference():
    stub = sys.modules.get("kernel") or types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm",
                 "sparse_attn", "hc_split_sinkhorn"):
        if not hasattr(stub, name):
            def _unavailable(*_a, __name=name, **_k):
                raise RuntimeError(f"kernel {__name!r} unavailable here")
            setattr(stub, name, _unavailable)
    sys.modules["kernel"] = stub
    sys.path.insert(0, str(INFERENCE))
    import model as reference_module

    return reference_module


def _positions(seqlen):
    """kv whose value encodes its own absolute position."""
    import mlx.core as mx

    return mx.arange(seqlen).reshape(1, seqlen, 1).astype(mx.float32)


@pytest.mark.parametrize("seqlen", [3, 8, 9, 17, 24])
def test_ring_slot_holds_its_own_position(seqlen):
    """Slot p % window must hold position p after prefill."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_ring_write

    ring = mx.zeros((1, WINDOW, 1))
    ring = window_ring_write(ring, _positions(seqlen), 0, WINDOW)
    assert ring.shape == (1, WINDOW, 1)
    values = np.array(ring).reshape(-1)

    retained = min(seqlen, WINDOW)
    for position in range(seqlen - retained, seqlen):
        assert values[position % WINDOW] == position, (
            f"seqlen={seqlen}: slot {position % WINDOW} holds "
            f"{values[position % WINDOW]}, expected {position}")


def test_contiguous_write_would_break_the_invariant():
    """Guard: the naive placement satisfies every shape and is still wrong."""
    import mlx.core as mx

    from runtime.deepseek_v4 import window_ring_write

    seqlen = 17  # cutoff = 1, so a contiguous write is off by one
    correct = np.array(window_ring_write(
        mx.zeros((1, WINDOW, 1)), _positions(seqlen), 0, WINDOW)).reshape(-1)
    naive = np.array(_positions(seqlen))[0, -WINDOW:, 0]
    assert not np.array_equal(correct, naive), (
        "rotated and contiguous writes agree here, so this test cannot "
        "detect the error")
    assert correct[seqlen % WINDOW - 1] == seqlen - 1


def test_decode_writes_one_slot_and_leaves_the_rest():
    import mlx.core as mx

    from runtime.deepseek_v4 import window_ring_write

    ring = window_ring_write(mx.zeros((1, WINDOW, 1)), _positions(WINDOW), 0,
                             WINDOW)
    before = np.array(ring).reshape(-1).copy()
    new = mx.array([[[99.0]]])
    ring = window_ring_write(ring, new, WINDOW, WINDOW)
    after = np.array(ring).reshape(-1)

    slot = WINDOW % WINDOW
    assert after[slot] == 99.0
    for index in range(WINDOW):
        if index != slot:
            assert after[index] == before[index], f"slot {index} was disturbed"


@pytest.mark.skipif(not has_reference, reason="checkpoint inference/ absent")
@pytest.mark.parametrize("start_pos,seqlen", [(0, 16), (0, 20), (7, 1),
                                              (63, 1)])
def test_compressed_indices_match_the_released_generator(start_pos, seqlen):
    import mlx.core as mx

    reference = _reference()
    from runtime.deepseek_v4 import compress_topk_idxs

    ratio, offset = 4, 100
    expected = reference.get_compress_topk_idxs(
        ratio, 1, seqlen, start_pos, offset).numpy()
    got = np.array(compress_topk_idxs(ratio, seqlen, start_pos, offset))
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    assert np.array_equal(got, expected), (
        f"start_pos={start_pos} seqlen={seqlen} diverged")


def test_gather_concatenates_window_then_compressed():
    import mlx.core as mx

    from runtime.deepseek_v4 import gather_indices, window_topk_idxs

    seqlen, ratio, offset = 16, 4, 64
    windowed = np.array(window_topk_idxs(WINDOW, seqlen, 0))
    combined = np.array(gather_indices(WINDOW, ratio, seqlen, 0, offset))
    assert combined.shape[-1] == windowed.shape[-1] + seqlen // ratio
    assert np.array_equal(combined[..., :windowed.shape[-1]], windowed)
    tail = combined[..., windowed.shape[-1]:]
    used = tail[tail >= 0]
    assert (used >= offset).all(), (
        "compressed indices must be offset past the window region")


def test_ratio_zero_layer_has_no_compressed_region():
    """compress_ratios contains 0; those layers gather the window only."""
    import mlx.core as mx

    from runtime.deepseek_v4 import gather_indices, window_topk_idxs

    windowed = np.array(window_topk_idxs(WINDOW, 12, 0))
    combined = np.array(gather_indices(WINDOW, 0, 12, 0, 64))
    assert np.array_equal(combined, windowed)


def test_compressed_indices_never_reach_the_current_group():
    import mlx.core as mx

    from runtime.deepseek_v4 import compress_topk_idxs

    ratio, seqlen, offset = 4, 20, 50
    idxs = np.array(compress_topk_idxs(ratio, seqlen, 0, offset))[0]
    for position in range(seqlen):
        used = idxs[position]
        used = used[used >= 0] - offset
        assert (used < (position + 1) // ratio).all(), (
            f"position {position} read a compressed entry it cannot see")
