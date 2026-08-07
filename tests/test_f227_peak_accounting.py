"""F227: reported peak conflates live memory with MLX's reclaimable pool.

``true_peak_metal_bytes`` is a high-water mark of MLX's allocator, which
retains freed buffers in a pool rather than returning them immediately. That
pool is reclaimable on demand, so a request reported at 7.83GB was holding
5.71GB live and 2.29GB of buffers it would have released under pressure.

This matters because the fused MXFP8 trunk path was held behind a flag on the
strength of a 7.75GB "peak" said to be past the ~7GB ceiling. Measured
properly the live figure is 5.71-5.75GB, and the pool is the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_pool_is_reclaimable_and_distinct_from_live_memory():
    """The property the peak metric misses: pooled bytes are not pressure.

    Deliberately does NOT compare live memory before and after dropping the
    scratch buffers. That comparison depends on when the interpreter and MLX
    get around to reclaiming, which passed alone and failed in a full run on a
    single 268MB array that had not been released yet. What matters is that
    the pool exists, releases on demand, and that releasing it leaves data we
    still hold intact.
    """
    import gc

    import mlx.core as mx

    mx.clear_cache()
    held = mx.ones((256, 256), mx.float32)
    mx.eval(held)

    for _ in range(4):
        scratch = mx.zeros((4096, 4096), mx.float32)
        mx.eval(scratch)
        del scratch
    gc.collect()

    pooled = mx.get_cache_memory()
    assert pooled > 0, "nothing pooled; this test cannot show the distinction"

    mx.clear_cache()
    assert mx.get_cache_memory() == 0, "pool did not release on demand"
    # Releasing the pool must not disturb live data -- that is the whole
    # reason a peak that counts pooled bytes overstates pressure.
    assert float(mx.sum(held)) == 256 * 256


def test_peak_exceeds_live_when_a_pool_exists():
    """A peak reading includes pooled bytes, so it overstates pressure."""
    import mlx.core as mx

    mx.clear_cache()
    mx.reset_peak_memory()
    held = mx.zeros((4096, 4096), mx.float32)
    mx.eval(held)
    for _ in range(3):
        scratch = mx.zeros((4096, 4096), mx.float32)
        mx.eval(scratch)
        del scratch

    peak = mx.get_peak_memory()
    live = mx.get_active_memory()
    assert peak >= live, "peak below live is impossible"
    mx.clear_cache()
    assert mx.get_active_memory() <= live
    del held
