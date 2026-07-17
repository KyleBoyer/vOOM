"""F74-v2: the missing engine+MLX liveness/true-peak artifact.

tests/test_expert_fetch_batch.py and tests/test_expert_batching.py prove the
lifetime-boundary CONTRACT (generator ownership, exception-closes-producer)
with plain Python objects and no MLX. That leaves the actual claim this
project cares about unverified: does bounding fetch+compute to one batch at a
time actually bound *Metal* memory, not just Python-object liveness?

This test drives the real `runtime.expert_batching.consume_expert_batches`
with real `mx.array` pages and `mx.eval()`, sized like real GLM-5.2 expert
pages (proportionally -- small absolute size so this stays cheap and safe on
a 16 GB machine), and measures actual `mx.get_peak_memory()` /
`mx.get_active_memory()`. It reproduces, at safe synthetic scale, the same
coupon-collector shape as the real 2026-07-14 incident (STATUS.md "Current
truth"): a full expert union materialized in one dict recreates a peak near
the FULL union size, while the bounded-batch path peaks near one batch.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from runtime.expert_batching import consume_expert_batches

PAGE_MB = 4
PAGE_ELEMS = (PAGE_MB * 1024 * 1024) // 4  # float32
N_EXPERTS = 32  # union == 128 MB
BATCH = 4  # bounded batch == 16 MB
UNION_BYTES = N_EXPERTS * PAGE_MB * 1024 * 1024
BATCH_BYTES = BATCH * PAGE_MB * 1024 * 1024


def _make_page() -> mx.array:
    return mx.zeros((PAGE_ELEMS,), dtype=mx.float32)


def _batches():
    for start in range(0, N_EXPERTS, BATCH):
        ids = list(range(start, min(start + BATCH, N_EXPERTS)))
        yield ids, {i: _make_page() for i in ids}


def test_unbounded_union_peaks_near_full_union_size():
    """Reproduces the disproven F74-v1 shape: one dict holding every page.

    `mx.get_active_memory()`/`get_peak_memory()` are process-wide counters,
    not scoped to this test -- an earlier test in the same pytest process
    (running the full suite in random order) may leave unrelated Metal memory
    resident. Every assertion here is therefore relative to a baseline
    captured immediately before this test's own allocations, never to an
    absolute zero.
    """
    baseline_active = mx.get_active_memory()
    mx.reset_peak_memory()
    union = {i: _make_page() for i in range(N_EXPERTS)}
    mx.eval(list(union.values()))
    peak_delta = mx.get_peak_memory() - baseline_active
    del union
    gc.collect()
    assert peak_delta >= UNION_BYTES * 0.9, (
        f"expected union materialization to add ~{UNION_BYTES / 1e6:.0f} MB over "
        f"baseline, got a delta of {peak_delta / 1e6:.1f} MB"
    )
    assert mx.get_active_memory() == baseline_active, (
        "union pages not released after del: active memory did not return to "
        f"its pre-test baseline ({baseline_active / 1e6:.1f} MB)"
    )


def test_bounded_batches_peak_near_one_batch_not_the_union():
    """The actual F74-v2 claim: real Metal peak tracks batch size, not union."""
    gc.collect()
    mx.clear_cache()
    baseline_active = mx.get_active_memory()
    mx.reset_peak_memory()

    def consume(ids, pages):
        mx.eval(list(pages.values()))

    consume_expert_batches(_batches(), consume)
    gc.collect()
    peak_delta = mx.get_peak_memory() - baseline_active

    # MLX may retain a few bytes of allocator/command metadata depending on
    # suite order (observed: 20 bytes over the exact 1.5x boundary). A 4 KiB
    # allowance is negligible beside a 16 MiB batch while keeping the union
    # (64 MiB) decisively outside the gate.
    assert peak_delta <= BATCH_BYTES * 1.5 + 4096, (
        f"expected batched peak to add at most ~{BATCH_BYTES / 1e6:.0f} MB over "
        f"baseline, got a delta of {peak_delta / 1e6:.1f} MB -- "
        f"consume_expert_batches is not bounding real Metal memory"
    )
    assert peak_delta < UNION_BYTES * 0.5, (
        "batched peak delta is not meaningfully smaller than the full union -- "
        "this is the exact regression F74-v1 shipped without catching"
    )
    assert mx.get_active_memory() == baseline_active, (
        "last batch's pages not released after consumption: active memory did "
        f"not return to its pre-test baseline ({baseline_active / 1e6:.1f} MB)"
    )


def test_mid_batch_compute_exception_releases_real_metal_memory():
    """The MLX/Metal exception-liveness proof STATUS.md says is still missing.

    A compute failure partway through must not stall Metal memory at the
    partial-accumulation high point -- the producer's generator must be
    closed and every already-yielded batch's pages already collectable.
    """
    gc.collect()
    mx.clear_cache()
    fail_at = N_EXPERTS // 2  # fail partway through, not on the first batch

    def consume(ids, pages):
        mx.eval(list(pages.values()))
        if ids[0] == fail_at:
            raise RuntimeError("injected compute failure")

    baseline_active = mx.get_active_memory()
    mx.reset_peak_memory()
    try:
        consume_expert_batches(_batches(), consume)
    except RuntimeError as exc:
        assert "injected compute failure" in str(exc)
    else:
        raise AssertionError("injected compute failure did not propagate")

    gc.collect()
    peak_delta = mx.get_peak_memory() - baseline_active
    active_after_unwind = mx.get_active_memory()

    # Same negligible allocator-metadata allowance as the success-path gate.
    # The full extra page that would indicate a real leak is 4 MiB, three
    # orders of magnitude larger than this tolerance.
    assert peak_delta <= BATCH_BYTES * 1.5 + 4096, (
        f"peak delta during the failing run was {peak_delta / 1e6:.1f} MB, "
        f"expected it bounded near one batch ({BATCH_BYTES / 1e6:.0f} MB) even "
        f"though the failure happened partway through the full union"
    )
    assert active_after_unwind == baseline_active, (
        f"active memory did not return to its pre-test baseline "
        f"({baseline_active / 1e6:.1f} MB, now {active_after_unwind / 1e6:.1f} MB) "
        f"after unwind -- the producer generator was not fully closed/released"
    )


def test_eval_batch_boundary_does_not_change_sequential_accumulation():
    """q=1 versus q=8 changes synchronization, not addition order."""
    contributions = [
        mx.arange(64, dtype=mx.float32).reshape(1, 64) * (index + 1) / 100
        for index in range(8)
    ]

    def accumulate(batch_size):
        out = mx.zeros_like(contributions[0])
        for start in range(0, len(contributions), batch_size):
            for contribution in contributions[start:start + batch_size]:
                out = out + contribution
            mx.eval(out)
        return out

    one = accumulate(1)
    eight = accumulate(8)
    assert mx.array_equal(one, eight)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__}: PASS")
            passed += 1
        except Exception as e:
            print(f"  {fn.__name__}: FAIL ({type(e).__name__}: {e})")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(fns)} tests passed")
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
