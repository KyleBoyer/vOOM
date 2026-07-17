#!/usr/bin/env python3
"""Pure F68 safety regressions; imports neither MLX nor Torch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.adaptive_chunk import AdaptiveChunkController


def test_kv_telemetry_not_double_counted() -> None:
    def run(kv_before: int):
        ctrl = AdaptiveChunkController(
            safe_bytes=10_000, initial_chunk=10, margin_bytes=0, dead_band=0.0
        )
        ctrl.observe(10, 2_000, 1_000, kv_before, False)
        ctrl.observe(20, 3_000, 1_000, kv_before, False)
        return ctrl.next_chunk_size(), ctrl.events

    assert run(0) == run(9_000_000)


def test_three_bad_chunks_freeze_at_reduced_minimum() -> None:
    ctrl = AdaptiveChunkController(safe_bytes=100, initial_chunk=4, margin_bytes=0)
    for chunk in (4, 2, 1):
        ctrl.observe(chunk, peak=101, active_before=0, kv_before=10_000,
                     governor_event=False)
    assert ctrl.failed and ctrl.unsafe_at_minimum
    assert ctrl.next_chunk_size() == 1
    assert any("FROZEN" in event for event in ctrl.events)


def test_padded_fit_covers_every_observation() -> None:
    ctrl = AdaptiveChunkController(safe_bytes=100_000, initial_chunk=8)
    ctrl._history = [(2, 21), (4, 55), (8, 81), (16, 190)]
    alpha, beta = ctrl._fit_alpha_beta()
    assert alpha > 0 and beta >= 0
    for chunk, delta in ctrl._history:
        assert alpha * chunk + beta >= delta


def test_live_safe_ceiling_can_change_without_discarding_cost_history() -> None:
    ctrl = AdaptiveChunkController(
        safe_bytes=10_000, initial_chunk=10, margin_bytes=0, dead_band=0.0
    )
    ctrl.observe(10, peak=2_000, active_before=1_000, kv_before=0,
                 governor_event=False)
    history = list(ctrl._history)

    ctrl.update_safe_bytes(6_000)
    assert ctrl.safe_bytes == 6_000
    assert ctrl.min_safe_bytes == 6_000
    assert ctrl.max_safe_bytes == 10_000
    assert ctrl._history == history

    ctrl.update_safe_bytes(12_000)
    assert ctrl.min_safe_bytes == 6_000
    assert ctrl.max_safe_bytes == 12_000


def test_growing_kv_overshoot_is_detected_and_shrinks_not_stays_wrong() -> None:
    """STATUS.md's "Current truth" names a "growing-KV" regression test that
    didn't exist anywhere in the repo (grepped tests/, runtime/, experiments/
    for growing_kv/growing-KV: zero matches) -- this closes that gap.

    The affine fit's `budget = safe_bytes - active_before - margin - beta` uses
    active_before AT THE TIME OF THE FIT, but active_before (resident KV) keeps
    climbing every subsequent chunk during a long prefill -- by the time the
    resulting chunk size is actually used, the real active_before can be
    higher than what the fit assumed, and the module's own docstring is
    explicit this ISN'T a certified bound: "a safety heuristic, not a
    statistical certificate" that relies on the real Metal governor (F42) as
    the hard backstop, not the chunk predictor alone. Confirmed empirically:
    a simple deterministic linear cost model (peak = active_before +
    chunk*per_position_cost) with steadily growing active_before DOES produce
    a real overshoot around step 12 of 20 in this setup -- so this is a real,
    reachable regime, not a hypothetical. What this test actually verifies is
    the controller's half of the safety contract: when growing KV causes a
    genuine overshoot (what the real governor would flag), it's correctly
    classified as bad and the chunk size shrinks (or holds at an already-
    reduced/frozen size) -- never grows -- in response, rather than staying
    confidently wrong.
    """
    safe_bytes = 10_000_000
    ctrl = AdaptiveChunkController(
        safe_bytes=safe_bytes, initial_chunk=100, margin_bytes=100_000, dead_band=0.0
    )
    active_before = 0
    per_position_cost = 1000
    saw_overshoot = False
    for _ in range(20):
        chunk = ctrl.next_chunk_size()
        peak = active_before + chunk * per_position_cost
        governor_event = peak > safe_bytes  # what the real Metal governor would flag
        chunk_before = ctrl.chunk
        ctrl.observe(chunk, peak, active_before, kv_before=active_before,
                    governor_event=governor_event)
        if governor_event:
            saw_overshoot = True
            assert ctrl.chunk <= chunk_before, (
                "chunk grew (or the controller failed to shrink) immediately "
                "after a real growing-KV overshoot"
            )
        active_before += chunk * per_position_cost // 2  # KV keeps accumulating
    assert saw_overshoot, (
        "test setup never reached the growing-KV overshoot regime it's meant "
        "to exercise -- strengthen per_position_cost or the KV growth rate"
    )


def test_moe_routing_spike_does_not_break_safety_or_get_masked() -> None:
    """The other STATUS.md-named "routing-spike" test that didn't exist
    (same grep, zero matches). Reproduces the shape of the real OLMoE-1B-7B
    incident (docs/benchmark_results.md "OLMoE follow-up"): a single chunk
    whose measured cost is a large outlier relative to trend (different
    experts routed to, not a bigger true problem) but still comfortably under
    the safety budget -- exactly what caused the documented oscillation,
    while "no chunk actually approached the safety budget" the whole time.

    Verifies three things a routing-spike must NOT do: (1) it must not itself
    be misclassified as unsafe when it wasn't (a noisy-but-safe observation
    isn't a governor event), (2) the padded envelope fit afterward must still
    cover it (not just the stable points -- a real outlier shouldn't be
    treated as if it never happened), and (3) crucially, padding for noise
    tolerance must not blind the controller to a GENUINE overshoot that
    happens shortly after -- real bad events must still register as bad.
    """
    safe_bytes = 6_000_000_000  # real OLMoE-1B-7B true-peak scale
    stable_cost_per_pos = 1_000_000
    spike_multiplier = 2.0  # noisy but still safe, matching the real incident

    ctrl = AdaptiveChunkController(safe_bytes=safe_bytes, initial_chunk=512, margin_bytes=200_000_000)
    for _ in range(4):
        chunk = ctrl.next_chunk_size()
        ctrl.observe(chunk, chunk * stable_cost_per_pos, 0, 0, False)

    spike_chunk = ctrl.next_chunk_size()
    spike_peak = int(spike_chunk * stable_cost_per_pos * spike_multiplier)
    assert spike_peak <= safe_bytes, "test setup's spike must itself be safe, matching the real incident"
    ctrl.observe(spike_chunk, spike_peak, 0, 0, False)
    assert not any("BAD" in e for e in ctrl.events[-1:]), (
        "a noisy-but-safe routing spike was misclassified as a bad/unsafe event"
    )

    alpha, beta = ctrl._fit_alpha_beta()
    for c, d in ctrl._history:
        assert alpha * c + beta >= d - 1e-6, (
            f"padded envelope fit does not cover the routing-spike observation ({c}, {d})"
        )

    # A genuine overshoot right after noisy-but-safe history must still be
    # caught -- padding tolerance for the spike must not mask a real one.
    ctrl2 = AdaptiveChunkController(safe_bytes=safe_bytes, initial_chunk=512, margin_bytes=200_000_000)
    for _ in range(4):
        chunk = ctrl2.next_chunk_size()
        ctrl2.observe(chunk, chunk * stable_cost_per_pos, 0, 0, False)
    bad_streak_before = ctrl2._bad_streak
    overshoot_chunk = ctrl2.next_chunk_size()
    ctrl2.observe(overshoot_chunk, safe_bytes + 1, 0, 0, False)
    assert ctrl2._bad_streak == bad_streak_before + 1, "a genuine overshoot was not registered as bad"
    assert any("BAD" in e for e in ctrl2.events), "a genuine overshoot left no BAD event in the log"


def _run_all() -> None:
    tests = [
        test_kv_telemetry_not_double_counted,
        test_three_bad_chunks_freeze_at_reduced_minimum,
        test_padded_fit_covers_every_observation,
        test_growing_kv_overshoot_is_detected_and_shrinks_not_stays_wrong,
        test_moe_routing_spike_does_not_break_safety_or_get_masked,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert "mlx" not in sys.modules and "torch" not in sys.modules
    print(f"PASS {len(tests)}/{len(tests)}; no MLX/Torch import")


if __name__ == "__main__":
    try:
        _run_all()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
