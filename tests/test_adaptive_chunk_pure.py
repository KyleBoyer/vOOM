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


def test_bad_first_observation_does_not_permanently_poison_growth() -> None:
    """Live-confirmed 2026-07-28 (EpistemeAI/VibeCoder-20B, a real 32-expert
    gpt_oss checkpoint): the very first chunk overshot (a cold-start
    expert-fetch spike -- nothing to do with steady-state per-token cost).
    Because the padded upper-envelope fit must cover EVERY history point
    forever (no aging/windowing), that one anomalous measurement inflated
    the fitted slope ~30-50x too high; it decayed back down only very
    slowly as later GREEN chunks accumulated, and the dead-band (added for
    a DIFFERENT problem, oscillation) then suppressed the still-too-small
    proposals -- across a real 4,000+ token prefill the chunk size never
    recovered past ~13-23 tokens from a starting point of 64.

    Reproduces the same shape here: one bad first observation with a huge
    delta relative to a subsequent honest, cheap, steady-state cost, then
    checks growth actually recovers within a handful of GREEN chunks
    instead of staying anchored near the bad observation's implied slope.
    """
    safe_bytes = 10_000_000_000  # 10 GB, matching real Metal-scale ceilings
    true_cost_per_pos = 50_000  # steady-state, once experts/caches are warm

    ctrl = AdaptiveChunkController(safe_bytes=safe_bytes, initial_chunk=64, margin_bytes=int(1e9))
    # Chunk 1: a cold-start spike that genuinely overshoots (64 tokens costs
    # as much as a real cold expert-fetch storm would).
    ctrl.observe(64, peak=safe_bytes + 1, active_before=0, kv_before=0, governor_event=False)
    assert any("BAD" in e for e in ctrl.events)
    assert ctrl._history == [], (
        "a bad/overshoot observation must not be added to the growth-fit history")

    # Subsequent chunks are honest, cheap, and safe -- the controller should
    # learn the TRUE slope from these, unpolluted by the first outlier.
    chunk_sizes_seen = []
    for _ in range(10):
        chunk = ctrl.next_chunk_size()
        chunk_sizes_seen.append(chunk)
        peak = chunk * true_cost_per_pos
        ctrl.observe(chunk, peak=peak, active_before=0, kv_before=0, governor_event=False)

    assert max(chunk_sizes_seen) > 200, (
        f"chunk size failed to recover after the bad first observation: {chunk_sizes_seen}")


def test_escalate_growth_cap_converges_faster_without_overshooting():
    """Opt-in growth-cap escalation (default off, identical behavior to the
    plain 2x/step cap when disabled): when the fit repeatedly wants to grow
    MORE than the current cap allows, several steps in a row, that's real
    signal the model is under-confident, not noise -- escalate the
    multiplier (2x -> 3x -> ...) so it reaches a good steady-state size in
    fewer GREEN-streak cycles. Must never actually overshoot `safe_bytes`
    regardless -- the escalation only affects how fast growth CAN happen
    when a chunk has already been measured safe, never the safety check
    itself."""
    safe_bytes = 10_000_000_000
    true_cost_per_pos = 50_000
    # The true safe chunk size is safe_bytes/true_cost_per_pos = 200,000 --
    # deliberately far above what a plain 2x/step cap can reach quickly from
    # a small starting point, so escalation has real room to help.

    def run(escalate: bool, steps: int = 30) -> list[int]:
        ctrl = AdaptiveChunkController(
            safe_bytes=safe_bytes, initial_chunk=64, margin_bytes=int(1e9),
            escalate_growth_cap=escalate)
        sizes = []
        for _ in range(steps):
            chunk = ctrl.next_chunk_size()
            sizes.append(chunk)
            peak = chunk * true_cost_per_pos
            assert peak <= safe_bytes, (
                f"escalation caused a real overshoot: chunk={chunk} peak={peak}")
            ctrl.observe(chunk, peak=peak, active_before=0, kv_before=0, governor_event=False)
        return sizes

    baseline = run(escalate=False)
    escalated = run(escalate=True)
    assert max(escalated) > max(baseline), (
        f"escalation did not converge faster: baseline max={max(baseline)}, "
        f"escalated max={max(escalated)}")


def test_expert_fetch_noise_does_not_collapse_chunk_size_when_residualized():
    """Live-confirmed 2026-07-29 (EpistemeAI/VibeCoder-20B, real 32-expert/
    4-active-per-token gpt_oss checkpoint): even with ONLY green/safe
    observations (the same-day bad-observation-exclusion fix already
    applied), the chunk collapsed 128->64->32->16->8->4->2->1 over ~60 real
    seconds while measured active memory stayed completely flat, nowhere
    near the safety ceiling. Root mechanism reproduced directly here: a
    SMALL chunk that luckily hit mostly-cached experts can show a SMALLER
    real (peak - active_before) delta than a BIGGER chunk that unluckily
    hit several cold ones -- exactly backwards from what the affine fit
    assumes (cost grows with chunk_size). Left inside `delta`, that single
    adversarial pair alone inflates the fitted per-token cost by >100,000x
    in this test's real numbers. Passing each chunk's own real
    `expert_fetch_bytes` lets the fit residualize that confound out and
    recover the true, much smaller per-token compute cost instead.
    """
    safe_bytes = 10_000_000_000
    margin = int(1e9)
    true_cost_per_token = 100  # real compute-scratch cost: small, stable
    expert_page_bytes = 75_000_000  # ~real per-expert page size scale
    # (chunk_size, expert_misses): the small chunk got lucky (few cold
    # experts), the bigger chunk got unlucky (many) -- both are real,
    # physically possible outcomes for the SAME model, since which experts
    # a chunk's tokens route to isn't a function of chunk_size at all.
    adversarial_pair = [(128, 2), (256, 25)]

    def fit_after(residualize: bool):
        ctrl = AdaptiveChunkController(
            safe_bytes=safe_bytes, initial_chunk=128, margin_bytes=margin)
        for chunk_size, misses in adversarial_pair:
            expert_bytes = misses * expert_page_bytes
            peak = chunk_size * true_cost_per_token + expert_bytes
            assert peak <= safe_bytes, "test setup itself must stay safe"
            ctrl.observe(
                chunk_size, peak=peak, active_before=0, kv_before=0,
                governor_event=False,
                expert_fetch_bytes=expert_bytes if residualize else 0)
        alpha, beta = ctrl._fit_alpha_beta()
        budget = safe_bytes - margin - beta
        proposed = int(budget / alpha) if alpha > 0 else None
        return alpha, proposed

    alpha_without, proposed_without = fit_after(residualize=False)
    alpha_with, proposed_with = fit_after(residualize=True)

    assert alpha_without > 1_000_000, (
        "test setup did not reproduce a spuriously inflated per-token cost "
        f"estimate without residualizing: alpha={alpha_without:.1f}")
    # 1.25x safety pad on the true 100 bytes/token still leaves real room.
    assert 100 <= alpha_with <= 200, (
        f"residualized alpha should track the true per-token cost: {alpha_with:.1f}")
    assert proposed_with >= proposed_without * 1000, (
        "residualizing expert-fetch noise out of the fit should let a much "
        f"larger, still-safe chunk be proposed: without={proposed_without} "
        f"with={proposed_with}")


def test_finite_expert_pool_stops_charging_for_impossible_new_experts():
    """The adversarial route bound becomes constant after all E experts."""
    page_bytes = 15_000_000
    per_token = 4 * page_bytes
    pool_bytes = 128 * page_bytes
    alpha = 1_000_000
    budget = 3_000_000_000
    ctrl = AdaptiveChunkController(
        safe_bytes=10_000_000_000,
        initial_chunk=64,
        worst_case_expert_bytes_per_token=per_token,
        max_expert_fetch_bytes=pool_bytes,
    )
    proposal = ctrl._safe_chunk_proposal(alpha, budget)
    assert proposal == int((budget - pool_bytes) / alpha)
    assert alpha * proposal + min(per_token * proposal, pool_bytes) <= budget

    # Below the 32-token saturation point, this is exactly the historical
    # worst-case per-token equation, not an expected-value approximation.
    small_budget = (alpha + per_token) * 17
    assert ctrl._safe_chunk_proposal(alpha, small_budget) == 17


def _run_all() -> None:
    tests = [
        test_kv_telemetry_not_double_counted,
        test_three_bad_chunks_freeze_at_reduced_minimum,
        test_padded_fit_covers_every_observation,
        test_growing_kv_overshoot_is_detected_and_shrinks_not_stays_wrong,
        test_moe_routing_spike_does_not_break_safety_or_get_masked,
        test_bad_first_observation_does_not_permanently_poison_growth,
        test_expert_fetch_noise_does_not_collapse_chunk_size_when_residualized,
        test_finite_expert_pool_stops_charging_for_impossible_new_experts,
        test_escalate_growth_cap_converges_faster_without_overshooting,
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
