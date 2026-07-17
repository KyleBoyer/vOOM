"""F68 regression: RuntimeConfig.adaptive_chunk_size (runtime/adaptive_chunk.py).

Learns a safe prefill chunk size ONLINE from observed peak-memory slope,
instead of trusting a fixed constant measured on a different model (4096
was measured on Qwen2.5-1.5B only — see docs/benchmark_results.md, "F60
chunked prefill AS a memory-transient fix"). It is intended as scheduling
only, but changed shapes can alter floating-point kernels. Token identity here is
an empirical regression gate; F33 block-output evidence is still required.

Uses the F65 architecture-faithful tiny GLM fixture (no NAS, sub-second).

  .venv/bin/python tests/test_adaptive_chunk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"
LONG_PROMPT = " ".join(["the quick brown fox jumps over the lazy dog"] * 10)


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def test_adaptive_and_fixed_chunking_produce_identical_tokens():
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    results = {}
    for adaptive in (False, True):
        kwargs = dict(max_weight_cache_mb=200, mla_compressed_kv=True, prefill_chunk_size=8)
        if adaptive:
            kwargs.update(adaptive_chunk_size=True, adaptive_chunk_safe_bytes=int(8.0e9))
        eng = StreamingEngine(str(FIXTURE), RuntimeConfig(**kwargs))
        result = eng.generate(LONG_PROMPT, max_tokens=4)
        results[adaptive] = result["tokens"]
        eng.close()
        mx.clear_cache()
    assert results[False] == results[True]


def test_controller_grows_chunk_size_after_green_streaks():
    """With a tiny, cheap model, the controller should learn it can afford
    a MUCH bigger chunk than the conservative starting point and grow
    toward it, not stay stuck at the initial size."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, mla_compressed_kv=True, prefill_chunk_size=8,
        adaptive_chunk_size=True,
    ))
    result = eng.generate(LONG_PROMPT, max_tokens=4)
    eng.close()
    mx.clear_cache()
    events = result["path_stats"]["adaptive_chunk_events"]
    assert result["path_stats"]["adaptive_chunk_dynamic_ceiling"] == 1
    assert result["path_stats"]["adaptive_chunk_safe_bytes_min"] > 0
    assert (result["path_stats"]["adaptive_chunk_safe_bytes_max"]
            >= result["path_stats"]["adaptive_chunk_safe_bytes_min"])
    assert any("GREEN" in e for e in events), f"expected at least one growth event, got: {events}"
    # fewer chunks than the fixed chunk=8 baseline would need for the same prompt
    assert result["path_stats"]["prefill_chunks"] < 11


def test_dead_band_ignores_a_proposal_close_to_the_current_chunk():
    """Unit-level, no model needed, deterministic: directly seeds the
    controller's history so its fitted envelope implies a proposed chunk
    size close to (but not exactly) the current one -- exactly the
    situation found live on real OLMoE (docs/benchmark_results.md, "OLMoE
    follow-up": routing-driven noise kept proposing slightly different
    chunk sizes near a settled value). The dead-band must recognize this
    as noise and leave the chunk size unchanged, rather than churning."""
    from runtime.adaptive_chunk import AdaptiveChunkController

    ctrl = AdaptiveChunkController(safe_bytes=int(8.0e9), initial_chunk=464, dead_band=0.2)
    active_before, kv_before = int(2.0e9), int(4.0e9)
    margin = ctrl.margin
    # kv_before is already part of active_before as far as MLX is concerned;
    # give it an intentionally huge independent value to catch double-counting.
    budget_target = int(8.0e9) - active_before - margin
    alpha_true = budget_target / 480  # implies a proposed chunk near 480, within 20% of 464
    # seed two history points at DIFFERENT chunk sizes so the fit has a
    # real (not degenerate) slope estimate
    ctrl._history = [(400, alpha_true * 400), (600, alpha_true * 600)]
    ctrl._green_streak = 1  # one observation away from a GREEN-streak decision
    ctrl.observe(
        chunk_size=464, peak=active_before + int(alpha_true * 464),
        active_before=active_before, kv_before=kv_before, governor_event=False)

    assert ctrl.events, "expected the GREEN-streak decision to fire and log something"
    assert "within dead_band" in ctrl.events[-1], ctrl.events[-1]
    assert ctrl.chunk == 464, "chunk size must stay unchanged when the proposal is within the dead band"


def test_controller_freezes_reduced_size_and_marks_size_one_unsafe():
    """Three bad chunks must not restore the original fixed size.

    This is a pure controller test of the exact failure sequence that the old
    engine handled unsafely by setting ``adaptive=None`` after three halvings.
    """
    from runtime.adaptive_chunk import AdaptiveChunkController

    ctrl = AdaptiveChunkController(safe_bytes=100, initial_chunk=4, margin_bytes=0)
    for chunk in (4, 2, 1):
        ctrl.observe(
            chunk_size=chunk, peak=101, active_before=0, kv_before=10_000,
            governor_event=False,
        )
    assert ctrl.failed
    assert ctrl.next_chunk_size() == 1
    assert ctrl.unsafe_at_minimum
    assert any("FROZEN" in event for event in ctrl.events)


def test_kv_telemetry_is_not_double_counted():
    """Changing the separate KV telemetry value cannot change a decision."""
    from runtime.adaptive_chunk import AdaptiveChunkController

    def run(kv_before):
        ctrl = AdaptiveChunkController(
            safe_bytes=10_000, initial_chunk=10, margin_bytes=0, dead_band=0.0
        )
        ctrl.observe(10, peak=2_000, active_before=1_000,
                     kv_before=kv_before, governor_event=False)
        ctrl.observe(20, peak=3_000, active_before=1_000,
                     kv_before=kv_before, governor_event=False)
        return ctrl.next_chunk_size(), list(ctrl.events)

    assert run(0) == run(9_000_000)


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
