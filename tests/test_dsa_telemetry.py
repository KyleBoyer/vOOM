"""F69 regression: proof-carrying DSA execution telemetry
(DSAState.stats / result["path_stats"]["dsa_*"]).

Configuring a feature is not evidence it ran -- this session's own real-GLM
validation script had a short prompt that silently never entered the F60
chunk loop despite its header calling chunking "tested" (caught by BRIEF 0,
docs/benchmark_results.md). These counters let a caller assert e.g.
`dsa_sparse_selects>0` and catch the same class of silent no-op instead of
mistaking "the model ran" for "the mechanism under test ran".

Uses the F65 architecture-faithful tiny GLM fixture (no NAS, sub-second).

  .venv/bin/python tests/test_dsa_telemetry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def test_dsa_counters_absent_for_non_dsa_model():
    """A model with no DSA indexer (e.g. a plain dense/GQA model) must not
    report dsa_* keys at all -- they're conditional on kv.dsa existing."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine("models/SmolLM2-135M", RuntimeConfig())
    result = eng.generate("The capital of France is", max_tokens=5)
    eng.close()
    mx.clear_cache()
    for key in ("dsa_observations", "dsa_sparse_selects", "dsa_shared_reuses"):
        assert key not in result["path_stats"], f"{key} should not appear for a non-DSA model"


def test_dsa_counters_nonzero_on_real_glm_fixture():
    """A real GLM-architecture forward pass (even the tiny fixture) must
    show non-zero DSA activity -- proof the indexer genuinely ran, not
    just that a GLM-shaped config was loaded."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    prompt = " ".join(["the quick brown fox jumps over the lazy dog"] * 10)
    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(max_weight_cache_mb=200, mla_compressed_kv=True))
    result = eng.generate(prompt, max_tokens=4)
    eng.close()
    mx.clear_cache()
    stats = result["path_stats"]
    assert stats["dsa_observations"] > 0
    assert stats["dsa_sparse_selects"] > 0
    assert stats["dsa_shared_reuses"] > 0


def test_short_prompt_shows_zero_sparse_selects_not_silently_passing():
    """This IS the exact gap F69 was built to catch: a prompt far shorter
    than index_topk must show dsa_sparse_selects==0 (dense fallback, S <=
    topk every time) even though DSA is fully configured and the model
    genuinely ran -- proving the counter distinguishes "ran" from "ran the
    sparse mechanism", not just always reporting non-zero."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(max_weight_cache_mb=200, mla_compressed_kv=True))
    result = eng.generate("hi", max_tokens=2)  # far below index_topk=32
    eng.close()
    mx.clear_cache()
    stats = result["path_stats"]
    assert stats["dsa_observations"] > 0  # the indexer still observed every position
    assert stats["dsa_sparse_selects"] == 0  # but never needed to select sparsely


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
