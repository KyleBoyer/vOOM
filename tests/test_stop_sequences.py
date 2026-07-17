"""Regression test for runtime/engine.py's `stop` parameter (OpenAI-style
stop sequences) — added 2026-07-13 alongside HTTP server usage-note work.
Uses local SmolLM2-135M (no download, fast).

  .venv/bin/python tests/test_stop_sequences.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "models/SmolLM2-135M"
PROMPT = "The capital of France is"


def test_no_stop_matches_default_behavior():
    """Passing stop=None (or omitting it) must produce byte-identical output
    to the pre-existing behavior — this is the backward-compatibility half
    of the change."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    r1 = eng.generate(PROMPT, 12)
    r2 = eng.generate(PROMPT, 12, stop=None)
    r3 = eng.generate(PROMPT, 12, stop=[])
    eng.close()
    assert r1["tokens"] == r2["tokens"] == r3["tokens"]
    assert r1.get("stopped") is False and r2.get("stopped") is False


def test_stop_string_truncates_output():
    """Find a real multi-token continuation, then use its own middle as a
    stop string — the returned text must end exactly before it, and
    `stopped` must be True."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    baseline = eng.generate(PROMPT, 16)
    full_text = baseline["text"]
    assert len(full_text) > 4, "need a long-enough baseline to pick a stop string from"
    stop_str = full_text[2:5]  # an arbitrary substring guaranteed to occur

    result = eng.generate(PROMPT, 16, stop=[stop_str])
    eng.close()
    assert result["stopped"] is True
    assert stop_str not in result["text"], \
        f"stop string {stop_str!r} leaked into output {result['text']!r}"
    assert result["text"] == full_text[:full_text.find(stop_str)]


def test_stop_never_fires_if_absent():
    """A stop string that never appears must not affect generation at all."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    baseline = eng.generate(PROMPT, 12)
    result = eng.generate(PROMPT, 12, stop=["zzz_never_appears_zzz"])
    eng.close()
    assert result["tokens"] == baseline["tokens"]
    assert result["stopped"] is False


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
