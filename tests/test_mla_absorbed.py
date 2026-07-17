"""F34 regression: MLA weight absorption during decode
(runtime/glm.py's `mla_absorbed` branch of `_mla_attention`) must produce
the SAME greedy token stream as the naive expand-then-attend path. Floating-
point association changes (the doc's own note), so bit-identical logits are
NOT the gate — greedy-token identity is, same standard as every other
lossless technique in this codebase. Uses the F65 fixture (no NAS, no real
GLM weights, sub-second).

  .venv/bin/python tests/test_mla_absorbed.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def _generate(prompt: str, max_tokens: int, absorbed: bool):
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
        mla_absorbed_decode=absorbed))
    result = eng.generate(prompt, max_tokens)
    eng.close()
    return result["tokens"]


def test_absorbed_matches_naive_greedy_tokens():
    _ensure_fixture()
    prompt = "Hi there, how are you today my friend"
    naive = _generate(prompt, 8, absorbed=False)
    absorbed = _generate(prompt, 8, absorbed=True)
    assert naive == absorbed, f"absorbed decode diverged: {absorbed} != {naive}"


def test_absorbed_matches_naive_across_multiple_prompts():
    """A few different prompts/lengths, to reduce the chance the first test
    just got lucky on one particular sequence of accept/argmax decisions."""
    _ensure_fixture()
    prompts = [
        ("The quick brown fox jumps", 6),
        ("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z", 10),
        ("Hi", 12),
    ]
    for prompt, n in prompts:
        naive = _generate(prompt, n, absorbed=False)
        absorbed = _generate(prompt, n, absorbed=True)
        assert naive == absorbed, \
            f"prompt {prompt!r}: absorbed diverged: {absorbed} != {naive}"


def test_absorbed_matches_naive_past_dsa_gather_threshold():
    """The fixture's index_topk=32 — a prompt+generation exceeding that
    triggers DSA's sparse gather (mx.take reducing lat_all/c_all/kr_all to
    the selected subset) BEFORE the absorbed math runs. The absorbed path
    must still match the naive path in that regime, not just the dense
    (S<=32) one exercised by the shorter prompts above."""
    _ensure_fixture()
    # The fixture tokenizer is byte-level, so this is exactly 40 positions:
    # enough to enter sparse gather without accidentally turning the intended
    # near-boundary probe into a 269-position numerical-stability stress test.
    long_prompt = "a" * 40  # > index_topk=32 tokens
    naive = _generate(long_prompt, 10, absorbed=False)
    absorbed = _generate(long_prompt, 10, absorbed=True)
    assert naive == absorbed, f"post-gather absorbed diverged: {absorbed} != {naive}"


def test_absorbed_flag_off_by_default():
    """mla_absorbed_decode defaults to False — must not change behavior for
    any existing caller that doesn't opt in."""
    from runtime.engine import RuntimeConfig

    assert RuntimeConfig().mla_absorbed_decode is False


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
