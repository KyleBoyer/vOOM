"""F32 regression: forced accept-none/partial/accept-all rollback boundary
tests on GLM's actual speculative MTP path, using the F65 architecture-
faithful tiny fixture (tests/fixtures/build_glm_fixture.py). No NAS, no real
GLM weights — runs in well under a second.

Requires the fixture to exist: run
  .venv/bin/python tests/fixtures/build_glm_fixture.py
first if models/glm-fixture-tiny/ is missing.

  .venv/bin/python tests/test_f32_rollback.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def _true_greedy(prompt: str, max_tokens: int):
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True))
    ref = eng.generate(prompt, max_tokens)
    eng.close()
    return ref["tokens"]


def _forced_boundary(n_match: int, k: int, prompt: str, max_tokens: int, true_tokens: list[int]):
    """Force MTPDrafter to propose n_match tokens matching the TRUE greedy
    continuation, then deliberately-wrong tokens for the rest. The emitted
    stream must equal true_tokens regardless — this is what F32 guarantees
    (target verification always wins, draft correctness only affects speed)."""
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True))
    dec = SpeculativeDecoder(eng, "mtp", k=k)
    call_count = [0]

    def forced_draft(h_last, last_token, kk, mtp_kv, offset):
        # advance the REAL MTP state (rollback correctness depends on this),
        # discard its predicted tokens, substitute a constructed sequence
        dec.mtp.__class__.draft_tokens(dec.mtp, h_last, last_token, kk, mtp_kv, offset)
        idx = call_count[0]
        call_count[0] += 1
        start = idx + 1
        matched = true_tokens[start:start + min(n_match, kk)]
        wrong = [(t + 12345) % 49000 for t in true_tokens[start + len(matched): start + kk]]
        return (matched + wrong)[:kk]

    dec.mtp.draft_tokens = forced_draft
    result = dec.generate(prompt, max_tokens=max_tokens)
    eng.close()
    return result["tokens"]


def test_rollback_accept_none():
    _ensure_fixture()
    prompt, max_tokens, k = "Hi", 6, 3
    truth = _true_greedy(prompt, max_tokens)
    emitted = _forced_boundary(0, k, prompt, max_tokens, truth)
    assert emitted == truth, f"accept-none: {emitted} != {truth}"


def test_rollback_partial_one():
    _ensure_fixture()
    prompt, max_tokens, k = "Hi", 6, 3
    truth = _true_greedy(prompt, max_tokens)
    emitted = _forced_boundary(1, k, prompt, max_tokens, truth)
    assert emitted == truth, f"partial(1): {emitted} != {truth}"


def test_rollback_partial_two():
    _ensure_fixture()
    prompt, max_tokens, k = "Hi", 6, 3
    truth = _true_greedy(prompt, max_tokens)
    emitted = _forced_boundary(2, k, prompt, max_tokens, truth)
    assert emitted == truth, f"partial(2): {emitted} != {truth}"


def test_rollback_accept_all():
    _ensure_fixture()
    prompt, max_tokens, k = "Hi", 6, 3
    truth = _true_greedy(prompt, max_tokens)
    emitted = _forced_boundary(3, k, prompt, max_tokens, truth)
    assert emitted == truth, f"accept-all: {emitted} != {truth}"


def test_rollback_above_dsa_threshold_is_quarantined():
    """MTP speculative decoding above a model's index_topk is deliberately
    refused (runtime/speculative.py), not silently run through an unproven
    path -- see the RuntimeError message for exactly what's missing (the
    released dynamic DSA rule for MTP: full indexer on draft step 0, reuse
    on steps 1+; plus rollback-state validation).

    This test previously exercised generation ABOVE that threshold directly
    (checking that rejected verify lanes also vanished from DSAState.k_idx),
    which is exactly the scenario the quarantine now blocks before any
    rollback logic runs. Once that dynamic rule and its rollback state are
    implemented and pass their own strict oracle validation, this test
    should go back to asserting a real accept-none rollback round like
    test_rollback_accept_none above -- not just that the guard fires.
    """
    _ensure_fixture()
    prompt = ("alpha beta gamma delta epsilon " * 12) + "answer:"
    max_tokens, k = 5, 3
    truth = _true_greedy(prompt, max_tokens)
    try:
        _forced_boundary(0, k, prompt, max_tokens, truth)
    except RuntimeError as exc:
        assert "quarantined" in str(exc), f"unexpected RuntimeError: {exc}"
    else:
        raise AssertionError(
            "expected MTP above index_topk to raise the quarantine RuntimeError; "
            "it did not -- either the guard was removed, or the dynamic DSA rule "
            "was actually implemented and this test needs to go back to checking "
            "real rollback correctness instead of the guard"
        )


def test_mtp_layer_derives_from_config():
    """F65 refactor: MTP layer index must come from config.num_hidden_layers,
    not the hardcoded 78 (which would break on any non-released-shape model,
    including this fixture)."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.speculative import SpeculativeDecoder

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(max_weight_cache_mb=200))
    dec = SpeculativeDecoder(eng, "mtp", k=2)
    assert dec.mtp.mtp_layer == eng.cfg.num_hidden_layers == 4
    eng.close()


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
