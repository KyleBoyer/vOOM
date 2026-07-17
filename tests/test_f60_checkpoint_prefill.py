"""F60 local correctness gate: chunked/checkpointed prefill must produce
token-identical output to a straight-through prefill, and a prefill that
resumes from a persisted mid-prompt checkpoint must match a fresh
straight-through run of the full (longer) prompt.

Uses the F65 architecture-faithful tiny GLM fixture (no NAS, no real GLM
weights, sub-second). This tests the CHECKPOINTING MECHANISM's correctness,
not real long-context (32K-1M) DSA/indexer behavior — that still requires
the real checkpoint over NAS per Goal 2.

  .venv/bin/python tests/test_f60_checkpoint_prefill.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"

# A prompt long enough to cross several checkpoint boundaries at ckpt=8.
LONG_PROMPT = "The quick brown fox jumps over the lazy dog while the " \
              "curious cat watches from atop the old wooden fence nearby"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def _straight_through(prompt: str, max_tokens: int) -> list[int]:
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True))
    ref = eng.generate(prompt, max_tokens)
    eng.close()
    return ref["tokens"]


def test_checkpointed_matches_straight_through():
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    truth = _straight_through(LONG_PROMPT, 6)

    tmpdir = tempfile.mkdtemp(prefix="f60_ckpt_")
    try:
        eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prompt_kv_dir=tmpdir, prefill_checkpoint_every=8))
        got = eng.generate(LONG_PROMPT, 6)["tokens"]
        eng.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert got == truth, f"checkpointed prefill diverged: {got} != {truth}"


def test_resume_from_partial_checkpoint():
    """Simulate a crash: prefill only a PREFIX of the prompt (checkpointing
    along the way), close the engine, then open a NEW engine on the same
    prompt_kv_dir and generate from the FULL prompt. The longest-prefix load
    should pick up the persisted mid-prompt checkpoint instead of redoing
    that work, and the final tokens must still match a fresh straight-through
    run of the full prompt."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.model_loader import WeightStore
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(FIXTURE / "tokenizer.json"))
    full_ids = tok.encode(LONG_PROMPT).ids
    prefix_len = len(full_ids) * 2 // 3  # crash partway through the prompt
    prefix_text = tok.decode(full_ids[:prefix_len])

    truth = _straight_through(LONG_PROMPT, 6)

    tmpdir = tempfile.mkdtemp(prefix="f60_resume_")
    try:
        cfg = RuntimeConfig(max_weight_cache_mb=200, pin_lm_head=True,
                            mla_compressed_kv=True, prompt_kv_dir=tmpdir,
                            prefill_checkpoint_every=8)
        eng1 = StreamingEngine(str(FIXTURE), cfg)
        eng1.generate(prefix_text, 1)  # "crashes" after only the prefix
        eng1.close()

        eng2 = StreamingEngine(str(FIXTURE), cfg)
        got = eng2.generate(LONG_PROMPT, 6)["tokens"]
        eng2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert got == truth, f"resumed prefill diverged: {got} != {truth}"


def test_uneven_checkpoint_boundary():
    """prompt length need not be a multiple of ckpt — the remainder must
    still be handled by the ordinary non-chunked tail path."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    truth = _straight_through(LONG_PROMPT, 4)
    tmpdir = tempfile.mkdtemp(prefix="f60_uneven_")
    try:
        eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True, mla_compressed_kv=True,
            prompt_kv_dir=tmpdir, prefill_checkpoint_every=7))  # 7 does not divide prompt length
        got = eng.generate(LONG_PROMPT, 4)["tokens"]
        eng.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert got == truth, f"uneven-boundary prefill diverged: {got} != {truth}"


def test_memory_chunking_does_not_write_each_chunk():
    """F60/F67 split: compute chunks bound Metal without O(n^2) snapshots."""
    _ensure_fixture()
    import tempfile
    from runtime.engine import RuntimeConfig, StreamingEngine

    with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent.parent)) as tmpdir:
        eng = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200,
            mla_compressed_kv=True,
            prompt_kv_dir=tmpdir,
            prefill_chunk_size=8,
            prefill_checkpoint_every=0,
        ))
        result = eng.generate(LONG_PROMPT, max_tokens=2)
        eng.close()
        assert result["path_stats"]["prefill_chunks"] > 0
        assert result["path_stats"]["prefill_checkpoints_saved"] == 0
        # Immutable segment manifests are expected and scale with the journal
        # deltas.  Only checkpoint manifests represent complete loadable
        # endpoints; their count must stay fixed rather than following the
        # number of compute chunks.
        assert len(list(Path(tmpdir).glob("*.ckpt.json"))) <= 2


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
