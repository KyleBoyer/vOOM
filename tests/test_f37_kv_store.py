"""F37 durability regression: the prompt-KV store's save/load round-trip and
runtime-identity fingerprint invalidation. Uses the F65 fixture, no NAS, no
real GLM weights, sub-second.

  .venv/bin/python tests/test_f37_kv_store.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"
PROMPT = "Hi there, how are you today my friend"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current
    if not is_current(FIXTURE):
        build(FIXTURE)


def test_save_load_round_trip_no_tmp_leftovers():
    """A second engine pointed at the same prompt_kv_dir must reproduce the
    first engine's tokens exactly (exact-hit restores stored logits), and no
    *.tmp / *.tmp.safetensors files should remain after a clean save — this
    is the regression test for the mx.save_safetensors force-appends-
    ".safetensors" bug caught while writing this fix (a tmp name of
    "{key}.safetensors.tmp" silently became "{key}.safetensors.tmp.safetensors"
    on disk, so os.open(tmp_st) raised FileNotFoundError)."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    tmpdir = tempfile.mkdtemp(prefix="f37_test_")
    try:
        cfg = RuntimeConfig(max_weight_cache_mb=200, pin_lm_head=True,
                            mla_compressed_kv=True, prompt_kv_dir=tmpdir)
        eng1 = StreamingEngine(str(FIXTURE), cfg)
        r1 = eng1.generate(PROMPT, 5)
        eng1.close()

        eng2 = StreamingEngine(str(FIXTURE), cfg)
        r2 = eng2.generate(PROMPT, 5)
        eng2.close()

        assert r1["tokens"] == r2["tokens"], \
            f"save/load round-trip diverged: {r1['tokens']} != {r2['tokens']}"

        import os
        files = os.listdir(tmpdir)
        assert not any(".tmp" in f for f in files), f"leftover tmp file(s): {files}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_fingerprint_changes_with_runtime_code():
    """A runtime source change must change model_fingerprint()'s output, so
    stale-code-computed cache entries are never silently reused. Uses an injected
    temporary source root; tests must never rewrite a live production module."""
    from runtime.kv_store import _runtime_fingerprint

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "engine.py"
        source.write_text("x = 1\n")
        fp_before = _runtime_fingerprint(root)
        source.write_text("x = 2\n")
        fp_touched = _runtime_fingerprint(root)
        source.write_text("x = 1\n")
        fp_reverted = _runtime_fingerprint(root)

    assert fp_before != fp_touched, "fingerprint did not change when runtime code changed"
    assert fp_before == fp_reverted, "fingerprint did not restore after reverting"


def test_corrupt_payload_falls_back_instead_of_crashing():
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    tmpdir = tempfile.mkdtemp(prefix="f37_corrupt_")
    try:
        cfg = RuntimeConfig(max_weight_cache_mb=200, pin_lm_head=True,
                            mla_compressed_kv=True, prompt_kv_dir=tmpdir)
        eng1 = StreamingEngine(str(FIXTURE), cfg)
        truth = eng1.generate(PROMPT, 3)["tokens"]
        eng1.close()
        for payload in Path(tmpdir).glob("*.safetensors"):
            payload.write_bytes(b"torn")

        eng2 = StreamingEngine(str(FIXTURE), cfg)
        got = eng2.generate(PROMPT, 3)["tokens"]
        eng2.close()
        assert got == truth
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_min_tokens_skips_short_prompt_store_without_changing_tokens():
    """Short requests should not pay a disk scan and two snapshots when the
    configured admission threshold says they are cheaper to recompute."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    tmpdir = tempfile.mkdtemp(prefix="f37_min_tokens_")
    try:
        baseline = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True,
            mla_compressed_kv=True))
        truth = baseline.generate(PROMPT, 3)["tokens"]
        baseline.close()

        gated = StreamingEngine(str(FIXTURE), RuntimeConfig(
            max_weight_cache_mb=200, pin_lm_head=True,
            mla_compressed_kv=True, prompt_kv_dir=tmpdir,
            prompt_kv_min_tokens=10_000))
        result = gated.generate(PROMPT, 3)
        gated.close()

        assert result["tokens"] == truth
        assert result["path_stats"]["prompt_cache_eligible"] == 0
        assert result["path_stats"]["prompt_snapshot_write_s"] == 0
        assert result["path_stats"]["postgen_snapshot_write_s"] == 0
        assert not list(Path(tmpdir).iterdir())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_fingerprint_includes_arithmetic_mode():
    _ensure_fixture()
    from runtime.kv_store import model_fingerprint

    a = model_fingerprint(FIXTURE, True, arithmetic="abs0-head0-chunk0")
    b = model_fingerprint(FIXTURE, True, arithmetic="abs1-head0-chunk0")
    assert a != b


def test_v6_upgrade_sweeps_unusable_legacy_snapshots_once(tmp_path):
    from runtime.kv_store import PromptKVStore

    legacy_id = "a" * 64
    legacy_meta = tmp_path / f"{legacy_id}.json"
    legacy_payload = tmp_path / f"{legacy_id}.safetensors"
    legacy_meta.write_text("{}")
    legacy_payload.write_bytes(b"obsolete")
    unrelated = tmp_path / "operator-note.json"
    unrelated.write_text("keep")

    PromptKVStore(tmp_path, "upgrade-test", max_bytes=1_000_000)

    assert not legacy_meta.exists()
    assert not legacy_payload.exists()
    assert unrelated.read_text() == "keep"
    marker = tmp_path / ".f37-v6-legacy-swept.json"
    assert marker.exists()

    # The marker makes later constructions a no-op and never broadens the
    # deletion pattern to arbitrary JSON files.
    PromptKVStore(tmp_path, "upgrade-test", max_bytes=1_000_000)
    assert unrelated.read_text() == "keep"


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
