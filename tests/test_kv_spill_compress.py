"""F07 regression: zstd-compressed KV page spilling (RuntimeConfig.
kv_spill_compress, opt-in) must round-trip byte-identical to the uncompressed
safetensors spill path, and must never change generated token IDs end-to-end.

  .venv/bin/python tests/test_kv_spill_compress.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from runtime.kv_paged import PagedKVCache

ROOT = Path(__file__).resolve().parent.parent


def test_compressed_page_round_trips_byte_identical():
    spill_dir = "/tmp/test_kv_spill_compress_unit"
    shutil.rmtree(spill_dir, ignore_errors=True)
    try:
        kv = PagedKVCache(num_layers=2, max_bytes=1, spill_dir=spill_dir,
                          page_positions=4, resident_pages=0, compress_spill=True)
        mx.random.seed(0)
        ref_k, ref_v = [], []
        for layer in range(2):
            k = mx.random.normal((1, 2, 4, 8)).astype(mx.bfloat16)
            v = mx.random.normal((1, 2, 4, 8)).astype(mx.bfloat16)
            mx.eval(k, v)
            ref_k.append(k)
            ref_v.append(v)
            kv.update(layer, k, v)
        # one more update forces the budget check + spill of the closed page
        for layer in range(2):
            k2 = mx.random.normal((1, 2, 1, 8)).astype(mx.bfloat16)
            v2 = mx.random.normal((1, 2, 1, 8)).astype(mx.bfloat16)
            mx.eval(k2, v2)
            kv.update(layer, k2, v2)

        assert kv.stats.spills > 0, "test setup didn't actually force a spill"
        page = kv._pages[0][0]
        assert not page.resident
        lk, lv = page.load()
        assert lk.dtype == mx.bfloat16
        assert bool(mx.array_equal(lk, ref_k[0]).item())
        assert bool(mx.array_equal(lv, ref_v[0]).item())
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)


def test_compressed_and_uncompressed_spill_produce_identical_tokens():
    """End-to-end: same model/prompt/budget, compress on vs off -> identical
    generated token IDs. This is a pure byte-transform of the spilled bf16
    bits; it must never move a token."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    model = str(ROOT / "models" / "SmolLM2-135M")
    prompt = "Once upon a time in a small village there lived a"
    results = {}
    for compress in (False, True):
        spill_dir = f"/tmp/test_kv_spill_compress_e2e_{compress}"
        shutil.rmtree(spill_dir, ignore_errors=True)
        rc = RuntimeConfig(max_kv_mb=4, kv_page_positions=64,
                          kv_spill_dir=spill_dir, kv_spill_compress=compress)
        engine = StreamingEngine(model, rc)
        result = engine.generate(prompt, max_tokens=220)
        assert engine.last_kv.stats.spills > 0, "test setup didn't actually force a spill"
        results[compress] = result["tokens"]
        engine.close()
        mx.clear_cache()
        shutil.rmtree(spill_dir, ignore_errors=True)
    assert results[False] == results[True]


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
