"""F93 real-weight end-to-end generate() for Kimi K2.5.

The other test_f93_k25_realweight_*_oracle.py files validate one MLA layer
and one MoE layer in isolation against the real modeling_deepseek.py. This
runs the actual production path instead: a real `StreamingEngine.generate()`
call through the full 61-layer stack (dense + MLA/MoE layers, real
INT4-dequantized expert weights via `WeightStore.fetch()`, real disk-tier
paging for the 554GB checkpoint against a small resident cache budget) --
the same kind of gate `tests/test_kimi_linear_smoke.py::
test_real_engine_generate_end_to_end` added for Kimi Linear. No
transformers-oracle comparison is attempted here (that would need the full
checkpoint loaded in real PyTorch, ~1TB+ in bf16 -- infeasible on this
machine's 16GB regardless of which packages are installed); this is a
coherence/plumbing gate; a prior session's note recorded a real end-to-end
K2.5 HTTP request failing on memory pressure, so this also stands as the
first confirmation that path can complete at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K2.5"
_model_skip = pytest.mark.skipif(
    not (MODEL_DIR / "modeling_deepseek.py").exists(),
    reason="Kimi-K2.5's real checkpoint is not available locally "
           "(a real ~554GB model, not fetched in CI)",
)


@_model_skip
def test_real_engine_generate_end_to_end():
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    rc = RuntimeConfig(
        prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(MODEL_DIR), rc)
    try:
        result = engine.generate(
            "The capital of France is", max_tokens=3,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    assert "Paris" in result["text"], result["text"]
    stats = result["path_stats"]
    assert stats["weight_cache_misses"] > 0
    assert stats["expert_cache_misses"] > 0
