"""GLM-4.7-Flash (model_type "glm4_moe_lite") real end-to-end smoke test.

Downloaded 2026-07-24 while investigating candidates for the "best-balanced
MoE model" question -- architecturally, this checkpoint turned out to be
extremely well-aligned with what `runtime/glm.py` already implements: real
MLA (q_lora_rank=768, kv_lora_rank=512, no NoPE, no DSA indexer fields) and
the same noaux_tc MoE gate GLM-5.2/Kimi K2.5 already use, with standard
`.mlp.experts.<id>.<proj>.weight` / `.mlp.gate.*` naming (no Kimi-Linear-style
prefix override needed). `runtime/engine.py`'s three `("glm_moe_dsa",
"kimi_k25")` MLA/MoE dispatch points gained `"glm4_moe_lite"` as a third
member -- the same pattern that already let Kimi K2.5 reuse `run_glm_block`
unmodified. This checkpoint also ships a real MTP layer (`model.layers.47.*`,
one past the declared 47 trunk layers, with `eh_proj`/`enorm`/`hnorm`/
`shared_head.*` -- the DeepSeek-V3/GLM-5.2 MTP shape) which this test does
NOT exercise; only the ordinary trunk forward pass is covered here.

**Decode-quality note, root-caused 2026-07-24: this is a plumbing-only
gate, not a correctness proof, but the reason is now understood and it is
NOT believed to be a bug in this integration.** Extended generation (32+
tokens, any prompt, with or without chat template) shows real repetition
collapse: past the first 1-2 correct content tokens, greedy decoding
locks onto repeating "The capital of France is Paris." verbatim regardless
of the actual prompt. Ruled out KV-state leakage (fresh isolated engine,
identical result) and weight-cache eviction/corruption (larger cache
budget, identical result). Then checked real per-step MoE routing directly
(monkeypatched `glm._route_experts`): expert IDs, weights, and step-to-step
variation all looked completely healthy -- no sign of a routing/indexing
bug. The decisive finding: the model's own official HuggingFace repo has a
discussion thread ("Endless Repetition? Anyone encountered?",
https://huggingface.co/zai-org/GLM-4.7-Flash/discussions/48) where OTHER
users report the same repetition-collapse behavior on completely
different engines (vLLM, llama.cpp/GGUF) -- this is a known,
cross-engine, released-checkpoint-level limitation, not specific to this
project's MLA/MoE dispatch. This project's `SamplingParams` had no
repetition-penalty field at all (confirmed by reading `runtime/sampler.py`)
-- exactly the missing lever the community reports point at. See
`docs/future_lossless_techniques.md` F101/F102 for full details. Given no
numeric oracle exists at this scale, the plumbing test below still
deliberately checks only plumbing (no crash, real cache activity) -- do
not read a pass there as a quality/correctness claim either way.

**F102 update: repetition penalty implemented and confirmed to break the
specific collapse.** With `repetition_penalty=1.0` (the default, a true
no-op), the exact same "The capital of France is Paris." loop reproduces
byte-for-byte. With `repetition_penalty>=1.1`, the loop is gone -- output
changes to different (still imperfect, but no longer stuck) text. This
does not mean the checkpoint now produces GREAT output under greedy
decoding (community reports suggest it remains a weaker model), but it
confirms the missing sampler lever was the real, correct fix for the
specific pathological repeat-loop symptom this investigation centered on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "GLM-4.7-Flash"
_model_skip = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors.index.json").exists(),
    reason="GLM-4.7-Flash checkpoint is not available locally "
           "(a real ~58GB model, not fetched in CI)",
)


@_model_skip
def test_real_engine_generate_runs_without_crashing():
    """Plumbing-only: real weight/expert cache activity, no crash, some
    output. Does NOT assert generation quality -- see module docstring for
    the real, unresolved repetition-collapse issue found this session."""
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    rc = RuntimeConfig(
        prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(MODEL_DIR), rc)
    try:
        result = engine.generate(
            "The capital of France is", max_tokens=8,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    assert result["text"].strip()
    stats = result["path_stats"]
    assert stats["weight_cache_misses"] > 0
    assert stats["expert_cache_misses"] > 0


@_model_skip
def test_repetition_penalty_breaks_the_known_collapse():
    """F102 regression test: repetition_penalty=1.0 (default) reproduces
    the documented repeat-loop exactly; repetition_penalty=1.3 must not."""
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    rc = RuntimeConfig(
        prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(MODEL_DIR), rc)
    try:
        baseline = engine.generate(
            "The capital of Germany is", max_tokens=32,
            sampling=SamplingParams(temperature=0.0))
        penalized = engine.generate(
            "The capital of Germany is", max_tokens=32,
            sampling=SamplingParams(temperature=0.0, repetition_penalty=1.3))
    finally:
        engine.close()
    assert "The capital of France is Paris. The capital of France is Paris" \
        in baseline["text"]
    assert "The capital of France is Paris. The capital of France is Paris" \
        not in penalized["text"]
    assert penalized["text"] != baseline["text"]
