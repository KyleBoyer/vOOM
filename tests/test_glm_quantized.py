"""End-to-end gates for the lossy GLM decode side-quest."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture() -> None:
    from tests.fixtures.build_glm_fixture import build, is_current

    if not is_current(FIXTURE):
        build(FIXTURE)


def _quantized_decode(decode_batch: int):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.quant import QTensor

    engine = StreamingEngine(str(FIXTURE), RuntimeConfig(
        max_weight_cache_mb=200,
        pin_lm_head=True,
        quant_bits=4,
        # The production default deliberately leaves tiny projections alone.
        # This fixture opts them in so router/indexer/expert QTensor branches
        # are genuinely exercised despite its 64-wide hidden state.
        quant_min_dim=0,
        expert_fetch_batch=1,
        decode_expert_fetch_batch=decode_batch,
        governor=False,
    ))
    try:
        prompt = engine.tokenizer.encode("quantized GLM parity proof").ids
        kv = engine.new_kv()
        logits = engine.forward_tokens(prompt, kv)
        tokens = []
        decode_logits = []
        for _ in range(6):
            token = int(mx.argmax(logits[0, -1]))
            tokens.append(token)
            logits = engine.forward_tokens([token], kv)
            decode_logits.append(mx.array(logits))
        mx.eval(decode_logits)

        quantized_names = {
            name
            for page in engine.cache._pages.values()
            for name, value in page.tensors.items()
            if isinstance(value, QTensor)
        }
        return (
            tokens,
            decode_logits,
            engine._expert_compute_batches,
            engine._max_experts_per_compute_batch,
            quantized_names,
        )
    finally:
        engine.close()


def test_glm_q4_decode_batch_eight_is_logit_and_token_identical_to_one():
    """The fast default changes synchronization only, not expert order/math."""
    _ensure_fixture()
    one = _quantized_decode(1)
    eight = _quantized_decode(8)

    assert one[0] == eight[0]
    assert all(mx.array_equal(a, b) for a, b in zip(one[1], eight[1]))
    assert eight[2] < one[2]
    assert one[3] == 1
    assert eight[3] == 2  # fixture top-k; real GLM's corresponding bound is 8

    # Prove the QTensor-aware branches added for standard/prequantized GLM are
    # active, rather than accidentally comparing two BF16 executions.
    assert any(name.endswith(".mlp.gate.weight") for name in eight[4])
    assert any(name.endswith(".indexer.weights_proj.weight") for name in eight[4])
    assert any(".mlp.experts." in name for name in eight[4])
    assert "lm_head.weight" in eight[4]


def test_lossy_runtime_modes_have_distinct_prompt_cache_fingerprints():
    """Persisted logits/KV must not cross arithmetic-mode boundaries."""
    _ensure_fixture()
    from runtime.engine import RuntimeConfig, StreamingEngine

    def fingerprint(**overrides):
        args = dict(
            max_weight_cache_mb=200,
            quant_bits=4,
            quant_min_dim=0,
            expert_fetch_batch=1,
            governor=False,
        )
        args.update(overrides)
        engine = StreamingEngine(str(FIXTURE), RuntimeConfig(**args))
        try:
            return engine._get_kv_fingerprint()
        finally:
            engine.close()

    base = fingerprint()
    assert fingerprint(quant_mode="mxfp4", quant_group_size=32) != base
    assert fingerprint(decode_expert_fetch_batch=8) != base
    assert fingerprint(quantize_tied_lm_head=True) != base
    assert fingerprint(resident_fast_decode=True) != base
    assert fingerprint(fused_swiglu=True) != base
    assert fingerprint(prefill_last_token_separate=True) != base
