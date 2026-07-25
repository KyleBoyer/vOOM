"""F113: real-checkpoint proof that forward_tokens_serial_positions() now
supports GLM-family targets (glm_moe_dsa/kimi_k25/glm4_moe_lite) correctly,
and that SpeculativeDecoder(draft="mtp") produces byte-identical output
against a real GLM-family checkpoint with real native MTP weights.

Background: SpeculativeDecoder's multi-position verify sweep used plain
forward_tokens() (batched multi-position GEMMs) for any MoE/hybrid target,
because forward_tokens_serial_positions() -- the numerically-exact,
one-position-at-a-time alternative -- explicitly refused them (its
layer_runner.run_block has no MoE/hybrid dispatch). A real end-to-end test
against GLM-4.7-Flash found this caused a REAL, reproducible divergence
from true sequential decode (this session's own F113 finding). Fixed by
giving forward_tokens_serial_positions a real GLM-aware per-position
dispatch (reusing runtime/glm.py's _glm_attention_residual/
_glm_mlp_residual, the same split F35 already uses), and routing
SpeculativeDecoder's verify sweep through it for GLM-family targets.

Two real-checkpoint proofs, matching this project's "greedy A/B,
byte-identical tokens" standard:

1. forward_tokens_serial_positions() vs true sequential forward_tokens()
   calls, one position at a time -- must be byte-identical. Uses the real
   Kimi-K2.5 checkpoint (554GB, shares GLM's exact MLA+MoE block code,
   available locally where a real GLM-5.2 checkpoint is not).
2. SpeculativeDecoder(draft="mtp") vs the plain target's own generate() --
   must be byte-identical. Uses the real GLM-4.7-Flash checkpoint (58GB,
   glm4_moe_lite, the only locally-available checkpoint with real native
   MTP weights -- K2.5 has num_nextn_predict_layers=0, none at all).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.engine import RuntimeConfig, StreamingEngine
from runtime.sampler import SamplingParams

_K25_DIR = Path(__file__).resolve().parent.parent / "models" / "Kimi-K2.5"
_GLM47_DIR = Path(__file__).resolve().parent.parent / "models" / "GLM-4.7-Flash"
_k25_skip = pytest.mark.skipif(
    not _K25_DIR.exists(), reason="real Kimi-K2.5 checkpoint not present on this machine")
_glm47_skip = pytest.mark.skipif(
    not _GLM47_DIR.exists(), reason="real GLM-4.7-Flash checkpoint not present on this machine")

_PROMPT = "The capital of France is"


@_k25_skip
def test_forward_tokens_serial_positions_matches_true_sequential_decode_for_glm_family():
    import mlx.core as mx

    rc = RuntimeConfig(prefill_chunk_size=256, min_weight_cache_mb=200, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(_K25_DIR), rc)
    try:
        ids = list(engine.tokenizer.encode(_PROMPT).ids)

        kv_probe = engine.new_kv()
        logits = engine.forward_tokens(ids, kv_probe)
        first = int(mx.argmax(logits[-1]))
        continuation = [first]
        cur = first
        for _ in range(3):
            step_logits = engine.forward_tokens([cur], kv_probe)
            cur = int(mx.argmax(step_logits[-1]))
            continuation.append(cur)
        del kv_probe

        verify_tokens = [ids[-1]] + continuation[:-1]

        kv_serial = engine.new_kv()
        engine.forward_tokens(ids[:-1], kv_serial)
        serial_logits = engine.forward_tokens_serial_positions(verify_tokens, kv_serial)
        mx.eval(serial_logits)

        kv_seq = engine.new_kv()
        engine.forward_tokens(ids[:-1], kv_seq)
        seq_logits_list = []
        for tok in verify_tokens:
            step_logits = engine.forward_tokens([tok], kv_seq)
            mx.eval(step_logits)
            seq_logits_list.append(step_logits[-1])

        for i in range(serial_logits.shape[0]):
            diff = float(mx.max(mx.abs(
                serial_logits[i].astype(mx.float32)
                - seq_logits_list[i].astype(mx.float32))))
            assert diff == 0.0, (
                f"position {i}: forward_tokens_serial_positions must be "
                f"byte-identical to true sequential decode for GLM-family "
                f"targets, got max abs diff {diff}"
            )
    finally:
        engine.close()


@_glm47_skip
def test_glm_native_mtp_matches_plain_target_byte_identical():
    from runtime.speculative import SpeculativeDecoder

    def _run(use_mtp: bool):
        rc = RuntimeConfig(
            prefill_chunk_size=512, min_weight_cache_mb=200, max_weight_cache_mb=6000)
        engine = StreamingEngine(str(_GLM47_DIR), rc)
        driver = SpeculativeDecoder(engine, draft="mtp") if use_mtp else engine
        try:
            if use_mtp:
                result = driver.generate(_PROMPT, max_tokens=8)
            else:
                result = driver.generate(
                    _PROMPT, max_tokens=8, sampling=SamplingParams(temperature=0.0))
        finally:
            engine.close()
        return result

    baseline = _run(use_mtp=False)
    mtp = _run(use_mtp=True)
    assert mtp["tokens"] == baseline["tokens"], (
        "SpeculativeDecoder(draft='mtp') must produce byte-identical greedy "
        "output to the plain target engine now that forward_tokens_"
        "serial_positions supports GLM-family targets"
    )
    assert mtp["text"] == baseline["text"]
