"""F116 follow-on: speculative-decoding verify path for Kimi Linear.

`Engine.forward_tokens_serial_positions`'s `kimi_family` branch (attention
and MLP/MoE run one position at a time, in order, reusing
`_kimi_linear_attention_residual`/`_kimi_linear_mlp_residual`) has existed
since F113's Kimi K3-readiness pass, but `runtime/speculative.py`'s
`SpeculativeDecoder` never exempted `kimi_linear` from its MoE-verify-
unsafe guard the way `glm_moe_dsa`/`kimi_k25`/`glm4_moe_lite` (and now
`kimi_k3`, see F128) already are -- a real, pre-existing gap noted but
deliberately not investigated during the kimi_k3 pass. This closes it: the
SAME proof glm_family's exemption cites ("verified byte-identical... on
the real checkpoint") applied here, on the real Kimi-Linear-48B-A3B-
Instruct checkpoint, comparing `forward_tokens_serial_positions` against
`forward_tokens` for a real multi-position window across the FULL 27-layer
stack (both the KDA and MLA/MoE layers, unlike the truncated 3-layer scope
K3's own equivalence tests needed given its much larger real size).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-Linear-48B-A3B-Instruct"
_MODEL_AVAILABLE = (MODEL_DIR / "config.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-Linear-48B-A3B-Instruct is not available locally "
           "(a real ~98GB model, not fetched in CI)",
)


@_model_skip
def test_forward_tokens_serial_positions_matches_forward_tokens():
    from runtime.engine import RuntimeConfig, StreamingEngine

    def _rc():
        # F74-v2 safety default, applied automatically by server.py for
        # real requests but not by a bare RuntimeConfig() -- see F128's
        # own real measurement of what omitting it costs.
        return RuntimeConfig(
            prefill_chunk_size=8, min_weight_cache_mb=200, max_weight_cache_mb=6000,
            expert_fetch_batch=1)

    tokens = [3, 7, 11, 13, 17, 19]

    # F128-style separate engines for clean transient-learning state (not
    # cross-contaminated by whichever call ran first), but NOT held open
    # concurrently -- two full un-truncated 27-layer weight caches alive at
    # once measurably exceeded this machine's real available memory (a
    # resource-tuning fact, not a logic bug); K3's own equivalence tests
    # got away with concurrent engines only because num_hidden_layers was
    # truncated to 3 there.
    engine_batched = StreamingEngine(str(MODEL_DIR), _rc())
    kv_batched = engine_batched.new_kv()
    batched_logits = engine_batched.forward_tokens(tokens, kv_batched)
    mx.eval(batched_logits)
    engine_batched.close()
    # close() stops background threads/governor polling but does not drop
    # the weight cache itself -- the tensors stay resident as long as
    # anything still references this engine object. Explicitly drop it and
    # force MLX to release the underlying Metal buffers before the second
    # engine's own construction/reservations run, or its baseline "active"
    # memory includes the first engine's entire cache (measured directly:
    # ~6.9GB active before engine_serial does anything at all).
    del engine_batched, kv_batched
    mx.clear_cache()

    engine_serial = StreamingEngine(str(MODEL_DIR), _rc())
    try:
        kv_serial = engine_serial.new_kv()
        serial_logits = engine_serial.forward_tokens_serial_positions(tokens, kv_serial)
        mx.eval(serial_logits)

        assert batched_logits.shape == serial_logits.shape
        max_diff = mx.max(mx.abs(
            batched_logits.astype(mx.float32) - serial_logits.astype(mx.float32)))
        mx.eval(max_diff)
        # Float32-rounding-only tolerance -- same reasoning as
        # tests/test_f128_k3_verify_positions_oracle.py: different matmul
        # batching (all positions in one chunk vs. one position at a
        # time), not a logic difference.
        assert max_diff.item() < 1e-3, (
            "forward_tokens_serial_positions and forward_tokens must "
            f"produce near-identical logits; max abs diff {max_diff.item()}")

        greedy_batched = mx.argmax(batched_logits, axis=-1)
        greedy_serial = mx.argmax(serial_logits, axis=-1)
        mx.eval(greedy_batched, greedy_serial)
        assert greedy_batched.tolist() == greedy_serial.tolist(), (
            "greedy argmax tokens must match exactly -- this is the "
            "property real speculative-decoding verification depends on, "
            "not just small float diffs")
    finally:
        engine_serial.close()
