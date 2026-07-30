"""F128: AttnRes-aware speculative-decoding verify path for Kimi K3.

`Engine.forward_tokens_serial_positions` (used by speculative decoding's
verify step to avoid batched-GEMM divergence risk) now has a kimi_k3_family
branch: attention and MLP/MoE still run one position at a time, in order,
exactly like every other family it already supports, but the AttnRes
pre/post mixing (`_apply_attn_res`) is batched across all positions in the
verify window at once, since `block_residual`'s per-layer boundary
snapshot must span ALL positions, not one at a time (see
attn_res_wrap_layer's own docstring). This test checks that path against
`forward_tokens` (the ordinary batched `_sweep`/`run_kimi_k3_block` path)
for the first 3 real layers, mirroring test_f128_k3_layer_stationary_
oracle.py's methodology and float32-rounding tolerance reasoning exactly
(same real reason: different matmul batching, not a logic difference).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
_MODEL_AVAILABLE = (MODEL_DIR / "model.safetensors.index.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 is not available locally (a real ~1.4TB model, not fetched in CI)",
)


@_model_skip
def test_forward_tokens_serial_positions_matches_forward_tokens_for_first_three_layers():
    from runtime.engine import RuntimeConfig, StreamingEngine

    def _rc():
        # F128: expert_fetch_batch=1 -- constructing RuntimeConfig directly
        # bypasses server.py's model_type-specific "fetch experts one at a
        # time" safety default (F74-v2, normally applied automatically for
        # kimi_k3/kimi_linear/glm_moe_dsa/... in real request handling).
        # Without it, _iter_expert_batches fetches all num_experts_per_tok
        # (16) routed experts as ONE unbounded batch; the MXFP4 dequant
        # path's several intermediate float32 arrays per weight then stay
        # alive simultaneously across all 16 experts in one lazy graph
        # before the single mx.eval() releases any of them, a real ~8.7GB
        # peak this test measured directly -- exactly the "16-22 GB false
        # demand" F74-v2 exists to prevent, just triggered here by this
        # test's own RuntimeConfig omission rather than a bug in the new
        # kimi_k3_family dispatch itself.
        return RuntimeConfig(
            prefill_chunk_size=4, min_weight_cache_mb=50, max_weight_cache_mb=1500,
            embed_rows=True, stream_lm_head=True, expert_fetch_batch=1)

    # F128: 2 positions, not 4 -- each of K3's real num_experts_per_tok=16
    # routes independently per position, so 4 largely-non-overlapping
    # positions can touch up to 64 distinct experts (~4-8GB dequantized)
    # simultaneously live in the weight cache across this one verify call,
    # a real measured MemoryError on this 16GB machine even with
    # expert_fetch_batch=1. 2 positions still exercises real multi-
    # position verification (the property this function's docstring and
    # this test both care about) at a scale this machine can hold.
    tokens = [3, 7]

    # F128: two SEPARATE engine instances, not one reused for both calls --
    # _layer_transient learning carries over within one engine (by design,
    # see F42), and forward_tokens' own prefill-shaped transient is much
    # larger than forward_tokens_serial_positions' per-position one on a
    # real K3 layer. Reusing one engine made the second call inherit the
    # first's inflated reservation and fail on a real MemoryError that had
    # nothing to do with this test's actual comparison.
    engine_batched = StreamingEngine(str(MODEL_DIR), _rc())
    engine_serial = StreamingEngine(str(MODEL_DIR), _rc())
    try:
        engine_batched.cfg.num_hidden_layers = 3
        engine_serial.cfg.num_hidden_layers = 3

        # Ordinary one-token decode is the exact arithmetic contract the
        # serial verifier preserves. Retain its position-1 recurrent endpoint
        # before advancing to position 2.
        kv_batched = engine_batched.new_kv()
        first_logits = engine_batched.forward_tokens(
            tokens[:1], kv_batched
        )
        first_kda_endpoint = kv_batched.kda_cache.fork()
        second_logits = engine_batched.forward_tokens(
            tokens[1:], kv_batched
        )
        batched_logits = mx.concatenate(
            [first_logits, second_logits], axis=0
        )
        mx.eval(batched_logits)

        kv_serial = engine_serial.new_kv()
        serial_logits = engine_serial.forward_tokens_serial_positions(
            tokens, kv_serial, capture_kda_endpoints=True
        )
        mx.eval(serial_logits)
        retained = engine_serial.consume_serial_kda_endpoint(1)
        assert retained is not None
        assert engine_serial._serial_kda_endpoints is None
        assert engine_serial._serial_kda_endpoint_retained_bytes == 0

        captured_layers = 0
        for layer in range(engine_serial.cfg.num_hidden_layers):
            expected_state = first_kda_endpoint.state(layer)
            actual_state = retained.state(layer)
            if expected_state is None:
                assert actual_state is None
                continue
            captured_layers += 1
            assert actual_state is not None
            assert np.array_equal(
                np.array(actual_state),
                np.array(expected_state),
            ), f"KDA state endpoint differs at layer {layer}"
            expected_history = first_kda_endpoint.conv_history(layer)
            actual_history = retained.conv_history(layer)
            assert expected_history is not None
            assert actual_history is not None
            for expected, actual in zip(
                expected_history, actual_history, strict=True
            ):
                assert np.array_equal(
                    np.array(actual), np.array(expected)
                ), f"KDA conv endpoint differs at layer {layer}"
        assert captured_layers > 0

        assert batched_logits.shape == serial_logits.shape
        max_diff = mx.max(mx.abs(
            batched_logits.astype(mx.float32) - serial_logits.astype(mx.float32)))
        mx.eval(max_diff)
        # Float32-rounding-only tolerance -- same reasoning as
        # test_f128_k3_layer_stationary_oracle.py: different matmul
        # batching (all positions in one chunk vs. one position at a
        # time), not a logic difference. Logits accumulate a few more
        # matmuls than the hidden-state comparison there, so a slightly
        # looser bound is appropriate.
        assert max_diff.item() < 1e-3, (
            "forward_tokens_serial_positions and forward_tokens must "
            f"produce near-identical logits; max abs diff {max_diff.item()}")
        assert (
            mx.argmax(batched_logits, axis=-1).tolist()
            == mx.argmax(serial_logits, axis=-1).tolist()
        )
    finally:
        engine_serial.close()
        engine_batched.close()
