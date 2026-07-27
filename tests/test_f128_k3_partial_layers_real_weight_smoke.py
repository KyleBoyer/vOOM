"""F128: Kimi K3 real-weight partial-layer smoke test.

Not a numerical oracle (no real transformers modeling_kimi_linear_k3.py
forward pass is feasible at 2.8T/896-expert scale on this machine) -- this
drives `run_kimi_k3_block` for the first few real layers of the actual
downloaded ~1.4TB checkpoint (dense layer 0, then KDA layer 1) through
StreamingEngine's own real disk-tier weight/expert paging, and checks the
result is finite and correctly shaped. This is the same "coherence/
plumbing gate, not byte-identical proof" category as
test_kimi_linear_smoke.py::test_real_engine_generate_end_to_end, scoped to
a handful of layers instead of a full generate() call so it stays fast
enough to run routinely against a checkpoint this large.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
_MODEL_AVAILABLE = (MODEL_DIR / "model.safetensors.index.json").exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 is not available locally (a real ~1.4TB model, not fetched in CI)",
)


@_model_skip
def test_first_three_real_layers_produce_finite_output():
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.kimi_linear import run_kimi_k3_block

    rc = RuntimeConfig(
        prefill_chunk_size=8, min_weight_cache_mb=500, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(MODEL_DIR), rc)
    try:
        assert engine.cfg.model_type == "kimi_k3"
        assert engine.cfg.attn_res_block_size == 12
        assert engine.cfg.hidden_act == "situ"
        assert engine.cfg.moe_latent_hidden_size > 0

        # F128: a single token keeps the worst-case distinct routed-expert
        # count at num_experts_per_tok (16) instead of up to
        # len(tokens)*16 -- K3's real per-expert dequantized size
        # (moe_latent_hidden_size x moe_intermediate_size, ~66MB each) made
        # even 4 tokens (up to 64 distinct experts, ~4-8GB) exceed this
        # test's deliberately small cache budget on a 16GB machine. This
        # is a real memory-governor constraint, not a correctness concern
        # this test is trying to exercise -- see test_first_three_real_
        # layers_produce_finite_output's docstring.
        tokens = [3]
        x = engine._embed(tokens)
        assert x.shape == (1, len(tokens), engine.cfg.hidden_size)
        mx.eval(x)

        kv = engine.new_kv()
        assert getattr(kv, "kda_cache", None) is not None

        block_residual = mx.zeros(
            (x.shape[0] * x.shape[1], 0, x.shape[2]), dtype=x.dtype)

        num_layers_to_run = 3
        for layer in range(num_layers_to_run):
            w = engine.cache.get(
                engine._layer_key(layer), engine._layer_names(layer))
            x, block_residual = run_kimi_k3_block(
                x, w, f"model.layers.{layer}", engine.cfg, kv, layer, 0,
                block_residual, engine._get_experts)
            mx.eval(x, block_residual)

            assert x.shape == (1, len(tokens), engine.cfg.hidden_size)
            assert not bool(mx.any(mx.isnan(x)).item()), f"layer {layer} produced NaN"
            assert not bool(mx.any(mx.isinf(x)).item()), f"layer {layer} produced Inf"

            expected_blocks = layer // engine.cfg.attn_res_block_size + 1
            assert block_residual.shape == (
                x.shape[0] * x.shape[1], expected_blocks, x.shape[2]), (
                f"layer {layer}: block_residual grew to "
                f"{block_residual.shape}, expected {expected_blocks} blocks")

        from runtime.kimi_linear import apply_output_attn_res

        out = apply_output_attn_res(
            x, {
                "model.output_attn_res_proj.weight": engine._output_attn_res_proj_w,
                "model.output_attn_res_norm.weight": engine._output_attn_res_norm_w,
            }, block_residual, engine.cfg)
        mx.eval(out)
        assert out.shape == x.shape
        assert not bool(mx.any(mx.isnan(out)).item())
    finally:
        engine.close()


@_model_skip
def test_real_engine_generate_end_to_end():
    """The full 93-layer stack (69 KDA + 24 MLA/MoE layers, real MXFP4
    expert dequant, real Stable LatentMoE projection, real AttnRes) through
    a real StreamingEngine.generate() call -- same "coherence/plumbing
    gate, not byte-identical proof" category as
    test_kimi_linear_smoke.py::test_real_engine_generate_end_to_end and
    test_f93_k25_realweight_generate_e2e.py, scaled to K3's real ~1.4TB
    checkpoint. No numerical oracle exists at this scale; this only proves
    the whole wired-up stack runs coherently against real weights and
    produces genuine disk-tier weight-cache activity, not a degenerate
    no-op path. May take a long time (K3 is ~2.5x K2.5's checkpoint size
    with far more layers/active-experts-per-token) -- run in the
    background, not inline in a fast test loop.
    """
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    rc = RuntimeConfig(
        prefill_chunk_size=8, min_weight_cache_mb=500, max_weight_cache_mb=6000)
    engine = StreamingEngine(str(MODEL_DIR), rc)
    try:
        result = engine.generate(
            "The capital of France is", max_tokens=1,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()
    assert result["text"]
    stats = result["path_stats"]
    assert stats["weight_cache_misses"] > 0
    assert stats["expert_cache_misses"] > 0
