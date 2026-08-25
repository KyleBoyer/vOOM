"""F103: this project's own hand-written mx.fast.metal_kernel fusion of the
Gated DeltaNet decode-step recurrence, real greedy-token quality gate.

Distinct from SQ26 (zmlx, a third-party kernel library found to be a net
decode slowdown despite winning an isolated microbenchmark -- attributed to
per-call dispatch/abstraction overhead a real generate() loop doesn't
amortize the way an isolated back-to-back loop does). This is a from-scratch
kernel written specifically to test whether a tightly-scoped fusion (no
external library overhead) fares better. Mirrors
tests/test_zmlx_fused_deltanet_decode.py's "greedy A/B, byte-identical
tokens" standard.

Real end-to-end result (docs/future_lossless_techniques.md F103): unlike
zmlx, this DOES show a real (if small) win in a genuinely compute-bound
regime (prequantized MXFP4 + resident_fast_decode) -- two independent runs
measured +5.9% and +2.8% decode throughput, byte-identical tokens both
times. A first attempt on a raw bf16 checkpoint with a bare config showed
~no difference, but that regime was disk-bound (~0.19 tok/s), not a fair
test of a compute kernel -- worth remembering before concluding "no
effect" from any single real-model timing without checking whether
compute is actually the bottleneck in that configuration.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from runtime.qwen35 import (
    _causal_depthwise_conv1d, _native_fused_causal_conv1d,
    _native_fused_gated_delta_step, _sequential_gated_delta_rule)

_REAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-4B")
_skip = pytest.mark.skipif(
    not _REAL_MODEL_DIR.exists(),
    reason="real Qwen3.5-4B checkpoint not present on this machine")

_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is"
)


def test_native_kernel_matches_reference_step_math():
    """Weights-free: the fused kernel's single-position recurrence must
    match _sequential_gated_delta_rule's own single-step body (real
    Qwen3.5-4B dimensions: H=32, Dk=Dv=128) to float32 accumulation-order
    noise, not just similar-looking output."""
    mx.random.seed(0)
    B, H, Dk, Dv = 1, 32, 128, 128
    q = mx.random.normal((B, H, Dk)).astype(mx.float32)
    k = mx.random.normal((B, H, Dk)).astype(mx.float32)
    v = mx.random.normal((B, H, Dv)).astype(mx.float32)
    beta = mx.random.uniform(0.0, 1.0, (B, H)).astype(mx.float32)
    decay = -mx.random.uniform(0.0, 2.0, (B, H)).astype(mx.float32)
    state = mx.random.normal((B, H, Dk, Dv)).astype(mx.float32) * 0.1

    ref_out, ref_state = _sequential_gated_delta_rule(
        q[:, None], k[:, None], v[:, None], beta[:, None], decay[:, None],
        state)
    fused_out, fused_state = _native_fused_gated_delta_step(
        q, k, v, beta, decay, state)
    mx.eval(ref_out, ref_state, fused_out, fused_state)

    out_diff = float(mx.max(mx.abs(ref_out[:, 0] - fused_out)))
    state_diff = float(mx.max(mx.abs(ref_state - fused_state)))
    assert out_diff < 1e-3, out_diff
    assert state_diff < 1e-3, state_diff


def test_native_conv1d_kernel_matches_reference_step_math():
    """Weights-free: the fused conv1d+SiLU kernel must match
    _causal_depthwise_conv1d's own decode-time (L=1) math at real
    Qwen3.5-4B dimensions (C=8192 combined q+k+v channels, K=4 taps)."""
    mx.random.seed(1)
    B, C, K = 1, 8192, 4
    x = mx.random.normal((B, 1, C)).astype(mx.float32)
    history = mx.random.normal((B, K - 1, C)).astype(mx.float32)
    weight = (mx.random.normal((C, 1, K)) * 0.3).astype(mx.float32)

    ref_out, ref_history = _causal_depthwise_conv1d(x, weight, history, K)
    fused_out, fused_history = _native_fused_causal_conv1d(x, weight, history, K)
    mx.eval(ref_out, ref_history, fused_out, fused_history)

    out_diff = float(mx.max(mx.abs(ref_out - fused_out)))
    history_diff = float(mx.max(mx.abs(ref_history - fused_history)))
    assert out_diff < 1e-3, out_diff
    assert history_diff == 0.0, history_diff  # pure slicing, must be exact


def test_native_kernel_is_a_true_no_op_when_disabled():
    """native_fused_deltanet_decode=False (the default) must not touch the
    kernel at all -- covered indirectly by the byte-identical real-model
    test below, asserted directly here via the dispatch condition itself:
    length>1 (prefill) never takes the fused branch regardless of the flag."""
    from runtime.qwen35 import _gated_delta_net
    import inspect
    src = inspect.getsource(_gated_delta_net)
    assert "native_fused_decode and length == 1" in src


def test_serial_verifier_forwards_fused_decode_policy_to_both_qwen_paths():
    """The dense batched-MLP and fallback/MoE verifier branches must not
    silently drop the decode-kernel policy while ordinary decode honors it."""
    import inspect

    from runtime.engine import StreamingEngine

    src = inspect.getsource(StreamingEngine.forward_tokens_serial_positions)
    assert src.count("self.rc.native_fused_deltanet_decode") == 2
    assert src.count("self.rc.zmlx_fused_deltanet_decode") == 2


def _run(native_fused: bool, max_tokens: int = 24):
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    rc = RuntimeConfig(
        prefill_chunk_size=512, native_fused_deltanet_decode=native_fused)
    engine = StreamingEngine(str(_REAL_MODEL_DIR), rc)
    try:
        return engine.generate(
            _PROMPT, max_tokens=max_tokens,
            sampling=SamplingParams(temperature=0.0))
    finally:
        engine.close()


@_skip
def test_native_fused_decode_matches_baseline_byte_identical():
    baseline = _run(native_fused=False)
    fused = _run(native_fused=True)
    assert fused["tokens"] == baseline["tokens"], (
        "the native fused DeltaNet decode kernel must produce byte-identical "
        "greedy output to the existing float32-accumulated implementation"
    )
    assert fused["text"] == baseline["text"]


_JET_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "models" / "Jet-Nemotron-4B")
_skip_jet = pytest.mark.skipif(
    not _JET_MODEL_DIR.exists(),
    reason="real Jet-Nemotron-4B checkpoint not present on this machine")


@_skip_jet
def test_native_kernel_reused_for_jet_nemotron_byte_identical():
    """Jet-Nemotron's JetBlock recurrence (runtime/jet_nemotron.py::
    _jet_block) is mathematically identical to qwen3_5's gated delta rule
    -- the SAME kernel is reused directly (dimension-agnostic via runtime
    shape reads), no new kernel written. Real dimensions differ from
    Qwen3.5-4B (16 heads / Dv=256 here vs 32 heads / Dv=128 there) -- a
    genuine test that the kernel generalizes, not just coincidentally
    works for the one shape it was designed against."""
    from runtime.engine import RuntimeConfig, StreamingEngine
    from runtime.sampler import SamplingParams

    prompt = ("The capital of France is Paris. The capital of Germany is "
              "Berlin. The capital of Italy is")

    def run(native_fused):
        rc = RuntimeConfig(
            prefill_chunk_size=512, native_fused_deltanet_decode=native_fused)
        engine = StreamingEngine(str(_JET_MODEL_DIR), rc)
        try:
            return engine.generate(
                prompt, max_tokens=16, sampling=SamplingParams(temperature=0.0))
        finally:
            engine.close()

    baseline = run(False)
    fused = run(True)
    assert fused["tokens"] == baseline["tokens"]
    assert fused["text"] == baseline["text"]
