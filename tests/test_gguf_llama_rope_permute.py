"""llama.cpp's real HF-to-GGUF converter (`conversion/llama.py`'s
`LlamaModel`, fetched 2026-07-28) permutes q_proj/k_proj weight (and bias)
rows on every Llama-family GGUF export, so its own interleaved-pair RoPE
kernel matches what was originally a HF rotate-half layout. Left
unreversed, this codebase's `_attention` (runtime/layer_runner.py,
`mx.fast.rope(..., traditional=False)` -- the rotate-half convention)
silently consumed the wrong row order for exactly those two projections:
measured directly against the real unquantized bf16
Llama-3-Groq-8B-Tool-Use weights, q_proj's mean abs error was ~0.014
against an original whose std is ~0.018 (~100% of signal -- reconstructing
noise, not the model), while every other tensor (ffn_down, embeddings,
the untied lm_head) showed the ordinary ~1-8% Q4_K/Q6_K quantization
noise. End to end this produced fluent-but-incoherent decode output
("I'm not sure, I'm not sure...") on a query ("What is the capital of
France?") the same checkpoint's real bf16 safetensors release answers
correctly and deterministically. Qwen2's own conversion module has no
such permutation step at all (confirmed against the real source), which
is why VibeThinker-3B's GGUF loading needed no equivalent fix -- this is
genuinely Llama-architecture-specific, not a general GGUF concern.

runtime.model_loader._undo_llama_cpp_gguf_rope_permute implements the
reversal; this test validates it three ways:

1. Against a verbatim transcription of the real `permute()` function
   (numeric round-trip: permute-then-unpermute must be the identity).
2. Against the real unquantized bf16 weights of a real downloaded
   checkpoint, both directly (calling the helper) and through the full
   `WeightStore.fetch()` path.
3. A real end-to-end `generate()` call reproducing the coherent-vs-
   incoherent proof described above.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from runtime.model_loader import WeightStore, _undo_llama_cpp_gguf_rope_permute

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "models" / "Llama-3-Groq-8B-Tool-Use-GGUF"
SAFETENSORS_DIR = ROOT / "models" / "Llama-3-Groq-8B-Tool-Use"
_GGUF_AVAILABLE = any(GGUF_DIR.glob("*.gguf")) if GGUF_DIR.exists() else False
_SAFETENSORS_AVAILABLE = (SAFETENSORS_DIR / "model.safetensors").exists()
_model_skip = pytest.mark.skipif(
    not (_GGUF_AVAILABLE and _SAFETENSORS_AVAILABLE),
    reason="Llama-3-Groq-8B-Tool-Use GGUF+safetensors pair not available locally")


# Verbatim from the real ggml-org/llama.cpp conversion/llama.py
# `LlamaModel.permute` (fetched 2026-07-28) -- the forward direction this
# codebase's helper must invert.
def _real_permute(weights: np.ndarray, n_head: int, n_head_kv: int | None) -> np.ndarray:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(weights.shape))


def test_unpermute_is_the_exact_inverse_of_the_real_permute_on_synthetic_data():
    rng = np.random.default_rng(0)

    # q_proj case: n_head == n_head_kv.
    original = rng.standard_normal((4096, 8)).astype(np.float32)
    permuted = _real_permute(original, 32, 32)
    recovered = np.array(_undo_llama_cpp_gguf_rope_permute(mx.array(permuted), 32, 32))
    assert np.array_equal(recovered, original)

    # k_proj/GQA case: n_head != n_head_kv.
    original_k = rng.standard_normal((1024, 8)).astype(np.float32)
    permuted_k = _real_permute(original_k, 32, 8)
    recovered_k = np.array(_undo_llama_cpp_gguf_rope_permute(mx.array(permuted_k), 32, 8))
    assert np.array_equal(recovered_k, original_k)


@_model_skip
def test_unpermuted_gguf_weight_matches_real_bf16_original_within_quant_noise():
    from formats.gguf_reader import GGUFFile

    gguf_path = next(GGUF_DIR.glob("*.gguf"))
    gguf_model = GGUFFile(gguf_path)
    bf16 = mx.load(str(SAFETENSORS_DIR / "model.safetensors"))

    cases = [
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", 32, 32),
        ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", 32, 8),
    ]
    for gguf_name, safetensors_name, n_head, n_head_kv in cases:
        raw = gguf_model.load(gguf_name, out_dtype=mx.float32)
        fixed = np.array(_undo_llama_cpp_gguf_rope_permute(raw, n_head, n_head_kv))
        orig = np.array(bf16[safetensors_name].astype(mx.float32))
        mean_abs_err = np.mean(np.abs(fixed - orig))
        # Ordinary Q4_K/Q6_K quantization noise on this checkpoint's other
        # tensors (ffn_down, embeddings, lm_head) measured at ~1-8% of the
        # original's std; the UNFIXED error was ~100% (reconstructing noise,
        # not signal). A generous 20%-of-std ceiling clearly separates "this
        # is quantization noise" from "the permutation bug is back".
        assert mean_abs_err < 0.2 * orig.std(), (
            f"{gguf_name}: mean abs err {mean_abs_err} too large relative to "
            f"orig std {orig.std()} -- permutation fix may be broken")


@_model_skip
def test_fetch_through_weightstore_applies_the_fix():
    store = WeightStore(GGUF_DIR)
    bf16 = mx.load(str(SAFETENSORS_DIR / "model.safetensors"))
    w, _, _ = store.fetch([
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ])
    for name in w:
        mine = np.array(w[name].astype(mx.float32))
        orig = np.array(bf16[name].astype(mx.float32))
        mean_abs_err = np.mean(np.abs(mine - orig))
        assert mean_abs_err < 0.2 * orig.std(), f"{name}: fetch() did not apply the fix"


@_model_skip
def test_real_engine_generate_answers_coherently():
    from runtime.engine import StreamingEngine

    engine = StreamingEngine(str(GGUF_DIR))
    prompt = ("<|start_header_id|>user<|end_header_id|>\n\n"
              "What is the capital of France?<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n")
    result = engine.generate(prompt, max_tokens=20)
    assert "paris" in result["text"].lower()
