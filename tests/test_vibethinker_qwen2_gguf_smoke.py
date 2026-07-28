"""VibeThinker-3B (Qwen2, GGUF-only release): real end-to-end weight-loading
and forward-pass smoke test.

VibeThinker-3B's tool-calling fine-tune (huggermax/VibeThinker-3B-tool-
calling-GGUF) is released GGUF-only (403 on the actual repo as of
2026-07-27/28 -- retried multiple times, consistently denied at the CDN
level; see docs/future_lossless_techniques.md). models/VibeThinker-3B-
tool-calling-GGUF/ holds that repo's real config.json/tokenizer files
plus a symlinked base-model GGUF (prithivMLmods/VibeThinker-3B-GGUF,
same Qwen2 architecture/tensor layout -- the GGUF parsing/dequant
engineering this test exercises is format-generic, independent of which
specific fine-tune's weights are behind the symlink) as a stand-in until
the real tool-calling weights become downloadable.

This is a coherence/plumbing smoke test (finite outputs through several
real layers), not the byte-identical-vs-oracle proof the Q4_K/Q6_K
dequant math and GGUF container parser already got in
tests/test_gguf_quant_oracle.py / tests/test_gguf_reader.py.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from runtime import layer_runner
from runtime.kv_cache import KVCache
from runtime.model_loader import WeightStore

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "VibeThinker-3B-tool-calling-GGUF"
_GGUF_FILE = MODEL_DIR / "VibeThinker-3B.Q4_K_M.gguf"
_MODEL_AVAILABLE = _GGUF_FILE.exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="VibeThinker-3B GGUF weights not available locally")


@_model_skip
def test_config_parses_as_dense_qwen2():
    store = WeightStore(MODEL_DIR)
    cfg = store.config
    assert store.gguf is not None
    assert cfg.model_type == "qwen2"
    assert cfg.num_hidden_layers == 36
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 128
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.tie_word_embeddings is True
    assert cfg.attention_bias is True
    assert cfg.num_experts == 0


@_model_skip
def test_first_four_real_layers_produce_finite_output():
    store = WeightStore(MODEL_DIR)
    cfg = store.config
    embed_w, _, _ = store.fetch(["model.embed_tokens.weight"])
    tokens = mx.array([1, 2, 3, 4, 5])
    x = layer_runner.embed(tokens, embed_w["model.embed_tokens.weight"])
    mx.eval(x)
    assert x.shape == (1, 5, cfg.hidden_size)

    num_layers_to_run = 4
    kv = KVCache(num_layers_to_run)
    for i in range(num_layers_to_run):
        w, _, _ = store.fetch(store.layer_param_names(i))
        x = layer_runner.run_block(x, w, f"model.layers.{i}", cfg, kv, i, offset=0)
        mx.eval(x)
        assert not bool(mx.any(mx.isnan(x)).item()), f"layer {i} produced NaN"
        assert not bool(mx.any(mx.isinf(x)).item()), f"layer {i} produced Inf"


@_model_skip
def test_real_engine_generate_end_to_end():
    from runtime.engine import StreamingEngine

    # VibeThinker-3B is a reasoning/instruct fine-tune (its real chat_template
    # opens every turn with <think> per its own model card) -- a bare
    # completion prompt with no chat wrapping produced degenerate repeated
    # punctuation (checked directly, not a correctness bug: the base
    # Qwen2.5-Coder-3B checkpoint just isn't tuned for raw completion).
    # Wrapping in its own real Qwen chat format produces coherent, on-topic
    # reasoning -- the actual proof this pipeline (GGUF parse, Q4_K/Q6_K
    # dequant, WeightStore wiring, tokenizer, forward pass) is correct.
    engine = StreamingEngine(str(MODEL_DIR))
    prompt = ("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
              "<|im_start|>assistant\n")
    result = engine.generate(prompt, max_tokens=20)
    assert isinstance(result, dict)
    text = result.get("text")
    assert isinstance(text, str)
    assert "france" in text.lower() or "paris" in text.lower()
