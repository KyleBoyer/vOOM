"""F202: numerical oracle for the native LFM2 port.

Every block in ``runtime/lfm2.py`` is checked against mlx-lm's own
``models/lfm2.py`` on real released weights. Inspection is not enough here:
the short convolution has two easy-to-invert conventions (kernel order, and
whether the carried history is prepended or appended), and both produce
plausible-looking output while being wrong.

Skips when the checkpoint is absent so the suite still runs on a bare tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

MODEL = ROOT / "models" / "LFM2.5-2.6B"

pytestmark = pytest.mark.skipif(
    not (MODEL / "config.json").is_file(), reason="LFM2.5-2.6B not present")


@pytest.fixture(scope="module")
def reference():
    import mlx_lm_shim
    mlx_lm_shim.apply()
    from mlx_lm import load

    model, tokenizer = load(str(MODEL), model_config={"block_ff_dim": 10752})
    return model, tokenizer


@pytest.fixture(scope="module")
def weights():
    import mlx.core as mx

    merged = {}
    for shard in sorted(MODEL.glob("model-*.safetensors")):
        merged.update(mx.load(str(shard)))
    return merged


@pytest.fixture(scope="module")
def cfg():
    from runtime.config import ModelConfig

    return ModelConfig.from_dir(str(MODEL))


def _hidden(mx, batch=1, length=6, dim=2048, seed=0):
    mx.random.seed(seed)
    return (mx.random.normal((batch, length, dim)) * 0.05).astype(mx.bfloat16)


def _close(mx, a, b, tol=2e-2):
    diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
    scale = max(mx.max(mx.abs(b.astype(mx.float32))).item(), 1e-6)
    return diff, diff / scale <= tol


def test_layer_types_split_is_22_conv_and_8_attention(cfg):
    from runtime.lfm2 import layer_is_attention

    kinds = [layer_is_attention(cfg, i) for i in range(cfg.num_hidden_layers)]
    assert cfg.num_hidden_layers == 30
    assert sum(kinds) == 8, f"expected 8 attention layers, got {sum(kinds)}"
    assert kinds.count(False) == 22


def test_short_conv_matches_mlx_lm(reference, weights, cfg):
    import mlx.core as mx

    from runtime.lfm2 import _lfm2_short_conv

    model, _ = reference
    layer_index = 0
    block = model.model.layers[layer_index]
    assert not block.is_attention_layer

    from runtime.kda_state import KDAStateCache

    x = _hidden(mx)
    expected = block.conv(x, mask=None, cache=None)
    # A fresh state cache has no history, so it is the zero-padded case that
    # mlx-lm's cache=None path also computes.
    got = _lfm2_short_conv(x, weights, f"model.layers.{layer_index}", cfg,
                           KDAStateCache(cfg.num_hidden_layers), layer_index)
    diff, ok = _close(mx, got, expected)
    assert ok, f"short conv max abs diff {diff}"


def test_short_conv_history_makes_split_prefill_match_one_shot(
        reference, weights, cfg):
    """Feeding two halves with carried state must equal one full pass.

    This is the property suffix decoding depends on: a rollback restores the
    conv history and replays, so the carried state has to be exactly the
    prefix's own continuation.
    """
    import mlx.core as mx

    from runtime.kda_state import KDAStateCache
    from runtime.lfm2 import _lfm2_short_conv

    layer_index = 0
    prefix = f"model.layers.{layer_index}"
    x = _hidden(mx, length=8)

    whole = _lfm2_short_conv(x, weights, prefix, cfg,
                             KDAStateCache(cfg.num_hidden_layers), layer_index)

    state = KDAStateCache(cfg.num_hidden_layers)
    first = _lfm2_short_conv(x[:, :5, :], weights, prefix, cfg, state,
                             layer_index)
    second = _lfm2_short_conv(x[:, 5:, :], weights, prefix, cfg, state,
                              layer_index)
    split = mx.concatenate([first, second], axis=1)

    diff, ok = _close(mx, split, whole)
    assert ok, f"split/one-shot conv mismatch {diff}"


def test_attention_matches_mlx_lm(reference, weights, cfg):
    import mlx.core as mx

    from runtime.kv_cache import KVCache
    from runtime.lfm2 import _lfm2_attention, layer_is_attention

    model, _ = reference
    layer_index = next(i for i in range(cfg.num_hidden_layers)
                       if layer_is_attention(cfg, i))
    block = model.model.layers[layer_index]
    assert block.is_attention_layer

    x = _hidden(mx)
    expected = block.self_attn(x, mask="causal", cache=None)

    kv = KVCache(cfg.num_hidden_layers)
    got = _lfm2_attention(x, weights, f"model.layers.{layer_index}", cfg, kv,
                          layer_index, 0)
    diff, ok = _close(mx, got, expected)
    assert ok, f"attention max abs diff {diff}"


def test_mlp_matches_mlx_lm(reference, weights, cfg):
    import mlx.core as mx

    from runtime.lfm2 import _lfm2_mlp

    model, _ = reference
    block = model.model.layers[0]
    x = _hidden(mx)
    expected = block.feed_forward(x)
    got = _lfm2_mlp(x, weights, "model.layers.0")
    diff, ok = _close(mx, got, expected)
    assert ok, f"mlp max abs diff {diff}"


@pytest.mark.parametrize("layer_index", [0, 2, 5, 29])
def test_full_block_matches_mlx_lm(reference, weights, cfg, layer_index):
    import mlx.core as mx

    from runtime.kda_state import KDAStateCache
    from runtime.kv_cache import KVCache
    from runtime.lfm2 import layer_is_attention, run_lfm2_block

    model, _ = reference
    block = model.model.layers[layer_index]
    x = _hidden(mx, seed=layer_index)
    mask = "causal" if block.is_attention_layer else None
    expected = block(x, mask, cache=None)

    kv = KVCache(cfg.num_hidden_layers)
    state = KDAStateCache(cfg.num_hidden_layers)
    got = run_lfm2_block(x, weights, f"model.layers.{layer_index}", cfg, kv,
                         layer_index, 0, state_cache=state)
    diff, ok = _close(mx, got, expected)
    assert ok, (f"layer {layer_index} "
                f"({'attention' if layer_is_attention(cfg, layer_index) else 'conv'})"
                f" max abs diff {diff}")


def test_mlp_last_only_matches_the_final_row(reference, weights, cfg):
    """The last-position shortcut must reproduce the full block's last row."""
    import mlx.core as mx

    from runtime.kda_state import KDAStateCache
    from runtime.kv_cache import KVCache
    from runtime.lfm2 import run_lfm2_block

    x = _hidden(mx, length=6, seed=3)
    full = run_lfm2_block(
        x, weights, "model.layers.0", cfg, KVCache(cfg.num_hidden_layers), 0, 0,
        state_cache=KDAStateCache(cfg.num_hidden_layers))
    tail = run_lfm2_block(
        x, weights, "model.layers.0", cfg, KVCache(cfg.num_hidden_layers), 0, 0,
        state_cache=KDAStateCache(cfg.num_hidden_layers), mlp_last_only=True)
    assert tail.shape[1] == 1
    diff, ok = _close(mx, tail, full[:, -1:, :])
    assert ok, f"mlp_last_only diverged from the full block's last row: {diff}"


def test_short_conv_refuses_a_missing_state_cache(weights, cfg):
    """Fail closed rather than silently restart from a zero history.

    The paged-KV layout carries no companion recurrent cache. Substituting
    zeros there does not raise -- it produced fluent nonsense (a repeated
    token pair) while every counter still looked healthy.
    """
    import mlx.core as mx

    from runtime.lfm2 import _lfm2_short_conv

    with pytest.raises(ValueError, match="conv-state cache"):
        _lfm2_short_conv(_hidden(mx), weights, "model.layers.0", cfg, None, 0)
