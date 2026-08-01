"""gpt-oss sliding-window KV retention (exactness + cache semantics).

gpt-oss alternates 128-token ``sliding_attention`` layers with full-attention
layers.  Its attention already slices (decode) and masks (prefill) to that
window, so keys older than the window are unreachable for those layers --
yet the cache retained every position for every layer.  Measured on the
2026-07-31 18,608-token run: 2.944GB of KV where 1.481GB is reachable.

Dropping unreachable keys must be EXACT, not approximate, so the decisive
test here compares real attention output with and without the window bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.config import ModelConfig
from runtime.gptoss import _attention_gptoss, yarn_params
from runtime.kv_cache import KVCache


WINDOW = 8
N_HEADS, N_KV, HEAD_DIM = 4, 2, 8


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="gpt_oss",
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        hidden_size=N_HEADS * HEAD_DIM,
        head_dim=HEAD_DIM,
        rope_theta=150000.0,
        rms_norm_eps=1e-5,
        vocab_size=32,
        intermediate_size=N_HEADS * HEAD_DIM,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        attention_bias=False,
        eos_token_ids=(0,),
        torch_dtype="float32",
        layer_types=("sliding_attention", "full_attention"),
        sliding_window=WINDOW,
    )


def _weights(cfg: ModelConfig, prefix: str) -> dict:
    mx.random.seed(0)
    hidden = cfg.hidden_size
    def rand(*shape):
        return mx.random.normal(shape).astype(mx.float32) * 0.05
    return {
        f"{prefix}.self_attn.q_proj.weight": rand(N_HEADS * HEAD_DIM, hidden),
        f"{prefix}.self_attn.k_proj.weight": rand(N_KV * HEAD_DIM, hidden),
        f"{prefix}.self_attn.v_proj.weight": rand(N_KV * HEAD_DIM, hidden),
        f"{prefix}.self_attn.o_proj.weight": rand(hidden, N_HEADS * HEAD_DIM),
        f"{prefix}.self_attn.sinks": rand(N_HEADS),
    }


def _run(cfg, weights, tokens, *, windowed: bool, chunk: int):
    """Feed ``tokens`` through one layer in ``chunk``-sized tiles."""
    kv = KVCache(cfg.num_hidden_layers)
    if windowed:
        assert kv.configure_sliding_windows(
            cfg.layer_types, cfg.sliding_window) == 1
    freqs, mscale = yarn_params(cfg)
    outputs = []
    for start in range(0, tokens.shape[1], chunk):
        piece = tokens[:, start:start + chunk, :]
        outputs.append(_attention_gptoss(
            piece, weights, "model.layers.0", cfg, kv, 0, kv.offset,
            freqs, mscale))
    return mx.concatenate(outputs, axis=1), kv


def test_windowed_kv_matches_unwindowed_attention_exactly():
    """The decisive proof: keys outside the window are unreachable, so
    dropping them cannot change a single output value."""
    cfg = _config()
    weights = _weights(cfg, "model.layers.0")
    mx.random.seed(1)
    # Well past the window so the bound actually discards keys.
    tokens = mx.random.normal((1, WINDOW * 5, cfg.hidden_size)).astype(mx.float32)

    for chunk in (1, 3, WINDOW * 5):
        full, _ = _run(cfg, weights, tokens, windowed=False, chunk=chunk)
        bounded, kv = _run(cfg, weights, tokens, windowed=True, chunk=chunk)
        mx.eval(full, bounded)
        assert full.shape == bounded.shape
        deviation = float(mx.abs(full - bounded).max())
        scale = float(mx.abs(full).max())
        if chunk == 1 or chunk >= tokens.shape[1]:
            # Decode, and any single-tile prefill, reduce over identically
            # shaped rows, so the bound is bit-exact.
            assert deviation == 0.0, f"chunk={chunk} deviated by {deviation}"
        else:
            # Multi-tile prefill sums a 10-key row where the unbounded arm
            # sums a 43-key row.  The extra entries are masked to exactly
            # -inf -> exp -> 0.0, so the SET attended to is identical; only
            # float32 reduction ORDER differs, which is ~1 ULP.  A wrong mask
            # would move these values by orders of magnitude, not 1e-7.
            assert deviation / scale < 1e-6, (
                f"chunk={chunk} deviated by {deviation} (rel "
                f"{deviation / scale}) -- too large for reduction-order noise")
        # And it really did bound the layer rather than trivially matching.
        # A single tile spanning the whole sequence has nothing older than
        # its own window to drop, so only tiled runs shrink.
        assert kv.keys[0].shape[2] <= WINDOW + chunk - 1
        if chunk < tokens.shape[1]:
            assert kv.keys[0].shape[2] < tokens.shape[1]
        assert kv.offset == tokens.shape[1]


def test_only_sliding_layers_are_bounded():
    cfg = _config()
    kv = KVCache(cfg.num_hidden_layers)
    assert kv.configure_sliding_windows(cfg.layer_types, WINDOW) == 1
    key = mx.zeros((1, N_KV, 1, HEAD_DIM))
    for _ in range(WINDOW * 3):
        kv.update(0, key, key)
        kv.update(1, key, key)
    assert kv.keys[0].shape[2] == WINDOW           # sliding: bounded (L=1)
    assert kv.keys[1].shape[2] == WINDOW * 3       # full: untouched
    assert kv.layer_start(0) == WINDOW * 3 - WINDOW
    assert kv.layer_start(1) == 0


def test_offset_reports_true_sequence_length_when_windowed():
    """offset feeds RoPE and every caller's position math, so a bounded
    layer must not shorten it."""
    cfg = _config()
    kv = KVCache(cfg.num_hidden_layers)
    kv.configure_sliding_windows(cfg.layer_types, WINDOW)
    key = mx.zeros((1, N_KV, 1, HEAD_DIM))
    for expected in range(1, WINDOW * 3 + 1):
        kv.update(0, key, key)
        assert kv.offset == expected


def test_unwindowed_cache_semantics_are_unchanged():
    kv = KVCache(2)
    key = mx.zeros((1, N_KV, 1, HEAD_DIM))
    for _ in range(5):
        kv.update(0, key, key)
    assert kv.offset == 5
    assert kv.layer_start(0) == 0
    kv.trim(3)
    assert kv.keys[0].shape[2] == 3
    assert kv.offset == 3


def test_trim_within_the_retained_window_rolls_back_correctly():
    cfg = _config()
    kv = KVCache(cfg.num_hidden_layers)
    kv.configure_sliding_windows(cfg.layer_types, WINDOW)
    key = mx.zeros((1, N_KV, 1, HEAD_DIM))
    for _ in range(WINDOW * 2):
        kv.update(0, key, key)
    start = kv.layer_start(0)
    kv.trim(start + 2)
    assert kv.keys[0].shape[2] == 2
    assert kv.offset == start + 2


def test_trim_below_the_window_fails_closed():
    """Speculative rollback past discarded keys must raise, never silently
    misalign every remaining position."""
    cfg = _config()
    kv = KVCache(cfg.num_hidden_layers)
    kv.configure_sliding_windows(cfg.layer_types, WINDOW)
    key = mx.zeros((1, N_KV, 1, HEAD_DIM))
    for _ in range(WINDOW * 3):
        kv.update(0, key, key)
    try:
        kv.trim(1)
    except ValueError as error:
        assert "sliding window" in str(error)
    else:
        raise AssertionError("trim below the window must raise")


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
