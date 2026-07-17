"""F62 (DSpark prep) regression: hidden-state taps must be purely additive —
capturing intermediate layer outputs must never change the model's actual
logits/tokens. Uses local SmolLM2-135M (no download, sub-second).

  .venv/bin/python tests/test_f62_hidden_taps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "models/SmolLM2-135M"
PROMPT = "The quick brown fox jumps over the lazy dog and then"


def test_tap_on_off_logit_identity():
    """Same prompt, same fresh KV cache, tap_layers=None vs a real set of
    layers — the returned logits must be EXACTLY equal (not just argmax-
    equal): taps must never perturb the computation graph."""
    import mlx.core as mx
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    tokens = eng.tokenizer.encode(PROMPT).ids

    kv_no_tap = eng.new_kv()
    logits_no_tap = eng.forward_tokens(tokens, kv_no_tap)
    mx.eval(logits_no_tap)

    kv_tap = eng.new_kv()
    logits_tap = eng.forward_tokens(tokens, kv_tap, tap_layers={2, 8, 15, 22})
    mx.eval(logits_tap)

    assert mx.array_equal(logits_no_tap, logits_tap).item(), \
        "tap-on logits differ from tap-off logits — taps are not side-effect-free"
    eng.close()


def test_tap_captures_requested_layers_only():
    """self._tap_hidden must contain exactly the requested layer indices,
    each shaped (1, L, hidden), and must be empty/reset when tap_layers=None
    on a later call (no stale taps leaking across calls)."""
    import mlx.core as mx
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    tokens = eng.tokenizer.encode(PROMPT).ids
    requested = {2, 8, 15}

    kv = eng.new_kv()
    eng.forward_tokens(tokens, kv, tap_layers=requested)
    assert set(eng._tap_hidden.keys()) == requested, \
        f"tap_hidden keys {set(eng._tap_hidden.keys())} != requested {requested}"
    for layer, h in eng._tap_hidden.items():
        assert h.shape == (1, len(tokens), eng.cfg.hidden_size), \
            f"layer {layer} tap shape {h.shape} unexpected"

    # a later no-tap call must clear the previous call's captured taps, not
    # leave them sitting around stale
    kv2 = eng.new_kv()
    eng.forward_tokens(tokens, kv2, tap_layers=None)
    assert eng._tap_hidden == {}, \
        f"tap_hidden should be empty after a tap_layers=None call, got keys {set(eng._tap_hidden.keys())}"
    eng.close()


def test_tap_reset_between_tap_calls():
    """A second forward_tokens call with a DIFFERENT tap_layers set must not
    carry over entries from the first call's set."""
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    tokens = eng.tokenizer.encode(PROMPT).ids

    eng.forward_tokens(tokens, eng.new_kv(), tap_layers={2, 8})
    assert set(eng._tap_hidden.keys()) == {2, 8}

    eng.forward_tokens(tokens, eng.new_kv(), tap_layers={15})
    assert set(eng._tap_hidden.keys()) == {15}, \
        "stale tap entries from a previous call leaked into the new tap set"
    eng.close()


def test_serial_position_verifier_captures_taps_without_changing_logits():
    """DSpark needs taps from the one-token-arithmetic verifier, not the
    batched-position path that can move a near-tied Qwen argmax."""
    import mlx.core as mx
    from runtime.engine import RuntimeConfig, StreamingEngine

    eng = StreamingEngine(MODEL, RuntimeConfig(max_weight_cache_mb=400, pin_lm_head=True))
    tokens = list(eng.tokenizer.encode(PROMPT).ids[-4:])
    requested = {2, 8, 15}

    plain = eng.forward_tokens_serial_positions(tokens, eng.new_kv())
    tapped = eng.forward_tokens_serial_positions(
        tokens, eng.new_kv(), tap_layers=requested)

    assert mx.array_equal(plain, tapped).item()
    assert set(eng._tap_hidden) == requested
    for hidden in eng._tap_hidden.values():
        assert hidden.shape == (1, len(tokens), eng.cfg.hidden_size)
    eng.close()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
