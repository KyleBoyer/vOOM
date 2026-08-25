from types import SimpleNamespace

import mlx.core as mx
import pytest

from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.qwen_mixed_depth_kv_persist import (
    QwenMixedDepthPromptPersistence,
)


def _config():
    return SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=4,
        hidden_size=3,
        vocab_size=11,
        layer_types=(
            "linear_attention", "full_attention",
            "linear_attention", "full_attention",
        ),
    )


def _mixed_kv():
    kv = KVCache(4)
    kv.keys[1] = mx.arange(14, dtype=mx.float32).reshape(1, 1, 7, 2)
    kv.values[1] = (kv.keys[1] + 100).astype(mx.float32)
    kv.keys[3] = mx.arange(6, dtype=mx.float32).reshape(1, 1, 3, 2)
    kv.values[3] = (kv.keys[3] + 200).astype(mx.float32)
    recurrent = KDAStateCache(4)
    for layer in (0, 2):
        recurrent.set_state(
            layer, mx.full((1, 2, 2), layer + 1, dtype=mx.float32))
        recurrent.set_conv_history(
            layer, (mx.full((1, 2), layer + 3, dtype=mx.bfloat16),))
    kv.kda_cache = recurrent
    return kv


def _store(tmp_path, fingerprint="fixture", *, allow_endpoint=False):
    return QwenMixedDepthPromptPersistence(
        tmp_path, fingerprint, 4, config=_config(),
        max_checkpoints=2, max_bytes=10_000_000,
        allow_prompt_endpoint=allow_endpoint,
    )


def test_mixed_depth_snapshot_restores_strict_extension_exactly(tmp_path):
    store = _store(tmp_path)
    tokens = tuple(range(7))
    source = _mixed_kv()
    chain = store.save(
        (), 0, tokens, source, None, None,
        prompt_length=7, reusable_prefix=7, approximate=True,
        cache_namespace="gateway", checkpoint_kind="stable_prefix",
    )
    assert len(chain) == 1
    # Re-saving the same semantic prefix reuses the verified generation.
    assert store.save(
        (), 0, tokens, source, None, None,
        prompt_length=7, reusable_prefix=7, approximate=True,
        cache_namespace="gateway", checkpoint_kind="stable_prefix",
    ) == chain

    assert store.find_best_match(
        tokens, 4, cache_namespace="gateway") is None
    match = store.find_best_match(
        tokens + (9, 10), 4, cache_namespace="gateway")
    assert match["case"] == "extension"
    assert match["matched"] == 7
    (restored_tokens, restored, exact_logits,
     exact_hidden) = store.load_matched_chain(match, 4)
    assert restored_tokens == tokens
    assert exact_logits is None
    assert exact_hidden is None
    assert restored.offset == 7
    assert restored.layer_lengths() == (0, 7, 0, 3)
    for layer in (1, 3):
        assert mx.array_equal(restored.keys[layer], source.keys[layer]).item()
        assert mx.array_equal(restored.values[layer], source.values[layer]).item()
    for layer in (0, 2):
        assert mx.array_equal(
            restored.kda_cache.state(layer),
            source.kda_cache.state(layer),
        ).item()
        assert mx.array_equal(
            restored.kda_cache.conv_history(layer)[0],
            source.kda_cache.conv_history(layer)[0],
        ).item()


def test_mixed_depth_snapshot_fails_closed_on_corruption_and_fingerprint(tmp_path):
    store = _store(tmp_path)
    tokens = tuple(range(7))
    chain = store.save(
        (), 0, tokens, _mixed_kv(), None, None,
        prompt_length=7, reusable_prefix=7, approximate=True,
        cache_namespace="gateway", checkpoint_kind="stable_prefix",
    )
    foreign = _store(tmp_path, fingerprint="different")
    assert foreign.find_best_match(
        tokens + (8,), 4, cache_namespace="gateway") is None

    payload = tmp_path / f"{chain[0]}.mixed.safetensors"
    data = bytearray(payload.read_bytes())
    data[len(data) // 2] ^= 1
    payload.write_bytes(data)
    assert store.find_best_match(
        tokens + (8,), 4, cache_namespace="gateway") is None


def test_mixed_depth_prompt_endpoint_restores_logits_only_on_exact_match(
        tmp_path):
    store = _store(tmp_path, allow_endpoint=True)
    tokens = tuple(range(7))
    source = _mixed_kv()
    logits = mx.arange(11, dtype=mx.float32).reshape(1, 1, 11)
    hidden = mx.arange(3, dtype=mx.bfloat16).reshape(1, 1, 3)
    chain = store.save_prompt_endpoint(
        tokens, source, logits, hidden, approximate=True,
        cache_namespace="gateway")
    assert store.save_prompt_endpoint(
        tokens, source, logits, hidden, approximate=True,
        cache_namespace="gateway") == chain

    exact = store.find_best_match(
        tokens, 4, cache_namespace="gateway")
    assert exact["case"] == "endpoint"
    (restored_tokens, restored, restored_logits,
     restored_hidden) = store.load_matched_chain(exact, 4)
    assert restored_tokens == tokens
    assert restored.offset == len(tokens)
    assert mx.array_equal(restored_logits, logits).item()
    assert mx.array_equal(restored_hidden, hidden).item()

    extension = store.find_best_match(
        tokens + (9,), 4, cache_namespace="gateway")
    assert extension["case"] == "extension"
    (_tokens, _restored, extension_logits,
     extension_hidden) = store.load_matched_chain(extension, 4)
    assert extension_logits is None
    assert extension_hidden is None


def test_mixed_depth_prompt_endpoint_is_opt_in_on_write_and_read(tmp_path):
    disabled = _store(tmp_path)
    with pytest.raises(ValueError, match="endpoint persistence is disabled"):
        disabled.save_prompt_endpoint(
            tuple(range(7)), _mixed_kv(),
            mx.arange(11, dtype=mx.float32).reshape(1, 1, 11),
            mx.arange(3, dtype=mx.bfloat16).reshape(1, 1, 3),
            approximate=True,
        )

    enabled = _store(tmp_path, allow_endpoint=True)
    tokens = tuple(range(7))
    enabled.save_prompt_endpoint(
        tokens, _mixed_kv(),
        mx.arange(11, dtype=mx.float32).reshape(1, 1, 11),
        mx.arange(3, dtype=mx.bfloat16).reshape(1, 1, 3),
        approximate=True,
    )
    assert disabled.find_best_match(tokens, 4) is None


def test_mixed_depth_generic_postgeneration_endpoint_is_still_ignored(
        tmp_path):
    store = _store(tmp_path)
    assert store.save(
        ("stable",), 7, tuple(range(8)), _mixed_kv(),
        mx.array([1]), mx.array([2]),
        prompt_length=8, reusable_prefix=0, approximate=True,
        checkpoint_kind="endpoint",
    ) == ("stable",)
    assert not list(tmp_path.glob("*.mixed.json"))


def test_mixed_depth_store_declares_approximate_stable_prefix_requirement(
        tmp_path):
    store = _store(tmp_path)
    assert store.requires_approximate_stable_prefix is True
