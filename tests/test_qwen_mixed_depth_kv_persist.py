from types import SimpleNamespace

import mlx.core as mx

from runtime.kda_state import KDAStateCache
from runtime.kv_cache import KVCache
from runtime.qwen_mixed_depth_kv_persist import (
    QwenMixedDepthPromptPersistence,
)


def _config():
    return SimpleNamespace(
        model_type="qwen3_5",
        num_hidden_layers=4,
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


def _store(tmp_path, fingerprint="fixture"):
    return QwenMixedDepthPromptPersistence(
        tmp_path, fingerprint, 4, config=_config(),
        max_checkpoints=2, max_bytes=10_000_000,
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
    restored_tokens, restored, exact_logits = store.load_matched_chain(match, 4)
    assert restored_tokens == tokens
    assert exact_logits is None
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


def test_mixed_depth_store_ignores_endpoint_checkpoint(tmp_path):
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
