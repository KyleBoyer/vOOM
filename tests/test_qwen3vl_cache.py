"""Exact image-embedding reuse for repeated Qwen3-VL requests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import runtime.qwen3vl as qwen3vl
from runtime.weight_cache import WeightCache
from runtime.kv_cache import KVCache
from runtime.kda_state import KDAStateCache


def _engine():
    return SimpleNamespace(store=object(), governor=None)


def test_identical_rgb_content_reuses_exact_embeddings(monkeypatch):
    calls = []
    embeds = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)
    deepstack = [mx.array([[float(i)]], dtype=mx.bfloat16) for i in range(3)]

    monkeypatch.setattr(
        qwen3vl, "preprocess_image",
        lambda image, **kwargs: (
            calls.append(image.tobytes()) or (object(), 16, 16)),
    )
    monkeypatch.setattr(
        qwen3vl, "_load_vision_weights", lambda store, cache=None: object(),
    )
    monkeypatch.setattr(
        qwen3vl, "vision_forward",
        lambda store, pixels, gh, gw, config, weights=None: (embeds, deepstack),
    )
    engine = _engine()
    first = Image.new("RGB", (64, 64), (0, 255, 0))
    second = Image.new("RGBA", (64, 64), (0, 255, 0, 255))

    e1, d1, hit1, weights = qwen3vl._cached_vision_forward(
        engine, first, (16, 16), {})
    e2, d2, hit2, _weights = qwen3vl._cached_vision_forward(
        engine, second, (16, 16), {}, weights)

    assert not hit1 and hit2
    assert len(calls) == 1
    assert e2 is e1 and d2 is d1


def test_changed_pixels_miss_and_lru_capacity_is_enforced(monkeypatch):
    calls = []

    def preprocess(image, **kwargs):
        calls.append(image.getpixel((0, 0)))
        return object(), 16, 16

    def forward(store, pixels, gh, gw, config, weights=None):
        value = float(len(calls))
        return mx.array([[value]]), [mx.array([[value]])]

    monkeypatch.setenv("VMODEL_VISION_CACHE_ENTRIES", "1")
    monkeypatch.setattr(
        qwen3vl, "_load_vision_weights", lambda store, cache=None: object())
    monkeypatch.setattr(qwen3vl, "preprocess_image", preprocess)
    monkeypatch.setattr(qwen3vl, "vision_forward", forward)
    engine = _engine()
    green = Image.new("RGB", (64, 64), (0, 255, 0))
    blue = Image.new("RGB", (64, 64), (0, 0, 255))

    qwen3vl._cached_vision_forward(engine, green, (16, 16), {})
    qwen3vl._cached_vision_forward(engine, blue, (16, 16), {})
    _e, _d, hit, _weights = qwen3vl._cached_vision_forward(
        engine, green, (16, 16), {})

    assert not hit
    assert calls == [(0, 255, 0), (0, 0, 255), (0, 255, 0)]
    assert len(engine._vision_embedding_cache) == 1


def test_weight_cache_can_keep_released_vision_page_raw():
    class Store:
        def fetch(self, names):
            value = mx.array([[1.0, 2.0]])
            return {names[0]: value}, 0.0, value.nbytes

    cache = WeightCache(
        Store(), max_bytes=1024,
        transform=lambda _name, value: value * 2,
    )

    raw = cache.get(
        "vision:released", ["model.visual.proj.weight"],
        apply_transform=False,
    )["model.visual.proj.weight"]
    transformed = cache.get(
        "text:fast", ["model.visual.proj.weight"],
    )["model.visual.proj.weight"]
    mx.eval(raw, transformed)

    assert raw.tolist() == [[1.0, 2.0]]
    assert transformed.tolist() == [[2.0, 4.0]]


def test_fast_vision_pixel_budget_is_config_driven():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(vision_config={
            "patch_size": 16, "spatial_merge_size": 2,
        }),
        rc=SimpleNamespace(vision_max_patches=1024),
    )
    assert qwen3vl._vision_max_pixels(engine) == 512 * 512

    engine.rc.vision_max_patches = 4096
    assert qwen3vl._vision_max_pixels(engine) == 1024 * 1024

    engine.rc.vision_max_patches = 128
    with pytest.raises(ValueError, match="vision_max_patches"):
        qwen3vl._vision_max_pixels(engine)


def test_vl_pipeline_requires_fully_resident_dense_text_trunk():
    resident = {"layer:0", "layer:1"}
    engine = SimpleNamespace(
        rc=SimpleNamespace(resident_fast_decode=True),
        cfg=SimpleNamespace(num_experts=0, num_hidden_layers=2),
        _embed_rows=None,
        cache=SimpleNamespace(contains=lambda key: key in resident),
        _layer_key=lambda layer: f"layer:{layer}",
    )
    kv = KVCache(2)
    assert qwen3vl._vl_resident_pipeline_ready(engine, kv)

    resident.remove("layer:1")
    assert not qwen3vl._vl_resident_pipeline_ready(engine, kv)
    resident.add("layer:1")
    engine._embed_rows = object()
    assert not qwen3vl._vl_resident_pipeline_ready(engine, kv)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize(("query_length", "key_length"), [(17, 17), (7, 23), (1, 23)])
def test_native_causal_attention_is_exact_lower_right(
        dtype, query_length, key_length):
    q = mx.arange(2 * query_length * 8, dtype=mx.float32).reshape(
        1, 2, query_length, 8).astype(dtype) / 100
    k = mx.arange(2 * key_length * 8, dtype=mx.float32).reshape(
        1, 2, key_length, 8).astype(dtype) / 110
    v = mx.sin(mx.arange(
        2 * key_length * 8, dtype=mx.float32)).reshape(
            1, 2, key_length, 8).astype(dtype)
    query_positions = mx.arange(
        key_length - query_length, key_length)[:, None]
    key_positions = mx.arange(key_length)[None, :]
    explicit_mask = mx.where(
        key_positions <= query_positions, 0.0, float("-inf")).astype(dtype)

    native = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=8 ** -0.5, mask="causal")
    explicit = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=8 ** -0.5, mask=explicit_mask)
    mx.eval(native, explicit)

    assert np.array_equal(
        np.array(native.astype(mx.float32)),
        np.array(explicit.astype(mx.float32)),
    )


def test_vision_path_stats_propagate_cache_and_pic_evidence():
    engine = SimpleNamespace(
        rope_profile="released",
        effective_max_position_embeddings=32_768,
    )
    result = {
        "prompt_tokens": 200,
        "prefill_s": 0.25,
        "vision_cache_hits": 1,
        "vision_cache_misses": 0,
        "vision_prompt_cache_hit": False,
        "vision_prompt_cache_exact_hit": False,
        "vision_prompt_cache_prefix_tokens": 0,
        "vision_prompt_cache_tower_skipped": 0,
        "vision_prompt_cache_stored": True,
        "vision_tool_pic": True,
        "vision_tool_pic_selected_tokens": 80,
        "vision_tool_pic_reused_tokens": 120,
        "vision_tool_pic_repaired_tokens": 4,
        "vision_tool_pic_memory_admitted": 1,
        "vision_tool_pic_projected_bytes": 123_456,
        "sampling_profile": "greedy",
        "constraint_profile": "required_tool",
    }

    stats = qwen3vl._vision_path_stats(
        engine, result, prompt_cache_mode=None,
        prompt_cache_lookup_s=0.002,
        prompt_state_approximate=True)

    assert stats["prompt_cache_source"] == "vision_tool_pic"
    assert stats["prompt_cache_write_tokens"] == 200
    assert stats["vision_cache_hits"] == 1
    assert stats["tool_pic"] == 1
    assert stats["tool_pic_selected_tokens"] == 80
    assert stats["tool_pic_reused_tokens"] == 120
    assert stats["tool_pic_repaired_tokens"] == 4
    assert stats["tool_pic_memory_admitted"] == 1
    assert stats["tool_pic_projected_bytes"] == 123_456
    assert stats["prompt_state_approximate"] == 1
    assert stats["constraint_profile"] == "required_tool"


def test_vision_prompt_cache_transfers_single_owner(monkeypatch):
    monkeypatch.setenv("VMODEL_VISION_PROMPT_CACHE", "1")
    engine = SimpleNamespace(governor=None)
    kv = SimpleNamespace(offset=3, trim=lambda length: None)
    logits = mx.array([1.0])
    key = ((1, 2, 3), ((b"image", (16, 16)),))

    assert qwen3vl._store_vision_prompt_cache(
        engine, key, kv, logits, 3)
    loaded = qwen3vl._take_vision_prompt_cache(engine, key, 3)
    assert loaded[:3] == (kv, logits, 3)
    assert loaded[3]["tokens"] == (1, 2, 3)
    assert engine._vision_prompt_cache is None
    assert qwen3vl._take_vision_prompt_cache(engine, key, 3) is None


def test_vision_prompt_cache_retains_same_image_edit_as_pic_source(monkeypatch):
    monkeypatch.setenv("VMODEL_VISION_PROMPT_CACHE", "1")
    engine = SimpleNamespace(governor=None)
    kv = SimpleNamespace(offset=3, trim=lambda length: None)
    logits = mx.array([1.0])
    engine._vision_prompt_cache = (((1, 2, 3), (b"first",)), kv, logits)

    edited = qwen3vl._take_vision_prompt_cache(
        engine, ((1, 2, 4), (b"first",)), 3)
    assert edited[:3] == (kv, logits, 3)
    assert edited[3]["tokens"] == (1, 2, 3)
    assert engine._vision_prompt_cache is None

    engine._vision_prompt_cache = (((1, 2, 3), (b"first",)), kv, logits)
    assert qwen3vl._take_vision_prompt_cache(
        engine, ((1, 2, 3), (b"second",)), 3) is None


def test_vision_prompt_cache_accepts_exact_text_extension(monkeypatch):
    monkeypatch.setenv("VMODEL_VISION_PROMPT_CACHE", "1")
    engine = SimpleNamespace(governor=None)
    kv = SimpleNamespace(offset=3, trim=lambda length: None)
    logits = mx.array([1.0])
    image_key = ((b"same-image", (16, 16)),)
    cached_key = ((1, 2, 3), image_key)
    extended_key = ((1, 2, 3, 4, 5), image_key)
    engine._vision_prompt_cache = (cached_key, kv, logits)

    loaded = qwen3vl._take_vision_prompt_cache(engine, extended_key, 5)
    assert loaded[:3] == (kv, logits, 3)
    assert loaded[3]["tokens"] == (1, 2, 3)


def test_hybrid_prompt_endpoint_snapshot_is_not_advanced_by_decode():
    kv = KVCache(2)
    kv.keys[1] = mx.array([[[[1.0], [2.0], [3.0]]]])
    kv.values[1] = mx.array([[[[4.0], [5.0], [6.0]]]])
    kv.kda_cache = KDAStateCache(2)
    kv.kda_cache.set_state(0, mx.array([[[[7.0]]]]))
    kv.kda_cache.set_conv_history(0, (mx.array([[[8.0]]]),))

    snapshot = qwen3vl._fork_hybrid_prompt_endpoint(kv)
    kv.update(1, mx.array([[[[9.0]]]]), mx.array([[[[10.0]]]]))
    kv.kda_cache.set_state(0, mx.array([[[[11.0]]]]))
    mx.eval(*kv.keys[1:], *kv.values[1:])

    assert snapshot.offset == 3
    assert snapshot.keys[1].tolist() == [[[[1.0], [2.0], [3.0]]]]
    assert snapshot.kda_cache.state(0).item() == 7.0
    assert snapshot.kda_cache.conv_history(0)[0].item() == 8.0


def test_prompt_kv_skips_tower_only_for_exact_or_text_suffix():
    cfg = SimpleNamespace(image_token_id=10, video_token_id=11)
    cached = (object(), object(), 3, {"tokens": (1, 2, 3)})

    assert qwen3vl._vision_prompt_cache_mode(cached, [1, 2, 3], cfg) == "exact"
    assert qwen3vl._vision_prompt_cache_mode(
        cached, [1, 2, 3, 4, 5], cfg) == "text_suffix"
    assert qwen3vl._vision_prompt_cache_mode(
        cached, [1, 2, 3, 10], cfg) is None
    assert qwen3vl._vision_prompt_cache_mode(
        cached, [1, 2, 3, 11], cfg) is None


def test_multimodal_expansion_maps_only_text_tool_capsules():
    class Prompt(str):
        token_ids = (1, 2, 9, 3, 4)
        tool_capsules = (("tool", 0, 2),)

    tokens, boundaries = qwen3vl._expand_multimodal_tokens_with_boundaries(
        list(Prompt.token_ids), 9, [3], 10, [])

    assert tokens == [1, 2, 9, 9, 9, 3, 4]
    assert qwen3vl._expanded_tool_capsules(Prompt("x"), boundaries) == (
        ("tool", 0, 2),)

    Prompt.tool_capsules = (("contains-image", 1, 4),)
    assert qwen3vl._expanded_tool_capsules(Prompt("x"), boundaries) == ()


@pytest.mark.parametrize("value", ["invalid", "-1"])
def test_invalid_cache_capacity_fails_closed(monkeypatch, value):
    monkeypatch.setenv("VMODEL_VISION_CACHE_ENTRIES", value)
    with pytest.raises(ValueError, match="VMODEL_VISION_CACHE_ENTRIES"):
        qwen3vl._cached_vision_forward(
            _engine(), Image.new("RGB", (32, 32)), (16, 16), {})
