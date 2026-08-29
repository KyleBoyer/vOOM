"""Small released-geometry gates for GLM-5.3's native vision backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _engine_config(*, depth=1):
    return SimpleNamespace(
        vision_config={
            "depth": depth,
            "hidden_size": 8,
            "num_heads": 2,
            "intermediate_size": 12,
            "out_hidden_size": 8,
            "projection_intermediate_size": 16,
            "patch_size": 1,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "rms_norm_eps": 1e-5,
            "swiglu_limit": 10.0,
            "min_image_tokens": 1,
            "max_image_tokens": 4,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
        },
    )


def test_glm53_smart_resize_matches_released_minimum_and_cap():
    from runtime.glm5_next_vision import smart_resize

    # Released defaults upscale a 64x64 image to 112x112 (8x8 ViT patches),
    # then the runtime's 4096-patch safety cap maps to at most 1024 merged
    # image tokens without changing the model weights or vision arithmetic.
    assert smart_resize(2, 64, 64) == (112, 112)
    assert smart_resize(
        2, 4000, 4000, max_image_tokens=1024) == (896, 896)


def test_glm53_preprocess_uses_released_channel_temporal_patch_order():
    from PIL import Image

    from runtime.glm5_next_vision import preprocess_image

    cfg = _engine_config(depth=0)
    cfg.vision_config.update({
        "patch_size": 14,
        "temporal_patch_size": 2,
        "min_image_tokens": 1,
        "max_image_tokens": 1,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
    })
    engine = SimpleNamespace(cfg=cfg, rc=SimpleNamespace(vision_max_patches=4))
    image = Image.new("RGB", (28, 28), color=(255, 128, 0))
    patches = preprocess_image(engine, image, (2, 2))

    assert patches.shape == (4, 3 * 2 * 14 * 14)
    first = patches[0].reshape(3, 2, 14, 14)
    np.testing.assert_allclose(first[0], 1.0)
    np.testing.assert_allclose(first[1], 128 / 255, rtol=0, atol=1e-7)
    np.testing.assert_allclose(first[2], 0.0)


def test_glm53_vision_tiny_forward_is_finite_and_merged():
    import mlx.core as mx

    from runtime.glm5_next_vision import V, vision_forward

    cfg = _engine_config()
    engine = SimpleNamespace(cfg=cfg)
    rng = np.random.default_rng(17)

    def weight(shape, scale=0.04):
        return mx.array(
            rng.normal(0, scale, size=shape).astype(np.float32),
            dtype=mx.bfloat16)

    weights = {
        f"{V}.patch_embed.proj.weight": weight((8, 3, 1, 1)),
        f"{V}.patch_embed.proj.bias": weight((8,)),
        f"{V}.post_layernorm.weight": mx.ones((8,), dtype=mx.bfloat16),
        f"{V}.downsample.weight": weight((8, 8, 2, 2)),
        f"{V}.downsample.bias": weight((8,)),
        f"{V}.merger.proj.weight": weight((8, 8)),
        f"{V}.merger.post_projection_norm.weight": mx.ones(
            (8,), dtype=mx.bfloat16),
        f"{V}.merger.post_projection_norm.bias": mx.zeros(
            (8,), dtype=mx.bfloat16),
        f"{V}.merger.gate_proj.weight": weight((16, 8)),
        f"{V}.merger.up_proj.weight": weight((16, 8)),
        f"{V}.merger.down_proj.weight": weight((8, 16)),
    }
    prefix = f"{V}.blocks.0"
    weights.update({
        f"{prefix}.norm1.weight": mx.ones((8,), dtype=mx.bfloat16),
        f"{prefix}.norm2.weight": mx.ones((8,), dtype=mx.bfloat16),
        f"{prefix}.attn.qkv.weight": weight((24, 8)),
        f"{prefix}.attn.qkv.bias": weight((24,)),
        f"{prefix}.attn.q_norm.weight": mx.ones((4,), dtype=mx.bfloat16),
        f"{prefix}.attn.k_norm.weight": mx.ones((4,), dtype=mx.bfloat16),
        f"{prefix}.attn.proj.weight": weight((8, 8)),
        f"{prefix}.attn.proj.bias": weight((8,)),
        f"{prefix}.mlp.gate_proj.weight": weight((12, 8)),
        f"{prefix}.mlp.gate_proj.bias": weight((12,)),
        f"{prefix}.mlp.up_proj.weight": weight((12, 8)),
        f"{prefix}.mlp.up_proj.bias": weight((12,)),
        f"{prefix}.mlp.down_proj.weight": weight((8, 12)),
        f"{prefix}.mlp.down_proj.bias": weight((8,)),
    })
    pixels = rng.normal(size=(4, 3)).astype(np.float32)
    output = vision_forward(engine, pixels, (2, 2), weights)
    mx.eval(output)

    assert output.shape == (1, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_server_admits_only_implemented_glm53_vision_backend():
    from runtime.server import _vision_request_error

    cfg = SimpleNamespace(
        vision_config={"model_type": "glm5_next_vision"},
        vision_backend="glm5_next", model_type="glm5_next")
    assert _vision_request_error(cfg, "GLM-5.3-Flash") is None
