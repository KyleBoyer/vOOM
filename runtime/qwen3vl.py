"""Qwen3-VL vision path: preprocessor, config-driven ViT with learned-position
bilinear interpolation and 2D RoPE, patch mergers, DeepStack injection, and an
interleaved-M-RoPE prefill that splices image embeddings into the token
stream. Decode after prefill reduces to the standard dense path (equal t/h/w
positions == standard rope), so the engine's sweep serves it unchanged.

Reference: Transformers ``modeling_qwen3_vl``. Vision weights stay
BF16-as-released and use the same governor-controlled cache as text pages.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import OrderedDict
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import layer_runner
from .kv_cache import KVCache
from .sampler import SamplingParams, sample
from .vision_positions import (build_multimodal_positions, image_grid_size,
                               MAX_GLOBAL_VISION_PATCHES,
                               MAX_RETAINED_VISION_TOKENS,
                               validate_global_attention_grids)

V = "model.visual"
_VISION_CACHE_MARGIN_BYTES = 400_000_000
_DEFAULT_VISION_CACHE_ENTRIES = 4


# ---------------------------------------------------------------- preprocess

def preprocess_image(img, patch: int = 16, merge: int = 2, tps: int = 2,
                     min_pixels: int = 65536, max_pixels: int = 16777216):
    """PIL image -> (patches (N, C*tps*patch*patch) float32, grid_h, grid_w).
    Patches are emitted in merge-block order (2x2 blocks contiguous), matching
    the Qwen2/3-VL processor."""
    img = img.convert("RGB")
    w, h = img.size
    gh, gw = image_grid_size(
        w, h, patch=patch, merge=merge,
        min_pixels=min_pixels, max_pixels=max_pixels)
    nw, nh = gw * patch, gh * patch
    img = img.resize((nw, nh))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5  # image_mean/std = 0.5
    x = np.transpose(x, (2, 0, 1))  # (C, H, W)
    x = np.stack([x] * tps, axis=1)  # duplicate temporal: (C, tps, H, W)
    # (C, tps, gh/m, m, p, gw/m, m, p) -> merge-block order
    x = x.reshape(3, tps, gh // merge, merge, patch, gw // merge, merge, patch)
    x = np.transpose(x, (2, 5, 3, 6, 0, 1, 4, 7))  # (ghb, gwb, m, m, C, tps, p, p)
    x = x.reshape(gh * gw, 3 * tps * patch * patch)
    return x, gh, gw


def preprocess_video(video, gh: int, gw: int, patch: int = 16,
                     merge: int = 2, tps: int = 2):
    """Sampled PIL frames -> official Qwen temporal/spatial patch order."""
    frames = list(video.frames)
    if not frames:
        raise ValueError("video has no sampled frames")
    if len(frames) % tps:
        frames.extend(frames[-1].copy() for _ in range(tps - len(frames) % tps))
    arrays = []
    for frame in frames:
        rgb = frame.convert("RGB").resize((gw * patch, gh * patch))
        value = np.asarray(rgb, dtype=np.float32) / 255.0
        arrays.append((value - 0.5) / 0.5)
    x = np.stack(arrays)                    # (T,H,W,C)
    x = np.transpose(x, (0, 3, 1, 2))      # (T,C,H,W)
    gt = len(frames) // tps
    x = x.reshape(
        gt, tps, 3, gh // merge, merge, patch,
        gw // merge, merge, patch)
    x = np.transpose(x, (0, 3, 6, 4, 7, 2, 1, 5, 8))
    x = x.reshape(gt * gh * gw, 3 * tps * patch * patch)
    return x, gt, gh, gw


@lru_cache(maxsize=32)
def _patch_positions(gh: int, gw: int, merge: int = 2):
    """(h, w) full-grid coordinates per patch, in merge-block order."""
    hs, ws = [], []
    for hb in range(gh // merge):
        for wb in range(gw // merge):
            for i in range(merge):
                for j in range(merge):
                    hs.append(hb * merge + i)
                    ws.append(wb * merge + j)
    return np.array(hs), np.array(ws)


@lru_cache(maxsize=32)
def _vision_position_data(gh: int, gw: int, merge: int, side: int, hd: int,
                          gt: int = 1):
    """Small, model-independent interpolation and 2D-RoPE geometry."""
    hs, ws = _patch_positions(gh, gw, merge)
    hc = hs * (side - 1) / max(gh - 1, 1)
    wc = ws * (side - 1) / max(gw - 1, 1)
    h0, w0 = np.floor(hc).astype(np.int32), np.floor(wc).astype(np.int32)
    h1 = np.minimum(h0 + 1, side - 1)
    w1 = np.minimum(w0 + 1, side - 1)
    fh, fw = hc - h0, wc - w0
    indices = np.stack((
        h0 * side + w0,
        h0 * side + w1,
        h1 * side + w0,
        h1 * side + w1,
    ), axis=1)
    coefficients = np.stack((
        (1 - fh) * (1 - fw),
        (1 - fh) * fw,
        fh * (1 - fw),
        fh * fw,
    ), axis=1)

    inv = 1.0 / (10000.0 ** (np.arange(0, hd // 4) / (hd // 4)))
    angles = np.concatenate(
        (hs[:, None] * inv[None, :], ws[:, None] * inv[None, :]), axis=1)
    rope = np.concatenate((angles, angles), axis=1)
    values = (
        mx.array(np.tile(indices, (gt, 1))),
        mx.array(np.tile(coefficients, (gt, 1)).astype(np.float32)),
        mx.array(np.tile(np.cos(rope), (gt, 1))).astype(mx.bfloat16),
        mx.array(np.tile(np.sin(rope), (gt, 1))).astype(mx.bfloat16),
    )
    mx.eval(*values)
    return values


def _vision_cache_capacity() -> int:
    raw = os.environ.get(
        "VMODEL_VISION_CACHE_ENTRIES", str(_DEFAULT_VISION_CACHE_ENTRIES))
    try:
        capacity = int(raw)
    except ValueError as error:
        raise ValueError("VMODEL_VISION_CACHE_ENTRIES must be an integer") from error
    if capacity < 0:
        raise ValueError("VMODEL_VISION_CACHE_ENTRIES must be >= 0")
    return capacity


def _vision_prompt_cache_enabled() -> bool:
    raw = os.environ.get("VMODEL_VISION_PROMPT_CACHE", "1").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        "VMODEL_VISION_PROMPT_CACHE must be a boolean (0/1, false/true)")


def _vision_max_patches(engine) -> int:
    vcfg = engine.cfg.vision_config
    patch = int(vcfg["patch_size"])
    configured = int(getattr(getattr(engine, "rc", None),
                             "vision_max_patches", 0) or 0)
    max_patches = configured or MAX_GLOBAL_VISION_PATCHES
    min_patches = (256 // patch) ** 2
    if max_patches < min_patches or max_patches > MAX_GLOBAL_VISION_PATCHES:
        raise ValueError(
            f"vision_max_patches must be in [{min_patches}, "
            f"{MAX_GLOBAL_VISION_PATCHES}]")
    return max_patches


def _vision_max_pixels(engine) -> int:
    patch = int(engine.cfg.vision_config["patch_size"])
    return _vision_max_patches(engine) * patch * patch


def _video_grid_size(engine, video) -> tuple[int, int, int]:
    vcfg = engine.cfg.vision_config
    patch = int(vcfg["patch_size"])
    merge = int(vcfg["spatial_merge_size"])
    tps = int(vcfg["temporal_patch_size"])
    num_frames = len(video.frames)
    if num_frames <= 0:
        raise ValueError("video has no sampled frames")
    gt = math.ceil(num_frames / tps)
    padded_frames = gt * tps

    width, height = video.frames[0].size
    factor = patch * merge
    if height < factor or width < factor:
        raise ValueError(
            f"video frame height:{height} or width:{width} must be at least "
            f"the patch/merge factor:{factor}")
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(
            f"video frame aspect ratio must be at most 200, got {aspect_ratio}")

    # The checkpoint processor applies its min/max bounds to T*H*W. Intersect
    # that released budget with the runtime's independent safety limits:
    # global attention sees one spatial segment, while the merged vision
    # embeddings for all ``gt`` segments remain live through text prefill.
    max_spatial_patches = min(
        _vision_max_patches(engine),
        MAX_RETAINED_VISION_TOKENS * merge * merge // gt,
    )
    if max_spatial_patches < merge * merge:
        raise ValueError(
            f"{len(video.frames)} sampled frames exceed the video patch budget; "
            "reduce VMODEL_VIDEO_MAX_FRAMES")
    video_min_pixels = getattr(engine.cfg, "video_min_pixels", 4_096)
    video_max_pixels = getattr(engine.cfg, "video_max_pixels", 25_165_824)
    for value, label in ((video_min_pixels, "video_min_pixels"),
                         (video_max_pixels, "video_max_pixels")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    safe_max_pixels = padded_frames * max_spatial_patches * patch * patch
    max_pixels = min(video_max_pixels, safe_max_pixels)
    if max_pixels < video_min_pixels:
        raise ValueError("video pixel minimum exceeds the active safety budget")
    if max_pixels < padded_frames * factor * factor:
        raise ValueError("video cannot fit one merged patch within the active budget")

    # Mirrors transformers.models.qwen3_vl.video_processing_qwen3_vl.smart_resize.
    # Its beta uses the sampled (unpadded) frame count, while the branch tests
    # use the temporal-patch-aligned count.
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    aligned_pixels = padded_frames * resized_height * resized_width
    source_pixels = num_frames * height * width
    if aligned_pixels > max_pixels:
        beta = math.sqrt(source_pixels / max_pixels)
        resized_height = max(
            factor, math.floor(height / beta / factor) * factor)
        resized_width = max(
            factor, math.floor(width / beta / factor) * factor)
    elif aligned_pixels < video_min_pixels:
        beta = math.sqrt(video_min_pixels / source_pixels)
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    gh, gw = resized_height // patch, resized_width // patch
    if (padded_frames * resized_height * resized_width > max_pixels
            or gh * gw > max_spatial_patches
            or gt * (gh // merge) * (gw // merge)
            > MAX_RETAINED_VISION_TOKENS):
        raise ValueError("official video resize exceeds the active vision budget")
    return gt, gh, gw


def _vision_cache_key(image, grid) -> tuple[bytes, tuple]:
    """Content identity after the same RGB conversion used by preprocessing."""
    digest = hashlib.sha256()
    if getattr(image, "is_video", False):
        digest.update(float(image.fps).hex().encode())
        digest.update(repr(tuple(image.frame_indices)).encode())
        frames = image.frames
    else:
        frames = (image,)
    for frame in frames:
        rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
        digest.update(rgb.width.to_bytes(8, "little"))
        digest.update(rgb.height.to_bytes(8, "little"))
        digest.update(rgb.tobytes())
    return digest.digest(), tuple(grid)


def _vision_cache(engine) -> OrderedDict:
    cache = getattr(engine, "_vision_embedding_cache", None)
    if cache is None:
        cache = OrderedDict()
        engine._vision_embedding_cache = cache
    return cache


def _vision_cache_has_headroom(engine) -> bool:
    governor = getattr(engine, "governor", None)
    if governor is None:
        return True
    return mx.get_active_memory() + _VISION_CACHE_MARGIN_BYTES <= governor.current_ceiling()


def _load_vision_weights(store, cache=None):
    names = store.names_with_prefix(V + ".")
    if cache is not None:
        return cache.get(
            "qwen3vl:vision:released", names, apply_transform=False)
    weights, _, _ = store.fetch(names)
    return weights


def _cached_vision_forward(engine, image, expected_grid, vcfg: dict,
                           vision_weights=None, *, max_pixels: int = 16_777_216,
                           cache_key=None):
    """Return exact evaluated vision outputs, reusing content-identical images.

    The cache is owned by one model-specific engine, so a model/mode swap drops
    it. Admission is rechecked against the live governor rather than reserving a
    fixed memory allowance; entries are already-active outputs, not a second
    copy of the vision tower or its scratch state.
    """
    capacity = _vision_cache_capacity()
    cache = _vision_cache(engine)
    if not _vision_cache_has_headroom(engine):
        cache.clear()
        mx.clear_cache()
    key = cache_key or _vision_cache_key(image, expected_grid)
    if capacity:
        cached = cache.pop(key, None)
        if cached is not None:
            cache[key] = cached
            return (*cached, True, vision_weights)

    if getattr(image, "is_video", False):
        gt, gh, gw = expected_grid
        px, actual_t, actual_h, actual_w = preprocess_video(
            image, gh, gw, patch=vcfg["patch_size"],
            merge=vcfg["spatial_merge_size"],
            tps=vcfg["temporal_patch_size"])
        if (actual_t, actual_h, actual_w) != tuple(expected_grid):
            raise ValueError("video dimensions changed after vision preflight")
    else:
        gt = 1
        px, gh, gw = preprocess_image(
            image,
            patch=int(vcfg.get("patch_size", 16)),
            merge=int(vcfg.get("spatial_merge_size", 2)),
            tps=int(vcfg.get("temporal_patch_size", 2)),
            max_pixels=max_pixels,
        )
        if (gh, gw) != tuple(expected_grid):
            raise ValueError("image dimensions changed after vision preflight")
    if vision_weights is None:
        vision_weights = _load_vision_weights(
            engine.store, getattr(engine, "cache", None))
    if getattr(image, "is_video", False):
        embeds, deepstack = vision_forward(
            engine.store, px, gh, gw, vcfg, weights=vision_weights, gt=gt)
    else:
        embeds, deepstack = vision_forward(
            engine.store, px, gh, gw, vcfg, weights=vision_weights)
    if capacity and _vision_cache_has_headroom(engine):
        cache[key] = (embeds, deepstack)
        while len(cache) > capacity:
            cache.popitem(last=False)
    return embeds, deepstack, False, vision_weights


def _take_vision_prompt_cache(engine, key, prompt_tokens: int):
    entry = getattr(engine, "_vision_prompt_cache", None)
    engine._vision_prompt_cache = None
    if not _vision_prompt_cache_enabled() or entry is None:
        return None
    if len(entry) == 3:  # checkpoint generations predating PIC metadata
        cached_key, kv, logits = entry
        metadata = {}
    else:
        cached_key, kv, logits, metadata = entry
    cached_tokens, cached_images = cached_key
    tokens, images = key
    prefix = len(cached_tokens)
    if cached_images != images or prefix <= 0 or kv.offset != prefix:
        return None
    return kv, logits, prefix, {
        **metadata,
        "tokens": tuple(cached_tokens),
    }


def _store_vision_prompt_cache(engine, key, kv, logits, prompt_tokens: int, *,
                               tool_capsules=(), pos3=None,
                               approximate: bool = False) -> bool:
    if not _vision_prompt_cache_enabled() or not _vision_cache_has_headroom(engine):
        engine._vision_prompt_cache = None
        return False
    kv.trim(prompt_tokens)
    mx.eval(logits)
    engine._vision_prompt_cache = (key, kv, logits, {
        "tool_capsules": tuple(tool_capsules),
        "pos3": (np.array(pos3, copy=True) if pos3 is not None else None),
        "approximate": bool(approximate),
    })
    return True


def _vision_prompt_cache_mode(cached_prompt, tokens, cfg) -> str | None:
    """Whether retained multimodal KV can serve this prompt without the tower."""
    if cached_prompt is None:
        return None
    prefix = int(cached_prompt[2])
    source_tokens = tuple(cached_prompt[3].get("tokens", ()))
    if (prefix > len(tokens) or len(source_tokens) != prefix
            or tuple(tokens[:prefix]) != source_tokens):
        return None
    if prefix == len(tokens):
        return "exact"
    vision_tokens = {cfg.image_token_id, cfg.video_token_id} - {0}
    if not any(token in vision_tokens for token in tokens[prefix:]):
        return "text_suffix"
    return None


# ------------------------------------------------------------------ ViT

def _ln(x, w, b, eps=1e-6):
    return mx.fast.layer_norm(x, w, b, eps)


def _rot_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _segmented_vision_attention(q, k, v, gt: int, *, scale: float):
    """Attend independently within each temporal patch/frame segment.

    Hugging Face supplies one ``cu_seqlens`` boundary per temporal grid item;
    batching those equal-sized segments gives the same result without allowing
    a frame to attend to patches from another frame. It also changes video ViT
    attention from O((T*HW)^2) to O(T*(HW)^2).
    """
    if gt <= 0 or q.shape[-2] % gt:
        raise ValueError("vision patch count must divide into temporal segments")
    if gt == 1:
        return mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=scale)[0]
    heads, patches, head_dim = q.shape
    frame_patches = patches // gt

    def segments(value):
        return value.reshape(heads, gt, frame_patches, head_dim).transpose(
            1, 0, 2, 3)

    attended = mx.fast.scaled_dot_product_attention(
        segments(q), segments(k), segments(v), scale=scale)
    return attended.transpose(1, 0, 2, 3).reshape(heads, patches, head_dim)


def vision_forward(store, pixels: np.ndarray, gh: int, gw: int, vcfg: dict, *,
                   weights=None, gt: int = 1):
    """Run the ViT. Returns (embeds (n_merged, out_hidden), deepstack list of
    3 x (n_merged, out_hidden)). Loads visual weights on demand via the store."""
    depth, hid = vcfg["depth"], vcfg["hidden_size"]
    heads, merge = vcfg["num_heads"], vcfg["spatial_merge_size"]
    hd = hid // heads
    w = weights if weights is not None else _load_vision_weights(store)

    x = mx.array(pixels).astype(mx.bfloat16) @ \
        w[f"{V}.patch_embed.proj.weight"].reshape(hid, -1).T + w[f"{V}.patch_embed.proj.bias"]

    # learned pos-embed, bilinearly interpolated from the SxS training grid
    side = int(math.sqrt(vcfg["num_position_embeddings"]))
    indices, coefficients, cos, sin = _vision_position_data(
        gh, gw, merge, side, hd, gt)
    pe = w[f"{V}.pos_embed.weight"].astype(mx.float32)
    interp = (pe[indices] * coefficients[..., None]).sum(axis=1)
    x = x + interp.astype(mx.bfloat16)

    # 2D rope: half the rotary dims carry h-position, half w-position

    def merger(h_in, prefix):
        wn = w[f"{prefix}.norm.weight"]
        if wn.shape[0] == hid:  # pre-shuffle norm (main merger)
            m = _ln(h_in, wn, w[f"{prefix}.norm.bias"]).reshape(-1, hid * merge * merge)
        else:  # post-shuffle norm (deepstack mergers)
            m = _ln(h_in.reshape(-1, hid * merge * merge), wn, w[f"{prefix}.norm.bias"])
        m = layer_runner._linear(m, w, f"{prefix}.linear_fc1")
        m = 0.5 * m * (1 + mx.tanh(0.7978845608 * (m + 0.044715 * m ** 3)))
        return layer_runner._linear(m, w, f"{prefix}.linear_fc2")

    deepstack = {i: None for i in vcfg["deepstack_visual_indexes"]}
    n = x.shape[0]
    for li in range(depth):
        p = f"{V}.blocks.{li}"
        hln = _ln(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"])
        qkv = layer_runner._linear(hln, w, f"{p}.attn.qkv")
        q, k, v = mx.split(qkv.reshape(n, 3, heads, hd).transpose(1, 2, 0, 3), 3, axis=0)
        q, k, v = q[0], k[0], v[0]  # (heads, N, hd)
        q = q * cos[None] + _rot_half(q) * sin[None]
        k = k * cos[None] + _rot_half(k) * sin[None]
        attn = _segmented_vision_attention(
            q, k, v, gt, scale=hd ** -0.5)
        attn = attn.transpose(1, 0, 2).reshape(n, hid)
        x = x + layer_runner._linear(attn, w, f"{p}.attn.proj")
        hln = _ln(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"])
        m = layer_runner._linear(hln, w, f"{p}.mlp.linear_fc1")
        m = 0.5 * m * (1 + mx.tanh(0.7978845608 * (m + 0.044715 * m ** 3)))
        x = x + layer_runner._linear(m, w, f"{p}.mlp.linear_fc2")
        if li in deepstack:
            ix = vcfg["deepstack_visual_indexes"].index(li)
            deepstack[li] = merger(x, f"{V}.deepstack_merger_list.{ix}")
    out = merger(x, f"{V}.merger")
    mx.eval(out, *[d for d in deepstack.values()])
    return out, [deepstack[i] for i in vcfg["deepstack_visual_indexes"]]


# ------------------------------------------------------ M-RoPE LLM prefill

def _mrope_cos_sin(pos3: np.ndarray, head_dim: int, theta: float, sections):
    """pos3: (3, L) t/h/w positions -> rotate-half cos/sin (L, head_dim) with
    interleaved sections: freq index i belongs to H if i%3==1 (i < 3*sec_h),
    W if i%3==2 (i < 3*sec_w), else T."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(half) / half))
    comp = np.zeros(half, dtype=int)
    for i in range(half):
        if i % 3 == 1 and i < 3 * sections[1]:
            comp[i] = 1
        elif i % 3 == 2 and i < 3 * sections[2]:
            comp[i] = 2
    ang = pos3[comp].T * inv[None, :]  # (L, half): position of each dim's component
    emb = np.concatenate([ang, ang], axis=1)
    return (mx.array(np.cos(emb)).astype(mx.bfloat16),
            mx.array(np.sin(emb)).astype(mx.bfloat16))


def vl_prefill(engine, tokens: list[int], image_embeds, deepstack, pos3, kv: KVCache):
    """Prefill with image embeddings spliced at image_token positions,
    interleaved M-RoPE, and DeepStack adds after the first LLM layers."""
    cfg = engine.cfg
    vision_tokens = {cfg.image_token_id, cfg.video_token_id} - {0}
    is_img = np.isin(np.array(tokens), list(vision_tokens))
    x = engine._embed(list(tokens))
    if is_img.any():
        xnp = x  # splice on the mx array via indices
        idx = mx.array(np.nonzero(is_img)[0])
        x = mx.zeros_like(xnp) + xnp
        x[0, idx, :] = image_embeds.astype(x.dtype)
    sections = cfg.rope_scaling["mrope_section"]
    cos, sin = _mrope_cos_sin(pos3, cfg.head_dim, cfg.rope_theta, sections)
    img_idx = mx.array(np.nonzero(is_img)[0]) if is_img.any() else None

    n = cfg.num_hidden_layers
    # MLX's native causal SDPA is lower-right aligned when K is longer than Q,
    # so the same form is exact for full and cached-suffix prefill. It avoids an
    # explicit O(L^2) mask allocation; live layer activations and KV still grow
    # linearly with the admitted expanded prompt length and remain bounded by
    # the active context limit plus the engine's memory governor.
    mask = "causal"
    for i in range(n):
        w = engine.cache.get(engine._layer_key(i), engine._layer_names(i))
        p = f"model.layers.{i}"
        x = _vl_text_block(x, w, p, cfg, kv, i, cos, sin, mask)
        if img_idx is not None and i < len(deepstack):
            add = mx.zeros_like(x[0])
            add[img_idx, :] = deepstack[i].astype(x.dtype)
            x = x + add[None]
        mx.eval(x)
    logits = layer_runner.final_logits(
        x, engine._norm_w, engine._lm_head_weight(), cfg.rms_norm_eps)
    mx.eval(logits)
    return logits


def vl_prefill_suffix(engine, tokens: list[int], pos3: np.ndarray, kv: KVCache,
                      prefix_tokens: int):
    """Extend an exact cached multimodal prefix with a text-only suffix."""
    suffix = tokens[prefix_tokens:]
    if not suffix:
        raise ValueError("vision prompt-cache suffix must not be empty")
    cfg = engine.cfg
    if (cfg.image_token_id in suffix
            or (cfg.video_token_id and cfg.video_token_id in suffix)):
        raise ValueError("vision prompt-cache suffix must be text-only")
    x = engine._embed(suffix)
    sections = cfg.rope_scaling["mrope_section"]
    cos, sin = _mrope_cos_sin(
        pos3[:, prefix_tokens:], cfg.head_dim, cfg.rope_theta, sections)
    mask = "causal"
    for layer in range(cfg.num_hidden_layers):
        w = engine.cache.get(engine._layer_key(layer), engine._layer_names(layer))
        x = _vl_text_block(
            x, w, f"model.layers.{layer}", cfg, kv, layer, cos, sin, mask)
        mx.eval(x)
    logits = layer_runner.final_logits(
        x, engine._norm_w, engine._lm_head_weight(), cfg.rms_norm_eps)
    mx.eval(logits)
    return logits


def vl_prefill_with_tool_capsules(
        engine, tokens: list[int], image_embeds, deepstack,
        pos3: np.ndarray, source_kv: KVCache, source_pos3: np.ndarray, plan):
    """Selective Qwen3-VL text-tool recomputation with explicit M-RoPE.

    Vision-token positions are always recomputed (tool spans never include
    media placeholders), with their exact tower embeddings and DeepStack adds.
    Reused text keys are relocated by the full three-axis M-RoPE delta rather
    than assuming KV offset equals semantic position.
    """
    from .tool_capsules import _layout_parts

    cfg = engine.cfg
    if not cfg.vision_config or not cfg.model_type.startswith("qwen3_vl"):
        raise ValueError("vision tool PIC requires Qwen3-VL")
    if source_kv.compressed_mla or source_kv.offset < len(source_pos3.T):
        raise ValueError("vision tool PIC requires complete dense source KV")
    if (source_pos3.shape != (3, source_kv.offset)
            or pos3.shape != (3, len(tokens))):
        raise ValueError("vision tool PIC position metadata mismatch")
    if not plan.selected_positions or plan.selected_positions[-1] != len(tokens) - 1:
        raise ValueError("vision tool PIC must recompute the prompt endpoint")

    selected_np = np.asarray(plan.selected_positions, dtype=np.int64)
    selected_positions = mx.array(selected_np, dtype=mx.int32)
    selected_tokens = [tokens[position] for position in plan.selected_positions]
    x = engine._embed(selected_tokens)
    vision_ids = {cfg.image_token_id, cfg.video_token_id} - {0}
    full_vision_positions = np.nonzero(
        np.isin(np.asarray(tokens), list(vision_ids)))[0]
    ordinal_by_position = {
        int(position): ordinal
        for ordinal, position in enumerate(full_vision_positions)
    }
    selected_vision = [
        (local, ordinal_by_position[int(position)])
        for local, position in enumerate(selected_np)
        if int(position) in ordinal_by_position
    ]
    if selected_vision:
        if image_embeds is None:
            raise ValueError("vision tool PIC is missing tower embeddings")
        local = mx.array([value[0] for value in selected_vision])
        ordinal = mx.array([value[1] for value in selected_vision])
        copied = mx.zeros_like(x) + x
        copied[0, local, :] = image_embeds[ordinal].astype(x.dtype)
        x = copied

    sections = cfg.rope_scaling["mrope_section"]
    selected_cos, selected_sin = _mrope_cos_sin(
        pos3[:, selected_np], cfg.head_dim, cfg.rope_theta, sections)
    destination = KVCache(cfg.num_hidden_layers)
    layout = tuple(_layout_parts(plan, len(tokens)))
    batch = 1
    selected_count = len(plan.selected_positions)

    for layer in range(cfg.num_hidden_layers):
        source_keys = source_kv.keys[layer]
        source_values = source_kv.values[layer]
        if source_keys is None or source_values is None:
            raise ValueError(
                f"vision tool PIC source is missing layer {layer} KV")
        weights = engine.cache.get(
            engine._layer_key(layer), engine._layer_names(layer))
        prefix = f"model.layers.{layer}"
        hidden = mx.fast.rms_norm(
            x, weights[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
        linear = layer_runner._linear
        q = linear(hidden, weights, f"{prefix}.self_attn.q_proj").reshape(
            batch, selected_count, cfg.num_attention_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        k = linear(hidden, weights, f"{prefix}.self_attn.k_proj").reshape(
            batch, selected_count, cfg.num_key_value_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        v = linear(hidden, weights, f"{prefix}.self_attn.v_proj").reshape(
            batch, selected_count, cfg.num_key_value_heads,
            cfg.head_dim).transpose(0, 2, 1, 3)
        q = mx.fast.rms_norm(
            q, weights[f"{prefix}.self_attn.q_norm.weight"],
            cfg.rms_norm_eps)
        k = mx.fast.rms_norm(
            k, weights[f"{prefix}.self_attn.k_norm.weight"],
            cfg.rms_norm_eps)
        q = (q * selected_cos[None, None]
             + _rot_half(q) * selected_sin[None, None])
        k = (k * selected_cos[None, None]
             + _rot_half(k) * selected_sin[None, None])

        key_parts = []
        value_parts = []
        for kind, start, end, source_or_selected in layout:
            width = end - start
            if kind == "selected":
                key_parts.append(
                    k[:, :, source_or_selected:source_or_selected + width, :])
                value_parts.append(
                    v[:, :, source_or_selected:source_or_selected + width, :])
                continue
            old_start = source_or_selected
            old_keys = source_keys[:, :, old_start:old_start + width, :]
            delta = (
                pos3[:, start:end]
                - source_pos3[:, old_start:old_start + width])
            if np.any(delta):
                delta_cos, delta_sin = _mrope_cos_sin(
                    delta, cfg.head_dim, cfg.rope_theta, sections)
                old_keys = (old_keys * delta_cos[None, None]
                            + _rot_half(old_keys) * delta_sin[None, None])
            key_parts.append(old_keys)
            value_parts.append(
                source_values[:, :, old_start:old_start + width, :])
        keys = key_parts[0] if len(key_parts) == 1 else mx.concatenate(
            key_parts, axis=2)
        values = value_parts[0] if len(value_parts) == 1 else mx.concatenate(
            value_parts, axis=2)
        destination.keys[layer] = keys
        destination.values[layer] = values

        key_positions = mx.arange(len(tokens), dtype=mx.int32)[None, :]
        mask = mx.where(
            key_positions <= selected_positions[:, None],
            0.0, float("-inf")).astype(q.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=cfg.head_dim ** -0.5, mask=mask)
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, selected_count, cfg.num_attention_heads * cfg.head_dim)
        x = x + linear(attended, weights, f"{prefix}.self_attn.o_proj")
        hidden = mx.fast.rms_norm(
            x, weights[f"{prefix}.post_attention_layernorm.weight"],
            cfg.rms_norm_eps)
        x = x + layer_runner._swiglu(hidden, weights, f"{prefix}.mlp")
        if selected_vision and layer < len(deepstack):
            local = mx.array([value[0] for value in selected_vision])
            ordinal = mx.array([value[1] for value in selected_vision])
            addition = mx.zeros_like(x[0])
            addition[local, :] = deepstack[layer][ordinal].astype(x.dtype)
            x = x + addition[None]
        mx.eval(x, keys, values)

    logits = layer_runner.final_logits(
        x, engine._norm_w, engine._lm_head_weight(), cfg.rms_norm_eps)
    mx.eval(logits)
    engine._h_window = x[:, -1:, :]
    engine._h_last = x[:, -1:, :]
    return destination, logits


def _vl_text_block(x, w, p, cfg, kv, layer, cos, sin, mask):
    """Qwen3 text block with explicit (mrope) cos/sin — mirrors
    layer_runner.run_block + per-head qk-norm, rotate-half rope."""
    B, L, _ = x.shape
    n_h, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    h = mx.fast.rms_norm(x, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
    lin = layer_runner._linear  # QTensor-aware (fast mode stores q4 weights)
    q = lin(h, w, f"{p}.self_attn.q_proj").reshape(B, L, n_h, hd).transpose(0, 2, 1, 3)
    k = lin(h, w, f"{p}.self_attn.k_proj").reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    v = lin(h, w, f"{p}.self_attn.v_proj").reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    q = mx.fast.rms_norm(q, w[f"{p}.self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k = mx.fast.rms_norm(k, w[f"{p}.self_attn.k_norm.weight"], cfg.rms_norm_eps)
    q = q * cos[None, None] + _rot_half(q) * sin[None, None]
    k = k * cos[None, None] + _rot_half(k) * sin[None, None]
    keys, values = kv.update(layer, k, v)
    attn = mx.fast.scaled_dot_product_attention(q, keys, values, scale=hd ** -0.5, mask=mask)
    attn = attn.transpose(0, 2, 1, 3).reshape(B, L, n_h * hd)
    x = x + lin(attn, w, f"{p}.self_attn.o_proj")
    h = mx.fast.rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
    return x + layer_runner._swiglu(h, w, f"{p}.mlp")


# ------------------------------------------------------------- entry point

def _video_token_expansion(engine, video, grid: tuple[int, int, int]) -> list[int]:
    gt, gh, gw = grid
    cfg = engine.cfg
    merge = cfg.vision_config["spatial_merge_size"]
    tps = cfg.vision_config["temporal_patch_size"]
    frame_tokens = (gh // merge) * (gw // merge)
    indices = list(video.frame_indices)
    while len(indices) < gt * tps:
        indices.append(indices[-1])
    expansion = []
    for temporal in range(gt):
        group = indices[temporal * tps:(temporal + 1) * tps]
        timestamp = sum(group) / len(group) / video.fps
        expansion.extend(engine.tokenizer.encode(
            f"<{timestamp:.1f} seconds>").ids)
        expansion.append(cfg.vision_start_token_id)
        expansion.extend([cfg.video_token_id] * frame_tokens)
        expansion.append(cfg.vision_end_token_id)
    return expansion


def _expand_video_placeholders(tokens: list[int], video_token_id: int,
                               expansions: list[list[int]]) -> list[int]:
    output = []
    index = 0
    for token in tokens:
        if token != video_token_id:
            output.append(token)
            continue
        if index >= len(expansions):
            raise ValueError("prompt has more video placeholders than supplied videos")
        output.extend(expansions[index])
        index += 1
    if index != len(expansions):
        raise ValueError("supplied videos exceed prompt video placeholders")
    return output


def _expand_multimodal_tokens_with_boundaries(
        tokens: list[int], image_token_id: int, image_counts: list[int],
        video_token_id: int, video_expansions: list[list[int]],
) -> tuple[list[int], list[int]]:
    """Expand media placeholders and map every original token boundary.

    The boundary map lets server-produced tool spans survive image/video token
    expansion without retokenizing or guessing offsets. A capsule containing a
    media placeholder is rejected by the caller rather than relabelled.
    """
    output: list[int] = []
    boundaries = [0]
    image_index = 0
    video_index = 0
    for token in tokens:
        if token == image_token_id:
            if image_index >= len(image_counts):
                raise ValueError(
                    "prompt has more image placeholders than supplied images")
            count = int(image_counts[image_index])
            if count <= 0:
                raise ValueError("image merged-patch token count must be positive")
            output.extend([token] * count)
            image_index += 1
        elif video_token_id and token == video_token_id:
            if video_index >= len(video_expansions):
                raise ValueError(
                    "prompt has more video placeholders than supplied videos")
            output.extend(video_expansions[video_index])
            video_index += 1
        else:
            output.append(token)
        boundaries.append(len(output))
    if image_index != len(image_counts):
        raise ValueError("supplied images exceed prompt image placeholders")
    if video_index != len(video_expansions):
        raise ValueError("supplied videos exceed prompt video placeholders")
    return output, boundaries


def _expanded_tool_capsules(prompt_text, boundaries: list[int]):
    capsules = tuple(getattr(prompt_text, "tool_capsules", ()))
    expanded = []
    for value in capsules:
        try:
            identity, start, end = value
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return ()
        if (not isinstance(identity, str) or not identity
                or not 0 <= start < end < len(boundaries)):
            return ()
        mapped_start = boundaries[start]
        mapped_end = boundaries[end]
        if mapped_end - mapped_start != end - start:
            return ()
        expanded.append((identity, mapped_start, mapped_end))
    return tuple(expanded)


def prepare_vl_prompt(engine, prompt_text: str, images) -> dict:
    """CPU-side media preprocessing and exact placeholder expansion.

    The server calls this before opening an SSE response so the expanded image
    token count can be checked against both the checkpoint window and any
    smaller runtime correctness bound. No KV or vision-model state is allocated.
    """
    if not images:
        raise ValueError("vision generation requires at least one image")
    cfg = engine.cfg
    merge = cfg.vision_config["spatial_merge_size"]
    patch = cfg.vision_config["patch_size"]
    max_pixels = _vision_max_pixels(engine)
    grids = []
    image_grids = []
    video_frame_grids = []
    video_expansions = []
    for media in images:
        if getattr(media, "is_video", False):
            grid = _video_grid_size(engine, media)
            grids.append(grid)
            video_frame_grids.extend([(grid[1], grid[2])] * grid[0])
            video_expansions.append(_video_token_expansion(engine, media, grid))
        else:
            grid = image_grid_size(
                *media.size, patch=patch, merge=merge,
                max_pixels=max_pixels)
            grids.append(grid)
            image_grids.append(grid)
    # This runs before any pixel/Metal allocation and, in the HTTP adapters,
    # before streaming headers. The checkpoint processor permits much larger
    # per-segment grids than this runtime's global spatial attention can serve.
    validate_global_attention_grids(grids, merge=merge)
    merged_counts = [(gh // merge) * (gw // merge) for gh, gw in image_grids]
    if video_expansions and (
            not cfg.video_token_id or not cfg.vision_start_token_id
            or not cfg.vision_end_token_id):
        raise ValueError("checkpoint is missing Qwen3-VL video token ids")
    ids = list(getattr(prompt_text, "token_ids", ())
               or engine.tokenizer.encode(prompt_text).ids)
    tokens, boundaries = _expand_multimodal_tokens_with_boundaries(
        ids, cfg.image_token_id, merged_counts,
        cfg.video_token_id, video_expansions)
    return {
        "grids": grids,
        "image_grids": image_grids,
        "video_frame_grids": video_frame_grids,
        "tokens": tokens,
        "tool_capsules": _expanded_tool_capsules(prompt_text, boundaries),
        "max_pixels": max_pixels,
    }


def _lazy_vl_resident_decode_step(engine, token: mx.array, kv: KVCache,
                                  rope_position: int):
    """Build one resident text-trunk step at the compressed M-RoPE offset."""
    x = layer_runner.embed(token.reshape(-1), engine._embed_weight())
    x = engine._sweep(x, kv, offset=rope_position)
    logits = layer_runner.final_logits(
        x, engine._norm_w, engine._lm_head_weight(), engine.cfg.rms_norm_eps)
    return mx.argmax(logits), logits


def _vl_resident_pipeline_ready(engine, kv) -> bool:
    return bool(
        engine.rc.resident_fast_decode
        and not engine.cfg.num_experts
        and getattr(engine, "_embed_rows", None) is None
        and isinstance(kv, KVCache)
        and all(engine.cache.contains(engine._layer_key(layer))
                for layer in range(engine.cfg.num_hidden_layers))
    )


def _vision_path_stats(engine, result: dict, *, prompt_cache_mode: str | None,
                       prompt_cache_lookup_s: float,
                       prompt_state_approximate: bool) -> dict:
    """Map vision-specific execution evidence onto the protocol telemetry schema."""
    tool_pic = int(bool(result.get("vision_tool_pic")))
    cache_source = (
        "vision_tool_pic" if tool_pic else
        "vision_memory" if prompt_cache_mode is not None else
        "cold"
    )
    prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
    cache_stored = int(bool(result.get("vision_prompt_cache_stored")))
    return {
        "prompt_cache_exact_hit": int(bool(
            result.get("vision_prompt_cache_exact_hit"))),
        "prompt_cache_prefix_tokens": int(
            result.get("vision_prompt_cache_prefix_tokens", 0) or 0),
        "prompt_cache_source": cache_source,
        "prompt_cache_lookup_s": float(prompt_cache_lookup_s),
        "prompt_cache_write_tokens": prompt_tokens if cache_stored else 0,
        "vision_cache_hits": int(result.get("vision_cache_hits", 0) or 0),
        "vision_cache_misses": int(result.get("vision_cache_misses", 0) or 0),
        "vision_prompt_cache_hit": int(bool(
            result.get("vision_prompt_cache_hit"))),
        "vision_prompt_cache_tower_skipped": int(
            result.get("vision_prompt_cache_tower_skipped", 0) or 0),
        "vision_prompt_cache_stored": cache_stored,
        "vision_tool_pic": tool_pic,
        "tool_pic": tool_pic,
        "tool_pic_selected_tokens": int(
            result.get("vision_tool_pic_selected_tokens", 0) or 0),
        "tool_pic_reused_tokens": int(
            result.get("vision_tool_pic_reused_tokens", 0) or 0),
        "tool_pic_repaired_tokens": int(
            result.get("vision_tool_pic_repaired_tokens", 0) or 0),
        "tool_pic_prefill_s": (
            float(result.get("prefill_s", 0.0) or 0.0) if tool_pic else 0.0),
        "tool_pic_memory_admitted": int(
            result.get("vision_tool_pic_memory_admitted", 0) or 0),
        "tool_pic_projected_bytes": int(
            result.get("vision_tool_pic_projected_bytes", 0) or 0),
        "prompt_state_approximate": int(bool(prompt_state_approximate)),
        "rope_profile": getattr(engine, "rope_profile", "released"),
        "effective_context_limit": int(getattr(
            engine, "effective_max_position_embeddings", 0) or 0),
        "sampling_profile": result.get("sampling_profile", "greedy"),
        "constraint_profile": result.get("constraint_profile", "none"),
    }


def generate_vl(engine, prompt_text: str, images, max_tokens: int = 64,
                on_token=None, stop=None, on_progress=None,
                *, prepared: dict | None = None,
                sampling: SamplingParams | None = None,
                constraint=None) -> dict:
    """prompt_text: chat-templated text containing one <|image_pad|> per
    image (pre-expansion). images: list of PIL images."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    total_t0 = time.perf_counter()
    sampling = sampling or SamplingParams()
    sampling.seed_rng()
    stop = stop or []
    engine._resident_fast_decode_sweeps = 0
    engine._resident_fast_prefill_sweeps = 0
    engine._true_peak_metal_bytes = mx.get_active_memory()
    # Vision bypasses StreamingEngine.generate(), whose per-layer path resets
    # MLX's process-global peak repeatedly. Start a request-local window here so
    # a previous exact engine cannot inflate a later lossy request's telemetry.
    mx.reset_peak_memory()
    if engine.governor is not None:
        engine.governor.reset_request_peak(engine._true_peak_metal_bytes)
    cfg = engine.cfg
    vcfg = cfg.vision_config
    prepared = prepared or prepare_vl_prompt(engine, prompt_text, images)
    # A direct caller may supply a hand-built ``prepared`` object and bypass
    # prepare_vl_prompt(), so enforce the same bound at the allocation boundary.
    validate_global_attention_grids(
        prepared["grids"], merge=vcfg["spatial_merge_size"])
    tokens = prepared["tokens"]
    limits = []
    for raw_limit in (
        getattr(engine, "effective_max_position_embeddings", 0),
        getattr(getattr(engine, "rc", None), "context_bound", 0),
    ):
        limit = int(raw_limit or 0)
        if limit > 0:
            limits.append(limit)
    active_limit = min(limits) if limits else 0
    if active_limit and len(tokens) + max_tokens > active_limit:
        raise ValueError(
            f"expanded vision prompt({len(tokens)})+max_tokens({max_tokens}) "
            f"exceeds active context limit={active_limit}")

    # A text request on this same vision-capable engine may have left a full KV
    # reachable through engine.last_kv. Vision owns a separate cache below, so
    # enforce single ownership before allocating pixels, embeddings, or new KV.
    from .request_state import release_generation_state

    release_generation_state(engine)
    mx.clear_cache()

    embeds_list, deep_list = [], []
    vision_t0 = time.perf_counter()
    vision_cache_hits = 0
    vision_weights = None
    max_pixels = int(prepared.get("max_pixels", _vision_max_pixels(engine)))
    vision_keys = [
        _vision_cache_key(image, tuple(grid))
        for image, grid in zip(images, prepared["grids"], strict=True)
    ]
    prompt_cache_key = (tuple(tokens), tuple(vision_keys))
    # Consume ownership before allocating new image state. A mismatch drops the
    # old KV immediately; a match keeps one local owner until suffix/full-prefill
    # dispatch below.
    cache_lookup_t0 = time.perf_counter()
    cached_prompt = _take_vision_prompt_cache(
        engine, prompt_cache_key, len(tokens))
    prompt_cache_lookup_s = time.perf_counter() - cache_lookup_t0
    prompt_cache_mode = _vision_prompt_cache_mode(cached_prompt, tokens, cfg)
    # A same-media edited catalog is not an exact prefix hit, but its source KV
    # remains useful to the separately gated PIC path below. Ownership stays
    # local until that admission decision; cold fallback drops it before
    # allocating the destination cache.
    if prompt_cache_mode is None and cached_prompt is not None:
        metadata = cached_prompt[3]
        pic_eligible_metadata = bool(
            engine.rc.tool_pic
            and prepared.get("tool_capsules")
            and metadata.get("tool_capsules")
            and metadata.get("pos3") is not None
            and not metadata.get("approximate", False))
        if not pic_eligible_metadata:
            cached_prompt = None
            mx.clear_cache()
    vision_prompt_cache_tower_skipped = (
        len(images) if prompt_cache_mode is not None else 0)
    if prompt_cache_mode is not None:
        if on_progress is not None:
            on_progress({"phase": "vision", "completed_images": len(images),
                         "total_images": len(images),
                         "cache_source": "vision_prompt_kv"})
    else:
        for image_index, (image, expected_grid) in enumerate(
                zip(images, prepared["grids"], strict=True), start=1):
            e, d, cache_hit, vision_weights = _cached_vision_forward(
                engine, image, tuple(expected_grid), vcfg, vision_weights,
                max_pixels=max_pixels, cache_key=vision_keys[image_index - 1])
            vision_cache_hits += int(cache_hit)
            embeds_list.append(e)
            deep_list.append(d)
            engine._note_true_peak()
            if on_progress is not None:
                on_progress({"phase": "vision", "completed_images": image_index,
                             "total_images": len(images)})
    vision_s = time.perf_counter() - vision_t0
    merge = vcfg["spatial_merge_size"]
    modality_grids = {}
    if prepared.get("image_grids"):
        modality_grids[cfg.image_token_id] = list(prepared["image_grids"])
    if prepared.get("video_frame_grids"):
        modality_grids[cfg.video_token_id] = list(prepared["video_frame_grids"])
    pos3, max_pos = build_multimodal_positions(tokens, modality_grids, merge)
    image_embeds = mx.concatenate(embeds_list, axis=0) if embeds_list else None
    deepstack = [mx.concatenate([d[i] for d in deep_list], axis=0)
                 for i in range(len(deep_list[0]))] if deep_list else []

    pic_plan = None
    pic_source = None
    pic_exact_prefix = 0
    pic_projected_bytes = 0
    pic_memory_admitted = False
    if (prompt_cache_mode is None and cached_prompt is not None
            and engine.rc.tool_pic and prepared.get("tool_capsules")):
        from .tool_capsules import ToolCapsuleSpan, build_pic_plan

        source_kv, _source_logits, source_length, source_meta = cached_prompt
        source_tokens = tuple(source_meta.get("tokens", ()))
        source_capsules = tuple(source_meta.get("tool_capsules", ()))
        source_pos3 = source_meta.get("pos3")
        if (not source_meta.get("approximate", False)
                and source_length == len(source_tokens)
                and source_capsules and source_pos3 is not None):
            lcp = 0
            for old, new in zip(source_tokens, tokens):
                if old != new:
                    break
                lcp += 1
            exact_prefix = min(lcp, max(0, len(tokens) - 1))
            try:
                pic_plan = build_pic_plan(
                    tokens,
                    tuple(ToolCapsuleSpan(*value)
                          for value in prepared["tool_capsules"]),
                    source_tokens,
                    tuple(ToolCapsuleSpan(*value)
                          for value in source_capsules),
                    exact_prefix_tokens=exact_prefix,
                    repair_tokens=engine.rc.tool_pic_repair_tokens)
            except ValueError:
                pic_plan = None
            if pic_plan is not None:
                pic_exact_prefix = exact_prefix
                savings = len(tokens) - pic_plan.selected_tokens
                if (savings < engine.rc.tool_pic_min_savings
                        or pic_plan.selected_tokens >= len(tokens) * 0.99):
                    pic_plan = None
            if pic_plan is not None:
                source_positions = max(1, source_kv.offset)
                destination_bytes = int(
                    source_kv.nbytes() * len(tokens) / source_positions)
                pic_projected_bytes = (
                    destination_bytes
                    + pic_plan.selected_tokens * len(tokens) * 4
                    + int(engine._layer_transient))
                pic_memory_admitted = True
                if engine.governor is not None:
                    pic_memory_admitted = (
                        mx.get_active_memory() + pic_projected_bytes + int(0.4e9)
                        <= engine.governor.current_ceiling())
                if pic_memory_admitted:
                    pic_source = (source_kv, source_pos3)

    vision_prompt_cache_hit = False
    vision_prompt_cache_prefix_tokens = 0
    vision_prompt_cache_exact_hit = False
    vision_tool_pic = False
    vision_tool_pic_selected_tokens = 0
    vision_tool_pic_reused_tokens = 0
    vision_tool_pic_repaired_tokens = 0
    prompt_state_approximate = False
    if prompt_cache_mode == "exact":
        kv, logits, vision_prompt_cache_prefix_tokens, source_meta = cached_prompt
        prompt_state_approximate = bool(source_meta.get("approximate", False))
        vision_prompt_cache_hit = True
        vision_prompt_cache_exact_hit = True
        prefill_s = 0.0
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": len(tokens),
                         "total_tokens": len(tokens)})
    elif prompt_cache_mode == "text_suffix":
        kv, _cached_logits, vision_prompt_cache_prefix_tokens, source_meta = cached_prompt
        prompt_state_approximate = bool(source_meta.get("approximate", False))
        vision_prompt_cache_hit = True
        if on_progress is not None:
            on_progress({"phase": "prefill",
                         "completed_tokens": vision_prompt_cache_prefix_tokens,
                         "total_tokens": len(tokens)})
        t0 = time.perf_counter()
        logits = vl_prefill_suffix(
            engine, tokens, pos3, kv, vision_prompt_cache_prefix_tokens)
        prefill_s = time.perf_counter() - t0
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": len(tokens),
                         "total_tokens": len(tokens)})
    elif pic_source is not None:
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": 0,
                         "total_tokens": len(tokens),
                         "cache_source": "vision_tool_pic"})
        source_kv, source_pos3 = pic_source
        t0 = time.perf_counter()
        try:
            if engine.governor is not None:
                engine.governor.reserve(pic_projected_bytes)
            kv, logits = vl_prefill_with_tool_capsules(
                engine, tokens, image_embeds, deepstack, pos3,
                source_kv, source_pos3, pic_plan)
        except (MemoryError, ValueError) as error:
            print(f"[vision-tool-pic] fallback to full prefill: "
                  f"{type(error).__name__}: {error}", flush=True)
            pic_source = None
        else:
            prefill_s = time.perf_counter() - t0
            prompt_state_approximate = True
            vision_tool_pic = True
            vision_prompt_cache_prefix_tokens = pic_exact_prefix
            vision_tool_pic_selected_tokens = pic_plan.selected_tokens
            vision_tool_pic_reused_tokens = pic_plan.capsule_tokens_reused
            vision_tool_pic_repaired_tokens = pic_plan.capsule_tokens_repaired
            if on_progress is not None:
                on_progress({"phase": "prefill",
                             "completed_tokens": len(tokens),
                             "total_tokens": len(tokens),
                             "cache_source": "vision_tool_pic"})
        if not vision_tool_pic:
            cached_prompt = None
            pic_source = None
            mx.clear_cache()
            kv = KVCache(cfg.num_hidden_layers)
            t0 = time.perf_counter()
            logits = vl_prefill(
                engine, tokens, image_embeds, deepstack, pos3, kv)
            prefill_s = time.perf_counter() - t0
    else:
        cached_prompt = None
        pic_source = None
        mx.clear_cache()
        kv = KVCache(cfg.num_hidden_layers)
        t0 = time.perf_counter()
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": 0,
                         "total_tokens": len(tokens)})
        logits = vl_prefill(engine, tokens, image_embeds, deepstack, pos3, kv)
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": len(tokens),
                         "total_tokens": len(tokens)})
        prefill_s = time.perf_counter() - t0
    prompt_logits = logits
    decode_t0 = time.perf_counter()
    sampled_logits = constraint.mask_logits(logits) if constraint is not None else logits
    next_tok = sample(sampled_logits, sampling)
    if constraint is not None:
        constraint.accept_token(next_tok)
    grammar_completed = bool(
        constraint is not None and constraint.completed)
    generated = [next_tok]
    stop_text = None
    matched_stop_sequence = None
    from .incremental_decode import IncrementalDetokenizer

    text_decoder = IncrementalDetokenizer(engine.tokenizer, stop)

    def accept_decoded_token(token_id: int) -> None:
        nonlocal stop_text, matched_stop_sequence
        delta = text_decoder.push_token(token_id)
        if text_decoder.matched_stop_sequence is not None:
            matched_stop_sequence = text_decoder.matched_stop_sequence
            stop_text = text_decoder.stop_text
        elif on_token is not None and delta:
            on_token(delta)

    accept_decoded_token(next_tok)
    pos = max_pos + 1
    remaining_decode = max_tokens - 1
    pipelined_decode_steps = 0
    can_pipeline = (
        sampling.is_greedy
        and constraint is None
        and stop_text is None
        and remaining_decode > 0
        and next_tok not in cfg.eos_token_ids
        and _vl_resident_pipeline_ready(engine, kv)
    )
    if can_pipeline:
        # The image span compresses the text RoPE position relative to kv.offset,
        # so the ordinary resident helper cannot be reused verbatim. The lazy
        # dependency chain is otherwise identical: construct token N+1 from the
        # unevaluated argmax for N, then synchronize only at the Python boundary.
        boundary = mx.get_active_memory()
        if engine.governor is not None and engine._token_transient:
            engine.governor.reserve(engine._token_transient)
        mx.reset_peak_memory()
        current_token, current_logits = _lazy_vl_resident_decode_step(
            engine, mx.array(next_tok), kv, pos)
        mx.async_eval(current_token, current_logits)

        for index in range(remaining_decode):
            schedule_future = index + 1 < remaining_decode
            if schedule_future:
                future_token, future_logits = _lazy_vl_resident_decode_step(
                    engine, current_token, kv, pos + index + 1)
                mx.async_eval(future_token, future_logits)

            next_tok = int(current_token)
            logits = current_logits
            generated.append(next_tok)
            pipelined_decode_steps += 1
            accept_decoded_token(next_tok)

            terminated = (
                stop_text is not None or next_tok in cfg.eos_token_ids)
            if terminated:
                if schedule_future:
                    mx.eval(future_token, future_logits)
                    kv.trim(len(tokens) + len(generated) - 1)
                break
            if not schedule_future:
                break
            current_token, current_logits = future_token, future_logits

        mx.eval(logits)
        engine._token_transient = max(
            engine._token_transient, mx.get_peak_memory() - boundary)
        engine._note_true_peak()
    elif stop_text is None:
        for _ in range(remaining_decode):
            if grammar_completed or next_tok in cfg.eos_token_ids:
                break
            x = engine._embed([next_tok])
            x = engine._sweep(x, kv, offset=pos)  # equal t/h/w == standard rope
            logits = layer_runner.final_logits(
                x, engine._norm_w, engine._lm_head_weight(), cfg.rms_norm_eps)
            sampled_logits = (
                constraint.mask_logits(logits)
                if constraint is not None else logits)
            next_tok = sample(sampled_logits, sampling)
            if constraint is not None:
                constraint.accept_token(next_tok)
                grammar_completed = bool(constraint.completed)
            pos += 1
            generated.append(next_tok)
            accept_decoded_token(next_tok)
            if stop_text is not None:
                break
    final_delta, final_text = text_decoder.finish_token_stream(
        final_text=stop_text)
    termination_reason = (
        "stop_sequence" if stop_text is not None else
        "grammar" if grammar_completed else
        "eos" if generated[-1] in cfg.eos_token_ids else
        "length"
    )
    if on_token is not None and final_delta:
        on_token(final_delta)
    decode_s = time.perf_counter() - decode_t0
    vision_prompt_cache_stored = _store_vision_prompt_cache(
        engine, prompt_cache_key, kv, prompt_logits, len(tokens),
        tool_capsules=prepared.get("tool_capsules", ()), pos3=pos3,
        approximate=prompt_state_approximate)
    engine._note_true_peak()
    result = {"text": final_text, "tokens": generated,
              "prefill_s": prefill_s, "prompt_tokens": len(tokens),
              "vision_s": vision_s,
              "vision_cache_hits": vision_cache_hits,
              "vision_cache_misses": (
                  len(images) - vision_cache_hits
                  - vision_prompt_cache_tower_skipped),
              "vision_prompt_cache_tower_skipped": (
                  vision_prompt_cache_tower_skipped),
              "video_inputs": sum(
                  int(getattr(media, "is_video", False)) for media in images),
              "video_sampled_frames": sum(
                  len(media.frames) for media in images
                  if getattr(media, "is_video", False)),
              "vision_grids": [list(grid) for grid in prepared["grids"]],
              "vision_prompt_cache_hit": vision_prompt_cache_hit,
              "vision_prompt_cache_exact_hit": vision_prompt_cache_exact_hit,
              "vision_prompt_cache_prefix_tokens": vision_prompt_cache_prefix_tokens,
              "vision_prompt_cache_stored": vision_prompt_cache_stored,
              "vision_tool_pic": vision_tool_pic,
              "vision_tool_pic_selected_tokens": vision_tool_pic_selected_tokens,
              "vision_tool_pic_reused_tokens": vision_tool_pic_reused_tokens,
              "vision_tool_pic_repaired_tokens": vision_tool_pic_repaired_tokens,
              "vision_tool_pic_memory_admitted": int(pic_memory_admitted),
              "vision_tool_pic_projected_bytes": pic_projected_bytes,
              "resident_pipelined_decode_steps": pipelined_decode_steps,
              "sampling_profile": sampling.profile,
              "constraint_profile": getattr(constraint, "profile", "none"),
              "decode_s": decode_s,
              "total_s": time.perf_counter() - total_t0,
              "true_peak_metal_bytes": engine._true_peak_metal_bytes,
              "stopped": stop_text is not None,
              "stop_sequence": matched_stop_sequence,
              "termination_reason": termination_reason}
    result["path_stats"] = _vision_path_stats(
        engine, result, prompt_cache_mode=prompt_cache_mode,
        prompt_cache_lookup_s=prompt_cache_lookup_s,
        prompt_state_approximate=prompt_state_approximate)
    return result
