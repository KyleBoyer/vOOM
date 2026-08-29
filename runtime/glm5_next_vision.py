"""Released GLM-5.3-Flash image tower and multimodal generation path.

The implementation follows Hugging Face's ``modeling_glm5_next`` and
``image_processing_pil_glm5_next`` operator order.  Vision tensors are loaded
only for the tower phase, evaluated, and released before the streamed text
trunk starts; retaining the 1.127-GB tower beside out-of-core text pages is a
large and unnecessary pressure penalty on the 16-GB target Mac.

Images are supported first.  Video requests fail closed until timestamp prompt
construction and temporal packing have an independent released-code oracle.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from . import layer_runner
from .incremental_decode import IncrementalDetokenizer
from .kv_cache import fork_hybrid_kv_endpoint
from .sampler import SamplingParams, sample
from .vision_positions import (MAX_GLOBAL_VISION_PATCHES,
                               MAX_RETAINED_VISION_TOKENS,
                               validate_global_attention_grids)


V = "model.visual"
_VISION_CACHE_ENTRIES = 4


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    """Exact dependency-light port of GLM-5.3's released smart resize."""
    if min(num_frames, height, width, temporal_factor, factor) <= 0:
        raise ValueError("image dimensions and resize factors must be positive")
    if max_image_tokens < min_image_tokens or min_image_tokens <= 0:
        raise ValueError("invalid GLM-5.3 image-token bounds")
    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = max(
        temporal_factor,
        round(num_frames / temporal_factor) * temporal_factor,
    )
    aligned_height = align(height)
    aligned_width = align(width)
    budget = aligned_frames * aligned_height * aligned_width
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        budget = aligned_frames * aligned_height * aligned_width
    if budget > max_pixels:
        if max_pixels < aligned_frames * factor**2:
            raise ValueError("GLM-5.3 image budget cannot fit one aligned patch")
        low, high = 1, height
        best_height = best_width = factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(
                1, math.floor(width * content_height / height))
            candidate_height = align(content_height)
            candidate_width = align(content_width)
            if (aligned_frames * candidate_height * candidate_width
                    <= max_pixels):
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        aligned_height, aligned_width = best_height, best_width
    return aligned_height, aligned_width


def _active_image_token_bounds(engine) -> tuple[int, int]:
    cfg = engine.cfg.vision_config
    merge = int(cfg["spatial_merge_size"])
    configured = int(getattr(engine.rc, "vision_max_patches", 0) or 0)
    patch_cap = configured or MAX_GLOBAL_VISION_PATCHES
    if not 1 <= patch_cap <= MAX_GLOBAL_VISION_PATCHES:
        raise ValueError(
            f"vision_max_patches must be in [1, {MAX_GLOBAL_VISION_PATCHES}]")
    released_min = int(cfg.get("min_image_tokens", 16))
    released_max = int(cfg.get("max_image_tokens", 8000))
    safe_max = min(released_max, patch_cap // (merge * merge))
    if safe_max < released_min:
        raise ValueError("active vision patch cap is below the released minimum")
    return released_min, safe_max


def image_grid(engine, image) -> tuple[int, int]:
    if getattr(image, "is_video", False):
        raise ValueError(
            "GLM-5.3 video input is not yet implemented; image input is supported")
    width, height = image.size
    cfg = engine.cfg.vision_config
    patch = int(cfg["patch_size"])
    merge = int(cfg["spatial_merge_size"])
    minimum, maximum = _active_image_token_bounds(engine)
    target_h, target_w = smart_resize(
        int(cfg["temporal_patch_size"]), height, width,
        temporal_factor=int(cfg["temporal_patch_size"]),
        factor=patch * merge,
        min_image_tokens=minimum,
        max_image_tokens=maximum,
    )
    grid = target_h // patch, target_w // patch
    validate_global_attention_grids(
        [grid], max_patches=(maximum * merge * merge), merge=merge,
        max_merged_tokens=MAX_RETAINED_VISION_TOKENS)
    return grid


def preprocess_image(engine, image, expected_grid: tuple[int, int]) -> np.ndarray:
    """PIL image -> released block-major flattened 3D-convolution patches."""
    from PIL import Image

    cfg = engine.cfg.vision_config
    patch = int(cfg["patch_size"])
    merge = int(cfg["spatial_merge_size"])
    temporal = int(cfg["temporal_patch_size"])
    gh, gw = expected_grid
    target_h, target_w = gh * patch, gw * patch
    rgb = image.convert("RGB")
    width, height = rgb.size
    minimum, maximum = _active_image_token_bounds(engine)
    computed_h, computed_w = smart_resize(
        temporal, height, width, temporal_factor=temporal,
        factor=patch * merge, min_image_tokens=minimum,
        max_image_tokens=maximum)
    if (computed_h, computed_w) != (target_h, target_w):
        raise ValueError("image dimensions changed after vision preflight")
    scale = min(target_h / height, target_w / width)
    pixels_per_token = temporal * (patch * merge) ** 2
    if temporal * height * width >= pixels_per_token * minimum:
        scale = min(1.0, scale)
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    if (content_w, content_h) != rgb.size:
        rgb = rgb.resize((content_w, content_h), Image.Resampling.BICUBIC)
    value = np.zeros((3, target_h, target_w), dtype=np.float32)
    source = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    value[:, :content_h, :content_w] = source
    mean = np.asarray(cfg["image_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(cfg["image_std"], dtype=np.float32)[:, None, None]
    value = (value - mean) / std
    value = value.reshape(
        3, gh // merge, merge, patch, gw // merge, merge, patch)
    value = value.transpose(1, 4, 2, 5, 0, 3, 6)
    value = np.broadcast_to(
        value[..., None, :, :],
        (*value.shape[:-2], temporal, patch, patch))
    return value.reshape(gh * gw, 3 * temporal * patch * patch)


def _rms_norm(value, weight, eps: float):
    dtype = value.dtype
    work = value.astype(mx.float32)
    work = work * mx.rsqrt(mx.mean(work * work, axis=-1, keepdims=True) + eps)
    return weight * work.astype(dtype)


def _rotate_half(value):
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _position_cos_sin(gh: int, gw: int, merge: int, head_dim: int):
    hs, ws = np.meshgrid(
        np.arange(gh, dtype=np.float32),
        np.arange(gw, dtype=np.float32), indexing="ij")
    shape = (gh // merge, merge, gw // merge, merge)
    hs = hs.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    ws = ws.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    rotary_dim = head_dim // 2
    inv = 1.0 / (
        10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32)
                    / rotary_dim))
    rotary = np.stack((hs, ws), axis=1)[..., None] * inv[None, None, :]
    rotary = rotary.reshape(gh * gw, -1)
    embedding = np.concatenate((rotary, rotary), axis=-1)
    return mx.array(np.cos(embedding)), mx.array(np.sin(embedding))


def _gelu_exact(value):
    return value * 0.5 * (1.0 + mx.erf(value / math.sqrt(2.0)))


def vision_forward(engine, pixels: np.ndarray, grid: tuple[int, int], weights):
    """Evaluate one released image tower and return merged text embeddings."""
    cfg = engine.cfg.vision_config
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_heads"])
    head_dim = hidden // heads
    merge = int(cfg["spatial_merge_size"])
    eps = float(cfg["rms_norm_eps"])
    limit = float(cfg["swiglu_limit"])
    gh, gw = grid
    count = gh * gw

    patch_weight = weights[f"{V}.patch_embed.proj.weight"].reshape(hidden, -1)
    x = (mx.array(pixels).astype(patch_weight.dtype) @ patch_weight.T
         + weights[f"{V}.patch_embed.proj.bias"])
    cos, sin = _position_cos_sin(gh, gw, merge, head_dim)
    for layer in range(int(cfg["depth"])):
        prefix = f"{V}.blocks.{layer}"
        normalized = _rms_norm(
            x, weights[f"{prefix}.norm1.weight"], eps)
        qkv = layer_runner._linear(
            normalized, weights, f"{prefix}.attn.qkv")
        q, k, value = mx.split(
            qkv.reshape(count, 3, heads, head_dim), 3, axis=1)
        q, k, value = q[:, 0], k[:, 0], value[:, 0]
        q = _rms_norm(q, weights[f"{prefix}.attn.q_norm.weight"], eps)
        k = _rms_norm(k, weights[f"{prefix}.attn.k_norm.weight"], eps)
        q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
        q = (q32 * cos[:, None] + _rotate_half(q32) * sin[:, None]).astype(
            q.dtype)
        k = (k32 * cos[:, None] + _rotate_half(k32) * sin[:, None]).astype(
            k.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            value.transpose(1, 0, 2)[None],
            scale=head_dim ** -0.5,
        )[0].transpose(1, 0, 2).reshape(count, hidden)
        x = x + layer_runner._linear(
            attended, weights, f"{prefix}.attn.proj")
        normalized = _rms_norm(
            x, weights[f"{prefix}.norm2.weight"], eps)
        gate = mx.minimum(
            layer_runner._linear(
                normalized, weights, f"{prefix}.mlp.gate_proj"), limit)
        up = mx.clip(
            layer_runner._linear(
                normalized, weights, f"{prefix}.mlp.up_proj"), -limit, limit)
        x = x + layer_runner._linear(
            mx.sigmoid(gate) * gate * up,
            weights, f"{prefix}.mlp.down_proj")
        mx.eval(x)

    x = _rms_norm(x, weights[f"{V}.post_layernorm.weight"], eps)
    grouped = x.reshape(-1, merge, merge, hidden).transpose(0, 3, 1, 2)
    down_weight = weights[f"{V}.downsample.weight"].reshape(
        int(cfg["out_hidden_size"]), -1)
    x = grouped.reshape(grouped.shape[0], -1) @ down_weight.T
    x = x + weights[f"{V}.downsample.bias"]
    x = layer_runner._linear(x, weights, f"{V}.merger.proj")
    x = _gelu_exact(mx.fast.layer_norm(
        x,
        weights[f"{V}.merger.post_projection_norm.weight"],
        weights[f"{V}.merger.post_projection_norm.bias"],
        1e-5,
    ))
    gate = mx.minimum(
        layer_runner._linear(x, weights, f"{V}.merger.gate_proj"), limit)
    up = mx.clip(
        layer_runner._linear(x, weights, f"{V}.merger.up_proj"), -limit, limit)
    output = layer_runner._linear(
        mx.sigmoid(gate) * gate * up, weights, f"{V}.merger.down_proj")
    mx.eval(output)
    return output


def _image_key(image, grid: tuple[int, int]):
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "little"))
    digest.update(rgb.height.to_bytes(8, "little"))
    digest.update(rgb.tobytes())
    return digest.digest(), tuple(grid)


def prepare_vl_prompt(engine, prompt_text: str, images) -> dict:
    if not images:
        raise ValueError("vision generation requires at least one image")
    if any(getattr(image, "is_video", False) for image in images):
        raise ValueError(
            "GLM-5.3 video input is not yet implemented; image input is supported")
    grids = [image_grid(engine, image) for image in images]
    merge = int(engine.cfg.vision_config["spatial_merge_size"])
    validate_global_attention_grids(grids, merge=merge)
    counts = [(gh // merge) * (gw // merge) for gh, gw in grids]
    ids = list(getattr(prompt_text, "token_ids", ())
               or engine.tokenizer.encode(prompt_text).ids)
    from .qwen3vl import (_expand_multimodal_tokens_with_boundaries,
                          _expanded_tool_capsules)

    tokens, boundaries = _expand_multimodal_tokens_with_boundaries(
        ids, engine.cfg.image_token_id, counts, 0, [])
    return {
        "grids": grids,
        "tokens": tokens,
        "tool_capsules": _expanded_tool_capsules(prompt_text, boundaries),
        "active_max_image_tokens": _active_image_token_bounds(engine)[1],
    }


def _load_vision_weights(engine):
    names = engine.store.names_with_prefix(f"{V}.")
    if not names:
        raise ValueError("GLM-5.3 checkpoint is missing model.visual tensors")
    weights, seconds, logical_bytes = engine.store.fetch(names)
    return weights, seconds, logical_bytes


def _vision_embeddings(engine, images, prepared, on_progress=None):
    cache = getattr(engine, "_glm53_vision_embedding_cache", None)
    if cache is None:
        cache = OrderedDict()
        engine._glm53_vision_embedding_cache = cache
    outputs = []
    misses = []
    keys = []
    hits = 0
    for index, (image, grid) in enumerate(
            zip(images, prepared["grids"], strict=True)):
        key = _image_key(image, tuple(grid))
        keys.append(key)
        cached = cache.pop(key, None)
        if cached is None:
            misses.append((index, image, tuple(grid), key))
            outputs.append(None)
        else:
            cache[key] = cached
            outputs.append(cached)
            hits += 1
    load_s = 0.0
    read_bytes = 0
    if misses:
        weights, load_s, read_bytes = _load_vision_weights(engine)
        for completed, (index, image, grid, key) in enumerate(misses, start=1):
            pixels = preprocess_image(engine, image, grid)
            value = vision_forward(engine, pixels, grid, weights)
            outputs[index] = value
            cache[key] = value
            while len(cache) > _VISION_CACHE_ENTRIES:
                cache.popitem(last=False)
            if on_progress is not None:
                on_progress({"phase": "vision", "completed_images": completed,
                             "total_images": len(misses)})
        del weights
        mx.clear_cache()
    elif on_progress is not None:
        on_progress({"phase": "vision", "completed_images": len(images),
                     "total_images": len(images), "cache_source": "embedding"})
    return mx.concatenate(outputs, axis=0), keys, hits, load_s, read_bytes


def _prefill(engine, tokens, image_embeds, kv):
    is_image = np.asarray(tokens) == int(engine.cfg.image_token_id)
    if int(is_image.sum()) != int(image_embeds.shape[0]):
        raise ValueError(
            "GLM-5.3 image embeddings and placeholder tokens do not match")
    x = engine._embed(tokens)
    if is_image.any():
        copied = mx.zeros_like(x) + x
        copied[0, mx.array(np.nonzero(is_image)[0]), :] = image_embeds.astype(
            x.dtype)
        x = copied
    x = engine._sweep(x, kv, offset=0)
    logits = engine._final_logits(x)
    mx.eval(logits)
    return logits


def generate_vl(
    engine, prompt_text: str, images, max_tokens: int = 64,
    on_token=None, stop=None, on_progress=None, *, prepared=None,
    sampling: SamplingParams | None = None, constraint=None,
) -> dict:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    started = time.perf_counter()
    sampling = sampling or SamplingParams()
    sampling.seed_rng()
    stop = stop or []
    prepared = prepared or prepare_vl_prompt(engine, prompt_text, images)
    tokens = prepared["tokens"]
    limits = [int(value) for value in (
        getattr(engine, "effective_max_position_embeddings", 0),
        getattr(engine.rc, "context_bound", 0),
    ) if int(value or 0) > 0]
    if limits and len(tokens) + max_tokens > min(limits):
        raise ValueError(
            f"expanded vision prompt({len(tokens)})+max_tokens({max_tokens}) "
            f"exceeds active context limit={min(limits)}")

    engine._true_peak_metal_bytes = mx.get_active_memory()
    engine._chunk_peak_metal_bytes = engine._true_peak_metal_bytes
    mx.reset_peak_memory()
    if engine.governor is not None:
        engine.governor.reset_request_peak(engine._true_peak_metal_bytes)
    from .request_state import release_generation_state

    release_generation_state(engine)
    mx.clear_cache()
    image_keys = [
        _image_key(image, tuple(grid))
        for image, grid in zip(images, prepared["grids"], strict=True)
    ]
    prompt_key = (tuple(tokens), tuple(image_keys))
    cached = getattr(engine, "_glm53_vision_prompt_cache", None)
    engine._glm53_vision_prompt_cache = None
    exact_hit = bool(cached is not None and cached[0] == prompt_key)
    if exact_hit:
        _key, kv, logits = cached
        cache_hits = len(images)
        load_s = 0.0
        vision_read_bytes = 0
        vision_s = 0.0
        prefill_s = 0.0
        tower_skipped = len(images)
    else:
        if cached is not None:
            del cached
            mx.clear_cache()
        vision_started = time.perf_counter()
        image_embeds, _keys, cache_hits, load_s, vision_read_bytes = (
            _vision_embeddings(engine, images, prepared, on_progress))
        vision_s = time.perf_counter() - vision_started
        engine._note_true_peak()
        kv = engine.new_kv()
        prefill_started = time.perf_counter()
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": 0,
                         "total_tokens": len(tokens)})
        logits = _prefill(engine, tokens, image_embeds, kv)
        prefill_s = time.perf_counter() - prefill_started
        tower_skipped = 0
    if on_progress is not None:
        on_progress({"phase": "prefill", "completed_tokens": len(tokens),
                     "total_tokens": len(tokens),
                     "cache_source": "vision_prompt_kv" if exact_hit else "cold"})
    prompt_endpoint = fork_hybrid_kv_endpoint(kv)
    prompt_logits = logits

    decode_started = time.perf_counter()
    sampled = constraint.mask_logits(logits) if constraint is not None else logits
    next_token = sample(sampled, sampling, history=tokens)
    if constraint is not None:
        constraint.accept_token(next_token)
    generated = [next_token]
    decoder = IncrementalDetokenizer(engine.tokenizer, stop)
    stop_text = None
    stop_sequence = None

    def accept(token):
        nonlocal stop_text, stop_sequence
        delta = decoder.push_token(token)
        if decoder.matched_stop_sequence is not None:
            stop_sequence = decoder.matched_stop_sequence
            stop_text = decoder.stop_text
        elif on_token is not None and delta:
            on_token(delta)

    accept(next_token)
    position = len(tokens)
    for _ in range(max_tokens - 1):
        if (stop_text is not None or next_token in engine.cfg.eos_token_ids
                or bool(constraint is not None and constraint.completed)):
            break
        x = engine._embed([next_token])
        x = engine._sweep(x, kv, offset=position)
        logits = engine._final_logits(x)
        sampled = constraint.mask_logits(logits) if constraint is not None else logits
        next_token = sample(
            sampled, sampling,
            history=(tokens + generated
                     if sampling.repetition_penalty != 1.0 else None))
        if constraint is not None:
            constraint.accept_token(next_token)
        generated.append(next_token)
        position += 1
        accept(next_token)
    final_delta, text = decoder.finish_token_stream(final_text=stop_text)
    if on_token is not None and final_delta:
        on_token(final_delta)
    decode_s = time.perf_counter() - decode_started
    engine._glm53_vision_prompt_cache = (
        prompt_key, prompt_endpoint, prompt_logits)
    engine._note_true_peak()
    if engine.governor is not None:
        engine._true_peak_metal_bytes = max(
            engine._true_peak_metal_bytes,
            int(engine.governor.request_peak()),
            int(mx.get_active_memory()),
        )
    termination = (
        "stop_sequence" if stop_text is not None else
        "grammar" if bool(constraint is not None and constraint.completed) else
        "eos" if generated[-1] in engine.cfg.eos_token_ids else "length")
    result = {
        "text": text,
        "tokens": generated,
        "prompt_tokens": len(tokens),
        "vision_s": vision_s,
        "vision_weight_load_s": load_s,
        "vision_weight_read_bytes": vision_read_bytes,
        "vision_cache_hits": cache_hits,
        "vision_cache_misses": len(images) - cache_hits,
        "vision_prompt_cache_hit": exact_hit,
        "vision_prompt_cache_exact_hit": exact_hit,
        "vision_prompt_cache_tower_skipped": tower_skipped,
        "vision_prompt_cache_stored": 1,
        "vision_grids": [list(grid) for grid in prepared["grids"]],
        "vision_active_max_image_tokens": prepared["active_max_image_tokens"],
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": time.perf_counter() - started,
        "true_peak_metal_bytes": engine._true_peak_metal_bytes,
        "resident_pipelined_decode_steps": 0,
        "sampling_profile": sampling.profile,
        "constraint_profile": getattr(constraint, "profile", "none"),
        "stopped": stop_text is not None,
        "stop_sequence": stop_sequence,
        "termination_reason": termination,
    }
    result["path_stats"] = {
        "prompt_cache_exact_hit": int(exact_hit),
        "prompt_cache_prefix_tokens": len(tokens) if exact_hit else 0,
        "prompt_cache_source": "vision_memory" if exact_hit else "cold",
        "vision_cache_hits": cache_hits,
        "vision_cache_misses": len(images) - cache_hits,
        "vision_prompt_cache_hit": int(exact_hit),
        "vision_prompt_cache_tower_skipped": tower_skipped,
        "vision_prompt_cache_stored": 1,
        "vision_weight_read_bytes": vision_read_bytes,
        "vision_weight_load_s": load_s,
        "vision_active_max_image_tokens": prepared["active_max_image_tokens"],
        "prompt_state_approximate": 0,
        "sampling_profile": sampling.profile,
        "constraint_profile": getattr(constraint, "profile", "none"),
    }
    return result
