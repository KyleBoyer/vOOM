"""Dependency-light Qwen-VL resize-grid and M-RoPE position helpers."""

from __future__ import annotations

import math

import numpy as np


# qwen3vl.vision_forward applies global attention within each image or temporal
# video segment. MLX's fused kernel can avoid materializing the complete score
# matrix, but O((H*W)^2) per segment is still unsafe at the checkpoint
# processor's 65,536-patch ceiling on a 16-GB Mac.
MAX_GLOBAL_VISION_PATCHES = 4096
# vision_forward retains one merged embedding plus three DeepStack tensors per
# image until text prefill. Bound that aggregate separately from per-image
# attention work; 4,096 merged tokens are about 134 MB at hidden=4,096/bf16.
MAX_RETAINED_VISION_TOKENS = 4096


def validate_global_attention_grids(
    grids: list[tuple[int, int] | tuple[int, int, int]],
    *,
    max_patches: int = MAX_GLOBAL_VISION_PATCHES,
    merge: int = 2,
    max_merged_tokens: int = MAX_RETAINED_VISION_TOKENS,
) -> None:
    """Reject per-frame attention or aggregate retained-state overflow."""
    if max_patches <= 0 or merge <= 0 or max_merged_tokens <= 0:
        raise ValueError("vision patch/token limits and merge must be positive")
    merged_tokens = 0
    for image_index, grid in enumerate(grids, start=1):
        if len(grid) == 2:
            grid_t, grid_h, grid_w = 1, *grid
        elif len(grid) == 3:
            grid_t, grid_h, grid_w = grid
        else:
            raise ValueError("vision grid must be (h,w) or (t,h,w)")
        patches = int(grid_h) * int(grid_w)
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError("image grid dimensions must be positive")
        if grid_h % merge or grid_w % merge:
            raise ValueError("image grid must be divisible by the spatial merge size")
        if patches > max_patches:
            raise ValueError(
                f"image {image_index} expands to {patches:,} vision patches; "
                f"the per-frame global-attention safety limit is {max_patches:,}. "
                "Resize the image before retrying"
            )
        merged_tokens += grid_t * (grid_h // merge) * (grid_w // merge)
    if merged_tokens > max_merged_tokens:
        raise ValueError(
            f"images retain {merged_tokens:,} merged vision tokens in aggregate; "
            f"the current safety limit is {max_merged_tokens:,}. "
            "Use fewer or smaller images before retrying"
        )


def image_grid_size(width: int, height: int, *, patch: int = 16, merge: int = 2,
                    min_pixels: int = 65_536,
                    max_pixels: int = 16_777_216,
                    max_aspect_ratio: float = 200.0) -> tuple[int, int]:
    """Return Qwen-style smart-resize patch grid without allocating pixels.

    Round normally inside the pixel budget, floor both aligned dimensions when
    shrinking, and ceil them when expanding. This mirrors the checkpoint
    processor's branch structure and avoids collapsing over-max images all the
    way to ``min_pixels``.
    """
    if width <= 0 or height <= 0 or patch <= 0 or merge <= 0:
        raise ValueError("image dimensions, patch, and merge must be positive")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("invalid image pixel bounds")
    if max(width, height) / min(width, height) > max_aspect_ratio:
        raise ValueError("image aspect ratio exceeds supported limit")
    factor = patch * merge
    area = width * height
    resized_width = max(factor, round(width / factor) * factor)
    resized_height = max(factor, round(height / factor) * factor)
    aligned_area = resized_width * resized_height
    if aligned_area > max_pixels:
        beta = math.sqrt(area / max_pixels)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
    elif aligned_area < min_pixels:
        beta = math.sqrt(min_pixels / area)
        resized_width = max(factor, math.ceil(width * beta / factor) * factor)
        resized_height = max(factor, math.ceil(height * beta / factor) * factor)
    return resized_height // patch, resized_width // patch


def build_positions(tokens: list[int], image_token: int,
                    grids: list[tuple[int, int]], merge: int = 2
                    ) -> tuple[np.ndarray, int]:
    """Build ordered multi-image M-RoPE positions and validate every span.

    Each grid consumes exactly one contiguous expanded image-token span. Text
    positions advance by one; an image advances by its largest merged dimension,
    matching Qwen's ``get_rope_index`` semantics.
    """
    if merge <= 0:
        raise ValueError("spatial merge size must be positive")
    pos = np.zeros((3, len(tokens)), dtype=np.int64)
    cursor = 0
    token_index = 0
    image_index = 0
    while token_index < len(tokens):
        if tokens[token_index] != image_token:
            pos[:, token_index] = cursor
            cursor += 1
            token_index += 1
            continue

        if image_index >= len(grids):
            raise ValueError("expanded prompt has more image spans than grids")
        grid_h, grid_w = grids[image_index]
        if grid_h <= 0 or grid_w <= 0 or grid_h % merge or grid_w % merge:
            raise ValueError("image grid must be positive and divisible by merge")
        merged_h, merged_w = grid_h // merge, grid_w // merge
        count = merged_h * merged_w
        end = token_index + count
        if end > len(tokens) or any(t != image_token for t in tokens[token_index:end]):
            raise ValueError("expanded image-token span does not match its grid")
        rows = np.repeat(np.arange(merged_h), merged_w)
        cols = np.tile(np.arange(merged_w), merged_h)
        pos[0, token_index:end] = cursor
        pos[1, token_index:end] = cursor + rows
        pos[2, token_index:end] = cursor + cols
        cursor += max(merged_h, merged_w)
        token_index = end
        image_index += 1

    if image_index != len(grids):
        raise ValueError("image grids exceed expanded prompt image spans")
    return pos, int(pos.max()) if tokens else -1


def build_multimodal_positions(
    tokens: list[int], modality_grids: dict[int, list[tuple[int, int]]],
    merge: int = 2,
) -> tuple[np.ndarray, int]:
    """Build M-RoPE positions for interleaved image and timestamped video spans.

    Each video temporal patch is represented by one ordinary spatial span,
    separated by timestamp/text tokens exactly like Qwen3-VL's processor.
    """
    if merge <= 0:
        raise ValueError("spatial merge size must be positive")
    remaining = {token: iter(grids) for token, grids in modality_grids.items()}
    consumed = {token: 0 for token in modality_grids}
    pos = np.zeros((3, len(tokens)), dtype=np.int64)
    cursor = 0
    token_index = 0
    while token_index < len(tokens):
        token = tokens[token_index]
        if token not in remaining:
            pos[:, token_index] = cursor
            cursor += 1
            token_index += 1
            continue
        try:
            grid_h, grid_w = next(remaining[token])
        except StopIteration as error:
            raise ValueError(
                f"expanded prompt has more token {token} spans than grids") from error
        if grid_h <= 0 or grid_w <= 0 or grid_h % merge or grid_w % merge:
            raise ValueError("vision grid must be positive and divisible by merge")
        merged_h, merged_w = grid_h // merge, grid_w // merge
        count = merged_h * merged_w
        end = token_index + count
        if end > len(tokens) or any(t != token for t in tokens[token_index:end]):
            raise ValueError("expanded vision-token span does not match its grid")
        rows = np.repeat(np.arange(merged_h), merged_w)
        cols = np.tile(np.arange(merged_w), merged_h)
        pos[0, token_index:end] = cursor
        pos[1, token_index:end] = cursor + rows
        pos[2, token_index:end] = cursor + cols
        cursor += max(merged_h, merged_w)
        token_index = end
        consumed[token] += 1
    for token, iterator in remaining.items():
        try:
            next(iterator)
        except StopIteration:
            continue
        raise ValueError(f"vision grids exceed expanded token {token} spans")
    return pos, int(pos.max()) if tokens else -1
