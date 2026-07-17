"""Pure resize-grid and multi-image M-RoPE position tests (no MLX)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.vision_positions import (MAX_GLOBAL_VISION_PATCHES,
                                      MAX_RETAINED_VISION_TOKENS,
                                      build_multimodal_positions, build_positions,
                                      image_grid_size,
                                      validate_global_attention_grids)


def test_grid_size_does_not_allocate_pixels_and_is_merge_aligned():
    assert image_grid_size(256, 256) == (16, 16)
    gh, gw = image_grid_size(1000, 300)
    assert gh > 0 and gw > 0 and gh % 2 == 0 and gw % 2 == 0


def test_grid_size_uses_distinct_canonical_over_and_under_budget_branches():
    # 8192x4096 is exactly 2x the 16,777,216-pixel ceiling: preserve its 2:1
    # shape near the ceiling instead of the old bug's ~65K-pixel collapse.
    gh, gw = image_grid_size(8192, 4096)
    assert (gh, gw) == (180, 362)
    resized_area = (gh * 16) * (gw * 16)
    assert 16_000_000 < resized_area <= 16_777_216
    assert image_grid_size(10, 10) == (16, 16)
    try:
        image_grid_size(10_000, 10)
    except ValueError as error:
        assert "aspect ratio" in str(error)
    else:
        raise AssertionError("extreme aspect ratio was accepted")


def test_global_vision_attention_caps_per_image_and_aggregate_state():
    validate_global_attention_grids([(64, 64)])
    try:
        validate_global_attention_grids([(66, 64)])
    except ValueError as error:
        assert f"{MAX_GLOBAL_VISION_PATCHES:,}" in str(error)
        assert "Resize" in str(error)
    else:
        raise AssertionError("unsafe global-attention image grid was accepted")

    try:
        validate_global_attention_grids([(64, 64)] * 5)
    except ValueError as error:
        assert f"{MAX_RETAINED_VISION_TOKENS:,}" in str(error)
        assert "fewer or smaller" in str(error)
    else:
        raise AssertionError("unsafe aggregate vision state was accepted")

    # Temporal segments attend independently, so several frames may each use
    # the per-frame budget. Their merged embeddings remain aggregate-bounded.
    validate_global_attention_grids([(5, 32, 32)])
    try:
        validate_global_attention_grids([(17, 32, 32)])
    except ValueError as error:
        assert "4,096" in str(error)
    else:
        raise AssertionError("unsafe temporal vision grid was accepted")

    try:
        validate_global_attention_grids([(2, 66, 64)])
    except ValueError as error:
        assert "per-frame" in str(error)
    else:
        raise AssertionError("unsafe per-frame video grid was accepted")


def test_two_mixed_grids_consume_distinct_ordered_spans():
    image = 9
    tokens = [10, image, image, 20] + [image] * 6 + [30]
    pos, maximum = build_positions(tokens, image, [(2, 4), (4, 6)], merge=2)
    assert pos.shape == (3, len(tokens))
    assert np.array_equal(pos[:, 0], [0, 0, 0])
    assert np.array_equal(pos[:, 1:3], [[1, 1], [1, 1], [1, 2]])
    assert np.array_equal(pos[:, 3], [3, 3, 3])
    assert np.array_equal(pos[:, 4:10], [
        [4, 4, 4, 4, 4, 4],
        [4, 4, 4, 5, 5, 5],
        [4, 5, 6, 4, 5, 6],
    ])
    assert np.array_equal(pos[:, 10], [7, 7, 7])
    assert maximum == 7


def test_position_builder_rejects_grid_span_mismatches():
    cases = [
        ([1, 9, 2], [(2, 4)]),
        ([1, 9, 9, 2], []),
        ([1, 2], [(2, 4)]),
    ]
    for tokens, grids in cases:
        try:
            build_positions(tokens, 9, grids)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid image span accepted: {tokens}, {grids}")


def test_timestamped_video_and_image_spans_use_distinct_token_ids():
    image, video = 9, 8
    # One image span, text timestamp, then two temporal video-patch spans.
    tokens = [1, image, image, 2, 3, video, video, 4, video, video, 5]
    pos, maximum = build_multimodal_positions(
        tokens, {image: [(2, 4)], video: [(2, 4), (2, 4)]}, merge=2)
    assert pos.shape == (3, len(tokens))
    assert np.array_equal(pos[:, 1:3], [[1, 1], [1, 1], [1, 2]])
    assert np.array_equal(pos[:, 5:7], [[5, 5], [5, 5], [5, 6]])
    assert np.array_equal(pos[:, 8:10], [[8, 8], [8, 8], [8, 9]])
    assert maximum == 10


def _run_all():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
