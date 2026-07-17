"""Qwen3-VL video normalization, sampling, budgeting, and attention tests."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import runtime.qwen3vl as qwen3vl
from runtime.toolcalls import VIDEO_SPAN, VideoFrames, load_video, normalize_messages
from runtime.vision_positions import validate_global_attention_grids


def _write_mp4(path, count: int, fps: int = 6, size: int = 64) -> None:
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            pixels = np.zeros((size, size, 3), dtype=np.uint8)
            pixels[:, :, index % 3] = 32 + index * 8
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_content_block_preserves_official_qwen_wrappers():
    messages, media = normalize_messages([{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "clip.mp4"}},
            {"type": "text", "text": "What changed?"},
        ],
    }])

    assert messages[0]["content"] == VIDEO_SPAN + "What changed?"
    assert VIDEO_SPAN == "<|vision_start|><|video_pad|><|vision_end|>"
    assert media == [{"type": "video", "source": "clip.mp4"}]


def test_synthetic_mp4_is_uniformly_sampled_and_bounded(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    _write_mp4(path, count=12, fps=6)
    monkeypatch.setenv("VMODEL_VIDEO_MAX_FRAMES", "6")
    monkeypatch.setenv("VMODEL_VIDEO_SAMPLE_FPS", "2")

    video = load_video(str(path))

    assert video.is_video
    assert len(video.frames) == 4
    assert video.frame_indices == (0, 4, 7, 11)
    assert video.fps == pytest.approx(6.0)
    assert all(frame.mode == "RGB" and frame.size == (64, 64)
               for frame in video.frames)


def test_one_frame_video_pads_only_to_temporal_patch(tmp_path):
    path = tmp_path / "still.mp4"
    _write_mp4(path, count=1)

    video = load_video(str(path))

    assert len(video.frames) == 2
    assert video.frame_indices == (0, 0)


@pytest.mark.parametrize(
    ("frame_count", "size", "expected"),
    [
        (2, 32, (1, 4, 4)),
        (2, 64, (1, 4, 4)),
        (8, 32, (4, 2, 2)),
        (8, 64, (4, 4, 4)),
        (16, 32, (8, 2, 2)),
        (16, 64, (8, 4, 4)),
    ],
)
def test_small_video_grids_match_official_processor_oracles(
        frame_count, size, expected):
    frames = tuple(Image.new("RGB", (size, size), (index, 0, 0))
                   for index in range(frame_count))
    video = VideoFrames(
        frames, 4.0, tuple(range(frame_count)), frame_count / 4.0)
    engine = SimpleNamespace(
        cfg=SimpleNamespace(
            vision_config={
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            video_min_pixels=4_096,
            video_max_pixels=25_165_824,
        ),
        rc=SimpleNamespace(vision_max_patches=4096),
    )

    assert qwen3vl._video_grid_size(engine, video) == expected


@pytest.mark.parametrize("size", [(16, 32), (32, 32 * 201)])
def test_video_grid_rejects_unsupported_processor_geometry(size):
    video = VideoFrames(
        (Image.new("RGB", size), Image.new("RGB", size)),
        2.0, (0, 1), 1.0)
    engine = SimpleNamespace(
        cfg=SimpleNamespace(vision_config={
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }),
        rc=SimpleNamespace(vision_max_patches=4096),
    )

    with pytest.raises(ValueError, match="factor|aspect ratio"):
        qwen3vl._video_grid_size(engine, video)


def test_video_grid_keeps_per_frame_detail_with_separate_attention_segments():
    frames = tuple(Image.new("RGB", (1024, 1024), (index, 0, 0))
                   for index in range(8))
    video = VideoFrames(frames, 4.0, tuple(range(8)), 2.0)
    engine = SimpleNamespace(
        cfg=SimpleNamespace(vision_config={
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }),
        rc=SimpleNamespace(vision_max_patches=4096),
    )

    grid = qwen3vl._video_grid_size(engine, video)

    assert grid == (4, 64, 64)
    validate_global_attention_grids([grid])


def test_segmented_video_attention_cannot_mix_frames():
    q = mx.zeros((1, 4, 2), dtype=mx.float32)
    k = mx.zeros((1, 4, 2), dtype=mx.float32)
    v = mx.array([[[1.0, 0.0], [3.0, 0.0],
                   [10.0, 0.0], [14.0, 0.0]]])

    output = qwen3vl._segmented_vision_attention(q, k, v, 2, scale=1.0)
    mx.eval(output)

    assert np.allclose(np.array(output)[0, :, 0], [2.0, 2.0, 12.0, 12.0])


def test_video_preprocess_uses_temporal_then_spatial_patch_order():
    arrays = []
    for index in range(4):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = 32 + index * 16
        arrays.append(pixels)
    video = VideoFrames(
        tuple(Image.fromarray(value) for value in arrays),
        4.0, (0, 1, 2, 3), 1.0,
    )

    patches, gt, gh, gw = qwen3vl.preprocess_video(video, 4, 4)

    assert (gt, gh, gw) == (2, 4, 4)
    assert patches.shape == (32, 3 * 2 * 16 * 16)
    # Each temporal group contributes one complete 4x4 spatial patch grid.
    assert np.allclose(patches[:16].mean(),
                       np.stack(arrays[:2]).astype(np.float32).mean() / 127.5 - 1,
                       atol=1e-6)
    assert np.allclose(patches[16:].mean(),
                       np.stack(arrays[2:]).astype(np.float32).mean() / 127.5 - 1,
                       atol=1e-6)
