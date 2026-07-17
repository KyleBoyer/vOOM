#!/usr/bin/env python3
"""Real Qwen3-VL video oracle, quality, latency, cache, and memory gate.

The small synthetic clips keep the gate deterministic and avoid treating a
network video or a language-only answer as proof.  The official Transformers
processor/model are the token, pixel, M-RoPE, and vision-embedding oracle.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import av
import mlx.core as mx
import numpy as np
import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.video_utils import VideoMetadata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig  # noqa: E402
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.model_loader import WeightStore  # noqa: E402
import runtime.qwen3vl as qwen3vl  # noqa: E402
from runtime.toolcalls import VideoFrames, load_video  # noqa: E402
from runtime.vision_positions import build_multimodal_positions  # noqa: E402


RAW_PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "Describe changes.<|im_end|>\n<|im_start|>assistant\n"
)
QUALITY_PROMPT = (
    "<|im_start|>user\n"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "The video changes from one full-screen color to another. Reply with only: "
    "FIRST=<color>; LAST=<color><|im_end|>\n<|im_start|>assistant\n"
)


def _oracle_frames() -> np.ndarray:
    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, :, :, index % 3] = 32 + index * 25
    return frames


def _write_quality_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 256
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            red = index < 4
            image = Image.new(
                "RGB", (256, 256), (230, 30, 30) if red else (30, 60, 230))
            draw = ImageDraw.Draw(image)
            draw.rectangle((28, 96, 228, 160), fill="white")
            draw.text((88, 116), "RED 1" if red else "BLUE 2", fill="black")
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float32, copy=False).reshape(-1)
    b = right.astype(np.float32, copy=False).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _official_oracle(model_dir: Path) -> tuple[dict, list[str]]:
    failures: list[str] = []
    frames = _oracle_frames()
    metadata = VideoMetadata(
        total_num_frames=8, fps=4, width=64, height=64,
        duration=2.0, frames_indices=list(range(8)),
    )
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True)
    official = processor(
        text=[RAW_PROMPT], videos=[frames], return_tensors="pt",
        videos_kwargs={
            "video_metadata": [metadata], "do_sample_frames": False,
        },
    )
    official_ids = official["input_ids"][0].tolist()
    official_pixels = official["pixel_values_videos"].float().numpy()
    official_grid = official["video_grid_thw"][0].tolist()

    cfg = ModelConfig.from_dir(model_dir)
    video = VideoFrames(
        tuple(Image.fromarray(frame) for frame in frames),
        4.0, tuple(range(8)), 2.0,
    )
    preflight_engine = SimpleNamespace(
        cfg=cfg,
        rc=SimpleNamespace(vision_max_patches=4096),
        tokenizer=Tokenizer.from_file(str(model_dir / "tokenizer.json")),
    )
    prepared = qwen3vl.prepare_vl_prompt(
        preflight_engine, RAW_PROMPT, [video])
    pixels, gt, gh, gw = qwen3vl.preprocess_video(
        video, prepared["grids"][0][1], prepared["grids"][0][2])

    token_equal = prepared["tokens"] == official_ids
    grid_equal = list(prepared["grids"][0]) == official_grid
    pixel_max_abs = float(np.max(np.abs(pixels - official_pixels)))
    if not token_equal:
        failures.append("runtime video token expansion differs from Transformers")
    if not grid_equal:
        failures.append("runtime video grid differs from Transformers")
    if pixel_max_abs > 1e-6:
        failures.append("runtime video pixels differ from Transformers")

    mm_types = np.zeros((1, len(official_ids)), dtype=np.int64)
    mm_types[0, np.asarray(official_ids) == cfg.video_token_id] = 2

    class RopeOracle:
        config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids

    official_positions, _delta = Qwen3VLModel.get_rope_index(
        RopeOracle(), torch.tensor([official_ids]), torch.tensor(mm_types),
        video_grid_thw=torch.tensor([official_grid]),
    )
    runtime_positions, _maximum = build_multimodal_positions(
        prepared["tokens"],
        {cfg.video_token_id: prepared["video_frame_grids"]},
        cfg.vision_config["spatial_merge_size"],
    )
    positions_equal = np.array_equal(
        official_positions[:, 0].numpy(), runtime_positions)
    if not positions_equal:
        failures.append("runtime video M-RoPE positions differ from Transformers")

    official_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    official_model.eval()
    with torch.inference_mode():
        vision = official_model.model.visual(
            official["pixel_values_videos"].to(torch.bfloat16),
            official["video_grid_thw"],
        )
        official_pool = vision.pooler_output.float().numpy()
        official_deep = [value.float().numpy()
                         for value in vision.deepstack_features]
    del vision, official_model, official, processor
    gc.collect()

    store = WeightStore(model_dir)
    weights = qwen3vl._load_vision_weights(store)
    runtime_pool, runtime_deep = qwen3vl.vision_forward(
        store, pixels, gh, gw, cfg.vision_config,
        weights=weights, gt=gt,
    )
    mx.eval(runtime_pool, *runtime_deep)
    runtime_pool_np = np.array(runtime_pool.astype(mx.float32))
    runtime_deep_np = [np.array(value.astype(mx.float32))
                       for value in runtime_deep]
    pool_cosine = _cosine(official_pool, runtime_pool_np)
    deep_cosines = [
        _cosine(reference, candidate)
        for reference, candidate in zip(official_deep, runtime_deep_np)
    ]
    if pool_cosine < 0.99 or min(deep_cosines) < 0.995:
        failures.append("segmented runtime vision output misses oracle envelope")

    segmented = qwen3vl._segmented_vision_attention

    def legacy_global(q, k, v, _gt, *, scale):
        return mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=scale)[0]

    qwen3vl._segmented_vision_attention = legacy_global
    try:
        global_pool, _global_deep = qwen3vl.vision_forward(
            store, pixels, gh, gw, cfg.vision_config,
            weights=weights, gt=gt,
        )
        mx.eval(global_pool)
        global_cosine = _cosine(
            official_pool, np.array(global_pool.astype(mx.float32)))
    finally:
        qwen3vl._segmented_vision_attention = segmented
    if global_cosine >= pool_cosine - 0.1:
        failures.append("legacy cross-frame attention control was not rejected")

    report = {
        "tokens_equal": token_equal,
        "positions_equal": positions_equal,
        "grid_equal": grid_equal,
        "grid": official_grid,
        "pixel_max_abs": pixel_max_abs,
        "pool_cosine": pool_cosine,
        "deepstack_cosines": deep_cosines,
        "legacy_global_pool_cosine": global_cosine,
    }
    del store, weights, runtime_pool, runtime_deep, global_pool, _global_deep
    gc.collect()
    mx.clear_cache()
    return report, failures


def _exact_config() -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=6000,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        resident_fast_decode=True,
        resident_fast_prefill_limit=2048,
        vision_max_patches=4096,
        governor=True,
    )


def _fast_config() -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=7000,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        quant_attention=False,
        quant_lm_head=False,
        quantize_tied_lm_head=False,
        resident_fast_decode=True,
        resident_fast_prefill_limit=512,
        fused_swiglu=True,
        stepped_kv_threshold=512,
        vision_max_patches=1024,
        governor=True,
    )


def _quality_and_speed(model_dir: Path, clip: Path) -> tuple[dict, list[str]]:
    failures: list[str] = []
    video = load_video(str(clip))
    exact_engine = StreamingEngine(model_dir, _exact_config())
    try:
        exact = qwen3vl.generate_vl(
            exact_engine, QUALITY_PROMPT, [video], max_tokens=16)
        repeated = qwen3vl.generate_vl(
            exact_engine, QUALITY_PROMPT, [video], max_tokens=16)
    finally:
        exact_engine.close()
    del exact_engine
    gc.collect()
    mx.clear_cache()

    fast_engine = StreamingEngine(model_dir, _fast_config())
    try:
        fast = qwen3vl.generate_vl(
            fast_engine, QUALITY_PROMPT, [video], max_tokens=16)
    finally:
        fast_engine.close()
    del fast_engine
    gc.collect()
    mx.clear_cache()

    def ordered_answer(value: str) -> bool:
        upper = value.upper()
        red = upper.find("RED")
        blue = upper.find("BLUE")
        return red >= 0 and blue > red

    if not ordered_answer(exact["text"]):
        failures.append("exact video answer did not preserve red-to-blue order")
    if not ordered_answer(fast["text"]):
        failures.append("lossy video answer did not preserve red-to-blue order")
    if repeated["tokens"] != exact["tokens"]:
        failures.append("repeated exact video request changed token IDs")
    if (not repeated["vision_cache_hits"]
            and not repeated["vision_prompt_cache_tower_skipped"]):
        failures.append(
            "repeated video neither reused embeddings nor skipped the tower")
    if not repeated["vision_prompt_cache_exact_hit"]:
        failures.append("repeated video did not reuse exact prompt KV")
    if repeated["path_stats"]["prompt_cache_source"] != "vision_memory":
        failures.append("repeated video cache source telemetry is incorrect")

    exact_rate = len(exact["tokens"]) / exact["decode_s"]
    fast_rate = len(fast["tokens"]) / fast["decode_s"]
    decode_speedup = fast_rate / exact_rate
    peak_ratio = (
        fast["true_peak_metal_bytes"] / exact["true_peak_metal_bytes"])
    if decode_speedup < 1.01:
        failures.append("lossy video decode improvement is below 1%")
    if peak_ratio >= 0.99:
        failures.append("lossy video peak-memory improvement is below 1%")

    report = {
        "sampled_frames": len(video.frames),
        "frame_indices": list(video.frame_indices),
        "exact": {
            "text": exact["text"],
            "tokens": exact["tokens"],
            "vision_grid": exact["vision_grids"][0],
            "vision_s": exact["vision_s"],
            "prefill_s": exact["prefill_s"],
            "decode_s": exact["decode_s"],
            "decode_tok_s": exact_rate,
            "total_s": exact["total_s"],
            "peak_metal_bytes": exact["true_peak_metal_bytes"],
        },
        "exact_repeat": {
            "text": repeated["text"],
            "embedding_cache_hits": repeated["vision_cache_hits"],
            "prompt_cache_tower_skipped": repeated[
                "vision_prompt_cache_tower_skipped"],
            "prompt_cache_exact_hit": repeated["vision_prompt_cache_exact_hit"],
            "prompt_cache_source": repeated["path_stats"][
                "prompt_cache_source"],
            "total_s": repeated["total_s"],
            "speedup": exact["total_s"] / repeated["total_s"],
        },
        "lossy": {
            "text": fast["text"],
            "tokens": fast["tokens"],
            "vision_grid": fast["vision_grids"][0],
            "vision_s": fast["vision_s"],
            "prefill_s": fast["prefill_s"],
            "decode_s": fast["decode_s"],
            "decode_tok_s": fast_rate,
            "total_s": fast["total_s"],
            "peak_metal_bytes": fast["true_peak_metal_bytes"],
        },
        "lossy_decode_speedup": decode_speedup,
        "lossy_peak_ratio": peak_ratio,
    }
    return report, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=(Path.home() / "hf_cache/modelscope/models/"
                 "Qwen3-VL-2B-Instruct"),
    )
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    model_dir = args.model.expanduser().resolve()
    if not (model_dir / "model.safetensors").is_file():
        parser.error(f"not a complete local Qwen3-VL checkpoint: {model_dir}")

    oracle, oracle_failures = _official_oracle(model_dir)
    with tempfile.TemporaryDirectory(prefix="voom-qwen3vl-video-") as temp:
        clip = Path(temp) / "quality.mp4"
        _write_quality_video(clip)
        quality, quality_failures = _quality_and_speed(model_dir, clip)

    failures = oracle_failures + quality_failures
    report = {
        "gate": "qwen3vl-video-v1",
        "passed": not failures,
        "failures": failures,
        "model": str(model_dir),
        "oracle": oracle,
        "quality_latency_memory": quality,
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="", flush=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
