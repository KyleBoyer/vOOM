#!/usr/bin/env python3
"""Real-model exactness gate for native lower-right Qwen3-VL causal masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.qwen3vl import generate_vl  # noqa: E402
from runtime.toolcalls import VISION_SPAN  # noqa: E402


PROMPT = (
    "<|im_start|>user\n"
    + VISION_SPAN
    + "Read the code in this image.<|im_end|>\n"
      "<|im_start|>assistant\n"
)


def _image() -> Image.Image:
    image = Image.new("RGB", (256, 256), (35, 90, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 70, 206, 186), fill=(240, 240, 240))
    draw.text((85, 120), "CODE 731", fill=(0, 0, 0))
    return image


def _run(engine, image, *, explicit: bool, max_tokens: int) -> dict:
    engine._vision_prompt_cache = None
    original = mx.fast.scaled_dot_product_attention
    causal_shapes = []

    def attention(q, k, v, *, scale, mask=None, **kwargs):
        if isinstance(mask, str) and mask == "causal":
            query_length = int(q.shape[-2])
            key_length = int(k.shape[-2])
            causal_shapes.append((query_length, key_length))
            if explicit:
                query_positions = mx.arange(
                    key_length - query_length, key_length)[:, None]
                key_positions = mx.arange(key_length)[None, :]
                mask = mx.where(
                    key_positions <= query_positions,
                    0.0, float("-inf")).astype(q.dtype)
        return original(q, k, v, scale=scale, mask=mask, **kwargs)

    mx.fast.scaled_dot_product_attention = attention
    try:
        cold = generate_vl(
            engine, PROMPT, [image], max_tokens=max_tokens)
        extended = generate_vl(
            engine, PROMPT + "The code is", [image],
            max_tokens=max_tokens)
    finally:
        mx.fast.scaled_dot_product_attention = original

    return {
        "cold_tokens": cold["tokens"],
        "suffix_tokens": extended["tokens"],
        "cold_text": cold["text"],
        "suffix_text": extended["text"],
        "cold_prefill_s": cold["prefill_s"],
        "suffix_prefill_s": extended["prefill_s"],
        "peak_metal_bytes": max(
            cold["true_peak_metal_bytes"],
            extended["true_peak_metal_bytes"]),
        "suffix_cache_hit": extended["vision_prompt_cache_hit"],
        "suffix_prefix_tokens": extended["vision_prompt_cache_prefix_tokens"],
        "causal_shapes": causal_shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=(Path.home()
                 / "hf_cache/modelscope/models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"model is not a local checkpoint: {model}")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    engine = StreamingEngine(model, RuntimeConfig(
        max_weight_cache_mb=6000,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        resident_fast_decode=False,
        vision_max_patches=4096,
        governor=True,
    ))
    image = _image()
    try:
        native = _run(
            engine, image, explicit=False, max_tokens=args.max_tokens)
        explicit = _run(
            engine, image, explicit=True, max_tokens=args.max_tokens)
        engine._vision_prompt_cache = None
        streamed = []
        stopped = generate_vl(
            engine, PROMPT, [image], max_tokens=args.max_tokens,
            stop=["731"], on_token=streamed.append)
    finally:
        engine.close()
        mx.clear_cache()

    full_native = any(q == k and q > 1 for q, k in native["causal_shapes"])
    suffix_native = any(q < k for q, k in native["causal_shapes"])
    full_explicit = any(q == k and q > 1 for q, k in explicit["causal_shapes"])
    suffix_explicit = any(q < k for q, k in explicit["causal_shapes"])
    passed = bool(
        native["cold_tokens"] == explicit["cold_tokens"]
        and native["suffix_tokens"] == explicit["suffix_tokens"]
        and native["suffix_cache_hit"]
        and explicit["suffix_cache_hit"]
        and full_native and suffix_native
        and full_explicit and suffix_explicit
        and stopped["text"] == "CODE"
        and "".join(streamed) == stopped["text"]
        and stopped["stop_sequence"] == "731"
        and stopped["termination_reason"] == "stop_sequence"
    )
    report = {
        "gate": "qwen3vl-native-causal-v1",
        "passed": passed,
        "model": str(model),
        "cold_ids_equal": native["cold_tokens"] == explicit["cold_tokens"],
        "suffix_ids_equal": native["suffix_tokens"] == explicit["suffix_tokens"],
        "stream_stop": {
            "text": stopped["text"],
            "emitted": "".join(streamed),
            "tokens": stopped["tokens"],
            "stop_sequence": stopped["stop_sequence"],
            "termination_reason": stopped["termination_reason"],
        },
        "native": native,
        "explicit": explicit,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
