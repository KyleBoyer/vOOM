"""Real HTTP gate for GLM-5.3 released image input and telemetry."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8077/v1/responses")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 255, 0)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_url = "data:image/png;base64," + base64.b64encode(
        image_bytes).decode("ascii")
    body = {
        "model": args.model,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url},
                {"type": "input_text", "text": (
                    "What is the dominant color? Answer with exactly one "
                    "lowercase color word.")},
            ],
        }],
        "temperature": 0.0,
        "max_output_tokens": args.max_output_tokens,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        value = json.load(response)
    wall_s = time.perf_counter() - started
    text = str(value.get("output_text") or "").strip().casefold()
    if not text:
        fragments = []
        for item in value.get("output") or []:
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    fragments.append(part["text"])
        text = "".join(fragments).strip().casefold()
    usage = value.get("usage") or {}
    timing = value.get("vmodel_timing") or {}
    passed = text == "green"
    document = {
        "schema": "voom.glm53-vision-http-gate.v1",
        "passed": passed,
        "request": {
            "model": args.model,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_bytes": len(image_bytes),
            "image_shape": [64, 64, 3],
            "max_output_tokens": args.max_output_tokens,
        },
        "result": {
            "answer": text,
            "wall_s": wall_s,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "vision_seconds": timing.get("vision_seconds"),
            "vision_cache_hits": timing.get("vision_cache_hits"),
            "vision_cache_misses": timing.get("vision_cache_misses"),
            "vision_prompt_cache_exact_hit": timing.get(
                "vision_prompt_cache_exact_hit"),
            "peak_metal_bytes": timing.get("true_peak_metal_bytes"),
            "weight_store_bytes_read": timing.get("weight_store_bytes_read"),
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
