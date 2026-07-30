#!/usr/bin/env python3
"""Gate generic strict-token-extension reuse in the resident Qwen backend.

This deliberately bypasses chat, tools, and provider request shapes.  It gives
the engine one deterministic token prefix and then the same prefix plus a small
forward-only suffix.  The proof therefore covers the cache primitive itself,
not a Plex- or message-boundary policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig
from runtime.engine import RuntimeConfig
from runtime.resident_mlx_lm import (
    ResidentMLXLMEngine,
    choose_resident_backend,
)
from runtime.sampler import SamplingParams


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        remaining = memoryview(payload)
        while remaining:
            remaining = remaining[os.write(descriptor, remaining):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _repeat_to_length(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise ValueError("seed text encoded to no tokens")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=5_755)
    parser.add_argument("--extension-tokens", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-extension-seconds", type=float, default=30.0)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-available-gb", type=float, default=3.2)
    args = parser.parse_args()

    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")
    if min(
        args.prefix_tokens, args.extension_tokens, args.max_output_tokens,
    ) <= 0:
        parser.error("token counts must be positive")
    if min(
        args.max_extension_seconds, args.max_peak_metal_gb,
        args.min_available_gb,
    ) <= 0:
        parser.error("time and memory thresholds must be positive")

    available = int(psutil.virtual_memory().available)
    required = int(args.min_available_gb * 1_000_000_000)
    if available < required:
        raise SystemExit(
            f"insufficient available memory: {available} < {required}")

    os.environ["VMODEL_RESIDENT_BACKEND"] = "mlx-lm"
    os.environ["VMODEL_MLX_LM_PROMPT_CACHE"] = "1"
    cfg = ModelConfig.from_dir(args.model_dir)
    decision = choose_resident_backend(
        args.model_dir, cfg, "fast", available_bytes=available)
    if not decision.admitted:
        raise SystemExit(
            f"resident backend was not admitted: {decision.reason}")

    engine = ResidentMLXLMEngine(
        args.model_dir, cfg, RuntimeConfig(), decision)
    report = {
        "schema": "voom.qwen-resident-prefix-extension.v1",
        "model_dir": str(args.model_dir),
        "request_shape": {
            "subject": "synthetic-neutral-runtime-prose",
            "tools": 0,
            "messages": 0,
            "prefix_tokens": args.prefix_tokens,
            "extension_tokens": args.extension_tokens,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "thresholds": {
            "extension_wall_seconds_strictly_below": (
                args.max_extension_seconds),
            "true_peak_metal_gb_strictly_below": args.max_peak_metal_gb,
            "cache_source": "hot-prompt-extension",
            "cached_prefix_tokens": args.prefix_tokens,
        },
        "failures": [],
        "passed": False,
    }
    try:
        seed_ids = engine.tokenizer.encode(
            "A generic runtime prefix discusses mathematics, systems, "
            "history, art, and unrelated everyday questions. ").ids
        suffix_ids = engine.tokenizer.encode(
            "\nA small independent addition follows.").ids
        prefix = _repeat_to_length(seed_ids, args.prefix_tokens)
        extension = _repeat_to_length(suffix_ids, args.extension_tokens)
        sampling = SamplingParams(
            temperature=args.temperature, seed=args.seed)
        cold = engine.generate(
            SimpleNamespace(token_ids=tuple(prefix)),
            max_tokens=args.max_output_tokens,
            sampling=sampling,
        )
        warm = engine.generate(
            SimpleNamespace(token_ids=tuple(prefix + extension)),
            max_tokens=args.max_output_tokens,
            sampling=sampling,
        )
        report["cold"] = cold
        report["extension"] = warm

        path = warm.get("path_stats", {})
        if path.get("prompt_cache_source") != "hot-prompt-extension":
            report["failures"].append(
                "strict extension did not use hot-prompt-extension")
        if path.get("execution_path") != "mlx_lm_prompt_extension":
            report["failures"].append(
                "strict extension did not use mlx_lm_prompt_extension")
        if path.get("prompt_cache_prefix_tokens") != args.prefix_tokens:
            report["failures"].append(
                "cached prefix length did not equal the complete base prefix")
        if warm.get("prompt_tokens") != (
                args.prefix_tokens + args.extension_tokens):
            report["failures"].append(
                "extension prompt token count changed unexpectedly")
        if not warm.get("total_s", float("inf")) < args.max_extension_seconds:
            report["failures"].append(
                "strict extension exceeded its wall-time threshold")
        peak = max(
            int(cold.get("true_peak_metal_bytes", 0)),
            int(warm.get("true_peak_metal_bytes", 0)),
        )
        if not peak < int(args.max_peak_metal_gb * 1_000_000_000):
            report["failures"].append(
                "resident prefix gate exceeded its Metal ceiling")
        report["passed"] = not report["failures"]
    finally:
        engine.close()

    _atomic_json(args.result_json, report)
    print(
        f"[prefix-extension] {'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}",
        flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
