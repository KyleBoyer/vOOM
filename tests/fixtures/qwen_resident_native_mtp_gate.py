#!/usr/bin/env python3
"""Real-checkpoint A/B gate for resident Qwen3.5 native MTP decoding.

The prompt is deliberately subject-neutral and contains no chat messages,
tools, or Plex-specific structure.  One engine load performs a cold MTP run,
an exact-cache plain decode, and an exact-cache MTP decode.  Greedy output must
be byte-identical.  Stochastic modes may record acceptance/latency, but remain
measurement-only failures until the greedy target gate is known to pass.
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
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")
    if min(args.prompt_tokens, args.max_output_tokens) <= 0:
        parser.error("token counts must be positive")

    available = int(psutil.virtual_memory().available)
    os.environ["VMODEL_RESIDENT_BACKEND"] = "mlx-lm"
    os.environ["VMODEL_MLX_LM_PROMPT_CACHE"] = "1"
    os.environ["VMODEL_MLX_LM_NATIVE_MTP"] = "1"
    cfg = ModelConfig.from_dir(args.model_dir)
    decision = choose_resident_backend(
        args.model_dir, cfg, "fast", available_bytes=available)
    if not decision.admitted:
        raise SystemExit(
            f"resident backend was not admitted: {decision.reason}")

    engine = ResidentMLXLMEngine(
        args.model_dir, cfg, RuntimeConfig(), decision)
    report = {
        "schema": "voom.qwen-resident-native-mtp-ab.v1",
        "model_dir": str(args.model_dir),
        "request_shape": {
            "subject": "heterogeneous-neutral-runtime-prose",
            "tools": 0,
            "messages": 0,
            "prompt_tokens": args.prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "thresholds": {
            "true_peak_metal_gb_strictly_below": args.max_peak_metal_gb,
            "minimum_decode_speedup": args.min_speedup,
        },
        "failures": [],
        "passed": False,
    }
    try:
        seed_ids = engine.tokenizer.encode(
            "A neutral benchmark alternates mathematics, cooking, travel, "
            "history, software, music, and ordinary household questions. "
        ).ids
        prompt_ids = _repeat_to_length(seed_ids, args.prompt_tokens)
        prompt = SimpleNamespace(token_ids=tuple(prompt_ids))
        sampling = SamplingParams(
            temperature=args.temperature, seed=args.seed)

        os.environ["VMODEL_MLX_LM_NATIVE_MTP_DECODE"] = "1"
        cold_mtp = engine.generate(
            prompt, max_tokens=args.max_output_tokens, sampling=sampling)
        os.environ["VMODEL_MLX_LM_NATIVE_MTP_DECODE"] = "0"
        plain = engine.generate(
            prompt, max_tokens=args.max_output_tokens, sampling=sampling)
        os.environ["VMODEL_MLX_LM_NATIVE_MTP_DECODE"] = "1"
        mtp = engine.generate(
            prompt, max_tokens=args.max_output_tokens, sampling=sampling)
        report.update({
            "cold_mtp": cold_mtp,
            "plain_exact": plain,
            "mtp_exact": mtp,
            "decode_speedup": (
                plain["decode_s"] / mtp["decode_s"]
                if mtp["decode_s"] else 0.0),
        })

        path = mtp.get("path_stats", {})
        if path.get("qwen_native_mtp_loaded") != 1:
            report["failures"].append("released MTP head was not loaded")
        if path.get("qwen_native_mtp_used") != 1:
            report["failures"].append("native MTP verification did not engage")
        if path.get("prompt_cache_source") != "hot-prompt-exact":
            report["failures"].append(
                "MTP A/B did not use the identical prompt endpoint")
        if (plain.get("path_stats", {}).get("prompt_cache_source")
                != "hot-prompt-exact"):
            report["failures"].append(
                "plain A/B did not use the identical prompt endpoint")
        if args.temperature == 0 and not (
                cold_mtp["tokens"] == plain["tokens"] == mtp["tokens"]):
            report["failures"].append(
                "greedy native MTP tokens differ from plain target")
        if args.temperature != 0:
            report["failures"].append(
                "stochastic native MTP is measurement-only until the "
                "greedy byte-identity gate passes")
        peak = max(
            int(cold_mtp.get("true_peak_metal_bytes", 0)),
            int(plain.get("true_peak_metal_bytes", 0)),
            int(mtp.get("true_peak_metal_bytes", 0)),
        )
        if not peak < int(args.max_peak_metal_gb * 1_000_000_000):
            report["failures"].append("native MTP crossed Metal ceiling")
        if report["decode_speedup"] < args.min_speedup:
            report["failures"].append(
                "native MTP decode speedup missed threshold")
        report["passed"] = not report["failures"]
    finally:
        engine.close()

    _atomic_json(args.result_json, report)
    print(
        f"[resident-native-mtp] "
        f"{'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
