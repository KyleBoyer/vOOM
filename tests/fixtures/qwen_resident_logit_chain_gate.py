#!/usr/bin/env python3
"""Real-checkpoint exactness/speed gate for resident generated-logit chains.

The retained path is content-blind: raw target logits are reused only while
the new sampled token prefix equals the previous token prefix.  This fixture
uses two different seeds for stochastic requests, then compares the chained
run with a plain exact-prompt run reset to the same second seed.  Equality
therefore covers both cached steps and the sequential divergence catch-up.
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
    parser.add_argument("--build-seed", type=int, default=101)
    parser.add_argument("--verify-seed", type=int, default=20260730)
    parser.add_argument("--max-peak-metal-gb", type=float, default=8.5)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    args = parser.parse_args()
    if args.result_json.exists():
        raise SystemExit(
            f"refusing to overwrite result artifact: {args.result_json}")

    available = int(psutil.virtual_memory().available)
    os.environ["VMODEL_RESIDENT_BACKEND"] = "mlx-lm"
    os.environ["VMODEL_MLX_LM_PROMPT_CACHE"] = "1"
    os.environ["VMODEL_MLX_LM_NATIVE_MTP"] = "0"
    os.environ["VMODEL_MLX_LM_LOGIT_CHAIN"] = "1"
    cfg = ModelConfig.from_dir(args.model_dir)
    decision = choose_resident_backend(
        args.model_dir, cfg, "fast", available_bytes=available)
    if not decision.admitted:
        raise SystemExit(
            f"resident backend was not admitted: {decision.reason}")

    engine = ResidentMLXLMEngine(
        args.model_dir, cfg, RuntimeConfig(), decision)
    report = {
        "schema": "voom.qwen-resident-logit-chain-ab.v1",
        "model_dir": str(args.model_dir),
        "request_shape": {
            "subject": "heterogeneous-neutral-runtime-prose",
            "tools": 0,
            "messages": 0,
            "prompt_tokens": args.prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "build_seed": args.build_seed,
            "verify_seed": args.verify_seed,
        },
        "failures": [],
        "passed": False,
    }
    try:
        seed_ids = engine.tokenizer.encode(
            "A neutral benchmark alternates mathematics, cooking, travel, "
            "history, software, music, and ordinary household questions. "
        ).ids
        prompt = SimpleNamespace(token_ids=tuple(
            _repeat_to_length(seed_ids, args.prompt_tokens)))
        build_sampling = SamplingParams(
            temperature=args.temperature, seed=args.build_seed)
        verify_sampling = SamplingParams(
            temperature=args.temperature, seed=args.verify_seed)

        cold = engine.generate(
            prompt, max_tokens=args.max_output_tokens,
            sampling=build_sampling)
        chained = engine.generate(
            prompt, max_tokens=args.max_output_tokens,
            sampling=verify_sampling)
        os.environ["VMODEL_MLX_LM_LOGIT_CHAIN"] = "0"
        plain = engine.generate(
            prompt, max_tokens=args.max_output_tokens,
            sampling=verify_sampling)
        speedup = (
            plain["decode_s"] / chained["decode_s"]
            if chained["decode_s"] else float("inf"))
        report.update({
            "cold_build": cold,
            "chained_exact": chained,
            "plain_exact": plain,
            "decode_speedup": speedup,
        })

        path = chained.get("path_stats", {})
        if chained["tokens"] != plain["tokens"]:
            report["failures"].append(
                "logit-chain tokens differ from same-seed plain target")
        if chained["text"] != plain["text"]:
            report["failures"].append(
                "logit-chain text differs from same-seed plain target")
        if path.get("resident_logit_chain_eligible") != 1:
            report["failures"].append("retained logit chain was not eligible")
        if args.temperature == 0:
            expected = max(0, len(chained["tokens"]) - 1)
            if path.get("resident_logit_chain_reused_step_logits") != expected:
                report["failures"].append(
                    "greedy repeat did not reuse the complete retained chain")
            if path.get("resident_logit_chain_catchup_sweeps") != 0:
                report["failures"].append(
                    "greedy complete-chain hit unexpectedly refed target")
        peak = max(
            int(cold.get("true_peak_metal_bytes", 0)),
            int(chained.get("true_peak_metal_bytes", 0)),
            int(plain.get("true_peak_metal_bytes", 0)))
        if not peak < int(args.max_peak_metal_gb * 1_000_000_000):
            report["failures"].append("logit-chain gate crossed Metal ceiling")
        if speedup < args.min_speedup:
            report["failures"].append("logit-chain speedup missed threshold")
        report["passed"] = not report["failures"]
    finally:
        engine.close()

    _atomic_json(args.result_json, report)
    print(
        f"[resident-logit-chain] "
        f"{'PASS' if report['passed'] else 'FAIL'} "
        f"{args.result_json}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
