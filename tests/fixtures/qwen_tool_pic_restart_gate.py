#!/usr/bin/env python3
"""Real-Qwen proof that the first edited catalog after restart can use PIC."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.engine import StreamingEngine  # noqa: E402
from tests.fixtures.qwen_tool_pic_gate import (  # noqa: E402
    _parsed, _prompt, _runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=(Path.home()
                 / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4"))
    parser.add_argument("--tools", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repair-tokens", type=int, default=4)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    persist = ROOT / "tests/_tool_pic_restart_scratch"
    shutil.rmtree(persist, ignore_errors=True)
    target = "tool_019"
    path = "src/restart.py"

    try:
        source_rc = _runtime(
            model, pic=True, cache_mb=3000,
            repair=args.repair_tokens, chunk=128)
        source_rc.hot_prompt_kv_persist_dir = str(persist)
        source_rc.hot_prompt_kv_persist_max_checkpoints = 4
        source_engine = StreamingEngine(model, source_rc)
        try:
            source = _prompt(
                source_engine, model, args.tools, edited=False,
                target=target, path=path, max_tokens=args.max_tokens)
            source_engine.generate(source, max_tokens=1, stop=[])
        finally:
            source_engine.close()
            mx.clear_cache()

        restarted_rc = _runtime(
            model, pic=True, cache_mb=3000,
            repair=args.repair_tokens, chunk=128)
        restarted_rc.hot_prompt_kv_persist_dir = str(persist)
        restarted_rc.hot_prompt_kv_persist_max_checkpoints = 4
        restarted = StreamingEngine(model, restarted_rc)
        try:
            restored_capsules = len(restarted._hot_prompt_slots[0].tool_capsules)
            edited = _prompt(
                restarted, model, args.tools, edited=True,
                target=target, path=path, max_tokens=args.max_tokens)
            candidate = restarted.generate(
                edited, max_tokens=args.max_tokens, stop=[])
        finally:
            restarted.close()
            mx.clear_cache()

        control_engine = StreamingEngine(
            model, _runtime(
                model, pic=False, cache_mb=3000,
                repair=args.repair_tokens, chunk=128))
        try:
            source = _prompt(
                control_engine, model, args.tools, edited=False,
                target=target, path=path, max_tokens=args.max_tokens)
            control_engine.generate(source, max_tokens=1, stop=[])
            edited = _prompt(
                control_engine, model, args.tools, edited=True,
                target=target, path=path, max_tokens=args.max_tokens)
            control = control_engine.generate(
                edited, max_tokens=args.max_tokens, stop=[])
        finally:
            control_engine.close()
            mx.clear_cache()

        name, arguments = _parsed(candidate["text"], "qwen2")
        stats = candidate["path_stats"]
        speedup = control["prefill_s"] / candidate["prefill_s"]
        passed = (
            restored_capsules == args.tools
            and stats["tool_pic"] == 1
            and candidate["tokens"] == control["tokens"]
            and name == target and arguments.get("path") == path
            and speedup >= 1.01)
        print(json.dumps({
            "gate": "qwen-tool-pic-restart-v1",
            "passed": passed,
            "restored_capsules": restored_capsules,
            "same_ids": candidate["tokens"] == control["tokens"],
            "pic_exercised": stats["tool_pic"] == 1,
            "selected_tokens": stats["tool_pic_selected_tokens"],
            "reused_tokens": stats["tool_pic_reused_tokens"],
            "pic_prefill_s": round(candidate["prefill_s"], 6),
            "exact_prefill_s": round(control["prefill_s"], 6),
            "speedup": round(speedup, 3),
        }, indent=2))
        return 0 if passed else 1
    finally:
        shutil.rmtree(persist, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
