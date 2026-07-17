#!/usr/bin/env python3
"""Exact CPU benchmarks for Qwen3-VL detokenization and tool boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

from jinja2.utils import htmlsafe_json_dumps
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.incremental_decode import IncrementalDetokenizer  # noqa: E402
from runtime.server import _tool_capsule_spans  # noqa: E402


def _measure(function, repeats: int = 1):
    rows = []
    value = None
    for _ in range(repeats):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        value = function()
        rows.append((
            time.process_time() - cpu_start,
            time.perf_counter() - wall_start,
        ))
    return value, statistics.median(row[0] for row in rows), statistics.median(
        row[1] for row in rows)


def _decode_benchmark(tokenizer, lengths):
    seed = tokenizer.encode(
        "The café 工具 says CODE731. Unicode 🙂 boundaries stay exact.\n").ids
    maximum = max(lengths)
    ids = (seed * ((maximum + len(seed) - 1) // len(seed)))[:maximum]
    rows = []
    for length in lengths:
        selected = ids[:length]

        def legacy():
            text = ""
            for end in range(1, len(selected) + 1):
                text = tokenizer.decode(selected[:end])
            return text

        def incremental():
            decoder = IncrementalDetokenizer(tokenizer)
            emitted = []
            for token_id in selected:
                emitted.append(decoder.push_token(token_id))
            delta, text = decoder.finish_token_stream()
            emitted.append(delta)
            return text, "".join(emitted)

        reference, legacy_cpu, legacy_wall = _measure(legacy)
        (candidate, streamed), incremental_cpu, incremental_wall = _measure(
            incremental, repeats=3)
        exact = reference == candidate == streamed
        rows.append({
            "tokens": length,
            "exact": exact,
            "legacy_cpu_s": legacy_cpu,
            "incremental_cpu_s": incremental_cpu,
            "cpu_speedup": legacy_cpu / incremental_cpu,
            "legacy_wall_s": legacy_wall,
            "incremental_wall_s": incremental_wall,
            "wall_speedup": legacy_wall / incremental_wall,
        })
    return rows


def _validate_random_token_streams(tokenizer, cases: int = 40) -> dict:
    rng = random.Random(731)
    lengths = (1, 2, 7, 31, 127)
    vocab_size = tokenizer.get_vocab_size()
    for case in range(cases):
        length = lengths[case % len(lengths)]
        ids = [rng.randrange(vocab_size) for _ in range(length)]
        decoder = IncrementalDetokenizer(tokenizer)
        emitted = [decoder.push_token(token_id) for token_id in ids]
        delta, text = decoder.finish_token_stream()
        emitted.append(delta)
        reference = tokenizer.decode(ids)
        if text != reference or "".join(emitted) != reference:
            return {"cases": case + 1, "exact": False}
    return {"cases": cases, "exact": True}


def _serialize(tool):
    return str(htmlsafe_json_dumps(
        tool, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True))


def _legacy_tool_capsule_spans(prompt, prompt_tools, token_ids, offsets):
    if not prompt_tools or len(offsets) != len(token_ids):
        return ()
    search_end = prompt.rfind("</tools>")
    search_start = prompt.rfind("<tools>", 0, search_end)
    if search_start < 0 or search_end < 0:
        return ()
    search_start += len("<tools>")
    spans = []
    previous_token_end = 0
    for tool in prompt_tools:
        serialized = _serialize(tool)
        char_start = prompt.find(serialized, search_start, search_end)
        if char_start < 0:
            return ()
        char_end = char_start + len(serialized)
        token_start = next((
            index for index, (start, end) in enumerate(offsets)
            if start == char_start and end > start), None)
        token_end = next((
            index + 1 for index, (start, end) in enumerate(offsets)
            if index >= (token_start or 0)
            and start < char_end <= end), None)
        if (token_start is None or token_end is None
                or token_start < previous_token_end):
            return ()
        identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        spans.append((identity, token_start, token_end))
        previous_token_end = token_end
        search_start = char_end
    return tuple(spans)


def _capsule_benchmark(prompt_tokens: int, tool_count: int):
    tools = [{
        "type": "function",
        "function": {
            "name": f"tool_{index:03d}",
            "description": f"Inspect artifact {index:03d} exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    } for index in range(tool_count)]
    catalog = "<tools>\n" + "\n".join(map(_serialize, tools)) + "\n</tools>"
    if len(catalog) > prompt_tokens:
        raise ValueError("tool catalog exceeds requested synthetic prompt")
    prompt = "x" * (prompt_tokens - len(catalog)) + catalog
    offsets = tuple((index, index + 1) for index in range(len(prompt)))
    token_ids = range(len(offsets))

    reference, legacy_cpu, legacy_wall = _measure(
        lambda: _legacy_tool_capsule_spans(
            prompt, tools, token_ids, offsets))
    candidate, indexed_cpu, indexed_wall = _measure(
        lambda: _tool_capsule_spans(
            prompt, tools, token_ids, offsets), repeats=3)
    return {
        "prompt_tokens": prompt_tokens,
        "tools": tool_count,
        "exact": candidate == reference and len(candidate) == tool_count,
        "legacy_cpu_s": legacy_cpu,
        "indexed_cpu_s": indexed_cpu,
        "cpu_speedup": legacy_cpu / indexed_cpu,
        "legacy_wall_s": legacy_wall,
        "indexed_wall_s": indexed_wall,
        "wall_speedup": legacy_wall / indexed_wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer", type=Path,
        default=(Path.home() / "hf_cache/modelscope/models/"
                 "Qwen3-VL-2B-Instruct/tokenizer.json"))
    parser.add_argument("--prompt-tokens", type=int, default=131_072)
    parser.add_argument("--tools", type=int, default=256)
    args = parser.parse_args()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not tokenizer_path.is_file():
        parser.error(f"tokenizer is not local: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    decode = _decode_benchmark(tokenizer, (512, 2_048, 4_096))
    random_validation = _validate_random_token_streams(tokenizer)
    capsules = _capsule_benchmark(args.prompt_tokens, args.tools)
    passed = bool(
        all(row["exact"] and row["cpu_speedup"] > 1.0 for row in decode)
        and random_validation["exact"]
        and capsules["exact"] and capsules["cpu_speedup"] > 1.0)
    report = {
        "gate": "qwen3vl-cpu-hotpaths-v1",
        "passed": passed,
        "tokenizer": str(tokenizer_path),
        "decode": decode,
        "random_token_streams": random_validation,
        "tool_capsules": capsules,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
