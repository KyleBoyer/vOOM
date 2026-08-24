#!/usr/bin/env python3
"""Real HTTP gate for explicit Qwen layer-stationary multi-request decode.

Runs two heterogeneous greedy legacy completions serially, then submits the
same raw prompts and budgets to the explicit batch endpoint.  The durable
artifact stores only hashes, token counts, timing, and scheduler telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


CASES = (
    {"id": "capital", "prompt": "The capital of France is", "max_tokens": 2},
    {"id": "sequence", "prompt": "Continue: one, two,", "max_tokens": 3},
)


def _post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(
        url, data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    return value, time.perf_counter() - started


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8077/v1")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    serial = []
    for case in CASES:
        response, wall = _post(
            f"{args.base_url}/completions",
            {
                "model": args.model,
                "prompt": case["prompt"],
                "max_tokens": case["max_tokens"],
                "temperature": 0,
                "stream": False,
            },
            args.timeout,
        )
        choice = response["choices"][0]
        text = str(choice["text"])
        serial.append({
            "id": case["id"], "text": text, "sha256": _sha256(text),
            "wall_seconds": wall,
        })

    batch, batch_wall = _post(
        f"{args.base_url}/qwen/layer-stationary/completions",
        {
            "model": args.model,
            "stream": False,
            "requests": [
                {**case, "temperature": 0} for case in CASES
            ],
        },
        args.timeout,
    )
    by_id = {choice["id"]: choice for choice in batch["choices"]}
    comparisons = []
    failures = []
    for expected in serial:
        actual = by_id.get(expected["id"])
        matched = bool(
            actual is not None and actual.get("text") == expected["text"])
        comparisons.append({
            "id": expected["id"],
            "serial_sha256": expected["sha256"],
            "batch_sha256": (
                _sha256(str(actual.get("text", ""))) if actual else None),
            "text_equal": matched,
            "completion_tokens": (
                int(actual.get("completion_tokens", 0)) if actual else 0),
        })
        if not matched:
            failures.append(f"{expected['id']}: batch text differs from serial")

    telemetry = batch.get("vmodel_multi_request_telemetry") or {}
    if telemetry.get("request_count") != len(CASES):
        failures.append("request_count telemetry mismatch")
    if telemetry.get("private_kv_endpoints") != len(CASES):
        failures.append("private KV endpoints are not independent")
    if telemetry.get("private_kda_endpoints") != len(CASES):
        failures.append("private KDA endpoints are not independent")
    if int(telemetry.get("layer_page_get_call_savings", 0)) <= 0:
        failures.append("layer-stationary scheduler saved no layer fetches")

    report = {
        "schema": "voom.qwen-multi-request-live.v1",
        "model": args.model,
        "cases": [
            {"id": case["id"], "prompt_sha256": _sha256(case["prompt"]),
             "max_tokens": case["max_tokens"]}
            for case in CASES
        ],
        "serial_wall_seconds": sum(row["wall_seconds"] for row in serial),
        "batch_wall_seconds": batch_wall,
        "comparisons": comparisons,
        "telemetry": telemetry,
        "failures": failures,
        "passed": not failures,
    }
    _write(args.result_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
