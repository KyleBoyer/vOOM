#!/usr/bin/env python3
"""Live CPU/BGE quality and memory gate for hybrid tool retrieval.

Run only after ``runtime.memory_preflight`` passes. The catalog and queries are
synthetic/public; the report contains no private harness content. Exit 75 means
the host did not meet the 6-GB model-I/O admission precondition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.tool_embeddings import (
    QUERY_PREFIX, EmbeddingConfig, ToolEmbeddingCache, build_tool_cache,
    encode_texts_subprocess)
from runtime.toolcalls import rank_tool_indices, tool_search_capsule


def _tool(name: str, description: str):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "description": "Primary input"},
        }},
    }}


TOOLS = [
    _tool("workspace_execute", "Execute a program or shell command in a workspace."),
    _tool("file_write", "Persist text or bytes to a local filesystem path."),
    _tool("browser_open", "Navigate a browser to a web URL."),
    _tool("calendar_create", "Schedule a calendar event or appointment."),
    _tool("email_send", "Deliver electronic mail to a recipient inbox."),
    _tool("database_query", "Retrieve relational database records using SQL."),
    _tool("image_generate", "Create a new picture or illustration from a description."),
    _tool("pdf_extract", "Extract readable text from a PDF document."),
    _tool("weather_lookup", "Read the weather forecast for a location."),
    _tool("spreadsheet_update", "Update cells in an Excel spreadsheet."),
    _tool("git_commit", "Record source repository changes in Git."),
    _tool("audio_transcribe", "Convert recorded speech into written text."),
]

CASES = [
    ("invoke a program from a command-line environment", "workspace_execute"),
    ("save these bytes permanently on disk", "file_write"),
    ("visit this internet address in a web client", "browser_open"),
    ("arrange an appointment for next Tuesday", "calendar_create"),
    ("compose a note for someone's inbox", "email_send"),
    ("retrieve matching rows from relational storage", "database_query"),
    ("make a new picture from a description", "image_generate"),
    ("read the words inside a portable document", "pdf_extract"),
]


def _pressure():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_bytes": int(memory.available),
        "swap_used_bytes": int(swap.used),
        "swap_out_bytes": int(swap.sout),
    }


def _write(path: Path | None, value: dict):
    if path is None:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    initial = _pressure()
    if initial["available_bytes"] < 6_000_000_000:
        report = {
            "schema": "vmodel-tool-embedding-live-gate-v1",
            "verdict": "DEFERRED_PRECONDITION",
            "initial_pressure": initial,
        }
        _write(args.result_json, report)
        return 75

    config = EmbeddingConfig.from_env()
    capsules = [tool_search_capsule(tool) for tool in TOOLS]
    build_started = time.perf_counter()
    build = build_tool_cache(capsules, config=config, verify_model=True)
    build_seconds = time.perf_counter() - build_started

    # Batch-prime the public evaluation queries once. Production novel queries
    # use the same disposable subprocess one at a time; exact repeats hit cache.
    query_texts = [QUERY_PREFIX + query for query, _expected in CASES]
    cache = ToolEmbeddingCache(config.cache_dir)
    vectors, query_hits = cache.get_many("query", query_texts)
    missing = [index for index, vector in enumerate(vectors) if vector is None]
    query_started = time.perf_counter()
    if missing:
        encoded = encode_texts_subprocess(
            [query_texts[index] for index in missing], config)
        cache.store_many(
            "query", [query_texts[index] for index in missing], encoded,
            max_objects=config.query_cache_max)
    query_seconds = time.perf_counter() - query_started

    old_env = {
        key: os.environ.get(key) for key in (
            "VMODEL_TOOL_EMBEDDINGS", "VMODEL_TOOL_EMBEDDINGS_REQUIRED")
    }
    os.environ["VMODEL_TOOL_EMBEDDINGS"] = "1"
    os.environ["VMODEL_TOOL_EMBEDDINGS_REQUIRED"] = "1"
    rows = []
    try:
        for query, expected in CASES:
            messages = [{"role": "user", "content": query}]
            lexical = rank_tool_indices(TOOLS, messages)
            hybrid, metadata = rank_tool_indices(
                TOOLS, messages, use_embeddings=True, return_metadata=True)
            lexical_names = [TOOLS[index]["function"]["name"] for index in lexical]
            hybrid_names = [TOOLS[index]["function"]["name"] for index in hybrid]
            rows.append({
                "expected": expected,
                "lexical_rank": lexical_names.index(expected) + 1,
                "hybrid_rank": hybrid_names.index(expected) + 1,
                "hybrid_top": hybrid_names[0],
                "embedding_status": metadata.get("tool_embedding_status"),
                "query_cache_hit": metadata.get("tool_embedding_query_cache_hit"),
                "catalog_id": metadata.get("tool_embedding_catalog_id"),
            })
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    final = _pressure()
    hybrid_top1 = sum(row["hybrid_rank"] == 1 for row in rows) / len(rows)
    hybrid_mrr = sum(1.0 / row["hybrid_rank"] for row in rows) / len(rows)
    lexical_mrr = sum(1.0 / row["lexical_rank"] for row in rows) / len(rows)
    passed = (
        hybrid_top1 >= 0.875
        and hybrid_mrr >= lexical_mrr
        and all(row["embedding_status"] == "hybrid" for row in rows)
        and all(row["query_cache_hit"] == 1 for row in rows)
        and final["available_bytes"] >= 4_000_000_000
        and final["swap_out_bytes"] - initial["swap_out_bytes"] <= 16_000_000
    )
    report = {
        "schema": "vmodel-tool-embedding-live-gate-v1",
        "verdict": "PASS" if passed else "FAIL",
        "build": build,
        "build_seconds": round(build_seconds, 4),
        "query_batch_cache_hits": query_hits,
        "query_batch_encoded": len(missing),
        "query_batch_seconds": round(query_seconds, 4),
        "hybrid_top1": hybrid_top1,
        "hybrid_mrr": round(hybrid_mrr, 4),
        "lexical_mrr": round(lexical_mrr, 4),
        "rows": rows,
        "initial_pressure": initial,
        "final_pressure": final,
        "swap_out_growth_bytes": (
            final["swap_out_bytes"] - initial["swap_out_bytes"]),
    }
    _write(args.result_json, report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
