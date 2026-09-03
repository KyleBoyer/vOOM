#!/usr/bin/env python3
"""Locate the first state drift in GLM-5.3 serial-position verification.

The oracle builds one released prompt endpoint, forks it twice, then feeds an
identical token window through (a) canonical one-token target calls and (b) the
layer-major serial-position verifier.  Per-layer hidden hashes and complete
endpoint hashes distinguish arithmetic-shape drift from rollback mistakes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.fixtures.glm53_mtp_real_probe import DEFAULT_PROMPT  # noqa: E402
from tests.fixtures.qwen38_dflash2_gate import (  # noqa: E402
    _array_payload,
    _state_digest,
)


def _sha(value) -> str:
    return hashlib.sha256(_array_payload(value)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--expert-fetch-batch", type=int, default=8)
    parser.add_argument("--expert-batch-prefetch-depth", type=int, default=2)
    parser.add_argument("--expert-batch-prefetch-workers", type=int, default=2)
    parser.add_argument(
        "--trace-layer", type=int, default=-1,
        help="capture the reduced input/output of one MLA attention layer")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    window = [int(value) for value in args.tokens.split(",") if value.strip()]
    if len(window) < 2:
        raise SystemExit("--tokens needs at least two comma-separated IDs")
    if args.result.exists():
        raise SystemExit(f"result already exists: {args.result}")

    os.environ["VMODEL_GLM53_MTP"] = "0"
    os.environ["VMODEL_GLM53_HOT_PROMPT_KV"] = "0"
    os.environ["VMODEL_EXECUTION_PROFILE"] = ""
    os.environ["VMODEL_GLM53_EXPERT_FETCH_BATCH"] = str(
        args.expert_fetch_batch)
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] = "1"
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] = str(
        args.expert_batch_prefetch_depth)
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] = str(
        args.expert_batch_prefetch_workers)

    from runtime.server import EngineManager

    manager = EngineManager()
    try:
        engine = manager.get(args.model, "lossless")
        seed = engine.generate(args.prompt, max_tokens=1, stop=[])
        if seed["tokens"] != window[:1]:
            raise SystemExit(
                f"window does not start at prompt token: "
                f"{window[0]} != {seed['tokens'][0]}")
        base = engine.last_kv
        if base is None:
            raise RuntimeError("plain prompt did not retain an endpoint")
        sequential_kv = base.fork()
        serial_kv = base.fork()
        layers = tuple(range(engine.cfg.num_hidden_layers))
        if args.trace_layer >= engine.cfg.num_hidden_layers:
            raise SystemExit("--trace-layer is outside the model")
        engine._request_profiler = None

        attention_trace = {"sequential": [], "serial": []}
        trace_mode = [""]
        if args.trace_layer >= 0:
            import runtime.glm5_next as glm5_next

            original_attention = glm5_next.glm5_next_mla_attention

            def traced_attention(*call_args, **call_kwargs):
                value = original_attention(*call_args, **call_kwargs)
                layer = int(call_args[5])
                mode = trace_mode[0]
                if layer == args.trace_layer and mode in attention_trace:
                    mx.eval(call_args[0], value)
                    attention_trace[mode].append((call_args[0], value))
                return value

            glm5_next.glm5_next_mla_attention = traced_attention

        sequential_taps: dict[int, list] = {layer: [] for layer in layers}
        sequential_top1 = []
        sequential_logits_sha = []
        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        trace_mode[0] = "sequential"
        for token in window:
            logits = engine.forward_tokens(
                [token], sequential_kv, tap_layers=layers)
            sequential_top1.append(int(mx.argmax(logits[-1])))
            sequential_logits_sha.append(_sha(logits[-1]))
            for layer in layers:
                sequential_taps[layer].append(engine._tap_hidden[layer])
        sequential_s = time.perf_counter() - started
        sequential_reads = engine.cache.stats.bytes_read - reads_before
        sequential_hidden = engine._h_last
        sequential_state = _state_digest(
            engine, kv=sequential_kv, hidden=sequential_hidden)
        sequential_layer_sha = {}
        for layer in layers:
            joined = mx.concatenate(sequential_taps[layer], axis=1)
            mx.eval(joined)
            sequential_layer_sha[str(layer)] = _sha(joined)

        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        trace_mode[0] = "serial"
        serial_logits = engine.forward_tokens_serial_positions(
            window, serial_kv, tap_layers=layers)
        serial_s = time.perf_counter() - started
        serial_reads = engine.cache.stats.bytes_read - reads_before
        serial_top1 = [int(value) for value in mx.argmax(
            serial_logits, axis=-1).tolist()]
        serial_logits_sha = [_sha(serial_logits[index])
                             for index in range(len(window))]
        serial_hidden = engine._h_last
        serial_state = _state_digest(
            engine, kv=serial_kv, hidden=serial_hidden)
        serial_layer_sha = {
            str(layer): _sha(engine._tap_hidden[layer]) for layer in layers
        }

        layer_equal = {
            str(layer): sequential_layer_sha[str(layer)]
            == serial_layer_sha[str(layer)]
            for layer in layers
        }
        state_keys = sorted(set(sequential_state["tensor_sha256"])
                            | set(serial_state["tensor_sha256"]))
        state_differences = [
            name for name in state_keys
            if sequential_state["tensor_sha256"].get(name)
            != serial_state["tensor_sha256"].get(name)
        ]
        trace = None
        if args.trace_layer >= 0:
            trace = {"layer": args.trace_layer}
            for mode, rows in attention_trace.items():
                trace[f"{mode}_calls"] = len(rows)
                trace[f"{mode}_input_sha"] = [_sha(value[0]) for value in rows]
                trace[f"{mode}_output_sha"] = [_sha(value[1]) for value in rows]
            trace["input_equal"] = (
                trace["sequential_input_sha"] == trace["serial_input_sha"])
            trace["output_equal"] = (
                trace["sequential_output_sha"] == trace["serial_output_sha"])
        document = {
            "schema": "voom.glm53-serial-position-state-oracle.v1",
            "model": str(args.model),
            "prompt_tokens": seed["prompt_tokens"],
            "window": window,
            "sequential_s": sequential_s,
            "serial_s": serial_s,
            "sequential_reads": sequential_reads,
            "serial_reads": serial_reads,
            "top1_equal": sequential_top1 == serial_top1,
            "sequential_top1": sequential_top1,
            "serial_top1": serial_top1,
            "logits_sha_equal": sequential_logits_sha == serial_logits_sha,
            "sequential_logits_sha": sequential_logits_sha,
            "serial_logits_sha": serial_logits_sha,
            "layer_equal": layer_equal,
            "first_different_layer": next(
                (layer for layer in layers if not layer_equal[str(layer)]),
                None),
            "sequential_layer_sha": sequential_layer_sha,
            "serial_layer_sha": serial_layer_sha,
            "state_equal": sequential_state["sha256"]
            == serial_state["sha256"],
            "state_tensor_equal": len(state_keys) - len(state_differences),
            "state_tensor_total": len(state_keys),
            "state_differences": state_differences,
            "attention_trace": trace,
            "sequential_state": {
                key: sequential_state[key]
                for key in ("sha256", "component_sha256", "kv_offset",
                            "layer_lengths")
            },
            "serial_state": {
                key: serial_state[key]
                for key in ("sha256", "component_sha256", "kv_offset",
                            "layer_lengths")
            },
        }
        rendered = json.dumps(document, indent=2, sort_keys=True)
        print(rendered)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
