#!/usr/bin/env python3
"""Run one real Kimi K3 streamed sweep under the configured Metal ceiling.

This is the narrow regression gate for one-position decode scheduling.  It
loads the real checkpoint through EngineManager, runs every K3 layer once,
and records the runtime's per-layer transient observations plus MLX's true
peak.  A fresh 30-second machine preflight runs before MLX is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _preflight(path: Path) -> dict:
    completed = subprocess.run([
        sys.executable, "-m", "runtime.memory_preflight",
        "--workspace", str(ROOT), "--sample-seconds", "30",
        "--min-root-free-gb", "10", "--result", str(path),
    ], cwd=ROOT, check=False)
    report = json.loads(path.read_text())
    if completed.returncode:
        raise RuntimeError(
            f"memory preflight deferred K3 decode: {report.get('reasons')}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--preflight-result", required=True, type=Path)
    parser.add_argument("--profile", default="kimi-k3-this-mac-fast-tier")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models" / "Kimi-K3")
    parser.add_argument("--max-metal-gb", type=float, default=8.5)
    parser.add_argument("--restore-fingerprint")
    parser.add_argument("--prewarm-prefixes", type=Path)
    args = parser.parse_args()
    if args.result.exists():
        parser.error(f"refusing existing result: {args.result}")

    preflight = _preflight(args.preflight_result.resolve())

    # Importing EngineManager imports MLX lazily from get(), so the mandatory
    # machine preflight above is immediately before the first Metal/model I/O.
    sys.path.insert(0, str(ROOT))
    from runtime.profiles import apply_runtime_profiles
    from runtime.server import EngineManager

    application = apply_runtime_profiles(
        (args.profile,), environ=os.environ, activate=True)
    manager = EngineManager()
    started = time.perf_counter()
    try:
        engine = manager.get(args.model_dir.resolve(), "fast")
        import mlx.core as mx

        restore = None
        if args.restore_fingerprint:
            if args.prewarm_prefixes is None:
                parser.error(
                    "--restore-fingerprint requires --prewarm-prefixes")
            from runtime.hot_kv_persist import HotPromptKVPersistence
            from runtime.prewarm import (
                _read_request,
                load_startup_prefixes,
                prepare_k3_static_prefix,
            )

            entries = load_startup_prefixes((args.prewarm_prefixes,))
            entry = entries[0]
            _raw, request = _read_request(entry)
            prompt, prefix_metadata = prepare_k3_static_prefix(
                engine, args.model_dir.resolve(), request, "fast",
                cache_namespace=entry.cache_namespace,
            )
            journal = HotPromptKVPersistence(
                engine.rc.hot_prompt_kv_persist_dir,
                args.restore_fingerprint,
                engine.rc.hot_prompt_kv_chunk_size,
                max_checkpoints=engine.rc.hot_prompt_kv_persist_max_checkpoints,
                max_bytes=engine.rc.hot_prompt_kv_persist_max_mb * 1_000_000,
                config=engine.cfg,
                require_recurrent=True,
            )
            match = journal.find_best_match(
                prompt.token_ids, engine.rc.hot_prompt_kv_chunk_size,
                cache_namespace=entry.cache_namespace,
            )
            if match is None:
                raise RuntimeError("no exact persisted endpoint matched")
            loaded = journal.load_matched_chain(
                match, engine.cfg.num_hidden_layers)
            if loaded is None:
                raise RuntimeError("persisted endpoint failed validation")
            loaded_tokens, loaded_kv, _loaded_logits = loaded
            kv = engine._configure_restored_k3_spill(loaded_kv)
            respilled = engine._respill_completed_k3_state(kv)
            offset = int(kv.offset)
            restore = {
                "fingerprint": args.restore_fingerprint,
                "match_case": match["case"],
                "matched_tokens": match["matched"],
                "loaded_tokens": len(loaded_tokens),
                "declared_prefix_tokens": prefix_metadata["tokens"],
                "respilled": respilled,
            }
        else:
            kv = engine.new_kv()
            offset = 0
        x = engine._embed([1])
        mx.eval(x)
        mx.clear_cache()
        baseline = int(mx.get_active_memory())
        mx.reset_peak_memory()
        sweep_started = time.perf_counter()
        sweep_error = None
        try:
            x = engine._sweep(x, kv, offset=offset)
            mx.eval(x)
        except Exception as error:
            sweep_error = {
                "type": type(error).__name__,
                "message": str(error),
            }
        sweep_seconds = time.perf_counter() - sweep_started
        engine._note_true_peak()
        peak = max(
            int(mx.get_peak_memory()),
            int(getattr(engine, "_true_peak_metal_bytes", 0) or 0),
        )
        learned = {
            f"{positions}:{signature}": int(value)
            for (positions, signature), value
            in sorted(engine._layer_transient_by_signature.items())
        }
        max_transient = max(learned.values(), default=0)
        ceiling = int(args.max_metal_gb * 1_000_000_000)
        gates = {
            "real_k3_checkpoint": engine.cfg.model_type == "kimi_k3",
            "sweep_completed": sweep_error is None,
            "one_position_sweep": (
                sweep_error is None and tuple(x.shape[:2]) == (1, 1)),
            "all_layers_observed": sum(
                engine._layer_transient_observation_counts.values()
            ) == engine.cfg.num_hidden_layers,
            "metal_within_ceiling": peak <= ceiling,
        }
        if restore is not None:
            gates["restored_exact_declared_endpoint"] = (
                restore["match_case"] == "endpoint"
                and restore["matched_tokens"]
                == restore["loaded_tokens"]
                == restore["declared_prefix_tokens"]
            )
        output = {
            "schema": "voom.kimi-k3-decode-memory-gate.v1",
            "verdict": "PASS" if all(gates.values()) else "FAIL",
            "profile": args.profile,
            "profile_digest": application.effective_digest,
            "model_dir": str(args.model_dir.resolve()),
            "preflight": preflight,
            "load_and_sweep_seconds": time.perf_counter() - started,
            "sweep_seconds": sweep_seconds,
            "baseline_metal_bytes": baseline,
            "true_peak_metal_bytes": peak,
            "max_layer_transient_bytes": max_transient,
            "layer_transient_bytes": learned,
            "last_k3_transient_observation": getattr(
                engine, "_last_k3_transient_observation", None),
            "sweep_error": sweep_error,
            "restore": restore,
            "gates": gates,
        }
    finally:
        manager.close()

    _atomic_json(args.result.resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
