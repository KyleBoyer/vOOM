"""Real-weight GLM-5.3 multi-position verifier and rollback oracle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", default="Hi there")
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.positions < 2:
        parser.error("positions must be at least two")

    import mlx.core as mx

    from runtime.server import EngineManager

    def array_equal(left, right) -> bool:
        if left is None or right is None:
            return left is right
        same = mx.array_equal(left, right)
        mx.eval(same)
        return bool(same.item())

    def histories_equal(left, right) -> bool:
        if left is None or right is None:
            return left is right
        return len(left) == len(right) and all(
            array_equal(a, b) for a, b in zip(left, right))

    manager = EngineManager()
    try:
        engine = manager.get(args.model, "lossless")
        encoded = list(engine.tokenizer.encode(args.text).ids)
        if not encoded:
            raise ValueError("oracle text must encode to at least one token")
        token_ids = [
            encoded[index % len(encoded)]
            for index in range(args.positions)
        ]

        reference = engine.new_kv()
        taps = tuple(range(engine.cfg.num_hidden_layers))
        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        reference_logits_rows = []
        reference_states = []
        reference_taps = []
        for token in token_ids:
            logits = engine.forward_tokens(
                [token], reference, tap_layers=taps)
            mx.eval(logits)
            reference_logits_rows.append(logits)
            reference_states.append(reference.fork())
            captured = {
                layer: engine._tap_hidden[layer]
                for layer in taps
            }
            mx.eval(*captured.values())
            reference_taps.append(captured)
        reference_s = time.perf_counter() - started
        reference_bytes = engine.cache.stats.bytes_read - reads_before

        candidate = engine.new_kv()
        base_kda = candidate.kda_cache.fork()
        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        candidate_logits = engine.forward_tokens_serial_positions(
            token_ids, candidate, tap_layers=taps,
            capture_kda_factors=True)
        mx.eval(candidate_logits)
        candidate_taps = dict(engine._tap_hidden)
        mx.eval(*candidate_taps.values())
        candidate_s = time.perf_counter() - started
        candidate_bytes = engine.cache.stats.bytes_read - reads_before

        factors = engine.consume_serial_kda_factors()
        if factors is None:
            raise RuntimeError("serial verifier did not retain KDA factors")
        prefix_reports = []
        for count, prefix_state in enumerate(reference_states, start=1):
            restored = candidate.fork()
            rollback = getattr(restored, "rollback", None)
            if callable(rollback):
                rollback(count)
            else:
                restored.trim(count)
            restored.kda_cache = factors.commit_prefix(base_kda, count)
            restored.kda_cache.synchronize()

            state_checks = {
                "attention": [],
                "dsa": [],
                "kda_state": [],
                "conv_history": [],
            }
            for layer in range(engine.cfg.num_hidden_layers):
                state_checks["kda_state"].append(array_equal(
                    prefix_state.kda_cache.state(layer),
                    restored.kda_cache.state(layer)))
                state_checks["conv_history"].append(histories_equal(
                    prefix_state.kda_cache.conv_history(layer),
                    restored.kda_cache.conv_history(layer)))
                length = prefix_state.layer_lengths()[layer]
                left = prefix_state.keys[layer]
                right = restored.keys[layer]
                if left is not None:
                    if prefix_state.compressed_mla:
                        left = left[:, :length, :]
                        right = right[:, :length, :]
                    else:
                        left = left[:, :, :length, :]
                        right = right[:, :, :length, :]
                state_checks["attention"].append(array_equal(left, right))
            for layer in engine.cfg.full_attn_layers:
                state_checks["dsa"].append(array_equal(
                    prefix_state.dsa.k_idx.get(layer),
                    restored.dsa.k_idx.get(layer)))
                state_checks["dsa"].append(array_equal(
                    prefix_state.dsa.pool_keys.get(layer),
                    restored.dsa.pool_keys.get(layer)))
            component_exact = {
                name: all(checks)
                for name, checks in state_checks.items()
            }
            prefix_reports.append({
                "positions": count,
                "state_checks": sum(map(len, state_checks.values())),
                "component_exact": component_exact,
                "state_exact": all(component_exact.values()),
            })

        tap_exact = {}
        first_tap_divergence = None
        for layer in taps:
            layer_checks = []
            for position in range(args.positions):
                exact = array_equal(
                    reference_taps[position][layer],
                    candidate_taps[layer][:, position:position + 1, :])
                layer_checks.append(exact)
                if not exact and first_tap_divergence is None:
                    first_tap_divergence = {
                        "layer": layer,
                        "position": position,
                    }
            tap_exact[str(layer)] = layer_checks

        reference_logits = mx.concatenate(reference_logits_rows, axis=0)
        mx.eval(reference_logits)
        result = {
            "schema": "voom.glm53-serial-verifier-oracle.v1",
            "tokens": token_ids,
            "greedy_reference": [
                int(value) for value in mx.argmax(reference_logits, axis=-1)
            ],
            "greedy_candidate": [
                int(value) for value in mx.argmax(candidate_logits, axis=-1)
            ],
            "logits_array_equal": array_equal(
                reference_logits, candidate_logits),
            "prefixes": prefix_reports,
            "first_tap_divergence": first_tap_divergence,
            "tap_exact": tap_exact,
            "prefix_state_exact": all(
                report["state_exact"] for report in prefix_reports),
            "reference_s": reference_s,
            "candidate_s": candidate_s,
            "speedup": reference_s / candidate_s,
            "reference_bytes": reference_bytes,
            "candidate_bytes": candidate_bytes,
            "read_reduction": reference_bytes / candidate_bytes,
            "factor_bytes": factors.nbytes(),
            "peak_metal_bytes": engine._true_peak_metal_bytes,
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.result is not None:
            result_path = args.result.resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw = tempfile.mkstemp(
                prefix=f".{result_path.name}.", suffix=".tmp",
                dir=result_path.parent)
            temporary = Path(raw)
            try:
                with os.fdopen(descriptor, "w") as handle:
                    handle.write(rendered + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, result_path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        if (not args.diagnostic and (
                not result["logits_array_equal"]
                or not result["prefix_state_exact"]
                or result["greedy_reference"] != result["greedy_candidate"])):
            raise SystemExit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
