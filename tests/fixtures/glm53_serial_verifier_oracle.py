"""Real-weight GLM-5.3 two-position verifier and KDA rollback oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", default="Hi there")
    args = parser.parse_args()

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
        token_ids = list(engine.tokenizer.encode(args.text).ids)[:2]
        if len(token_ids) != 2:
            raise ValueError("oracle text must encode to at least two tokens")

        reference = engine.new_kv()
        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        first_logits = engine.forward_tokens([token_ids[0]], reference)
        mx.eval(first_logits)
        prefix_state = reference.fork()
        second_logits = engine.forward_tokens([token_ids[1]], reference)
        mx.eval(second_logits)
        reference_s = time.perf_counter() - started
        reference_bytes = engine.cache.stats.bytes_read - reads_before

        candidate = engine.new_kv()
        base_kda = candidate.kda_cache.fork()
        reads_before = engine.cache.stats.bytes_read
        started = time.perf_counter()
        candidate_logits = engine.forward_tokens_serial_positions(
            token_ids, candidate, capture_kda_factors=True)
        mx.eval(candidate_logits)
        candidate_s = time.perf_counter() - started
        candidate_bytes = engine.cache.stats.bytes_read - reads_before

        factors = engine.consume_serial_kda_factors()
        if factors is None:
            raise RuntimeError("serial verifier did not retain KDA factors")
        candidate.trim(1)
        candidate.kda_cache = factors.commit_prefix(base_kda, 1)
        candidate.kda_cache.synchronize()

        state_checks = []
        for layer in range(engine.cfg.num_hidden_layers):
            state_checks.extend((
                array_equal(
                    prefix_state.kda_cache.state(layer),
                    candidate.kda_cache.state(layer)),
                histories_equal(
                    prefix_state.kda_cache.conv_history(layer),
                    candidate.kda_cache.conv_history(layer)),
                array_equal(prefix_state.keys[layer], candidate.keys[layer]),
            ))
        for layer in engine.cfg.full_attn_layers:
            state_checks.append(array_equal(
                prefix_state.dsa.k_idx.get(layer),
                candidate.dsa.k_idx.get(layer)))

        reference_logits = mx.concatenate(
            [first_logits, second_logits], axis=0)
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
            "prefix_state_checks": len(state_checks),
            "prefix_state_exact": all(state_checks),
            "reference_s": reference_s,
            "candidate_s": candidate_s,
            "speedup": reference_s / candidate_s,
            "reference_bytes": reference_bytes,
            "candidate_bytes": candidate_bytes,
            "read_reduction": reference_bytes / candidate_bytes,
            "factor_bytes": factors.nbytes(),
            "peak_metal_bytes": engine._true_peak_metal_bytes,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if (not result["logits_array_equal"]
                or not result["prefix_state_exact"]
                or result["greedy_reference"] != result["greedy_candidate"]):
            raise SystemExit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
