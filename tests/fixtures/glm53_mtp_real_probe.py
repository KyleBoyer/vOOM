"""One released GLM-5.3 NextN round with exact target verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_PROMPT = (
    "You are a careful assistant. Summarize this request in one short clause: "
    "compare two exact runtime schedules while preserving every released-model operation."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--draft-tokens", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.draft_tokens <= 5:
        raise ValueError("draft-tokens must be in [1, 5]")

    # The cold max-one request retains exactly the prompt endpoint and logits;
    # no capture-specific content participates in cache selection.
    os.environ["VMODEL_GLM53_HOT_PROMPT_KV"] = "1"
    os.environ["VMODEL_GLM53_HOT_KV_SLOTS"] = "1"
    os.environ["VMODEL_GLM53_HOT_KV_MIN_TOKENS"] = "0"

    import mlx.core as mx

    from runtime.glm_mtp import MTPDrafter
    from runtime.kv_cache import KVCache
    from runtime.server import EngineManager

    manager = EngineManager()
    try:
        engine = manager.get(args.model, "lossless")
        prompt_ids = list(engine.tokenizer.encode(args.prompt).ids)
        seed = engine.generate(args.prompt, max_tokens=1, stop=[])
        first_token = int(seed["tokens"][0])
        if first_token in engine.cfg.eos_token_ids:
            raise RuntimeError("probe prompt terminated before an MTP round")
        if engine._h_window is None or engine._h_window.shape[1] != len(prompt_ids):
            raise RuntimeError("cold prefill did not retain the target hidden window")
        slot = next(
            (value for value in engine._hot_prompt_slots
             if value.tokens == tuple(prompt_ids)),
            None,
        )
        if slot is None:
            raise RuntimeError("cold prefill did not retain the exact prompt endpoint")

        drafter = MTPDrafter(engine)
        mtp_kv = KVCache(drafter.mtp_layer + 1)
        mtp_kv.compressed_mla = True
        draft_reads_before = engine.cache.stats.bytes_read
        draft_started = time.perf_counter()
        if len(prompt_ids) > 1:
            drafter.prefill(prompt_ids, engine._h_window, mtp_kv)
        proposals = drafter.draft_tokens(
            engine._h_last, first_token, args.draft_tokens, mtp_kv,
            offset=len(prompt_ids) - 1)
        draft_s = time.perf_counter() - draft_started
        draft_bytes = engine.cache.stats.bytes_read - draft_reads_before

        target_kv = slot.kv.fork()
        base_kda = target_kv.kda_cache.fork()
        verify_tokens = [first_token] + proposals
        verify_reads_before = engine.cache.stats.bytes_read
        verify_started = time.perf_counter()
        logits = engine.forward_tokens_serial_positions(
            verify_tokens, target_kv, capture_kda_factors=True)
        mx.eval(logits)
        verify_s = time.perf_counter() - verify_started
        verify_bytes = engine.cache.stats.bytes_read - verify_reads_before
        factors = engine.consume_serial_kda_factors()
        if factors is None:
            raise RuntimeError("target verifier did not retain KDA factors")

        greedy = [int(value) for value in mx.argmax(logits, axis=-1)]
        accepted = 0
        while (accepted < len(proposals)
               and greedy[accepted] == proposals[accepted]):
            accepted += 1
        fed = accepted + 1
        target_kv.trim(len(prompt_ids) + fed)
        if fed < len(verify_tokens):
            target_kv.kda_cache = factors.commit_prefix(base_kda, fed)
        target_kv.kda_cache.synchronize()
        emitted = [first_token] + proposals[:accepted] + [greedy[accepted]]

        result = {
            "schema": "voom.glm53-mtp-real-probe.v1",
            "prompt_tokens": len(prompt_ids),
            "seed_token": first_token,
            "proposals": proposals,
            "target_greedy": greedy,
            "accepted": accepted,
            "acceptance_rate": accepted / len(proposals),
            "emitted": emitted,
            "seed_s": seed["total_s"],
            "seed_bytes": seed["path_stats"]["weight_store_bytes_read"],
            "draft_s": draft_s,
            "draft_bytes": draft_bytes,
            "verify_s": verify_s,
            "verify_bytes": verify_bytes,
            "round_s": draft_s + verify_s,
            "round_bytes": draft_bytes + verify_bytes,
            "committed_tokens": accepted + 1,
            "seconds_per_committed_token": (
                (draft_s + verify_s) / (accepted + 1)),
            "factor_bytes": factors.nbytes(),
            "peak_metal_bytes": engine._true_peak_metal_bytes,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        manager.close()


if __name__ == "__main__":
    main()
