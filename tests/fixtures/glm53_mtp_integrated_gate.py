"""Exercise the explicit GLM-5.3 native-MTP serving adapter on real weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fixtures.glm53_mtp_real_probe import DEFAULT_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--execution-profile", choices=("", "layers", "ops"), default="")
    parser.add_argument("--expert-fetch-batch", type=int, default=1)
    parser.add_argument("--expert-batch-prefetch", action="store_true")
    parser.add_argument("--trunk-prefetch-depth", type=int, default=0)
    parser.add_argument("--trunk-prefetch-workers", type=int, default=1)
    parser.add_argument("--expected-tokens", default="")
    args = parser.parse_args()

    os.environ["VMODEL_GLM53_MTP"] = "0" if args.plain else "1"
    os.environ["VMODEL_GLM53_MTP_DEPTH"] = str(args.depth)
    os.environ["VMODEL_GLM53_MTP_MAX_PROMPT_TOKENS"] = "2048"
    os.environ["VMODEL_GLM53_HOT_PROMPT_KV"] = "0"
    os.environ["VMODEL_EXECUTION_PROFILE"] = args.execution_profile
    os.environ["VMODEL_GLM53_EXPERT_FETCH_BATCH"] = str(
        args.expert_fetch_batch)
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] = (
        "1" if args.expert_batch_prefetch else "0")
    os.environ["VMODEL_GLM53_TRUNK_PREFETCH_DEPTH"] = str(
        args.trunk_prefetch_depth)
    os.environ["VMODEL_GLM53_TRUNK_PREFETCH_WORKERS"] = str(
        args.trunk_prefetch_workers)

    from runtime.server import EngineManager

    manager = EngineManager()
    try:
        engine = manager.get(args.model, "lossless")
        if args.runs <= 0:
            raise SystemExit("--runs must be positive")
        runs = []
        for run_index in range(args.runs):
            # Keep the cold cache-building call timing unperturbed; when a
            # repeat is requested, attribute only the decisive hot call.
            engine.rc.execution_profile = (
                args.execution_profile
                if args.runs == 1 or run_index == args.runs - 1 else "")
            result = engine.generate(
                args.prompt, max_tokens=args.max_tokens, stop=[])
            stats = result["path_stats"]
            runs.append({
                "run_index": run_index,
                "tokens": result["tokens"],
                "text": result["text"],
                "prefill_s": result["prefill_s"],
                "decode_s": result["decode_s"],
                "first_token_s": result["first_token_s"],
                "total_s": result["total_s"],
                "true_peak_metal_bytes": result["true_peak_metal_bytes"],
                "prompt_cache_exact_hit": stats.get(
                    "prompt_cache_exact_hit", 0),
                "prompt_cache_source": stats.get(
                    "prompt_cache_source", ""),
                "prompt_cache_kind": stats.get("prompt_cache_kind", ""),
                "prompt_cache_resident_bytes": stats.get(
                    "prompt_cache_resident_bytes", 0),
                "proposed": stats.get("speculative_proposed", 0),
                "accepted": stats.get("speculative_accepted", 0),
                "target_sweeps": stats.get(
                    "speculative_target_sweeps", 0),
                "draft_s": stats.get("speculative_draft_s", 0.0),
                "verify_s": stats.get(
                    "speculative_verify_decode_s", 0.0),
                "draft_bytes": stats.get("speculative_draft_bytes", 0),
                "weight_store_bytes_read": stats.get(
                    "weight_store_bytes_read", 0),
                "prefill_weight_store_bytes_read": stats.get(
                    "prefill_weight_store_bytes_read", 0),
                "decode_weight_store_bytes_read": stats.get(
                    "decode_weight_store_bytes_read", 0),
                "weight_fast_tier_bytes": stats.get(
                    "weight_fast_tier_bytes", 0),
                "weight_archive_bytes": stats.get(
                    "weight_archive_bytes", 0),
                "weight_prefetch_waits": stats.get(
                    "weight_prefetch_waits", 0),
                "weight_prefetch_wait_s": stats.get(
                    "weight_prefetch_wait_s", 0.0),
                "weight_prefetch_useful_pages": stats.get(
                    "weight_prefetch_useful_pages", 0),
                "weight_prefetch_useful_bytes": stats.get(
                    "weight_prefetch_useful_bytes", 0),
                "weight_prefetch_hidden_lower_bound_s": stats.get(
                    "weight_prefetch_hidden_lower_bound_s", 0.0),
                "expert_batch_prefetch_submitted": stats.get(
                    "expert_batch_prefetch_submitted", 0),
                "expert_batch_prefetch_wait_s": stats.get(
                    "expert_batch_prefetch_wait_s", 0.0),
                "expert_batch_prefetch_hidden_s": stats.get(
                    "expert_batch_prefetch_hidden_s", 0.0),
                "execution_profile": result.get("execution_profile"),
            })
        document = {
            "schema": "voom.glm53-mtp-integrated-gate.v1",
            "tokens": runs[-1]["tokens"],
            "text": runs[-1]["text"],
            "prompt_tokens": result["prompt_tokens"],
            "runs": runs,
            "prefill_s": runs[-1]["prefill_s"],
            "decode_s": runs[-1]["decode_s"],
            "first_token_s": runs[-1]["first_token_s"],
            "total_s": runs[-1]["total_s"],
            "true_peak_metal_bytes": runs[-1]["true_peak_metal_bytes"],
            "mtp_enabled": stats.get("glm53_mtp_enabled", 0),
            "mtp_used": stats.get("glm53_mtp_used", 0),
            "mtp_depth": stats.get("glm53_mtp_depth", 0),
            "proposed": stats.get("speculative_proposed", 0),
            "accepted": stats.get("speculative_accepted", 0),
            "acceptance_rate": stats.get(
                "speculative_acceptance_rate", 0.0),
            "target_sweeps": stats.get("speculative_target_sweeps", 0),
            "rounds": stats.get("speculative_rounds", []),
            "draft_s": stats.get("speculative_draft_s", 0.0),
            "verify_s": stats.get("speculative_verify_decode_s", 0.0),
            "draft_bytes": stats.get("speculative_draft_bytes", 0),
            "weight_store_bytes_read": stats.get(
                "weight_store_bytes_read", 0),
        }
        print(json.dumps(document, indent=2, sort_keys=True))
        if args.expected_tokens:
            expected = [
                int(value) for value in args.expected_tokens.split(",")
                if value.strip()
            ]
            if document["tokens"] != expected:
                raise SystemExit(
                    f"token mismatch: {document['tokens']} != {expected}")
        if not args.plain and not document["mtp_used"]:
            raise SystemExit("native MTP adapter did not engage")
        if args.runs > 1:
            expected = runs[0]["tokens"]
            if any(run["tokens"] != expected for run in runs[1:]):
                raise SystemExit("repeat token mismatch")
            if not args.plain and not runs[-1]["prompt_cache_exact_hit"]:
                raise SystemExit("native MTP repeat did not hit prompt cache")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
