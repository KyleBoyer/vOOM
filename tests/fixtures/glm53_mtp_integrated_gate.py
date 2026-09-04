"""Exercise the explicit GLM-5.3 native-MTP serving adapter on real weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fixtures.glm53_mtp_real_probe import DEFAULT_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mtp-max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--mtp-confidence-telemetry", action="store_true")
    parser.add_argument("--mtp-min-logit-margin", type=float, default=0.0)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--execution-profile", choices=("", "layers", "ops"), default="")
    parser.add_argument("--expert-fetch-batch", type=int, default=1)
    parser.add_argument("--expert-batch-prefetch", action="store_true")
    parser.add_argument("--expert-batch-prefetch-depth", type=int, default=1)
    parser.add_argument("--expert-batch-prefetch-workers", type=int, default=1)
    parser.add_argument("--trunk-prefetch-depth", type=int, default=0)
    parser.add_argument("--trunk-prefetch-workers", type=int, default=1)
    parser.add_argument("--weight-cache-mb", type=int, default=5000)
    parser.add_argument("--state-digest", action="store_true")
    parser.add_argument(
        "--dsa-row-digest", action="store_true",
        help=("include per-position hashes for DSA index-key caches; "
              "requires --state-digest and is diagnostic-only"))
    parser.add_argument(
        "--probe-every", type=int, default=0,
        help="fixture-only adaptive-controller probe interval (0 keeps default)")
    parser.add_argument("--expected-tokens", default="")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.dsa_row_digest and not args.state_digest:
        parser.error("--dsa-row-digest requires --state-digest")

    os.environ["VMODEL_GLM53_MTP"] = "0" if args.plain else "1"
    os.environ["VMODEL_GLM53_MTP_DEPTH"] = str(args.depth)
    os.environ["VMODEL_GLM53_MTP_MAX_PROMPT_TOKENS"] = str(
        args.mtp_max_prompt_tokens)
    os.environ["VMODEL_GLM53_MTP_CONFIDENCE_TELEMETRY"] = (
        "1" if args.mtp_confidence_telemetry else "0")
    os.environ["VMODEL_GLM53_MTP_MIN_LOGIT_MARGIN"] = str(
        args.mtp_min_logit_margin)
    os.environ["VMODEL_GLM53_HOT_PROMPT_KV"] = "0"
    os.environ["VMODEL_EXECUTION_PROFILE"] = args.execution_profile
    os.environ["VMODEL_GLM53_EXPERT_FETCH_BATCH"] = str(
        args.expert_fetch_batch)
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] = (
        "1" if args.expert_batch_prefetch else "0")
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] = str(
        args.expert_batch_prefetch_depth)
    os.environ["VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] = str(
        args.expert_batch_prefetch_workers)
    os.environ["VMODEL_GLM53_TRUNK_PREFETCH_DEPTH"] = str(
        args.trunk_prefetch_depth)
    os.environ["VMODEL_GLM53_TRUNK_PREFETCH_WORKERS"] = str(
        args.trunk_prefetch_workers)
    os.environ["VMODEL_GLM53_SHORT_WEIGHT_CACHE_MB"] = str(
        args.weight_cache_mb)

    from runtime.server import EngineManager

    manager = EngineManager()
    try:
        engine = manager.get(args.model, "lossless")
        if args.probe_every < 0:
            raise SystemExit("--probe-every must be nonnegative")
        if args.probe_every:
            if args.plain:
                raise SystemExit("--probe-every requires native MTP")
            engine.decoder.PROBE_EVERY = args.probe_every
        if args.state_digest and not args.plain:
            engine.decoder._diagnostic_retain_generation_endpoint = True
        if args.state_digest and args.plain:
            # ``StreamingEngine._h_last`` is a drafting aid, not a generation
            # endpoint: ordinary token-at-a-time generation deliberately does
            # not refresh it after every decode sweep.  Capture the final fed
            # token's trunk row without changing its lifetime in production so
            # the plain arm can be compared to the speculative arm's committed
            # endpoint.  The wrapper adds no evaluation or arithmetic.
            target = getattr(engine, "target", engine)
            original_sweep = target._sweep

            def capture_endpoint_hidden(*sweep_args, **sweep_kwargs):
                hidden = original_sweep(*sweep_args, **sweep_kwargs)
                target._diagnostic_generation_hidden = hidden[:, -1:, :]
                return hidden

            target._diagnostic_generation_hidden = None
            target._sweep = capture_endpoint_hidden
        if args.runs <= 0:
            raise SystemExit("--runs must be positive")
        runs = []
        for run_index in range(args.runs):
            # Keep the cold cache-building call timing unperturbed; when a
            # repeat is requested, attribute only the decisive hot call.
            engine.rc.execution_profile = (
                args.execution_profile
                if args.runs == 1 or run_index == args.runs - 1 else "")
            pressure_before = psutil.swap_memory()
            result = engine.generate(
                args.prompt, max_tokens=args.max_tokens, stop=[])
            pressure_after = psutil.swap_memory()
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
                "swap_used_growth_bytes": max(
                    0, int(pressure_after.used) - int(pressure_before.used)),
                "swap_out_growth_bytes": max(
                    0, int(pressure_after.sout) - int(pressure_before.sout)),
                "governor_swap_pressure_events": stats.get(
                    "governor_swap_pressure_events", 0),
                "governor_swap_used_growth_bytes": stats.get(
                    "governor_swap_used_growth_bytes", 0),
                "governor_swap_out_growth_bytes": stats.get(
                    "governor_swap_out_growth_bytes", 0),
                "governor_reservation_failures": stats.get(
                    "governor_reservation_failures", 0),
                "glm53_exact_expert_shape": {
                    key.removeprefix("glm53_exact_expert_"): value
                    for key, value in stats.items()
                    if key.startswith("glm53_exact_expert_")
                },
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
                "controller_disabled_rounds": stats.get(
                    "speculative_controller_disabled_rounds", 0),
                "controller_probe_every": stats.get(
                    "speculative_controller_probe_every", 0),
                "controller_cost_estimate": stats.get(
                    "speculative_controller_cost_estimate", 0.0),
                "controller_acceptance_estimate": stats.get(
                    "speculative_controller_acceptance_estimate", 0.0),
                "mtp_confidence_enabled": stats.get(
                    "glm53_mtp_confidence_enabled", 0),
                "mtp_min_logit_margin": stats.get(
                    "glm53_mtp_min_logit_margin", 0.0),
                "mtp_confidence_candidates": stats.get(
                    "glm53_mtp_confidence_candidates", 0),
                "mtp_confidence_withheld": stats.get(
                    "glm53_mtp_confidence_withheld", 0),
                "mtp_logit_margin_min": stats.get(
                    "glm53_mtp_logit_margin_min", 0.0),
                "mtp_logit_margin_mean": stats.get(
                    "glm53_mtp_logit_margin_mean", 0.0),
                "mtp_logit_margin_max": stats.get(
                    "glm53_mtp_logit_margin_max", 0.0),
                "mtp_logit_margins": stats.get(
                    "glm53_mtp_logit_margins", ""),
                "mtp_sync_confidence_candidates": stats.get(
                    "glm53_mtp_sync_confidence_candidates", 0),
                "mtp_sync_logit_margins": stats.get(
                    "glm53_mtp_sync_logit_margins", ""),
                "mtp_state_only_prefill_tokens": stats.get(
                    "glm53_mtp_state_only_prefill_tokens", 0),
                "draft_s": stats.get("speculative_draft_s", 0.0),
                "verify_s": stats.get(
                    "speculative_verify_decode_s", 0.0),
                "draft_bytes": stats.get("speculative_draft_bytes", 0),
                "weight_store_bytes_read": stats.get(
                    "weight_store_bytes_read", 0),
                "weight_cache_hits": stats.get("weight_cache_hits", 0),
                "weight_cache_misses": stats.get("weight_cache_misses", 0),
                "weight_cache_evictions": stats.get(
                    "weight_cache_evictions", 0),
                "weight_cache_resident_bytes": stats.get(
                    "weight_cache_resident_bytes", 0),
                "weight_cache_budget_bytes": stats.get(
                    "weight_cache_budget_bytes", 0),
                "weight_cache_prefetch_hits": stats.get(
                    "weight_cache_prefetch_hits", 0),
                "expert_cache_hits": stats.get("expert_cache_hits", 0),
                "expert_cache_misses": stats.get("expert_cache_misses", 0),
                "glm53_native_fp8_dequant": stats.get(
                    "glm53_native_fp8_dequant", 0),
                "glm53_fp8_direct_qmv": stats.get(
                    "glm53_fp8_direct_qmv", 0),
                "glm53_fp8_direct_pages": stats.get(
                    "glm53_fp8_direct_pages", 0),
                "glm53_fp8_direct_resident_bytes": stats.get(
                    "glm53_fp8_direct_resident_bytes", 0),
                "glm53_fp8_direct_qmv_calls": stats.get(
                    "glm53_fp8_direct_qmv_calls", 0),
                "glm53_fp8_direct_qmv_positions": stats.get(
                    "glm53_fp8_direct_qmv_positions", 0),
                "glm53_fp8_direct_fallback_calls": stats.get(
                    "glm53_fp8_direct_fallback_calls", 0),
                "glm53_fp8_direct_fallback_positions": stats.get(
                    "glm53_fp8_direct_fallback_positions", 0),
                "glm53_fp8_direct_fallback_reconstruct_s": stats.get(
                    "glm53_fp8_direct_fallback_reconstruct_ns", 0) / 1e9,
                "glm53_fp8_direct_fallback_reconstruct_bytes": stats.get(
                    "glm53_fp8_direct_fallback_reconstruct_bytes", 0),
                "glm53_fp8_transform_s": stats.get(
                    "glm53_fp8_transform_ns", 0) / 1e9,
                "glm53_fp8_transform_calls": stats.get(
                    "glm53_fp8_transform_calls", 0),
                "glm53_fp8_native_calls": stats.get(
                    "glm53_fp8_native_calls", 0),
                "glm53_fp8_input_bytes": stats.get(
                    "glm53_fp8_input_bytes", 0),
                "glm53_fp8_resident_bytes": stats.get(
                    "glm53_fp8_resident_bytes", 0),
                "prefill_weight_store_bytes_read": stats.get(
                    "prefill_weight_store_bytes_read", 0),
                "decode_weight_store_bytes_read": stats.get(
                    "decode_weight_store_bytes_read", 0),
                "weight_fast_tier_bytes": stats.get(
                    "weight_fast_tier_bytes", 0),
                "weight_archive_bytes": stats.get(
                    "weight_archive_bytes", 0),
                "parallel_tier_fetches": stats.get(
                    "parallel_tier_fetches", 0),
                "parallel_tier_fast_bytes": stats.get(
                    "parallel_tier_fast_bytes", 0),
                "parallel_tier_archive_bytes": stats.get(
                    "parallel_tier_archive_bytes", 0),
                "parallel_tier_wall_s": stats.get(
                    "parallel_tier_wall_s", 0.0),
                "parallel_tier_fast_service_s": stats.get(
                    "parallel_tier_fast_service_s", 0.0),
                "parallel_tier_archive_service_s": stats.get(
                    "parallel_tier_archive_service_s", 0.0),
                "parallel_tier_hidden_s": stats.get(
                    "parallel_tier_hidden_s", 0.0),
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
                "glm53_layer_stationary_disk_spool": stats.get(
                    "glm53_layer_stationary_disk_spool", 0),
                "glm53_layer_stationary_disk_spool_logical_bytes": stats.get(
                    "glm53_layer_stationary_disk_spool_logical_bytes", 0),
                "glm53_layer_stationary_disk_spool_bytes_written": stats.get(
                    "glm53_layer_stationary_disk_spool_bytes_written", 0),
                "glm53_layer_stationary_disk_spool_bytes_read": stats.get(
                    "glm53_layer_stationary_disk_spool_bytes_read", 0),
                "glm53_layer_stationary_disk_spool_write_calls": stats.get(
                    "glm53_layer_stationary_disk_spool_write_calls", 0),
                "glm53_layer_stationary_disk_spool_read_calls": stats.get(
                    "glm53_layer_stationary_disk_spool_read_calls", 0),
                "glm53_layer_stationary_disk_spool_write_s": stats.get(
                    "glm53_layer_stationary_disk_spool_write_s", 0.0),
                "glm53_layer_stationary_disk_spool_read_s": stats.get(
                    "glm53_layer_stationary_disk_spool_read_s", 0.0),
                "glm53_layer_stationary_disk_spool_uncached_descriptors": (
                    stats.get(
                        "glm53_layer_stationary_disk_spool_uncached_descriptors",
                        0)),
                "direct_io_fd_cache_enabled": stats.get(
                    "direct_io_fd_cache_enabled", 0),
                "direct_io_fd_opens": stats.get("direct_io_fd_opens", 0),
                "direct_io_fd_hits": stats.get("direct_io_fd_hits", 0),
                "direct_io_fd_closes": stats.get("direct_io_fd_closes", 0),
                "direct_io_fd_open_s": stats.get(
                    "direct_io_fd_open_ns", 0) / 1e9,
                "direct_io_fd_cached": stats.get("direct_io_fd_cached", 0),
                "direct_io_pread_calls": stats.get(
                    "direct_io_pread_calls", 0),
                "direct_io_pread_requested_bytes": stats.get(
                    "direct_io_pread_requested_bytes", 0),
                "direct_io_pread_bytes": stats.get(
                    "direct_io_pread_bytes", 0),
                "direct_io_pread_s": stats.get(
                    "direct_io_pread_ns", 0) / 1e9,
                "direct_io_pread_short_reads": stats.get(
                    "direct_io_pread_short_reads", 0),
                "direct_io_nocache_enabled": stats.get(
                    "direct_io_nocache_enabled", 0),
                "direct_io_fd_nocache_applied": stats.get(
                    "direct_io_fd_nocache_applied", 0),
                "execution_profile": result.get("execution_profile"),
            })
        endpoint_state = None
        if args.state_digest:
            from tests.fixtures.qwen38_dflash2_gate import _state_digest
            target = getattr(engine, "target", engine)
            require_recurrent = target.cfg.model_type == "glm5_next"
            if args.plain:
                hidden = getattr(
                    target, "_diagnostic_generation_hidden", None)
                if hidden is None:
                    raise RuntimeError(
                        "plain decoder did not retain its committed hidden row")
                endpoint_state = _state_digest(
                    target, hidden=hidden,
                    require_recurrent=require_recurrent,
                    dsa_row_digest=args.dsa_row_digest)
            else:
                endpoint = engine.decoder._diagnostic_generation_endpoint
                hidden = engine.decoder._diagnostic_generation_hidden
                if endpoint is None or hidden is None:
                    raise RuntimeError(
                        "native MTP verifier did not retain its committed endpoint")
                endpoint_state = _state_digest(
                    target, kv=endpoint, hidden=hidden,
                    require_recurrent=require_recurrent,
                    dsa_row_digest=args.dsa_row_digest)
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
            "swap_used_growth_bytes": runs[-1][
                "swap_used_growth_bytes"],
            "swap_out_growth_bytes": runs[-1][
                "swap_out_growth_bytes"],
            "governor_swap_pressure_events": runs[-1][
                "governor_swap_pressure_events"],
            "governor_swap_used_growth_bytes": runs[-1][
                "governor_swap_used_growth_bytes"],
            "governor_swap_out_growth_bytes": runs[-1][
                "governor_swap_out_growth_bytes"],
            "governor_reservation_failures": runs[-1][
                "governor_reservation_failures"],
            "mtp_enabled": stats.get("glm53_mtp_enabled", 0),
            "mtp_used": stats.get("glm53_mtp_used", 0),
            "mtp_depth": stats.get("glm53_mtp_depth", 0),
            "proposed": stats.get("speculative_proposed", 0),
            "accepted": stats.get("speculative_accepted", 0),
            "acceptance_rate": stats.get(
                "speculative_acceptance_rate", 0.0),
            "target_sweeps": stats.get("speculative_target_sweeps", 0),
            "controller_disabled_rounds": stats.get(
                "speculative_controller_disabled_rounds", 0),
            "controller_probe_every": stats.get(
                "speculative_controller_probe_every", 0),
            "controller_cost_estimate": stats.get(
                "speculative_controller_cost_estimate", 0.0),
            "controller_acceptance_estimate": stats.get(
                "speculative_controller_acceptance_estimate", 0.0),
            "rounds": stats.get("speculative_rounds", []),
            "draft_s": stats.get("speculative_draft_s", 0.0),
            "verify_s": stats.get("speculative_verify_decode_s", 0.0),
            "draft_bytes": stats.get("speculative_draft_bytes", 0),
            "weight_store_bytes_read": stats.get(
                "weight_store_bytes_read", 0),
            "glm53_native_fp8_dequant": runs[-1][
                "glm53_native_fp8_dequant"],
            "glm53_fp8_direct_qmv": runs[-1]["glm53_fp8_direct_qmv"],
            "glm53_fp8_direct_pages": runs[-1][
                "glm53_fp8_direct_pages"],
            "glm53_fp8_direct_resident_bytes": runs[-1][
                "glm53_fp8_direct_resident_bytes"],
            "glm53_fp8_direct_qmv_calls": runs[-1][
                "glm53_fp8_direct_qmv_calls"],
            "glm53_fp8_direct_qmv_positions": runs[-1][
                "glm53_fp8_direct_qmv_positions"],
            "glm53_fp8_direct_fallback_calls": runs[-1][
                "glm53_fp8_direct_fallback_calls"],
            "glm53_fp8_direct_fallback_positions": runs[-1][
                "glm53_fp8_direct_fallback_positions"],
            "glm53_fp8_direct_fallback_reconstruct_s": runs[-1][
                "glm53_fp8_direct_fallback_reconstruct_s"],
            "glm53_fp8_direct_fallback_reconstruct_bytes": runs[-1][
                "glm53_fp8_direct_fallback_reconstruct_bytes"],
            "glm53_fp8_transform_s": runs[-1]["glm53_fp8_transform_s"],
            "glm53_fp8_transform_calls": runs[-1][
                "glm53_fp8_transform_calls"],
            "glm53_fp8_native_calls": runs[-1]["glm53_fp8_native_calls"],
            "glm53_fp8_input_bytes": runs[-1]["glm53_fp8_input_bytes"],
            "glm53_fp8_resident_bytes": runs[-1][
                "glm53_fp8_resident_bytes"],
            "state": endpoint_state,
        }
        rendered = json.dumps(document, indent=2, sort_keys=True)
        print(rendered)
        if args.result is not None:
            if args.result.exists():
                raise SystemExit(f"result already exists: {args.result}")
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(rendered + "\n")
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
