"""Server adapter checks that do not import MLX or start a model process."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.server import (Handler, INFER_LOCK, PreparedPrompt, PriorityLock, RequestValidationError,
                            _TokenOffsetIndex,
                            _active_context_limit,
                            _advertised_model_ids,
                            _cache_phase_telemetry,
                            _execution_profile_fields,
                            _fast_dense_resident_kv_projection,
                            _HarmonyChannelGate,
                            _harmony_split_channels,
                            _harmony_visible_text,
                            _hidden_gateway_catalogs,
                            _hidden_gateway_abstention_policy,
                            _hidden_gateway_execution_abstention_policy,
                            _hidden_gateway_activation_clear,
                            _hidden_gateway_activation_get,
                            _hidden_gateway_activation_put,
                            _hidden_gateway_conversation_key,
                            _hidden_gateway_execution_context,
                            _hidden_gateway_execution_context_policy,
                            _hidden_gateway_execution_messages,
                            _hidden_gateway_result_suffix_anchor,
                            _HIDDEN_GATEWAY_TERMINAL_PAGINATION_POLICY,
                            _hidden_gateway_terminal_context,
                            _hidden_gateway_terminal_pagination_synthesis,
                            _hidden_gateway_deterministic_policy_render,
                            _hidden_gateway_host_action,
                            _hidden_gateway_initial_pagination_defaults,
                            _hidden_gateway_pagination_call,
                            _HiddenDecisionStream,
                            _hidden_gateway_decision_choice,
                            _hidden_gateway_force_reason,
                            _hidden_gateway_semantic_query,
                            _hidden_gateway_search_result_limit,
                            _hidden_tool_gateway_enabled,
                            _hidden_tool_abstain_pair,
                            _hidden_tool_enable_pair,
                            _hidden_tool_search_pair,
                            _hidden_gateway_virtual_pairs,
                            _load_vision_images,
                            _MarkerHoldback,
                            _chat_prompt,
                            _omitted_output_token_limit,
                            _positive_token_limit,
                            _prepare_chat_prompt, _render_template,
                            _compiled_template,
                            _openai_finish_reason, _responses_output_items, _safe_emit_len,
                            _parse_request_tool_calls, _tool_request_controls,
                            _is_voom_lossy_checkpoint,
                            _preferred_fast_artifact,
                            _dspark_draft_for,
                            _engine_generate,
                            _grammar_jump_forward_policy,
                            _has_own_method,
                            _qwen_compiled_delta_policy,
                            _qwen_chunked_delta_policy,
                            _qwen_delta_prefill_policies,
                            _qwen_lossy_suffix_prefill_policy,
                            _speculative_draft_for,
                            _request_reasoning_controls, _request_sampling,
                            _registry,
                            _tool_capsule_spans,
                            _vision_protocol_timing,
                            _vision_request_error,
                            _validate_fast_dense_resident_kv,
                            _validate_context_budget, _validate_generation_controls,
                            split_model_mode)


class _CharTokenizer:
    def encode(self, text):
        return SimpleNamespace(
            ids=list(text), offsets=[(index, index + 1)
                                     for index in range(len(text))])


class _CountingCharTokenizer(_CharTokenizer):
    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        return super().encode(text)


def test_vision_dispatch_distinguishes_tower_presence_from_backend_support():
    text = SimpleNamespace(model_type="qwen2", vision_config=None,
                           vision_backend="")
    kimi = SimpleNamespace(
        model_type="kimi_k3",
        vision_config={"patch_size": 14, "merge_kernel_size": [2, 2]},
        vision_backend="kimi_k3",
    )
    qwen = SimpleNamespace(
        model_type="qwen3_vl",
        vision_config={"patch_size": 16, "spatial_merge_size": 2},
        vision_backend="qwen3vl",
    )

    assert "has no vision tower" in _vision_request_error(text, "text-only")
    error = _vision_request_error(kimi, "Kimi-K3")
    assert "kimi_k3 vision tower" in error
    assert "not yet implement" in error
    assert _vision_request_error(qwen, "Qwen3-VL") is None


def test_route_ignores_query_string():
    handler = object.__new__(Handler)
    handler.path = "/v1/responses?trace=true"
    assert handler._route() == "/responses"


def test_vision_protocol_timing_uses_generic_path_stats():
    result = {
        "vision_cache_hits": 99,
        "vision_tool_pic_reused_tokens": 98,
        "path_stats": {
            "vision_cache_hits": 2,
            "vision_cache_misses": 1,
            "vision_prompt_cache_tower_skipped": 1,
            "prompt_cache_prefix_tokens": 144,
            "prompt_cache_exact_hit": 1,
            "vision_prompt_cache_stored": 1,
            "prompt_cache_source": "vision_tool_pic",
            "tool_pic": 1,
            "tool_pic_reused_tokens": 80,
            "tool_pic_selected_tokens": 64,
            "tool_pic_repaired_tokens": 4,
            "tool_pic_memory_admitted": 1,
            "tool_pic_projected_bytes": 123_456,
            "prompt_state_approximate": 1,
        },
    }

    timing = _vision_protocol_timing(result)

    assert timing == {
        "true_peak_metal_bytes": 0,
        "kv_bytes": 0,
        "kv_positions": 0,
        "execution_backend": "voom",
        "execution_path": "",
        "prefill_step_size": 0,
        "request_incremental_projection_bytes": 0,
        "request_system_available_bytes": 0,
        "request_system_required_bytes": 0,
        "prompt_kv_projected_bytes": 0,
        "prompt_kv_projection": "",
        "vision_cache_hits": 2,
        "vision_cache_misses": 1,
        "vision_prompt_cache_tower_skipped": 1,
        "vision_prompt_cache_prefix_tokens": 144,
        "vision_prompt_cache_exact_hit": 1,
        "vision_prompt_cache_stored": 1,
        "cache_source": "vision_tool_pic",
        "tool_pic": 1,
        "tool_pic_reused_tokens": 80,
        "tool_pic_selected_tokens": 64,
        "tool_pic_repaired_tokens": 4,
        "tool_pic_memory_admitted": 1,
        "tool_pic_projected_bytes": 123_456,
        "tool_pic_system_available_bytes": 0,
        "tool_pic_system_floor_bytes": 0,
        "tool_pic_system_memory_admitted": 0,
        "prompt_state_approximate": 1,
    }


def test_vision_protocol_timing_exposes_qwen_mtp_round_trace():
    replay = [{"depth": 1, "draft_token_ids": [7, 9]}]
    timing = _vision_protocol_timing({
        "path_stats": {
            "qwen_mtp_used": 1,
            "qwen_mtp_round_outcomes": "AARRA",
            "qwen_mtp_depth": 2,
            "qwen_mtp_max_verify_width_observed": 5,
            "qwen_mtp_proposal_weight_representation": "mxfp4-q4-g32",
            "qwen_mtp_proposal_page_round_loads": 3,
            "qwen_mtp_proposal_page_round_releases": 3,
            "qwen_mtp_proposal_page_read_bytes": 5678,
            "qwen_mtp_proposal_page_load_s": 0.125,
            "qwen_mtp_proposal_page_release_s": 0.025,
            "qwen_mtp_bf16_sidecar_round_loads": 3,
            "qwen_mtp_bf16_sidecar_round_releases": 3,
            "qwen_mtp_bf16_sidecar_read_bytes": 1234,
            "qwen_mtp_target_sweeps": 5,
            "qwen_mtp_plain_equivalent_target_sweeps": 8,
            "qwen_mtp_target_sweeps_avoided": 3,
            "qwen_mtp_target_tokens_per_sweep": 1.6,
            "qwen_mtp_verifier_input_positions": 10,
            "qwen_mtp_verifier_committed_positions": 8,
            "qwen_mtp_verifier_rolled_back_positions": 2,
            "qwen_mtp_verifier_output_tokens": 8,
            "qwen_mtp_verifier_tokens_per_sweep": 1.6,
            "qwen_mtp_verifier_accepted_draft_tokens": 4,
            "qwen_mtp_verifier_correction_tokens": 1,
            "qwen_mtp_verifier_bonus_tokens": 3,
            "qwen_mtp_draft_round_s": 0.75,
            "qwen_mtp_verifier_round_s": 2.5,
            "qwen_mtp_ngram_first_enabled": 1,
            "qwen_mtp_ngram_first_eligible": 1,
            "qwen_mtp_ngram_first_max_draft_tokens": 4,
            "qwen_mtp_ngram_first_max_proposed_per_round": 4,
            "qwen_mtp_ngram_first_attempts": 5,
            "qwen_mtp_ngram_first_matches": 2,
            "qwen_mtp_ngram_first_proposed": 2,
            "qwen_mtp_ngram_first_accepted": 1,
            "qwen_mtp_ngram_first_rejected": 1,
            "qwen_mtp_ngram_first_native_draft_bypasses": 2,
            "qwen_mtp_native_draft_proposed": 3,
            "qwen_mtp_native_draft_accepted": 2,
            "qwen_mtp_native_draft_rejected": 1,
            "qwen_mtp_proposal_sources": "NMNMN",
            "qwen_mtp_accepted_by_step": [2, 1],
            "qwen_mtp_ngram_first_accepted_by_step": [1, 0, 0, 0],
            "qwen_mtp_ngram_first_verified_by_step": [1, 1, 0, 0],
            "qwen_mtp_q_policy": {"kind": "flat", "top_k": 4},
            "qwen_mtp_proposal_q_replay": replay,
        },
    })

    assert timing["qwen_mtp_used"] == 1
    assert timing["qwen_mtp_round_outcomes"] == "AARRA"
    assert timing["qwen_mtp_depth"] == 2
    assert timing["qwen_mtp_max_verify_width_observed"] == 5
    assert timing["qwen_mtp_proposal_weight_representation"] == "mxfp4-q4-g32"
    assert timing["qwen_mtp_proposal_page_round_loads"] == 3
    assert timing["qwen_mtp_proposal_page_round_releases"] == 3
    assert timing["qwen_mtp_proposal_page_read_bytes"] == 5678
    assert timing["qwen_mtp_proposal_page_load_s"] == 0.125
    assert timing["qwen_mtp_proposal_page_release_s"] == 0.025
    assert timing["qwen_mtp_bf16_sidecar_round_loads"] == 3
    assert timing["qwen_mtp_bf16_sidecar_round_releases"] == 3
    assert timing["qwen_mtp_bf16_sidecar_read_bytes"] == 1234
    assert timing["qwen_mtp_target_sweeps"] == 5
    assert timing["qwen_mtp_plain_equivalent_target_sweeps"] == 8
    assert timing["qwen_mtp_target_sweeps_avoided"] == 3
    assert timing["qwen_mtp_target_tokens_per_sweep"] == 1.6
    assert timing["qwen_mtp_verifier_input_positions"] == 10
    assert timing["qwen_mtp_verifier_committed_positions"] == 8
    assert timing["qwen_mtp_verifier_rolled_back_positions"] == 2
    assert timing["qwen_mtp_verifier_output_tokens"] == 8
    assert timing["qwen_mtp_verifier_tokens_per_sweep"] == 1.6
    assert timing["qwen_mtp_verifier_accepted_draft_tokens"] == 4
    assert timing["qwen_mtp_verifier_correction_tokens"] == 1
    assert timing["qwen_mtp_verifier_bonus_tokens"] == 3
    assert timing["qwen_mtp_draft_round_s"] == 0.75
    assert timing["qwen_mtp_verifier_round_s"] == 2.5
    assert timing["qwen_mtp_ngram_first_enabled"] == 1
    assert timing["qwen_mtp_ngram_first_eligible"] == 1
    assert timing["qwen_mtp_ngram_first_max_draft_tokens"] == 4
    assert timing["qwen_mtp_ngram_first_max_proposed_per_round"] == 4
    assert timing["qwen_mtp_ngram_first_attempts"] == 5
    assert timing["qwen_mtp_ngram_first_matches"] == 2
    assert timing["qwen_mtp_ngram_first_proposed"] == 2
    assert timing["qwen_mtp_ngram_first_accepted"] == 1
    assert timing["qwen_mtp_ngram_first_rejected"] == 1
    assert timing["qwen_mtp_ngram_first_native_draft_bypasses"] == 2
    assert timing["qwen_mtp_native_draft_proposed"] == 3
    assert timing["qwen_mtp_native_draft_accepted"] == 2
    assert timing["qwen_mtp_native_draft_rejected"] == 1
    assert timing["qwen_mtp_proposal_sources"] == "NMNMN"
    assert timing["qwen_mtp_accepted_by_step"] == [2, 1]
    assert timing["qwen_mtp_ngram_first_accepted_by_step"] == [1, 0, 0, 0]
    assert timing["qwen_mtp_ngram_first_verified_by_step"] == [1, 1, 0, 0]
    assert timing["qwen_mtp_q_policy"] == {"kind": "flat", "top_k": 4}
    assert timing["qwen_mtp_proposal_q_replay"] == replay


def test_protocol_timing_exposes_dspark_round_and_context_suspension():
    timing = _vision_protocol_timing({
        "path_stats": {
            "speculative_round_proposed": [3, 3, 1],
            "speculative_round_accepted": [1, 0, 1],
            "speculative_round_draft_s": [0.5, 0.4, 0.2],
            "speculative_round_verify_s": [7.0, 7.1, 6.8],
            "speculative_round_context_s": [0.2, 0.0, 0.1],
            "dspark_draft_context_suspend_rounds": 3,
            "dspark_draft_context_restore_rounds": 2,
            "dspark_draft_context_suspend_s": 0.125,
            "dspark_draft_context_restore_s": 0.25,
            "dspark_draft_context_suspended_bytes": 4096,
            "dspark_draft_context_released_active_bytes": 2048,
        },
    })

    assert timing["speculative_round_proposed"] == [3, 3, 1]
    assert timing["speculative_round_accepted"] == [1, 0, 1]
    assert timing["speculative_round_draft_s"] == [0.5, 0.4, 0.2]
    assert timing["speculative_round_verify_s"] == [7.0, 7.1, 6.8]
    assert timing["speculative_round_context_s"] == [0.2, 0.0, 0.1]
    assert timing["dspark_draft_context_suspend_rounds"] == 3
    assert timing["dspark_draft_context_restore_rounds"] == 2
    assert timing["dspark_draft_context_suspend_s"] == 0.125
    assert timing["dspark_draft_context_restore_s"] == 0.25
    assert timing["dspark_draft_context_suspended_bytes"] == 4096
    assert timing["dspark_draft_context_released_active_bytes"] == 2048


def test_protocol_timing_exposes_qwen35_prefill_ceiling_and_selection():
    timing = _vision_protocol_timing({
        "path_stats": {
            "prefill_step_size": 128,
            "qwen35_prefill_chunk_ceiling": 128,
            "qwen35_prefill_chunk_selected": 128,
        },
    })

    assert timing["prefill_step_size"] == 128
    assert timing["qwen35_prefill_chunk_ceiling"] == 128
    assert timing["qwen35_prefill_chunk_selected"] == 128


def test_protocol_timing_exposes_qwen_delta_arithmetic_mode():
    timing = _vision_protocol_timing({
        "path_stats": {
            "qwen_compiled_delta_prefill": 1,
            "qwen_native_fused_delta_prefill": 0,
            "qwen_chunked_delta_prefill": 0,
            "qwen_lossy_suffix_prefill_enabled": 1,
            "qwen_lossy_suffix_prefill_used": 1,
            "qwen_lossy_suffix_prefill_early_layers": 8,
            "qwen_lossy_suffix_prefill_prefix_tokens": 0,
            "qwen_lossy_suffix_prefill_tokens": 256,
            "hot_prompt_kv_disk_hit": 1,
            "hot_prompt_hybrid_prefix_snapshot_tokens": 6332,
            "hot_prompt_admission_positions": 7,
            "disk_prompt_lookup_s": 0.42,
            "hot_prompt_kv_persist_write_s": 1.25,
        },
    })

    assert timing["qwen_compiled_delta_prefill"] == 1
    assert timing["qwen_native_fused_delta_prefill"] == 0
    assert timing["qwen_chunked_delta_prefill"] == 0
    assert timing["qwen_lossy_suffix_prefill_enabled"] == 1
    assert timing["qwen_lossy_suffix_prefill_used"] == 1
    assert timing["qwen_lossy_suffix_prefill_early_layers"] == 8
    assert timing["qwen_lossy_suffix_prefill_prefix_tokens"] == 0
    assert timing["qwen_lossy_suffix_prefill_tokens"] == 256
    assert timing["hot_prompt_kv_disk_hit"] == 1
    assert timing["hot_prompt_hybrid_prefix_snapshot_tokens"] == 6332
    assert timing["hot_prompt_admission_positions"] == 7
    assert timing["disk_prompt_lookup_s"] == 0.42
    assert timing["hot_prompt_kv_persist_write_s"] == 1.25


def test_protocol_timing_exposes_pin_and_prefetch_measurements():
    timing = _vision_protocol_timing({
        "path_stats": {
            "weight_cache_pinned_hits": 11,
            "weight_cache_prefetch_hits": 7,
            "weight_prefetch_waits": 3,
            "weight_prefetch_wait_ns": 125_000_000,
            "weight_prefetch_wait_s": 0.125,
            "weight_prefetch_loads": 9,
            "weight_prefetch_useful_pages": 7,
            "weight_prefetch_wasted_pages": 2,
            "weight_prefetch_hidden_lower_bound_s": 0.75,
            "parallel_tier_fetches": 64,
            "parallel_tier_fast_bytes": 40_000_000,
            "parallel_tier_archive_bytes": 60_000_000,
            "parallel_tier_hidden_s": 1.5,
            "weight_cache_pinned_bytes": 400_000_000,
            "weight_cache_prefetched_bytes": 200_000_000,
            "planned_trunk_pin_layers": 4,
            "planned_trunk_pin_bytes": 350_000_000,
        },
    })

    assert timing["weight_cache_pinned_hits"] == 11
    assert timing["weight_cache_prefetch_hits"] == 7
    assert timing["weight_prefetch_waits"] == 3
    assert timing["weight_prefetch_wait_ns"] == 125_000_000
    assert timing["weight_prefetch_wait_s"] == 0.125
    assert timing["weight_prefetch_loads"] == 9
    assert timing["weight_prefetch_useful_pages"] == 7
    assert timing["weight_prefetch_wasted_pages"] == 2
    assert timing["weight_prefetch_hidden_lower_bound_s"] == 0.75
    assert timing["parallel_tier_fetches"] == 64
    assert timing["parallel_tier_fast_bytes"] == 40_000_000
    assert timing["parallel_tier_archive_bytes"] == 60_000_000
    assert timing["parallel_tier_hidden_s"] == 1.5
    assert timing["weight_cache_pinned_bytes"] == 400_000_000
    assert timing["weight_cache_prefetched_bytes"] == 200_000_000
    assert timing["planned_trunk_pin_layers"] == 4
    assert timing["planned_trunk_pin_bytes"] == 350_000_000


def test_protocol_timing_exposes_governor_admission_measurements():
    timing = _vision_protocol_timing({
        "path_stats": {
            "weight_cache_resident_bytes": 675_000_000,
            "weight_cache_budget_bytes": 860_000_000,
            "governor_reservations": 12,
            "governor_reservation_calls": 300,
            "governor_reservation_fast_path_calls": 280,
            "governor_reservation_clear_cache_only_calls": 3,
            "governor_serial_verify_page_reservation_calls": 64,
            "governor_serial_verify_transient_reservation_calls": 64,
            "governor_qwen_prefill_page_reservation_calls": 64,
            "governor_qwen_prefill_transient_reservation_calls": 128,
            "governor_reservation_requested_bytes": 1_000_000_000,
            "governor_reservation_budget_reduced_bytes": 400_000_000,
            "governor_reservation_budget_restored_bytes": 200_000_000,
            "governor_reservation_cache_released_bytes": 200_000_000,
            "governor_reservation_unproductive_shrinks": 5,
            "governor_reservation_failures": 0,
        },
    })

    assert timing["weight_cache_resident_bytes"] == 675_000_000
    assert timing["weight_cache_budget_bytes"] == 860_000_000
    assert timing["governor_reservations"] == 12
    assert timing["governor_reservation_calls"] == 300
    assert timing["governor_serial_verify_page_reservation_calls"] == 64
    assert timing["governor_qwen_prefill_page_reservation_calls"] == 64
    assert timing["governor_reservation_budget_reduced_bytes"] == 400_000_000
    assert timing["governor_reservation_budget_restored_bytes"] == 200_000_000
    assert timing["governor_reservation_cache_released_bytes"] == 200_000_000
    assert timing["governor_reservation_unproductive_shrinks"] == 5


def test_protocol_timing_exposes_row_paged_head_recall_measurements():
    timing = _vision_protocol_timing({
        "path_stats": {
            # Depth-1 short8 ARRRA: bootstrap target=1, five verifier
            # windows=10 target rows, and five shared-head draft rows.
            "reranked_lm_head_calls": 16,
            "reranked_lm_head_positions": 16,
            "reranked_lm_head_candidate_winner_changes": 2,
            "reranked_lm_head_candidate_recall_probes": 16,
            "reranked_lm_head_candidate_recall_hits": 16,
            "reranked_lm_head_candidate_recall": 1.0,
            "reranked_lm_head_candidate_read_calls": 16,
            "reranked_lm_head_candidate_bytes_read": 10_485_760,
            "reranked_lm_head_candidate_recall_full_scan_calls": 16,
            "reranked_lm_head_candidate_recall_full_scan_bytes": 40_684_748_800,
            "qwen_mtp_target_reranked_lm_head_calls": 11,
            "qwen_mtp_target_reranked_lm_head_positions": 11,
            "qwen_mtp_target_reranked_lm_head_candidate_recall_probes": 11,
            "qwen_mtp_target_reranked_lm_head_candidate_recall_hits": 11,
            "qwen_mtp_target_reranked_lm_head_candidate_recall": 1.0,
            "qwen_mtp_draft_reranked_lm_head_calls": 5,
            "qwen_mtp_draft_reranked_lm_head_positions": 5,
            "qwen_mtp_draft_reranked_lm_head_candidate_recall_probes": 5,
            "qwen_mtp_draft_reranked_lm_head_candidate_recall_hits": 5,
            "qwen_mtp_draft_reranked_lm_head_candidate_recall": 1.0,
        },
    })

    assert timing["reranked_lm_head_calls"] == 16
    assert timing["reranked_lm_head_candidate_winner_changes"] == 2
    assert timing["reranked_lm_head_candidate_recall_probes"] == 16
    assert timing["reranked_lm_head_candidate_recall_hits"] == 16
    assert timing["reranked_lm_head_candidate_recall"] == 1.0
    assert timing["reranked_lm_head_candidate_bytes_read"] == 10_485_760
    assert timing[
        "reranked_lm_head_candidate_recall_full_scan_bytes"] == 40_684_748_800
    assert timing["qwen_mtp_target_reranked_lm_head_calls"] == 11
    assert timing["qwen_mtp_target_reranked_lm_head_positions"] == 11
    assert timing[
        "qwen_mtp_target_reranked_lm_head_candidate_recall"] == 1.0
    assert timing["qwen_mtp_draft_reranked_lm_head_calls"] == 5
    assert timing["qwen_mtp_draft_reranked_lm_head_positions"] == 5
    assert timing[
        "qwen_mtp_draft_reranked_lm_head_candidate_recall"] == 1.0


def test_response_write_timeout_releases_inference_lock(monkeypatch):
    class Connection:
        def __init__(self):
            self.timeout = None
            self.values = []

        def gettimeout(self):
            return self.timeout

        def settimeout(self, value):
            self.timeout = value
            self.values.append(value)

    handler = object.__new__(Handler)
    handler.connection = Connection()
    handler._read_json_request = lambda: (b"{}", {}, 2)
    handler._preflight_nested_request = lambda req: ([], [], [])

    def timeout():
        raise TimeoutError("client stopped reading")

    handler._do_post_locked = timeout
    monkeypatch.setenv("VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS", "0.05")
    handler.do_POST()

    assert handler.close_connection
    assert handler.connection.values == [0.05, None]
    assert INFER_LOCK.acquire(blocking=False)
    INFER_LOCK.release()


def _fake_engine(*, model_limit=1_000_000, context_bound=0, model_type="qwen2"):
    return SimpleNamespace(
        tokenizer=_CharTokenizer(),
        cfg=SimpleNamespace(
            model_type=model_type, max_position_embeddings=model_limit),
        rc=SimpleNamespace(context_bound=context_bound),
        effective_max_position_embeddings=model_limit,
        rope_profile="released",
    )


def test_compact_json_is_sorted_and_minified():
    template = "{{ value | tojson }}"
    rendered = _render_template(
        template, compact_json=True, value={"z": 1, "a": {"y": 2, "b": 3}})
    assert rendered == '{"a":{"b":3,"y":2},"z":1}'


def test_compact_json_preserves_jinja_htmlsafe_escaping():
    rendered = _render_template(
        "{{ value | tojson }}", compact_json=True,
        value={"text": "</tools> & it's safe"})
    assert r"\u003c/tools\u003e" in rendered
    assert r"\u0026" in rendered
    assert r"\u0027" in rendered
    assert "</tools>" not in rendered


def test_compact_json_accepts_standard_indent_argument_but_stays_canonical():
    rendered = _render_template(
        "{{ value | tojson(indent=2) }}", compact_json=True,
        value={"z": 1, "a": 2})
    assert rendered == '{"a":2,"z":1}'


def test_raise_exception_global_surfaces_as_request_validation_error():
    """Real chat templates (Qwen's included) call raise_exception(msg) to
    reject malformed conversations. Without a registered global, Jinja
    raised its own opaque UndefinedError instead of the template's actual
    message, which crashed as an unhandled 500 (2026-07-20,
    live-confirmed against a real Codex/Kai request with two leading
    system messages). It must surface as a clean, catchable
    RequestValidationError with the template's own message."""
    import pytest

    template = "{{ raise_exception('boom') }}"
    with pytest.raises(RequestValidationError, match="boom"):
        _render_template(template)
    with pytest.raises(RequestValidationError, match="boom"):
        _render_template(template, compact_json=True)


def test_template_compilation_is_cached_by_text_and_render_profile():
    _compiled_template.cache_clear()
    template = "{{ value | tojson }}"
    _render_template(template, value={"x": 1})
    _render_template(template, value={"x": 2})
    after_released = _compiled_template.cache_info()
    assert (after_released.hits, after_released.misses) == (1, 1)

    _render_template(template, compact_json=True, value={"x": 3})
    after_compact = _compiled_template.cache_info()
    assert (after_compact.hits, after_compact.misses) == (1, 2)


def test_exact_long_rendered_prompt_token_ids_are_engine_local_lru(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0].content }}")
    engine = _fake_engine()
    engine.tokenizer = _CountingCharTokenizer()
    messages = [{"role": "user", "content": "x" * 1500}]
    args = (engine, tmp_path, messages, "low", [], [], "lossless", 1)

    first = _prepare_chat_prompt(*args)
    second = _prepare_chat_prompt(*args)

    assert engine.tokenizer.calls == 1
    assert first[0].token_ids == second[0].token_ids
    assert first[4]["prompt_token_cache_hit"] == 0
    assert second[4]["prompt_token_cache_hit"] == 1


def test_native_template_history_renders_tool_arguments_as_object_not_string():
    template = (
        "{% for message in messages %}{% if message.tool_calls %}"
        "{{ message.tool_calls[0].function.arguments | tojson }}"
        "{% endif %}{% endfor %}"
    )
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_weather", "type": "function", "function": {
            "name": "weather", "arguments": '{"city":"Chicago"}'}}]}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": template}))
        released = _chat_prompt(_fake_engine(), model_dir, messages, "low")
        compact = _chat_prompt(
            _fake_engine(), model_dir, messages, "low", compact_json=True)
    assert released == '{"city": "Chicago"}'
    assert compact == '{"city":"Chicago"}'


def test_native_harmony_history_normalizes_null_assistant_content():
    # GPT-OSS's released template checks ``"<|channel|>" in
    # message.content`` whenever the content key exists. Responses function
    # calls legitimately carry content=None, which otherwise raises TypeError
    # before the template reaches its tool-call branch.
    template = (
        "{% for message in messages %}{% if message.role == 'assistant' %}"
        "{% if 'forbidden' in message.content %}BAD{% endif %}"
        "{{ message.tool_calls[0].function.name }}"
        "{% endif %}{% endfor %}"
    )
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_weather", "type": "function", "function": {
            "name": "weather", "arguments": "{}"}}]}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": template}))
        rendered = _chat_prompt(
            _fake_engine(model_type="gpt_oss"), model_dir, messages, "low")
    assert rendered == "weather"


def test_fast_qwen35_tool_history_uses_one_canonical_hermes_prefix(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}"
        "{{ message.role }}={{ message.content or '' }}|"
        "{% for call in message.tool_calls or [] %}NATIVE={{ call.function.name }}{% endfor %}"
        "{% endfor %}{% if add_generation_prompt %}assistant={% endif %}")
    engine = _fake_engine(model_type="qwen3_5_moe")
    tool = _named_tool("get_weather")
    first_messages = [{"role": "user", "content": "Weather?"}]
    call = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_weather", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"value":"Chicago"}'}}]}
    result = {"role": "tool", "tool_call_id": "call_weather",
              "content": '{"temperature":72}'}

    first, *_ = _prepare_chat_prompt(
        engine, tmp_path, first_messages, "low", [tool], [tool], "fast", 8)
    followup, *_ = _prepare_chat_prompt(
        engine, tmp_path, [*first_messages, call, result], "low",
        [tool], [tool], "fast", 8)
    canonical_call = (
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"value": "Chicago"}}</tool_call>')

    assert "NATIVE=" not in followup
    assert followup.startswith(str(first) + canonical_call)


def test_native_template_bos_token_concatenation_does_not_raise():
    # Groq's real Llama-3-Groq-8B-Tool-Use chat_template.jinja does exactly
    # this: `{% set content = bos_token + content %}` for the first message
    # only. bos_token was never supplied anywhere in this codebase's
    # template-rendering context, so this previously raised
    # `UndefinedError: 'bos_token' is undefined` (live-confirmed 2026-07-28)
    # -- Jinja's default Undefined tolerates a bare `{{ bos_token }}` output
    # (renders empty) but not a `+` concatenation.
    template = (
        "{% for message in messages %}"
        "{% set content = message.role + ':' + message.content %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
    )
    messages = [{"role": "user", "content": "hi"}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"bos_token": "<|begin_of_text|>"}))
        rendered = _chat_prompt(_fake_engine(), model_dir, messages, "low")
    assert rendered == "<|begin_of_text|>user:hi"


def test_native_template_tojson_ensure_ascii_kwarg_does_not_raise():
    # ai9stars/G9v3-3B's real chat_template.jinja does exactly this:
    # `{{ tool | tojson(ensure_ascii=False) }}` when rendering its tool
    # definitions. Jinja2's own built-in `tojson` filter only accepts
    # `indent` -- passing `ensure_ascii` previously raised
    # `TypeError: do_tojson() got an unexpected keyword argument
    # 'ensure_ascii'` (live-confirmed 2026-07-28) on the ordinary (lossless,
    # non-compact_json) rendering path, which had no custom `tojson`
    # override at all.
    from runtime.server import _render_template

    template = "{{ value | tojson(ensure_ascii=False) }}"
    rendered = _render_template(template, value={"city": "北京"})
    assert rendered == '{"city": "北京"}'


def test_native_template_tojson_default_ensure_ascii_matches_jinja_builtin():
    from runtime.server import _render_template

    template = "{{ value | tojson }}"
    rendered = _render_template(template, value={"city": "北京"})
    assert rendered == '{"city": "\\u5317\\u4eac"}'


def test_bos_token_accepts_added_token_dict_shape():
    template = "{{ bos_token }}{{ messages[0].content }}"
    messages = [{"role": "user", "content": "hi"}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        (model_dir / "tokenizer_config.json").write_text(json.dumps({
            "bos_token": {"content": "<s>", "lstrip": False}}))
        rendered = _chat_prompt(_fake_engine(), model_dir, messages, "low")
    assert rendered == "<s>hi"


def test_bos_token_falls_back_to_special_tokens_map():
    template = "{{ bos_token }}{{ messages[0].content }}"
    messages = [{"role": "user", "content": "hi"}]
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        (model_dir / "tokenizer_config.json").write_text(json.dumps({}))
        (model_dir / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<s>"}))
        rendered = _chat_prompt(_fake_engine(), model_dir, messages, "low")
    assert rendered == "<s>hi"


def test_standalone_jinja_template_receives_reasoning_and_thinking_controls():
    template = (
        "{{ reasoning_effort }}|"
        "{{ enable_thinking if enable_thinking is defined else 'unset' }}|"
        "{{ messages[0].content }}"
    )
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        engine = _fake_engine(model_type="glm_moe_dsa")
        released = _chat_prompt(
            engine, model_dir, [{"role": "user", "content": "Hello"}], "high")
        fastest = _chat_prompt(
            engine, model_dir, [{"role": "user", "content": "Hello"}], "high",
            enable_thinking=False)
    assert released == "high|unset|Hello"
    assert fastest == "high|False|Hello"


def test_fast_mode_disables_template_thinking_while_lossless_keeps_default():
    template = (
        "{{ 'thinking' if enable_thinking is not defined or enable_thinking "
        "else 'no-thinking' }}"
    )
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        engine = _fake_engine(model_type="glm_moe_dsa")
        args = (engine, model_dir, [{"role": "user", "content": "x"}],
                "high", [], [])
        released, *_ = _prepare_chat_prompt(*args, "lossless", 1)
        fastest, *_ = _prepare_chat_prompt(*args, "fast", 1)
    assert released == "thinking"
    assert fastest == "no-thinking"


def test_selected_execution_tools_can_restore_parameter_descriptions(tmp_path):
    """The hidden gateway compacts the large discovery catalog, but once it
    has selected a tiny execution set the field-level distinctions are part of
    the model's job, not disposable retrieval prose."""
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}"
        "{{ tool.function.parameters.properties.path.description "
        "if tool.function.parameters.properties.path.description is defined "
        "else 'MISSING' }}"
        "{% endfor %}")
    tool = {"type": "function", "function": {
        "name": "list_media", "description": "List media.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description":
                     "A root folder path, not a library section name."}},
            "required": ["path"], "additionalProperties": False}}}
    args = (
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}],
        "low", [tool], [tool], "fast", 1)

    compact, *_rest, compact_meta = _prepare_chat_prompt(*args)
    hydrated, *_rest, hydrated_meta = _prepare_chat_prompt(
        *args, preserve_tool_parameter_prose=True)

    assert compact == "MISSING"
    assert compact_meta["schema_profile"] == "compact-no-nested-prose"
    assert hydrated == "A root folder path, not a library section name."
    assert hydrated_meta["schema_profile"] == (
        "selected-full-prose-compact-json")


def test_lossless_prompt_preserves_prose_but_applies_x_optional(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}"
        "{{ tool.function.parameters.required|join(',') }}|"
        "{{ tool.function.parameters.properties.path.description }}"
        "{% endfor %}")
    tool = {"type": "function", "function": {
        "name": "list_media", "parameters": {
            "type": "object", "properties": {
                "path": {"type": ["string", "null"],
                         "description": "Root folder path."},
                "limit": {"type": "integer"},
            },
            "required": ["path", "limit"], "x-optional": ["path"],
        }}}
    prompt, *_rest, metadata = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}],
        "low", [tool], [tool], "lossless", 1)
    assert prompt == "limit|Root folder path."
    assert metadata["schema_profile"] == "released-effective-optionality"
    assert tool["function"]["parameters"]["required"] == ["path", "limit"]


def test_explicit_high_effort_overrides_fast_no_thinking_default():
    template = "{{ 'thinking' if enable_thinking else 'no-thinking' }}"
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(template)
        prompt, *_ = _prepare_chat_prompt(
            _fake_engine(model_type="qwen3"), model_dir,
            [{"role": "user", "content": "x"}], "high", [], [], "fast", 1,
            enable_thinking=True, reasoning_requested=True)
    assert prompt == "thinking"


def test_explicit_effort_injects_instruction_for_non_reasoning_template():
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "chat_template.jinja").write_text(
            "{% for message in messages %}{{ message.role }}={{ message.content }}|{% endfor %}")
        prompt = _chat_prompt(
            _fake_engine(), model_dir, [{"role": "user", "content": "Solve it"}],
            "high", enable_thinking=True, reasoning_requested=True)
    assert prompt.startswith("system=Reason thoroughly")
    assert prompt.endswith("user=Solve it|")


def _named_tool(name: str, marker: str | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name,
        "description": marker or f"Call {name}",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "string", "description": f"Value for {name}"}}},
    }}


def test_tool_request_controls_implement_auto_none_and_parallel_policy():
    chat_tools = [_named_tool("weather")]
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"parallel_tool_calls": False}, chat_tools)
    assert effective == chat_tools
    assert choice == "auto"
    assert not parallel

    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"tool_choice": "none"}, chat_tools)
    assert effective == []
    assert choice == "none"
    assert parallel

    anthropic_tools = [{"name": "weather", "input_schema": {"type": "object"}}]
    effective, choice, parallel = _tool_request_controls(
        "/messages", {"tool_choice": {
            "type": "auto", "disable_parallel_tool_use": True}}, anthropic_tools)
    assert effective == anthropic_tools
    assert choice == "auto"
    assert not parallel


def test_tool_request_controls_reject_unenforceable_and_malformed_choices():
    chat_tools = [_named_tool("weather")]
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {"tool_choice": "required"}, chat_tools)
    assert effective == chat_tools and choice == "required" and parallel
    effective, choice, parallel = _tool_request_controls(
        "/chat/completions", {
            "tool_choice": {"type": "function", "function": {"name": "weather"}}},
        chat_tools)
    assert effective == chat_tools and choice == "specific:weather" and parallel
    bad_requests = [
        {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
        {"parallel_tool_calls": "false"},
    ]
    for request in bad_requests:
        try:
            _tool_request_controls("/chat/completions", request, chat_tools)
        except RequestValidationError:
            pass
        else:
            raise AssertionError(f"malformed/unenforceable controls accepted: {request}")


def test_unsupported_generation_controls_fail_instead_of_being_ignored():
    sampling = _validate_generation_controls(
        "/chat/completions", {"n": 1, "response_format": {"type": "text"}})
    assert sampling.is_greedy
    sampling = _validate_generation_controls(
        "/messages", {"temperature": 0.7, "top_p": 0.9, "top_k": 10,
                      "seed": 123})
    assert not sampling.is_greedy
    assert (sampling.temperature, sampling.top_p, sampling.top_k, sampling.seed) == (
        0.7, 0.9, 10, 123)
    assert not _validate_generation_controls(
        "/responses", {"top_p": 0.8}).is_greedy
    assert _validate_generation_controls(
        "/chat/completions", {"response_format": {"type": "json_object"}})
    bad = [
        ("/chat/completions", {"n": 2}),
        ("/chat/completions", {"logprobs": True}),
        ("/chat/completions", {"presence_penalty": 0.5}),
        ("/chat/completions", {"logit_bias": {"42": 100}}),
        ("/chat/completions", {"functions": []}),
        ("/completions", {"best_of": 2}),
        ("/completions", {"echo": True}),
        ("/responses", {"top_logprobs": 5}),
        ("/responses", {"previous_response_id": "resp_old"}),
        ("/responses", {"background": True}),
        ("/responses", {"truncation": "auto"}),
        ("/responses", {"text": {"format": {"type": "json_schema"}}}),
        ("/responses", {"text": {"verbosity": "low"}}),
        ("/responses", {"temperature": "0.7"}),
        ("/responses", {"top_p": 1.5}),
        ("/responses", {"reasoning": {"effort": 7}}),
        ("/messages", {"top_k": -1}),
        ("/responses", {"seed": -1}),
        ("/messages", {"thinking": {"type": "mystery"}}),
        ("/messages", {"thinking": {"type": "enabled", "budget_tokens": 0}}),
    ]
    for route, request in bad:
        try:
            _validate_generation_controls(route, request)
        except RequestValidationError:
            pass
        else:
            raise AssertionError(f"unsupported control was ignored: {request}")


def test_reasoning_controls_map_all_protocols():
    assert _request_reasoning_controls(
        "/chat/completions", {"reasoning_effort": "high"})[:3] == (
            "high", True, True)
    assert _request_reasoning_controls(
        "/responses", {"reasoning": {"effort": "minimal"}})[:3] == (
            "minimal", False, True)
    assert _request_reasoning_controls(
        "/messages", {"thinking": {"type": "enabled", "budget_tokens": 128}}) == (
            "high", True, True, 128)


def test_parallel_tool_calls_false_keeps_only_first_parsed_call():
    text = (
        '<tool_call>{"name":"alpha","arguments":{}}</tool_call>'
        '<tool_call>{"name":"beta","arguments":{}}</tool_call>')
    tools = [_named_tool("alpha"), _named_tool("beta")]
    content, calls = _parse_request_tool_calls(
        text, tools, "qwen2", allow_parallel=False)
    assert content == ""
    assert [call["function"]["name"] for call in calls] == ["alpha"]


def test_fast_tool_catalog_permutations_render_identically_but_wire_order_survives(
        tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool | tojson }}\n{% endfor %}"
        "{% for message in messages %}{{ message.content }}{% endfor %}")
    engine = _fake_engine()
    tools = [_named_tool("zeta"), _named_tool("alpha"), _named_tool("mu")]
    raw = [{"type": "function", "name": t["function"]["name"],
            "parameters": t["function"]["parameters"]} for t in tools]
    messages = [{"role": "user", "content": "hello"}]

    first = _prepare_chat_prompt(
        engine, tmp_path, messages, "low", tools, raw, "fast", 1)
    permutation = [2, 0, 1]
    second = _prepare_chat_prompt(
        engine, tmp_path, messages, "low", [tools[i] for i in permutation],
        [raw[i] for i in permutation], "fast", 1)

    assert first[0] == second[0]
    assert isinstance(first[0], PreparedPrompt)
    assert first[0].token_ids == tuple(engine.tokenizer.encode(first[0]).ids)
    assert first[1] == second[1]
    assert first[4]["tool_catalog_id"] == second[4]["tool_catalog_id"]
    assert first[4]["tool_order_profile"] == "canonical-name-v1"
    # Prompt ordering is an internal cache optimization; response schemas retain
    # the exact request order expected by each protocol adapter.
    assert [t["function"]["name"] for t in second[2]] == ["mu", "zeta", "alpha"]
    assert [t["name"] for t in second[3]] == ["mu", "zeta", "alpha"]


def test_relevant_parameter_prompt_returns_packed_constraint_schema_but_full_wire(
        tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool | tojson }}{% endfor %}"
        "{% for message in messages %}{{ message.content }}{% endfor %}")
    engine = _fake_engine()
    properties = {
        "query": {"type": "string"},
        "excludeWarehouseId": {"type": "string"},
        "minimumQuantity": {"type": "number"},
        "maximumQuantity": {"type": "number"},
        "cursor": {"type": "string"},
        "limit": {"type": "integer"},
        **{f"unrelated{index}": {"type": "string"} for index in range(10)},
    }
    tool = _named_tool("inventory_list_stock")
    tool["function"]["parameters"] = {
        "type": "object", "properties": properties,
        "additionalProperties": False,
    }
    raw = [{
        "type": "function", "name": "inventory_list_stock",
        "parameters": tool["function"]["parameters"],
    }]
    messages = [{
        "role": "user",
        "content": "list stock below five excluding warehouse west and paginate",
    }]

    prepared = _prepare_chat_prompt(
        engine, tmp_path, messages, "low", [tool], raw, "fast", 1,
        preserve_tool_parameter_prose=True,
        relevant_parameter_messages=messages)

    prompt_properties = prepared[2][0]["function"]["parameters"]["properties"]
    assert len(prompt_properties) == 12
    assert {"excludeWarehouseId", "maximumQuantity", "cursor", "limit"} <= set(
        prompt_properties)
    assert len(prepared[3][0]["parameters"]["properties"]) == len(properties)


def test_fast_qwen_style_tools_carry_token_aligned_capsule_spans(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "<tools>{% for tool in tools %}\n{{ tool | tojson }}{% endfor %}"
        "\n</tools>{{ messages[0].content }}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]

    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "fast", 1)

    assert len(prompt.tool_capsules) == 2
    bodies = ["".join(prompt.token_ids[start:end])
              for _identity, start, end in prompt.tool_capsules]
    assert '"name":"alpha"' in bodies[0]
    assert '"name":"zeta"' in bodies[1]


def test_tool_capsule_spans_preserve_duplicate_occurrences_and_boundaries():
    from jinja2.utils import htmlsafe_json_dumps

    tool = _named_tool("duplicate")
    serialized = str(htmlsafe_json_dumps(
        tool, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True))
    prompt = (
        "The literal <tools></tools> is documentation."
        f"<tools>\n{serialized}\n{serialized}\n</tools>")
    token_ids = tuple(prompt)
    offsets = tuple((index, index + 1) for index in range(len(prompt)))

    spans = _tool_capsule_spans(
        prompt, [tool, tool], token_ids, offsets)

    assert len(spans) == 2
    assert spans[0][0] == spans[1][0]
    assert spans[0][2] <= spans[1][1]
    assert prompt[spans[0][1]:spans[0][2]] == serialized
    assert prompt[spans[1][1]:spans[1][2]] == serialized


def test_tool_capsule_offset_index_uses_first_nonempty_duplicate_start():
    from jinja2.utils import htmlsafe_json_dumps

    tool = _named_tool("alpha")
    serialized = str(htmlsafe_json_dumps(
        tool, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True))
    prompt = f"<tools>{serialized}</tools>"
    char_start = prompt.index(serialized)
    offsets = [(index, index + 1) for index in range(len(prompt))]
    offsets.insert(char_start, (char_start, char_start))
    token_ids = tuple(range(len(offsets)))

    spans = _tool_capsule_spans(prompt, [tool], token_ids, tuple(offsets))

    assert len(spans) == 1
    assert spans[0][1] == char_start + 1
    assert spans[0][2] == char_start + len(serialized) + 1


def test_token_offset_index_matches_first_interval_reference_across_blocks():
    offsets = tuple(
        (index // 2, min(1_000, index // 2 + 1 + (index * 17) % 47))
        for index in range(1_024))
    index = _TokenOffsetIndex(offsets, 1_000)

    for token_start in (0, 1, 127, 255, 256, 511, 700, 1_023):
        for char_end in (1, 17, 128, 255, 400, 511):
            expected = next((
                position + 1
                for position, (start, end) in enumerate(offsets)
                if position >= token_start and start < char_end <= end
            ), None)
            assert index.token_end(char_end, token_start) == expected


def test_tool_capsule_spans_fail_closed_on_nonmonotonic_offsets():
    tool = _named_tool("alpha")
    prompt = "<tools>{}</tools>"
    offsets = tuple(
        [(0, 1), (2, 3), (1, 2)]
        + [(index, index + 1) for index in range(3, len(prompt))])

    assert _tool_capsule_spans(
        prompt, [tool], tuple(range(len(offsets))), offsets) == ()


def test_native_template_that_ignores_tools_gets_canonical_tool_preamble(
        tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role }}:"
        "{{ message.content }}|{% endfor %}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]

    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "fast", 1)

    assert prompt.startswith("system:You have access to the following tools.")
    assert "<tools>" in prompt and "</tools>" in prompt
    assert prompt.index('"name":"alpha"') < prompt.index('"name":"zeta"')
    assert len(prompt.tool_capsules) == 2


def test_lossless_tool_order_remains_request_order(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool.function.name }}|{% endfor %}")
    tools = [_named_tool("zeta"), _named_tool("alpha")]
    prompt, *_ = _prepare_chat_prompt(
        _fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low",
        tools, tools, "lossless", 1)
    assert prompt == "zeta|alpha|"


def test_parallel_tool_completion_order_renders_one_canonical_prompt(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role }}:"
        "{% for call in message.tool_calls or [] %}{{ call.id }};{% endfor %}"
        "{{ message.tool_call_id or '' }}={{ message.content or '' }}|{% endfor %}")
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_alpha", "type": "function",
         "function": {"name": "alpha", "arguments": "{}"}},
        {"id": "call_beta", "type": "function",
         "function": {"name": "beta", "arguments": "{}"}},
    ]}
    alpha = {"role": "tool", "tool_call_id": "call_alpha", "content": "A"}
    beta = {"role": "tool", "tool_call_id": "call_beta", "content": "B"}
    args = (_fake_engine(), tmp_path)

    first = _prepare_chat_prompt(
        *args, [assistant, alpha, beta], "low", [], [], "lossless", 1)
    second = _prepare_chat_prompt(
        *args, [assistant, beta, alpha], "low", [], [], "lossless", 1)

    assert first[0] == second[0]
    assert first[0].token_ids == second[0].token_ids


def test_fast_added_or_removed_tool_preserves_canonical_catalog_prefix(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "HEADER|{% for tool in tools %}[{{ tool.function.name }}]"
        "{% endfor %}|MESSAGES")
    base = [_named_tool(name) for name in ("zeta", "beta", "delta")]
    added = base + [_named_tool("epsilon")]
    args = (_fake_engine(), tmp_path, [{"role": "user", "content": "x"}], "low")
    base_prompt, *_ = _prepare_chat_prompt(*args, base, base, "fast", 1)
    added_prompt, *_ = _prepare_chat_prompt(*args, added, added, "fast", 1)
    assert base_prompt == "HEADER|[beta][delta][zeta]|MESSAGES"
    assert added_prompt == "HEADER|[beta][delta][epsilon][zeta]|MESSAGES"
    common = os.path.commonprefix((base_prompt, added_prompt))
    assert common == "HEADER|[beta][delta]["


def test_fast_duplicate_tool_names_fail_before_rendering(tmp_path):
    tools = [_named_tool("same", "one"), _named_tool("same", "two")]
    try:
        _prepare_chat_prompt(
            _fake_engine(), tmp_path, [{"role": "user", "content": "x"}],
            "low", tools, tools, "fast", 1)
    except RequestValidationError as error:
        assert str(error) == "duplicate tool function name: 'same'"
    else:
        raise AssertionError("ambiguous duplicate tools were rendered")


def test_fast_shortlist_is_permutation_invariant_at_score_ties(tmp_path):
    from unittest.mock import patch

    (tmp_path / "chat_template.jinja").write_text(
        "{% for tool in tools %}{{ tool.function.name }}|{% endfor %}")
    tools = [_named_tool(name) for name in ("zeta", "beta", "alpha")]
    permutation = [tools[1], tools[0], tools[2]]
    args = (_fake_engine(), tmp_path, [{"role": "user", "content": "unrelated"}],
            "low")
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_LIMIT": "2"}):
        first = _prepare_chat_prompt(*args, tools, tools, "fast", 1)
        second = _prepare_chat_prompt(*args, permutation, permutation, "fast", 1)
    assert first[0] == second[0] == "alpha|beta|"
    assert {t["function"]["name"] for t in first[2]} == {"alpha", "beta"}
    assert {t["function"]["name"] for t in second[2]} == {"alpha", "beta"}


def test_hidden_tool_gateway_starts_virtual_only_then_retrieves_real_tools():
    from unittest.mock import patch

    tools = [
        _named_tool("workspace_execute", "Execute a shell command in the workspace."),
        _named_tool("browser_open", "Open a web page in a browser."),
        _named_tool("calendar_create", "Create a calendar event."),
    ]
    raw = [{
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "parameters": tool["function"]["parameters"],
    } for tool in tools]
    messages = [{"role": "user", "content": "Tell me a NodeJS joke."}]
    initial, initial_raw, pinned, retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, limit=2)
    assert initial == [] and initial_raw == [] and pinned == 0
    assert retrieval["tool_embedding_status"] == "not_queried"

    selected, selected_raw, pinned, retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, query="open this page in the browser", limit=2)
    assert "browser_open" in {
        tool["function"]["name"] for tool in selected
    }
    assert "browser_open" in {tool["name"] for tool in selected_raw}
    assert len(selected) == 2 and pinned == 0
    assert "tool_retrieval_profile" in retrieval

    virtual, virtual_raw = _hidden_tool_search_pair()
    assert virtual["function"]["name"] == "vmodel_search_tools"
    assert virtual_raw["name"] == "vmodel_search_tools"
    enable, enable_raw = _hidden_tool_enable_pair()
    assert enable["function"]["name"] == "vmodel_enable_tools"
    assert enable_raw["name"] == "vmodel_enable_tools"
    virtuals, virtuals_raw = _hidden_gateway_virtual_pairs()
    assert [tool["function"]["name"] for tool in virtuals] == [
        "vmodel_search_tools", "vmodel_enable_tools"]
    assert [tool["name"] for tool in virtuals_raw] == [
        "vmodel_search_tools", "vmodel_enable_tools"]
    abstain, abstain_raw = _hidden_tool_abstain_pair()
    assert abstain["function"]["name"] == "vmodel_no_suitable_tool"
    assert abstain_raw["parameters"]["required"] == ["reason"]
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_GATEWAY": "1"}):
        assert _hidden_tool_gateway_enabled("fast", len(tools), "auto")
        assert not _hidden_tool_gateway_enabled("lossless", len(tools), "auto")
        assert not _hidden_tool_gateway_enabled("fast", len(tools), "specific:browser_open")


def test_hidden_gateway_abstention_defaults_safe_and_real_tool_mode_is_explicit():
    assert _hidden_gateway_abstention_policy("auto") == (
        True, "auto-safe-abstention")
    assert _hidden_gateway_abstention_policy("1") == (
        True, "operator-enabled")
    assert _hidden_gateway_abstention_policy("0") == (
        False, "operator-required-real-tool")
    with pytest.raises(ValueError, match="must be auto, 0, or 1"):
        _hidden_gateway_abstention_policy("required")


def test_required_real_tool_mode_applies_only_to_forced_external_actions():
    assert _hidden_gateway_execution_abstention_policy(
        False, "operator-required-real-tool",
        "external-action-imperative",
    ) == (
        False,
        "operator-required-real-tool:external-action-imperative",
    )
    assert _hidden_gateway_execution_abstention_policy(
        False, "operator-required-real-tool", None,
    ) == (True, "unforced-search-safety-fallback")
    assert _hidden_gateway_execution_abstention_policy(
        True, "auto-safe-abstention", "external-action-imperative",
    ) == (True, "auto-safe-abstention")


def test_plex_transcript_keeps_fixed_decision_catalog_and_pinned_execution_set():
    """Regression for the user's real 2026-07-19 Plex pagination transcript."""
    from unittest.mock import patch

    tools = [
        _named_tool("plugin__plex__plex_list_library", "List Plex media with pagination."),
        _named_tool("workspace_execute", "Execute a workspace command."),
        _named_tool("browser_open", "Open a browser page."),
        _named_tool("calendar_create", "Create a calendar event."),
    ]
    raw = [{
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "parameters": tool["function"]["parameters"],
    } for tool in tools]
    first_turn = [{
        "role": "user",
        "content": (
            "list the plex movies/tv shows that are age rating PG13 or TV-7 "
            "or less(for younger kids) and whose root folder does NOT contain "
            "\"/Kids/\"\nMake sure to paginate the plex listing\n"),
    }]
    later_turn = first_turn + [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_plex", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library",
                "arguments": '{"limit":32,"offset":0}',
            },
        }],
    }, {
        "role": "tool", "tool_call_id": "call_plex",
        "name": "plugin__plex__plex_list_library",
        "content": '{"movies":[{"title":"A Christmas Carol",'
                   '"contentRating":"PG"}],"movieHasMore":true}',
    }, {
        "role": "user", "content": "try just doing no query?",
    }]

    # The decision schemas never include transcript-pinned real functions.
    virtuals, _raw_virtuals = _hidden_gateway_virtual_pairs()
    assert [tool["function"]["name"] for tool in virtuals] == [
        "vmodel_search_tools", "vmodel_enable_tools"]

    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([0, 1, 2, 3], {"tool_embedding_status": "test"})):
        selected, _selected_raw, _pinned, _meta = _hidden_gateway_catalogs(
            tools, raw, first_turn, query="list Plex media", limit=2)
    activated = tuple(tool["function"]["name"] for tool in selected)
    assert activated == ("plugin__plex__plex_list_library",)

    # Page/corrected-argument intent ranks an already-activated tool first:
    # preserve the exact schema set even though call history now hard-pins Plex.
    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([0, 2, 3, 1], {"tool_embedding_status": "test"})):
        stable, _stable_raw, pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, later_turn, query="continue Plex pagination without query",
            limit=2, activated_names=activated,
            expansion_limit=2, max_activated=4)
    assert tuple(tool["function"]["name"] for tool in stable) == activated
    assert pinned == 1
    assert metadata["gateway_activation_profile"] == "stable-hit"

    # A genuinely different top capability is still admitted rather than being
    # trapped behind the old tool choice.
    with patch("runtime.toolcalls.rank_tool_indices",
               return_value=([2, 3, 0, 1], {"tool_embedding_status": "test"})):
        expanded, _expanded_raw, _pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, later_turn, query="open a page in the browser",
            limit=2, activated_names=activated,
            expansion_limit=2, max_activated=4)
    assert {tool["function"]["name"] for tool in expanded} == {
        *activated, "browser_open", "calendar_create"}
    assert metadata["gateway_activation_profile"] == "expanded"


def test_hidden_gateway_search_hydrates_at_most_four_without_forcing_four():
    assert _hidden_gateway_search_result_limit(32, 4, 32) == 4
    assert _hidden_gateway_search_result_limit(32, 4, 2) == 2
    assert _hidden_gateway_search_result_limit(32, 4, 0) == 1
    assert _hidden_gateway_search_result_limit(32, 4, "many") == 4
    assert _hidden_gateway_search_result_limit(3, 4, 32) == 3


def test_hidden_gateway_preserves_explicit_plugin_namespace_in_router_query():
    tools = [
        _named_tool("plugin__plex__plex_list_library",
                    "List Plex movies and TV series."),
        _named_tool("mastra_workspace_list_files",
                    "List files and folders in the workspace."),
    ]
    raw = [{"type": "function", **tool["function"]} for tool in tools]
    selected, _raw, _pinned, metadata = _hidden_gateway_catalogs(
        tools, raw,
        [{"role": "user", "content": (
            "Use Plex to list movies and TV outside a root folder")}],
        query="list files movies tv root folder", limit=1)
    assert selected[0]["function"]["name"] == (
        "plugin__plex__plex_list_library")
    assert metadata["gateway_explicit_namespaces"] == ["plex"]


def test_gateway_activation_key_survives_appended_tool_turns_without_raw_state():
    tools = [_named_tool("plugin__plex__plex_list_library")]
    anchor = [{"role": "system", "content": "stable harness prompt"}, {
        "role": "user", "content": "list Plex media and paginate"}]
    continuation = anchor + [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library", "arguments": "{}"}}]}, {
        "role": "tool", "tool_call_id": "call_1", "content": "page one"}, {
        "role": "user", "content": "get page two"}]
    first_key = _hidden_gateway_conversation_key("lossy-Qwen3-4B", tools, anchor)
    next_key = _hidden_gateway_conversation_key(
        "lossy-Qwen3-4B", tools, continuation)
    assert first_key == next_key
    assert len(first_key) == 64

    _hidden_gateway_activation_clear()
    try:
        _hidden_gateway_activation_put(first_key, tools)
        assert _hidden_gateway_activation_get(next_key, tools) == (
            "plugin__plex__plex_list_library",)
    finally:
        _hidden_gateway_activation_clear()


def test_hidden_gateway_execution_prompt_is_prefix_stable_across_tool_result():
    tools = [_named_tool("plugin__plex__plex_list_library_media")]
    anchor = [{"role": "system", "content": "stable harness prompt"}, {
        "role": "user", "content": "list Plex media and paginate"}]
    continuation = anchor + [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library_media",
                "arguments": "{\"limit\":500,\"offset\":0}"}}]}, {
        "role": "tool", "tool_call_id": "call_1",
        "name": "plugin__plex__plex_list_library_media",
        "content": "{\"movieHasMore\":true}"}]
    activation_key = "a" * 64

    initial = _hidden_gateway_execution_messages(
        anchor, tools, activation_key)
    followup = _hidden_gateway_execution_messages(
        continuation, tools, activation_key)

    assert followup[:len(initial)] == initial
    assert followup[len(initial):] == continuation[len(anchor):]
    assert initial[-2]["tool_calls"][0]["function"] == {
        "name": "vmodel_enable_tools", "arguments": "{}"}
    assert initial[-1]["name"] == "vmodel_enable_tools"


def test_hidden_gateway_execution_activation_changes_with_catalog():
    anchor = [{"role": "user", "content": "inspect Plex"}]
    first = _hidden_gateway_execution_messages(
        anchor, [_named_tool("plex_list")], "b" * 64)
    expanded = _hidden_gateway_execution_messages(
        anchor, [_named_tool("plex_list"), _named_tool("plex_search")],
        "b" * 64)
    assert first[-2]["tool_calls"][0]["id"] != \
        expanded[-2]["tool_calls"][0]["id"]
    assert json.loads(expanded[-1]["content"])["tools"] == [
        "plex_list", "plex_search"]


def test_hidden_gateway_suffix_contract_is_selected_schema_local_not_subject_local():
    tool = _named_tool("calendar_list_events")
    tool["function"].update({
        "description": "List calendar events with optional time and attendee filters.",
        "parameters": {"type": "object", "properties": {
            "afterTime": {"type": "string", "description": "Lower time bound"},
            "attendee": {"type": "string", "description": "Attendee email"},
        }},
    })
    messages = [{"role": "user", "content": "list tomorrow's events for Sam"}]

    execution = _hidden_gateway_execution_messages(
        messages, [tool], "c" * 64, include_interface_contract=True)
    activation = json.loads(execution[-1]["content"])

    assert activation["tools"] == ["calendar_list_events"]
    interface = activation["interfaces"][0]["function"]
    assert interface["name"] == "calendar_list_events"
    assert set(interface["parameters"]["properties"]) == {
        "afterTime", "attendee"}
    assert "Plex" not in execution[-1]["content"]


@pytest.mark.parametrize(("user_text", "tool_name"), [
    ("list inventory below its reorder point", "inventory_list_stock"),
    ("show failed build jobs from today", "ci_list_jobs"),
    ("find unread messages and paginate", "mail_list_messages"),
])
def test_hidden_gateway_result_suffix_anchor_retains_original_intent_across_domains(
        user_text, tool_name):
    tool = _named_tool(tool_name)
    tool["function"].update({
        "description": "List rows under a declared ordering contract.",
        "parameters": {"type": "object", "properties": {
            "threshold": {"type": "string", "description": (
                "Ordered low < medium < high")},
        }},
    })
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": tool_name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "{\"hasMore\":false,\"rows\":[]}"},
    ]

    anchored = _hidden_gateway_result_suffix_anchor(
        messages, [tool])

    assert anchored[:-1] == messages[:-1]
    assert messages[-1]["content"] == \
        "{\"hasMore\":false,\"rows\":[]}"
    assert user_text in anchored[-1]["content"]
    assert tool_name in anchored[-1]["content"]
    assert "answer now" in anchored[-1]["content"]
    assert "Ordered low < medium < high" in anchored[-1]["content"]


def test_hidden_gateway_task_context_keeps_task_history_and_reports_omission():
    messages = [
        {"role": "system", "content": "global harness policy"},
        {"role": "developer", "content": "global developer policy"},
        {"role": "user", "content": "list Plex media and paginate"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "{\"movieHasMore\":true}"},
    ]
    full, full_omitted = _hidden_gateway_execution_context(messages, "full")
    assert full == messages
    assert full is not messages
    assert full_omitted == 0

    task, task_omitted = _hidden_gateway_execution_context(messages, "task")
    assert [message["role"] for message in task] == [
        "user", "assistant", "tool"]
    assert task_omitted == len(
        "global harness policyglobal developer policy")
    try:
        _hidden_gateway_execution_context(messages, "unknown")
    except ValueError as error:
        assert "full" in str(error) and "task" in str(error)
    else:
        raise AssertionError("an unknown execution context was accepted")


def test_hidden_gateway_terminal_context_reuses_task_projection_across_domains():
    tool = _named_tool("inventory_list_stock")
    tool["function"].update({
        "description": "List inventory under a declared filter contract.",
        "parameters": {"type": "object", "properties": {
            "maximumQuantity": {
                "type": "number", "description": "Inclusive upper bound"},
        }},
    })
    messages = [
        {"role": "system", "content": "unrelated global harness policy"},
        {"role": "developer", "content": "unrelated developer examples"},
        {"role": "user", "content": "list all stock below five and paginate"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "inventory_list_stock",
                "arguments": "{\"maximumQuantity\":5,\"offset\":0}",
            },
        }]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "{\"hasMore\":true,\"rows\":[{\"sku\":\"A\"}]}"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_2", "type": "function", "function": {
                "name": "inventory_list_stock",
                "arguments": "{\"maximumQuantity\":5,\"offset\":100}",
            },
        }]},
        {"role": "tool", "tool_call_id": "call_2",
         "content": "{\"hasMore\":false,\"rows\":[{\"sku\":\"B\"}]}"},
    ]

    projected, omitted = _hidden_gateway_terminal_context(
        messages, [tool], "task", suffix_contract=True)

    assert all(message["role"] not in ("system", "developer")
               for message in projected)
    assert [message["tool_call_id"] for message in projected
            if message["role"] == "tool"] == ["call_1", "call_2"]
    assert "list all stock below five and paginate" in projected[-1]["content"]
    assert "Inclusive upper bound" in projected[-1]["content"]
    assert "\"sku\":\"B\"" in projected[-1]["content"]
    assert "enabled_tools" not in projected[-1]["content"]
    assert "answer now instead" not in projected[-1]["content"]
    interface = json.loads(projected[-1]["content"].split(
        "<vmodel_private_context>", 1)[1].split(
            "</vmodel_private_context>", 1)[0])[
                "selected_interfaces"][0]["function"]
    assert set(interface["parameters"]["properties"]) == {
        "maximumQuantity"}
    assert omitted == len(
        "unrelated global harness policyunrelated developer examples")


def test_hidden_gateway_task_context_auto_always_stays_full():
    # 2026-07-26 correction: this narrow "task" capsule (plus its downstream
    # top-2 expert routing and prose stripping) was only ever validated
    # against one pinned captured request via a test harness that also
    # substituted a compact tool schema for the real one -- the unmodified
    # capture does not reliably reproduce its claimed win. `auto` no longer
    # activates it automatically for any shape, read-only or not; explicit
    # `VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT=task` still does (covered
    # below), preserving the code for future validated opt-in use.
    messages = [
        {"role": "system", "content": "x" * 5000},
        {"role": "user", "content": "list Plex media and paginate"},
    ]
    read_tool = _named_tool("plugin__plex__plex_list_library_media")
    assert _hidden_gateway_execution_context_policy(
        "auto", messages=messages, selected_tools=[read_tool],
        mode="fast", model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    ) == ("full", "auto-disabled-pending-broad-corpus-validation")

    mutating = _named_tool("plugin__mail__mail_delete_message")
    assert _hidden_gateway_execution_context_policy(
        "auto", messages=messages, selected_tools=[mutating],
        mode="fast", model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    )[0] == "full"
    mixed = _named_tool("plugin__mail__mail_get_and_delete_message")
    assert _hidden_gateway_execution_context_policy(
        "auto", messages=messages, selected_tools=[mixed],
        mode="fast", model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    )[0] == "full"
    assert _hidden_gateway_execution_context_policy(
        "auto", messages=[
            *messages, {"role": "developer", "content": "keep me"}],
        selected_tools=[read_tool], mode="fast",
        model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    )[0] == "full"
    assert _hidden_gateway_execution_context_policy(
        "auto", messages=messages, selected_tools=[read_tool],
        mode="lossless", model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    )[0] == "full"


def test_hidden_gateway_task_context_explicit_operator_request_still_works():
    messages = [
        {"role": "system", "content": "x" * 5000},
        {"role": "user", "content": "list Plex media and paginate"},
    ]
    read_tool = _named_tool("plugin__plex__plex_list_library_media")
    assert _hidden_gateway_execution_context_policy(
        "task", messages=messages, selected_tools=[read_tool],
        mode="fast", model_type="qwen3_5_moe", host_route=True,
        force_reason="external-action-imperative",
    ) == ("task", "operator-task")
    try:
        _hidden_gateway_execution_context_policy(
            "task", messages=messages, selected_tools=[read_tool],
            mode="lossless", model_type="qwen3_5_moe", host_route=True,
            force_reason="external-action-imperative")
    except ValueError as error:
        assert "lossy fast modes" in str(error)
    else:
        raise AssertionError("task context was accepted for a lossless mode")


def test_qwen_chunked_delta_auto_is_lossy_only_and_overridable():
    assert _qwen_chunked_delta_policy(
        "auto", mode="fast", model_type="qwen3_5_moe",
    ) == (True, "auto-lossy-qwen")
    assert _qwen_chunked_delta_policy(
        "auto", mode="lossless", model_type="qwen3_5",
    ) == (False, "lossless-checkpoint-boundary")
    assert _qwen_chunked_delta_policy(
        "1", mode="lossless", model_type="qwen3_5",
    ) == (True, "operator-forced")
    assert _qwen_chunked_delta_policy(
        "auto", mode="fast", model_type="glm_moe_dsa",
    ) == (False, "unsupported-architecture")


def test_qwen_compiled_delta_is_explicit_and_architecture_scoped():
    assert _qwen_compiled_delta_policy(
        "0", model_type="qwen3_5",
    ) == (False, "operator-disabled")
    assert _qwen_compiled_delta_policy(
        "1", model_type="qwen3_5_moe",
    ) == (True, "operator-forced")
    assert _qwen_compiled_delta_policy(
        "1", model_type="glm_moe_dsa",
    ) == (False, "unsupported-architecture")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _qwen_compiled_delta_policy("auto", model_type="qwen3_5")


def test_qwen_compiled_delta_displaces_only_auto_chunked_mode():
    assert _qwen_delta_prefill_policies(
        "1", "auto", mode="fast", model_type="qwen3_5",
    ) == (
        True, "operator-forced", False, "compiled-delta-selected",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _qwen_delta_prefill_policies(
            "1", "1", mode="fast", model_type="qwen3_5")


def test_qwen_lossy_suffix_prefill_is_explicit_fixed_and_hybrid_safe():
    layer_types = [
        "linear_attention", "linear_attention", "linear_attention",
        "full_attention",
    ] * 10
    assert _qwen_lossy_suffix_prefill_policy(
        "", mode="fast", total_layers=40, layer_types=layer_types,
    ) == (0, 0, 0)
    assert _qwen_lossy_suffix_prefill_policy(
        "4:128", mode="fast", total_layers=40, layer_types=layer_types,
    ) == (4, 0, 128)
    assert _qwen_lossy_suffix_prefill_policy(
        "8:256:128", mode="fast", total_layers=40,
        layer_types=layer_types,
    ) == (8, 256, 128)
    with pytest.raises(ValueError, match="PREFIX_TOKENS"):
        _qwen_lossy_suffix_prefill_policy(
            "8:-1:128", mode="fast", total_layers=40,
            layer_types=layer_types)
    with pytest.raises(ValueError, match="lossy fast mode"):
        _qwen_lossy_suffix_prefill_policy(
            "4:128", mode="lossless", total_layers=40,
            layer_types=layer_types)
    with pytest.raises(ValueError, match="first full-attention"):
        _qwen_lossy_suffix_prefill_policy(
            "3:128", mode="fast", total_layers=40,
            layer_types=layer_types)
    with pytest.raises(ValueError, match="EARLY_LAYERS:SUFFIX_TOKENS"):
        _qwen_lossy_suffix_prefill_policy(
            "4", mode="fast", total_layers=40, layer_types=layer_types)


def test_grammar_jump_forward_auto_is_disabled_pending_validation():
    # 2026-07-26 correction: this used to default-enable for every lossy
    # fast-mode request. That reproduces a real measured correctness risk
    # (it can change free-choice tool arguments, e.g. a pagination `limit`)
    # that was already documented as "rejected as an automatic default,
    # remains opt-in" before a later commit silently reversed it. `auto` now
    # matches that original, tested-safe behavior again; explicit opt-in via
    # `VMODEL_GRAMMAR_JUMP_FORWARD_LOSSY=1` is unchanged.
    assert _grammar_jump_forward_policy(
        "auto", mode="fast"
    ) == (False, "auto-disabled-pending-argument-fidelity-validation")
    assert _grammar_jump_forward_policy(
        "0", mode="fast") == (False, "operator-disabled")
    assert _grammar_jump_forward_policy(
        "1", mode="fast-long") == (True, "operator-forced")
    assert _grammar_jump_forward_policy(
        "1", mode="lossless") == (False, "lossless-route")


def test_hidden_gateway_forces_only_high_confidence_external_intents():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Tell me a joke about Node.js."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "How does the Node.js event loop work?"},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Write one coherent paragraph about streaming."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "List three reasons streaming feels faster."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Create a short poem."},
    ]) is None
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "What folder are we in?"},
    ]) == "external-state-inspection"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Whats the largest top level directory?"},
    ]) == "external-state-inspection"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Check for real."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Write this file in the workspace."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Search the web."},
    ]) == "external-action-imperative"
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Use an available tool to inspect it."},
    ]) == "explicit-tool-request"


def test_hidden_gateway_forces_confirmed_deferred_action_but_not_bare_ack():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "Which directory is largest?"},
        {"role": "assistant", "content": "I'll run a command to check it."},
        {"role": "user", "content": "do it"},
    ]) == "confirmed-deferred-action"
    assert _hidden_gateway_force_reason([
        {"role": "assistant", "content": "That sounds good."},
        {"role": "user", "content": "okay"},
    ]) is None


def test_confirmed_action_query_uses_prior_intent_not_bare_ack():
    messages = [
        {"role": "user", "content": "Which workspace directory is largest?"},
        {
            "role": "assistant",
            "content": "I'll run a shell command to inspect directory sizes.",
        },
        {"role": "user", "content": "do it"},
    ]
    query, profile = _hidden_gateway_semantic_query(
        messages, "confirmed-deferred-action")
    assert profile == "confirmed-action-context"
    assert "directory" in query
    assert "command" in query
    assert query != "do it"

    latest, latest_profile = _hidden_gateway_semantic_query(messages, None)
    assert latest_profile == "latest-user-intent"
    assert latest == "do it"


def test_hidden_gateway_does_not_force_again_after_tool_result():
    assert _hidden_gateway_force_reason([
        {"role": "user", "content": "What folder are we in?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "pwd", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
    ]) is None


def test_hidden_gateway_forces_activated_catalog_for_explicit_pagination():
    messages = [{
        "role": "tool", "tool_call_id": "call_1",
        "content": json.dumps({
            "movieHasMore": True,
            "nested": {"series_has_more": False},
        }),
    }]
    assert _hidden_gateway_force_reason(messages) == "tool-result-pagination"
    assert _hidden_gateway_decision_choice(
        "auto", "tool-result-pagination", activated_tools=True,
    ) == "specific:vmodel_enable_tools"
    assert _hidden_gateway_decision_choice(
        "auto", "tool-result-pagination", activated_tools=False,
    ) == "specific:vmodel_search_tools"


def test_hidden_gateway_terminal_pagination_synthesis_is_explicit_and_generic():
    base = [
        {"role": "user", "content": "List all inventory and paginate."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "inventory_list", "arguments": "{\"offset\":50}",
            },
        }]},
    ]
    assert _hidden_gateway_terminal_pagination_synthesis(base + [{
        "role": "tool", "tool_call_id": "call_1",
        "content": json.dumps({"items": [], "has_more": False}),
    }]) is True
    assert _hidden_gateway_terminal_pagination_synthesis(base + [{
        "role": "tool", "tool_call_id": "call_1",
        "content": json.dumps({"items": [], "has_more": True}),
    }]) is False
    assert _hidden_gateway_terminal_pagination_synthesis([
        {"role": "user", "content": "List current inventory."},
        {"role": "tool", "content": json.dumps({"hasMore": False})},
    ]) is False
    assert _hidden_gateway_terminal_pagination_synthesis([
        {"role": "user", "content": "List all inventory and paginate."},
        {"role": "tool", "content": json.dumps({"items": []})},
    ]) is False
    assert "completed retrieval mechanism" in \
        _HIDDEN_GATEWAY_TERMINAL_PAGINATION_POLICY
    assert "not a requested answer format" in \
        _HIDDEN_GATEWAY_TERMINAL_PAGINATION_POLICY
    assert "compact plain list" in \
        _HIDDEN_GATEWAY_TERMINAL_PAGINATION_POLICY
    assert "do not add a table" in \
        _HIDDEN_GATEWAY_TERMINAL_PAGINATION_POLICY


def test_hidden_gateway_deterministic_policy_is_explicit_and_production_connected():
    user = (
        "list the plex movies/tv shows rated PG13 or TV-7 or less and whose "
        'root does not contain /Kids/. Make sure to paginate the listing')
    messages = [{"role": "user", "content": user}]
    for index, has_more in enumerate((True, False)):
        call_id = f"call_{index}"
        messages.extend(({
            "role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": "plugin__plex__plex_list_library",
                    "arguments": json.dumps({
                        "mediaType": "all", "ratingOperator": "lte",
                        "movieRatingValue": "PG-13",
                        "showRatingValue": "TV-Y7",
                        "excludeRootFolderPath": "/Kids/",
                        "limit": 50, "offset": index * 50,
                    }),
                },
            }],
        }, {
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps({
                "movies": ([{
                    "title": "ALPHA_G", "contentRating": "G",
                    "rootFolderPath": "/Media/Movies",
                    "plexLibrarySectionName": "Movies",
                }] if index == 0 else []),
                "series": [],
                "movieHasMore": has_more,
                "seriesHasMore": has_more,
            }),
        }))

    disabled = _hidden_gateway_deterministic_policy_render(messages, False)
    assert disabled.render is None
    assert disabled.reason == "disabled"
    enabled = _hidden_gateway_deterministic_policy_render(messages, True)
    assert enabled.reason == "rendered"
    assert enabled.render is not None
    assert enabled.render.text == "Movies: ALPHA_G (G)\nTV Shows: None"


def test_responses_deterministic_policy_bypasses_prompt_and_model(
        monkeypatch, tmp_path):
    call_id = "call_terminal"
    messages = [{
        "role": "user",
        "content": (
            "list the plex movies/tv shows rated PG13 or TV-7 or less and "
            "whose root does not contain /Kids/. Make sure to paginate the "
            "listing"),
    }, {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": "plugin__plex__plex_list_library",
                "arguments": json.dumps({
                    "mediaType": "all", "ratingOperator": "lte",
                    "movieRatingValue": "PG-13",
                    "showRatingValue": "TV-Y7",
                    "excludeRootFolderPath": "/Kids/",
                    "limit": 50, "offset": 0,
                }),
            },
        }],
    }, {
        "role": "tool", "tool_call_id": call_id,
        "content": json.dumps({
            "movies": [{
                "title": "ALPHA_G", "contentRating": "G",
                "rootFolderPath": "/Media/Movies",
                "plexLibrarySectionName": "Movies",
            }],
            "series": [], "movieHasMore": False, "seriesHasMore": False,
        }),
    }]
    raw_tool = {
        "type": "function", "name": "plugin__plex__plex_list_library",
        "description": "List Plex media.",
        "parameters": {"type": "object", "properties": {}},
    }
    engine = _fake_engine(model_type="qwen3_5")
    handler = object.__new__(Handler)
    handler._structured_output = None
    handler._reasoning_effort = "low"
    handler._enable_thinking = False
    handler._reasoning_requested = False
    handler._sampling = SimpleNamespace(profile="greedy")
    handler._output_token_budget_source = "request"
    responses = []
    handler._json = lambda code, body: responses.append((code, body))
    monkeypatch.setenv("VMODEL_FAST_TOOL_GATEWAY", "1")
    monkeypatch.setenv(
        "VMODEL_FAST_TOOL_GATEWAY_DETERMINISTIC_POLICY", "1")
    monkeypatch.setattr(
        "runtime.server._prepare_chat_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(
            "deterministic render must not build a model prompt")))
    monkeypatch.setattr(
        "runtime.server._engine_generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(
            "deterministic render must not run a model")))

    handler._do_responses(
        {}, "fixture", tmp_path, engine, "fast", 256, False, [],
        [raw_tool], [raw_tool], "auto", True, messages, [])

    assert responses[0][0] == 200
    body = responses[0][1]
    assert body["output_text"] == "Movies: ALPHA_G (G)\nTV Shows: None"
    assert body["usage"]["input_tokens"] == 0
    assert body["vmodel_tool_selection"][
        "gateway_deterministic_policy_rendered"] == 1
    assert body["vmodel_timing"]["total_engine_seconds"] == 0


def test_hidden_gateway_host_route_only_skips_already_forced_decisions():
    assert _hidden_gateway_host_action(
        "external-action-imperative",
        activated_tools=False, semantic_query="list Plex media",
    ) == "vmodel_search_tools"
    assert _hidden_gateway_host_action(
        "tool-result-pagination",
        activated_tools=True, semantic_query="original user request",
    ) == "vmodel_enable_tools"
    assert _hidden_gateway_host_action(
        None, activated_tools=True, semantic_query="maybe use a tool",
    ) is None
    assert _hidden_gateway_host_action(
        "client-required", activated_tools=False, semantic_query="",
    ) is None


def test_hidden_gateway_host_pagination_advances_offset_from_schema_defaults():
    tool = _named_tool("plugin__plex__plex_list_library_media")
    tool["function"]["parameters"] = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 100},
            "offset": {"anyOf": [
                {"type": "integer", "default": 0}, {"type": "null"}]},
            "query": {"type": "string"},
        },
    }
    messages = [{
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "plugin__plex__plex_list_library_media",
                "arguments": "{\"query\":\"kids\"}",
            },
        }],
    }, {
        "role": "tool", "tool_call_id": "call_1",
        "content": "{\"movieHasMore\":true}",
    }]
    assert _hidden_gateway_pagination_call(messages, [tool]) == {
        "name": "plugin__plex__plex_list_library_media",
        "arguments": {"query": "kids", "limit": 100, "offset": 100},
    }


def test_hidden_gateway_initial_pagination_defaults_preserve_model_arguments():
    tool = _named_tool("plugin__plex__plex_list_library_media")
    tool["function"]["parameters"] = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 100},
            "offset": {"anyOf": [
                {"type": "integer", "default": 0}, {"type": "null"}]},
            "query": {"type": "string"},
        },
    }
    messages = [{
        "role": "user",
        "content": "Pull the full list from Plex and make sure to paginate.",
    }]
    empty_call = {"function": {
        "name": "plugin__plex__plex_list_library_media",
        "arguments": "{}",
    }}
    assert _hidden_gateway_initial_pagination_defaults(
        messages, [tool], empty_call) == {
            "name": "plugin__plex__plex_list_library_media",
            "arguments": {"limit": 100, "offset": 0},
        }
    authored_call = {"function": {
        "name": "plugin__plex__plex_list_library_media",
        "arguments": "{\"limit\":500}",
    }}
    assert _hidden_gateway_initial_pagination_defaults(
        messages, [tool], authored_call) == {
            "name": "plugin__plex__plex_list_library_media",
            "arguments": {"limit": 500, "offset": 0},
        }
    assert _hidden_gateway_initial_pagination_defaults(
        [{"role": "user", "content": "Search Plex for Dune."}],
        [tool], empty_call) is None


def test_hidden_gateway_host_pagination_falls_back_for_ambiguous_contracts():
    tool = _named_tool("search")
    tool["function"]["parameters"] = {
        "type": "object", "properties": {"cursor": {"type": "string"}}}
    messages = [{
        "role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }],
    }, {
        "role": "tool", "tool_call_id": "call_1",
        "content": "{\"hasMore\":true,\"nextCursor\":\"opaque\"}",
    }]
    assert _hidden_gateway_pagination_call(messages, [tool]) is None


def test_hidden_gateway_does_not_treat_false_or_text_has_more_as_pagination():
    assert _hidden_gateway_force_reason([{
        "role": "tool", "content": json.dumps({"hasMore": False}),
    }]) is None
    assert _hidden_gateway_force_reason([{
        "role": "tool", "content": json.dumps({"hasMore": "true"}),
    }]) is None
    assert _hidden_gateway_force_reason([{
        "role": "tool", "content": "hasMore=true but not JSON",
    }]) is None


def test_hidden_gateway_required_client_or_intent_targets_only_search():
    assert _hidden_gateway_decision_choice(
        "auto", "external-state-inspection") == \
        "specific:vmodel_search_tools"
    assert _hidden_gateway_decision_choice(
        "required", "client-required") == "specific:vmodel_search_tools"
    assert _hidden_gateway_decision_choice("auto", None) == "auto"


def test_hidden_tool_gateway_hard_pins_transcript_tools():
    tools = [_named_tool("workspace_execute"), _named_tool("browser_open")]
    raw = [{
        "type": "function", "name": tool["function"]["name"],
        "description": "", "parameters": {},
    } for tool in tools]
    messages = [{
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "workspace_execute", "arguments": "{}"},
        }],
    }]
    selected, _raw, pinned, _retrieval = _hidden_gateway_catalogs(
        tools, raw, messages, limit=1)
    assert pinned == 1
    assert [tool["function"]["name"] for tool in selected] == ["workspace_execute"]


def test_hidden_search_query_is_not_diluted_by_large_system_prompt():
    from unittest.mock import patch

    tools = [
        _named_tool("workspace_execute", "Execute a shell command."),
        _named_tool("calendar_create", "Create a calendar meeting."),
    ]
    raw = [{
        "type": "function", "name": tool["function"]["name"],
        "description": tool["function"]["description"], "parameters": {},
    } for tool in tools]
    messages = [{
        "role": "system",
        "content": "calendar meeting appointment event " * 5_000,
    }]
    with patch.dict(os.environ, {"VMODEL_TOOL_EMBEDDINGS": "0"}):
        selected, _raw, _pinned, metadata = _hidden_gateway_catalogs(
            tools, raw, messages, query="run a terminal command", limit=1)
    assert [tool["function"]["name"] for tool in selected] == [
        "workspace_execute"]
    assert metadata["tool_embedding_status"] == "unavailable"


def test_glm_fast_mode_enables_quantized_cache_pages():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="glm_moe_dsa", tie_word_embeddings=False,
        index_topk=2048, vision_config=None)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-glm"), "fast")

    assert captured[0].quant_bits == 4
    assert captured[0].max_weight_cache_mb == 5000


def test_k25_lossless_uses_demand_paging_without_speculative_prefetch():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k25", tie_word_embeddings=False,
        index_topk=0, vision_config=None)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-k25"), "lossless")

    rc = captured[0]
    assert rc.max_weight_cache_mb == 1500
    assert rc.prefetch_depth == 0
    assert rc.stream_lm_head
    assert not rc.pin_lm_head
    assert rc.quant_bits == 0


def test_k3_native_profile_uses_proven_layer_stationary_prefetch_schedule():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k3", tie_word_embeddings=False,
        index_topk=0, vision_config=None)
    with patch.dict(
            "os.environ", {
                "VMODEL_CT_MXFP4_NATIVE": "1",
                "VMODEL_EXPERT_BATCH_PREFETCH": "1",
                "VMODEL_K3_SCALE_SIDECAR_DIR": "/tmp/k3-scales",
            }), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-k3"), "lossless")

    rc = captured[0]
    assert rc.native_ct_mxfp4
    assert rc.kimi_k3_scale_sidecar_dir == "/tmp/k3-scales"
    assert rc.layer_stationary_prefill
    assert rc.prefill_chunk_size == 1
    assert rc.max_weight_cache_mb == 3000
    assert rc.min_weight_cache_mb == 150
    assert rc.prefetch_depth == 1
    assert rc.prefetch_workers == 1
    assert rc.expert_batch_prefetch
    assert rc.prompt_kv_dir == ""
    assert rc.stream_lm_head
    assert not rc.pin_lm_head
    assert not rc.suffix_decoding


def test_k3_long_context_math_candidates_are_explicit_and_forwarded():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k3",
        tie_word_embeddings=False,
        index_topk=0,
        vision_config=None,
    )
    with patch.dict(
        "os.environ",
        {
            "VMODEL_K3_COMPRESSED_MLA": "1",
            "VMODEL_K3_ABSORBED_MLA": "1",
            "VMODEL_K3_MLA_KEY_TILE_SIZE": "1024",
            "VMODEL_K3_FUSED_ATTNRES_TILE_SIZE": "128",
            "VMODEL_K3_DENSE_MLP_TILE_SIZE": "256",
            "VMODEL_K3_PREFILL_TILE_WIDTH": "256",
            "VMODEL_K3_PREFILL_TILE_POLICY": "prompt-length",
            "VMODEL_K3_PREFILL_LONG_CONTEXT_TOKENS": "384",
            "VMODEL_K3_PREFILL_SHORT_TILE_WIDTH": "2",
            "VMODEL_K3_DENSE_MLP_SHORT_TILE_SIZE": "8",
        },
        clear=True,
    ), patch(
        "runtime.config.ModelConfig.from_dir", return_value=cfg
    ), patch(
        "runtime.path_resolver.resolve_model_dir",
        side_effect=lambda path: path,
    ), patch(
        "runtime.engine.StreamingEngine", FakeEngine
    ):
        EngineManager().get(Path("/tmp/fake-k3"), "lossless")

    rc = captured[0]
    assert rc.kimi_k3_compressed_mla
    assert rc.kimi_k3_absorbed_mla
    assert rc.kimi_k3_mla_key_tile_size == 1024
    assert rc.kimi_k3_fused_attnres_tile_size == 128
    assert rc.kimi_k3_dense_mlp_tile_size == 256
    assert rc.prefill_chunk_size == 256
    assert rc.kimi_k3_prefill_tile_policy == "prompt-length"
    assert rc.kimi_k3_prefill_long_context_tokens == 384
    assert rc.kimi_k3_prefill_short_tile_width == 2
    assert rc.kimi_k3_dense_mlp_short_tile_size == 8


def test_k3_suffix_verification_is_explicit_and_bounded():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k3",
        tie_word_embeddings=False,
        index_topk=0,
        vision_config=None,
    )
    with patch.dict(
        "os.environ",
        {
            "VMODEL_K3_SUFFIX_DECODING": "1",
            "VMODEL_K3_SUFFIX_K": "3",
            "VMODEL_K3_SUFFIX_MAX_DEPTH": "7",
            "VMODEL_K3_SUFFIX_FACTOR": "1.5",
            "VMODEL_K3_SUFFIX_MIN_PROBABILITY": "0.8",
            "VMODEL_K3_SUFFIX_MAX_LOCAL_TOKENS": "32768",
        },
        clear=True,
    ), patch(
        "runtime.config.ModelConfig.from_dir", return_value=cfg
    ), patch(
        "runtime.path_resolver.resolve_model_dir",
        side_effect=lambda path: path,
    ), patch(
        "runtime.engine.StreamingEngine", FakeEngine
    ):
        EngineManager().get(Path("/tmp/fake-k3"), "lossless")

    rc = captured[0]
    assert rc.suffix_decoding
    assert rc.suffix_decoding_k == 3
    assert rc.suffix_decoding_max_depth == 7
    assert rc.suffix_decoding_factor == 1.5
    assert rc.suffix_decoding_min_probability == 0.8
    assert rc.suffix_decoding_max_local_tokens == 32768
    assert rc.suffix_decoding_max_nodes == 800_000
    assert rc.suffix_decoding_max_bytes == 256_000_000


def test_k3_absorbed_mla_requires_compressed_mla():
    from unittest.mock import patch

    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(
        "os.environ",
        {"VMODEL_K3_ABSORBED_MLA": "1"},
        clear=True,
    ):
        with pytest.raises(
            RequestValidationError,
            match="requires VMODEL_K3_COMPRESSED_MLA=1",
        ):
            EngineManager().get(Path("/tmp/fake-k3"), "lossless")


def test_k3_scale_sidecar_server_setting_requires_native_mxfp4():
    from unittest.mock import patch

    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(
        "os.environ",
        {
            "VMODEL_CT_MXFP4_NATIVE": "0",
            "VMODEL_K3_SCALE_SIDECAR_DIR": "/tmp/k3-scales",
        },
        clear=True,
    ):
        with pytest.raises(
            RequestValidationError, match="requires VMODEL_CT_MXFP4_NATIVE=1"
        ):
            EngineManager().get(Path("/tmp/fake-k3"), "lossless")


def test_k3_nf12_uncached_profile_is_explicit_and_carried_to_runtime():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="kimi_k3", tie_word_embeddings=False,
        index_topk=0, vision_config=None)
    with patch.dict(
            "os.environ", {
                "VMODEL_CT_MXFP4_NATIVE": "1",
                "VMODEL_K3_NF12_SIDECAR_DIR": "/tmp/k3-nf12",
                "VMODEL_K3_NF12_UNCACHED": "1",
            }, clear=True), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-k3"), "lossless")

    rc = captured[0]
    assert rc.bf16_nf12_sidecar_dir == "/tmp/k3-nf12"
    assert rc.bf16_nf12_uncached_reads


def test_k3_nf12_uncached_setting_requires_sidecar():
    from unittest.mock import patch

    from runtime.server import EngineManager, RequestValidationError

    with patch.dict(
        "os.environ",
        {
            "VMODEL_CT_MXFP4_NATIVE": "1",
            "VMODEL_K3_NF12_UNCACHED": "1",
        },
        clear=True,
    ):
        with pytest.raises(
            RequestValidationError,
            match="requires VMODEL_K3_NF12_SIDECAR_DIR",
        ):
            EngineManager().get(Path("/tmp/fake-k3"), "lossless")


def test_qwen36_profiles_bound_experts_and_use_hybrid_endpoint_cache():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config={"depth": 27}, num_hidden_layers=40)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen36"), "lossless")
        EngineManager().get(Path("/tmp/fake-qwen36"), "fast")

    lossless, fast = captured
    for rc in captured:
        assert rc.prompt_kv_dir == ""
        assert rc.hot_prompt_kv
        assert rc.hot_prompt_kv_slots == 2
        assert rc.hot_prompt_kv_min_tokens == 16
        assert rc.hot_prompt_kv_min_available_mb == 0
        # F95 (2026-07-21): this construction-time pick is now a
        # best-effort DEFAULT only -- StreamingEngine.generate() resamples
        # the same ladder fresh per conversation (see engine.py's
        # hybrid_prefill_chunk_size docstring), so the wider tiers are safe
        # to use again here too. available=8GB -> 512 (the >=4GB tier).
        assert rc.prefill_chunk_size == 512
        assert rc.hot_prompt_kv_chunk_size == rc.prefill_chunk_size
        assert rc.expert_fetch_batch == 16
        assert rc.decode_expert_fetch_batch == 8
        assert rc.layer_stationary_prefill
        assert rc.fast_dirs[0].endswith("vmodel_fast_tier/fake-qwen36")
        assert rc.parallel_storage_reads
        assert not rc.pin_lm_head
        assert rc.stream_lm_head
        # F94 (2026-07-21): pin_first_layers is never set for qwen3_5_moe
        # regardless of available memory -- live-reconfirmed the gated
        # >=4GB bet backfires (a fresh server's construction-time reading
        # doesn't predict what real request traffic drains memory to), same
        # lesson as min_weight_cache_mb/prefill_chunk_size elsewhere in this
        # file. Fully-evictable is the only behavior proven not to make a
        # real failure worse.
        assert rc.pin_first_layers == 0
    assert lossless.quant_bits == 0
    assert lossless.max_weight_cache_mb == 7000
    assert fast.quant_bits == 4
    assert fast.quant_mode == "mxfp4"
    assert not fast.quant_attention
    assert not fast.quant_router
    assert not fast.quant_lm_head
    # The real 35B lossy capture crossed the paging cliff at 5.5/7.0 GB.
    # 5.0 GB completed safely at 7.24 GB Metal; lossless remains unchanged.
    assert fast.max_weight_cache_mb == 5000


def test_qwen36_explicit_quantized_head_replaces_repeated_bf16_stream():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_hidden_layers=40,
        num_experts=256, num_experts_per_tok=8,
        layer_types=[
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention",
        ] * 10,
    )
    with patch.dict(
            "os.environ", {"VMODEL_QWEN35_QUANT_LM_HEAD": "1"}), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen36-qhead"), "fast")

    rc = captured[0]
    assert rc.quant_mode == "mxfp4"
    assert rc.quant_lm_head
    assert rc.pin_lm_head
    assert not rc.stream_lm_head


def test_qwen36_explicit_reranked_head_keeps_exact_candidate_scores():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_hidden_layers=40,
        num_experts=256, num_experts_per_tok=8,
        layer_types=[
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention",
        ] * 10,
    )
    with patch.dict("os.environ", {
            "VMODEL_QWEN35_RERANK_LM_HEAD": "1",
            "VMODEL_QWEN35_RERANK_LM_HEAD_CANDIDATES": "64",
         }), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen36-rerank-head"), "fast")

    rc = captured[0]
    assert not rc.quant_lm_head
    assert rc.rerank_lm_head
    assert rc.rerank_lm_head_candidates == 64
    assert (rc.rerank_lm_head_mode, rc.rerank_lm_head_bits,
            rc.rerank_lm_head_group_size) == ("mxfp4", 4, 32)
    assert rc.pin_lm_head
    assert not rc.stream_lm_head


def test_qwen_native_mtp_auto_policy_targets_only_out_of_core_dense_models():
    from runtime.server import _qwen_native_mtp_policy

    enabled, reason = _qwen_native_mtp_policy(
        "auto", model_type="qwen3_5", num_experts=0,
        payload_bytes=16_800_000_000, cache_bytes=7_000_000_000)
    assert enabled
    assert reason == "auto-out-of-core-dense"

    enabled, reason = _qwen_native_mtp_policy(
        "auto", model_type="qwen3_5", num_experts=0,
        payload_bytes=3_200_000_000, cache_bytes=7_000_000_000)
    assert not enabled
    assert reason == "resident-dense-overhead"

    enabled, reason = _qwen_native_mtp_policy(
        "auto", model_type="qwen3_5_moe", num_experts=256,
        payload_bytes=23_400_000_000, cache_bytes=7_000_000_000)
    assert not enabled
    assert reason == "moe-serial-verifier-not-auto-validated"

    enabled, reason = _qwen_native_mtp_policy(
        "1", model_type="qwen3_5_moe", num_experts=256,
        payload_bytes=23_400_000_000, cache_bytes=7_000_000_000)
    assert enabled
    assert reason == "operator-forced"

    enabled, reason = _qwen_native_mtp_policy(
        "auto", model_type="qwen3_5", num_experts=0,
        payload_bytes=16_800_000_000, cache_bytes=7_000_000_000,
        ngram_enabled=True)
    assert not enabled
    assert reason == "explicit-ngram-selected"


def test_qwen_native_mtp_policy_validates_operator_mode():
    import pytest

    from runtime.server import _qwen_native_mtp_policy

    with pytest.raises(ValueError, match="auto, 0, or 1"):
        _qwen_native_mtp_policy(
            "sometimes", model_type="qwen3_5", num_experts=0,
            payload_bytes=16_800_000_000, cache_bytes=7_000_000_000)


def test_hybrid_prefill_chunk_size_ladder_reinstated_for_per_conversation_use():
    """F94 (2026-07-20/21): a binary 512/128 split was proven insufficient
    live, and the wider ladder was STILL insufficient once this server
    started caching and reusing ONE engine across every subsequent
    request forever -- a healthy available_bytes reading taken once, at a
    server's first-ever request, has no bearing on real memory conditions
    hours later. The 512/128 tiers were removed and 32 became the ceiling
    for everything, regardless of size or reading.

    F95 (2026-07-21): reinstated 512/128, now that this function (moved to
    runtime.engine, see hybrid_prefill_chunk_size there) is called PER
    CONVERSATION instead of once per engine lifetime -- a fresh reading
    only has to stay valid for one conversation's own lifetime, not a
    whole server's. The floor tier (chunk=1) still must stay reachable no
    matter how tight memory gets, so a request degrades to token-by-token
    prefill instead of ever hard-failing purely due to chunk size."""
    from runtime.engine import hybrid_prefill_chunk_size

    assert hybrid_prefill_chunk_size(10_000_000_000) == 512
    assert hybrid_prefill_chunk_size(4_000_000_000) == 512
    assert hybrid_prefill_chunk_size(3_999_999_999) == 128
    assert hybrid_prefill_chunk_size(2_000_000_000) == 128
    assert hybrid_prefill_chunk_size(1_999_999_999) == 32
    assert hybrid_prefill_chunk_size(1_000_000_000) == 32
    assert hybrid_prefill_chunk_size(999_999_999) == 8
    assert hybrid_prefill_chunk_size(500_000_000) == 8
    assert hybrid_prefill_chunk_size(499_999_999) == 1
    assert hybrid_prefill_chunk_size(0) == 1


def test_k3_prompt_prefill_schedule_uses_only_token_boundary():
    from runtime.engine import (
        kimi_k3_prefill_schedule_compatible,
        kimi_k3_prompt_prefill_schedule,
    )

    settings = {
        "policy": "prompt-length",
        "long_context_tokens": 256,
        "short_tile_width": 256,
        "long_tile_width": 256,
        "short_dense_mlp_tile_size": 0,
        "long_dense_mlp_tile_size": 256,
    }
    assert kimi_k3_prompt_prefill_schedule(255, **settings) == (
        256, 0, "short")
    assert kimi_k3_prompt_prefill_schedule(256, **settings) == (
        256, 256, "long")
    assert kimi_k3_prompt_prefill_schedule(
        1,
        policy="fixed",
        long_context_tokens=256,
        short_tile_width=1,
        long_tile_width=128,
        short_dense_mlp_tile_size=0,
        long_dense_mlp_tile_size=64,
    ) == (128, 64, "fixed")
    assert kimi_k3_prefill_schedule_compatible(
        policy="prompt-length",
        active_schedule="short:prefill=256:dense=0",
        cached_schedule="short:prefill=256:dense=0",
    )
    assert not kimi_k3_prefill_schedule_compatible(
        policy="prompt-length",
        active_schedule="long:prefill=256:dense=256",
        cached_schedule="short:prefill=256:dense=0",
    )
    assert kimi_k3_prefill_schedule_compatible(
        policy="fixed", active_schedule="fixed:new", cached_schedule="")


def test_hybrid_prefill_chunk_size_model_scale_no_longer_changes_the_ceiling():
    """model_scale used to lower the ceiling further for large dense models
    specifically, but a direct probe found chunk size barely moves
    _layer_transient for a fixed model size anyway (~7% from 128 to 32) --
    the real lever was always the weight-cache floor. model_scale is now
    an accepted-but-inert parameter, kept only so existing callers don't
    need to change shape. Any value must produce the exact same result as
    omitting it (2026-07-21)."""
    from runtime.engine import hybrid_prefill_chunk_size

    for available in (10_000_000_000, 4_000_000_000, 1_999_999_999,
                      999_999_999, 499_999_999, 0):
        baseline = hybrid_prefill_chunk_size(available)
        for model_scale in (0, 50_000_000, 89_000_000, 1_000_000_000):
            assert hybrid_prefill_chunk_size(
                available, model_scale=model_scale) == baseline


def test_hybrid_min_weight_cache_floor_is_unconditionally_low():
    """Live-reproduced TWICE (2026-07-20): a real Qwen3.6-27B request
    constructed its engine at a healthy available=10.27GB, yet still hit
    "unsafe Metal reservation refused" ~3s later at available=2.80GB --
    a single 30K-token/64-layer dense sweep drained 7.5GB of system memory
    over its OWN lifetime before the failing layer was even reached. A
    construction-time available_bytes reading (gating an earlier version
    of this floor) cannot predict that a long sweep will drain memory
    this much by itself, so the floor must stay conservative regardless
    of what memory looked like at construction -- unlike prefill_chunk_size
    (which F95 now resamples per conversation), nothing requires this
    floor to vary with a point-in-time reading at all; it stays engine-wide."""
    from runtime.engine import hybrid_min_weight_cache_floor_mb

    assert hybrid_min_weight_cache_floor_mb(10_000_000_000) == 64
    assert hybrid_min_weight_cache_floor_mb(4_000_000_000) == 64
    assert hybrid_min_weight_cache_floor_mb(0) == 64


def test_qwen36_never_pins_trunk_and_adapts_chunk_size_under_low_memory():
    """pin_first_layers guarantees the whole trunk stays resident for the
    engine's entire lifetime -- unlike the weight-cache BUDGET, which the
    live governor keeps adapting via reserve()/admissible_units(), there is
    no way to un-pin mid-request. Live-confirmed this backfires under real
    memory pressure TWICE (2026-07-20, then again 2026-07-21 against a real
    Qwen3.6-35B-A3B request even with a memory-gated >=4GB threshold,
    because a fresh server's construction-time reading is comfortable but
    doesn't predict what real request traffic later drains memory to): a
    request failed at a HIGHER active-memory point than pre-pinning
    failures ever reached, because the pinned trunk could no longer be shed
    to make room the way a fully-evictable one could. Pinning is therefore
    never attempted at all now, regardless of available memory at
    construction (F94).

    Same reasoning applies to prefill_chunk_size: every real
    governor.reserve() failure this session traced back to _layer_transient
    (the measured per-chunk compute-scratch high-water mark, which scales
    with chunk size), not expert-fetch bytes. hot_prompt_kv requires FIXED
    chunks (rules out GLM's live adaptive_chunk_size resampling), but a
    smaller fixed chunk is still fixed -- so low memory should also pick a
    smaller constant instead of the generous-memory default."""
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config={"depth": 27}, num_hidden_layers=40)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=2_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen36-tight"), "fast")

    assert captured[0].pin_first_layers == 0
    # lm_head streaming is unconditional -- it costs nothing to keep even
    # when there isn't room to also pin the trunk.
    assert not captured[0].pin_lm_head
    assert captured[0].stream_lm_head
    # F95: this is now just the construction-time default (available=2GB
    # -> 128, the >=2GB tier); StreamingEngine.generate() resamples fresh
    # per conversation instead of trusting this for the engine's lifetime.
    assert captured[0].prefill_chunk_size == 128
    assert captured[0].hot_prompt_kv_chunk_size == 128
    assert captured[0].hot_prompt_kv_min_available_mb == 0
    # F94: the weight-cache floor stays unconditionally low regardless of
    # available memory at construction -- a fixed 1.5GB floor left
    # governor.reserve() with nothing further to shed on the real
    # Qwen3.6-27B failure this was modeled on, even though available
    # memory looked healthy when the engine was constructed.
    assert captured[0].min_weight_cache_mb == 64


def test_qwen36_fast_mode_respects_configured_weight_cache_budget():
    """fast/fast-long mode's quantization block used to unconditionally
    reset max_weight_cache_mb to a literal 6000 right after the env-var
    read above it, silently discarding whatever an operator configured via
    VMODEL_QWEN35_WEIGHT_CACHE_MB (2026-07-20, live-confirmed: a real
    tool-heavy request needed a smaller resident budget to leave headroom
    for the expert-fetch reserve, and this exact mode -- the one a large
    tool count routes to -- ignored the knob meant to free that room)."""
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5_moe", tie_word_embeddings=False,
        index_topk=0, vision_config={"depth": 27}, num_hidden_layers=40)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch.dict(os.environ, {"VMODEL_QWEN35_WEIGHT_CACHE_MB": "2500"}):
        EngineManager().get(Path("/tmp/fake-qwen36-budget"), "lossless")
        EngineManager().get(Path("/tmp/fake-qwen36-budget"), "fast")

    lossless, fast = captured
    assert lossless.max_weight_cache_mb == 2500
    assert fast.max_weight_cache_mb == 2500


def test_dense_fast_mode_uses_validated_mxfp4_and_pipelined_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits, rc.quant_min_dim) == (
        "mxfp4", 32, 4, 0)
    assert rc.quantize_tied_lm_head
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 512
    assert rc.fused_swiglu
    assert rc.stepped_kv_threshold == 512
    assert not rc.embed_rows
    assert rc.min_weight_cache_mb == 1500


def test_dense_fast_gateway_uses_reclaimable_floor_and_2k_kv_boundaries():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    with patch.dict(os.environ, {"VMODEL_FAST_TOOL_GATEWAY": "1"}), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen-gateway"), "fast")

    rc = captured[0]
    assert rc.min_weight_cache_mb == 600
    assert rc.prefill_chunk_size == 512
    assert rc.hot_prompt_kv_chunk_size == 512
    assert rc.adaptive_kv_spill_mb == 256
    assert rc.adaptive_kv_spill_prefill_chunk_size == 512
    assert rc.hot_prompt_kv_min_available_mb == 0


def test_dense_fast_paged_kv_profile_disables_incompatible_hot_paths(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)
    env = {
        "VMODEL_FAST_KV_MAX_MB": "2200",
        "VMODEL_FAST_KV_SPILL_DIR": str(tmp_path / "spill"),
        "VMODEL_FAST_KV_SPILL_COMPRESS": "1",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3"), "fast")

    rc = captured[0]
    assert rc.max_kv_mb == 2200
    assert rc.release_paged_kv_after_generate
    assert rc.prefill_chunk_size == 512
    assert rc.mlx_cache_limit_mb == 64
    assert rc.kv_spill_dir == str(tmp_path / "spill")
    assert rc.kv_spill_compress
    assert not rc.hot_prompt_kv
    assert not rc.tool_pic
    assert not rc.tool_pic_shared_pages
    assert rc.hot_prompt_kv_persist_dir == ""


def test_dense_qwen35_honors_postgen_cleanup_floor():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN35_POSTGEN_MIN_AVAILABLE_MB": "6000",
        "VMODEL_QWEN35_WEIGHT_CACHE_MB": "2200",
        "VMODEL_QWEN35_PREFILL_CHUNK_CEILING": "128",
        "VMODEL_QWEN35_HOT_KV": "1",
    }
    manager = EngineManager()
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        manager.get(Path("/tmp/fake-qwen38-dense"), "lossless")
        os.environ["VMODEL_QWEN35_PREFILL_CHUNK_CEILING"] = "32"
        manager.get(Path("/tmp/fake-qwen38-dense"), "lossless")
        os.environ["VMODEL_QWEN35_HOT_KV"] = "0"
        manager.get(Path("/tmp/fake-qwen38-dense"), "lossless")

    assert len(captured) == 3  # ceiling and hot-state policy are identities
    assert captured[0].qwen_postgen_min_available_mb == 6000
    assert captured[0].max_weight_cache_mb == 2200
    assert captured[0].qwen35_prefill_chunk_ceiling == 128
    assert captured[1].qwen35_prefill_chunk_ceiling == 32
    assert captured[1].hot_prompt_kv
    assert not captured[2].hot_prompt_kv


@pytest.mark.parametrize("value", ["", "2", "true", "bad"])
def test_qwen35_hot_kv_rejects_non_boolean_values(value):
    from unittest.mock import patch

    from runtime.server import EngineManager

    with patch.dict(os.environ, {"VMODEL_QWEN35_HOT_KV": value}):
        with pytest.raises(
                RequestValidationError, match="VMODEL_QWEN35_HOT_KV"):
            EngineManager().get(Path("/tmp/not-opened-invalid-hot-kv"), "fast")


def test_dense_qwen35_paged_durable_prefix_is_explicit_and_disk_only(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN35_KV_MAX_MB": "64",
        "VMODEL_QWEN35_KV_PREFILL_CHUNK_SIZE": "32",
        "VMODEL_QWEN35_KV_SPILL_DIR": str(tmp_path / "spill"),
        "VMODEL_QWEN35_HOT_KV_PERSIST_DIR": str(tmp_path / "journal"),
        "VMODEL_QWEN35_PAGED_KV_PERSIST": "1",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen38-paged-durable"), "lossless")

    rc = captured[0]
    assert rc.max_kv_mb == 64
    assert rc.paged_kv_persist
    assert rc.hot_prompt_kv
    assert rc.release_paged_kv_after_generate
    assert rc.hot_prompt_kv_persist_dir == str(tmp_path / "journal")
    assert rc.kv_spill_dir == str(tmp_path / "spill")
    assert rc.prefill_chunk_size == rc.hot_prompt_kv_chunk_size == 32
    assert not rc.qwen_fused_boundary_scaffold_prefill


@pytest.mark.parametrize("value", ["-1", "2", "64", "513", "auto", "bad"])
def test_qwen35_prefill_chunk_ceiling_rejects_non_ladder_values(value):
    from unittest.mock import patch

    from runtime.server import EngineManager

    with patch.dict(os.environ, {
            "VMODEL_QWEN35_PREFILL_CHUNK_CEILING": value,
         }):
        with pytest.raises(
                RequestValidationError,
                match="VMODEL_QWEN35_PREFILL_CHUNK_CEILING"):
            EngineManager().get(Path("/tmp/not-opened-invalid-ceiling"), "fast")


def test_dense_qwen35_wires_mixed_depth_suffix_and_disables_durable_store():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL": "16:1024",
        "VMODEL_QWEN35_HOT_KV_PERSIST_DIR": "/tmp/must-not-be-used",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen38-dense-suffix"), "fast")

    rc = captured[0]
    assert rc.qwen_lossy_suffix_prefill_early_layers == 16
    assert rc.qwen_lossy_suffix_prefill_prefix_tokens == 0
    assert rc.qwen_lossy_suffix_prefill_tokens == 1024
    assert rc.layer_stationary_prefill
    assert rc.hot_prompt_kv_persist_dir == ""


def test_dense_qwen35_wires_explicit_mixed_depth_durable_store(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    persist = tmp_path / "mixed"
    env = {
        "VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL": "16:1024",
        "VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST": "1",
        "VMODEL_QWEN35_HOT_KV_PERSIST_DIR": str(persist),
    }
    manager = EngineManager()
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        manager.get(Path("/tmp/fake-qwen38-dense-mixed-persist"), "fast")
        os.environ["VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST"] = "0"
        manager.get(Path("/tmp/fake-qwen38-dense-mixed-persist"), "fast")

    assert len(captured) == 2
    assert captured[0].hot_prompt_kv_persist_dir == str(persist)
    assert captured[0].hot_prompt_kv_persist_max_checkpoints == 4
    assert captured[0].hot_prompt_kv_persist_max_mb == 2048
    assert captured[1].hot_prompt_kv_persist_dir == ""


def test_dense_qwen35_wires_explicit_pin_and_prefetch_ladder():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    env = {
        "VMODEL_QWEN35_PIN_LM_HEAD": "1",
        "VMODEL_QWEN35_PIN_TRUNK_BUDGET_MB": "1000",
        "VMODEL_QWEN35_PREFETCH_DEPTH": "3",
        "VMODEL_QWEN35_WEIGHT_CACHE_MB": "2200",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-qwen38-dense-pin"), "fast")

    rc = captured[0]
    assert rc.pin_lm_head and not rc.stream_lm_head
    assert rc.pin_trunk_budget_mb == 1000
    assert rc.prefetch_depth == 3
    assert rc.max_weight_cache_mb == 2200


def test_dense_huihui_rerank64_uses_explicit_row_paged_exact_source():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    fingerprint = "a" * 64
    env = {
        "VMODEL_QWEN35_RERANK_LM_HEAD": "1",
        "VMODEL_QWEN35_RERANK_LM_HEAD_CANDIDATES": "64",
        "VMODEL_QWEN35_RERANK_LM_HEAD_SOURCE": "/exact/huihui",
        "VMODEL_QWEN35_RERANK_LM_HEAD_SOURCE_FINGERPRINT": fingerprint,
        "VMODEL_QWEN35_RERANK_LM_HEAD_RECALL_PROBE_EVERY": "7",
        "VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE": (
            "/tmp/huihui-authoritative-ranks.jsonl"),
        "VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE_MAX_POSITIONS": "1100",
        "VMODEL_QWEN35_RERANK_LM_HEAD_RANK_CAPTURE_MAX_PER_REQUEST": "64",
    }
    with patch.dict(os.environ, env), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        EngineManager().get(Path("/tmp/fake-huihui-all-mxfp4"), "fast")

    rc = captured[0]
    assert rc.rerank_lm_head
    assert rc.rerank_lm_head_candidates == 64
    assert rc.rerank_lm_head_source == "/exact/huihui"
    assert rc.rerank_lm_head_source_fingerprint == fingerprint
    assert rc.rerank_lm_head_recall_probe_every == 7
    assert (rc.rerank_lm_head_rank_capture_path
            == "/tmp/huihui-authoritative-ranks.jsonl")
    assert rc.rerank_lm_head_rank_capture_max_positions == 1100
    assert rc.rerank_lm_head_rank_capture_max_per_request == 64
    assert rc.pin_lm_head and rc.quant_lm_head
    assert not rc.stream_lm_head


def test_dense_huihui_rerank_fails_closed_without_exact_source():
    from unittest.mock import patch

    from runtime.server import EngineManager

    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    with patch.dict(os.environ, {
            "VMODEL_QWEN35_RERANK_LM_HEAD": "1",
         }), \
         patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        with pytest.raises(ValueError, match="requires.*SOURCE"):
            EngineManager().get(
                Path("/tmp/fake-huihui-missing-source"), "fast")


def test_qwen3_fast_resident_kv_projection_rejects_real_harness_scale():
    engine = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="qwen3", vision_config=None, num_experts=0,
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128),
        rc=SimpleNamespace(max_kv_mb=0),
    )
    safe = _fast_dense_resident_kv_projection(engine, "fast", 10_774, 16)
    assert safe["bytes_per_token"] == 147_456
    assert safe["projected_bytes"] < safe["limit_bytes"]
    assert safe["positions"] == 10_774
    assert safe["declared_positions"] == 10_790
    assert safe["declared_projected_bytes"] > safe["projected_bytes"]

    try:
        _validate_fast_dense_resident_kv(engine, "fast", 28_307, 64)
    except RequestValidationError as error:
        message = str(error)
        assert "resident BF16 KV projection" in message
        assert "VMODEL_FAST_TOOL_GATEWAY=1" in message
        assert "VMODEL_FAST_TOOL_LIMIT=32" in message
        assert "quarantined" in message
    else:
        raise AssertionError("unsafe real-harness resident KV was accepted")

    engine.rc.adaptive_kv_spill_mb = 256
    adaptive = _validate_fast_dense_resident_kv(
        engine, "fast", 28_307, 64)
    assert adaptive["adaptive_spill_required"] == 1
    assert adaptive["adaptive_spill_mb"] == 256

    engine.rc.max_kv_mb = 2200
    assert _validate_fast_dense_resident_kv(
        engine, "fast", 28_307, 64) is None


def test_dense_kv_preflight_subtracts_evictable_retained_state():
    base = dict(
        active_metal_bytes=7_830_000_000,
        retained_prompt_kv_bytes=1_680_000_000,
        orphan_prompt_kv_bytes=0,
        evictable_prompt_kv_bytes=1_680_000_000,
        hot_prompt_slots=1,
        metal_ceiling_bytes=9_050_000_000,
    )
    engine = SimpleNamespace(
        cfg=SimpleNamespace(
            model_type="qwen3", vision_config=None, num_experts=0,
            num_hidden_layers=36, num_key_value_heads=8, head_dim=128),
        rc=SimpleNamespace(max_kv_mb=0),
        prompt_cache_memory_snapshot=lambda: base,
    )
    projection = _validate_fast_dense_resident_kv(
        engine, "fast", 10_453, 4_096)
    assert projection["retained_prompt_kv_bytes"] == 1_680_000_000
    assert projection["dynamic_projected_bytes"] < 9_050_000_000

    engine.prompt_cache_memory_snapshot = lambda: {
        **base,
        "retained_prompt_kv_bytes": 0,
        "evictable_prompt_kv_bytes": 0,
    }
    try:
        _validate_fast_dense_resident_kv(engine, "fast", 10_453, 4_096)
    except RequestValidationError as error:
        assert "live dense-Qwen Metal projection" in str(error)
        assert "before generation" in str(error)
    else:
        raise AssertionError("live projection ignored retained-cache pressure")


def test_resident_qwen35_projects_only_growing_full_attention_kv():
    engine = SimpleNamespace(
        backend_name="mlx-lm",
        cfg=SimpleNamespace(
            model_type="qwen3_5", vision_config={"model_type": "qwen3_5"},
            num_experts=0, num_hidden_layers=32,
            full_attention_interval=4, num_key_value_heads=4, head_dim=256),
        rc=SimpleNamespace(max_kv_mb=0, adaptive_kv_spill_mb=256),
        prompt_cache_memory_snapshot=lambda: {
            "active_metal_bytes": 6_300_000_000,
            "retained_prompt_kv_bytes": 0,
            "orphan_prompt_kv_bytes": 0,
            "evictable_prompt_kv_bytes": 0,
            "metal_ceiling_bytes": 8_300_000_000,
        },
    )

    measured_shape = _validate_fast_dense_resident_kv(
        engine, "fast", 14_375, 16)
    assert measured_shape["bytes_per_token"] == 32_768
    assert measured_shape["projected_bytes"] == 471_040_000
    assert measured_shape["dynamic_projected_bytes"] == 7_171_040_000

    try:
        _validate_fast_dense_resident_kv(engine, "fast", 60_000, 16)
    except RequestValidationError as error:
        assert "live dense-Qwen Metal projection" in str(error)
    else:
        raise AssertionError("unsafe resident Qwen3.5 context was accepted")


def test_cache_phase_telemetry_keeps_hidden_phases_separate():
    decision = _cache_phase_telemetry("gateway_decision", {
        "prompt_tokens": 2_000,
        "prefill_s": 0.25,
        "path_stats": {
            "prompt_cache_namespace": "gateway_decision",
            "prompt_cache_prefix_tokens": 1_900,
            "prompt_cache_source": "hot_disk",
            "prompt_cache_exact_hit": 0,
        },
    })
    execution = _cache_phase_telemetry("gateway_execution", {
        "prompt_tokens": 10_000,
        "prefill_s": 42.0,
        "execution_profile": {"schema_version": 1, "level": "layers"},
        "path_stats": {
            "prompt_cache_namespace": "gateway_execution",
            "prompt_cache_prefix_tokens": 0,
            "prompt_cache_source": "cold",
            "tool_pic_reused_tokens": 128,
            "tool_pic_selected_tokens": 9_872,
            "hot_prompt_admission_evicted_slots": 1,
            "hot_prompt_admission_evicted_bytes": 1_500_000_000,
        },
    })
    assert decision["cached_tokens"] == 1_900
    assert decision["cache_source"] == "hot_disk"
    assert execution["cached_tokens"] == 0
    assert execution["effective_reused_tokens"] == 128
    assert execution["admission_evicted_bytes"] == 1_500_000_000
    assert execution["execution_profile"]["level"] == "layers"


def test_responses_stream_emits_terminal_failure_instead_of_truncated_sse():
    import io

    handler = Handler.__new__(Handler)
    handler.wfile = io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None
    handler._sampling = SimpleNamespace()
    handler._constraint = None

    class Engine:
        cfg = SimpleNamespace(model_type="qwen3")

        def __init__(self):
            self.cleaned = 0

        def discard_failed_request_state(self):
            self.cleaned += 1

    engine = Engine()

    def fail(_on_token, _on_progress):
        raise MemoryError("projected working set exceeds ceiling")

    handler._stream_responses(
        "prompt", 64, [], engine, [], lambda *_args: {},
        "resp_test", "Qwen3-4B", 1, None, None, None, [],
        "msg_test", "auto", False, generate_fn=fail)

    wire = handler.wfile.getvalue().decode()
    assert statuses == [200]
    assert '"type": "response.failed"' in wire
    assert '"code": "server_memory_error"' in wire
    assert '"status": "failed"' in wire
    assert engine.cleaned == 1


def test_failed_request_cleanup_releases_only_failed_kv():
    from runtime.engine import StreamingEngine

    class State:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    failed = State()
    survivor = State()
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.last_kv = failed
    engine._hot_prompt_slots = [
        SimpleNamespace(kv=survivor),
        SimpleNamespace(kv=failed),
    ]
    engine._h_window = object()
    engine._h_last = object()
    engine._provisional = object()

    engine.discard_failed_request_state()

    assert failed.releases == 1
    assert survivor.releases == 0
    assert [slot.kv for slot in engine._hot_prompt_slots] == [survivor]
    assert engine.last_kv is None
    assert engine._h_window is None
    assert engine._h_last is None
    assert engine._provisional is None


def test_interrupted_prefill_retains_only_complete_exact_chunk():
    from runtime.engine import KVCache, StreamingEngine

    class State(KVCache):
        def __init__(self, offset):
            self._offset = offset
            self.releases = 0

        @property
        def offset(self):
            return self._offset

        def release(self):
            self.releases += 1

    survivor = State(128)
    partial = State(4096)
    engine = StreamingEngine.__new__(StreamingEngine)
    engine.rc = SimpleNamespace(
        hot_prompt_kv=True, max_kv_mb=0,
        hot_prompt_kv_min_tokens=2048, hot_prompt_kv_slots=1,
        prefill_chunk_size=4096)
    engine._hot_kv_persist = None
    engine._hot_prompt_slots = [SimpleNamespace(kv=survivor)]
    engine.last_kv = partial
    engine._h_window = object()
    engine._h_last = object()
    tokens = list(range(8192))
    capsules = (("inside", 100, 200), ("crosses", 4000, 4200))

    assert engine._retain_interrupted_prefill(
        tokens, partial, 4096, capsules)

    assert survivor.releases == 1
    assert len(engine._hot_prompt_slots) == 1
    slot = engine._hot_prompt_slots[0]
    assert slot.kv is partial
    assert slot.tokens == tuple(tokens[:4096])
    assert slot.logits is None and slot.prompt_logits is None
    assert slot.reusable_prefix == 4096
    # F95: records whatever chunk size was actually driving this
    # interrupted request, so a retry resumes with the same value.
    assert slot.chunk_size == 4096
    assert slot.tool_capsules == (("inside", 100, 200),)
    assert engine._h_window is None and engine._h_last is None


def test_vision_fast_mode_quantizes_only_quality_gated_text_mlp():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_vl", tie_word_embeddings=True,
        index_topk=0, vision_config={"depth": 24}, num_experts=0,
        hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False,
    )
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3-vl"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits) == (
        "mxfp4", 32, 4)
    assert rc.quant_mlp
    assert not rc.quant_attention
    assert not rc.quant_lm_head
    assert not rc.quantize_tied_lm_head
    assert rc.resident_fast_decode
    assert rc.vision_max_patches == 1024
    assert rc.tool_pic


def test_small_lossless_vision_model_uses_exact_resident_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen3_vl", tie_word_embeddings=True,
        index_topk=0, vision_config={"depth": 24}, num_experts=0,
        hidden_size=2048, intermediate_size=6144,
        num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False,
    )
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=4_000_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen3-vl"), "lossless")

    rc = captured[0]
    assert rc.quant_bits == 0
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 2048
    assert not rc.embed_rows
    assert not rc.fused_swiglu
    assert not rc.hot_prompt_kv


def test_small_dense_lossless_mode_uses_exact_resident_decode():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12,
        num_key_value_heads=2, head_dim=128, vocab_size=151936,
        attention_bias=True)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-qwen-1.5b"), "lossless")

    rc = captured[0]
    assert rc.quant_bits == 0
    assert rc.resident_fast_decode
    assert rc.resident_fast_prefill_limit == 2048
    assert not rc.fused_swiglu
    assert not rc.embed_rows
    assert rc.stepped_kv_threshold == 2048
    assert rc.hot_prompt_kv
    assert rc.hot_prompt_kv_slots == 1
    assert rc.hot_prompt_kv_min_tokens == 2048
    assert rc.prompt_kv_min_tokens == 2048


def test_generic_moe_fast_mode_quantizes_experts_but_preserves_sensitive_trunk():
    from unittest.mock import patch

    from runtime.server import EngineManager

    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(Path("/tmp/fake-olmoe"), "fast")

    rc = captured[0]
    assert (rc.quant_mode, rc.quant_group_size, rc.quant_bits) == ("mxfp4", 32, 4)
    assert rc.quant_mlp
    assert not rc.quant_attention
    assert not rc.quant_router
    assert not rc.quant_lm_head
    assert not rc.resident_fast_decode
    assert rc.resident_moe_decode
    assert rc.fused_swiglu
    assert rc.rerank_lm_head
    assert rc.rerank_lm_head_candidates == 32
    assert (rc.rerank_lm_head_mode, rc.rerank_lm_head_bits,
            rc.rerank_lm_head_group_size) == ("affine", 2, 64)
    assert rc.resident_attention_mode == "mxfp8"
    assert rc.resident_attention_bits == 8
    assert rc.stepped_kv_threshold == 1
    assert not rc.embed_rows
    assert rc.prefill_chunk_size == 2048
    assert rc.prefill_last_token_separate
    assert rc.tool_pic
    assert rc.expert_top_k_by_layer == ()


def test_lossless_olmoe_expands_exact_cache_when_governor_admits_it():
    from unittest.mock import patch

    from runtime.server import EngineManager

    made = []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc, self.closes = rc, 0
            self.cache = SimpleNamespace(max_bytes=15_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=13_840_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        engine = EngineManager().get(Path("/tmp/fake-olmoe"), "lossless")

    assert engine is made[0]
    assert len(made) == 1
    assert made[0].rc.max_weight_cache_mb == 14_809
    assert made[0].rc.quant_bits == 0
    assert not made[0].rc.resident_moe_decode


def test_lossless_olmoe_rebuilds_streamed_cache_when_admission_fails():
    from unittest.mock import patch

    from runtime.server import EngineManager

    made = []

    class FakeEngine:
        def __init__(self, _path, rc):
            self.rc, self.closes = rc, 0
            self.cache = SimpleNamespace(max_bytes=8_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._checkpoint_payload_bytes",
               return_value=13_840_000_000), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        engine = EngineManager().get(Path("/tmp/fake-olmoe"), "lossless")

    assert engine is made[1]
    assert len(made) == 2
    assert made[0].closes == 1
    assert made[0].rc.max_weight_cache_mb == 14_809
    assert made[1].rc.max_weight_cache_mb == 6000


def test_dense_embedding_pin_estimate_keeps_large_models_row_paged():
    from runtime.server import (
        _dense_fast_resident_bytes, _dense_lossless_resident_bytes)

    qwen_14b = SimpleNamespace(
        hidden_size=5120, intermediate_size=13824,
        num_hidden_layers=48, num_attention_heads=40,
        num_key_value_heads=8, head_dim=128, vocab_size=152064,
        attention_bias=True, tie_word_embeddings=False)
    assert _dense_fast_resident_bytes(qwen_14b) > int(7_000_000_000 * 0.85)

    qwen_1_5b = SimpleNamespace(
        hidden_size=1536, intermediate_size=8960,
        num_hidden_layers=28, num_attention_heads=12,
        num_key_value_heads=2, head_dim=128, vocab_size=151936,
        attention_bias=True, tie_word_embeddings=True)
    qwen_7b = SimpleNamespace(
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True, tie_word_embeddings=False)
    assert _dense_lossless_resident_bytes(qwen_1_5b) < int(6_000_000_000 * 0.85)
    assert _dense_lossless_resident_bytes(qwen_7b) > int(6_000_000_000 * 0.85)


def test_lossless_qwen_discovers_only_same_family_complete_mxfp4_draft(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen2.5-7B-Instruct"
    preferred = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    wrong_variant = tmp_path / "Qwen2.5-0.5B-Base-mlx-mxfp4"
    unvalidated_size = tmp_path / "Qwen2.5-0.5B-Instruct-mlx-mxfp4"
    for path in (target, preferred, wrong_variant, unvalidated_size):
        path.mkdir()
    common = {
        "model_type": "qwen2", "vision_config": None, "num_experts": 0,
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
    }
    (preferred / "config.json").write_text(json.dumps({
        **common, "hidden_size": 1536}))
    (preferred / "model.safetensors").write_bytes(b"complete")
    (wrong_variant / "config.json").write_text(json.dumps({
        **common, "hidden_size": 896}))
    (wrong_variant / "model.safetensors").write_bytes(b"complete")
    (unvalidated_size / "config.json").write_text(json.dumps({
        **common, "hidden_size": 896}))
    (unvalidated_size / "model.safetensors").write_bytes(b"complete")

    cfg = SimpleNamespace(hidden_size=3584)
    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        assert _speculative_draft_for(target, cfg) == preferred.resolve()


def test_speculative_draft_can_be_disabled_or_explicitly_overridden(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen2.5-7B-Instruct"
    draft = tmp_path / "custom-draft"
    target.mkdir()
    draft.mkdir()
    (draft / "config.json").write_text("{}")
    (draft / "model.safetensors").write_bytes(b"complete")
    cfg = SimpleNamespace(hidden_size=3584)

    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "off"}):
        assert _speculative_draft_for(target, cfg) is None
    with patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": str(draft)}):
        assert _speculative_draft_for(target, cfg) == draft.resolve()


def test_qwen3_dspark_discovers_only_shape_compatible_block7(tmp_path):
    from unittest.mock import patch

    target = tmp_path / "Qwen3-4B"
    good = tmp_path / "dspark_qwen3_4b_block7"
    wrong = tmp_path / "dspark_qwen3_8b_block7"
    for path in (target, good, wrong):
        path.mkdir()
    common = {
        "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3",
        "vocab_size": 151936, "num_target_layers": 36, "block_size": 7,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }
    (good / "config.json").write_text(json.dumps({
        **common, "hidden_size": 2560}))
    (good / "model.safetensors").write_bytes(b"complete")
    (wrong / "config.json").write_text(json.dumps({
        **common, "hidden_size": 4096}))
    (wrong / "model.safetensors").write_bytes(b"complete")
    cfg = SimpleNamespace(hidden_size=2560, vocab_size=151936,
                          num_hidden_layers=36)

    with patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "auto"}):
        assert _dspark_draft_for(target, cfg) == good.resolve()
    with patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "off"}):
        assert _dspark_draft_for(target, cfg) is None


def test_engine_manager_wraps_streamed_lossless_qwen3_with_dspark(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen3-4B"
    draft_path = tmp_path / "dspark_qwen3_4b_block7"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            # Simulate a governor-constrained host: the fitted target cache is
            # below the exact 4B footprint, so DSpark remains useful.
            self.cache = SimpleNamespace(max_bytes=6_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeDSparkEngine:
        def __init__(self, target, draft_dir, *, max_draft_tokens,
                     max_prompt_tokens, confidence_threshold,
                     prompt_cache_min_tokens, context_window_tokens):
            self.target = target
            self.draft_dir = Path(draft_dir)
            self.max_draft_tokens = max_draft_tokens
            self.max_prompt_tokens = max_prompt_tokens
            self.confidence_threshold = confidence_threshold
            self.prompt_cache_min_tokens = prompt_cache_min_tokens
            self.context_window_tokens = context_window_tokens
            self.closes = 0

        def close(self):
            self.closes += 1
            self.target.close()

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine", FakeDSparkEngine), \
         patch.dict("os.environ", {
             "VMODEL_DSPARK_DRAFT": "auto",
             "VMODEL_DSPARK_MAX_DRAFT_TOKENS": "4",
             "VMODEL_DSPARK_MAX_PROMPT_TOKENS": "2048",
         }):
        wrapped = manager.get(target_path, "lossless")
        assert isinstance(wrapped, FakeDSparkEngine)
        assert wrapped.draft_dir == draft_path
        assert wrapped.max_draft_tokens == 4
        assert wrapped.max_prompt_tokens == 2048
        assert wrapped.confidence_threshold == 0.0
        assert wrapped.prompt_cache_min_tokens == 2048
        assert len(made) == 1
        assert made[0].rc.max_weight_cache_mb > 9000
        assert made[0].rc.prefetch_workers == 2
        assert made[0].rc.prefetch_depth == 4
        assert made[0].rc.hot_prompt_kv
        assert made[0].rc.hot_prompt_kv_min_tokens == 2048

        manager.get(target_path, "fast")

    assert wrapped.closes == 1
    assert wrapped.target.closes == 1


def test_engine_manager_exposes_k3_dspark_only_by_explicit_path(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Kimi-K3"
    draft_path = tmp_path / "Kimi-K3-DSpark-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="kimi_k3", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=896,
        hidden_size=7168, intermediate_size=33792,
        num_hidden_layers=93, num_attention_heads=64,
        num_key_value_heads=64, head_dim=192, vocab_size=163840,
        attention_bias=False)

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=3_000_000_000)
            self.governor = None

        def close(self):
            self.closes += 1

    class FakeDSparkEngine:
        def __init__(self, target, draft_dir, *, max_draft_tokens,
                     max_prompt_tokens, confidence_threshold,
                     prompt_cache_min_tokens, context_window_tokens):
            self.target = target
            self.draft_dir = Path(draft_dir)
            self.max_draft_tokens = max_draft_tokens
            self.max_prompt_tokens = max_prompt_tokens

        def close(self):
            self.target.close()

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for",
               return_value=draft_path) as discover, \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine",
               FakeDSparkEngine), \
         patch.dict("os.environ", {
             "VMODEL_DSPARK_DRAFT": str(draft_path),
             "VMODEL_DSPARK_MAX_DRAFT_TOKENS": "6",
         }):
        wrapped = manager.get(target_path, "lossless")

    assert wrapped.draft_dir == draft_path
    assert wrapped.max_draft_tokens == 6
    assert wrapped.max_prompt_tokens == 1_048_576
    discover.assert_called_once()


def test_engine_manager_exposes_qwen38_dspark_only_by_explicit_path(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Huihui-Qwen3.8-27B"
    draft_path = tmp_path / "Qwen3.8-27B-DSpark-Agentic-4bit"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=2_000_000_000)
            self.governor = None

        def close(self):
            self.closes += 1

    class FakeDSparkEngine:
        def __init__(self, target, draft_dir, **kwargs):
            self.target = target
            self.draft_dir = Path(draft_dir)
            self.kwargs = kwargs
            self.closes = 0
            made.append(self)

        def close(self):
            self.closes += 1
            self.target.close()

    manager = EngineManager()
    env = {
        "VMODEL_DSPARK_DRAFT": str(draft_path),
        "VMODEL_DSPARK_MAX_DRAFT_TOKENS": "2",
        "VMODEL_DSPARK_MAX_PROMPT_TOKENS": "262144",
        "VMODEL_QWEN_MTP_SPECULATIVE": "0",
    }
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for",
               return_value=draft_path) as discover, \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine", FakeDSparkEngine), \
         patch.dict("os.environ", env), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        first = manager.get(target_path, "fast")
        assert first.kwargs["max_draft_tokens"] == 2
        assert first.kwargs["max_prompt_tokens"] == 262144
        assert first.kwargs["prompt_cache_min_tokens"] == 2048
        assert first.kwargs["context_window_tokens"] == 1024
        assert first.kwargs["release_between_sweeps"] is True
        assert first.kwargs["drafter_load_margin_bytes"] == 64_000_000
        assert first.target.rc.max_weight_cache_mb == 2200
        assert first.target.rc.prefetch_workers == 2
        assert first.target.rc.prefetch_depth == 4

        # Drafter lifetime is part of engine identity; disabling the Qwen
        # default must not reuse a release-enabled engine.
        os.environ["VMODEL_DSPARK_RELEASE_BETWEEN_SWEEPS"] = "0"
        second = manager.get(target_path, "fast")
        assert "release_between_sweeps" not in second.kwargs

        # The rest of the DSpark settings remain identity inputs as well.
        os.environ["VMODEL_DSPARK_MAX_DRAFT_TOKENS"] = "3"
        third = manager.get(target_path, "fast")

    assert len(made) == 3
    assert third.kwargs["max_draft_tokens"] == 3
    assert first.closes == 1
    assert discover.call_count == 3


def test_engine_manager_wraps_qwen38_with_explicit_dflash2_and_keys_policy(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Huihui-Qwen3.8-27B"
    draft_path = tmp_path / "Qwen3.8-27B-DFlash2-mlx-affine4-g64"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3_5", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=5120, intermediate_size=17408,
        num_hidden_layers=64, num_attention_heads=24,
        num_key_value_heads=4, head_dim=256, vocab_size=248320,
        attention_bias=False, layer_types=(
            "linear_attention", "linear_attention", "linear_attention",
            "full_attention") * 16,
    )
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=2_000_000_000)
            self.governor = None

        def close(self):
            self.closes += 1

    class FakeDFlash2Engine:
        def __init__(self, target, draft_dir, **kwargs):
            self.target = target
            self.draft_dir = Path(draft_dir)
            self.kwargs = kwargs
            self.closes = 0
            made.append(self)

        def close(self):
            self.closes += 1
            self.target.close()

    manager = EngineManager()
    env = {
        "VMODEL_QWEN_DFLASH2_DRAFT": str(draft_path),
        "VMODEL_QWEN_DFLASH2_MAX_DRAFT_TOKENS": "4",
        "VMODEL_QWEN_DFLASH2_MAX_PROMPT_TOKENS": "262144",
        "VMODEL_QWEN_DFLASH2_PROMPT_CACHE_MIN_TOKENS": "0",
        "VMODEL_QWEN_DFLASH2_PROPOSAL_POLICY": "unary",
        "VMODEL_QWEN_DFLASH2_RELEASE_BETWEEN_SWEEPS": "1",
        "VMODEL_QWEN_DFLASH2_FUSED_DYNAMIC_CONV": "1",
        "VMODEL_QWEN_DFLASH2_ABLATION_DIRECTION": str(
            tmp_path / "direction"),
        "VMODEL_QWEN_DFLASH2_ABLATION_STRENGTH": "0.75",
        "VMODEL_QWEN_DFLASH2_LOAD_MARGIN_MB": "400",
        "VMODEL_QWEN_DFLASH2_NATIVE_MTP_FALLBACK": "1",
        "VMODEL_QWEN_DFLASH2_FALLBACK_MIN_ROUNDS": "5",
        "VMODEL_QWEN_DFLASH2_FALLBACK_MIN_ACCEPTED_PER_ROUND": "1.25",
        "VMODEL_QWEN_DFLASH2_TREE_BUDGET": "8",
        "VMODEL_QWEN_MTP_SPECULATIVE": "1",
    }
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir",
               side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for", return_value=None), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dflash2_adapter.DFlash2SpeculativeEngine",
               FakeDFlash2Engine), \
         patch.dict("os.environ", env), \
         patch("runtime.server.psutil.virtual_memory",
               return_value=SimpleNamespace(available=8_000_000_000)):
        first = manager.get(target_path, "fast")
        assert first.draft_dir == draft_path
        assert first.kwargs == {
            "max_draft_tokens": 4,
            "max_prompt_tokens": 262144,
            "prompt_cache_min_tokens": 0,
            "release_between_sweeps": True,
            "drafter_load_margin_bytes": 400_000_000,
            "proposal_policy": "unary",
            "fused_dynamic_conv": True,
            "ablation_direction_dir": str(tmp_path / "direction"),
            "ablation_strength": 0.75,
            "native_mtp_fallback": True,
            "fallback_min_dflash_rounds": 5,
            "fallback_min_accepted_per_round": 1.25,
            "tree_budget": 8,
        }
        os.environ["VMODEL_QWEN_DFLASH2_PROPOSAL_POLICY"] = "selector"
        second = manager.get(target_path, "fast")

    assert len(made) == 2
    assert first.closes == 1
    assert first.target.closes == 1
    assert second.kwargs["proposal_policy"] == "selector"


def test_engine_manager_prefers_full_resident_qwen3_when_governor_admits_it(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen3-4B"
    draft_path = tmp_path / "dspark_qwen3_4b_block7"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen3", tie_word_embeddings=True,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, vocab_size=151936,
        attention_bias=False)

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=10_000_000_000)
            self.governor = None

        def close(self):
            self.closes += 1

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._dspark_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.dspark.DSparkSpeculativeEngine",
               side_effect=AssertionError("resident target should win")), \
         patch.dict("os.environ", {"VMODEL_DSPARK_DRAFT": "auto"}):
        engine = manager.get(target_path, "lossless")

    assert isinstance(engine, FakeEngine)
    assert engine.rc.resident_fast_decode
    assert engine.rc.resident_fast_prefill_limit == 2048
    assert engine.rc.stepped_kv_threshold == 2048
    assert engine.rc.hot_prompt_kv
    assert engine.rc.hot_prompt_kv_min_tokens == 2048


def test_engine_manager_wraps_large_lossless_qwen_and_swaps_both_owners(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            # Simulate the constrained target machine: a model-sized probe is
            # fitted below the complete exact 7B footprint.
            self.cache = SimpleNamespace(max_bytes=6_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeSpeculativeEngine:
        def __init__(self, target, draft, *, k, max_prompt_tokens,
                     prompt_cache_min_tokens):
            self.target, self.draft = target, draft
            self.k, self.max_prompt_tokens, self.closes = k, max_prompt_tokens, 0
            self.prompt_cache_min_tokens = prompt_cache_min_tokens

        def close(self):
            self.closes += 1
            self.draft.close()
            self.target.close()

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine", FakeSpeculativeEngine), \
         patch.dict("os.environ", {
             "VMODEL_SPECULATIVE_DRAFT": "auto",
             "VMODEL_SPECULATIVE_K": "6",
             "VMODEL_SPECULATIVE_MAX_PROMPT_TOKENS": "2048",
         }):
        wrapped = manager.get(target_path, "lossless")
        assert isinstance(wrapped, FakeSpeculativeEngine)
        assert wrapped.prompt_cache_min_tokens == 2048
        assert len(made) == 3
        assert made[0].rc.max_weight_cache_mb > 16_000
        assert made[0].closes == 1
        assert made[1].path == target_path
        assert made[1].rc.max_weight_cache_mb == 6000
        assert made[1].rc.prefetch_workers == 2
        assert made[1].rc.prefetch_depth == 4
        assert made[2].path == draft_path
        assert made[2].rc.resident_fast_decode
        assert made[2].rc.max_weight_cache_mb == 1200

        manager.get(target_path, "fast")

    assert wrapped.closes == 1
    assert (wrapped.target.closes, wrapped.draft.closes) == (1, 1)


def test_engine_manager_prefers_full_resident_qwen2_when_governor_admits_it(
        tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            self.cache = SimpleNamespace(max_bytes=18_000_000_000)
            self.governor = None
            made.append(self)

        def close(self):
            self.closes += 1

    manager = EngineManager()
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine",
               side_effect=AssertionError("resident target should win")), \
         patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        engine = manager.get(target_path, "lossless")

    assert engine is made[1]
    assert len(made) == 2
    assert made[0].closes == 1
    assert made[0].rc.max_weight_cache_mb > 16_000
    assert made[0].rc.embed_rows
    assert not made[0].rc.resident_fast_decode
    assert made[1].rc.max_weight_cache_mb > 16_000
    assert not made[1].rc.embed_rows
    assert made[1].rc.resident_fast_decode
    assert made[1].rc.resident_fast_prefill_limit == 2048
    assert made[1].rc.stepped_kv_threshold == 2048
    assert made[1].rc.hot_prompt_kv
    assert made[1].rc.hot_prompt_kv_min_tokens == 2048
    assert made[1].rc.prompt_kv_min_tokens == 2048


def test_qwen2_admission_race_falls_back_to_streamed_speculation(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    target_path = tmp_path / "Qwen2.5-7B-Instruct"
    draft_path = tmp_path / "Qwen2.5-1.5B-Instruct-mlx-mxfp4"
    target_path.mkdir()
    draft_path.mkdir()
    cfg = SimpleNamespace(
        model_type="qwen2", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=0,
        hidden_size=3584, intermediate_size=18944,
        num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, head_dim=128, vocab_size=152064,
        attention_bias=True)
    made = []

    class FakeGovernor:
        def __init__(self, fitted):
            self.fitted = fitted

        def fit_cache_to_live_headroom(self):
            return self.fitted

    class FakeEngine:
        def __init__(self, path, rc):
            self.path, self.rc, self.closes = Path(path), rc, 0
            index = len(made)
            fitted = (18_000_000_000 if index == 0 else
                      8_000_000_000 if index == 1 else
                      rc.max_weight_cache_mb * 1_000_000)
            self.cache = SimpleNamespace(
                max_bytes=rc.max_weight_cache_mb * 1_000_000)
            self.governor = FakeGovernor(fitted)
            made.append(self)

        def close(self):
            self.closes += 1

    class FakeSpeculativeEngine:
        def __init__(self, target, draft, **_kwargs):
            self.target, self.draft = target, draft

        def close(self):
            self.draft.close()
            self.target.close()

    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.server._speculative_draft_for", return_value=draft_path), \
         patch("runtime.engine.StreamingEngine", FakeEngine), \
         patch("runtime.speculative.SpeculativeEngine", FakeSpeculativeEngine), \
         patch.dict("os.environ", {"VMODEL_SPECULATIVE_DRAFT": "auto"}):
        wrapped = EngineManager().get(target_path, "lossless")

    assert isinstance(wrapped, FakeSpeculativeEngine)
    assert len(made) == 4
    assert made[0].closes == 1
    assert made[1].closes == 1
    assert wrapped.target is made[2]
    assert wrapped.target.rc.max_weight_cache_mb == 6000
    assert not wrapped.target.rc.resident_fast_decode
    assert wrapped.draft is made[3]


def test_fast_long_prefix_is_distinct_from_fast_prefix():
    assert split_model_mode("lossy-long-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast-long")
    assert split_model_mode("lossy-Qwen2.5-1.5B") == (
        "Qwen2.5-1.5B", "fast")


def test_derived_quantized_checkpoint_is_advertised_only_as_lossy(tmp_path):
    from unittest.mock import patch

    released = tmp_path / "released"
    derived = tmp_path / "derived"
    released.mkdir()
    derived.mkdir()
    (released / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
    (derived / "config.json").write_text(json.dumps({
        "model_type": "olmoe",
        "voom_quantization": {"profile": "experts", "source": str(released)},
    }))

    with patch("runtime.server._registry", return_value={
            "released": released, "derived": derived}):
        ids = _advertised_model_ids()
    assert "released" in ids
    assert "lossy-released" in ids
    assert "derived" not in ids
    assert "lossy-derived" in ids


def test_registry_does_not_advertise_auxiliary_embedding_encoders(tmp_path):
    from unittest.mock import patch

    from runtime.local_config import StorageConfig

    models = tmp_path / "models"
    chat = models / "Qwen-test"
    embed = models / "tool-embed-bge-small-en-v1.5"
    chat.mkdir(parents=True)
    embed.mkdir()
    (chat / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (embed / "config.json").write_text(json.dumps({"model_type": "bert"}))
    with patch("runtime.server.ROOT", tmp_path), \
         patch("runtime.local_config.get_storage_config",
               return_value=StorageConfig()):
        registry = _registry()
    assert "Qwen-test" in registry
    assert "tool-embed-bge-small-en-v1.5" not in registry


def test_engine_manager_rejects_derived_checkpoint_as_lossless(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    (tmp_path / "config.json").write_text(json.dumps({
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64)
    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path):
        try:
            EngineManager().get(tmp_path, "lossless")
        except RequestValidationError as error:
            assert "vOOM-derived lossy artifact" in str(error)
        else:
            raise AssertionError("derived checkpoint was accepted as lossless")


def test_base_olmoe_prefers_complete_expert_mxfp4_sibling(tmp_path):
    source = tmp_path / "OLMoE"
    q4 = tmp_path / "OLMoE-mlx-expert-mxfp4"
    q8 = tmp_path / "OLMoE-mlx-expert-mxfp8"
    source.mkdir()
    for path, mode, bits in ((q4, "mxfp4", 4), (q8, "mxfp8", 8)):
        path.mkdir()
        (path / "config.json").write_text(json.dumps({
            "model_type": "olmoe",
            "quantization": {"mode": mode, "bits": bits, "group_size": 32},
            "voom_quantization": {
                "profile": "experts", "source": str(source.resolve())},
        }))
        (path / "model.safetensors.index.json").write_text("{}")

    assert _preferred_fast_artifact(source) == q4
    assert _preferred_fast_artifact(q8) == q8


def test_base_qwen36_prefers_complete_expert_mxfp4_sibling(tmp_path):
    source = tmp_path / "Qwen3.6-35B-A3B"
    q4 = tmp_path / "Qwen3.6-35B-A3B-mlx-expert-mxfp4"
    source.mkdir()
    q4.mkdir()
    (q4 / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {
            "profile": "experts", "source": str(source.resolve())},
    }))
    (q4 / "model.safetensors.index.json").write_text("{}")

    assert _preferred_fast_artifact(source) == q4


def _write_mtplx_fixture(source: Path, artifact: Path, revision: str) -> None:
    (source / ".cache/huggingface/trees").mkdir(parents=True)
    (source / ".cache/huggingface/trees" / f"{revision}.json").write_text("{}")
    artifact.mkdir()
    base_model = f"empero-ai/{source.name}"
    (artifact / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
    }))
    (artifact / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"mtplx_mtp_sidecar": "mtp.safetensors"},
        "weight_map": {},
    }))
    (artifact / "mtp.safetensors").write_bytes(b"fixture")
    (artifact / "mtplx_runtime.json").write_text(json.dumps({
        "base_model": base_model,
        "base_revision": revision,
        "mtp_source": base_model,
        "mtp_sidecar_file": "mtp.safetensors",
        "mtp_sidecar_format": "bf16",
        "body_quantization": "mxfp4",
    }))
    (artifact / "RELEASE_MANIFEST.json").write_text(json.dumps({
        "base_model": base_model,
        "source_revision": revision,
    }))


def test_base_qwythos_prefers_revision_bound_mtplx_mxfp4_sibling(tmp_path):
    revision = "7c72a9c714cf66281cb222c4aa0aef368d84c94f"
    source = tmp_path / "Qwythos-27B-v1"
    artifact = tmp_path / "Qwythos-27B-v1-mlx-all-mxfp4"
    source.mkdir()
    _write_mtplx_fixture(source, artifact, revision)

    assert _is_voom_lossy_checkpoint(artifact)
    assert _preferred_fast_artifact(source) == artifact


def test_mtplx_source_revision_mismatch_is_not_selected(tmp_path):
    source_revision = "7c72a9c714cf66281cb222c4aa0aef368d84c94f"
    artifact_revision = "8aa70bf12de66ea70d7543daf71150e14dcff01d"
    source = tmp_path / "Qwythos-27B-v1"
    artifact = tmp_path / "Qwythos-27B-v1-mlx-all-mxfp4"
    source.mkdir()
    _write_mtplx_fixture(source, artifact, artifact_revision)
    tree = source / ".cache/huggingface/trees"
    (tree / f"{artifact_revision}.json").unlink()
    (tree / f"{source_revision}.json").write_text("{}")

    # The artifact remains explicitly lossy, but it must not be substituted
    # for a different local source revision.
    assert _is_voom_lossy_checkpoint(artifact)
    assert _preferred_fast_artifact(source) == source


def test_mtplx_checkpoint_is_advertised_only_as_lossy(tmp_path):
    from unittest.mock import patch

    revision = "7c72a9c714cf66281cb222c4aa0aef368d84c94f"
    source = tmp_path / "Qwythos-27B-v1"
    artifact = tmp_path / "Qwythos-27B-v1-mlx-all-mxfp4"
    source.mkdir()
    _write_mtplx_fixture(source, artifact, revision)

    with patch("runtime.server._registry", return_value={
            source.name: source, artifact.name: artifact}):
        ids = _advertised_model_ids()
    assert source.name in ids
    assert f"lossy-{source.name}" in ids
    assert artifact.name not in ids
    assert f"lossy-{artifact.name}" in ids


def test_execution_profile_discloses_effective_derived_artifact(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(
            quantization={"mode": "mxfp4", "bits": 4, "group_size": 32},
            on_disk_quantized=True),
        rc=SimpleNamespace(quant_bits=4, quant_mode="mxfp4", quant_group_size=32),
    )

    assert _execution_profile_fields(engine) == {
        "vmodel_checkpoint": tmp_path.name,
        "vmodel_weight_profile": "experts-mxfp4-q4-g32",
        "vmodel_backend": "voom",
    }


def test_execution_profile_discloses_qwen35_prefill_chunk_ceiling(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(quantization={}, on_disk_quantized=False),
        rc=SimpleNamespace(
            quant_bits=0,
            rerank_lm_head=False,
            resident_attention_mode="",
            expert_top_k_by_layer=(),
            qwen35_prefill_chunk_ceiling=128,
        ),
    )

    assert _execution_profile_fields(engine)[
        "vmodel_qwen35_prefill_chunk_ceiling"] == 128


def test_execution_profile_discloses_reranked_head(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(
            quantization={"mode": "mxfp4", "bits": 4, "group_size": 32},
            on_disk_quantized=True),
        rc=SimpleNamespace(
            quant_bits=4, quant_mode="mxfp4", quant_group_size=32,
            rerank_lm_head=True, rerank_lm_head_mode="mxfp4",
            rerank_lm_head_bits=4, rerank_lm_head_group_size=32,
            rerank_lm_head_candidates=32,
            resident_attention_mode="mxfp8",
            resident_attention_bits=8,
            resident_attention_group_size=32),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "experts-mxfp4-q4-g32+head-mxfp4-q4-g32-rerank32"
        "+attn-mxfp8-q8-g32")


def test_execution_profile_discloses_resident_quantized_head(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(
            quantization={"mode": "mxfp4", "bits": 4, "group_size": 32},
            on_disk_quantized=True),
        rc=SimpleNamespace(
            quant_bits=4, quant_mode="mxfp4", quant_group_size=32,
            quant_lm_head=True, pin_lm_head=True, stream_lm_head=False,
            rerank_lm_head=False, resident_attention_mode=""),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "experts-mxfp4-q4-g32+head-mxfp4-q4-g32")


def test_execution_profile_discloses_olmoe_top_k_schedule(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(quantization={}, on_disk_quantized=False),
        rc=SimpleNamespace(
            quant_bits=0,
            rerank_lm_head=False,
            resident_attention_mode="",
            expert_top_k_by_layer=(7, 7, 8, 8),
        ),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "released+olmoe-topk-7.7.8.8")


def test_execution_profile_discloses_native_compressed_tensors_mxfp4(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    engine = SimpleNamespace(
        _model_dir=tmp_path,
        store=SimpleNamespace(quantization={}, on_disk_quantized=False),
        rc=SimpleNamespace(
            quant_bits=0,
            rerank_lm_head=False,
            resident_attention_mode="",
            expert_top_k_by_layer=(),
            native_ct_mxfp4=True,
        ),
    )

    assert _execution_profile_fields(engine)["vmodel_weight_profile"] == (
        "released+ct-mxfp4-native")


def test_mxfp8_olmoe_profile_gets_resident_admission_budget(tmp_path):
    from unittest.mock import patch

    from runtime.server import EngineManager

    (tmp_path / "config.json").write_text(json.dumps({
        "quantization": {"mode": "mxfp8", "bits": 8, "group_size": 32},
        "voom_quantization": {"profile": "experts", "source": "/source"},
    }))
    cfg = SimpleNamespace(
        model_type="olmoe", tie_word_embeddings=False,
        index_topk=0, vision_config=None, num_experts=64,
        hidden_size=2048, intermediate_size=1024,
        num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, head_dim=128, vocab_size=50304,
        attention_bias=False)
    captured = []

    class FakeEngine:
        def __init__(self, _path, rc):
            captured.append(rc)

        def close(self):
            pass

    with patch("runtime.config.ModelConfig.from_dir", return_value=cfg), \
         patch("runtime.path_resolver.resolve_model_dir", side_effect=lambda path: path), \
         patch("runtime.engine.StreamingEngine", FakeEngine):
        EngineManager().get(tmp_path, "fast")

    assert captured[0].resident_moe_decode
    assert captured[0].max_weight_cache_mb == 9000
    assert captured[0].resident_attention_mode == ""
    assert captured[0].stepped_kv_threshold == 1


def test_active_context_limit_uses_stricter_runtime_correctness_bound():
    engine = _fake_engine(model_limit=1_000_000, context_bound=2_048)
    assert _active_context_limit(engine) == 2_048
    assert _validate_context_budget(
        engine, 2_000, 48, prompt_label="prompt", output_label="max_tokens") == 2_048
    try:
        _validate_context_budget(
            engine, 2_000, 49, prompt_label="prompt", output_label="max_tokens")
    except RequestValidationError as error:
        assert "active context limit=2048" in str(error)
    else:
        raise AssertionError("runtime correctness-bound overflow was accepted")


def test_chat_prompt_rejects_correctness_bound_before_generation():
    with tempfile.TemporaryDirectory() as directory:
        engine = _fake_engine(model_limit=1_000_000, context_bound=40)
        args = (engine, Path(directory), [{"role": "user", "content": "x"}],
                "low", [], [], "lossless")
        _prompt, prompt_tokens, *_rest, metadata = _prepare_chat_prompt(*args, 1)
        assert metadata["context_limit"] == 40
        remaining = 40 - prompt_tokens
        _prepare_chat_prompt(*args, remaining)
        try:
            _prepare_chat_prompt(*args, remaining + 1)
        except RequestValidationError as error:
            assert "active context limit=40" in str(error)
        else:
            raise AssertionError("chat correctness-bound overflow was accepted")


def test_invalid_image_payload_is_a_request_validation_error():
    try:
        _load_vision_images(["data:image/png;base64,%%%"])
    except RequestValidationError as error:
        assert str(error) == "invalid image 1: image data URI contains invalid base64"
    else:
        raise AssertionError("invalid image payload was accepted")


def test_positive_token_limit_rejects_zero_negative_bool_fraction_and_text():
    assert _positive_token_limit(1, "max_tokens") == 1
    assert _positive_token_limit("2", "max_tokens") == 2
    for value in (0, -1, True, 1.5, "nope", None):
        try:
            _positive_token_limit(value, "max_tokens")
        except RequestValidationError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(f"invalid token limit accepted: {value!r}")


def test_omitted_output_budget_is_eos_safety_ceiling_not_legacy_64():
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VMODEL_OMITTED_MAX_OUTPUT_TOKENS", None)
        assert _omitted_output_token_limit() == 4096
    with patch.dict(os.environ, {"VMODEL_OMITTED_MAX_OUTPUT_TOKENS": "768"}):
        assert _omitted_output_token_limit() == 768
    with patch.dict(os.environ, {"VMODEL_OMITTED_MAX_OUTPUT_TOKENS": "0"}):
        try:
            _omitted_output_token_limit()
        except RequestValidationError:
            pass
        else:
            raise AssertionError("accepted a zero omitted-output safety ceiling")


def test_responses_mixed_text_and_tool_call_keeps_both_items_and_id():
    text = (
        "I will check.\n"
        '<tool_call>{"name":"weather","arguments":{"city":"Chicago"}}</tool_call>'
        "\nThen I will summarize."
    )
    content, output = _responses_output_items(
        text, [{"type": "function", "function": {
            "name": "weather", "parameters": {}}}], "qwen2", "msg_fixed")
    assert "I will check." in content
    assert "Then I will summarize." in content
    assert [item["type"] for item in output] == ["message", "function_call"]
    assert output[0]["id"] == "msg_fixed"
    assert output[0]["content"][0]["text"] == content
    assert output[1]["name"] == "weather"


def test_responses_incomplete_container_has_incomplete_message_item():
    _content, output = _responses_output_items(
        "partial", [], "qwen2", "msg_partial", message_status="incomplete")
    assert output[0]["status"] == "incomplete"


def test_release_generation_state_drops_every_previous_request_owner():
    from runtime.request_state import release_generation_state

    marker = object()
    owner = SimpleNamespace(
        _hot_prompt_slots=[marker, marker],
        last_kv=marker, _h_window=marker, _h_last=marker, _provisional=marker,
    )
    release_generation_state(owner)
    assert owner._hot_prompt_slots == []
    for name in ("last_kv", "_h_window", "_h_last", "_provisional"):
        assert getattr(owner, name) is None


def test_release_generation_state_releases_aliased_refcounted_kv_once():
    from runtime.request_state import release_generation_state

    class Releasable:
        calls = 0

        def release(self):
            self.calls += 1

    state = Releasable()
    owner = SimpleNamespace(
        _hot_prompt_slots=[SimpleNamespace(kv=state)],
        last_kv=state, _h_window=None, _h_last=None, _provisional=None,
    )
    release_generation_state(owner)
    assert state.calls == 1


def test_openai_finish_reason_distinguishes_length_eos_and_tool_calls():
    assert _openai_finish_reason({"termination_reason": "length"}) == "length"
    assert _openai_finish_reason({"termination_reason": "eos"}) == "stop"
    assert _openai_finish_reason(
        {"termination_reason": "length"}, has_tool_calls=True) == "tool_calls"


def test_stream_holdback_catches_complete_marker_with_trailing_text():
    markers = ("<tool_call>",)
    assert _safe_emit_len("<tool_call>", markers) == 0
    assert _safe_emit_len("<tool_call>{", markers) == 0
    assert _safe_emit_len("hello<tool_call>{", markers) == len("hello")
    assert _safe_emit_len("hello", markers) == len("hello")


def test_stream_holdback_survives_marker_split_across_decode_pieces():
    markers = ("<tool_call>",)
    pending = ""
    emitted = ""
    for piece in ("hello<tool_", "call>{"):
        pending += piece
        safe = _safe_emit_len(pending, markers)
        emitted += pending[:safe]
        pending = pending[safe:]
    assert emitted == "hello"
    assert pending == "<tool_call>{"
    assert _safe_emit_len(pending, markers) == 0


def test_marker_holdback_streams_safe_text_and_replays_post_call_text():
    holdback = _MarkerHoldback(("<tool_call>",))
    assert holdback.feed("hello<tool_") == "hello"
    assert holdback.feed("call>{\"name\":\"x\"}") == ""
    assert holdback.holding
    assert holdback.final_remainder("hello after") == " after"


def test_hidden_decision_streams_after_first_irreversible_prefix_mismatch():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    for piece in ("H", "ello", " from", " the model"):
        decision.feed(piece)
    decision.finish_direct("Hello from the model")
    assert decision.branch == "direct"
    assert "".join(emitted) == "Hello from the model"
    assert len(emitted) == 4


def test_hidden_decision_holds_marker_at_every_decode_split():
    marker = "<tool_call>"
    for split in range(len(marker) + 1):
        emitted = []
        decision = _HiddenDecisionStream("qwen3", emitted.append)
        decision.feed(" \n" + marker[:split])
        decision.feed(marker[split:] + '{"name":"vmodel_search_tools"}')
        assert decision.branch == "tool"
        assert emitted == []


def test_hidden_decision_releases_marker_like_direct_text():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    decision.feed("<tool_calls> is ordinary text")
    decision.finish_direct("<tool_calls> is ordinary text")
    assert decision.branch == "direct"
    assert "".join(emitted) == "<tool_calls> is ordinary text"


def test_hidden_decision_never_leaks_late_virtual_marker():
    emitted = []
    decision = _HiddenDecisionStream("qwen3", emitted.append)
    before = "Before. "
    marker = (
        '<tool_call>{"name":"vmodel_search_tools",'
        '"arguments":{"query":"browser"}}</tool_call>')
    after = " After."
    decision.feed(before)
    decision.feed("<tool_")
    decision.feed(marker[len("<tool_"):] + after)
    assert decision.branch == "direct"
    assert decision.late_marker_detected
    assert "<tool" not in "".join(emitted)
    virtual, _raw = _hidden_tool_search_pair()
    content, calls = _parse_request_tool_calls(
        before + marker + after, [virtual], "qwen3", allow_parallel=False)
    assert [call["function"]["name"] for call in calls] == [
        "vmodel_search_tools"]
    decision.finish_direct(content)
    assert "".join(emitted) == before + after


def test_hidden_decision_handles_harmony_spacing_and_final_channel():
    emitted = []
    tool_decision = _HiddenDecisionStream("gpt_oss", emitted.append)
    tool_decision.feed("commentary   to=functions.vmodel_search_tools")
    assert tool_decision.branch == "tool"
    assert emitted == []

    direct = _HiddenDecisionStream("gpt_oss", emitted.append)
    direct.feed("<|channel|>final")
    direct.finish_direct("<|channel|>final")
    assert direct.branch == "direct"
    assert "".join(emitted) == "<|channel|>final"


def test_resident_adjusted_transient_excludes_persistent_cache_growth():
    from runtime.engine import (
        _layer_transient_for_positions, _layer_transient_reserve_margin,
        _resident_adjusted_transient)

    assert _resident_adjusted_transient(1_000, 2_500, 2_500) == 0
    assert _resident_adjusted_transient(1_000, 2_500, 2_900) == 400
    assert _resident_adjusted_transient(2_500, 1_000, 2_900) == 400
    assert _layer_transient_reserve_margin(1) == 0
    assert _layer_transient_reserve_margin(2) == 400_000_000
    assert _layer_transient_for_positions(1, 1_280, 96) == (96, 0)
    assert _layer_transient_for_positions(32, 1_280, 96) == (
        1_280, 400_000_000)


def test_layer_transient_isolated_by_architecture_signature():
    """A dense-layer outlier must not poison an adjacent routed MoE reserve."""
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    engine = object.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="synthetic_hybrid",
        layer_types=(),
        kda_layers=(0, 1, 2),
        full_attn_layers=(),
        indexer_types=(),
        num_experts=256,
        mlp_layer_types=(),
        first_k_dense_replace=1,
    )
    engine._layer_transient = 0
    engine._prefill_layer_transient = 0
    engine._prefill_layer_transient_by_positions = {}
    engine._decode_layer_transient = 0
    engine._layer_transient_by_signature = {}
    engine._layer_transient_margin = 0

    # The first layer uses a large dense MLP; following layers use routed MoE.
    StreamingEngine._record_layer_transient(
        engine, 1, 0, 3_200_000_000)
    assert StreamingEngine._select_layer_transient(engine, 1, 1) == 0

    # Once one MoE layer has been measured, another layer with the same
    # architecture signature reuses that narrower observation.
    StreamingEngine._record_layer_transient(engine, 1, 1, 125_000_000)
    assert StreamingEngine._select_layer_transient(
        engine, 1, 2) == 125_000_000

    # Request-level admission remains conservative and sees the aggregate.
    assert StreamingEngine._restore_aggregate_layer_transient(
        engine, 1) == 3_200_000_000


def test_layer_transient_drops_one_time_signature_warmup_peak():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    engine = object.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="synthetic_moe",
        layer_types=(),
        kda_layers=(0, 1, 2, 3),
        full_attn_layers=(),
        indexer_types=(),
        num_experts=256,
        mlp_layer_types=(),
        first_k_dense_replace=0,
    )
    engine._layer_transient = 0
    engine._prefill_layer_transient = 0
    engine._prefill_layer_transient_by_positions = {}
    engine._decode_layer_transient = 0
    engine._layer_transient_by_signature = {}
    engine._layer_transient_observation_counts = {}
    engine._layer_transient_recurring_max = {}
    engine._layer_transient_margin = 0

    StreamingEngine._record_layer_transient(
        engine, 5, 0, 3_200_000_000)
    assert StreamingEngine._select_layer_transient(
        engine, 5, 1) == 3_200_000_000

    # The second execution establishes the recurring high-water; later
    # recurring increases remain monotonic and therefore fail closed.
    StreamingEngine._record_layer_transient(
        engine, 5, 1, 900_000_000)
    assert StreamingEngine._select_layer_transient(
        engine, 5, 2) == 900_000_000
    StreamingEngine._record_layer_transient(
        engine, 5, 2, 1_200_000_000)
    assert StreamingEngine._select_layer_transient(
        engine, 5, 3) == 1_200_000_000


def test_serial_verify_transient_does_not_poison_one_token_decode():
    from types import SimpleNamespace

    from runtime.engine import StreamingEngine

    engine = object.__new__(StreamingEngine)
    engine.cfg = SimpleNamespace(
        model_type="synthetic_moe",
        layer_types=(),
        kda_layers=(0, 1),
        full_attn_layers=(),
        indexer_types=(),
        num_experts=256,
        mlp_layer_types=(),
        first_k_dense_replace=0,
    )
    engine._layer_transient = 0
    engine._layer_transient_margin = 0
    engine._layer_transient_by_signature = {}
    engine._layer_transient_observation_counts = {}
    engine._layer_transient_recurring_max = {}
    engine._decode_layer_transient = 0
    engine._prefill_layer_transient = 0
    engine._prefill_layer_transient_by_positions = {}
    engine._serial_verify_layer_transient = {}
    engine._serial_verify_layer_transient_counts = {}
    engine._serial_verify_layer_transient_recurring_max = {}

    StreamingEngine._record_layer_transient(
        engine, 1, 0, 500_000_000
    )
    StreamingEngine._record_serial_verify_layer_transient(
        engine, 3, 0, 1_400_000_000
    )

    assert StreamingEngine._select_serial_verify_layer_transient(
        engine, 3, 1
    ) == 1_400_000_000
    StreamingEngine._record_serial_verify_layer_transient(
        engine, 3, 1, 600_000_000
    )
    assert StreamingEngine._select_serial_verify_layer_transient(
        engine, 3, 1
    ) == 600_000_000
    assert StreamingEngine._select_layer_transient(
        engine, 1, 1
    ) == 500_000_000


def test_cache_io_delta_reports_only_current_request():
    from types import SimpleNamespace

    from runtime.engine import _cache_io_snapshot, _record_cache_io_delta

    cache_stats = SimpleNamespace(
        hits=10, misses=20, evictions=3, bytes_read=1_000)
    store_stages = [1_000, 2, 3_000, 4_000]
    scale_stages = [5_000, 6_000, 7_000, 8]
    engine = SimpleNamespace(
        cache=SimpleNamespace(
            stats=cache_stats, total_bytes=400, max_bytes=800),
        store=SimpleNamespace(
            fast_tier_bytes=100, archive_bytes=900,
            stage_snapshot=lambda: tuple(store_stages),
            k3_scale_sidecar_snapshot=lambda: tuple(scale_stages)),
        governor=SimpleNamespace(
            reservations=2,
            reservation_calls=3,
            reservation_fast_path_calls=1,
            reservation_clear_cache_only_calls=1,
            reservation_reason_counts={
                "serial-verify-layer-page": 2,
                "serial-verify-transient": 3,
                "qwen-prefill-layer-page": 4,
                "qwen-prefill-transient": 5,
            },
            reservation_requested_bytes=100,
            reservation_budget_reduced_bytes=200,
            reservation_budget_restored_bytes=50,
            reservation_cache_released_bytes=150,
            reservation_unproductive_shrinks=1,
            reservation_failures=1,
        ),
        expert_hits=4, expert_misses=5,
        _layer_transient=60, _token_transient=70,
    )
    before = _cache_io_snapshot(engine)
    cache_stats.hits += 2
    cache_stats.misses += 3
    cache_stats.evictions += 4
    cache_stats.bytes_read += 5_000
    engine.expert_hits += 6
    engine.expert_misses += 7
    engine.governor.reservations += 8
    engine.governor.reservation_calls += 9
    engine.governor.reservation_fast_path_calls += 10
    engine.governor.reservation_clear_cache_only_calls += 11
    engine.governor.reservation_reason_counts[
        "serial-verify-layer-page"] += 12
    engine.governor.reservation_reason_counts[
        "serial-verify-transient"] += 13
    engine.governor.reservation_reason_counts[
        "qwen-prefill-layer-page"] += 14
    engine.governor.reservation_reason_counts[
        "qwen-prefill-transient"] += 15
    engine.governor.reservation_requested_bytes += 16
    engine.governor.reservation_budget_reduced_bytes += 17
    engine.governor.reservation_budget_restored_bytes += 18
    engine.governor.reservation_cache_released_bytes += 19
    engine.governor.reservation_unproductive_shrinks += 20
    engine.store.fast_tier_bytes += 9
    engine.store.archive_bytes += 10
    store_stages[0] += 11
    store_stages[1] += 12
    store_stages[2] += 13
    store_stages[3] += 14
    scale_stages[0] += 15
    scale_stages[1] += 16
    scale_stages[2] += 17
    scale_stages[3] += 18
    stats = {}
    _record_cache_io_delta(engine, before, stats)

    assert stats["weight_cache_hits"] == 2
    assert stats["weight_cache_misses"] == 3
    assert stats["weight_cache_evictions"] == 4
    assert stats["weight_store_bytes_read"] == 5_000
    assert stats["expert_cache_hits"] == 6
    assert stats["expert_cache_misses"] == 7
    assert stats["governor_reservations"] == 8
    assert stats["governor_reservation_calls"] == 9
    assert stats["governor_reservation_fast_path_calls"] == 10
    assert stats["governor_reservation_clear_cache_only_calls"] == 11
    assert stats["governor_serial_verify_page_reservation_calls"] == 12
    assert stats["governor_serial_verify_transient_reservation_calls"] == 13
    assert stats["governor_qwen_prefill_page_reservation_calls"] == 14
    assert stats["governor_qwen_prefill_transient_reservation_calls"] == 15
    assert stats["governor_reservation_requested_bytes"] == 16
    assert stats["governor_reservation_budget_reduced_bytes"] == 17
    assert stats["governor_reservation_budget_restored_bytes"] == 18
    assert stats["governor_reservation_cache_released_bytes"] == 19
    assert stats["governor_reservation_unproductive_shrinks"] == 20
    assert stats["weight_fast_tier_bytes"] == 9
    assert stats["weight_archive_bytes"] == 10
    assert stats["ct_mxfp4_transform_ns"] == 11
    assert stats["ct_mxfp4_transform_calls"] == 12
    assert stats["ct_mxfp4_input_bytes"] == 13
    assert stats["ct_mxfp4_resident_bytes"] == 14
    assert stats["k3_scale_sidecar_read_bytes"] == 15
    assert stats["k3_scale_sidecar_output_bytes"] == 16
    assert stats["k3_scale_sidecar_decode_ns"] == 17
    assert stats["k3_scale_sidecar_decode_calls"] == 18
    assert stats["weight_cache_resident_bytes"] == 400
    assert stats["layer_transient_bytes"] == 60


class _WrapperEngine:
    """Mirrors SpeculativeEngine/DSparkSpeculativeEngine/
    QwenMTPSpeculativeEngine's shape: a plain object (no inheritance from
    the wrapped target) that defines its OWN .generate() and forwards
    every other attribute via __getattr__."""

    def __init__(self, target):
        self.target = target
        self.own_generate_calls = 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    def generate(self, *args, **kwargs):
        self.own_generate_calls += 1
        return {"via": "wrapper"}


class _PlainTarget:
    def __init__(self):
        self.retry_calls = 0
        self.plain_calls = 0

    def generate(self, *args, **kwargs):
        self.plain_calls += 1
        return {"via": "target-plain"}

    def generate_with_memory_retry(self, *args, **kwargs):
        self.retry_calls += 1
        return {"via": "target-retry"}


def test_has_own_method_distinguishes_real_methods_from_getattr_proxy():
    target = _PlainTarget()
    wrapper = _WrapperEngine(target)
    assert _has_own_method(target, "generate_with_memory_retry")
    assert not _has_own_method(wrapper, "generate_with_memory_retry")
    # The wrapper's OWN .generate is still a real method, unaffected.
    assert _has_own_method(wrapper, "generate")


def test_engine_generate_uses_wrappers_own_generate_not_targets_retry():
    """2026-07-22 regression: a naive getattr(engine,
    "generate_with_memory_retry", engine.generate) resolves through a
    wrapper's __getattr__ straight to the TARGET's bound retry method,
    silently bypassing the wrapper's own speculative .generate() override
    for every request -- live-confirmed this made MTP speculative decoding
    never actually engage despite being "enabled". A wrapper adapter that
    defines its own .generate() must have that method called, never the
    wrapped target's."""
    target = _PlainTarget()
    wrapper = _WrapperEngine(target)
    result = _engine_generate(wrapper, "prompt", 16)
    assert result == {"via": "wrapper"}
    assert wrapper.own_generate_calls == 1
    assert target.retry_calls == 0
    assert target.plain_calls == 0


def test_engine_generate_uses_plain_targets_own_retry_method():
    """A real (unwrapped) engine's own generate_with_memory_retry must
    still be preferred, exactly as before this fix."""
    target = _PlainTarget()
    result = _engine_generate(target, "prompt", 16)
    assert result == {"via": "target-retry"}
    assert target.retry_calls == 1
    assert target.plain_calls == 0


def test_engine_generate_scopes_privacy_safe_authoritative_rank_capture():
    class Capture:
        def __init__(self):
            self.shapes = []
            self.active = False

        @contextmanager
        def request(self, shape):
            self.shapes.append(shape)
            self.active = True
            try:
                yield
            finally:
                self.active = False

    capture = Capture()

    class Target:
        def __init__(self):
            self._lm_head_w = SimpleNamespace(recall_rank_capture=capture)
            self.cfg = SimpleNamespace(model_type="qwen3_5")
            self.rc = SimpleNamespace(expert_top_k_by_layer=())
            self.tokenizer = _CharTokenizer()

        def generate_with_memory_retry(self, *args, **kwargs):
            assert capture.active
            return {"via": "captured"}

    prompt = PreparedPrompt(
        "rendered", range(900),
        rerank_capture_shape={
            "prompt_tokens": 900,
            "system_chars": 2000,
            "tool_count": 134,
            "message_count": 7,
            "developer": True,
        },
    )
    result = _engine_generate(
        Target(), prompt, 8,
        sampling=SimpleNamespace(is_greedy=False),
        constraint=object(),
        on_token=lambda _token: None,
    )
    assert result == {"via": "captured"}
    assert capture.shapes == [{
        "prompt_tokens_bucket": "513-2048",
        "system_chars_bucket": "1025-8192",
        "tool_count_bucket": "129+",
        "message_count_bucket": "5-16",
        "developer": True,
        "streaming": True,
        "temperature_class": "stochastic",
        "constrained": True,
    }]


def test_engine_generate_scopes_and_restores_qwen_expert_top_k():
    class Target:
        def __init__(self):
            self.cfg = SimpleNamespace(
                model_type="qwen3_5_moe", num_hidden_layers=3,
                num_experts_per_tok=8, expert_top_k_by_layer=())
            self.rc = SimpleNamespace(expert_top_k_by_layer=())

        def generate_with_memory_retry(self, *args, **kwargs):
            assert self.cfg.expert_top_k_by_layer == (2, 2, 2)
            assert self.rc.expert_top_k_by_layer == (2, 2, 2)
            return {"via": "request-top-k"}

    target = Target()
    assert _engine_generate(
        target, "prompt", 16, expert_top_k=2,
    ) == {"via": "request-top-k"}
    assert target.cfg.expert_top_k_by_layer == ()
    assert target.rc.expert_top_k_by_layer == ()


def test_priority_lock_serves_waiters_in_priority_order_not_arrival_order():
    lock = PriorityLock()
    lock.acquire()  # hold it so every thread below queues up as a waiter

    order = []
    order_lock = threading.Lock()
    # Deliberately spawned in an order that does NOT match priority, so a
    # pass here can only be explained by real priority-based scheduling.
    priorities = [30, 10, 20, 0]

    def worker(priority):
        lock.acquire(priority=priority)
        with order_lock:
            order.append(priority)
        lock.release()

    threads = [threading.Thread(target=worker, args=(p,)) for p in priorities]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + 5.0
    while len(lock._waiters) < len(priorities) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(lock._waiters) == len(priorities), (
        "not all threads registered as waiters before the deadline")

    lock.release()  # let the queued waiters run, lowest priority first
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert order == sorted(priorities), (
        f"expected priority order {sorted(priorities)}, got {order}")


def test_priority_lock_ties_broken_by_arrival_order():
    lock = PriorityLock()
    lock.acquire()

    order = []
    order_lock = threading.Lock()

    def worker(label):
        lock.acquire(priority=5)  # same priority for every waiter
        with order_lock:
            order.append(label)
        lock.release()

    labels = ["first", "second", "third"]
    threads = []
    for label in labels:
        thread = threading.Thread(target=worker, args=(label,))
        threads.append(thread)
        thread.start()
        # Register one at a time so arrival order is deterministic; the
        # heap's secondary (seq) key is what this test actually verifies.
        deadline = time.monotonic() + 5.0
        while (len(lock._waiters) < len(threads)
               and time.monotonic() < deadline):
            time.sleep(0.005)

    lock.release()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert order == labels


def test_priority_lock_nonblocking_matches_threading_lock_contract():
    lock = PriorityLock()
    assert lock.acquire(blocking=False) is True
    assert lock.acquire(blocking=False) is False
    lock.release()
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_priority_lock_context_manager_uses_default_priority():
    lock = PriorityLock()
    with lock:
        assert lock._locked is True
    assert lock._locked is False


def test_harmony_visible_text_returns_final_channel_only():
    # The exact shape measured from the real gpt-oss-120b Plex gate on
    # 2026-07-31: glyphs stripped by decode, channel names left as bare words.
    raw = ("analysisECHO_R is rated R so exclude. GOLF_TV14 excluded.\n\n"
           "Thus final list: ALPHA_G, CHARLIE_TVY.\n\n"
           "Return plain list.assistantfinalALPHA_G\nCHARLIE_TVY")
    assert _harmony_visible_text(raw, "gpt_oss") == "ALPHA_G\nCHARLIE_TVY"


def test_harmony_visible_text_handles_special_token_glyphs():
    raw = ("<|channel|>analysis<|message|>secret reasoning<|end|>"
           "<|start|>assistant<|channel|>final<|message|>the answer<|return|>")
    assert _harmony_visible_text(raw, "gpt_oss") == "the answer"


def test_harmony_visible_text_preserves_text_without_a_final_channel():
    # Generation cut off inside analysis: the analysis text is all the client
    # has, so emptying it would destroy the response rather than clean it.
    truncated = "analysisI still need to verify the remaining rows"
    assert _harmony_visible_text(truncated, "gpt_oss") == truncated


def test_harmony_visible_text_leaves_other_model_families_alone():
    raw = "assistantfinal is an ordinary word here"
    assert _harmony_visible_text(raw, "qwen3_5") == raw


def test_harmony_visible_text_uses_the_last_final_channel():
    raw = "assistantfinalfirst<|start|>assistant<|channel|>final<|message|>second"
    assert _harmony_visible_text(raw, "gpt_oss") == "second"


def test_harmony_visible_text_ignores_empty_final_channel():
    raw = "analysisreal content here.assistantfinal   "
    assert _harmony_visible_text(raw, "gpt_oss") == raw


def _stream_through_gate(raw: str, chunk: int = 3):
    """Feed text through the gate one small piece at a time, as decode does."""
    gate = _HarmonyChannelGate("gpt_oss")
    streamed = "".join(
        gate.feed(raw[i:i + chunk]) for i in range(0, len(raw), chunk))
    return gate, streamed


def test_harmony_gate_streams_only_the_final_channel():
    raw = ("analysisECHO_R is rated R so exclude.assistantfinalALPHA_G\n"
           "CHARLIE_TVY")
    _gate, streamed = _stream_through_gate(raw)
    assert streamed == "ALPHA_G\nCHARLIE_TVY"
    assert "ECHO_R" not in streamed


def test_harmony_gate_holds_everything_when_no_final_channel_arrives():
    raw = "analysisstill verifying the remaining rows"
    gate, streamed = _stream_through_gate(raw)
    assert streamed == ""
    # Fail-safe: the whole text is owed at end-of-stream, so the client still
    # receives it rather than an empty response.
    assert gate.remainder(raw) == raw


def test_harmony_gate_streamed_text_is_a_prefix_of_the_parsed_content():
    # _MarkerHoldback.final_remainder raises unless this holds, so a trailing
    # newline arriving mid-stream must not outrun the final parsed text.
    raw = "analysisreasoning here.assistantfinal  the answer  \n\n"
    gate, streamed = _stream_through_gate(raw, chunk=1)
    visible = _harmony_visible_text(raw, "gpt_oss")
    assert visible == "the answer"
    assert visible.startswith(streamed)
    assert gate.remainder(visible) == visible[len(streamed):]


def test_harmony_gate_survives_a_marker_split_across_decode_pieces():
    raw = "analysisx.assistantfinalDONE"
    for chunk in (1, 2, 4, 7):
        _gate, streamed = _stream_through_gate(raw, chunk=chunk)
        assert streamed == "DONE", f"chunk={chunk}"


def test_harmony_gate_is_inert_for_other_model_families():
    gate = _HarmonyChannelGate("qwen3_5")
    assert gate.feed("plain text") == "plain text"
    assert gate.remainder("plain text") == ""


def test_harmony_split_channels_reports_reasoning_separately():
    raw = ("<|channel|>analysis<|message|>FOXTROT_KIDS_PG is under /Kids/ so "
           "exclude<|end|><|start|>assistant<|channel|>final<|message|>"
           "ALPHA_G<|return|>")
    visible, analysis = _harmony_split_channels(raw, "gpt_oss")
    assert visible == "ALPHA_G"
    # The reasoning is preserved out-of-band: a model may do its filtering
    # there rather than through a tool argument.
    assert "FOXTROT_KIDS_PG" in analysis
    assert "<|" not in analysis


def test_harmony_split_channels_reports_no_reasoning_without_a_final_channel():
    raw = "analysiscut off mid-thought"
    visible, analysis = _harmony_split_channels(raw, "gpt_oss")
    assert visible == raw
    assert analysis == ""


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  {test.__name__}: PASS")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
