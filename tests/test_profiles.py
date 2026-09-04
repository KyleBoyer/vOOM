"""Pure tests for named runtime profiles (no MLX or model I/O)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from runtime.profiles import (
    PROFILE_SCHEMA,
    RuntimeProfileError,
    active_runtime_profile_fields,
    apply_runtime_profiles,
    clear_active_runtime_profiles,
    discover_runtime_profiles,
    load_runtime_profile,
    parse_runtime_profile_names,
    resolve_runtime_profiles,
    runtime_profile_dirs,
)


ROOT = Path(__file__).resolve().parent.parent


def _write_profile(
    directory: Path,
    name: str,
    *,
    settings: dict | None = None,
    extends: list[str] | None = None,
    notes: list[str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PROFILE_SCHEMA,
        "name": name,
        "description": f"Profile {name}",
    }
    if notes is not None:
        payload["notes"] = notes
    if extends is not None:
        payload["extends"] = extends
    if settings is not None:
        payload["settings"] = settings
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_builtin_profiles_resolve_complete_agent_group():
    catalog = discover_runtime_profiles((ROOT / "profiles",))
    order, settings = resolve_runtime_profiles(
        ("qwen35-a3b-endpoint-packed-agent",), catalog)

    assert order == (
        "qwen35-a3b-endpoint-packed",
        "agent-tool-gateway",
        "qwen35-a3b-endpoint-packed-agent",
    )
    assert settings["VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL"] == "4:128:64"
    assert settings["VMODEL_QWEN_MOE_EXPERT_TOP_K"] == "2"
    assert settings["VMODEL_FAST_TOOL_GATEWAY"] == "1"
    assert settings["VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT"] == "full"
    assert settings["VMODEL_FAST_TOOL_GATEWAY_QWEN_MOE_TOP_K"] == "released"

    mtp_order, mtp_settings = resolve_runtime_profiles(
        ("qwen35-a3b-endpoint-packed-mtp",), catalog)
    assert mtp_order == (
        "qwen35-a3b-endpoint-packed",
        "qwen35-a3b-endpoint-packed-mtp",
    )
    assert mtp_settings["VMODEL_QWEN_MTP_SPECULATIVE"] == "1"
    assert mtp_settings["VMODEL_QWEN_MTP_MIN_OUTPUT_TOKENS"] == "2"
    assert mtp_settings["VMODEL_QWEN_MTP_STOCHASTIC_DRAFT_TOP_K"] == "4"

    rerank_order, rerank_settings = resolve_runtime_profiles(
        ("qwen35-a3b-endpoint-packed-head-rerank64",), catalog)
    assert rerank_order == (
        "qwen35-a3b-endpoint-packed",
        "qwen35-a3b-endpoint-packed-head-rerank64",
    )
    assert rerank_settings["VMODEL_QWEN35_RERANK_LM_HEAD"] == "1"
    assert rerank_settings[
        "VMODEL_QWEN35_RERANK_LM_HEAD_CANDIDATES"] == "64"

    huihui_order, huihui_settings = resolve_runtime_profiles(
        ("huihui-qwen38-27b-fast-agent",), catalog)
    assert huihui_order == (
        "agent-tool-gateway",
        "huihui-qwen38-27b-fast-agent",
    )
    assert huihui_settings[
        "VMODEL_QWEN35_PREFILL_CHUNK_CEILING"] == "128"
    assert huihui_settings["VMODEL_QWEN35_MIN_AVAILABLE_MB"] == "5300"
    assert huihui_settings[
        "VMODEL_QWEN35_POSTGEN_MIN_AVAILABLE_MB"] == "5300"
    assert huihui_settings["VMODEL_QWEN_MTP_DEPTH"] == "4"
    assert huihui_settings[
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_TOKENS"] == "128"
    assert huihui_settings[
        "VMODEL_QWEN_MTP_PROMPT_HISTORY_MIN_PROMPT_TOKENS"] == "4096"
    assert huihui_settings[
        "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD"] == "1"
    assert huihui_settings[
        "VMODEL_QWEN35_SERIAL_VERIFY_SUSPEND_LM_HEAD_MIN_PROMPT_TOKENS"] == (
            "4096")
    assert huihui_settings[
        "VMODEL_QWEN_MTP_GRAMMAR_AWARE_DRAFT"] == "1"
    assert huihui_settings[
        "VMODEL_QWEN35_SERIAL_VERIFY_BATCHED_MLP"] == "1"
    assert huihui_settings[
        "VMODEL_QWEN_MTP_STOCHASTIC_DRAFT_TOP_K"] == "1"
    assert huihui_settings["VMODEL_QWEN35_PIN_LM_HEAD"] == "1"
    assert huihui_settings["VMODEL_QWEN35_PREFETCH_DEPTH"] == "2"
    assert huihui_settings[
        "VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST"] == "1"
    assert huihui_settings[
        "VMODEL_QWEN35_REUSABLE_USER_PREFIX"] == "1"

    huihui_quant_order, huihui_quant_settings = resolve_runtime_profiles(
        ("huihui-qwen38-27b-fast-agent-mtpquant",), catalog)
    assert huihui_quant_order == (
        "agent-tool-gateway",
        "huihui-qwen38-27b-fast-agent",
        "huihui-qwen38-27b-fast-agent-mtpquant",
    )
    assert huihui_quant_settings["VMODEL_QWEN_MTP_DEPTH"] == "4"
    assert huihui_quant_settings[
        "VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL"] == "16:1024"
    assert huihui_quant_settings[
        "VMODEL_FAST_TOOL_GATEWAY_MIN_TOOLS"] == "1"

    flash_order, flash_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-instrumented-lossless",), catalog)
    assert flash_order == ("qwen38-flash-next-instrumented-lossless",)
    assert flash_settings["VMODEL_QWEN4_WEIGHT_CACHE_MB"] == "400"
    assert flash_settings["VMODEL_QWEN4_PREFETCH_DEPTH"] == "1"
    assert flash_settings["VMODEL_QWEN4_MTP_DEPTH"] == "3"
    assert flash_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_SUSPEND_LM_HEAD"] == "1"
    assert flash_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_EXACT_BF16_GEMV"] == "1"
    assert flash_settings["VMODEL_QWEN4_PARALLEL_STORAGE_READS"] == "1"
    assert flash_settings["VMODEL_QWEN4_FAST_TIER_DECODE_ONLY"] == "1"

    jang_order, jang_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-crack-jang6s",), catalog)
    assert jang_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-crack-jang6s",
    )
    assert jang_settings["VMODEL_QWEN4_MTP_DEPTH"] == "0"
    assert jang_settings["VMODEL_QWEN4_COMPILED_DELTA"] == "1"
    assert jang_settings["VMODEL_QWEN4_PLE_READ_WORKERS"] == "8"
    assert jang_settings["VMODEL_QWEN4_PARALLEL_STORAGE_READS"] == "0"
    assert jang_settings["VMODEL_QWEN4_HOT_PROMPT_KV"] == "0"

    jang_fused_order, jang_fused_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-crack-jang6s-exact-fused-delta",), catalog)
    assert jang_fused_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-crack-jang6s",
        "qwen38-flash-next-crack-jang6s-exact-fused-delta",
    )
    assert jang_fused_settings["VMODEL_QWEN4_COMPILED_DELTA"] == "0"
    assert jang_fused_settings[
        "VMODEL_QWEN4_NATIVE_FUSED_DELTA_PREFILL"] == "1"
    assert jang_fused_settings[
        "VMODEL_QWEN4_SPARSE_EXPERT_BATCH_ROWS"] == "0"
    assert jang_fused_settings["VMODEL_QWEN4_MTP_DEPTH"] == "0"

    jang_prefetch_order, jang_prefetch_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-crack-jang6s-exact-fused-delta-prefetch",),
        catalog,
    )
    assert jang_prefetch_order[-2:] == (
        "qwen38-flash-next-crack-jang6s-exact-fused-delta",
        "qwen38-flash-next-crack-jang6s-exact-fused-delta-prefetch",
    )
    assert jang_prefetch_settings[
        "VMODEL_QWEN4_EXPERT_BATCH_PREFETCH_PREFILL_ONLY"] == "1"
    assert jang_prefetch_settings[
        "VMODEL_QWEN4_NATIVE_FUSED_DELTA_PREFILL"] == "1"

    qwen_exact_fused_order, qwen_exact_fused_settings = (
        resolve_runtime_profiles((
            "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-"
            "direct-fp8-qmv-prefill-pipeline-exact-fused-delta",
        ), catalog))
    assert qwen_exact_fused_order[-2:] == (
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-"
        "direct-fp8-qmv-prefill-pipeline",
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-"
        "direct-fp8-qmv-prefill-pipeline-exact-fused-delta",
    )
    assert qwen_exact_fused_settings[
        "VMODEL_QWEN4_COMPILED_DELTA"] == "0"
    assert qwen_exact_fused_settings[
        "VMODEL_QWEN4_NATIVE_FUSED_DELTA_PREFILL"] == "1"

    glm_flash_order, glm_flash_settings = resolve_runtime_profiles(
        ("glm53-flash-lossless-compiled-kda",), catalog)
    assert glm_flash_order == ("glm53-flash-lossless-compiled-kda",)
    assert glm_flash_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "1"
    assert glm_flash_settings[
        "VMODEL_GLM53_INCREMENTAL_DSA_POOL"] == "1"
    assert "VMODEL_GLM53_SPARSE_FUSED_ATTENTION" not in glm_flash_settings
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        glm_flash_settings)

    glm_segment16_order, glm_segment16_settings = resolve_runtime_profiles(
        ("glm53-flash-lossless-compiled-kda-segment16",), catalog)
    assert glm_segment16_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-lossless-compiled-kda-segment16",
    )
    assert glm_segment16_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "1"
    assert glm_segment16_settings[
        "VMODEL_GLM53_COMPILED_KDA_SEGMENT"] == "16"

    glm_flash_prefetch_order, glm_flash_prefetch_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-lossless-expert-prefetch",), catalog))
    assert glm_flash_prefetch_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-lossless-expert-prefetch",
    )
    assert glm_flash_prefetch_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "1"
    assert glm_flash_prefetch_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert "VMODEL_GLM53_SPARSE_FUSED_ATTENTION" not in (
        glm_flash_prefetch_settings)

    glm_flash_batch8_order, glm_flash_batch8_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-lossless-expert-prefetch-batch8",), catalog))
    assert glm_flash_batch8_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-lossless-expert-prefetch",
        "glm53-flash-lossless-expert-prefetch-batch8",
    )
    assert glm_flash_batch8_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_flash_batch8_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert "VMODEL_GLM53_SPARSE_FUSED_ATTENTION" not in (
        glm_flash_batch8_settings)
    assert "VMODEL_GLM53_SPARSE_FUSED_KV_INT8" not in (
        glm_flash_batch8_settings)
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        glm_flash_batch8_settings)

    glm_direct_order, glm_direct_settings = resolve_runtime_profiles(
        ("glm53-flash-lossless-native-mtp3-workers2-direct-fp8-qmv",),
        catalog,
    )
    assert glm_direct_order[-2:] == (
        "glm53-flash-lossless-native-mtp3-workers2",
        "glm53-flash-lossless-native-mtp3-workers2-direct-fp8-qmv",
    )
    assert glm_direct_settings["VMODEL_GLM53_FP8_DIRECT_QMV"] == "1"
    assert glm_direct_settings["VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_direct_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "2"
    assert glm_direct_settings["VMODEL_GLM53_MTP_DEPTH"] == "3"

    glm_decode_direct_order, glm_decode_direct_settings = (
        resolve_runtime_profiles((
            "glm53-flash-lossless-native-mtp3-workers2-"
            "direct-fp8-qmv-decode-only",
        ), catalog))
    assert glm_decode_direct_order[-2:] == (
        "glm53-flash-lossless-native-mtp3-workers2-direct-fp8-qmv",
        "glm53-flash-lossless-native-mtp3-workers2-"
        "direct-fp8-qmv-decode-only",
    )
    assert glm_decode_direct_settings[
        "VMODEL_GLM53_FP8_DIRECT_QMV"] == "1"
    assert glm_decode_direct_settings[
        "VMODEL_GLM53_FP8_DIRECT_QMV_DECODE_ONLY"] == "1"

    qwen_direct_order, qwen_direct_settings = resolve_runtime_profiles((
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-"
        "direct-fp8-qmv-decode-only",
    ), catalog)
    assert qwen_direct_order[-2:] == (
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp",
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-"
        "direct-fp8-qmv-decode-only",
    )
    assert qwen_direct_settings["VMODEL_QWEN4_FP8_DIRECT_QMV"] == "1"
    assert qwen_direct_settings[
        "VMODEL_QWEN4_FP8_DIRECT_QMV_DECODE_ONLY"] == "1"

    glm_absorbed_order, glm_absorbed_settings = resolve_runtime_profiles(
        ("glm53-flash-e-compact-mla",), catalog)
    assert glm_absorbed_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-e-compact-mla",
    )
    assert glm_absorbed_settings[
        "VMODEL_GLM53_SPARSE_ABSORBED_MLA"] == "1"
    assert glm_absorbed_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "0"
    assert "VMODEL_GLM53_SPARSE_FUSED_ATTENTION" not in (
        glm_absorbed_settings)
    assert "VMODEL_GLM53_SPARSE_FUSED_KV_INT8" not in (
        glm_absorbed_settings)

    glm_compact_coalesced_order, glm_compact_coalesced_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-e-compact-mla-coalesced-batch8",), catalog))
    assert glm_compact_coalesced_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-e-compact-mla",
        "glm53-flash-lossless-expert-prefetch",
        "glm53-flash-lossless-expert-prefetch-batch8",
        "glm53-flash-e-compact-mla-prefetch-batch8",
        "glm53-flash-e-compact-mla-coalesced-batch8",
    )
    assert glm_compact_coalesced_settings[
        "VMODEL_GLM53_SPARSE_ABSORBED_MLA"] == "1"
    assert glm_compact_coalesced_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_compact_coalesced_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert glm_compact_coalesced_settings[
        "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS"] == "1"
    assert glm_compact_coalesced_settings[
        "VMODEL_GLM53_COALESCED_EXPERT_MAX_POSITIONS"] == "512"
    assert "VMODEL_GLM53_SPARSE_FUSED_ATTENTION" not in (
        glm_compact_coalesced_settings)
    assert "VMODEL_GLM53_SPARSE_FUSED_KV_INT8" not in (
        glm_compact_coalesced_settings)

    glm_compact_tile64_order, glm_compact_tile64_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-e-compact-mla-tile64",), catalog))
    assert glm_compact_tile64_order[-2:] == (
        "glm53-flash-e-compact-mla-prefetch-batch8",
        "glm53-flash-e-compact-mla-tile64",
    )
    assert glm_compact_tile64_settings[
        "VMODEL_GLM53_SPARSE_ABSORBED_QUERY_TILE_SIZE"] == "64"
    assert glm_compact_tile64_settings[
        "VMODEL_GLM53_PREFILL_TILE_WIDTH"] == "64"
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        glm_compact_tile64_settings)

    glm_compact_tile128_query_order, glm_compact_tile128_query_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-e-compact-mla-tile128",), catalog))
    assert glm_compact_tile128_query_order[-2:] == (
        "glm53-flash-e-compact-mla-tile64",
        "glm53-flash-e-compact-mla-tile128",
    )
    assert glm_compact_tile128_query_settings[
        "VMODEL_GLM53_SPARSE_ABSORBED_QUERY_TILE_SIZE"] == "128"
    assert glm_compact_tile128_query_settings[
        "VMODEL_GLM53_PREFILL_TILE_WIDTH"] == "128"
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        glm_compact_tile128_query_settings)

    tile128_workers2_order, tile128_workers2_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-e-compact-mla-tile128-workers2",), catalog))
    assert tile128_workers2_order[-2:] == (
        "glm53-flash-e-compact-mla-tile128",
        "glm53-flash-e-compact-mla-tile128-workers2",
    )
    assert tile128_workers2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "2"
    assert tile128_workers2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] == "2"
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        tile128_workers2_settings)

    glm_prefetch2_order, glm_prefetch2_settings = resolve_runtime_profiles(
        ("glm53-flash-e-compact-mla-coalesced-prefetch2",), catalog)
    assert glm_prefetch2_order[-2:] == (
        "glm53-flash-e-compact-mla-coalesced-batch8",
        "glm53-flash-e-compact-mla-coalesced-prefetch2",
    )
    assert glm_prefetch2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "2"
    assert glm_prefetch2_settings[
        "VMODEL_GLM53_PREFILL_TILE_WIDTH"] == "32"

    glm_prefetch2_workers2_order, glm_prefetch2_workers2_settings = (
        resolve_runtime_profiles((
            "glm53-flash-e-compact-mla-coalesced-prefetch2-workers2",),
            catalog))
    assert glm_prefetch2_workers2_order[-2:] == (
        "glm53-flash-e-compact-mla-coalesced-prefetch2",
        "glm53-flash-e-compact-mla-coalesced-prefetch2-workers2",
    )
    assert glm_prefetch2_workers2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "2"
    assert glm_prefetch2_workers2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] == "2"

    glm_compact_tile128_order, glm_compact_tile128_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-e-compact-mla-coalesced-tile128",), catalog))
    assert glm_compact_tile128_order[-2:] == (
        "glm53-flash-e-compact-mla-coalesced-batch8",
        "glm53-flash-e-compact-mla-coalesced-tile128",
    )
    assert glm_compact_tile128_settings[
        "VMODEL_GLM53_PREFILL_TILE_WIDTH"] == "128"
    assert glm_compact_tile128_settings[
        "VMODEL_GLM53_COALESCED_EXPERT_MAX_POSITIONS"] == "512"
    assert "VMODEL_GLM53_SPARSE_FUSED_KV_INT8" not in (
        glm_compact_tile128_settings)

    glm_fast_order, glm_fast_settings = resolve_runtime_profiles(
        ("glm53-flash-long-context-fast",), catalog)
    assert glm_fast_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-long-context-fast",
    )
    assert glm_fast_settings["VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "1"
    assert glm_fast_settings["VMODEL_GLM53_PREFILL_TILE_WIDTH"] == "128"
    assert glm_fast_settings["VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_fast_settings["VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert glm_fast_settings[
        "VMODEL_GLM53_SPARSE_FUSED_ATTENTION"] == "1"
    assert glm_fast_settings["VMODEL_GLM53_SPARSE_FUSED_KV_INT8"] == "1"
    assert glm_fast_settings[
        "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS"] == "1"
    assert glm_fast_settings[
        "VMODEL_GLM53_COALESCED_EXPERT_MAX_POSITIONS"] == "512"

    glm_native_order, glm_native_settings = resolve_runtime_profiles(
        ("glm53-flash-long-context-native-kda",), catalog)
    assert glm_native_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-long-context-fast",
        "glm53-flash-long-context-native-kda",
    )
    assert glm_native_settings["VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "0"
    assert glm_native_settings[
        "VMODEL_GLM53_NATIVE_FUSED_KDA_PREFILL"] == "1"

    glm_striped_order, glm_striped_settings = resolve_runtime_profiles(
        ("glm53-flash-lossless-striped-kda",), catalog)
    assert glm_striped_order[-2:] == (
        "glm53-flash-lossless-expert-prefetch-batch8-workers2",
        "glm53-flash-lossless-striped-kda",
    )
    assert glm_striped_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "0"
    assert glm_striped_settings[
        "VMODEL_GLM53_NATIVE_FUSED_KDA_PREFILL"] == "1"

    glm_striped_long_order, glm_striped_long_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-lossless-striped-kda-long-context",), catalog))
    assert glm_striped_long_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-lossless-striped-kda-long-context",
    )
    assert glm_striped_long_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "0"
    assert glm_striped_long_settings[
        "VMODEL_GLM53_NATIVE_FUSED_KDA_PREFILL"] == "1"

    glm_disk_spool_order, glm_disk_spool_settings = (
        resolve_runtime_profiles((
            "glm53-flash-lossless-striped-kda-disk-spool-long-context",
        ), catalog))
    assert glm_disk_spool_order[-2:] == (
        "glm53-flash-lossless-striped-kda-long-context",
        "glm53-flash-lossless-striped-kda-disk-spool-long-context",
    )
    assert glm_disk_spool_settings[
        "VMODEL_GLM53_LAYER_STATIONARY_HOST_SPOOL"] == "0"
    assert glm_disk_spool_settings[
        "VMODEL_GLM53_LAYER_STATIONARY_DISK_SPOOL_DIR"] == (
            "/Volumes/Workspace NVME/vmodel-spill/glm53-flash-activations")
    assert glm_disk_spool_settings[
        "VMODEL_GLM53_EXPANDED_KV_HOST_SPOOL"] == "1"
    assert glm_disk_spool_settings[
        "VMODEL_GLM53_NATIVE_FUSED_KDA_PREFILL"] == "1"
    assert glm_disk_spool_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "0"
    for lossy_setting in (
            "VMODEL_GLM53_SPARSE_FUSED_ATTENTION",
            "VMODEL_GLM53_SPARSE_FUSED_KV_INT8",
            "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS"):
        assert lossy_setting not in glm_disk_spool_settings

    glm_disk_prefetch_order, glm_disk_prefetch_settings = (
        resolve_runtime_profiles((
            "glm53-flash-lossless-striped-kda-disk-spool-prefetch-"
            "long-context",
        ), catalog))
    assert glm_disk_prefetch_order[-2:] == (
        "glm53-flash-lossless-striped-kda-disk-spool-long-context",
        "glm53-flash-lossless-striped-kda-disk-spool-prefetch-long-context",
    )
    assert glm_disk_prefetch_settings[
        "VMODEL_GLM53_EXPANDED_KV_HOST_SPOOL"] == "1"
    assert glm_disk_prefetch_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_disk_prefetch_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert glm_disk_prefetch_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "2"
    assert glm_disk_prefetch_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] == "2"
    for lossy_setting in (
            "VMODEL_GLM53_SPARSE_FUSED_ATTENTION",
            "VMODEL_GLM53_SPARSE_FUSED_KV_INT8",
            "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS"):
        assert lossy_setting not in glm_disk_prefetch_settings

    glm_isolated_kda_order, glm_isolated_kda_settings = (
        resolve_runtime_profiles(
            ("glm53-flash-lossy-native-kda-isolated",), catalog))
    assert glm_isolated_kda_order[-2:] == (
        "glm53-flash-lossless-striped-kda",
        "glm53-flash-lossy-native-kda-isolated",
    )
    assert glm_isolated_kda_settings[
        "VMODEL_GLM53_COMPILED_KDA_PREFILL"] == "0"
    assert glm_isolated_kda_settings[
        "VMODEL_GLM53_NATIVE_FUSED_KDA_PREFILL"] == "1"
    assert "VMODEL_GLM53_SPARSE_FUSED_KV_INT8" not in (
        glm_isolated_kda_settings)
    assert "VMODEL_GLM53_COALESCED_EXPERT_POSITIONS" not in (
        glm_isolated_kda_settings)

    glm_fp8_hybrid_order, glm_fp8_hybrid_settings = resolve_runtime_profiles(
        ("glm53-flash-lossless-expert-prefetch-batch8-workers2-"
         "native-fp8-foreground",), catalog)
    assert glm_fp8_hybrid_order[-2:] == (
        "glm53-flash-lossless-expert-prefetch-batch8-workers2-native-fp8",
        "glm53-flash-lossless-expert-prefetch-batch8-workers2-"
        "native-fp8-foreground",
    )
    assert glm_fp8_hybrid_settings[
        "VMODEL_GLM53_NATIVE_FP8_DEQUANT"] == "1"
    assert glm_fp8_hybrid_settings[
        "VMODEL_GLM53_NATIVE_FP8_PREFETCH"] == "0"

    glm_dflash_order, glm_dflash_settings = resolve_runtime_profiles(
        ("glm53-flash-dflash2-e-fast",), catalog)
    assert glm_dflash_order == (
        "glm53-flash-lossless-compiled-kda",
        "glm53-flash-dflash2-e-fast",
    )
    assert "VMODEL_QWEN_DFLASH2_DRAFT" not in glm_dflash_settings
    assert glm_dflash_settings[
        "VMODEL_QWEN_DFLASH2_MAX_PROMPT_TOKENS"] == "1048576"
    assert glm_dflash_settings[
        "VMODEL_QWEN_DFLASH2_PROPOSAL_POLICY"] == "selector"
    assert glm_dflash_settings[
        "VMODEL_QWEN_DFLASH2_RELEASE_AFTER_ROUND"] == "1"
    assert glm_dflash_settings[
        "VMODEL_QWEN_DFLASH2_FUSED_DYNAMIC_CONV"] == "0"

    glm_full_order, glm_full_settings = resolve_runtime_profiles(
        ("glm53-full-lossless-long-context",), catalog)
    assert glm_full_order == ("glm53-full-lossless-long-context",)
    assert glm_full_settings["VMODEL_GLM_DSA_LONG_CONTEXT"] == "1"
    assert glm_full_settings[
        "VMODEL_GLM_DSA_SPARSE_ABSORBED_MLA"] == "1"
    assert glm_full_settings["VMODEL_GLM_DSA_KEY_TILE_SIZE"] == "1024"
    assert glm_full_settings["VMODEL_GLM_DSA_PREFILL_TILE_WIDTH"] == "32"
    assert glm_full_settings[
        "VMODEL_GLM_DSA_SELECTION_QUERY_TILE_SIZE"] == "64"
    assert glm_full_settings["VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "1"
    assert glm_full_settings["VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "0"

    glm_full_direct_order, glm_full_direct_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-direct-fp8-qmv",), catalog))
    assert glm_full_direct_order == (
        "glm53-full-lossless-direct-fp8-qmv",)
    assert glm_full_direct_settings[
        "VMODEL_GLM53_FP8_DIRECT_QMV"] == "1"
    assert glm_full_direct_settings[
        "VMODEL_GLM53_TRUNK_PREFETCH_DEPTH"] == "1"
    assert "VMODEL_GLM53_MTP" not in glm_full_direct_settings
    assert "VMODEL_GLM53_EXPERT_BATCH_PREFETCH" not in (
        glm_full_direct_settings)

    glm_full_pipeline_order, glm_full_pipeline_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-direct-fp8-qmv-expert-pipeline",),
            catalog))
    assert glm_full_pipeline_order == (
        "glm53-full-lossless-direct-fp8-qmv",
        "glm53-full-lossless-direct-fp8-qmv-expert-pipeline",
    )
    assert glm_full_pipeline_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert glm_full_pipeline_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_DEPTH"] == "1"
    assert glm_full_pipeline_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH_WORKERS"] == "1"
    assert glm_full_pipeline_settings[
        "VMODEL_GLM53_SHORT_WEIGHT_CACHE_MB"] == "2500"
    assert glm_full_pipeline_settings[
        "VMODEL_GLM53_SHORT_STREAM_LM_HEAD"] == "1"

    glm_full_decode_direct_order, glm_full_decode_direct_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-direct-fp8-qmv-decode-only",), catalog))
    assert glm_full_decode_direct_order == (
        "glm53-full-lossless-direct-fp8-qmv",
        "glm53-full-lossless-direct-fp8-qmv-decode-only",
    )
    assert glm_full_decode_direct_settings[
        "VMODEL_GLM53_FP8_DIRECT_QMV_DECODE_ONLY"] == "1"

    glm_full_native_order, glm_full_native_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-long-context-native-fp8",), catalog))
    assert glm_full_native_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-long-context-native-fp8",
    )
    assert glm_full_native_settings[
        "VMODEL_GLM53_NATIVE_FP8_DEQUANT"] == "1"
    assert glm_full_native_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "0"
    assert glm_full_native_settings[
        "VMODEL_GLM_DSA_PREFILL_TILE_WIDTH"] == "32"

    glm_preallocate_order, glm_preallocate_settings = resolve_runtime_profiles(
        ("glm53-full-lossless-index-preallocate",), catalog)
    assert glm_preallocate_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-index-preallocate",
    )
    assert glm_preallocate_settings[
        "VMODEL_GLM_DSA_INDEX_PREALLOCATE"] == "1"

    glm_full_composed_order, glm_full_composed_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-preallocate-prefetch-batch8",), catalog))
    assert glm_full_composed_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-index-preallocate",
        "glm53-full-lossless-preallocate-prefetch-batch8",
    )
    assert glm_full_composed_settings[
        "VMODEL_GLM_DSA_INDEX_PREALLOCATE"] == "1"
    assert glm_full_composed_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "8"
    assert glm_full_composed_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"
    assert glm_full_composed_settings[
        "VMODEL_GLM_DSA_SELECTION_QUERY_TILE_SIZE"] == "64"
    assert "VMODEL_GLM_DSA_KEY_TILE_SIZE" in glm_full_composed_settings

    glm_full_batch4_order, glm_full_batch4_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-preallocate-prefetch-batch4",), catalog))
    assert glm_full_batch4_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-index-preallocate",
        "glm53-full-lossless-preallocate-prefetch-batch4",
    )
    assert glm_full_batch4_settings[
        "VMODEL_GLM_DSA_INDEX_PREALLOCATE"] == "1"
    assert glm_full_batch4_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "4"
    assert glm_full_batch4_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"

    glm_full_batch2_order, glm_full_batch2_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-preallocate-prefetch-batch2",), catalog))
    assert glm_full_batch2_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-index-preallocate",
        "glm53-full-lossless-preallocate-prefetch-batch2",
    )
    assert glm_full_batch2_settings[
        "VMODEL_GLM_DSA_INDEX_PREALLOCATE"] == "1"
    assert glm_full_batch2_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "2"
    assert glm_full_batch2_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"

    glm_full_batch1_order, glm_full_batch1_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-preallocate-prefetch-batch1",), catalog))
    assert glm_full_batch1_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-index-preallocate",
        "glm53-full-lossless-preallocate-prefetch-batch1",
    )
    assert glm_full_batch1_settings[
        "VMODEL_GLM_DSA_INDEX_PREALLOCATE"] == "1"
    assert glm_full_batch1_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "1"
    assert glm_full_batch1_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"

    glm_full_prefetch_order, glm_full_prefetch_settings = (
        resolve_runtime_profiles(
            ("glm53-full-lossless-expert-prefetch",), catalog))
    assert glm_full_prefetch_order == (
        "glm53-full-lossless-long-context",
        "glm53-full-lossless-expert-prefetch",
    )
    assert glm_full_prefetch_settings[
        "VMODEL_GLM53_EXPERT_FETCH_BATCH"] == "1"
    assert glm_full_prefetch_settings[
        "VMODEL_GLM53_EXPERT_BATCH_PREFETCH"] == "1"

    candidate_order, candidate_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-candidate-bf16",), catalog)
    assert candidate_order == ("qwen38-flash-next-candidate-bf16",)
    assert candidate_settings["VMODEL_QWEN4_PREFILL_CHUNK"] == "1024"
    assert candidate_settings["VMODEL_QWEN4_COMPILED_DELTA"] == "1"
    assert candidate_settings["VMODEL_QWEN4_HOT_PROMPT_KV"] == "0"
    assert candidate_settings["VMODEL_QWEN4_MTP_DEPTH"] == "0"
    assert "VMODEL_QWEN4_FAST_TIER_DIR" not in candidate_settings

    uncensored_fp8_order, uncensored_fp8_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-uncensored-fp8",), catalog)
    assert uncensored_fp8_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-uncensored-fp8",
    )
    assert uncensored_fp8_settings[
        "VMODEL_QWEN4_NATIVE_FP8_DEQUANT"] == "1"
    assert "VMODEL_QWEN4_FAST_TIER_DIR" not in uncensored_fp8_settings

    uncensored_mtp_order, uncensored_mtp_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-uncensored-fp8-mtp",), catalog)
    assert uncensored_mtp_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-uncensored-fp8",
        "qwen38-flash-next-uncensored-fp8-mtp",
    )
    assert uncensored_mtp_settings[
        "VMODEL_QWEN4_NATIVE_FP8_DEQUANT"] == "1"
    assert uncensored_mtp_settings["VMODEL_QWEN4_MTP_DEPTH"] == "3"
    assert uncensored_mtp_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_SUSPEND_LM_HEAD"] == "1"
    assert uncensored_mtp_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_EXACT_BF16_GEMV"] == "1"
    assert uncensored_mtp_settings["VMODEL_QWEN4_FAST_TIER_DECODE_ONLY"] == "0"
    assert "VMODEL_QWEN4_FAST_TIER_DIR" not in uncensored_mtp_settings
    assert uncensored_mtp_settings["VMODEL_QWEN4_HOT_PROMPT_KV"] == "0"

    uncensored_fast_order, uncensored_fast_settings = (
        resolve_runtime_profiles(
            ("qwen38-flash-next-uncensored-fp8-fast-tier-mtp",),
            catalog))
    assert uncensored_fast_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-uncensored-fp8",
        "qwen38-flash-next-uncensored-fp8-mtp",
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp",
    )
    assert uncensored_fast_settings[
        "VMODEL_QWEN4_FAST_TIER_DIR"].endswith(
            "Qwen3.8-Flash-Next-uncensored-fp8-plex-hot-all-corpus3")
    assert uncensored_fast_settings[
        "VMODEL_QWEN4_PARALLEL_STORAGE_READS"] == "1"
    assert uncensored_fast_settings[
        "VMODEL_QWEN4_FAST_TIER_DECODE_ONLY"] == "0"
    assert uncensored_fast_settings[
        "VMODEL_QWEN4_WEIGHT_CACHE_MB"] == "256"
    assert uncensored_fast_settings["VMODEL_QWEN4_MTP_DEPTH"] == "3"

    uncensored_hot_order, uncensored_hot_settings = (
        resolve_runtime_profiles(
            ("qwen38-flash-next-uncensored-fp8-fast-tier-mtp-hot-kv",),
            catalog))
    assert uncensored_hot_order[-2:] == (
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp",
        "qwen38-flash-next-uncensored-fp8-fast-tier-mtp-hot-kv",
    )
    assert uncensored_hot_settings["VMODEL_QWEN4_HOT_PROMPT_KV"] == "1"
    assert uncensored_hot_settings["VMODEL_QWEN4_HOT_KV_SLOTS"] == "2"
    assert uncensored_hot_settings[
        "VMODEL_QWEN4_HOT_KV_MIN_TOKENS"] == "16"
    assert uncensored_hot_settings["VMODEL_QWEN4_HOT_KV_PERSIST_DIR"] == ""

    candidate_fast_order, candidate_fast_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-candidate-fast-tier",), catalog)
    assert candidate_fast_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-candidate-fast-tier",
    )
    assert candidate_fast_settings["VMODEL_QWEN4_FAST_TIER_DIR"].endswith(
        "Qwen3.8-Flash-Next-Abliterated-trace-v3-hot24")
    assert candidate_fast_settings[
        "VMODEL_QWEN4_PARALLEL_STORAGE_READS"] == "1"
    assert candidate_fast_settings[
        "VMODEL_QWEN4_FAST_TIER_DECODE_ONLY"] == "0"
    assert candidate_fast_settings["VMODEL_QWEN4_MTP_DEPTH"] == "0"

    candidate_mtp_order, candidate_mtp_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-candidate-fast-tier-mtp",), catalog)
    assert candidate_mtp_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-candidate-fast-tier",
        "qwen38-flash-next-candidate-fast-tier-mtp",
    )
    assert candidate_mtp_settings["VMODEL_QWEN4_MTP_DEPTH"] == "3"
    assert candidate_mtp_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_SUSPEND_LM_HEAD"] == "1"
    assert candidate_mtp_settings[
        "VMODEL_QWEN4_SERIAL_VERIFY_EXACT_BF16_GEMV"] == "1"
    assert "VMODEL_QWEN4_MTP_Q_CALIBRATION_SCALES" not in (
        candidate_mtp_settings)

    candidate_hot_order, candidate_hot_settings = resolve_runtime_profiles(
        ("qwen38-flash-next-candidate-fast-tier-mtp-hot-kv",), catalog)
    assert candidate_hot_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-candidate-fast-tier",
        "qwen38-flash-next-candidate-fast-tier-mtp",
        "qwen38-flash-next-candidate-fast-tier-mtp-hot-kv",
    )
    assert candidate_hot_settings["VMODEL_QWEN4_HOT_PROMPT_KV"] == "1"
    assert candidate_hot_settings["VMODEL_QWEN4_HOT_KV_SLOTS"] == "2"
    assert candidate_hot_settings[
        "VMODEL_QWEN4_HOT_KV_MIN_TOKENS"] == "16"

    candidate_compact_order, candidate_compact_settings = (
        resolve_runtime_profiles(
            ("qwen38-flash-next-candidate-fast-tier-mtp-compact-kda",),
            catalog))
    assert candidate_compact_order == (
        "qwen38-flash-next-candidate-bf16",
        "qwen38-flash-next-candidate-fast-tier",
        "qwen38-flash-next-candidate-fast-tier-mtp",
        "qwen38-flash-next-candidate-fast-tier-mtp-compact-kda",
    )
    assert candidate_compact_settings[
        "VMODEL_QWEN4_MTP_COMPACT_KDA_ROLLBACK"] == "1"
    assert candidate_compact_settings["VMODEL_QWEN4_MTP_DEPTH"] == "3"

    long_order, long_settings = resolve_runtime_profiles(
        ("huihui-qwen38-27b-fast-long-context",), catalog)
    assert long_order == (
        "agent-tool-gateway",
        "huihui-qwen38-27b-fast-agent",
        "huihui-qwen38-27b-fast-long-context",
    )
    assert long_settings[
        "VMODEL_QWEN35_LOSSY_SUFFIX_PREFILL"] == "12:128:1024"
    assert long_settings[
        "VMODEL_QWEN35_MIXED_DEPTH_HOT_KV_PERSIST"] == "0"
    assert long_settings["VMODEL_QWEN35_KV_MAX_MB"] == "64"
    assert long_settings[
        "VMODEL_QWEN35_KV_PAGE_POSITIONS"] == "1024"
    assert long_settings[
        "VMODEL_QWEN35_PAGED_ONLINE_ATTENTION"] == "1"
    assert "VMODEL_QWEN35_PAGED_ONLINE_PAGE_NATIVE" not in long_settings
    assert long_settings[
        "VMODEL_QWEN35_PAGED_ONLINE_TILE_POSITIONS"] == "4096"
    assert long_settings["VMODEL_QWEN_MTP_DEPTH"] == "4"
    assert long_settings[
        "VMODEL_QWEN_MTP_SELECTIVE_TREE_MARGIN"] == "0"

    mtpquant_order, mtpquant_settings = resolve_runtime_profiles(
        ("huihui-qwen38-27b-fast-long-context-mtpquant",), catalog)
    assert mtpquant_order == (
        "agent-tool-gateway",
        "huihui-qwen38-27b-fast-agent",
        "huihui-qwen38-27b-fast-long-context",
        "huihui-qwen38-27b-fast-long-context-mtpquant",
    )
    assert mtpquant_settings["VMODEL_QWEN_MTP_DEPTH"] == "7"
    assert mtpquant_settings[
        "VMODEL_QWEN35_PAGED_ONLINE_TILE_POSITIONS"] == "4096"

    persist_order, persist_settings = resolve_runtime_profiles(
        ("huihui-qwen38-27b-lossless-paged-persist",), catalog)
    assert persist_order == (
        "huihui-qwen38-27b-lossless",
        "huihui-qwen38-27b-lossless-paged-persist",
    )
    assert persist_settings["VMODEL_QWEN35_KV_MAX_MB"] == "768"
    assert persist_settings["VMODEL_QWEN35_PAGED_KV_PERSIST"] == "1"
    assert persist_settings[
        "VMODEL_QWEN35_FUSED_BOUNDARY_SCAFFOLD_PREFILL"] == "0"

def test_builtin_k3_and_qwen9_profiles_pin_validated_values():
    catalog = discover_runtime_profiles((ROOT / "profiles",))

    qwen_order, qwen_settings = resolve_runtime_profiles(
        ("qwen35-9b-depth-adaptive-agent",), catalog)
    assert qwen_order == (
        "qwen35-9b-depth-adaptive",
        "agent-tool-gateway",
        "qwen35-9b-depth-adaptive-agent",
    )
    assert qwen_settings["VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL"] == "8:256"
    assert qwen_settings["VMODEL_MLX_LM_SYSTEM_RESERVE_MB"] == "1500"

    balanced_order, balanced_settings = resolve_runtime_profiles(
        ("qwen35-9b-tool-balanced-agent",), catalog)
    assert balanced_order == (
        "qwen35-9b-depth-adaptive",
        "agent-tool-gateway",
        "qwen35-9b-tool-balanced-agent",
    )
    assert balanced_settings["VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL"] == "16:1024"
    assert balanced_settings[
        "VMODEL_FAST_TOOL_GATEWAY_SUFFIX_CONTRACT"] == "1"
    assert balanced_settings[
        "VMODEL_FAST_TOOL_GATEWAY_TERMINAL_PAGINATION_SYNTHESIS"] == "1"
    assert "VMODEL_FAST_TOOL_GATEWAY_LITERAL_GROUNDING" not in (
        balanced_settings)

    grounded_order, grounded_settings = resolve_runtime_profiles(
        ("qwen35-9b-tool-grounded-agent",), catalog)
    assert grounded_order == (
        "qwen35-9b-depth-adaptive",
        "agent-tool-gateway",
        "qwen35-9b-tool-balanced-agent",
        "qwen35-9b-tool-grounded-agent",
    )
    assert grounded_settings[
        "VMODEL_FAST_TOOL_GATEWAY_LITERAL_GROUNDING"] == "1"

    gptoss_order, gptoss_settings = resolve_runtime_profiles(
        ("gpt-oss-120b-tool-agent",), catalog)
    assert gptoss_order == (
        "agent-tool-gateway",
        "gpt-oss-120b-tool-agent",
    )
    assert gptoss_settings["VMODEL_GPTOSS_PREFILL_CHUNK_SIZE"] == "512"
    assert gptoss_settings["VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT"] == "full"
    assert gptoss_settings[
        "VMODEL_FAST_TOOL_GATEWAY_SUFFIX_CONTRACT"] == "1"
    assert gptoss_settings[
        "VMODEL_FAST_TOOL_GATEWAY_LITERAL_GROUNDING"] == "1"
    assert gptoss_settings[
        "VMODEL_FAST_TOOL_GATEWAY_TERMINAL_PAGINATION_SYNTHESIS"] == "1"

    task_order, task_settings = resolve_runtime_profiles(
        ("gpt-oss-120b-tool-agent-task",), catalog)
    assert task_order == (
        "agent-tool-gateway",
        "gpt-oss-120b-tool-agent",
        "gpt-oss-120b-tool-agent-task",
    )
    assert task_settings[
        "VMODEL_FAST_TOOL_GATEWAY_EXECUTION_CONTEXT"] == "task"

    stationary_order, stationary_settings = resolve_runtime_profiles(
        ("gpt-oss-120b-tool-agent-layer-stationary",), catalog)
    assert stationary_order == (
        "agent-tool-gateway",
        "gpt-oss-120b-tool-agent",
        "gpt-oss-120b-tool-agent-task",
        "gpt-oss-120b-tool-agent-layer-stationary",
    )
    assert stationary_settings[
        "VMODEL_GPTOSS_LAYER_STATIONARY_PREFILL"] == "1"
    assert stationary_settings["VMODEL_GPTOSS_PREFILL_CHUNK_SIZE"] == "128"
    assert stationary_settings["VMODEL_GPTOSS_PREFILL_EXPERT_BATCH"] == "8"
    assert stationary_settings["VMODEL_GPTOSS_HOT_PROMPT_KV"] == "1"
    assert stationary_settings["VMODEL_GRAMMAR_FAST_FORWARD"] == "1"

    k3_order, k3_settings = resolve_runtime_profiles((
        "kimi-k3-exact-streaming",
        "kimi-k3-adaptive-context",
        "kimi-k3-suffix-verification",
    ), catalog)
    assert k3_order == (
        "kimi-k3-exact-streaming",
        "kimi-k3-memory-core",
        "kimi-k3-adaptive-context",
        "kimi-k3-suffix-verification",
    )
    assert k3_settings["VMODEL_CT_MXFP4_NATIVE"] == "1"
    assert k3_settings["VMODEL_K3_ABSORBED_MLA"] == "1"
    assert k3_settings["VMODEL_K3_FUSED_ATTNRES_TILE_SIZE"] == "128"
    assert k3_settings["VMODEL_K3_PREFILL_TILE_POLICY"] == "prompt-length"
    assert k3_settings["VMODEL_K3_PREFILL_LONG_CONTEXT_TOKENS"] == "256"
    assert k3_settings["VMODEL_K3_PREFILL_SHORT_TILE_WIDTH"] == "256"
    assert k3_settings["VMODEL_K3_PREFILL_TILE_WIDTH"] == "256"
    assert k3_settings["VMODEL_K3_SUFFIX_K"] == "2"
    assert k3_settings["VMODEL_K3_SUFFIX_MIN_PROBABILITY"] == "0.75"

    short_order, short_settings = resolve_runtime_profiles(
        ("kimi-k3-short-first-token",), catalog)
    assert short_order == (
        "kimi-k3-memory-core",
        "kimi-k3-short-first-token",
    )
    assert short_settings["VMODEL_K3_PREFILL_TILE_POLICY"] == "fixed"
    assert short_settings["VMODEL_K3_PREFILL_TILE_WIDTH"] == "256"
    assert short_settings["VMODEL_K3_DENSE_MLP_TILE_SIZE"] == "0"


def test_profile_inheritance_and_later_selection_precedence(tmp_path):
    _write_profile(tmp_path, "base", settings={
        "VMODEL_SHARED": 1,
        "VMODEL_BASE_ONLY": True,
    })
    _write_profile(tmp_path, "child", extends=["base"], settings={
        "VMODEL_SHARED": 2,
        "VMODEL_CHILD_ONLY": False,
    })
    catalog = discover_runtime_profiles((tmp_path,))

    order, settings = resolve_runtime_profiles(("child", "base"), catalog)

    assert order == ("base", "child", "base")
    assert settings == {
        "VMODEL_SHARED": "1",
        "VMODEL_BASE_ONLY": "1",
        "VMODEL_CHILD_ONLY": "0",
    }


def test_explicit_environment_wins_and_changes_effective_digest(tmp_path):
    _write_profile(tmp_path, "speed", settings={
        "VMODEL_CACHE_MB": 2300,
        "VMODEL_FEATURE": True,
    })
    env = {"VMODEL_CACHE_MB": "4096", "UNRELATED": "preserved"}

    application = apply_runtime_profiles(
        ("speed",), search_dirs=(tmp_path,), environ=env)

    assert application is not None
    assert env == {
        "VMODEL_CACHE_MB": "4096",
        "VMODEL_FEATURE": "1",
        "UNRELATED": "preserved",
    }
    assert application.overridden_keys == ("VMODEL_CACHE_MB",)
    assert application.profile_digest != application.effective_digest


def test_equivalent_application_has_stable_digests(tmp_path):
    _write_profile(tmp_path, "stable", settings={
        "VMODEL_B": "two",
        "VMODEL_A": 1,
    })

    first = apply_runtime_profiles(
        ("stable",), search_dirs=(tmp_path,), environ={})
    second = apply_runtime_profiles(
        ("stable",), search_dirs=(tmp_path,), environ={})

    assert first is not None and second is not None
    assert first.profile_digest == second.profile_digest
    assert first.effective_digest == second.effective_digest


def test_active_telemetry_never_discloses_setting_values(tmp_path):
    _write_profile(tmp_path, "telemetry", settings={
        "VMODEL_PRIVATE_PATH": "/a/machine-specific/path",
    })
    try:
        application = apply_runtime_profiles(
            ("telemetry",), search_dirs=(tmp_path,), environ={}, activate=True)
        fields = active_runtime_profile_fields()
        assert application is not None
        assert fields["vmodel_runtime_profiles"] == ["telemetry"]
        assert fields["vmodel_runtime_profile_groups"] == ["telemetry"]
        assert fields["vmodel_runtime_profile_digest"] == application.profile_digest
        assert fields["vmodel_runtime_effective_digest"] == application.effective_digest
        assert "/a/machine-specific/path" not in json.dumps(fields)
        assert "VMODEL_PRIVATE_PATH" not in json.dumps(fields)
    finally:
        clear_active_runtime_profiles()
    assert active_runtime_profile_fields() == {}


def test_server_execution_fields_include_active_profile_identity(tmp_path):
    from runtime.server import _execution_profile_fields

    _write_profile(tmp_path, "response", settings={"VMODEL_FEATURE": True})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    engine = SimpleNamespace(
        _model_dir=model_dir,
        store=SimpleNamespace(quantization={}, on_disk_quantized=False),
        rc=SimpleNamespace(
            quant_bits=0,
            rerank_lm_head=False,
            resident_attention_mode="",
            expert_top_k_by_layer=(),
            native_ct_mxfp4=False,
            kimi_k3_scale_sidecar_dir="",
            bf16_nf12_sidecar_dir="",
        ),
    )
    try:
        apply_runtime_profiles(
            ("response",), search_dirs=(tmp_path,), environ={}, activate=True)
        fields = _execution_profile_fields(engine)
        assert fields["vmodel_runtime_profiles"] == ["response"]
        assert fields["vmodel_runtime_profile_groups"] == ["response"]
        assert len(fields["vmodel_runtime_profile_digest"]) == 64
        assert len(fields["vmodel_runtime_effective_digest"]) == 64
    finally:
        clear_active_runtime_profiles()


def test_profile_selection_parses_repeated_and_comma_separated_values():
    assert parse_runtime_profile_names(("one,two", "three")) == (
        "one", "two", "three")
    with pytest.raises(RuntimeProfileError, match="invalid runtime profile name"):
        parse_runtime_profile_names("../escape")


def test_environment_and_explicit_search_dirs_follow_defaults(tmp_path):
    env_one = tmp_path / "env-one"
    env_two = tmp_path / "env-two"
    explicit = tmp_path / "explicit"
    env = {"VMODEL_PROFILE_DIR": os.pathsep.join((str(env_one), str(env_two)))}

    directories = runtime_profile_dirs((explicit,), environ=env)

    assert directories[-3:] == (
        env_one.resolve(), env_two.resolve(), explicit.resolve())


def test_profile_rejects_non_vmodel_and_recursive_settings(tmp_path):
    bad = _write_profile(tmp_path, "bad", settings={"PATH": "/tmp"})
    with pytest.raises(RuntimeProfileError, match="setting names"):
        load_runtime_profile(bad)

    recursive = _write_profile(
        tmp_path, "recursive", settings={"VMODEL_PROFILE": "bad"})
    with pytest.raises(RuntimeProfileError, match="cannot be set by a profile"):
        load_runtime_profile(recursive)

    nonfinite = _write_profile(
        tmp_path, "nonfinite", settings={"VMODEL_VALUE": float("inf")})
    with pytest.raises(RuntimeProfileError, match="NaN or infinite"):
        load_runtime_profile(nonfinite)


def test_profile_rejects_duplicate_names_and_inheritance_cycles(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_profile(first, "duplicate", settings={"VMODEL_X": 1})
    _write_profile(second, "duplicate", settings={"VMODEL_X": 2})
    with pytest.raises(RuntimeProfileError, match="duplicate runtime profile"):
        discover_runtime_profiles((first, second))

    cycle_dir = tmp_path / "cycles"
    _write_profile(cycle_dir, "alpha", extends=["beta"])
    _write_profile(cycle_dir, "beta", extends=["alpha"])
    catalog = discover_runtime_profiles((cycle_dir,))
    with pytest.raises(RuntimeProfileError, match="alpha -> beta -> alpha"):
        resolve_runtime_profiles(("alpha",), catalog)


def test_unknown_parent_fails_with_available_catalog(tmp_path):
    _write_profile(tmp_path, "child", extends=["missing"])
    catalog = discover_runtime_profiles((tmp_path,))
    with pytest.raises(RuntimeProfileError, match="unknown runtime profile 'missing'"):
        resolve_runtime_profiles(("child",), catalog)
