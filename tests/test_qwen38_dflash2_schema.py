"""Pure-Python gates for the fail-closed DFlash2 metadata contract."""

from __future__ import annotations

import copy

import pytest

from runtime.dflash2_schema import (
    DFlash2Config,
    GLM53_FLASH_CONFIG,
    GLM53_FLASH_PARAMETER_COUNT,
    GLM53_FLASH_RELEASE,
    OFFICIAL_CONFIG,
    OFFICIAL_PARAMETER_COUNT,
    validate_source_header,
)
from runtime.dflash2_sidecar import plan_sidecar


def _header(config: DFlash2Config):
    result = {}
    offset = 0
    for name, spec in config.expected_tensor_specs().items():
        result[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    return result, offset


def test_official_plan_is_pinned_default_off_and_header_complete():
    report = plan_sidecar()

    assert report["architecture"] == {
        "architecture": "DFlash2DraftModel",
        "tensor_count": 81,
        "parameter_count": OFFICIAL_PARAMETER_COUNT,
        "tensor_schema_sha256": (
            "ff72a992215a097089d90bbd019cb96a78c5f73af21396757b75db196e8ce9be"),
        "target_layer_ids": [5, 19, 33, 47, 61],
        "checkpoint_block_size": 8,
        "checkpoint_proposal_count": 7,
        "selector_top_k": 16,
        "selector_rank": 256,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "is_causal": False,
    }
    assert report["conversion"]["quantized_tensors"] == 49
    assert report["conversion"]["estimated_output_tensor_bytes"] == 1_082_862_080
    assert report["serving"]["runtime_supported"] is False
    assert report["serving"]["enabled_by_default"] is False
    assert report["serving"]["planned_proposal_count"] == 4
    assert report["source"]["local_header_validated"] is False


def test_official_tensor_schema_matches_published_81_tensor_header():
    config = DFlash2Config.from_mapping(OFFICIAL_CONFIG)
    header, payload_bytes = _header(config)

    report = validate_source_header(
        config, header, payload_bytes=payload_bytes)

    assert report["tensor_count"] == 81
    assert report["parameter_count"] == 1_924_404_480
    assert report["tensor_bytes"] == 3_848_808_960


def test_glm53_flash_plan_is_independently_pinned_and_header_complete():
    release = GLM53_FLASH_RELEASE
    report = plan_sidecar(
        repository=release.repository,
        revision=release.revision,
        expected_config_sha256=release.config_sha256,
        expected_weights_sha256=release.weights_sha256,
        expected_weights_bytes=release.weights_bytes,
    )

    assert report["architecture"] == {
        "architecture": "DFlash2DraftModel",
        "tensor_count": 81,
        "parameter_count": GLM53_FLASH_PARAMETER_COUNT,
        "tensor_schema_sha256": (
            "d46fe79fe98bed4c2f0df126ee6ec3024711a658200378151221eb073285b51c"),
        "target_layer_ids": [5, 14, 24, 33, 42],
        "checkpoint_block_size": 8,
        "checkpoint_proposal_count": 7,
        "selector_top_k": 16,
        "selector_rank": 256,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "is_causal": False,
    }
    assert report["conversion"]["quantized_tensors"] == 49
    assert report["conversion"]["retained_bf16_bytes"] == 428_544
    assert report["conversion"]["estimated_output_tensor_bytes"] == 659_040_768
    assert report["source"]["repository"] == release.repository
    assert report["source"]["revision"] == release.revision


def test_glm53_flash_tensor_schema_matches_published_81_tensor_header():
    config = DFlash2Config.from_mapping(GLM53_FLASH_CONFIG)
    config.validate_official_glm53_flash()
    config.validate_official_release(GLM53_FLASH_RELEASE)
    header, payload_bytes = _header(config)

    report = validate_source_header(
        config, header, payload_bytes=payload_bytes)

    assert report["tensor_count"] == 81
    assert report["parameter_count"] == GLM53_FLASH_PARAMETER_COUNT
    assert report["tensor_bytes"] == 2_342_160_896


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(architectures=["DFlashDraftModel"]),
         "DFlash1"),
        (lambda raw: raw.update(is_causal=True), "is_causal"),
        (lambda raw: raw["dflash_config"].update(block_size=1), "block_size"),
        (lambda raw: raw["dflash_config"].update(
            target_layer_ids=[5, 19, 19, 47, 61]), "unique"),
    ],
)
def test_config_rejects_silent_architecture_changes(mutation, message):
    raw = copy.deepcopy(OFFICIAL_CONFIG)
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        DFlash2Config.from_mapping(raw)


def test_header_rejects_missing_conv_selector_shape_and_payload_gap():
    config = DFlash2Config.from_mapping(OFFICIAL_CONFIG)
    header, payload_bytes = _header(config)

    missing = copy.deepcopy(header)
    missing.pop("layers.4.attention_conv.base_kernel")
    with pytest.raises(ValueError, match="tensor set mismatch"):
        validate_source_header(config, missing)

    wrong_selector = copy.deepcopy(header)
    wrong_selector["candidate_selector.predecessor_codebook"]["shape"] = [
        248320, 255]
    with pytest.raises(ValueError, match="predecessor_codebook mismatch"):
        validate_source_header(config, wrong_selector)

    gap = copy.deepcopy(header)
    first = min(gap, key=lambda name: gap[name]["data_offsets"][0])
    gap[first]["data_offsets"] = [1, gap[first]["data_offsets"][1] + 1]
    with pytest.raises(ValueError, match="not contiguous"):
        validate_source_header(config, gap, payload_bytes=payload_bytes + 1)
