"""Pure artifact gates for opt-in target-verified DFlash2 ablation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from runtime.dflash2_ablation import (
    coherent_mean_direction,
    contrastive_direction,
    load_artifact,
    write_artifact,
)


def test_contrastive_direction_is_unit_and_can_remove_harmless_axis():
    harmful = np.array([[3, 1, 0], [5, 1, 0]], dtype=np.float32)
    harmless = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
    direction = contrastive_direction(harmful, harmless)
    np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-7)
    projected = contrastive_direction(
        harmful,
        np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
        project_harmless=True,
    )
    np.testing.assert_allclose(projected, [0, 1, 0], atol=1e-7)


def test_coherent_mean_rejects_unrelated_rank1_axes():
    direction, minimum, count = coherent_mean_direction(
        {
            "layer.1": np.array([1, 0, 0], dtype=np.float32),
            "layer.2": np.array([-1, 0, 0], dtype=np.float32),
        },
        hidden_size=3,
    )
    np.testing.assert_allclose(direction, [1, 0, 0], atol=1e-7)
    assert minimum == pytest.approx(1.0)
    assert count == 2
    with pytest.raises(ValueError, match="not one global axis"):
        coherent_mean_direction(
            {
                "layer.1": np.array([1, 0, 0], dtype=np.float32),
                "layer.2": np.array([0, 1, 0], dtype=np.float32),
            },
            hidden_size=3,
        )


def test_artifact_is_target_bound_hashed_and_default_off(tmp_path):
    target = tmp_path / "config.json"
    target.write_text('{"model_type":"qwen3_5","hidden_size":3}\n')
    output = tmp_path / "direction"
    built = write_artifact(
        output,
        np.array([3, 4, 0], dtype=np.float32),
        target_config=target,
        draft_revision="draft-revision",
        source={"kind": "fixture"},
        method="fixture-contrast",
    )
    assert built.manifest["enabled_by_default"] is False
    loaded = load_artifact(
        output,
        target_config=target,
        draft_revision="draft-revision",
        hidden_size=3,
    )
    np.testing.assert_allclose(loaded.direction, [0.6, 0.8, 0], atol=1e-7)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["direction_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_artifact(
            output,
            target_config=target,
            draft_revision="draft-revision",
            hidden_size=3,
        )
