"""Real-file checks for tests/fixtures/inspect_checkpoint_format.py.

Written ahead of Kimi K3's open-weight release: this script is the tool
that will actually get run against the real checkpoint the moment it
exists, so it needs its own proof it reads real safetensors headers
correctly -- against checkpoints already on disk, not a synthetic mock.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import pytest

from inspect_checkpoint_format import _categorize, inspect

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "models" / "glm-fixture-tiny"


def test_categorize_matches_known_tensor_name_shapes():
    assert _categorize("model.layers.0.mlp.experts.3.gate_proj.weight") == "expert"
    assert _categorize("model.layers.0.mlp.shared_expert.gate_proj.weight") == "shared_expert"
    assert _categorize("model.layers.0.self_attn.q_proj.weight") == "attention"
    assert _categorize("model.layers.0.linear_attn.conv1d.weight") == "linear_attention"
    assert _categorize("model.layers.0.mlp.gate.weight") == "router_gate"
    assert _categorize("model.layers.0.input_layernorm.weight") == "norm"
    assert _categorize("model.embed_tokens.weight") == "embedding"
    assert _categorize("lm_head.weight") == "lm_head"
    assert _categorize("model.layers.0.something_unrecognized.weight") == "other"


@pytest.mark.skipif(
    not FIXTURE_DIR.exists(),
    reason="glm-fixture-tiny not present locally")
def test_inspect_reads_real_single_file_checkpoint_header():
    report = inspect(FIXTURE_DIR)
    assert report["config_model_type"] == "glm_moe_dsa"
    assert report["index_shard_count"] == 1
    assert report["total_tensors_inspected"] > 0
    assert "config_error" not in report
    assert "index_error" not in report
    # Every category this fixture actually has tensors for must report a
    # real dtype string, not an empty/missing entry -- catches a header
    # -parsing regression that silently drops tensors rather than erroring.
    dtypes = report["dtype_by_category"]
    assert dtypes["attention"]
    assert dtypes["expert"]
    assert all(isinstance(count, int) and count > 0
               for counts in dtypes.values() for count in counts.values())


def test_inspect_reports_error_for_missing_directory(tmp_path):
    report = inspect(tmp_path)
    assert "config_error" in report
    assert "index_error" in report
    assert report["total_tensors_inspected"] == 0
