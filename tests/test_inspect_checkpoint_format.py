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
    assert _categorize(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    ) == "ple_ngram"
    assert _categorize("mtp.layers.0.self_attn.q_proj.weight") == "mtp"
    assert _categorize(
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
    ) == "qsa_indexer"
    assert _categorize(
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
    ) == "gated_residual"
    assert _categorize(
        "model.visual.patch_embed.proj.weight") == "vision"
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
    assert report["total_tensor_bytes"] > 0
    assert sum(report["tensor_bytes_by_category"].values()) == report[
        "total_tensor_bytes"]
    assert report["unknown_dtype_tensors"] == []
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


def test_inspect_lifts_nested_qwen4_exp_text_config(tmp_path):
    (tmp_path / "config.json").write_text("""{
      "model_type": "qwen4_exp",
      "architectures": ["Qwen4ExpForConditionalGeneration"],
      "text_config": {
        "model_type": "qwen4_exp_text",
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "layer_types": ["linear_attention", "full_attention"],
        "ple_layer_ids": [2],
        "mtp_num_hidden_layers": 1
      }
    }""")

    report = inspect(tmp_path)

    assert report["config_model_type"] == "qwen4_exp"
    assert report["config_text_model_type"] == "qwen4_exp_text"
    assert report["config_num_hidden_layers"] == 48
    assert report["config_num_experts"] == 512
    assert report["config_num_experts_per_tok"] == 10
    assert report["config_layer_types_present"] is True
    assert report["config_ple_layer_ids"] == [2]
    assert report["config_mtp_num_hidden_layers"] == 1
    assert "index_error" in report
