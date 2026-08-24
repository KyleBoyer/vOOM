"""Header/load witness for the real Huihui mixed-precision candidate."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from runtime.model_loader import WeightStore
from runtime.quant import QTensor


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / (
    "Huihui-Qwen3.8-27B-abliterated-mlx-mixed-a8-last4-mtpbf16")
PLAN = ROOT / "logs" / "huihui_qwen38_mixed_a8_last4_mtpbf16.plan.json"
_skip = pytest.mark.skipif(
    not MODEL.is_dir() or not PLAN.is_file(),
    reason="real Huihui mixed-precision artifact is not installed",
)


@_skip
def test_real_huihui_mixed_artifact_matches_selected_plan():
    plan = json.loads(PLAN.read_text())
    config = json.loads((MODEL / "config.json").read_text())
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())

    assert plan["spec"] == {
        "attention": "mxfp8",
        "body": "mxfp4",
        "last_bf16_layers": 4,
        "mtp": "bf16",
    }
    assert config["voom_quantization"]["precision_plan_digest"] == \
        plan["plan_digest"]
    assert index["metadata"]["total_size"] == \
        plan["summary"]["estimated_bytes"] == 23_081_758_624
    assert len(set(index["weight_map"].values())) == 18
    assert all((MODEL / shard).is_file()
               for shard in set(index["weight_map"].values()))

    store = WeightStore(MODEL)
    names = [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.60.linear_attn.in_proj_qkv.weight",
        "model.layers.60.mlp.up_proj.weight",
        "mtp.fc.weight",
        "lm_head.weight",
    ]
    values, _seconds, _bytes = store.fetch(names)
    assert isinstance(values[names[0]], QTensor)
    assert values[names[0]].mode == "mxfp8"
    assert isinstance(values[names[1]], QTensor)
    assert values[names[1]].mode == "mxfp4"
    assert not isinstance(values[names[2]], QTensor)
    assert not isinstance(values[names[3]], QTensor)
    assert values[names[2]].dtype == mx.bfloat16
    assert values[names[3]].dtype == mx.bfloat16
    assert not isinstance(values[names[4]], QTensor)
    assert values[names[4]].dtype == mx.bfloat16
    assert isinstance(values[names[5]], QTensor)
    assert values[names[5]].mode == "mxfp4"
