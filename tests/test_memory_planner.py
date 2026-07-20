"""Architecture-aware memory-planner accounting gates."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path(__file__).resolve().parent.parent / "models" / "glm-fixture-tiny"


def _ensure_fixture():
    from tests.fixtures.build_glm_fixture import build, is_current

    if not is_current(FIXTURE):
        build(FIXTURE)


def test_glm_planner_matches_actual_fixture_trunk_tensor_bytes():
    _ensure_fixture()
    from runtime.memory_planner import MemoryPlanner
    from runtime.model_loader import WeightStore

    store = WeightStore(FIXTURE)
    planner = MemoryPlanner(store, budget_mb=200, disk_mb_per_s=300)
    actual = 0
    for layer in range(store.config.num_hidden_layers):
        tensors, _seconds, _read_bytes = store.fetch(store.layer_param_names(layer))
        actual += sum(tensor.nbytes for tensor in tensors.values())

    assert planner.layer_bytes * store.config.num_hidden_layers == actual
    assert planner.active_layer_bytes < planner.layer_bytes


def test_released_scale_glm_no_longer_looks_like_one_dense_mlp_per_layer():
    _ensure_fixture()
    from runtime.config import ModelConfig
    from runtime.memory_planner import MemoryPlanner

    base = ModelConfig.from_dir(FIXTURE)
    released_scale = replace(
        base,
        hidden_size=6144,
        intermediate_size=12288,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=154_880,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        n_shared_experts=1,
        mlp_layer_types=(),
        tie_word_embeddings=False,
    )
    store = SimpleNamespace(config=released_scale, on_disk_quantized=False)
    planner = MemoryPlanner(store, budget_mb=8_000, disk_mb_per_s=300)
    plan = planner.plan("latency")

    # A routed layer owns all 256 expert pages (~19 GB BF16), even though one
    # decode position reads only top-k=8. The old dense-only formula reported
    # ~0.45 GB/layer and could recommend impossible residency.
    assert planner.layer_bytes > 15_000_000_000
    assert planner.active_layer_bytes < 2_000_000_000
    assert plan.resident_layers <= 1


def test_fully_prequantized_plan_uses_packed_layer_and_traffic_bytes():
    _ensure_fixture()
    from runtime.config import ModelConfig
    from runtime.memory_planner import MemoryPlanner

    config = ModelConfig.from_dir(FIXTURE)
    packed_ratio = (4 / 8 + 1 / 32) / 2
    store = SimpleNamespace(
        config=config,
        on_disk_quantized=True,
        quantization_ratio=lambda name: (
            packed_ratio if name == "lm_head.weight" else 1.0),
        uniform_quantization_ratio=lambda fragment: (
            packed_ratio if fragment in (".self_attn.", ".mlp.") else 1.0),
    )
    planner = MemoryPlanner(store, budget_mb=200, disk_mb_per_s=300)
    plan = planner.plan("lru")
    raw = MemoryPlanner(
        SimpleNamespace(config=config, on_disk_quantized=False),
        budget_mb=200, disk_mb_per_s=300,
    )

    assert plan.rc.quant_bits == 0
    assert plan.resident_layer_bytes == planner.layer_bytes
    assert plan.active_layer_bytes == planner.active_layer_bytes
    assert planner.layer_bytes < 0.4 * raw.layer_bytes
