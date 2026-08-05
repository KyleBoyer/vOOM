"""F198: the cross-layer router predictor must find the gate it needs.

``StreamingEngine._router_lookahead`` (F45) predicts layer L+1's routed experts
by running that layer's own released router on layer L's hidden state.  It
looked the gate up at a hardcoded ``mlp.gate.weight`` / ``mlp.router.weight``.
Kimi Linear and Kimi K3 ship theirs under ``block_sparse_moe.gate.*``, so both
lookups returned ``None``, the function took its dense-layer early return, and
the predictor was silently inert on exactly the architectures whose expert
paging dominates wall time.

These tests pin the name derivation against the real checkpoint indices, so a
future architecture with a third prefix fails here rather than by quietly
prefetching nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.config import ModelConfig  # noqa: E402


def gate_prefix(moe_expert_prefix: str) -> str:
    """Exercise the real derivation, not a copy of it."""
    return ModelConfig.moe_module_prefix(
        type("_Stub", (), {"moe_expert_prefix": moe_expert_prefix})())


@pytest.mark.parametrize("expert_prefix,expected", [
    ("mlp.experts", "mlp"),                       # GLM, Qwen MoE
    ("block_sparse_moe.experts", "block_sparse_moe"),  # Kimi Linear, K3
    ("experts", ""),                              # degenerate, no dotted parent
])
def test_gate_prefix_derivation(expert_prefix, expected):
    assert gate_prefix(expert_prefix) == expected


def _index_names(model_dir: Path) -> set[str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return set(index["weight_map"])


@pytest.mark.parametrize("model_name", ["Kimi-K3", "Kimi-Linear-48B-A3B-Instruct"])
def test_derived_gate_name_exists_in_real_checkpoint(model_name):
    """The derived name must match a tensor the real checkpoint actually ships."""
    model_dir = ROOT / "models" / model_name
    if not (model_dir / "model.safetensors.index.json").is_file():
        pytest.skip(f"{model_name} not present")

    config = ModelConfig.from_dir(str(model_dir))
    prefix = gate_prefix(config.moe_expert_prefix)
    assert prefix, f"{model_name}: empty gate prefix from {config.moe_expert_prefix}"
    names = _index_names(model_dir)

    # Find any MoE layer and require the derived gate name to resolve. Both
    # canonical and checkpoint-prefixed forms are accepted, matching how the
    # runtime resolves names.
    candidates = [
        name for name in names
        if name.endswith(f".{prefix}.gate.weight")
    ]
    assert candidates, (
        f"{model_name}: no tensor ends with .{prefix}.gate.weight; "
        f"the router lookahead would find no gate and prefetch nothing")

    # The old hardcoded lookup must genuinely have missed, otherwise this test
    # is not protecting anything.
    if prefix != "mlp":
        stale = [name for name in names if name.endswith(".mlp.gate.weight")]
        assert not stale, (
            f"{model_name} also ships .mlp.gate.weight; the original lookup "
            f"was not actually broken here and this test proves nothing")


def test_k3_gate_and_expert_names_share_the_derived_parent():
    """Gate and experts must hang off the same parent module, per layer."""
    model_dir = ROOT / "models" / "Kimi-K3"
    if not (model_dir / "model.safetensors.index.json").is_file():
        pytest.skip("Kimi-K3 not present")

    config = ModelConfig.from_dir(str(model_dir))
    prefix = gate_prefix(config.moe_expert_prefix)
    names = _index_names(model_dir)

    layer = 1
    gate = [n for n in names
            if f".layers.{layer}.{prefix}.gate.weight" in n]
    experts = [n for n in names
               if f".layers.{layer}.{config.moe_expert_prefix}.0." in n]
    assert gate, f"layer {layer}: no gate under {prefix}"
    assert experts, f"layer {layer}: no experts under {config.moe_expert_prefix}"
    assert config.num_experts_per_tok == 16
