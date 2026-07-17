"""Focused exactness gates for sorted weight-name prefix lookup."""

from runtime.model_loader import WeightStore


def _store_with_names(*names):
    store = object.__new__(WeightStore)
    store._names = sorted(names)
    return store


def test_prefix_lookup_returns_only_the_contiguous_sorted_range():
    store = _store_with_names(
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.20.gate_proj.weight",
        "model.layers.1.mlp.experts.3.gate_proj.weight",
        "model.layers.10.input_layernorm.weight",
        "model.norm.weight",
    )

    assert store.names_with_prefix("model.layers.1.mlp.experts.2.") == [
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight",
    ]
    assert store.layer_param_names(1) == [
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.20.gate_proj.weight",
        "model.layers.1.mlp.experts.3.gate_proj.weight",
    ]


def test_prefix_lookup_handles_edges_and_missing_prefixes():
    store = _store_with_names("b.one", "b.two", "c.one")
    assert store.names_with_prefix("") == ["b.one", "b.two", "c.one"]
    assert store.names_with_prefix("a") == []
    assert store.names_with_prefix("b.") == ["b.one", "b.two"]
    assert store.names_with_prefix("z") == []
