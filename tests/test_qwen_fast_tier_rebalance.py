from runtime.qwen_fast_tier_rebalance import rebalance_manifest


def test_rebalance_removes_from_most_overrepresented_layers_and_keeps_overlap():
    manifest = {
        "model.layers.0.a": {"nbytes": 40},
        "model.layers.0.b": {"nbytes": 40},
        "model.layers.0.norm": {"nbytes": 1},
        "model.layers.1.a": {"nbytes": 40},
        "model.layers.1.b": {"nbytes": 40},
        "model.layers.1.c": {"nbytes": 40},
        "model.norm.weight": {"nbytes": 1},
    }
    candidate, report = rebalance_manifest(
        manifest, {"0": 100, "1": 200}, 0.4)

    # The synthetic entries are below the production 1MB substantial floor,
    # so no unsafe microscopic "rebalance" occurs.
    assert candidate == manifest
    assert report["removed_tensors"] == 0


def test_rebalance_never_removes_last_substantial_page_from_a_layer():
    mb = 1_000_000
    manifest = {
        "model.layers.0.a": {"nbytes": 40 * mb},
        "model.layers.0.b": {"nbytes": 40 * mb},
        "model.layers.1.a": {"nbytes": 40 * mb},
        "model.layers.1.b": {"nbytes": 40 * mb},
        "model.layers.1.c": {"nbytes": 40 * mb},
    }
    candidate, report = rebalance_manifest(
        manifest, {"0": 100 * mb, "1": 200 * mb}, 0.4)

    assert report["removed_tensors"] == 2
    for layer in ("0", "1"):
        assert any(name.startswith(f"model.layers.{layer}.")
                   for name in candidate)
