"""F128: raw-safetensors fast-tier mirror for Kimi K3 (formats/kimi_k3_fast_tier.py).

Distinct from the vpack2 overlay mechanism (predicted-hot-expert staging,
requires the packed vpack/vpack2 format): this mirrors a DETERMINISTIC
subset of always-touched, non-expert tensors (self_attn/KDA, Stable-
LatentMoE's routed_expert_down_proj/up_proj/norm, the MoE gate, layer-0's
dense MLP, norms, AttnRes projections) as raw per-tensor byte files on a
second local disk, read concurrently with the main external volume during
WeightStore.fetch(). This test checks the read path is byte-exact against
the ordinary (non-fast-tier) fetch of the SAME real tensors, both alone and
mixed with a slow-tier (expert) tensor in the same fetch() call to exercise
the concurrent ThreadPoolExecutor path.
"""

from __future__ import annotations

from pathlib import Path
import json
import threading

import mlx.core as mx
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Kimi-K3"
FAST_ROOT = Path.home() / "vmodel_fast_tier"
_MANIFEST = FAST_ROOT / "Kimi-K3" / "fast_tier_manifest.json"
_MODEL_AVAILABLE = (MODEL_DIR / "model.safetensors.index.json").exists()
_FAST_TIER_STAGED = _MANIFEST.exists()
_model_skip = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Kimi-K3 is not available locally (a real ~1.4TB model, not fetched in CI)",
)
_fast_tier_skip = pytest.mark.skipif(
    not _FAST_TIER_STAGED,
    reason="Kimi-K3 fast tier is not staged locally "
           "(run formats/kimi_k3_fast_tier.py first)",
)


def test_k3_fast_tier_budget_fails_before_writing(tmp_path):
    from formats.kimi_k3_fast_tier import build_fast_tier

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    mx.save_safetensors(
        str(shard), {name: mx.ones((8, 8), dtype=mx.bfloat16)})
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard.name},
    }))
    target_root = tmp_path / "fast"

    with pytest.raises(ValueError, match="no files were written"):
        build_fast_tier(
            model_dir, target_root, max_bytes=1)
    assert not (target_root / model_dir.name).exists()


def test_budgeted_selection_balances_layers():
    from formats.kimi_k3_fast_tier import _select_budgeted

    candidates = [
        {
            "name": f"model.layers.{layer}.tensor.{index}",
            "shard": f"{layer}.safetensors",
            "offset": index * 60,
            "nbytes": size,
        }
        for layer in (0, 1)
        for index, size in enumerate((60, 40))
    ]
    selected = _select_budgeted(candidates, 120)
    selected_by_layer = {
        layer: sum(
            item["nbytes"]
            for item in selected
            if f"layers.{layer}." in item["name"]
        )
        for layer in (0, 1)
    }
    assert selected_by_layer == {0: 60, 1: 60}


def test_fast_tier_excludes_qwen4_direct_ple_rows():
    from formats.kimi_k3_fast_tier import _category

    assert _category(
        "model.layers.1.ple.embedding.weight") is None
    assert _category(
        "model.layers.1.self_attn.q_proj.weight") == "keep"


def test_fast_tier_excludes_glm53_routed_experts_but_keeps_shared_trunk():
    from formats.kimi_k3_fast_tier import _category

    assert _category(
        "model.layers.4.mlp.experts.17.gate_proj.weight") is None
    assert _category(
        "model.layers.4.mlp.shared_experts.gate_proj.weight") == "keep"
    assert _category(
        "model.layers.4.self_attn.q_proj.weight") == "keep"


def test_raw_fast_tier_preserves_glm53_e4m3_codes_as_uint8():
    from formats.packed import to_mx

    raw = bytes((0x00, 0x38, 0xB8, 0x7E))
    got = to_mx({"dtype": "F8_E4M3", "shape": [2, 2]}, raw)
    mx.eval(got)

    assert got.dtype == mx.uint8
    assert bytes(memoryview(__import__("numpy").array(got))) == raw


def test_budgeted_selection_balances_qwen_multimodal_wrapper_layers():
    """Qwen uses model.language_model.layers, unlike K3's older fixture."""
    from formats.kimi_k3_fast_tier import _select_budgeted

    candidates = [
        {
            "name": (
                f"model.language_model.layers.{layer}.mlp.proj{index}.weight"
            ),
            "shard": f"{layer}.safetensors",
            "offset": index * 60,
            "nbytes": size,
        }
        for layer in (0, 1)
        for index, size in enumerate((60, 40))
    ]
    selected = _select_budgeted(candidates, 120)
    selected_by_layer = {
        layer: sum(
            item["nbytes"]
            for item in selected
            if f"layers.{layer}." in item["name"]
        )
        for layer in (0, 1)
    }
    assert selected_by_layer == {0: 60, 1: 60}


def test_fast_tier_packs_selected_tensors_per_shard(tmp_path):
    from formats.kimi_k3_fast_tier import build_fast_tier, validate_fast_tier

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "test",
    }))
    shard = model_dir / "model-00001-of-00001.safetensors"
    names = [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
    ]
    mx.save_safetensors(
        str(shard),
        {
            names[0]: mx.arange(64, dtype=mx.uint8),
            names[1]: mx.arange(64, dtype=mx.uint8) + 1,
        },
    )
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: shard.name for name in names},
    }))
    fast_root = tmp_path / "fast"

    report = build_fast_tier(
        model_dir, fast_root, max_bytes=3_000_000,
        min_free_bytes=0, container_format="safetensors",
    )
    target = fast_root / model_dir.name
    manifest = json.loads(
        (target / "fast_tier_manifest.json").read_text()
    )
    entries = list(manifest.values())

    assert report["selected_tensors"] == 2
    assert len({entry["file"] for entry in entries}) == 1
    assert sorted(entry["offset"] for entry in entries) == [0, 64]
    containers = list(target.glob("*.safetensors"))
    assert len(containers) == 1
    packed = mx.load(str(containers[0]))
    for raw_name in names:
        canonical = raw_name.removeprefix("language_model.")
        mx.eval(packed[canonical])
        assert packed[canonical].shape == (64,)

    # WeightStore is used both directly with the parent root and by server
    # autodiscovery with the already model-specific directory.  Both must
    # resolve the same manifest rather than appending the model name twice.
    from runtime.model_loader import WeightStore

    for configured_dir in (fast_root, target):
        store = WeightStore.__new__(WeightStore)
        store.fast_dirs = [configured_dir]
        store.dir = model_dir
        store._raw_fast_tier_manifest = None
        store._raw_fast_tier_root = None
        store._ensure_raw_fast_tier_loaded()
        assert store._raw_fast_tier_root == target
        assert set(store._raw_fast_tier_manifest) == {
            raw_name.removeprefix("language_model.") for raw_name in names
        }

    validation = validate_fast_tier(model_dir, target)
    assert validation["verdict"] == "PASS"
    assert validation["checked_tensors"] == 2
    assert validation["checked_bytes"] == 128


def test_raw_fast_tier_reuses_one_store_lifetime_worker():
    """Long decode must not construct a new MLX-calling thread per layer."""
    from runtime.model_loader import WeightStore

    store = WeightStore.__new__(WeightStore)
    store._stage_lock = threading.Lock()
    store._raw_fast_tier_executor = None
    first = store._raw_fast_tier_executor_for_reads()
    second = store._raw_fast_tier_executor_for_reads()
    assert first is second
    assert first.submit(lambda: 17).result() == 17
    store.close()
    assert store._raw_fast_tier_executor is None
    with pytest.raises(RuntimeError, match="shutdown"):
        first.submit(lambda: 18)


@_model_skip
@_fast_tier_skip
def test_fast_tier_read_matches_slow_tier_for_a_real_tensor():
    from runtime.model_loader import WeightStore

    name = "model.layers.1.self_attn.q_proj.weight"

    store_slow = WeightStore(MODEL_DIR)
    assert store_slow.has(name)
    out_slow, _elapsed, _nbytes = store_slow.fetch([name])
    slow_arr = out_slow[name]
    mx.eval(slow_arr)

    store_fast = WeightStore(MODEL_DIR, fast_dirs=[FAST_ROOT])
    out_fast, _elapsed, _nbytes = store_fast.fetch([name])
    fast_arr = out_fast[name]
    mx.eval(fast_arr)

    assert store_fast.fast_tier_tensors == 1
    assert store_fast.fast_tier_bytes == fast_arr.nbytes

    assert slow_arr.shape == fast_arr.shape
    assert slow_arr.dtype == fast_arr.dtype
    max_diff = mx.max(mx.abs(
        slow_arr.astype(mx.float32) - fast_arr.astype(mx.float32)))
    mx.eval(max_diff)
    assert max_diff.item() == 0.0, (
        "fast-tier read must be byte-exact with the slow-tier read of the "
        f"same real tensor; max abs diff {max_diff.item()}")


@_model_skip
@_fast_tier_skip
def test_mixed_fast_and_slow_tier_fetch_matches_slow_only():
    """Exercises the concurrent ThreadPoolExecutor path: one name covered
    by the fast tier (self_attn), one NOT covered (a real MXFP4 expert
    weight, still only servable from the main external volume)."""
    from runtime.model_loader import WeightStore

    fast_name = "model.layers.1.self_attn.q_proj.weight"
    slow_name = "model.layers.1.block_sparse_moe.experts.0.w1.weight"

    store_slow = WeightStore(MODEL_DIR)
    out_slow, _elapsed, _nbytes = store_slow.fetch([fast_name, slow_name])
    mx.eval(out_slow[fast_name], out_slow[slow_name])

    store_mixed = WeightStore(
        MODEL_DIR,
        fast_dirs=[FAST_ROOT],
        parallel_storage_reads=True,
    )
    out_mixed, _elapsed, _nbytes = store_mixed.fetch([fast_name, slow_name])
    mx.eval(out_mixed[fast_name], out_mixed[slow_name])

    assert store_mixed.fast_tier_tensors == 1
    assert store_mixed.parallel_tier_fetches == 1
    assert store_mixed.parallel_tier_fast_bytes > 0
    assert store_mixed.parallel_tier_archive_bytes > 0

    for key in (fast_name, slow_name):
        a, b = out_slow[key], out_mixed[key]
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        max_diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
        mx.eval(max_diff)
        assert max_diff.item() == 0.0, (
            f"{key}: mixed-tier fetch must match slow-only fetch; "
            f"max abs diff {max_diff.item()}")


@_model_skip
def test_weightstore_without_fast_dirs_is_unaffected():
    """No fast_dirs configured at all -- the common case for every other
    model -- must never touch the new manifest-lookup machinery (this is
    the regression this session's own bug (None manifest -> `in None`
    TypeError) would have hit on literally every other model's first
    fetch() call had it shipped)."""
    from runtime.model_loader import WeightStore

    store = WeightStore(MODEL_DIR)
    name = "model.norm.weight"
    assert store.has(name)
    out, _elapsed, _nbytes = store.fetch([name])
    mx.eval(out[name])
    assert store.fast_tier_tensors == 0
    assert store.fast_tier_bytes == 0
