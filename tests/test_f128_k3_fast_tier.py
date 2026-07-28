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

    store_mixed = WeightStore(MODEL_DIR, fast_dirs=[FAST_ROOT])
    out_mixed, _elapsed, _nbytes = store_mixed.fetch([fast_name, slow_name])
    mx.eval(out_mixed[fast_name], out_mixed[slow_name])

    assert store_mixed.fast_tier_tensors == 1

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
