"""F210: exercise the DeepSeek V4 ports against REAL checkpoint weights.

F204-F209 verified each subsystem's arithmetic against the released reference,
but four of the six did so with random weights. That checks the math and
nothing about layout: a transposed matrix, a mis-sliced fused tensor, or a
wrong name mapping all survive a random-weight test because both sides see the
same wrong thing.

These tests dequantize real released tensors and drive the ports with them, so
the weight layout itself is under test. They are the first checks in this port
that would fail on a naming or reshape error.

Skips when the checkpoint is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "DeepSeek-V4-Flash-0731"

pytestmark = pytest.mark.skipif(
    not (MODEL / "model.safetensors.index.json").is_file(),
    reason="DeepSeek-V4-Flash-0731 not present")

LAYER = 5


@pytest.fixture(scope="module")
def config():
    return json.loads((MODEL / "config.json").read_text())


@pytest.fixture(scope="module")
def index():
    return json.loads(
        (MODEL / "model.safetensors.index.json").read_text())["weight_map"]


@pytest.fixture(scope="module")
def shards():
    return {}


def load(index, shards, name):
    import mlx.core as mx

    shard = index[name]
    if shard not in shards:
        shards[shard] = mx.load(str(MODEL / shard))
    return shards[shard][name]


def dequant(index, shards, stem):
    """Dequantize a released weight/scale pair by its actual dtype."""
    import mlx.core as mx

    from runtime.quant import (dequantize_deepseek_v4_fp8,
                               dequantize_deepseek_v4_int8)

    weight = load(index, shards, f"{stem}.weight")
    if f"{stem}.scale" not in index:
        return weight
    scale = load(index, shards, f"{stem}.scale")
    if weight.dtype == mx.int8:
        return dequantize_deepseek_v4_int8(weight, scale)
    return dequantize_deepseek_v4_fp8(weight, scale)


def test_real_attention_weight_shapes_match_the_config(index, shards, config):
    """Every derived dimension must agree with the released tensors."""
    dim = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    q_rank = config["q_lora_rank"]
    o_rank = config["o_lora_rank"]
    groups = config["o_groups"]
    prefix = f"layers.{LAYER}.attn"

    assert load(index, shards, f"{prefix}.wq_a.weight").shape == (q_rank, dim)
    assert load(index, shards, f"{prefix}.wq_b.weight").shape == (
        heads * head_dim, q_rank)
    # Single shared KV across all heads (num_key_value_heads == 1).
    assert load(index, shards, f"{prefix}.wkv.weight").shape == (head_dim, dim)
    assert load(index, shards, f"{prefix}.wo_a.weight").shape == (
        groups * o_rank, heads * head_dim // groups)
    assert load(index, shards, f"{prefix}.wo_b.weight").shape == (
        dim, groups * o_rank)
    assert load(index, shards, f"{prefix}.attn_sink.weight"
                if f"{prefix}.attn_sink.weight" in index
                else f"{prefix}.attn_sink").shape == (heads,)


def test_real_o_projection_matches_a_grouped_reference(index, shards, config):
    """Drive the grouped o-LoRA with real dequantized weights."""
    import mlx.core as mx

    from runtime.deepseek_v4 import attention_output_projection

    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    prefix = f"layers.{LAYER}.attn"

    wo_a = dequant(index, shards, f"{prefix}.wo_a")
    wo_b = dequant(index, shards, f"{prefix}.wo_b")

    rng = np.random.default_rng(0)
    o = mx.array((rng.normal(size=(1, 2, heads, head_dim)) * 0.05
                  ).astype(np.float32))
    got = np.array(attention_output_projection(
        o, wo_a, wo_b, n_groups=groups, o_lora_rank=o_rank))

    # Independent reference: loop the groups explicitly rather than einsum.
    flat = np.array(o).reshape(1, 2, groups, -1)
    a = np.array(wo_a).reshape(groups, o_rank, flat.shape[-1])
    codes = np.zeros((1, 2, groups, o_rank), np.float32)
    for g in range(groups):
        codes[:, :, g] = flat[:, :, g] @ a[g].T
    expected = codes.reshape(1, 2, groups * o_rank) @ np.array(wo_b).T

    assert got.shape == (1, 2, config["hidden_size"])
    diff = np.abs(got - expected).max()
    assert diff < 2e-2, f"grouped projection diverged, max abs diff {diff}"


def test_group_major_reshape_is_not_interchangeable(index, shards, config):
    """The other reshape keeps every shape valid but pairs wrong groups."""
    import mlx.core as mx

    from runtime.deepseek_v4 import attention_output_projection

    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    groups = config["o_groups"]
    o_rank = config["o_lora_rank"]
    wo_a = dequant(index, shards, f"layers.{LAYER}.attn.wo_a")
    wo_b = dequant(index, shards, f"layers.{LAYER}.attn.wo_b")

    rng = np.random.default_rng(1)
    o = mx.array((rng.normal(size=(1, 1, heads, head_dim)) * 0.05
                  ).astype(np.float32))
    correct = np.array(attention_output_projection(
        o, wo_a, wo_b, n_groups=groups, o_lora_rank=o_rank))

    # rank-major instead of group-major on the output axis
    swapped = mx.swapaxes(
        wo_a.reshape(o_rank, groups, -1), 0, 1).reshape(wo_a.shape)
    other = np.array(attention_output_projection(
        o, swapped, wo_b, n_groups=groups, o_lora_rank=o_rank))
    assert not np.allclose(correct, other, atol=1e-3), (
        "the two reshapes agree, so this test cannot detect the error")


def test_real_compressor_weights_are_bf16_not_quantized(index, shards):
    """The released compressor keeps wkv/wgate in bf16; pin that assumption."""
    import mlx.core as mx

    prefix = f"layers.{LAYER}.attn.compressor"
    for name in ("wkv", "wgate"):
        weight = load(index, shards, f"{prefix}.{name}.weight")
        assert weight.dtype == mx.bfloat16, (
            f"{name} is {weight.dtype}; the port pools in float32 assuming an "
            "unquantized source")
        assert f"{prefix}.{name}.scale" not in index


def test_real_dequantized_weights_are_finite_and_nontrivial(index, shards,
                                                            config):
    """Guard against a dequant that silently returns zeros or NaNs."""
    import mlx.core as mx

    for stem in (f"layers.{LAYER}.attn.wq_a", f"layers.{LAYER}.attn.wo_b",
                 f"layers.{LAYER}.ffn.experts.0.w1",
                 f"layers.{LAYER}.ffn.shared_experts.w1"):
        value = dequant(index, shards, stem)
        array = np.array(value.astype(mx.float32))
        assert np.isfinite(array).all(), f"{stem} dequantized to non-finite"
        assert np.abs(array).max() > 0, f"{stem} dequantized to all zeros"
        # A real weight matrix should not be dominated by a single magnitude.
        assert len(np.unique(np.abs(array).round(6))) > 100, (
            f"{stem} has suspiciously few distinct magnitudes")


def test_hyper_connection_weights_load_and_reduce_real_shapes(index, shards,
                                                              config):
    """Run hc_pre/hc_post on real per-layer HC parameters."""
    import mlx.core as mx

    from runtime.deepseek_v4 import hc_post, hc_pre

    hc_mult = config["hc_mult"]
    dim = config["hidden_size"]
    rng = np.random.default_rng(2)
    x = mx.array((rng.normal(size=(1, 2, hc_mult, dim)) * 0.02
                  ).astype(np.float32))

    reduced, post, comb = hc_pre(
        x,
        load(index, shards, f"layers.{LAYER}.hc_attn_fn"),
        load(index, shards, f"layers.{LAYER}.hc_attn_scale"),
        load(index, shards, f"layers.{LAYER}.hc_attn_base"),
        hc_mult=hc_mult, norm_eps=config["rms_norm_eps"],
        sinkhorn_iters=config["hc_sinkhorn_iters"], eps=config["hc_eps"])
    assert reduced.shape == (1, 2, dim)
    assert np.isfinite(np.array(reduced)).all()

    out = hc_post(reduced, x, post, comb)
    assert out.shape == (1, 2, hc_mult, dim)
    assert np.isfinite(np.array(out)).all()
    # Real Sinkhorn weights are only APPROXIMATELY doubly stochastic. Measured
    # across layers 0/5/20/42 the worst marginal deviation is 0.0788 (layer 0),
    # with 0.020-0.038 elsewhere -- 20 iterations do not fully converge on real
    # weights, though they do on the better-conditioned random ones the F205
    # unit test uses at atol 5e-3. Asserting the tight bound here would be
    # asserting a property the released model does not have.
    value = np.array(comb)
    deviation = max(np.abs(value.sum(axis=-1) - 1).max(),
                    np.abs(value.sum(axis=-2) - 1).max())
    assert deviation < 0.15, (
        f"marginal deviation {deviation:.4f} far exceeds the 0.079 measured "
        "across sampled layers; Sinkhorn is not converging as it should")
    assert deviation > 0, "exactly doubly stochastic would be suspicious"
