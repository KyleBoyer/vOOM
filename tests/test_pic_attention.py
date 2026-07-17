from __future__ import annotations

import mlx.core as mx
import pytest

from runtime.pic_attention import (
    apply_rope, position_free_paged_attention, rope_cache,
)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")
@pytest.mark.parametrize("dtype,dim", [
    (mx.float32, 64), (mx.bfloat16, 64), (mx.bfloat16, 128),
])
def test_position_free_paged_attention_matches_rotated_reference(dtype, dim):
    mx.random.seed(713)
    batch, nq, nkv, tq, tk, block = 1, 4, 2, 3, 19, 4
    q_raw = (mx.random.normal((batch, nq, tq, dim)) * 0.1).astype(dtype)
    k_logical = (mx.random.normal((batch, nkv, tk, dim)) * 0.1).astype(dtype)
    v_logical = (mx.random.normal((batch, nkv, tk, dim)) * 0.1).astype(dtype)
    q_positions = mx.array([16, 17, 18], dtype=mx.int32)
    key_positions = mx.arange(tk, dtype=mx.int32)
    cos, sin = rope_cache(32, dim, 10_000.0, dtype=dtype)
    q = apply_rope(q_raw, q_positions, cos, sin)
    k = apply_rope(k_logical, key_positions, cos, sin)
    mask = mx.where(
        key_positions[None, :] <= q_positions[:, None],
        0.0, float("-inf")).astype(dtype)
    reference = mx.fast.scaled_dot_product_attention(
        q, k, v_logical, scale=dim ** -0.5, mask=mask)

    padded = ((tk + block - 1) // block) * block
    pad = padded - tk
    k_padded = mx.pad(k_logical, [(0, 0), (0, 0), (0, pad), (0, 0)])
    v_padded = mx.pad(v_logical, [(0, 0), (0, 0), (0, pad), (0, 0)])
    k_pages = k_padded.reshape(batch, nkv, padded // block, block, dim)
    v_pages = v_padded.reshape(batch, nkv, padded // block, block, dim)
    k_pages = k_pages.transpose(0, 2, 1, 3, 4).reshape(
        padded // block, nkv, block, dim)
    v_pages = v_pages.transpose(0, 2, 1, 3, 4).reshape(
        padded // block, nkv, block, dim)
    table = mx.arange(padded // block, dtype=mx.int32)
    candidate = position_free_paged_attention(
        q, k_pages, v_pages, table, key_positions, q_positions, cos, sin,
        scale=dim ** -0.5)
    mx.eval(reference, candidate)

    tolerance = 2e-5 if dtype == mx.float32 else 2e-2
    assert mx.allclose(candidate, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")
def test_block_table_can_share_one_physical_unrotated_page_at_two_positions():
    mx.random.seed(19)
    nq = nkv = 2
    dim = 64
    block = 4
    # Logical sequence A,B,A references physical page A twice without copying.
    k_pages = (mx.random.normal((2, nkv, block, dim)) * 0.1).astype(mx.float32)
    v_pages = (mx.random.normal((2, nkv, block, dim)) * 0.1).astype(mx.float32)
    table = mx.array([0, 1, 0], dtype=mx.int32)
    key_positions = mx.arange(12, dtype=mx.int32)
    query_positions = mx.array([11], dtype=mx.int32)
    q_raw = (mx.random.normal((1, nq, 1, dim)) * 0.1).astype(mx.float32)
    cos, sin = rope_cache(16, dim, 10_000.0, dtype=mx.float32)
    q = apply_rope(q_raw, query_positions, cos, sin)

    candidate = position_free_paged_attention(
        q, k_pages, v_pages, table, key_positions, query_positions, cos, sin,
        scale=dim ** -0.5)
    gathered_k = mx.concatenate((k_pages[0], k_pages[1], k_pages[0]), axis=1)
    gathered_v = mx.concatenate((v_pages[0], v_pages[1], v_pages[0]), axis=1)
    reference = mx.fast.scaled_dot_product_attention(
        q, apply_rope(gathered_k[None], key_positions, cos, sin),
        gathered_v[None], scale=dim ** -0.5)
    mx.eval(candidate, reference)

    assert mx.allclose(candidate, reference, rtol=2e-5, atol=2e-5)
    physical_bytes = k_pages.nbytes + v_pages.nbytes
    logical_duplicate_bytes = gathered_k.nbytes + gathered_v.nbytes
    assert physical_bytes * 3 == logical_duplicate_bytes * 2


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")
@pytest.mark.parametrize("dim", [64, 128])
def test_fp32_rope_table_matches_mlx_bfloat16_keys_within_tolerance(dim):
    """CORRECTED (2026-07-17): this test's original name and body asserted
    `mx.array_equal` -- exact bit equality between this project's own
    fp32-table RoPE and MLX's opaque, Metal-compiled `mx.fast.rope` kernel.
    That held on the local M4 dev machine for both dim=64 and dim=128, but
    a real GitHub Actions run on its macos-14 runner (a different Apple
    Silicon generation) diverged by roughly one bfloat16 ULP at dim=128
    only -- confirmed via an actual CI failure, not guessed. `mx.fast.rope`
    is a vendor-compiled kernel whose internal accumulation/tiling can
    legitimately differ by GPU generation; asserting byte-for-byte equality
    against it across different hardware was a stronger claim than this
    project can actually guarantee. Loosened to `mx.allclose`, using this
    file's own `test_position_free_paged_attention_matches_rotated_reference`
    convention of `tolerance = 2e-5 if dtype == mx.float32 else 2e-2` --
    both `candidate` and `reference` here are bfloat16 (only ~2-3 significant
    decimal digits), so 2e-2 is the correct bucket, not a new, weaker
    standard invented just for this test. (A first attempt at this fix
    wrongly reused the float32 branch's 2e-5 tolerance for this bfloat16
    comparison and still failed on CI -- corrected here.)
    """
    mx.random.seed(991)
    values = mx.random.normal((1, 2, 7, dim)).astype(mx.bfloat16)
    positions = mx.arange(3, 10, dtype=mx.int32)
    cos, sin = rope_cache(16, dim, 10_000.0, dtype=mx.float32)
    candidate = apply_rope(values, positions, cos, sin).astype(mx.bfloat16)
    reference = mx.fast.rope(
        values, dim, traditional=False, base=10_000.0,
        scale=1.0, offset=3)
    mx.eval(candidate, reference)
    assert mx.allclose(candidate, reference, rtol=2e-2, atol=2e-2)
