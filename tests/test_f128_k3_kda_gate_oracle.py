"""F128: Kimi K3's KDA gate formula (safe_gate/lower_bound, full-rank gate).

K3's real config.json linear_attn_config sets `use_full_rank_gate: true` and
`gate_lower_bound: -5.0` -- both absent (None/False) in the original Kimi
Linear 48B checkpoint F92 already verified, and both genuinely change the
real fla-org/flash-linear-attention KDA gate formula (fetched directly from
github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py on
2026-07-27, both the pure-torch `naive_kda_gate`/`naive_kda_lowerbound_gate`
reference functions AND the real Triton kernel's own gate branch, which
agree). This test transcribes those two real reference functions verbatim
and checks runtime.kimi_linear._kda_attention's inline gate computation
(extracted here as a standalone function mirroring its exact code, since
_kda_attention itself needs a full real-weight KDA layer to run end to end)
against them on synthetic data.

Also confirmed directly against a real downloaded K3 shard (not re-derived
here, see runtime/kimi_linear.py's own comment): A_log is saved with
head_dim elements (128), not num_heads (96) -- the real gate.py Triton
kernel indexes A_log strictly by head index up to H (passed explicitly,
never inferred from A_log's own tensor length), so this test also checks
that slicing to the first H elements before use is what the real gate
functions expect (they take an already-H-length A_log as their contract).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_torch_skip = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this venv")


# Verbatim from the real fla-org/flash-linear-attention fla/ops/kda/gate.py
# (fetched 2026-07-27), used here only as a numerical oracle, never
# imported into the runtime.
def _real_naive_kda_gate(g, A_log, dt_bias=None):
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = -A_log.view(H, 1).float().exp() * F.softplus(g.float())
    return g


def _real_naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0):
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g


# Mirrors runtime.kimi_linear._kda_attention's exact gate computation
# (extracted so this test can exercise it on plain synthetic arrays without
# needing a full real-weight KDA layer).
def _mx_kda_gate(g_input, dt_bias, A_log_full, H, lower_bound):
    g_raw = g_input.astype(mx.float32) + dt_bias
    A = mx.exp(A_log_full[:H].astype(mx.float32)).reshape(1, 1, H, 1)
    if lower_bound:
        return lower_bound * mx.sigmoid(A * g_raw)
    softplus_g = mx.logaddexp(g_raw, mx.zeros_like(g_raw))
    return -A * softplus_g


@_torch_skip
def test_kda_gate_no_lower_bound_matches_real_reference():
    rng = np.random.default_rng(0)
    B, L, H, D = 2, 3, 4, 8
    g_np = rng.standard_normal((B, L, H, D)).astype(np.float32)
    dt_bias_np = rng.standard_normal((H, D)).astype(np.float32)
    a_log_np = rng.standard_normal((H,)).astype(np.float32)

    ref = _real_naive_kda_gate(
        torch.from_numpy(g_np), torch.from_numpy(a_log_np),
        torch.from_numpy(dt_bias_np)).numpy()

    mine = _mx_kda_gate(
        mx.array(g_np), mx.array(dt_bias_np), mx.array(a_log_np), H, 0.0)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)


@_torch_skip
def test_kda_gate_with_lower_bound_matches_real_reference():
    rng = np.random.default_rng(1)
    B, L, H, D = 2, 3, 4, 8
    lower_bound = -5.0
    g_np = rng.standard_normal((B, L, H, D)).astype(np.float32)
    dt_bias_np = rng.standard_normal((H, D)).astype(np.float32)
    a_log_np = rng.standard_normal((H,)).astype(np.float32)

    ref = _real_naive_kda_lowerbound_gate(
        torch.from_numpy(g_np), torch.from_numpy(a_log_np),
        torch.from_numpy(dt_bias_np), lower_bound).numpy()

    mine = _mx_kda_gate(
        mx.array(g_np), mx.array(dt_bias_np), mx.array(a_log_np), H, lower_bound)
    mx.eval(mine)

    np.testing.assert_allclose(np.array(mine), ref, atol=1e-5, rtol=1e-5)
    # Sanity: the whole point of "safe_gate" is clamping to [lower_bound, 0).
    assert bool(mx.all(mine >= lower_bound).item())
    assert bool(mx.all(mine < 0).item())


@_torch_skip
def test_kda_gate_a_log_over_allocated_tail_is_ignored():
    """F128's real finding: K3's on-disk A_log has head_dim (128) elements
    but only the first num_heads (96) are ever read by the real kernel.
    Appending garbage after the first H elements must not change the
    result -- this is the actual real-checkpoint shape this project has to
    handle, not just a same-length synthetic case."""
    rng = np.random.default_rng(2)
    B, L, H, D = 1, 2, 4, 8
    g_np = rng.standard_normal((B, L, H, D)).astype(np.float32)
    dt_bias_np = rng.standard_normal((H, D)).astype(np.float32)
    a_log_exact_np = rng.standard_normal((H,)).astype(np.float32)
    a_log_over_allocated_np = np.concatenate(
        [a_log_exact_np, rng.standard_normal((D - H,)).astype(np.float32)])
    assert a_log_over_allocated_np.shape == (D,)

    exact = _mx_kda_gate(
        mx.array(g_np), mx.array(dt_bias_np), mx.array(a_log_exact_np), H, 0.0)
    over_allocated = _mx_kda_gate(
        mx.array(g_np), mx.array(dt_bias_np), mx.array(a_log_over_allocated_np), H, 0.0)
    mx.eval(exact, over_allocated)

    np.testing.assert_array_equal(np.array(exact), np.array(over_allocated))
