"""Compact KDA factor-buffer commit invariants."""

from __future__ import annotations

import mlx.core as mx
import pytest

from runtime.kda_state import KDAStateCache
from runtime.kda_state import _native_fused_kda_factor_step


@pytest.mark.parametrize('seed', [64013, 91703])
@pytest.mark.parametrize('populated', [False, True])
def test_qwen_scalar_factors_match_every_serial_prefix_at_huihui_geometry(seed, populated):
    """Existing scalar factor path, not the Kimi per-key/native fused path.

    Installed Huihui config has 48 expanded value heads and 128x128 FP32
    recurrent matrices. A depth-four draft verifies five serial positions.
    No model weights or captured prompt are needed for this arithmetic gate.
    """
    from runtime.qwen35 import _sequential_gated_delta_rule

    mx.random.seed(seed)
    heads, dim, positions = 48, 128, 5
    base = KDAStateCache(2)  # the second, untouched layer must stay untouched
    initial = (mx.random.normal((1, heads, dim, dim)) / 16
        if populated else mx.zeros((1, heads, dim, dim)))
    initial_history = (mx.zeros((1, 3, 10240), dtype=mx.bfloat16),)
    mx.eval(initial, *initial_history)
    if populated:
        base.set_state(0, initial)
        base.set_conv_history(0, initial_history)
    live = base.fork()
    live.begin_factor_capture()
    current = initial
    endpoints, histories = [], []
    for position in range(positions):
        q = mx.random.normal((1, 1, heads, dim)) / (dim ** 0.5)
        k = mx.random.normal(q.shape) / (dim ** 0.5)
        v = mx.random.normal(q.shape)
        beta = mx.sigmoid(mx.random.normal((1, 1, heads)))
        decay = -mx.exp(mx.random.normal((1, 1, heads))) / 10
        history = (mx.full((1, 3, 10240), position + 1, dtype=mx.bfloat16),)
        mx.eval(q, k, v, beta, decay, *history)
        _output, current = _sequential_gated_delta_rule(q, k, v, beta, decay, current)
        # Ordinary verifier state is materialized between one-position calls.
        mx.eval(current)
        endpoints.append(current)
        histories.append(history)
        live.capture_factor_step(0, gate=decay[:, 0], key=k[:, 0],
            value=v[:, 0], beta=beta[:, 0], conv_history=history)
    window = live.finish_factor_capture(positions)
    assert window is not None and not live.factor_capture_active
    assert window.nbytes() < (positions - 1) * initial.nbytes / 10
    for count in range(positions + 1):
        restored = window.commit_prefix(base, count, native_fused=False)
        assert restored.state(1) is None and restored.conv_history(1) is None
        if count == 0:
            assert restored.state(0) is base.state(0)
            assert restored.conv_history(0) is base.conv_history(0)
        else:
            assert restored.state(0).dtype == mx.float32
            assert bool(mx.array_equal(restored.state(0), endpoints[count - 1]).item())
            assert bool(mx.array_equal(restored.conv_history(0)[0], histories[count - 1][0]).item())
    assert base.state(0) is (initial if populated else None)
    assert base.conv_history(0) is (initial_history if populated else None)


def _advance(state, gate, key, value, beta):
    state = state * mx.exp(gate)[..., None]
    pred = mx.sum(key[..., None] * state, axis=-2)
    residual = value - pred
    return state + (
        beta[..., None] * key
    )[..., None] * residual[..., None, :]


def test_factor_window_replays_every_prefix_exactly():
    mx.random.seed(20260730)
    layers, positions, heads, width = 3, 4, 2, 5
    base = KDAStateCache(layers)
    original = []
    for layer in range(layers):
        state = mx.random.normal((1, heads, width, width))
        history = tuple(
            mx.random.normal((1, 2, heads * width))
            for _ in range(3)
        )
        mx.eval(state, *history)
        base.set_state(layer, state)
        base.set_conv_history(layer, history)
        original.append(state)

    live = base.fork()
    live.begin_factor_capture()
    expected_by_prefix = []
    for position in range(positions):
        expected = []
        for layer in range(layers):
            gate = -mx.abs(mx.random.normal((1, heads, width)))
            key = mx.random.normal((1, heads, width))
            value = mx.random.normal((1, heads, width))
            beta = mx.sigmoid(mx.random.normal((1, heads)))
            history = tuple(
                mx.full((1, 2, heads * width), position + layer + index)
                for index in range(3)
            )
            live.capture_factor_step(
                layer,
                gate=gate,
                key=key,
                value=value,
                beta=beta,
                conv_history=history,
            )
            original[layer] = _advance(
                original[layer], gate, key, value, beta)
            # Reference ordinary decode has a materialized recurrent endpoint
            # between positions; preserve that rounding boundary in the
            # oracle rather than comparing two equivalent lazy graphs.
            mx.eval(original[layer])
            expected.append(original[layer])
        expected_by_prefix.append(expected)
    window = live.finish_factor_capture(positions)
    assert window is not None

    for prefix in range(1, positions + 1):
        restored = window.commit_prefix(base, prefix)
        for layer in range(layers):
            actual = restored.state(layer)
            expected = expected_by_prefix[prefix - 1][layer]
            mx.eval(actual, expected)
            assert mx.array_equal(actual, expected).item()
            assert restored.conv_history(layer)[0][0, 0, 0].item() == (
                prefix - 1 + layer)


def test_factor_storage_is_smaller_than_dense_endpoints_for_k3_geometry():
    # Per K3 KDA layer/position: gate+k+v are H*D fp32, beta is H fp32.
    # A dense endpoint is H*D*D fp32. Conv histories are included
    # conservatively for all q/k/v channels and two prior positions.
    heads, width, conv_history = 96, 128, 2
    factor_bytes = (
        (3 * heads * width + heads) * 4
        + 3 * heads * width * conv_history * 2
    )
    endpoint_bytes = heads * width * width * 4
    assert factor_bytes < endpoint_bytes / 6


def test_factor_storage_is_smaller_than_dense_qwen_flash_endpoints():
    # Released Qwen3.8-Flash-Next geometry: 36 DeltaNet layers, 48 expanded
    # value/key heads, Dk=Dv=128, and one BF16 3-row convolution history over
    # the 10,240-wide fused Q/K/V input. A depth-three verifier retains three
    # strict-prefix dense endpoints today, versus four token-factor records.
    layers, positions, strict_prefixes = 36, 4, 3
    heads, dim, history_rows, fused_qkv = 48, 128, 3, 10240
    factor_step_bytes = (
        (heads + heads * dim + heads * dim + heads) * 4
        + history_rows * fused_qkv * 2
    )
    factor_window_bytes = layers * positions * factor_step_bytes
    dense_endpoint_bytes = (
        layers * strict_prefixes * heads * dim * dim * 4)
    assert factor_window_bytes < dense_endpoint_bytes / 20


def test_native_factor_step_matches_plain_mlx_recurrence():
    mx.random.seed(20260731)
    batch, heads, width = 1, 3, 17
    state = mx.random.normal((batch, heads, width, width))
    gate = -5.0 * mx.sigmoid(
        mx.random.normal((batch, heads, width)))
    key = mx.random.normal((batch, heads, width))
    key = key * mx.rsqrt(
        mx.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    value = mx.random.normal((batch, heads, width))
    beta = mx.sigmoid(mx.random.normal((batch, heads)))

    expected = _advance(state, gate, key, value, beta)
    actual = _native_fused_kda_factor_step(
        gate, key, value, beta, state)
    mx.eval(expected, actual)

    # The custom kernel serially reduces each state column while MLX may use a
    # tree reduction.  This is a numerical-equivalence probe, not the
    # released-model token-identity admission gate.
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
