"""Compact KDA factor-buffer commit invariants."""

from __future__ import annotations

import mlx.core as mx

from runtime.kda_state import KDAStateCache
from runtime.kda_state import _native_fused_kda_factor_step


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
