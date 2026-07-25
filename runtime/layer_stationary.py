"""F94: layer-stationary (layer-major) dense prefill.

The ordinary prefill sweep (``StreamingEngine._sweep``, calling
``layer_runner.run_block`` inside a chunk loop in ``generate()``) is
chunk-major: for each chunk of prompt positions, iterate every layer. Every
layer's weights are therefore re-fetched once per chunk -- at a narrow
memory-safety chunk width this can mean dozens of complete re-reads of the
whole checkpoint for one prompt (see F94's proposal in
docs/future_lossless_techniques.md).

This module provides the inverted schedule instead: for each layer, iterate
every tile of the prompt (in causal/sequential order), so that layer's
weights are fetched exactly once regardless of how many tiles the prompt is
split into. Mathematically this is the SAME computation in a different
order -- ``run_block`` itself is unchanged, and per-layer KV state still
accumulates causally exactly as it does in chunk-major mode, since tiles of
one layer are still processed strictly in position order before moving to
the next layer. The only genuinely new bookkeeping is staging each layer's
per-tile output hidden states so the NEXT layer's pass can consume them
(chunk-major never needs this: within one chunk, hidden state already flows
naturally through consecutive ``run_block`` calls for that chunk alone).
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from . import layer_runner
from .config import ModelConfig


def run_layer_stationary_sweep(
    x: mx.array,
    cfg: ModelConfig,
    kv: "object",
    offset: int,
    tile_width: int,
    get_layer_weights: Callable[[int], dict],
    *,
    mlp_last_only: bool = False,
    rope_freqs: mx.array | None = None,
    rope_mscale: float = 1.0,
    fused_swiglu: bool = False,
) -> mx.array:
    """Layer-major dense prefill sweep. x: (1, L, hidden) fresh embeddings.

    ``get_layer_weights(layer)`` is called exactly once per layer regardless
    of ``tile_width`` -- callers wanting to prove the I/O-reduction property
    (not just numerical equivalence) should wrap it with a call counter, as
    tests/test_f94_layer_stationary_oracle.py does.

    ``mlp_last_only`` matches chunk-major's own existing convention exactly:
    applied at every tile once ``layer == num_hidden_layers - 1``, whether or
    not that particular tile contains the true final prompt position. Only
    the tile that DOES contain it (necessarily the last tile processed,
    since tiles run in causal order) produces a value anything downstream
    ever reads; every other last-layer tile's sliced-to-one-position output
    is discarded exactly as chunk-major already discards every non-final
    chunk's last-layer output today -- this function does not introduce
    that discard, it only relocates where it happens.
    """
    if tile_width <= 0:
        raise ValueError("tile_width must be positive")
    n = cfg.num_hidden_layers
    total = int(x.shape[1])
    for layer in range(n):
        w = get_layer_weights(layer)
        last_layer = layer == n - 1
        slice_last = mlp_last_only and last_layer
        tiles = []
        pos = 0
        while pos < total:
            end = min(pos + tile_width, total)
            xt = x[:, pos:end, :]
            yt = layer_runner.run_block(
                xt, w, f"model.layers.{layer}", cfg, kv, layer,
                offset + pos,
                mlp_last_only=slice_last,
                rope_freqs=rope_freqs, rope_mscale=rope_mscale,
                fused_swiglu=fused_swiglu,
            )
            if slice_last:
                # Every tile got sliced to its OWN last position, not the
                # true final prompt position -- only the tile actually
                # containing it (necessarily the last one processed, since
                # tiles run in causal order) is meaningful. Keep replacing
                # rather than concatenating, matching chunk-major's own
                # existing discard of every non-final chunk's sliced output
                # (see this function's docstring).
                tiles = [yt]
            else:
                tiles.append(yt)
            pos = end
        x = tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=1)
    return x
