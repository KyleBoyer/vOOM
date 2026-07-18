"""Kimi Delta Attention (KDA) per-layer recurrent state.

Structurally unrelated to the token-indexed KVCache family in kv_cache.py /
kv_paged.py: a KDA layer's state is a fixed-size (num_heads, head_dim,
head_dim) matrix plus a tiny (kernel_size - 1)-token causal-conv history,
both O(1) in context length -- there is nothing to page or spill regardless
of how long the sequence gets. See docs/future_lossless_techniques.md F92.
"""

from __future__ import annotations

import mlx.core as mx


class KDAStateCache:
    """Holds one recurrent state + conv history per KDA layer."""

    def __init__(self, num_layers: int):
        self._state: list[mx.array | None] = [None] * num_layers
        self._conv: list[tuple | None] = [None] * num_layers

    def state(self, layer: int) -> mx.array | None:
        return self._state[layer]

    def set_state(self, layer: int, state: mx.array) -> None:
        self._state[layer] = state

    def conv_history(self, layer: int) -> tuple | None:
        return self._conv[layer]

    def set_conv_history(self, layer: int, history: tuple) -> None:
        self._conv[layer] = history

    def reset(self) -> None:
        for i in range(len(self._state)):
            self._state[i] = None
            self._conv[i] = None
