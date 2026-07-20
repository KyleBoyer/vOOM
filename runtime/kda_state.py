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

    def nbytes(self) -> int:
        """Resident bytes owned by recurrent matrices and conv histories."""
        total = sum(value.nbytes for value in self._state if value is not None)
        for history in self._conv:
            if history is not None:
                total += sum(value.nbytes for value in history if value is not None)
        return total

    def fork(self) -> "KDAStateCache":
        """Capture an immutable exact endpoint without copying array buffers.

        Recurrent updates construct replacement arrays and install them with
        ``set_state``/``set_conv_history``; they never mutate an installed
        endpoint in place. A new owner can therefore share evaluated arrays
        until either branch advances, at which point copy-on-write graph
        construction naturally separates them.
        """
        result = KDAStateCache(len(self._state))
        result._state = list(self._state)
        result._conv = [
            tuple(history) if history is not None else None
            for history in self._conv
        ]
        arrays = [value for value in result._state if value is not None]
        arrays.extend(
            value
            for history in result._conv if history is not None
            for value in history if value is not None)
        if arrays:
            mx.eval(*arrays)
        return result

    def synchronize(self) -> None:
        """Finish every endpoint array before ownership crosses HTTP threads."""
        arrays = [value for value in self._state if value is not None]
        arrays.extend(
            value
            for history in self._conv if history is not None
            for value in history if value is not None)
        if arrays:
            mx.eval(*arrays)
            mx.synchronize()

    def export_arrays(self) -> dict[str, mx.array]:
        """Stable safetensors mapping for one exact recurrent endpoint."""
        arrays: dict[str, mx.array] = {}
        for layer, value in enumerate(self._state):
            if value is not None:
                arrays[f"kda_state_{layer}"] = value
        for layer, history in enumerate(self._conv):
            if history is None:
                continue
            for index, value in enumerate(history):
                if value is not None:
                    arrays[f"kda_conv_{layer}_{index}"] = value
        return arrays

    @classmethod
    def from_arrays(
        cls, num_layers: int, arrays: dict[str, mx.array], *,
        expected_layers=(),
    ) -> "KDAStateCache":
        """Restore a validated endpoint from ``export_arrays`` output."""
        result = cls(num_layers)
        histories: dict[int, dict[int, mx.array]] = {}
        for name, value in arrays.items():
            if name.startswith("kda_state_"):
                suffix = name[len("kda_state_"):]
                if not suffix.isdigit():
                    raise ValueError("invalid recurrent state tensor name")
                layer = int(suffix)
                if not 0 <= layer < num_layers or result._state[layer] is not None:
                    raise ValueError("invalid duplicate recurrent state layer")
                result._state[layer] = value
            elif name.startswith("kda_conv_"):
                suffix = name[len("kda_conv_"):]
                parts = suffix.split("_")
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    raise ValueError("invalid recurrent conv tensor name")
                layer, index = map(int, parts)
                if not 0 <= layer < num_layers:
                    raise ValueError("invalid recurrent conv layer")
                values = histories.setdefault(layer, {})
                if index in values:
                    raise ValueError("duplicate recurrent conv tensor")
                values[index] = value
        for layer, values in histories.items():
            if set(values) != set(range(len(values))):
                raise ValueError("recurrent conv history has an index gap")
            result._conv[layer] = tuple(
                values[index] for index in range(len(values)))
        expected = tuple(int(layer) for layer in expected_layers)
        for layer in expected:
            if (not 0 <= layer < num_layers
                    or result._state[layer] is None
                    or result._conv[layer] is None):
                raise ValueError(
                    f"recurrent checkpoint is missing layer {layer}")
        result.synchronize()
        return result
