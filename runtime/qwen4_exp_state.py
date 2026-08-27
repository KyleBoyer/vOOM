"""Exact request-local auxiliary state for Qwen3.8 Flash-Next."""

from __future__ import annotations

import mlx.core as mx


class Qwen4ExpStateCache:
    """PLE histories and QSA raw index keys alongside the ordinary KV cache."""

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError("Qwen4-Exp state cache needs decoder layers")
        self.qsa_keys: list[mx.array | None] = [None] * num_layers
        self.qsa_positions: list[mx.array | None] = [None] * num_layers
        self.ple_conv: list[mx.array | None] = [None] * num_layers
        self.ple_context: list[tuple[int, ...] | None] = [None] * num_layers
        self.ple_lengths: list[int] = [0] * num_layers

    def update_qsa(
        self, layer: int, keys: mx.array, positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.qsa_keys[layer] is None:
            self.qsa_keys[layer] = keys
            self.qsa_positions[layer] = positions
        else:
            self.qsa_keys[layer] = mx.concatenate(
                [self.qsa_keys[layer], keys], axis=1)
            self.qsa_positions[layer] = mx.concatenate(
                [self.qsa_positions[layer], positions], axis=-1)
        return self.qsa_keys[layer], self.qsa_positions[layer]

    def fork(self) -> "Qwen4ExpStateCache":
        branch = Qwen4ExpStateCache(len(self.qsa_keys))
        branch.qsa_keys = list(self.qsa_keys)
        branch.qsa_positions = list(self.qsa_positions)
        branch.ple_conv = list(self.ple_conv)
        branch.ple_context = list(self.ple_context)
        branch.ple_lengths = list(self.ple_lengths)
        return branch

    def trim(self, length: int) -> None:
        """Trim append-only QSA state; recurrent PLE rewind is unsupported."""
        if length < 0:
            raise ValueError("Qwen4-Exp trim length must be non-negative")
        for layer, current in enumerate(self.ple_lengths):
            if current and length != current:
                raise ValueError(
                    "Qwen4-Exp PLE recurrence cannot be rewound without an "
                    f"exact checkpoint (layer {layer}, {current} -> {length})")
        pending = []
        for layer, keys in enumerate(self.qsa_keys):
            if keys is None:
                continue
            if keys.shape[1] > length:
                self.qsa_keys[layer] = keys[:, :length]
                self.qsa_positions[layer] = self.qsa_positions[layer][..., :length]
                pending.extend((
                    self.qsa_keys[layer], self.qsa_positions[layer]))
        if pending:
            mx.eval(*pending)

    def restore_recurrent_prefix(
        self,
        endpoint: "Qwen4ExpStateCache",
        length: int,
    ) -> None:
        """Restore exact PLE recurrence, then trim append-only QSA state.

        Serial speculative verification retains only PLE convolution/context
        tensors for strict prefixes.  Copying every long-context QSA key array
        at every proposal depth would scale as ``context * depth`` for state
        that is already safely trimmable.  Install the non-trimmable PLE
        endpoint first so :meth:`trim` may then shorten QSA in place.
        """
        if not isinstance(endpoint, Qwen4ExpStateCache):
            raise TypeError("Qwen4 recurrent endpoint has the wrong type")
        if len(endpoint.ple_conv) != len(self.ple_conv):
            raise ValueError("Qwen4 recurrent endpoint layer count mismatch")
        target = int(length)
        if target < 0:
            raise ValueError("Qwen4 recurrent endpoint length must be non-negative")
        for layer, current in enumerate(self.ple_conv):
            candidate = endpoint.ple_conv[layer]
            if current is None:
                if candidate is not None:
                    raise ValueError(
                        "Qwen4 recurrent endpoint contains an unexpected PLE layer")
                continue
            if (
                candidate is None
                or endpoint.ple_context[layer] is None
                or endpoint.ple_lengths[layer] != target
            ):
                raise ValueError(
                    f"Qwen4 recurrent endpoint is incomplete at layer {layer}")
            self.ple_conv[layer] = candidate
            self.ple_context[layer] = endpoint.ple_context[layer]
            self.ple_lengths[layer] = target
        self.trim(target)

    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for values in (
                self.qsa_keys, self.qsa_positions, self.ple_conv)
            for value in values
            if value is not None
        )

    def synchronize(self) -> None:
        arrays = [
            value
            for values in (
                self.qsa_keys, self.qsa_positions, self.ple_conv)
            for value in values
            if value is not None
        ]
        if arrays:
            mx.eval(*arrays)
            mx.synchronize()

    def validate(
        self,
        *,
        expected_qsa_layers=(),
        expected_ple_layers=(),
        expected_length: int,
        indexer_dim: int,
        ple_context_len: int,
        ple_state_len: int,
        ple_width: int,
    ) -> None:
        """Fail closed unless this is one complete released-model endpoint."""
        length = int(expected_length)
        if length < 0:
            raise ValueError("Qwen4 endpoint length must be non-negative")
        qsa_layers = tuple(int(layer) for layer in expected_qsa_layers)
        ple_layers = tuple(int(layer) for layer in expected_ple_layers)
        if any(not 0 <= layer < len(self.qsa_keys) for layer in (*qsa_layers, *ple_layers)):
            raise ValueError("Qwen4 auxiliary layer is outside the decoder")
        if {
            index for index, value in enumerate(self.qsa_keys)
            if value is not None
        } != set(qsa_layers):
            raise ValueError("Qwen4 checkpoint has incomplete QSA keys")
        if {
            index for index, value in enumerate(self.qsa_positions)
            if value is not None
        } != set(qsa_layers):
            raise ValueError("Qwen4 checkpoint has incomplete QSA positions")
        if {
            index for index, value in enumerate(self.ple_conv)
            if value is not None
        } != set(ple_layers):
            raise ValueError("Qwen4 checkpoint has incomplete PLE convolution state")
        for layer in qsa_layers:
            keys = self.qsa_keys[layer]
            positions = self.qsa_positions[layer]
            if (keys.dtype != mx.bfloat16
                    or tuple(keys.shape) != (1, length, int(indexer_dim))):
                raise ValueError(
                    f"Qwen4 QSA key geometry mismatch at layer {layer}")
            if (positions.dtype != mx.int32
                    or tuple(positions.shape) != (1, length)):
                raise ValueError(
                    f"Qwen4 QSA position geometry mismatch at layer {layer}")
            expected = mx.arange(length, dtype=mx.int32)[None]
            if not bool(mx.all(positions == expected).item()):
                raise ValueError(
                    f"Qwen4 QSA positions are not canonical at layer {layer}")
        for layer in ple_layers:
            conv = self.ple_conv[layer]
            context = self.ple_context[layer]
            if (conv.dtype != mx.bfloat16
                    or tuple(conv.shape) != (
                        1, int(ple_state_len), int(ple_width))):
                raise ValueError(
                    f"Qwen4 PLE convolution geometry mismatch at layer {layer}")
            if context is None or len(context) != int(ple_context_len):
                raise ValueError(
                    f"Qwen4 PLE token context is incomplete at layer {layer}")
            if int(self.ple_lengths[layer]) != length:
                raise ValueError(
                    f"Qwen4 PLE length mismatch at layer {layer}")
        self.synchronize()

    def export_arrays(self) -> dict[str, mx.array]:
        """Stable safetensors mapping for one exact QSA/PLE endpoint."""
        arrays: dict[str, mx.array] = {}
        for layer, value in enumerate(self.qsa_keys):
            if value is not None:
                arrays[f"qwen4_qsa_key_{layer}"] = value
        for layer, value in enumerate(self.qsa_positions):
            if value is not None:
                arrays[f"qwen4_qsa_position_{layer}"] = value
        for layer, value in enumerate(self.ple_conv):
            if value is None:
                continue
            context = self.ple_context[layer]
            if context is None:
                raise ValueError(
                    f"Qwen4 PLE layer {layer} is missing token context")
            arrays[f"qwen4_ple_conv_{layer}"] = value
            arrays[f"qwen4_ple_context_{layer}"] = mx.array(
                context, dtype=mx.int32)
            arrays[f"qwen4_ple_length_{layer}"] = mx.array(
                [self.ple_lengths[layer]], dtype=mx.int32)
        return arrays

    @classmethod
    def from_arrays(
        cls,
        num_layers: int,
        arrays: dict[str, mx.array],
        *,
        expected_qsa_layers=(),
        expected_ple_layers=(),
        expected_length: int,
        indexer_dim: int,
        ple_context_len: int,
        ple_state_len: int,
        ple_width: int,
    ) -> "Qwen4ExpStateCache":
        result = cls(num_layers)
        prefixes = {
            "qwen4_qsa_key_": result.qsa_keys,
            "qwen4_qsa_position_": result.qsa_positions,
            "qwen4_ple_conv_": result.ple_conv,
        }
        contexts: dict[int, mx.array] = {}
        lengths: dict[int, mx.array] = {}
        for name, value in arrays.items():
            target = None
            suffix = ""
            for prefix, candidate in prefixes.items():
                if name.startswith(prefix):
                    target = candidate
                    suffix = name[len(prefix):]
                    break
            if target is not None:
                if (not suffix.isdigit() or not 0 <= int(suffix) < num_layers
                        or target[int(suffix)] is not None):
                    raise ValueError("invalid duplicate Qwen4 auxiliary tensor")
                target[int(suffix)] = value
                continue
            if name.startswith("qwen4_ple_context_"):
                suffix = name[len("qwen4_ple_context_"):]
                mapping = contexts
            elif name.startswith("qwen4_ple_length_"):
                suffix = name[len("qwen4_ple_length_"):]
                mapping = lengths
            else:
                raise ValueError("unknown Qwen4 auxiliary tensor")
            if (not suffix.isdigit() or not 0 <= int(suffix) < num_layers
                    or int(suffix) in mapping):
                raise ValueError("invalid duplicate Qwen4 PLE metadata tensor")
            mapping[int(suffix)] = value
        for layer in set(contexts) | set(lengths):
            context = contexts.get(layer)
            length = lengths.get(layer)
            if (context is None or length is None
                    or context.dtype != mx.int32
                    or tuple(context.shape) != (int(ple_context_len),)
                    or length.dtype != mx.int32
                    or tuple(length.shape) != (1,)):
                raise ValueError("invalid Qwen4 PLE metadata geometry")
            result.ple_context[layer] = tuple(int(value) for value in context.tolist())
            result.ple_lengths[layer] = int(length.item())
        result.validate(
            expected_qsa_layers=expected_qsa_layers,
            expected_ple_layers=expected_ple_layers,
            expected_length=expected_length,
            indexer_dim=indexer_dim,
            ple_context_len=ple_context_len,
            ple_state_len=ple_state_len,
            ple_width=ple_width,
        )
        return result
