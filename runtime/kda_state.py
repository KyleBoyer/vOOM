"""Kimi Delta Attention (KDA) per-layer recurrent state.

Structurally unrelated to the token-indexed KVCache family in kv_cache.py /
kv_paged.py: a KDA layer's state is a fixed-size (num_heads, head_dim,
head_dim) matrix plus a tiny (kernel_size - 1)-token causal-conv history,
both O(1) in context length -- there is nothing to page or spill regardless
of how long the sequence gets. See docs/future_lossless_techniques.md F92.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import mlx.core as mx
import numpy as np

from .uncached_io import set_darwin_nocache


_NATIVE_KDA_FACTOR_STEP_SOURCE = """
    uint dv = thread_position_in_grid.x;
    uint h  = thread_position_in_grid.y;
    uint b  = thread_position_in_grid.z;
    uint Dk = state_shape[2];
    uint Dv = state_shape[3];
    uint H  = state_shape[1];
    if (dv >= Dv) return;

    uint vector_base = (b * H + h) * Dk;
    uint state_base = vector_base * Dv + dv;
    float predicted = 0.0f;
    for (uint dk = 0; dk < Dk; dk++) {
        float decayed = float(state[state_base + dk * Dv])
            * exp(float(gate[vector_base + dk]));
        out_state[state_base + dk * Dv] = T(decayed);
        predicted += float(key[vector_base + dk]) * decayed;
    }

    float residual = float(value[(b * H + h) * Dv + dv]) - predicted;
    float scaled_residual = float(beta[b * H + h]) * residual;
    for (uint dk = 0; dk < Dk; dk++) {
        uint index = state_base + dk * Dv;
        float updated = float(out_state[index])
            + float(key[vector_base + dk]) * scaled_residual;
        out_state[index] = T(updated);
    }
"""

_native_kda_factor_step_kernel = mx.fast.metal_kernel(
    name="kimi_kda_factor_commit_step",
    input_names=["gate", "key", "value", "beta", "state"],
    output_names=["out_state"],
    source=_NATIVE_KDA_FACTOR_STEP_SOURCE,
)


def _native_fused_kda_factor_step(
    gate: mx.array,
    key: mx.array,
    value: mx.array,
    beta: mx.array,
    state: mx.array,
) -> mx.array:
    """Commit one captured KDA update in one state-resident Metal dispatch.

    K3's decay is per key channel, unlike Qwen/Jet DeltaNet's scalar-per-head
    decay, so their existing fused step kernel cannot be reused.  This kernel
    assigns one thread to each value-channel column: the complete recurrent
    matrix column stays within that thread through decay, prediction, and
    rank-one correction.  It never reloads target weights or materializes the
    intermediate decayed matrix/prediction/residual arrays used by the plain
    MLX expression.

    The fused arithmetic can differ from MLX's reduction order by ordinary
    float32 roundoff.  Consequently callers must opt in until a released-model
    greedy token-identity gate has admitted it for a serving profile.
    """
    batch, heads, key_dim, value_dim = state.shape
    if gate.shape != (batch, heads, key_dim):
        raise ValueError(
            f"KDA gate shape {gate.shape} != {(batch, heads, key_dim)}")
    if key.shape != gate.shape:
        raise ValueError(f"KDA key shape {key.shape} != gate {gate.shape}")
    if value.shape != (batch, heads, value_dim):
        raise ValueError(
            f"KDA value shape {value.shape} != "
            f"{(batch, heads, value_dim)}")
    if beta.shape != (batch, heads):
        raise ValueError(
            f"KDA beta shape {beta.shape} != {(batch, heads)}")
    return _native_kda_factor_step_kernel(
        inputs=[gate, key, value, beta, state],
        template=[("T", state.dtype)],
        grid=(value_dim, heads, batch),
        threadgroup=(min(value_dim, 256), 1, 1),
        output_shapes=[state.shape],
        output_dtypes=[state.dtype],
    )[0]


@dataclass(frozen=True)
class KDAFactorStep:
    """Compact sufficient statistics for one released KDA state update."""

    gate: mx.array
    key: mx.array
    value: mx.array
    beta: mx.array
    conv_history: tuple

    def nbytes(self) -> int:
        return (
            self.gate.nbytes
            + self.key.nbytes
            + self.value.nbytes
            + self.beta.nbytes
            + sum(value.nbytes for value in self.conv_history
                  if value is not None)
        )


class KDAFactorWindow:
    """Per-layer factors captured during a speculative verify window.

    Replaying accepted factors touches no target weights and allocates only one
    final recurrent matrix per KDA layer.  This is the MLX counterpart of
    SpecLA's factor buffering: rollback storage scales with the low-rank update
    factors, not ``positions * heads * D * D`` dense endpoints.
    """

    def __init__(self, steps: list[list[KDAFactorStep]], positions: int):
        self.steps = steps
        self.positions = int(positions)

    def nbytes(self) -> int:
        return sum(step.nbytes() for layer in self.steps for step in layer)

    def commit_prefix(
        self,
        base: "KDAStateCache",
        positions: int,
        *,
        native_fused: bool = False,
    ) -> "KDAStateCache":
        count = int(positions)
        if not 0 <= count <= self.positions:
            raise ValueError(
                f"KDA factor prefix {count} is outside [0, {self.positions}]")
        result = base.fork()
        for layer, steps in enumerate(self.steps):
            if not steps or count == 0:
                continue
            if len(steps) < count:
                raise ValueError(
                    f"KDA layer {layer} captured {len(steps)} factors, "
                    f"needs {count}")
            state = result.state(layer)
            if state is None:
                key = steps[0].key
                state = mx.zeros(
                    (
                        key.shape[0],
                        key.shape[1],
                        key.shape[2],
                        steps[0].value.shape[2],
                    ),
                    dtype=mx.float32,
                )
            for step in steps[:count]:
                if native_fused:
                    state = _native_fused_kda_factor_step(
                        step.gate,
                        step.key,
                        step.value,
                        step.beta,
                        state,
                    )
                else:
                    state = state * mx.exp(step.gate)[..., None]
                    pred_v = mx.sum(
                        step.key[..., None] * state, axis=-2)
                    residual = step.value - pred_v
                    state = state + (
                        step.beta[..., None] * step.key
                    )[..., None] * residual[..., None, :]
            mx.eval(state)
            result.set_state(layer, state)
            result.set_conv_history(
                layer, tuple(steps[count - 1].conv_history))
        return result


class KDAStateCache:
    """Holds one recurrent state + conv history per KDA layer."""

    def __init__(self, num_layers: int):
        self._state: list[mx.array | None] = [None] * num_layers
        self._conv: list[tuple | None] = [None] * num_layers
        self._factor_capture: list[list[KDAFactorStep]] | None = None
        self._spill_root: Path | None = None
        self._spill_temporary = None
        self._spill_meta: dict[int, dict[str, tuple]] = {}
        self.spill_bytes_written = 0
        self.spill_bytes_read = 0
        self.spill_layers = 0
        self.spill_reloads = 0
        self.spill_uncached_descriptors = 0

    def enable_disk_spill(self, root: str | Path) -> None:
        """Enable exact lazy FP32/BF16 recurrent-state spill by layer."""
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._spill_root = root
        self._spill_temporary = tempfile.TemporaryDirectory(
            prefix="voom-k3-kda-", dir=root)

    @property
    def spill_enabled(self) -> bool:
        return self._spill_temporary is not None

    def _spill_directory(self) -> Path:
        if self._spill_temporary is None:
            raise RuntimeError("KDA state spill is not enabled")
        return Path(self._spill_temporary.name)

    @staticmethod
    def _array_payload(value: mx.array) -> tuple[bytes, str]:
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            return np.asarray(value.view(mx.uint16)).tobytes(), "bf16"
        if value.dtype == mx.float32:
            return np.asarray(value).astype(
                np.float32, copy=False).tobytes(), "f32"
        if value.dtype == mx.float16:
            return np.asarray(value).astype(
                np.float16, copy=False).tobytes(), "f16"
        raise TypeError(f"unsupported KDA spill dtype {value.dtype}")

    @staticmethod
    def _array_from_payload(
        payload: bytes, shape: tuple[int, ...], dtype: str,
    ) -> mx.array:
        if dtype == "bf16":
            host = np.frombuffer(payload, dtype=np.uint16).reshape(shape)
            return mx.array(host).view(mx.bfloat16)
        if dtype == "f32":
            host = np.frombuffer(payload, dtype=np.float32).reshape(shape)
            return mx.array(host)
        if dtype == "f16":
            host = np.frombuffer(payload, dtype=np.float16).reshape(shape)
            return mx.array(host)
        raise TypeError(f"unsupported KDA spill dtype token {dtype}")

    def spill_layer(self, layer: int) -> bool:
        """Persist one completed layer and release its Metal endpoints."""
        if not self.spill_enabled or self._state[layer] is None:
            return False
        arrays = {"state": self._state[layer]}
        history = self._conv[layer]
        if history is not None:
            arrays.update({
                f"conv_{index}": value
                for index, value in enumerate(history)
                if value is not None
            })
        metadata: dict[str, tuple] = {}
        for name, value in arrays.items():
            payload, dtype = self._array_payload(value)
            path = self._spill_directory() / f"layer-{layer:03d}-{name}.bin"
            with path.open("wb", buffering=0) as output:
                self.spill_uncached_descriptors += int(
                    set_darwin_nocache(output.fileno()))
                output.write(payload)
            metadata[name] = (path, tuple(map(int, value.shape)), dtype)
            self.spill_bytes_written += len(payload)
        self._spill_meta[layer] = metadata
        self._state[layer] = None
        self._conv[layer] = None
        self.spill_layers += 1
        return True

    def _reload_layer(self, layer: int) -> None:
        metadata = self._spill_meta.get(layer)
        if not metadata:
            return
        loaded = {}
        for name, (path, shape, dtype) in metadata.items():
            with path.open("rb", buffering=0) as source:
                self.spill_uncached_descriptors += int(
                    set_darwin_nocache(source.fileno()))
                payload = source.read()
            loaded[name] = self._array_from_payload(payload, shape, dtype)
            self.spill_bytes_read += len(payload)
        state = loaded.get("state")
        if state is not None:
            self._state[layer] = state
        conv = [
            loaded[name]
            for name in sorted(
                (name for name in loaded if name.startswith("conv_")),
                key=lambda name: int(name.split("_")[1]),
            )
        ]
        if conv:
            self._conv[layer] = tuple(conv)
        arrays = [value for value in loaded.values()]
        if arrays:
            mx.eval(*arrays)
        self.spill_reloads += 1

    def spill_stats(self) -> dict[str, int]:
        return {
            "layers": self.spill_layers,
            "bytes_written": self.spill_bytes_written,
            "bytes_read": self.spill_bytes_read,
            "reloads": self.spill_reloads,
            "resident_bytes": self.nbytes(),
            "uncached_descriptors": self.spill_uncached_descriptors,
        }

    def state(self, layer: int) -> mx.array | None:
        if self._state[layer] is None and layer in self._spill_meta:
            self._reload_layer(layer)
        return self._state[layer]

    def set_state(self, layer: int, state: mx.array) -> None:
        self._state[layer] = state

    def conv_history(self, layer: int) -> tuple | None:
        if self._conv[layer] is None and layer in self._spill_meta:
            self._reload_layer(layer)
        return self._conv[layer]

    def set_conv_history(self, layer: int, history: tuple) -> None:
        self._conv[layer] = history

    def reset(self) -> None:
        for i in range(len(self._state)):
            self._state[i] = None
            self._conv[i] = None
        self._factor_capture = None
        self._spill_meta.clear()
        if self._spill_temporary is not None:
            root = self._spill_root
            self._spill_temporary.cleanup()
            self._spill_temporary = tempfile.TemporaryDirectory(
                prefix="voom-k3-kda-", dir=root)

    def begin_factor_capture(self) -> None:
        if self._factor_capture is not None:
            raise RuntimeError("KDA factor capture is already active")
        self._factor_capture = [[] for _ in self._state]

    @property
    def factor_capture_active(self) -> bool:
        return self._factor_capture is not None

    def capture_factor_step(
        self,
        layer: int,
        *,
        gate: mx.array,
        key: mx.array,
        value: mx.array,
        beta: mx.array,
        conv_history: tuple,
    ) -> None:
        if self._factor_capture is None:
            return
        arrays = [gate, key, value, beta]
        arrays.extend(item for item in conv_history if item is not None)
        mx.eval(*arrays)
        self._factor_capture[layer].append(KDAFactorStep(
            gate=gate,
            key=key,
            value=value,
            beta=beta,
            conv_history=tuple(conv_history),
        ))

    def finish_factor_capture(self, positions: int) -> KDAFactorWindow | None:
        capture = self._factor_capture
        self._factor_capture = None
        if capture is None:
            return None
        count = int(positions)
        for layer, steps in enumerate(capture):
            if steps and len(steps) != count:
                raise RuntimeError(
                    f"KDA layer {layer} captured {len(steps)} factor steps "
                    f"for a {count}-position window")
        return KDAFactorWindow(capture, count)

    def cancel_factor_capture(self) -> None:
        self._factor_capture = None

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
        # Layer-stationary K3 prefill deliberately spills completed recurrent
        # layers and clears their resident owners.  A durable endpoint export
        # must not interpret that absence as "this checkpoint has no KDA".
        # Reload each immutable spill payload exactly once before constructing
        # the safetensors mapping, mirroring compressed MLA persistence.
        for layer in sorted(self._spill_meta):
            if self._state[layer] is None:
                self._reload_layer(layer)
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
