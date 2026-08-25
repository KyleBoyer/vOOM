"""Isolated DFlash 2 MLX architecture primitives (no serving adapter).

Adapted from ``z-lab/dflash`` ``dflash/model_mlx.py`` at pinned revision
``07ebd93db9f472af339b644bb70221ad8428328a``.  The upstream implementation
is Copyright (c) 2026 Z Lab and licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Only the DFlash2-specific math primitives live here: grouped dynamic causal
convolution, parent-conditioned candidate selection, and draft-only residual
projection.  Target verification, recurrent rollback, server registration,
and checkpoint loading stay in their dedicated runtime modules.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


_FUSED_GROUPED_DYNAMIC_CONV_SOURCE = r"""
    uint h = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint H = hidden_shape[2];
    uint L = hidden_shape[1];
    uint K = base_shape[0];
    uint G = dynamic_shape[3];
    if (h >= H || t >= L || b >= hidden_shape[0]) return;

    uint group = h / (H / G);
    float value = 0.0f;
    for (uint offset = 0; offset < K; ++offset) {
        if (offset > t) break;
        size_t source_index = ((size_t)b * L + (t - offset)) * H + h;
        size_t base_index = (size_t)offset * H + h;
        size_t dynamic_index =
            (((size_t)b * L + t) * K + offset) * G + group;
        value += static_cast<float>(hidden[source_index])
            * (static_cast<float>(base[base_index])
               + static_cast<float>(dynamic[dynamic_index]));
    }
    output[((size_t)b * L + t) * H + h] = static_cast<T>(value);
"""


_fused_grouped_dynamic_conv_kernel = mx.fast.metal_kernel(
    name="voom_dflash2_grouped_dynamic_conv",
    input_names=["hidden", "dynamic", "base"],
    output_names=["output"],
    source=_FUSED_GROUPED_DYNAMIC_CONV_SOURCE,
)


def _validate_grouped_dynamic_convolution(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> tuple[int, int, int, int]:
    """Validate the shared reference/fused convolution contract."""
    if hidden.ndim != 3:
        raise ValueError("DFlash2 convolution hidden must be rank 3")
    if dynamic.ndim != 4 or base.ndim != 2:
        raise ValueError("DFlash2 convolution kernel ranks are invalid")
    if isinstance(group_size, bool) or not isinstance(group_size, int) \
            or group_size <= 0:
        raise ValueError("DFlash2 convolution group_size must be positive")
    batch, length, hidden_size = map(int, hidden.shape)
    if hidden_size % group_size:
        raise ValueError("DFlash2 hidden width is not divisible by group_size")
    groups = hidden_size // group_size
    kernel_size = int(base.shape[0])
    if tuple(base.shape) != (kernel_size, hidden_size):
        raise ValueError("DFlash2 base kernel shape does not match hidden width")
    if tuple(dynamic.shape) != (batch, length, kernel_size, groups):
        raise ValueError("DFlash2 dynamic kernel shape mismatch")
    return batch, length, hidden_size, kernel_size


def fused_grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    """One-dispatch Metal implementation of DFlash2's causal convolution.

    This is deliberately draft-only.  It preserves the formula but uses a
    serial FP32 accumulation rather than MLX's elementwise graph reduction,
    so its proposals remain subject to the exact target verifier and the path
    stays explicit/default-off until a real acceptance and wall-time gate.
    """
    batch, length, hidden_size, _kernel_size = (
        _validate_grouped_dynamic_convolution(
            hidden, dynamic, base, group_size))
    if not mx.metal.is_available():
        raise RuntimeError("DFlash2 fused convolution requires Metal")
    return _fused_grouped_dynamic_conv_kernel(
        inputs=[hidden, dynamic, base],
        template=[("T", hidden.dtype)],
        grid=(hidden_size, length, batch),
        threadgroup=(min(hidden_size, 256), 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
    )[0]


def project_out_direction(
    hidden: mx.array,
    direction: mx.array,
    strength: float = 1.0,
) -> mx.array:
    """Remove a measured residual-stream direction from draft activations."""
    if hidden.ndim < 2:
        raise ValueError("DFlash2 ablation hidden tensor must be rank >=2")
    if direction.ndim != 1 or direction.shape[0] != hidden.shape[-1]:
        raise ValueError("DFlash2 ablation direction width mismatch")
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) \
            or not 0.0 <= float(strength) <= 2.0:
        raise ValueError("DFlash2 ablation strength must be in [0, 2]")
    if not float(strength):
        return hidden
    dtype = hidden.dtype
    hidden32 = hidden.astype(mx.float32)
    direction32 = direction.astype(mx.float32)
    component = mx.sum(hidden32 * direction32, axis=-1, keepdims=True)
    return (
        hidden32 - float(strength) * component * direction32
    ).astype(dtype)


def grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
    *,
    fused: bool = False,
) -> mx.array:
    """Apply DFlash2's causal depthwise/group-dynamic convolution.

    ``hidden`` is ``[batch, length, hidden]``, ``dynamic`` is
    ``[batch, length, kernel, groups]``, and ``base`` is
    ``[kernel, hidden]``.  Offset ``d`` sees only position ``t-d``.
    """
    batch, length, hidden_size, kernel_size = (
        _validate_grouped_dynamic_convolution(
            hidden, dynamic, base, group_size))
    if fused:
        return fused_grouped_dynamic_convolve(
            hidden, dynamic, base, group_size)
    groups = hidden_size // group_size

    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, kernel_size, groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(kernel_size):
        values = blocks if offset == 0 else mx.concatenate(
            (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]), axis=1)
        kernel = base[offset].reshape(
            1, 1, groups, group_size).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    """Two-sided DFlash2 wrapper around one generated grouped kernel."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        group_size: int,
        *,
        fused: bool = False,
    ):
        super().__init__()
        if hidden_size <= 0 or kernel_size <= 0 or group_size <= 0:
            raise ValueError("DFlash2 convolution dimensions must be positive")
        if hidden_size % group_size:
            raise ValueError("DFlash2 hidden_size must divide into conv groups")
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.fused = bool(fused)
        groups = hidden_size // group_size
        # The published checkpoint owns this parameter.  Zero initialization
        # is only a construction default for synthetic tests/load-before-use.
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups)
        prepared = grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
            fused=self.fused,
        )
        return prepared, dynamic[..., 1, :, :]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return grouped_dynamic_convolve(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.group_size,
            fused=self.fused,
        )


def _sampling_probabilities(scores: mx.array, temperature: float) -> mx.array:
    if temperature <= 0:
        raise ValueError("DFlash2 stochastic selector temperature must be positive")
    return mx.softmax(scores.astype(mx.float32) / temperature, axis=-1)


class CandidateSelector(nn.Module):
    """Select one parent-consistent chain from each slot's top-k candidates.

    The returned stochastic ``q`` rows are over candidate indices, accompanied
    by their vocabulary IDs.  Any future verifier must use both when computing
    exact target-minus-draft rejection correction.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        selector_rank: int,
        selector_top_k: int,
    ):
        super().__init__()
        if min(hidden_size, vocab_size, selector_rank, selector_top_k) <= 0:
            raise ValueError("DFlash2 selector dimensions must be positive")
        if selector_top_k > vocab_size:
            raise ValueError("DFlash2 selector top-k exceeds vocabulary size")
        self.top_k = selector_top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, selector_rank)
        self.successor_codebook = nn.Embedding(vocab_size, selector_rank)
        self.hidden_projection = nn.Linear(
            hidden_size, selector_rank, bias=False)

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        temperature: float = 0.0,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        if hidden.ndim != 3 or logits.ndim != 3:
            raise ValueError("DFlash2 selector hidden/logits must be rank 3")
        if hidden.shape[:2] != logits.shape[:2]:
            raise ValueError("DFlash2 selector hidden/logit positions differ")
        if anchor_ids.ndim != 1 or anchor_ids.shape[0] != hidden.shape[0]:
            raise ValueError("DFlash2 selector needs one anchor per batch row")
        if temperature < 0:
            raise ValueError("DFlash2 selector temperature must be non-negative")

        candidates = mx.argpartition(
            logits, -self.top_k, axis=-1)[..., -self.top_k:]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path: list[mx.array] = []
        q_rows: list[mx.array] = []
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * projected[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            if temperature > 0:
                q = _sampling_probabilities(scores, temperature)
                selected = mx.random.categorical(mx.log(q))
                q_rows.append(q)
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1)[:, 0]
            path.append(predecessor)
        return (
            mx.stack(path, axis=1),
            candidates,
            mx.stack(q_rows, axis=1) if q_rows else None,
        )
