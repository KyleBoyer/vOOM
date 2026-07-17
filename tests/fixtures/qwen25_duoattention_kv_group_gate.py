#!/usr/bin/env python3
"""Standalone DuoAttention feasibility gate for dense Qwen2.5-1.5B.

This fixture deliberately does not modify the serving runtime.  It profiles
actual layer/KV-group far-context attention on a calibration-only prompt set,
checks several retrieval-group budgets against exact dense continuation NLL,
then evaluates one frozen pattern on disjoint 8K/16K needle/tool prompts.

The approximate cache is fixture-local, cannot be serialized, rejects unsafe
rollback after an eviction, and is never inserted into the engine's prompt/PIC
state.  Run model work only through the repository-wide lock, for example:

  lock=/tmp/voom-mlx-benchmark.lock
  until mkdir "$lock" 2>/dev/null; do sleep 1; done
  echo $$ > "$lock/owner"
  trap 'rm -f "$lock/owner"; rmdir "$lock"' EXIT INT TERM
  ~/.hf-pull/bin/python tests/fixtures/qwen25_duoattention_kv_group_gate.py \
    --profile-only

After copying the reported suggested pattern into ``FROZEN_FULL_GROUPS``, run
without ``--profile-only`` for the held-out gate.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import layer_runner  # noqa: E402
from runtime.engine import RuntimeConfig, StreamingEngine  # noqa: E402
from runtime.kv_cache import KVCache, SteppedKVCache  # noqa: E402
from runtime.toolcalls import tools_preamble  # noqa: E402


# Frozen by the 2026-07-16 locked calibration run.  Each tuple is the one full
# retrieval KV group for that layer; the other of Qwen2.5-1.5B's two groups is
# sink+recent.  This measured pattern failed the calibration quality threshold,
# but remains frozen so a small held-out run can decisively confirm the no-go.
FROZEN_FULL_GROUPS: tuple[tuple[int, ...], ...] = (
    (0,), (1,), (0,), (1,), (0,), (0,), (1,),
    (0,), (0,), (1,), (1,), (1,), (0,), (0,),
    (0,), (0,), (0,), (1,), (0,), (0,), (0,),
    (1,), (1,), (0,), (1,), (1,), (0,), (0,),
)
FROZEN_MODEL_GEOMETRY = (28, 12, 2, 128)

CALIBRATION_SPECS = (
    ("cal-needle-early", 1536, 0.18, "needle", "CALIBER-ORCHID-17"),
    ("cal-tool-middle", 2560, 0.52, "tool", "calibration_probe_11"),
    ("cal-needle-late", 3584, 0.81, "needle", "CALIBER-CEDAR-29"),
)

HELDOUT_SPECS = (
    ("heldout-8k-needle-early", 8192, 0.13, "needle", "HELIOS-8A-731"),
    ("heldout-8k-tool-late", 8192, 0.79, "tool", "archive_probe_23"),
    ("heldout-16k-needle-middle", 16384, 0.43, "needle", "TUNDRA-16M-947"),
    ("heldout-16k-tool-late", 16384, 0.87, "tool", "archive_probe_41"),
)


def validate_full_groups(
    full_by_layer: Sequence[Sequence[int]], num_layers: int, num_kv_heads: int,
) -> tuple[tuple[int, ...], ...]:
    """Canonicalize a layer-specific retrieval-group pattern or fail closed."""
    if len(full_by_layer) != num_layers:
        raise ValueError(
            f"retrieval pattern has {len(full_by_layer)} layers, expected {num_layers}")
    result = []
    for layer, groups in enumerate(full_by_layer):
        canonical = tuple(sorted(int(group) for group in groups))
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"layer {layer} repeats a KV group")
        if any(group < 0 or group >= num_kv_heads for group in canonical):
            raise ValueError(f"layer {layer} contains an out-of-range KV group")
        result.append(canonical)
    if all(len(groups) == num_kv_heads for groups in result):
        raise ValueError("all-dense pattern is not a DuoAttention candidate")
    return tuple(result)


def query_heads_for_groups(groups: Sequence[int], repeats: int) -> tuple[int, ...]:
    if repeats <= 0:
        raise ValueError("GQA repeats must be positive")
    return tuple(
        head for group in groups
        for head in range(int(group) * repeats, (int(group) + 1) * repeats)
    )


def retained_stream_positions(
    existing: Sequence[int], start: int, width: int, sink: int, recent: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (attention-view positions, persisted sink+recent positions)."""
    if min(start, width, sink, recent) < 0 or recent == 0:
        raise ValueError("invalid streaming-cache geometry")
    view = tuple(int(position) for position in existing) + tuple(
        range(start, start + width))
    if len(set(view)) != len(view) or tuple(sorted(view)) != view:
        raise ValueError("streaming positions must be unique and chronological")
    end = start + width
    kept = tuple(
        position for position in view
        if position < sink or position >= max(sink, end - recent)
    )
    return view, kept


def logical_kv_bytes(
    full_by_layer: Sequence[Sequence[int]], length: int, num_kv_heads: int,
    head_dim: int, sink: int, recent: int, bytes_per_element: int = 2,
) -> int:
    """Pure reference byte count for two K/V tensors per layer."""
    total_elements = 0
    stream_length = min(length, sink + recent)
    for full in full_by_layer:
        full_count = len(tuple(full))
        total_elements += (
            full_count * length + (num_kv_heads - full_count) * stream_length)
    return total_elements * head_dim * 2 * bytes_per_element


@dataclass(frozen=True)
class AttentionView:
    full_keys: mx.array | None
    full_values: mx.array | None
    stream_keys: mx.array | None
    stream_values: mx.array | None
    stream_positions: tuple[int, ...]
    full_groups: tuple[int, ...]
    stream_groups: tuple[int, ...]
    end: int


@dataclass
class ComponentMetrics:
    synchronize: bool = False
    projection_s: float = 0.0
    update_gather_s: float = 0.0
    sdpa_s: float = 0.0
    projection_calls: int = 0
    update_calls: int = 0
    sdpa_calls: int = 0

    def as_dict(self) -> dict:
        return {
            "synchronized": self.synchronize,
            "projection_s": self.projection_s,
            "update_gather_s": self.update_gather_s,
            "sdpa_s": self.sdpa_s,
            "projection_calls": self.projection_calls,
            "update_calls": self.update_calls,
            "sdpa_calls": self.sdpa_calls,
        }


class DuoKVCache(KVCache):
    """Fixture-local, stepped, non-serializable DuoAttention KV cache."""

    approximate = True
    serializable = False
    step = 256

    def __init__(
        self, full_by_layer: Sequence[Sequence[int]], num_kv_heads: int,
        sink: int = 64, recent: int = 256,
        metrics: ComponentMetrics | None = None,
    ):
        pattern = validate_full_groups(
            full_by_layer, len(full_by_layer), num_kv_heads)
        super().__init__(len(pattern))
        self.full_by_layer = pattern
        self.num_kv_heads = int(num_kv_heads)
        self.sink = int(sink)
        self.recent = int(recent)
        if self.sink < 0 or self.recent <= 0:
            raise ValueError("sink must be nonnegative and recent must be positive")
        self.metrics = metrics or ComponentMetrics()
        self.full_keys: list[mx.array | None] = [None] * len(pattern)
        self.full_values: list[mx.array | None] = [None] * len(pattern)
        self.stream_keys: list[mx.array | None] = [None] * len(pattern)
        self.stream_values: list[mx.array | None] = [None] * len(pattern)
        self.stream_positions: list[tuple[int, ...]] = [()] * len(pattern)
        self._lengths = [0] * len(pattern)
        self._evicted = [False] * len(pattern)

    @property
    def offset(self) -> int:
        return self._lengths[0] if self._lengths else 0

    def _take_groups(self, value: mx.array, groups: Sequence[int]) -> mx.array:
        return mx.take(value, mx.array(tuple(groups), dtype=mx.int32), axis=1)

    def _append_full(
        self, layer: int, keys: mx.array, values: mx.array,
        previous: int, end: int,
    ) -> tuple[mx.array, mx.array]:
        current = self.full_keys[layer]
        if current is None or end > current.shape[2]:
            growth = ((end - previous + self.step - 1) // self.step) * self.step
            new_k = mx.zeros(
                (*keys.shape[:2], growth, keys.shape[3]), dtype=keys.dtype)
            new_v = mx.zeros(
                (*values.shape[:2], growth, values.shape[3]), dtype=values.dtype)
            if current is not None:
                new_k = mx.concatenate([current[..., :previous, :], new_k], axis=2)
                new_v = mx.concatenate([
                    self.full_values[layer][..., :previous, :], new_v], axis=2)
            self.full_keys[layer] = new_k
            self.full_values[layer] = new_v
        self.full_keys[layer][..., previous:end, :] = keys
        self.full_values[layer][..., previous:end, :] = values
        return (
            self.full_keys[layer][..., :end, :],
            self.full_values[layer][..., :end, :],
        )

    def update_duo(self, layer: int, keys: mx.array, values: mx.array) -> AttentionView:
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Duo cache requires equal rank-4 K/V")
        if keys.shape[1] != self.num_kv_heads:
            raise ValueError("Duo cache KV-group geometry changed")
        previous = self._lengths[layer]
        width = int(keys.shape[2])
        end = previous + width
        full_groups = self.full_by_layer[layer]
        stream_groups = tuple(
            group for group in range(self.num_kv_heads)
            if group not in full_groups)

        full_k = full_v = None
        if full_groups:
            full_k, full_v = self._append_full(
                layer,
                self._take_groups(keys, full_groups),
                self._take_groups(values, full_groups),
                previous,
                end,
            )

        stream_view_k = stream_view_v = None
        view_positions: tuple[int, ...] = ()
        if stream_groups:
            new_k = self._take_groups(keys, stream_groups)
            new_v = self._take_groups(values, stream_groups)
            prior_k = self.stream_keys[layer]
            prior_v = self.stream_values[layer]
            stream_view_k = (
                new_k if prior_k is None else mx.concatenate([prior_k, new_k], axis=2))
            stream_view_v = (
                new_v if prior_v is None else mx.concatenate([prior_v, new_v], axis=2))
            view_positions, kept_positions = retained_stream_positions(
                self.stream_positions[layer], previous, width,
                self.sink, self.recent)
            kept_set = set(kept_positions)
            keep_indices = tuple(
                index for index, position in enumerate(view_positions)
                if position in kept_set)
            keep = mx.array(keep_indices, dtype=mx.int32)
            self.stream_keys[layer] = mx.take(stream_view_k, keep, axis=2)
            self.stream_values[layer] = mx.take(stream_view_v, keep, axis=2)
            self.stream_positions[layer] = kept_positions
            self._evicted[layer] |= len(kept_positions) < len(view_positions)

        self._lengths[layer] = end
        return AttentionView(
            full_k, full_v, stream_view_k, stream_view_v, view_positions,
            full_groups, stream_groups, end)

    def clone_for_branch(self) -> "DuoKVCache":
        clone = DuoKVCache(
            self.full_by_layer, self.num_kv_heads,
            self.sink, self.recent, self.metrics)
        clone.full_keys = [
            None if value is None else value[..., :self._lengths[layer], :]
            for layer, value in enumerate(self.full_keys)]
        clone.full_values = [
            None if value is None else value[..., :self._lengths[layer], :]
            for layer, value in enumerate(self.full_values)]
        clone.stream_keys = list(self.stream_keys)
        clone.stream_values = list(self.stream_values)
        clone.stream_positions = list(self.stream_positions)
        clone._lengths = list(self._lengths)
        clone._evicted = list(self._evicted)
        return clone

    def nbytes(self) -> int:
        total = 0
        for layer, length in enumerate(self._lengths):
            full_k = self.full_keys[layer]
            if full_k is not None:
                total += full_k[..., :length, :].nbytes
                total += self.full_values[layer][..., :length, :].nbytes
            stream_k = self.stream_keys[layer]
            if stream_k is not None:
                total += stream_k.nbytes + self.stream_values[layer].nbytes
        return total

    def allocated_nbytes(self) -> int:
        return sum(
            value.nbytes for arrays in (
                self.full_keys, self.full_values,
                self.stream_keys, self.stream_values)
            for value in arrays if value is not None)

    def trim(self, length: int):
        if not 0 <= length <= self.offset:
            raise ValueError("Duo cache trim is outside the committed range")
        if length == self.offset:
            return
        if any(self._evicted):
            raise RuntimeError(
                "Duo cache rollback is unsafe after streaming eviction")
        for layer in range(len(self._lengths)):
            if self.full_keys[layer] is not None:
                self.full_keys[layer] = self.full_keys[layer][..., :length, :]
                self.full_values[layer] = self.full_values[layer][..., :length, :]
            if self.stream_keys[layer] is not None:
                keep_count = sum(
                    position < length for position in self.stream_positions[layer])
                self.stream_keys[layer] = self.stream_keys[layer][..., :keep_count, :]
                self.stream_values[layer] = self.stream_values[layer][..., :keep_count, :]
                self.stream_positions[layer] = tuple(
                    position for position in self.stream_positions[layer]
                    if position < length)
            self._lengths[layer] = min(self._lengths[layer], length)


@dataclass
class RetrievalProfiler:
    num_layers: int
    num_kv_heads: int
    totals: list[list[float]] = field(init=False)
    counts: list[list[int]] = field(init=False)

    def __post_init__(self):
        self.totals = [
            [0.0] * self.num_kv_heads for _ in range(self.num_layers)]
        self.counts = [
            [0] * self.num_kv_heads for _ in range(self.num_layers)]

    def observe(self, layer: int, mass: Sequence[float]):
        if len(mass) != self.num_kv_heads:
            raise ValueError("profiled KV-group count changed")
        for group, value in enumerate(mass):
            self.totals[layer][group] += float(value)
            self.counts[layer][group] += 1

    def scores(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(
            self.totals[layer][group] / self.counts[layer][group]
            if self.counts[layer][group] else 0.0
            for group in range(self.num_kv_heads))
            for layer in range(self.num_layers))


class ProfileKVCache(SteppedKVCache):
    def __init__(self, num_layers: int, profiler: RetrievalProfiler):
        super().__init__(num_layers)
        self.profiler = profiler


_ORIGINAL_ATTENTION = layer_runner._attention


def _project_qkv(h, w, prefix, cfg, offset, rope_freqs, rope_mscale):
    batch, length, _ = h.shape
    n_q, n_kv, head_dim = (
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    q = layer_runner._linear(h, w, f"{prefix}.self_attn.q_proj")
    k = layer_runner._linear(h, w, f"{prefix}.self_attn.k_proj")
    v = layer_runner._linear(h, w, f"{prefix}.self_attn.v_proj")
    q_norm = w.get(f"{prefix}.self_attn.q_norm.weight")
    k_norm = w.get(f"{prefix}.self_attn.k_norm.weight")
    per_head_norm = q_norm is not None and q_norm.shape[0] == head_dim
    if q_norm is not None and not per_head_norm:
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
    q = q.reshape(batch, length, n_q, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, length, n_kv, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, length, n_kv, head_dim).transpose(0, 2, 1, 3)
    if per_head_norm:
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
    if rope_mscale != 1.0:
        q = q * rope_mscale
        k = k * rope_mscale
    if rope_freqs is None:
        q = mx.fast.rope(
            q, head_dim, traditional=False, base=cfg.rope_theta,
            scale=1.0, offset=offset)
        k = mx.fast.rope(
            k, head_dim, traditional=False, base=cfg.rope_theta,
            scale=1.0, offset=offset)
    else:
        q = mx.fast.rope(
            q, head_dim, traditional=False, base=None, freqs=rope_freqs,
            scale=1.0, offset=offset)
        k = mx.fast.rope(
            k, head_dim, traditional=False, base=None, freqs=rope_freqs,
            scale=1.0, offset=offset)
    return q, k, v


def _mask(q_positions, key_positions, dtype, *, sink=None, recent=None):
    q_pos = mx.array(tuple(q_positions), dtype=mx.int32)[:, None]
    k_pos = mx.array(tuple(key_positions), dtype=mx.int32)[None, :]
    allowed = k_pos <= q_pos
    if sink is not None:
        allowed = allowed & (
            (k_pos < int(sink)) | (k_pos >= q_pos - int(recent) + 1))
    return mx.where(allowed, 0.0, float("-inf")).astype(dtype)


def _profile_far_mass(q, keys, layer: int, cache: ProfileKVCache, sink: int, recent: int):
    sequence = int(keys.shape[2])
    query_position = sequence - 1
    far_end = query_position - recent + 1
    if far_end <= sink:
        return
    batch, n_q, _, head_dim = q.shape
    n_kv = keys.shape[1]
    repeats = n_q // n_kv
    q_grouped = q[:, :, -1:, :].reshape(
        batch, n_kv, repeats, 1, head_dim).astype(mx.float32)
    k_grouped = mx.expand_dims(keys.astype(mx.float32), axis=2)
    scores = mx.matmul(
        q_grouped * (head_dim ** -0.5), k_grouped.swapaxes(-1, -2))
    probs = mx.softmax(scores, axis=-1, precise=True)
    positions = mx.arange(sequence)
    far = (positions >= sink) & (positions < far_end)
    mass = (probs * far[None, None, None, None, :]).sum(axis=-1)
    mass = mass.mean(axis=(0, 2, 3))
    mx.eval(mass)
    cache.profiler.observe(layer, mass.tolist())


def _q_for_groups(q: mx.array, groups: Sequence[int], repeats: int) -> mx.array:
    pieces = [
        q[:, group * repeats:(group + 1) * repeats]
        for group in groups]
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)


def _attention_dispatch(
    h, w, prefix, cfg, kv, layer, offset,
    rope_freqs=None, rope_mscale=1.0,
):
    if not isinstance(kv, (DuoKVCache, ProfileKVCache)):
        return _ORIGINAL_ATTENTION(
            h, w, prefix, cfg, kv, layer, offset,
            rope_freqs=rope_freqs, rope_mscale=rope_mscale)
    if cfg.model_type != "qwen2":
        raise ValueError("Duo fixture is restricted to dense Qwen2/Qwen2.5")

    length = int(h.shape[1])
    metrics = kv.metrics if isinstance(kv, DuoKVCache) else None
    projection_started = time.perf_counter()
    q, k, v = _project_qkv(
        h, w, prefix, cfg, offset, rope_freqs, rope_mscale)
    if metrics is not None and metrics.synchronize:
        mx.eval(q, k, v)
        mx.synchronize()
        metrics.projection_s += time.perf_counter() - projection_started
        metrics.projection_calls += 1

    if isinstance(kv, ProfileKVCache):
        keys, values = kv.update(layer, k, v)
        _profile_far_mass(q, keys, layer, kv, sink=64, recent=256)
        mask = None
        if length > 1:
            mask = _mask(
                range(offset, offset + length), range(keys.shape[2]), q.dtype)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=cfg.head_dim ** -0.5, mask=mask)
    else:
        if metrics.synchronize:
            mx.synchronize()
        update_started = time.perf_counter()
        view = kv.update_duo(layer, k, v)
        if metrics.synchronize:
            arrays = [value for value in (
                view.full_keys, view.full_values,
                view.stream_keys, view.stream_values) if value is not None]
            mx.eval(*arrays)
            mx.synchronize()
            metrics.update_gather_s += time.perf_counter() - update_started
            metrics.update_calls += 1

        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        q_positions = range(offset, offset + length)
        pieces: dict[int, mx.array] = {}
        if metrics.synchronize:
            mx.synchronize()
        sdpa_started = time.perf_counter()
        sdpa_results = []
        if view.full_groups:
            q_full = _q_for_groups(q, view.full_groups, repeats)
            full_mask = None if length == 1 else _mask(
                q_positions, range(view.end), q.dtype)
            full_out = mx.fast.scaled_dot_product_attention(
                q_full, view.full_keys, view.full_values,
                scale=cfg.head_dim ** -0.5, mask=full_mask)
            sdpa_results.append(full_out)
            for index, group in enumerate(view.full_groups):
                pieces[group] = full_out[
                    :, index * repeats:(index + 1) * repeats]
        if view.stream_groups:
            q_stream = _q_for_groups(q, view.stream_groups, repeats)
            stream_mask = None if length == 1 else _mask(
                q_positions, view.stream_positions, q.dtype,
                sink=kv.sink, recent=kv.recent)
            stream_out = mx.fast.scaled_dot_product_attention(
                q_stream, view.stream_keys, view.stream_values,
                scale=cfg.head_dim ** -0.5, mask=stream_mask)
            sdpa_results.append(stream_out)
            for index, group in enumerate(view.stream_groups):
                pieces[group] = stream_out[
                    :, index * repeats:(index + 1) * repeats]
        attended = mx.concatenate(
            [pieces[group] for group in range(cfg.num_key_value_heads)], axis=1)
        if metrics.synchronize:
            mx.eval(*sdpa_results, attended)
            mx.synchronize()
            metrics.sdpa_s += time.perf_counter() - sdpa_started
            metrics.sdpa_calls += len(sdpa_results)

    batch = h.shape[0]
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch, length, cfg.num_attention_heads * cfg.head_dim)
    return layer_runner._linear(attended, w, f"{prefix}.self_attn.o_proj")


@contextlib.contextmanager
def patched_attention():
    if layer_runner._attention is not _ORIGINAL_ATTENTION:
        raise RuntimeError("another attention patch is already active")
    layer_runner._attention = _attention_dispatch
    try:
        yield
    finally:
        layer_runner._attention = _ORIGINAL_ATTENTION


@dataclass(frozen=True)
class GateCase:
    label: str
    target_tokens: int
    needle_ratio: float
    kind: str
    prompt_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]
    expected_ids: tuple[str, ...]


def _encode(engine: StreamingEngine, text: str) -> list[int]:
    return list(engine.tokenizer.encode(text).ids)


def _tools() -> list[dict]:
    result = []
    # Keep the tool prompt representative but small enough for the disjoint
    # 2.5K calibration case.  Every target used by this fixture is present.
    for index in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 41, 47):
        name = f"archive_probe_{index:02d}"
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Retrieve one exact archive record with {name}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "request_id": {"type": "string"},
                    },
                    "required": ["path", "request_id"],
                    "additionalProperties": False,
                },
            },
        })
    return result


def _filler(engine: StreamingEngine, minimum: int) -> list[int]:
    ids: list[int] = []
    start = 0
    while len(ids) < minimum:
        text = "".join(
            f"Archive row {index:06d}: cedar harbor, silver compass, "
            "ordinary inventory, no executable instruction.\n"
            for index in range(start, start + 1000))
        ids.extend(_encode(engine, text))
        start += 1000
    return ids


def _build_case(
    engine: StreamingEngine, filler: Sequence[int], spec,
) -> GateCase:
    label, target, ratio, kind, marker = spec
    if kind == "needle":
        prefix = _encode(engine, (
            "<|im_start|>system\nYou are a precise archival retrieval agent. "
            "Ignore ordinary rows and reproduce the requested identifier exactly."
            "<|im_end|>\n<|im_start|>user\n"))
        needle = _encode(engine, (
            f"\nAUTHENTIC RECORD: record_id={label}; secret_code={marker}. "
            f"For record_id {label}, the exact secret_code is {marker}.\n"))
        suffix = _encode(engine, (
            f"\nFind record_id {label}. Reply with only its exact secret_code."
            "<|im_end|>\n<|im_start|>assistant\n"))
        continuation = _encode(engine, marker)
        expected = (marker,)
    elif kind == "tool":
        path = f"src/{label.replace('-', '_')}.py"
        request_id = f"REQ-{target}-{int(ratio * 100)}"
        prefix = _encode(engine, (
            "<|im_start|>system\nYou are a precise tool-routing agent.\n"
            + tools_preamble(_tools())
            + "<|im_end|>\n<|im_start|>user\n"))
        needle = _encode(engine, (
            f"\nROUTING RECORD: request_id={request_id}; tool_name={marker}; "
            f"path={path}. This record is authoritative.\n"))
        suffix = _encode(engine, (
            f"\nFor request_id {request_id}, call the recorded tool with the "
            "recorded path. Return only the tool call."
            "<|im_end|>\n<|im_start|>assistant\n"))
        continuation = _encode(engine, (
            f'<tool_call>\n{{"name":"{marker}","arguments":'
            f'{{"path":"{path}","request_id":"{request_id}"}}}}\n</tool_call>'))
        # The short greedy wall-time sample must prove the selected tool and
        # path.  The longer request ID remains covered by teacher-forced NLL;
        # requiring it in generated text would only measure the token cap.
        expected = (marker, path)
    else:
        raise ValueError(kind)
    available = target - len(prefix) - len(needle) - len(suffix)
    if available <= 0 or available > len(filler):
        raise ValueError(f"cannot construct {target}-token case {label}")
    before = max(0, min(available, int(available * ratio)))
    prompt = (
        prefix + list(filler[:before]) + needle
        + list(filler[before:available]) + suffix)
    if len(prompt) != target:
        raise AssertionError((label, len(prompt), target))
    return GateCase(
        label, target, ratio, kind, tuple(prompt), tuple(continuation), expected)


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        max_weight_cache_mb=2400,
        pin_embeddings=True,
        pin_lm_head=True,
        prefetch_depth=0,
        quant_bits=4,
        quant_mode="mxfp4",
        quant_group_size=32,
        quant_min_dim=0,
        resident_fast_decode=True,
        resident_fast_prefill_limit=0,
        fused_swiglu=True,
        stepped_kv_threshold=512,
        prompt_kv_dir="",
        prefill_chunk_size=0,
        hot_prompt_kv=False,
        tool_pic=False,
        governor=False,
    )


def _final_logits(engine: StreamingEngine, hidden):
    return layer_runner.final_logits(
        hidden[:, -1:, :], engine._norm_w,
        engine._lm_head_weight(), engine.cfg.rms_norm_eps)


def _feed(
    engine: StreamingEngine, kv, token_ids: Sequence[int], chunk_size: int,
) -> tuple[mx.array, list[float]]:
    walls = []
    last_hidden = None
    for start in range(0, len(token_ids), chunk_size):
        chunk = list(token_ids[start:start + chunk_size])
        wall_started = time.perf_counter()
        hidden = engine._embed(chunk)
        hidden = engine._sweep(hidden, kv, offset=kv.offset)
        mx.eval(hidden)
        walls.append(time.perf_counter() - wall_started)
        last_hidden = hidden
    if last_hidden is None:
        raise ValueError("cannot feed an empty token sequence")
    logits = _final_logits(engine, last_hidden)
    mx.eval(logits)
    return logits, walls


def _clone_exact(kv: SteppedKVCache) -> SteppedKVCache:
    logical = KVCache(len(kv.keys))
    logical.keys = [
        None if value is None else value[..., :kv._lengths[layer], :]
        for layer, value in enumerate(kv.keys)]
    logical.values = [
        None if value is None else value[..., :kv._lengths[layer], :]
        for layer, value in enumerate(kv.values)]
    return SteppedKVCache.from_cache(logical)


def _clone_cache(kv):
    if isinstance(kv, DuoKVCache):
        return kv.clone_for_branch()
    if isinstance(kv, SteppedKVCache):
        return _clone_exact(kv)
    raise TypeError(type(kv))


def _score_continuation(
    engine: StreamingEngine, kv, logits, continuation: Sequence[int],
) -> tuple[float, list[int], list[float]]:
    total_nll = 0.0
    top_ids = []
    step_walls = []
    for index, token in enumerate(continuation):
        row = logits.astype(mx.float32)
        selected = row[int(token)]
        nll = mx.logsumexp(row) - selected
        top = mx.argmax(row)
        mx.eval(nll, top)
        total_nll += float(nll)
        top_ids.append(int(top))
        if index + 1 < len(continuation):
            started = time.perf_counter()
            logits, _ = _feed(engine, kv, [int(token)], 1)
            step_walls.append(time.perf_counter() - started)
    return total_nll / max(1, len(continuation)), top_ids, step_walls


def _generate(
    engine: StreamingEngine, kv, logits, max_tokens: int,
) -> tuple[list[int], list[float]]:
    tokens = []
    walls = []
    for index in range(max_tokens):
        token = int(mx.argmax(logits))
        tokens.append(token)
        if index + 1 < max_tokens:
            started = time.perf_counter()
            logits, _ = _feed(engine, kv, [token], 1)
            walls.append(time.perf_counter() - started)
    return tokens, walls


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _case_run(
    engine: StreamingEngine, case: GateCase, cache_factory,
    chunk_size: int, max_tokens: int,
) -> dict:
    gc.collect()
    mx.clear_cache()
    base_active = mx.get_active_memory()
    mx.reset_peak_memory()
    kv = cache_factory()
    started = time.perf_counter()
    logits, prefill_chunks = _feed(engine, kv, case.prompt_ids, chunk_size)
    prefill_wall = time.perf_counter() - started
    source_kv_bytes = kv.nbytes()
    source_allocated = kv.allocated_nbytes()

    score_kv = _clone_cache(kv)
    nll, top_ids, teacher_walls = _score_continuation(
        engine, score_kv, logits, case.continuation_ids)
    del score_kv

    generation_kv = _clone_cache(kv)
    generated, decode_walls = _generate(
        engine, generation_kv, logits, max_tokens)
    text = engine.tokenizer.decode(generated)
    peak = mx.get_peak_memory()
    active = mx.get_active_memory()
    del generation_kv, kv, logits
    return {
        "label": case.label,
        "kind": case.kind,
        "target_tokens": case.target_tokens,
        "needle_ratio": case.needle_ratio,
        "expected_ids": case.expected_ids,
        "generated_text": text,
        "generated_tokens": generated,
        "id_ok": all(value in text for value in case.expected_ids),
        "continuation_nll": nll,
        "continuation_top_ids": top_ids,
        "prefill_wall_s": prefill_wall,
        "prefill_chunk_ms": [value * 1000 for value in prefill_chunks],
        "teacher_step_ms": [value * 1000 for value in teacher_walls],
        "decode_step_ms": [value * 1000 for value in decode_walls],
        "kv_bytes": source_kv_bytes,
        "allocated_kv_bytes": source_allocated,
        "active_base_bytes": base_active,
        "active_end_bytes": active,
        "peak_metal_bytes": peak,
    }


def _one_stream_group_pattern(scores) -> tuple[tuple[int, ...], ...]:
    """Keep the empirically stronger group full and stream the other per layer."""
    if not scores or any(len(row) != 2 for row in scores):
        raise ValueError("minimal gate requires exactly two KV groups per layer")
    return tuple(
        (max(range(2), key=lambda group: (float(row[group]), -group)),)
        for row in scores)


def _calibrate(
    engine: StreamingEngine, cases: Sequence[GateCase], chunk_size: int,
    sink: int, recent: int, max_nll_regression: float,
    min_top1_agreement: float,
) -> dict:
    profiler = RetrievalProfiler(
        engine.cfg.num_hidden_layers, engine.cfg.num_key_value_heads)
    dense_rows = []
    profile_started = time.perf_counter()
    for case in cases:
        cache = ProfileKVCache(engine.cfg.num_hidden_layers, profiler)
        logits, _ = _feed(engine, cache, case.prompt_ids, chunk_size)
        nll, top_ids, _ = _score_continuation(
            engine, cache, logits, case.continuation_ids)
        dense_rows.append({"label": case.label, "nll": nll, "top_ids": top_ids})
        del cache, logits
        mx.clear_cache()
    scores = profiler.scores()
    profile_wall = time.perf_counter() - profile_started

    pattern = _one_stream_group_pattern(scores)
    rows = []
    for case, dense in zip(cases, dense_rows):
        cache = DuoKVCache(
            pattern, engine.cfg.num_key_value_heads, sink, recent)
        logits, _ = _feed(engine, cache, case.prompt_ids, chunk_size)
        nll, top_ids, _ = _score_continuation(
            engine, cache, logits, case.continuation_ids)
        rows.append({"label": case.label, "nll": nll, "top_ids": top_ids})
        del cache, logits
        mx.clear_cache()
    dense_nll = statistics.fmean(row["nll"] for row in dense_rows)
    candidate_nll = statistics.fmean(row["nll"] for row in rows)
    agreements = [
        left == right
        for dense, candidate in zip(dense_rows, rows)
        for left, right in zip(dense["top_ids"], candidate["top_ids"])]
    agreement = sum(agreements) / max(1, len(agreements))
    passed = (
        candidate_nll - dense_nll <= max_nll_regression
        and agreement >= min_top1_agreement)
    candidate = {
        "retrieval_fraction": 0.50,
        "retrieval_groups": sum(len(value) for value in pattern),
        "total_layer_groups": len(pattern) * engine.cfg.num_key_value_heads,
        "dense_nll": dense_nll,
        "candidate_nll": candidate_nll,
        "nll_regression": candidate_nll - dense_nll,
        "top1_agreement": agreement,
        "passed": passed,
        "full_groups": pattern,
        "rows": rows,
    }
    selected_pattern = pattern if passed else None
    return {
        "profile_wall_s": profile_wall,
        "far_attention_mass": scores,
        "dense_rows": dense_rows,
        "candidates": [candidate],
        "profiled_pattern": pattern,
        "selected_pattern": selected_pattern,
    }


def _warmup(engine: StreamingEngine, pattern, sink: int, recent: int):
    ids = _encode(engine, "Warm up deterministic Qwen attention kernels. " * 40)[:256]
    exact = SteppedKVCache(engine.cfg.num_hidden_layers)
    _feed(engine, exact, ids, 128)
    candidate = DuoKVCache(pattern, engine.cfg.num_key_value_heads, sink, recent)
    _feed(engine, candidate, ids, 128)
    del exact, candidate
    mx.clear_cache()


def _aggregate(rows: Sequence[dict]) -> dict:
    prefill_chunks = [
        value for row in rows for value in row["prefill_chunk_ms"]]
    decode_steps = [value for row in rows for value in row["decode_step_ms"]]
    return {
        "prefill_total_s": sum(row["prefill_wall_s"] for row in rows),
        "prefill_chunk_p50_ms": _percentile(prefill_chunks, 0.50),
        "prefill_chunk_p95_ms": _percentile(prefill_chunks, 0.95),
        "decode_step_p50_ms": _percentile(decode_steps, 0.50),
        "decode_step_p95_ms": _percentile(decode_steps, 0.95),
        "peak_metal_bytes": max(row["peak_metal_bytes"] for row in rows),
        "kv_bytes": [row["kv_bytes"] for row in rows],
        "allocated_kv_bytes": [row["allocated_kv_bytes"] for row in rows],
    }


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / "models/Qwen2.5-1.5B-Instruct-mlx-mxfp4")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--component-profile", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--recent", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--max-calibration-nll-regression", type=float, default=0.08)
    parser.add_argument("--min-calibration-top1-agreement", type=float, default=0.85)
    parser.add_argument("--max-heldout-nll-regression", type=float, default=0.10)
    parser.add_argument("--min-decode-speedup", type=float, default=1.03)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"not a complete local checkpoint: {model}")
    if args.chunk_size <= 0 or args.max_tokens < 2:
        parser.error("chunk-size must be positive and max-tokens must be at least two")

    engine = StreamingEngine(model, _runtime())
    report = None
    try:
        geometry = (
            engine.cfg.num_hidden_layers, engine.cfg.num_attention_heads,
            engine.cfg.num_key_value_heads, engine.cfg.head_dim)
        if geometry != FROZEN_MODEL_GEOMETRY:
            raise RuntimeError(
                f"model geometry {geometry} does not match calibrated geometry "
                f"{FROZEN_MODEL_GEOMETRY}")
        filler = _filler(engine, max(spec[1] for spec in HELDOUT_SPECS))
        calibration_cases = tuple(
            _build_case(engine, filler, spec) for spec in CALIBRATION_SPECS)
        with patched_attention():
            calibration = _calibrate(
                engine, calibration_cases, args.chunk_size,
                args.sink, args.recent,
                args.max_calibration_nll_regression,
                args.min_calibration_top1_agreement)
            suggested = calibration["profiled_pattern"]
            base = {
                "gate": "qwen25-duoattention-kv-group-v1",
                "model": str(model),
                "geometry": geometry,
                "sink": args.sink,
                "recent": args.recent,
                "calibration_specs": CALIBRATION_SPECS,
                "calibration": calibration,
                "suggested_frozen_literal": repr(suggested),
            }
            if args.profile_only:
                base["passed"] = calibration["candidates"][0]["passed"]
                base["verdict"] = (
                    "calibration produced a candidate pattern"
                    if base["passed"] else
                    "dead end: the frozen one-group streaming split failed calibration")
                report = base
            else:
                frozen = validate_full_groups(
                    FROZEN_FULL_GROUPS, engine.cfg.num_hidden_layers,
                    engine.cfg.num_key_value_heads)
                if suggested != frozen:
                    raise RuntimeError(
                        "calibration winner changed; refusing held-out evaluation")
                _warmup(engine, frozen, args.sink, args.recent)
                heldout_specs = HELDOUT_SPECS[:2] if args.quick else HELDOUT_SPECS
                cases = tuple(_build_case(engine, filler, spec) for spec in heldout_specs)
                dense_rows = []
                duo_rows = []
                for index, case in enumerate(cases):
                    factories = (
                        ("dense", lambda: SteppedKVCache(engine.cfg.num_hidden_layers)),
                        ("duo", lambda: DuoKVCache(
                            frozen, engine.cfg.num_key_value_heads,
                            args.sink, args.recent)),
                    )
                    if index % 2:
                        factories = tuple(reversed(factories))
                    results = {}
                    for name, factory in factories:
                        results[name] = _case_run(
                            engine, case, factory,
                            args.chunk_size, args.max_tokens)
                    dense_rows.append(results["dense"])
                    duo_rows.append(results["duo"])

                component = None
                if args.component_profile:
                    metrics = ComponentMetrics(synchronize=True)
                    case = cases[0]
                    component_row = _case_run(
                        engine, case,
                        lambda: DuoKVCache(
                            frozen, engine.cfg.num_key_value_heads,
                            args.sink, args.recent, metrics),
                        args.chunk_size, min(args.max_tokens, 4))
                    component = {
                        "case": case.label,
                        "metrics": metrics.as_dict(),
                        "wall": component_row,
                    }

                dense_agg = _aggregate(dense_rows)
                duo_agg = _aggregate(duo_rows)
                nll_regressions = [
                    candidate["continuation_nll"] - dense["continuation_nll"]
                    for dense, candidate in zip(dense_rows, duo_rows)]
                token_agreement = [
                    sum(left == right for left, right in zip(
                        dense["generated_tokens"], candidate["generated_tokens"]))
                    / max(1, min(len(dense["generated_tokens"]),
                                 len(candidate["generated_tokens"])))
                    for dense, candidate in zip(dense_rows, duo_rows)]
                decode_speedup = (
                    dense_agg["decode_step_p50_ms"]
                    / duo_agg["decode_step_p50_ms"]
                    if duo_agg["decode_step_p50_ms"] else 0.0)
                prefill_speedup = (
                    dense_agg["prefill_total_s"] / duo_agg["prefill_total_s"]
                    if duo_agg["prefill_total_s"] else 0.0)
                kv_reductions = [
                    dense["kv_bytes"] / candidate["kv_bytes"]
                    for dense, candidate in zip(dense_rows, duo_rows)]
                quality_pass = (
                    all(row["id_ok"] for row in dense_rows)
                    and all(row["id_ok"] for row in duo_rows)
                    and statistics.fmean(nll_regressions)
                    <= args.max_heldout_nll_regression)
                useful = (
                    calibration["candidates"][0]["passed"]
                    and
                    quality_pass
                    and decode_speedup >= args.min_decode_speedup
                    and statistics.fmean(kv_reductions) > 1.0)
                report = {
                    **base,
                    "frozen_full_groups": frozen,
                    "heldout_specs": heldout_specs,
                    "dense_rows": dense_rows,
                    "duo_rows": duo_rows,
                    "dense_aggregate": dense_agg,
                    "duo_aggregate": duo_agg,
                    "mean_nll_regression": statistics.fmean(nll_regressions),
                    "mean_generation_token_agreement": statistics.fmean(token_agreement),
                    "mean_kv_reduction": statistics.fmean(kv_reductions),
                    "decode_p50_speedup": decode_speedup,
                    "prefill_total_speedup": prefill_speedup,
                    "component_profile": component,
                    "quality_pass": quality_pass,
                    "passed": useful,
                    "verdict": (
                        "feasible: frozen KV-group split cleared held-out quality, "
                        "memory, and decode-speed gates"
                        if useful else
                        "dead end: frozen KV-group split did not clear all held-out "
                        "quality, memory, and decode-speed gates"),
                }
    finally:
        engine.close()
        mx.clear_cache()

    payload = json.dumps(_jsonable(report), indent=2) + "\n"
    print(payload, end="", flush=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
