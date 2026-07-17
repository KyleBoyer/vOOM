#!/usr/bin/env python3
"""Fixture-only, explicitly lossy FastKV-style token-selective propagation.

This module is intentionally outside ``runtime``.  It is a falsifiable Qwen2
prototype, not a reusable cache implementation: every request owns a fresh
ragged KV state, every layer carries explicit absolute positions, and the state
is never installed in ``StreamingEngine`` or admitted to prompt/PIC/hot caches.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import mlx.core as mx
import numpy as np

from runtime import layer_runner
from runtime.tool_capsules import _apply_rope


PROFILE_FAMILY = "lossy-fastkv-tsp-qwen-v0"


class UnsupportedFastKVTSP(ValueError):
    """The fixture cannot prove that this architecture/config is admissible."""


class StateIsolationError(RuntimeError):
    """The fixture observed a mutation of engine-owned request state."""


@dataclass(frozen=True)
class TSPProfile:
    tsp_layer: int
    retention: float
    recent_window: int = 8
    pool_width: int = 7
    query_chunk: int = 1024

    @property
    def approximate(self) -> bool:
        return self.retention < 1.0

    @property
    def name(self) -> str:
        percent = round(self.retention * 100)
        return f"{PROFILE_FAMILY}-r{percent:02d}-l{self.tsp_layer:02d}"


@dataclass
class LayerPositionKV:
    """One layer's K/V and the absolute position represented by every row."""

    positions: mx.array
    keys: mx.array
    values: mx.array

    def clone(self) -> "LayerPositionKV":
        # MLX arrays are immutable. Decode replaces, rather than mutates, each
        # concatenated view, so prompt arrays can be safely shared by branches.
        return LayerPositionKV(self.positions, self.keys, self.values)

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.positions.nbytes + self.keys.nbytes + self.values.nbytes)

    def validate(self, *, layer: int, num_kv_heads: int, head_dim: int) -> None:
        if self.positions.ndim != 1:
            raise AssertionError(f"layer {layer} positions are not rank one")
        if self.keys.shape != self.values.shape:
            raise AssertionError(f"layer {layer} K/V shapes differ")
        expected = (1, num_kv_heads, self.count, head_dim)
        if tuple(self.keys.shape) != expected:
            raise AssertionError(
                f"layer {layer} K/V shape {self.keys.shape} != {expected}")
        positions = np.asarray(self.positions.tolist(), dtype=np.int64)
        if positions.size and np.any(np.diff(positions) <= 0):
            raise AssertionError(f"layer {layer} positions are not strictly increasing")


@dataclass
class RaggedTSPState:
    layers: list[LayerPositionKV]
    prompt_length: int
    profile_name: str
    approximate: bool

    def clone(self) -> "RaggedTSPState":
        return RaggedTSPState(
            [layer.clone() for layer in self.layers], self.prompt_length,
            self.profile_name, self.approximate)

    @property
    def cache_bytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def validate(self, cfg) -> None:
        if len(self.layers) != cfg.num_hidden_layers:
            raise AssertionError("ragged state layer count mismatch")
        for index, layer in enumerate(self.layers):
            layer.validate(
                layer=index, num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim)

    def metadata(self) -> dict:
        """Metadata is deliberately ineligible for any exact/reuse contract."""
        return {
            "profile": self.profile_name,
            "approximate": bool(self.approximate),
            "exact": not self.approximate,
            "persistent": False,
            "reusable": False,
            "cache_scope": "fixture-local-request-only",
        }

    def position_geometry(self) -> list[dict]:
        rows = []
        for index, layer in enumerate(self.layers):
            values = np.asarray(layer.positions.tolist(), dtype=np.int32)
            rows.append({
                "layer": index,
                "count": int(values.size),
                "first": int(values[0]) if values.size else None,
                "last": int(values[-1]) if values.size else None,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            })
        return rows


@dataclass
class PrefillResult:
    logits: mx.array
    state: RaggedTSPState
    selected_positions: tuple[int, ...]
    selected_attention_mass: float
    wall_s: float
    selector_s: float
    layer_s: tuple[float, ...]
    active_before_bytes: int
    active_after_bytes: int
    peak_bytes: int

    def metrics(self) -> dict:
        return {
            "profile": self.state.profile_name,
            "approximate": self.state.approximate,
            "wall_s": self.wall_s,
            "selector_s": self.selector_s,
            "layer_s": list(self.layer_s),
            "selected_tokens": len(self.selected_positions),
            "selected_attention_mass": self.selected_attention_mass,
            "cache_bytes": self.state.cache_bytes,
            "active_before_bytes": self.active_before_bytes,
            "active_after_bytes": self.active_after_bytes,
            "active_delta_bytes": self.active_after_bytes - self.active_before_bytes,
            "peak_bytes": self.peak_bytes,
            "peak_delta_bytes": max(0, self.peak_bytes - self.active_before_bytes),
            "state": self.state.metadata(),
            "positions": self.state.position_geometry(),
        }


@dataclass
class GreedyResult:
    tokens: list[int]
    text: str
    nll: list[float]
    state: RaggedTSPState
    logits: mx.array
    completed_constraint: bool
    wall_s: float


_MISSING = object()
_STATE_ATTRS = (
    "last_kv",
    "_hot_prompt_slots",
    "_prompt_kv_store",
    "_hot_kv_persist",
    "_vision_prompt_cache",
    "_position_free_pool",
    "_provisional",
    "_h_window",
    "_h_last",
    "_tap_hidden",
)


def _snapshot_value(value):
    if value is _MISSING:
        return ("missing",)
    if isinstance(value, list):
        return ("list", id(value), tuple(id(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict", id(value),
            tuple(sorted((repr(key), id(item)) for key, item in value.items())),
        )
    return ("object", id(value))


def engine_state_snapshot(engine) -> dict[str, tuple]:
    return {
        name: _snapshot_value(getattr(engine, name, _MISSING))
        for name in _STATE_ATTRS
    }


T = TypeVar("T")


def isolated_engine_call(engine, callback: Callable[[], T]) -> T:
    """Run fixture math and prove that engine request state did not change."""
    before = engine_state_snapshot(engine)
    try:
        return callback()
    finally:
        after = engine_state_snapshot(engine)
        if after != before:
            changed = sorted(name for name in before if before[name] != after[name])
            raise StateIsolationError(
                "fixture mutated engine-owned request state: " + ", ".join(changed))


def _runtime_flag(engine, name: str):
    return getattr(getattr(engine, "rc", None), name, False)


def validate_admission(engine, profile: TSPProfile, prompt_length: int | None = None) -> None:
    """Fail closed unless the fixture's dense text Qwen2 assumptions are proven."""
    cfg = engine.cfg
    if cfg.model_type != "qwen2":
        raise UnsupportedFastKVTSP(
            f"{PROFILE_FAMILY} requires model_type='qwen2', got {cfg.model_type!r}")
    if int(getattr(cfg, "num_experts", 0)):
        raise UnsupportedFastKVTSP("MoE checkpoints are unsupported")
    if getattr(cfg, "vision_config", None):
        raise UnsupportedFastKVTSP("vision/M-RoPE checkpoints are unsupported")
    if getattr(cfg, "layer_types", ()):
        raise UnsupportedFastKVTSP("mixed/sliding layer types are unsupported")
    if getattr(cfg, "rope_interleave", False):
        raise UnsupportedFastKVTSP("interleaved RoPE is unsupported")
    if cfg.num_key_value_heads <= 0 or (
            cfg.num_attention_heads % cfg.num_key_value_heads):
        raise UnsupportedFastKVTSP("attention requires an integral GQA ratio")
    if cfg.head_dim <= 0 or cfg.head_dim % 2:
        raise UnsupportedFastKVTSP("RoPE requires a positive even head dimension")
    if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size:
        raise UnsupportedFastKVTSP("attention head geometry does not span hidden_size")
    if not 0 <= profile.tsp_layer < cfg.num_hidden_layers - 1:
        raise UnsupportedFastKVTSP(
            "tsp_layer must leave at least one late propagation layer")
    if not 0 < profile.retention <= 1:
        raise UnsupportedFastKVTSP("retention must be within (0, 1]")
    if profile.recent_window <= 0:
        raise UnsupportedFastKVTSP("recent_window must be positive")
    if profile.pool_width <= 0 or profile.pool_width % 2 == 0:
        raise UnsupportedFastKVTSP("pool_width must be a positive odd integer")
    if profile.query_chunk <= 0:
        raise UnsupportedFastKVTSP("query_chunk must be positive")
    if prompt_length is not None:
        maximum = int(getattr(
            engine, "effective_max_position_embeddings",
            cfg.max_position_embeddings))
        if prompt_length <= profile.recent_window:
            raise UnsupportedFastKVTSP(
                "prompt must be longer than the protected recent window")
        if prompt_length > maximum:
            raise UnsupportedFastKVTSP(
                f"prompt length {prompt_length} exceeds admitted maximum {maximum}")

    forbidden_flags = (
        "hot_prompt_kv", "prompt_kv_dir", "hot_prompt_kv_persist_dir",
        "tool_pic", "tool_pic_shared_pages", "max_kv_mb",
        "prefill_checkpoint_every", "adaptive_chunk_size",
        "resident_fast_decode", "resident_moe_decode",
    )
    enabled = [name for name in forbidden_flags if _runtime_flag(engine, name)]
    if enabled:
        raise UnsupportedFastKVTSP(
            "fixture requires isolated nonpersistent runtime flags; enabled: "
            + ", ".join(enabled))

    nonempty_state = []
    for name in (
            "last_kv", "_prompt_kv_store", "_hot_kv_persist",
            "_vision_prompt_cache", "_position_free_pool", "_provisional",
            "_h_window", "_h_last"):
        if getattr(engine, name, None) is not None:
            nonempty_state.append(name)
    for name in ("_hot_prompt_slots", "_tap_hidden"):
        if getattr(engine, name, None):
            nonempty_state.append(name)
    if nonempty_state:
        raise UnsupportedFastKVTSP(
            "engine already owns request/speculative state: "
            + ", ".join(nonempty_state))

    # ModelConfig intentionally omits Qwen's use_sliding_window flag. Check the
    # raw architecture declaration when a real engine exposes its checkpoint.
    model_dir = getattr(engine, "_model_dir", None)
    config_path = Path(model_dir) / "config.json" if model_dir else None
    if config_path is not None and config_path.is_file():
        raw = json.loads(config_path.read_text())
        if raw.get("model_type") != "qwen2":
            raise UnsupportedFastKVTSP("raw checkpoint is not Qwen2")
        architectures = raw.get("architectures", [])
        if architectures and architectures != ["Qwen2ForCausalLM"]:
            raise UnsupportedFastKVTSP(
                f"unsupported architecture declaration: {architectures!r}")
        if raw.get("use_sliding_window", False):
            raise UnsupportedFastKVTSP("Qwen sliding-window attention is unsupported")
        if raw.get("vision_config") or raw.get("num_experts"):
            raise UnsupportedFastKVTSP("raw checkpoint is not dense text-only Qwen2")


def select_positions(
    salience, *, retention: float, recent_window: int, pool_width: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Deterministic max-pooled top-k with a mandatory recent-token budget.

    The retention fraction is the total budget, including the protected recent
    window. Ties prefer the earlier absolute position. Returned positions are
    sorted for causal propagation, never ranked order.
    """
    values = np.asarray(salience, dtype=np.float64).reshape(-1)
    length = int(values.size)
    if length == 0 or not np.all(np.isfinite(values)):
        raise ValueError("salience must be a nonempty finite vector")
    if not 0 < retention <= 1:
        raise ValueError("retention must be within (0, 1]")
    if recent_window <= 0 or recent_window > length:
        raise ValueError("recent_window must be within the sequence")
    if pool_width <= 0 or pool_width % 2 == 0:
        raise ValueError("pool_width must be positive and odd")

    radius = pool_width // 2
    padded = np.pad(values, (radius, radius), constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, pool_width)
    pooled = windows.max(axis=-1)
    budget = max(recent_window, int(math.ceil(retention * length)))
    budget = min(length, budget)
    mandatory = set(range(length - recent_window, length))
    candidates = np.asarray(
        [index for index in range(length) if index not in mandatory],
        dtype=np.int64)
    remaining = budget - len(mandatory)
    if remaining:
        order = np.lexsort((candidates, -pooled[candidates]))
        mandatory.update(int(value) for value in candidates[order[:remaining]])
    selected = tuple(sorted(mandatory))
    if selected[-1] != length - 1 or len(selected) != budget:
        raise AssertionError("selector violated its endpoint/budget invariant")
    return selected, pooled


def _weights(engine, layer: int) -> dict:
    return engine.cache.get(
        engine._layer_key(layer), engine._layer_names(layer))


def _qkv(engine, x: mx.array, positions: mx.array, weights: dict, prefix: str):
    cfg = engine.cfg
    hidden = mx.fast.rms_norm(
        x, weights[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
    q = layer_runner._linear(hidden, weights, f"{prefix}.self_attn.q_proj")
    k = layer_runner._linear(hidden, weights, f"{prefix}.self_attn.k_proj")
    v = layer_runner._linear(hidden, weights, f"{prefix}.self_attn.v_proj")
    q_norm = weights.get(f"{prefix}.self_attn.q_norm.weight")
    k_norm = weights.get(f"{prefix}.self_attn.k_norm.weight")
    per_head_norm = q_norm is not None and q_norm.shape[0] == cfg.head_dim
    if q_norm is not None and not per_head_norm:
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
    batch, length, _ = q.shape
    q = q.reshape(
        batch, length, cfg.num_attention_heads,
        cfg.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(
        batch, length, cfg.num_key_value_heads,
        cfg.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(
        batch, length, cfg.num_key_value_heads,
        cfg.head_dim).transpose(0, 2, 1, 3)
    if per_head_norm:
        q = mx.fast.rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = mx.fast.rms_norm(k, k_norm, cfg.rms_norm_eps)
    q = _apply_rope(
        q, positions, cfg, engine._rope_freqs, scale=engine._mscale)
    k = _apply_rope(
        k, positions, cfg, engine._rope_freqs, scale=engine._mscale)
    return q, k, v


def _run_layer(
    engine, x: mx.array, query_positions: mx.array,
    previous: LayerPositionKV | None, layer: int, query_chunk: int,
    *, last_mlp_only: bool = False, expose_q: bool = False,
) -> tuple[mx.array, LayerPositionKV, mx.array | None]:
    cfg = engine.cfg
    weights = _weights(engine, layer)
    prefix = f"model.layers.{layer}"
    q, new_k, new_v = _qkv(engine, x, query_positions, weights, prefix)
    if previous is None:
        key_positions = query_positions
        keys, values = new_k, new_v
    else:
        if int(query_positions[0]) <= int(previous.positions[-1]):
            raise AssertionError("decode positions must append monotonically")
        key_positions = mx.concatenate((previous.positions, query_positions))
        keys = mx.concatenate((previous.keys, new_k), axis=2)
        values = mx.concatenate((previous.values, new_v), axis=2)
    mx.eval(q, keys, values, key_positions)

    attended_parts = []
    length = int(query_positions.shape[0])
    for start in range(0, length, query_chunk):
        end = min(length, start + query_chunk)
        q_part = q[:, :, start:end, :]
        mask = mx.where(
            key_positions[None, :] <= query_positions[start:end, None],
            0.0, float("-inf")).astype(q.dtype)
        part = mx.fast.scaled_dot_product_attention(
            q_part, keys, values, scale=cfg.head_dim ** -0.5, mask=mask)
        mx.eval(part)
        attended_parts.append(part)
    attended = (attended_parts[0] if len(attended_parts) == 1
                else mx.concatenate(attended_parts, axis=2))
    attended = attended.transpose(0, 2, 1, 3).reshape(
        x.shape[0], length, cfg.num_attention_heads * cfg.head_dim)
    residual = x + layer_runner._linear(
        attended, weights, f"{prefix}.self_attn.o_proj")
    if last_mlp_only:
        residual = residual[:, -1:, :]
    hidden = mx.fast.rms_norm(
        residual, weights[f"{prefix}.post_attention_layernorm.weight"],
        cfg.rms_norm_eps)
    output = residual + layer_runner._swiglu(
        hidden, weights, f"{prefix}.mlp", fused=False)
    layer_kv = LayerPositionKV(key_positions, keys, values)
    mx.eval(output, layer_kv.positions, layer_kv.keys, layer_kv.values)
    return output, layer_kv, q if expose_q else None


def _attention_salience(
    q: mx.array, keys: mx.array, query_positions: mx.array,
    key_positions: mx.array, cfg, window: int,
) -> np.ndarray:
    q_tail = q[:, :, -window:, :]
    q_positions = query_positions[-window:]
    ratio = cfg.num_attention_heads // cfg.num_key_value_heads
    expanded_keys = (
        keys if ratio == 1 else mx.repeat(keys, repeats=ratio, axis=1))
    scores = (q_tail @ expanded_keys.transpose(0, 1, 3, 2)) * (
        cfg.head_dim ** -0.5)
    mask = mx.where(
        key_positions[None, :] <= q_positions[:, None],
        0.0, float("-inf")).astype(scores.dtype)
    probabilities = mx.softmax(scores + mask[None, None], axis=-1, precise=True)
    salience = probabilities.astype(mx.float32).mean(axis=(0, 1, 2))
    mx.eval(salience)
    return np.asarray(salience.tolist(), dtype=np.float64)


def run_prefill(engine, tokens: list[int], profile: TSPProfile) -> PrefillResult:
    """Run full layers through TSP, then propagate only selected token rows."""
    tokens = list(tokens)
    validate_admission(engine, profile, len(tokens))

    def execute() -> PrefillResult:
        mx.synchronize()
        active_before = int(mx.get_active_memory())
        mx.reset_peak_memory()
        started = time.perf_counter()
        x = engine._embed(tokens)
        positions = mx.arange(len(tokens), dtype=mx.int32)
        layers: list[LayerPositionKV | None] = [None] * engine.cfg.num_hidden_layers
        layer_times = []
        selector_s = 0.0
        selected = tuple(range(len(tokens)))
        selected_mass = 1.0

        for layer in range(engine.cfg.num_hidden_layers):
            layer_started = time.perf_counter()
            x, layer_kv, selection_q = _run_layer(
                engine, x, positions, None, layer, profile.query_chunk,
                last_mlp_only=(layer == engine.cfg.num_hidden_layers - 1),
                expose_q=(profile.approximate and layer == profile.tsp_layer),
            )
            layers[layer] = layer_kv
            layer_times.append(time.perf_counter() - layer_started)

            if profile.approximate and layer == profile.tsp_layer:
                selector_started = time.perf_counter()
                salience = _attention_salience(
                    selection_q, layer_kv.keys, positions, layer_kv.positions,
                    engine.cfg, profile.recent_window)
                selected, _pooled = select_positions(
                    salience, retention=profile.retention,
                    recent_window=profile.recent_window,
                    pool_width=profile.pool_width)
                selected_index = mx.array(selected, dtype=mx.int32)
                x = mx.take(x, selected_index, axis=1)
                positions = mx.take(positions, selected_index)
                mx.eval(x, positions)
                selected_mass = float(salience[list(selected)].sum())
                selector_s = time.perf_counter() - selector_started

        logits = layer_runner.final_logits(
            x, engine._norm_w, engine._lm_head_weight(),
            engine.cfg.rms_norm_eps)
        mx.eval(logits)
        mx.synchronize()
        wall_s = time.perf_counter() - started
        state = RaggedTSPState(
            [value for value in layers if value is not None], len(tokens),
            profile.name, profile.approximate)
        state.validate(engine.cfg)
        if selected[-1] != len(tokens) - 1:
            raise AssertionError("TSP dropped the final prompt position")
        return PrefillResult(
            logits=logits,
            state=state,
            selected_positions=selected,
            selected_attention_mass=selected_mass,
            wall_s=wall_s,
            selector_s=selector_s,
            layer_s=tuple(layer_times),
            active_before_bytes=active_before,
            active_after_bytes=int(mx.get_active_memory()),
            peak_bytes=int(mx.get_peak_memory()),
        )

    return isolated_engine_call(engine, execute)


def _token_nll(logits: mx.array, token: int) -> float:
    scores = logits.astype(mx.float32)
    value = mx.logsumexp(scores) - scores[int(token)]
    mx.eval(value)
    return float(value)


def _decode_token(engine, state: RaggedTSPState, token: int, position: int,
                  query_chunk: int) -> mx.array:
    x = engine._embed([int(token)])
    positions = mx.array([position], dtype=mx.int32)
    for layer in range(engine.cfg.num_hidden_layers):
        x, layer_kv, _ = _run_layer(
            engine, x, positions, state.layers[layer], layer, query_chunk,
            last_mlp_only=False, expose_q=False)
        state.layers[layer] = layer_kv
    logits = layer_runner.final_logits(
        x, engine._norm_w, engine._lm_head_weight(), engine.cfg.rms_norm_eps)
    mx.eval(logits)
    return logits


def generate_greedy(
    engine, prefill: PrefillResult, *, max_tokens: int,
    constraint=None,
) -> GreedyResult:
    """Greedy decode from a private clone; the final emitted token is not fed."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    def execute() -> GreedyResult:
        state = prefill.state.clone()
        logits = prefill.logits
        generated: list[int] = []
        nll: list[float] = []
        started = time.perf_counter()
        for index in range(max_tokens):
            sampled = constraint.mask_logits(logits) if constraint is not None else logits
            token = int(mx.argmax(sampled))
            generated.append(token)
            nll.append(_token_nll(logits, token))
            if constraint is not None:
                constraint.accept_token(token)
            if ((constraint is not None and constraint.completed)
                    or token in engine.cfg.eos_token_ids
                    or index + 1 == max_tokens):
                break
            logits = _decode_token(
                engine, state, token, prefill.state.prompt_length + index,
                query_chunk=1)
        state.validate(engine.cfg)
        return GreedyResult(
            tokens=generated,
            text=engine.tokenizer.decode(generated),
            nll=nll,
            state=state,
            logits=logits,
            completed_constraint=bool(
                constraint is not None and constraint.completed),
            wall_s=time.perf_counter() - started,
        )

    return isolated_engine_call(engine, execute)


def score_teacher_tokens(
    engine, prefill: PrefillResult, tokens: list[int], *, query_chunk: int = 1,
) -> list[float]:
    """Score fixed tokens without allowing a divergent candidate path to leak."""
    target = [int(value) for value in tokens]
    if not target:
        return []

    def execute() -> list[float]:
        state = prefill.state.clone()
        logits = prefill.logits
        losses = []
        for index, token in enumerate(target):
            losses.append(_token_nll(logits, token))
            if index + 1 < len(target):
                logits = _decode_token(
                    engine, state, token, prefill.state.prompt_length + index,
                    query_chunk=query_chunk)
        state.validate(engine.cfg)
        return losses

    return isolated_engine_call(engine, execute)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count
