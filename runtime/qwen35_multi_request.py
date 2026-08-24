"""Experimental exact layer-stationary decode for independent Qwen requests.

This module deliberately stops below the HTTP/generation layer.  It consumes
one already-selected token from each request, advances that request's own
hybrid KV/KDA endpoint, and returns the resulting next-token logits.  The
caller remains responsible for admission, queueing, sampling, stop handling,
and serializing access to the shared ``StreamingEngine``.

The optimization is I/O scheduling only: every decoder block still sees the
ordinary ``(1, 1, hidden)`` shape once per request.  A target trunk layer page
is fetched once, retained while all requests execute that layer independently,
then released before moving to the next layer.  No activations, attention KV,
DeltaNet state, convolution history, router calls, or expert arithmetic are
batched across requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
import time
from typing import Any, Mapping, Sequence

import mlx.core as mx
import numpy as np

from .lm_head_stream import StreamedLMHead
from .qwen35 import qwen35_rms_norm, run_qwen35_block


_QWEN35_MODEL_TYPES = frozenset(("qwen3_5", "qwen3_5_moe"))
_HARD_MAX_REQUESTS = 16


@dataclass(frozen=True)
class QwenLayerStationaryRequest:
    """One independent one-token Qwen decode input.

    ``kv`` must be the request's private hybrid endpoint.  In particular, its
    ordinary attention cache and companion ``kda_cache`` may not be shared
    with another item in the same step.  ``positions3`` is the optional
    one-token multimodal M-RoPE coordinate matrix, with shape ``(3, 1)``.
    """

    request_id: str
    token: int
    kv: Any
    positions3: np.ndarray | mx.array | None = None


@dataclass(frozen=True)
class QwenLayerStationaryOutput:
    request_id: str
    logits: mx.array
    greedy_token: int


@dataclass(frozen=True)
class QwenLayerStationaryTelemetry:
    request_count: int
    layer_count: int
    request_tokens: int
    wall_seconds: float
    request_tokens_per_second: float
    layer_page_get_calls: int
    serial_equivalent_layer_page_get_calls: int
    layer_page_get_call_savings: int
    layer_page_get_call_reduction_fraction: float
    target_layer_page_bytes_read: int
    total_cache_bytes_read: int
    expert_or_other_bytes_read: int
    streamed_lm_head_read_calls: int
    streamed_lm_head_bytes_read: int
    total_weight_bytes_read: int
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    cache_disk_seconds: float
    bytes_read_per_request_token: float
    shared_streamed_lm_head: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class QwenLayerStationaryBatchResult:
    outputs: tuple[QwenLayerStationaryOutput, ...]
    telemetry: QwenLayerStationaryTelemetry

    def by_request_id(self) -> Mapping[str, QwenLayerStationaryOutput]:
        """Return outputs in caller order, keyed by the validated unique id."""
        return {output.request_id: output for output in self.outputs}


def _cache_stat(cache, name: str, default=0):
    stats = getattr(cache, "stats", None)
    return getattr(stats, name, default) if stats is not None else default


def _resident_adjusted_transient(
    start_active: int, end_active: int, peak_active: int,
) -> int:
    """Scratch above resident growth; kept local to avoid an engine import."""
    return max(0, int(peak_active) - max(int(start_active), int(end_active)))


class QwenLayerStationaryScheduler:
    """Callable, synchronous layer-stationary Qwen decode building block.

    This is intentionally not automatic server batching.  One scheduler may
    not be re-entered, and callers must ensure no other operation is using the
    engine while :meth:`advance` runs.  ``max_requests`` is an explicit memory
    and tail-latency bound; a hard ceiling prevents accidental unbounded
    activation/KV admission in an experimental API.
    """

    HARD_MAX_REQUESTS = _HARD_MAX_REQUESTS

    def __init__(self, engine, *, max_requests: int = 4):
        if (not isinstance(max_requests, Integral)
                or isinstance(max_requests, bool)
                or not 1 <= int(max_requests) <= self.HARD_MAX_REQUESTS):
            raise ValueError(
                "Qwen layer-stationary max_requests must be in "
                f"[1, {self.HARD_MAX_REQUESTS}]")
        self.engine = engine
        self.max_requests = int(max_requests)
        self._advancing = False

    def _validate(
        self, requests: Sequence[QwenLayerStationaryRequest],
    ) -> tuple[QwenLayerStationaryRequest, ...]:
        cfg = getattr(self.engine, "cfg", None)
        if cfg is None or cfg.model_type not in _QWEN35_MODEL_TYPES:
            actual = None if cfg is None else cfg.model_type
            raise ValueError(
                "Qwen layer-stationary scheduling supports only qwen3_5 and "
                f"qwen3_5_moe engines, got {actual!r}")
        items = tuple(requests)
        if not items:
            raise ValueError("Qwen layer-stationary decode needs requests")
        if len(items) > self.max_requests:
            raise ValueError(
                f"Qwen layer-stationary batch has {len(items)} requests; "
                f"configured maximum is {self.max_requests}")

        ids: set[str] = set()
        kv_ids: set[int] = set()
        kda_ids: set[int] = set()
        for item in items:
            if not isinstance(item, QwenLayerStationaryRequest):
                raise TypeError(
                    "Qwen layer-stationary items must be "
                    "QwenLayerStationaryRequest instances")
            if not isinstance(item.request_id, str) or not item.request_id:
                raise ValueError("request_id must be a non-empty string")
            if item.request_id in ids:
                raise ValueError(
                    f"duplicate Qwen request_id {item.request_id!r}")
            ids.add(item.request_id)
            if (not isinstance(item.token, Integral)
                    or isinstance(item.token, bool)
                    or not 0 <= int(item.token) < int(cfg.vocab_size)):
                raise ValueError(
                    f"request {item.request_id!r} token {item.token!r} is "
                    f"outside [0, {cfg.vocab_size})")
            if item.kv is None:
                raise ValueError(
                    f"request {item.request_id!r} has no KV endpoint")
            if id(item.kv) in kv_ids:
                raise ValueError(
                    "Qwen layer-stationary requests must not share a KV "
                    "endpoint")
            kv_ids.add(id(item.kv))
            kda = getattr(item.kv, "kda_cache", None)
            if kda is None:
                raise ValueError(
                    f"request {item.request_id!r} KV endpoint has no "
                    "DeltaNet state cache")
            if id(kda) in kda_ids:
                raise ValueError(
                    "Qwen layer-stationary requests must not share a KDA "
                    "state cache")
            kda_ids.add(id(kda))
            offset = getattr(item.kv, "offset", None)
            if not isinstance(offset, Integral) or int(offset) < 0:
                raise ValueError(
                    f"request {item.request_id!r} has invalid KV offset "
                    f"{offset!r}")
            if item.positions3 is not None:
                shape = tuple(int(value) for value in item.positions3.shape)
                if shape != (3, 1):
                    raise ValueError(
                        f"request {item.request_id!r} positions3 must have "
                        f"shape (3, 1), got {shape}")
        return items

    def advance(
        self, requests: Sequence[QwenLayerStationaryRequest],
    ) -> QwenLayerStationaryBatchResult:
        """Advance every independent request by exactly one consumed token."""
        if self._advancing:
            raise RuntimeError("Qwen layer-stationary scheduler is not reentrant")
        items = self._validate(requests)
        self._advancing = True
        try:
            return self._advance_validated(items)
        finally:
            self._advancing = False

    def _advance_validated(
        self, items: tuple[QwenLayerStationaryRequest, ...],
    ) -> QwenLayerStationaryBatchResult:
        engine = self.engine
        cfg = engine.cfg
        cache = engine.cache
        count = len(items)
        layer_count = int(cfg.num_hidden_layers)
        started = time.perf_counter()
        stats_before = {
            "bytes_read": int(_cache_stat(cache, "bytes_read")),
            "hits": int(_cache_stat(cache, "hits")),
            "misses": int(_cache_stat(cache, "misses")),
            "evictions": int(_cache_stat(cache, "evictions")),
            "disk_s": float(_cache_stat(cache, "disk_s", 0.0)),
        }

        # Separate calls retain the ordinary embedding lookup shape too.
        hidden = [engine._embed([int(item.token)]) for item in items]
        offsets = [int(item.kv.offset) for item in items]
        target_bytes_read = 0
        layer_page_get_calls = 0
        rc = engine.rc

        for layer in range(layer_count):
            select_transient = getattr(
                engine, "_select_serial_verify_layer_transient", None)
            if select_transient is not None:
                select_transient(count, layer)

            key = engine._layer_key(layer)
            names = engine._layer_names(layer)
            contains = getattr(cache, "contains", None)
            if contains is not None and not contains(key):
                estimate = int(engine._layer_fetch_bytes_estimate(layer))
                prepare = getattr(cache, "prepare_for", None)
                if prepare is not None:
                    prepare(estimate)
                governor = getattr(engine, "governor", None)
                if governor is not None:
                    governor.reserve(estimate)

            wait_started = time.perf_counter()
            bytes_before_get = int(_cache_stat(cache, "bytes_read"))
            weights = cache.get(key, names)
            target_bytes_read += max(
                0,
                int(_cache_stat(cache, "bytes_read")) - bytes_before_get,
            )
            layer_page_get_calls += 1
            timer = getattr(engine, "timer", None)
            if timer is not None:
                timer.add("weights_wait", time.perf_counter() - wait_started)

            governor = getattr(engine, "governor", None)
            transient = int(getattr(engine, "_layer_transient", 0))
            if governor is not None and transient:
                governor.reserve(
                    transient,
                    margin=int(getattr(engine, "_layer_transient_margin", 0)),
                )

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            next_hidden = []
            prefix = f"model.layers.{layer}"
            for item, value, offset in zip(items, hidden, offsets, strict=True):
                # Deliberately one request at a time.  In particular, MoE
                # routing and expert matmuls never acquire a batch dimension.
                next_hidden.append(run_qwen35_block(
                    value,
                    weights,
                    prefix,
                    cfg,
                    item.kv,
                    layer,
                    offset,
                    engine._get_experts,
                    iter_expert_batches=engine._iter_expert_batches,
                    positions3=item.positions3,
                    zmlx_fused_decode=rc.zmlx_fused_deltanet_decode,
                    native_fused_decode=rc.native_fused_deltanet_decode,
                    chunked_delta_prefill=rc.qwen_chunked_delta_prefill,
                    compiled_delta_prefill=rc.qwen_compiled_delta_prefill,
                    native_fused_delta_prefill=(
                        rc.qwen_native_fused_delta_prefill),
                ))
            # One barrier per shared layer, after independent one-token graphs.
            mx.eval(*next_hidden)
            hidden = next_hidden

            record_transient = getattr(
                engine, "_record_serial_verify_layer_transient", None)
            if record_transient is not None:
                record_transient(
                    count,
                    layer,
                    _resident_adjusted_transient(
                        active_before,
                        mx.get_active_memory(),
                        mx.get_peak_memory(),
                    ),
                )
            note_peak = getattr(engine, "_note_true_peak", None)
            if note_peak is not None:
                note_peak()
            del weights

        head = engine._lm_head_weight()
        shared_streamed_head = isinstance(head, StreamedLMHead)
        if shared_streamed_head:
            # Stream each vocab block once while still issuing a distinct
            # one-row matmul for every request (logits_serial_rows' contract).
            normalized = [
                qwen35_rms_norm(
                    value[:, -1:, :], engine._norm_w, cfg.rms_norm_eps)
                for value in hidden
            ]
            rows = mx.concatenate(normalized, axis=1)
            all_logits = head.logits_serial_rows(rows)
            logits = [all_logits[0, index] for index in range(count)]
        else:
            # A batched LM-head matmul can select a different reduction kernel;
            # preserve the ordinary independent call shape instead.
            logits = [
                engine._final_logits(value, head=head) for value in hidden
            ]
        mx.eval(*logits)
        outputs = tuple(
            QwenLayerStationaryOutput(
                request_id=item.request_id,
                logits=value,
                greedy_token=int(mx.argmax(value).item()),
            )
            for item, value in zip(items, logits, strict=True)
        )

        wall_seconds = max(0.0, time.perf_counter() - started)
        stats_after = {
            "bytes_read": int(_cache_stat(cache, "bytes_read")),
            "hits": int(_cache_stat(cache, "hits")),
            "misses": int(_cache_stat(cache, "misses")),
            "evictions": int(_cache_stat(cache, "evictions")),
            "disk_s": float(_cache_stat(cache, "disk_s", 0.0)),
        }
        total_bytes = max(
            0, stats_after["bytes_read"] - stats_before["bytes_read"])
        streamed_head_calls = (
            (int(head.vocab) + int(head.block_rows) - 1)
            // int(head.block_rows)
            if shared_streamed_head else 0
        )
        streamed_head_bytes = (
            int(head.vocab) * int(head.row_bytes)
            if shared_streamed_head else 0
        )
        total_weight_bytes = total_bytes + streamed_head_bytes
        serial_gets = count * layer_count
        savings = serial_gets - layer_page_get_calls
        telemetry = QwenLayerStationaryTelemetry(
            request_count=count,
            layer_count=layer_count,
            request_tokens=count,
            wall_seconds=wall_seconds,
            request_tokens_per_second=(
                count / wall_seconds if wall_seconds > 0 else float("inf")),
            layer_page_get_calls=layer_page_get_calls,
            serial_equivalent_layer_page_get_calls=serial_gets,
            layer_page_get_call_savings=savings,
            layer_page_get_call_reduction_fraction=(
                savings / serial_gets if serial_gets else 0.0),
            target_layer_page_bytes_read=target_bytes_read,
            total_cache_bytes_read=total_bytes,
            expert_or_other_bytes_read=max(0, total_bytes - target_bytes_read),
            streamed_lm_head_read_calls=streamed_head_calls,
            streamed_lm_head_bytes_read=streamed_head_bytes,
            total_weight_bytes_read=total_weight_bytes,
            cache_hits=max(0, stats_after["hits"] - stats_before["hits"]),
            cache_misses=max(
                0, stats_after["misses"] - stats_before["misses"]),
            cache_evictions=max(
                0, stats_after["evictions"] - stats_before["evictions"]),
            cache_disk_seconds=max(
                0.0, stats_after["disk_s"] - stats_before["disk_s"]),
            bytes_read_per_request_token=total_weight_bytes / count,
            shared_streamed_lm_head=shared_streamed_head,
        )
        return QwenLayerStationaryBatchResult(outputs, telemetry)
