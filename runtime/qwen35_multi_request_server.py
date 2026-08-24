"""Explicit HTTP-generation coordinator for exact Qwen layer scheduling.

The low-level scheduler in :mod:`runtime.qwen35_multi_request` consumes one
token for each independent request.  This module supplies the deliberately
narrow serving policy around it: bounded admission, isolated prompt endpoints,
independent deterministic sampling/stop state, and aggregate I/O telemetry.

It is not an automatic queue.  The server calls it only from an explicit,
default-off endpoint while holding its ordinary global inference lock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import time
from typing import Any, Callable, Sequence

import mlx.core as mx

from .kv_cache import fork_hybrid_kv_endpoint
from .qwen35_multi_request import (
    QwenLayerStationaryRequest,
    QwenLayerStationaryScheduler,
)
from .sampler import SamplingParams, sample


ROUTE = "/qwen/layer-stationary/completions"
_MODEL_TYPES = frozenset(("qwen3_5", "qwen3_5_moe"))
_ALLOWED_TOP_LEVEL = frozenset(("model", "vmodel_mode", "requests", "stream"))
_ALLOWED_ITEM = frozenset((
    "id", "prompt", "max_tokens", "stop", "temperature", "top_p",
    "top_k", "seed", "repetition_penalty",
))


class QwenMultiRequestValidationError(ValueError):
    """A fail-closed batch request/configuration error (HTTP 400)."""


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise QwenMultiRequestValidationError(
            f"{name} must be an integer") from error
    if not low <= value <= high:
        raise QwenMultiRequestValidationError(
            f"{name} must be in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class QwenMultiRequestServerConfig:
    """Bounded, server-only admission policy; it does not alter engine math."""

    enabled: bool = False
    max_requests: int = 4
    max_prompt_tokens: int = 4096
    max_total_prompt_tokens: int = 8192
    max_output_tokens: int = 256
    max_total_output_tokens: int = 512

    @classmethod
    def from_env(cls) -> "QwenMultiRequestServerConfig":
        enabled_raw = os.environ.get(
            "VMODEL_QWEN_MULTI_REQUEST_BATCH", "0").strip()
        if enabled_raw not in ("0", "1"):
            raise QwenMultiRequestValidationError(
                "VMODEL_QWEN_MULTI_REQUEST_BATCH must be 0 or 1")
        return cls(
            enabled=enabled_raw == "1",
            max_requests=_env_int(
                "VMODEL_QWEN_MULTI_REQUEST_MAX_REQUESTS", 4,
                low=1, high=QwenLayerStationaryScheduler.HARD_MAX_REQUESTS),
            max_prompt_tokens=_env_int(
                "VMODEL_QWEN_MULTI_REQUEST_MAX_PROMPT_TOKENS", 4096,
                low=1, high=32768),
            max_total_prompt_tokens=_env_int(
                "VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_PROMPT_TOKENS", 8192,
                low=1, high=65536),
            max_output_tokens=_env_int(
                "VMODEL_QWEN_MULTI_REQUEST_MAX_OUTPUT_TOKENS", 256,
                low=1, high=1024),
            max_total_output_tokens=_env_int(
                "VMODEL_QWEN_MULTI_REQUEST_MAX_TOTAL_OUTPUT_TOKENS", 512,
                low=1, high=4096),
        )

    @property
    def identity(self) -> str:
        """Stable telemetry identity without forcing an engine-cache swap."""
        return (
            "qwen-ls-http-v1:"
            f"r{self.max_requests}:p{self.max_prompt_tokens}:"
            f"pt{self.max_total_prompt_tokens}:o{self.max_output_tokens}:"
            f"ot{self.max_total_output_tokens}"
        )

    def as_dict(self) -> dict[str, int | bool | str]:
        return {**asdict(self), "identity": self.identity}


@dataclass(frozen=True)
class QwenMultiRequestItem:
    request_id: str
    prompt: Any
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    sampling: SamplingParams
    stop: tuple[str, ...] = ()


def _sampling_from_item(item: dict) -> SamplingParams:
    for field in ("temperature", "top_p"):
        value = item.get(field)
        if value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise QwenMultiRequestValidationError(
                f"{field} must be a finite number")
        upper = 2.0 if field == "temperature" else 1.0
        if not 0 <= float(value) <= upper:
            raise QwenMultiRequestValidationError(
                f"{field} must be between 0 and {upper:g}")

    top_k = item.get("top_k", 0)
    if top_k is None:
        top_k = 0
    if (isinstance(top_k, bool) or not isinstance(top_k, int)
            or top_k < 0):
        raise QwenMultiRequestValidationError(
            "top_k must be a non-negative integer")
    repetition = item.get("repetition_penalty", 1.0)
    if (isinstance(repetition, bool)
            or not isinstance(repetition, (int, float))
            or not math.isfinite(float(repetition))
            or float(repetition) <= 0):
        raise QwenMultiRequestValidationError(
            "repetition_penalty must be a finite number > 0")
    explicit = any(item.get(field) is not None for field in (
        "temperature", "top_p", "top_k", "seed"))
    temperature = item.get("temperature")
    if temperature is None:
        temperature = 1.0 if explicit else 0.0
    try:
        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(1.0 if item.get("top_p") is None
                        else item["top_p"]),
            top_k=top_k,
            seed=item.get("seed"),
            repetition_penalty=float(repetition),
        )
    except (TypeError, ValueError) as error:
        raise QwenMultiRequestValidationError(str(error)) from error
    if not params.is_greedy:
        raise QwenMultiRequestValidationError(
            "multi-request layer-stationary serving currently requires "
            "deterministic/greedy sampling per request; categorical sampling "
            "uses a process-global MLX RNG and is not request-order independent")
    return params


def parse_batch_payload(
    payload: dict, config: QwenMultiRequestServerConfig,
) -> tuple[QwenMultiRequestItem, ...]:
    """Validate the wire shape before model lookup or tokenization."""
    unknown = sorted(set(payload) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise QwenMultiRequestValidationError(
            "unsupported multi-request fields: " + ", ".join(unknown))
    if payload.get("stream", False) is not False:
        raise QwenMultiRequestValidationError(
            "multi-request layer-stationary serving is non-streaming only")
    raw_items = payload.get("requests")
    if not isinstance(raw_items, list) or not raw_items:
        raise QwenMultiRequestValidationError(
            "requests must be a non-empty array")
    if len(raw_items) > config.max_requests:
        raise QwenMultiRequestValidationError(
            f"batch has {len(raw_items)} requests; configured maximum is "
            f"{config.max_requests}")

    ids: set[str] = set()
    total_output = 0
    parsed = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise QwenMultiRequestValidationError(
                f"requests[{index}] must be an object")
        unknown_item = sorted(set(raw) - _ALLOWED_ITEM)
        if unknown_item:
            raise QwenMultiRequestValidationError(
                f"requests[{index}] has unsupported fields: "
                + ", ".join(unknown_item))
        request_id = raw.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise QwenMultiRequestValidationError(
                f"requests[{index}].id must be a non-empty string")
        if request_id in ids:
            raise QwenMultiRequestValidationError(
                f"duplicate request id {request_id!r}")
        ids.add(request_id)
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise QwenMultiRequestValidationError(
                f"requests[{index}].prompt must be a non-empty string")
        max_tokens = raw.get("max_tokens")
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or not 1 <= max_tokens <= config.max_output_tokens):
            raise QwenMultiRequestValidationError(
                f"requests[{index}].max_tokens must be in "
                f"[1, {config.max_output_tokens}]")
        total_output += max_tokens
        stop_value = raw.get("stop")
        if stop_value is None:
            stop = ()
        elif isinstance(stop_value, str):
            stop = (stop_value,)
        elif (isinstance(stop_value, list)
              and all(isinstance(value, str) for value in stop_value)):
            stop = tuple(stop_value)
        else:
            raise QwenMultiRequestValidationError(
                f"requests[{index}].stop must be a string or list of strings")
        if any(not value for value in stop):
            raise QwenMultiRequestValidationError(
                f"requests[{index}].stop strings must be non-empty")
        parsed.append(QwenMultiRequestItem(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=(),
            max_tokens=max_tokens,
            sampling=_sampling_from_item(raw),
            stop=stop,
        ))
    if total_output > config.max_total_output_tokens:
        raise QwenMultiRequestValidationError(
            f"batch requests {total_output} total output tokens; configured "
            f"maximum is {config.max_total_output_tokens}")
    return tuple(parsed)


def _first_stop(text: str, stop: Sequence[str]) -> tuple[int, str] | None:
    matches = [
        (text.find(value), order, value)
        for order, value in enumerate(stop)
        if text.find(value) >= 0
    ]
    if not matches:
        return None
    index, _order, value = min(matches)
    return index, value


def _release_endpoint(engine, kv) -> None:
    release = getattr(engine, "_release_kv", None)
    if callable(release):
        release(kv)
        return
    release = getattr(kv, "release", None)
    if callable(release):
        release()


def unwrap_qwen_target(engine):
    """Skip speculative wrappers; this API schedules the exact target only."""
    current = engine
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        cfg = getattr(current, "cfg", None)
        if cfg is not None and getattr(cfg, "model_type", None) in _MODEL_TYPES:
            # A speculative wrapper delegates cfg. Prefer its concrete target
            # when one exists so no draft state is accidentally shared.
            target = getattr(current, "target", None)
            if target is None or target is current:
                return current
            current = target
            continue
        target = getattr(current, "target", None)
        if target is None or target is current:
            break
        current = target
    actual = getattr(getattr(current, "cfg", None), "model_type", None)
    raise QwenMultiRequestValidationError(
        "layer-stationary HTTP batching supports only qwen3_5/qwen3_5_moe "
        f"targets, got {actual!r}")


@dataclass
class _GenerationState:
    item: QwenMultiRequestItem
    kv: Any
    generated: list[int]
    done: bool = False
    termination_reason: str = "length"
    stop_sequence: str | None = None


def run_qwen_multi_request_batch(
    engine,
    items: Sequence[QwenMultiRequestItem],
    *,
    max_requests: int,
    bootstrap_generate: Callable[..., dict] | None = None,
) -> dict:
    """Generate bounded independent deterministic requests layer-major.

    Bootstrap uses the ordinary engine for each prompt and one sampled token.
    Its prompt endpoint is then forked into a private KV/KDA object.  Subsequent
    rounds share only target weight residency; arithmetic and sampling remain
    per request.
    """
    target = unwrap_qwen_target(engine)
    cfg = target.cfg
    rc = target.rc
    try:
        scheduler = QwenLayerStationaryScheduler(
            target, max_requests=max_requests)
    except ValueError as error:
        raise QwenMultiRequestValidationError(str(error)) from error
    items = tuple(items)
    if not items:
        raise QwenMultiRequestValidationError(
            "multi-request generation needs at least one request")
    if len(items) > max_requests:
        raise QwenMultiRequestValidationError(
            f"batch has {len(items)} requests; configured maximum is "
            f"{max_requests}")
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, QwenMultiRequestItem):
            raise QwenMultiRequestValidationError(
                "generation items must be QwenMultiRequestItem instances")
        if not item.request_id or item.request_id in ids:
            raise QwenMultiRequestValidationError(
                f"duplicate or empty request id {item.request_id!r}")
        ids.add(item.request_id)
        if not item.prompt_token_ids:
            raise QwenMultiRequestValidationError(
                f"request {item.request_id!r} has no validated prompt tokens")
        if (isinstance(item.max_tokens, bool)
                or not isinstance(item.max_tokens, int)
                or item.max_tokens <= 0):
            raise QwenMultiRequestValidationError(
                f"request {item.request_id!r} max_tokens must be positive")
        if not item.sampling.is_greedy:
            raise QwenMultiRequestValidationError(
                f"request {item.request_id!r} uses categorical sampling; "
                "only request-order-independent deterministic sampling is "
                "admitted")
    if getattr(rc, "max_kv_mb", 0):
        raise QwenMultiRequestValidationError(
            "layer-stationary HTTP batching requires resident plain KV; "
            "max_kv_mb paging is configured")
    if getattr(rc, "tool_pic_shared_pages", False):
        raise QwenMultiRequestValidationError(
            "layer-stationary HTTP batching does not support shared-page PIC KV")
    stepped_threshold = int(getattr(rc, "stepped_kv_threshold", 0) or 0)
    if stepped_threshold and any(
            len(item.prompt_token_ids) >= stepped_threshold for item in items):
        raise QwenMultiRequestValidationError(
            "layer-stationary HTTP batching requires plain KV; a prompt "
            "crosses the configured stepped_kv_threshold")

    generator = bootstrap_generate
    if generator is None:
        generator = getattr(
            target, "generate_with_memory_retry", target.generate)
    states: list[_GenerationState] = []
    round_telemetry = []
    batch_started = time.perf_counter()

    # Batch-local endpoints must neither alias nor populate the ordinary hot
    # or disk prompt caches. This policy toggle changes no model arithmetic and
    # is restored even on failed bootstrap.
    prompt_cache_fields = {
        "hot_prompt_kv": getattr(rc, "hot_prompt_kv", False),
        "prompt_kv_dir": getattr(rc, "prompt_kv_dir", ""),
    }
    try:
        rc.hot_prompt_kv = False
        rc.prompt_kv_dir = ""
        for item in items:
            # ``last_kv`` is only a diagnostic alias; retained hot slots own
            # their endpoints separately. Clear the alias so a failed
            # bootstrap cannot be mistaken for a fresh endpoint.
            target.last_kv = None
            try:
                result = generator(
                    item.prompt,
                    1,
                    stop=list(item.stop),
                    sampling=item.sampling,
                )
            except BaseException:
                failed_kv = getattr(target, "last_kv", None)
                if failed_kv is not None:
                    target.last_kv = None
                    _release_endpoint(target, failed_kv)
                    mx.clear_cache()
                raise
            source = getattr(target, "last_kv", None)
            if source is None:
                raise RuntimeError("Qwen batch bootstrap produced no KV endpoint")
            try:
                generated = [int(value) for value in result.get("tokens", ())]
                if len(generated) != 1:
                    raise RuntimeError(
                        "Qwen batch bootstrap must produce exactly one token")
                if not 0 <= generated[0] < int(cfg.vocab_size):
                    raise RuntimeError(
                        "Qwen batch bootstrap produced an out-of-vocabulary token")
                endpoint = fork_hybrid_kv_endpoint(source)
            except (TypeError, ValueError) as error:
                raise QwenMultiRequestValidationError(
                    "Qwen batch bootstrap endpoint is not an exact, resident "
                    f"plain hybrid KV: {error}") from error
            finally:
                if getattr(target, "last_kv", None) is source:
                    target.last_kv = None
                _release_endpoint(target, source)
            state = _GenerationState(item, endpoint, generated)
            termination = str(result.get("termination_reason", "length"))
            if item.max_tokens == 1 or termination != "length":
                state.done = True
                state.termination_reason = termination
                state.stop_sequence = result.get("stop_sequence")
            states.append(state)
    except BaseException:
        for state in states:
            _release_endpoint(target, state.kv)
        states.clear()
        mx.clear_cache()
        raise
    finally:
        rc.hot_prompt_kv = prompt_cache_fields["hot_prompt_kv"]
        rc.prompt_kv_dir = prompt_cache_fields["prompt_kv_dir"]

    try:
        while True:
            active = [state for state in states if not state.done]
            if not active:
                break
            batch = scheduler.advance(tuple(
                QwenLayerStationaryRequest(
                    state.item.request_id,
                    state.generated[-1],
                    state.kv,
                )
                for state in active
            ))
            round_telemetry.append(batch.telemetry)
            outputs = batch.by_request_id()
            for state in active:
                item = state.item
                token = sample(
                    outputs[item.request_id].logits,
                    item.sampling,
                    history=(*item.prompt_token_ids, *state.generated),
                )
                state.generated.append(token)
                decoded = target.tokenizer.decode(state.generated)
                stop_match = _first_stop(decoded, item.stop)
                if stop_match is not None:
                    state.done = True
                    state.termination_reason = "stop_sequence"
                    state.stop_sequence = stop_match[1]
                elif token in cfg.eos_token_ids:
                    state.done = True
                    state.termination_reason = "eos"
                elif len(state.generated) >= item.max_tokens:
                    state.done = True
                    state.termination_reason = "length"

        choices = []
        for state in states:
            text = target.tokenizer.decode(state.generated)
            stop_match = _first_stop(text, state.item.stop)
            if stop_match is not None:
                text = text[:stop_match[0]]
            choices.append({
                "id": state.item.request_id,
                "text": text,
                "tokens": list(state.generated),
                "finish_reason": (
                    "length" if state.termination_reason == "length"
                    else "stop"),
                "termination_reason": state.termination_reason,
                "stop_sequence": state.stop_sequence,
                "prompt_tokens": len(state.item.prompt_token_ids),
                "completion_tokens": len(state.generated),
                "sampling": state.item.sampling.profile,
            })

        def total(name: str, cast=int):
            return cast(sum(getattr(value, name) for value in round_telemetry))

        scheduler_seconds = total("wall_seconds", float)
        scheduled_tokens = total("request_tokens")
        telemetry = {
            "default_on": False,
            "request_count": len(states),
            "scheduler_rounds": len(round_telemetry),
            "scheduled_request_tokens": scheduled_tokens,
            "scheduler_wall_seconds": scheduler_seconds,
            "batch_wall_seconds": max(0.0, time.perf_counter() - batch_started),
            "scheduled_tokens_per_second": (
                scheduled_tokens / scheduler_seconds
                if scheduler_seconds > 0 else 0.0),
            "layer_page_get_calls": total("layer_page_get_calls"),
            "serial_equivalent_layer_page_get_calls": total(
                "serial_equivalent_layer_page_get_calls"),
            "layer_page_get_call_savings": total(
                "layer_page_get_call_savings"),
            "target_layer_page_bytes_read": total(
                "target_layer_page_bytes_read"),
            "total_cache_bytes_read": total("total_cache_bytes_read"),
            "total_weight_bytes_read": total("total_weight_bytes_read"),
            "cache_hits": total("cache_hits"),
            "cache_misses": total("cache_misses"),
            "cache_evictions": total("cache_evictions"),
            "cache_disk_seconds": total("cache_disk_seconds", float),
            "sampling_policy": "independent-deterministic",
            "cache_identity_policy": "private-kv-and-kda-per-request",
            "prompt_cache_policy": "disabled-during-bootstrap",
            "private_kv_endpoints": len({id(state.kv) for state in states}),
            "private_kda_endpoints": len({
                id(state.kv.kda_cache) for state in states}),
        }
        return {"choices": choices, "telemetry": telemetry}
    finally:
        for state in states:
            _release_endpoint(target, state.kv)
        states.clear()
        mx.clear_cache()
