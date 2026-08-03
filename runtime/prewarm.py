"""Declarative, fail-closed startup prefix warming for Kimi K3.

The configuration stores only a path and an expected SHA-256 for a captured
Responses request.  At startup we retain the tools plus leading
system/developer messages, derive a tokenizer-safe boundary before the first
dynamic item, and seed that exact endpoint through the ordinary engine path.
When durable hot-KV is enabled, the endpoint includes compressed MLA latents,
KDA recurrent state, and endpoint logits and therefore survives a restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path


_SCHEMA = "voom.startup-prefixes.v1"
_DERIVATION = "kimi-k3-responses-static-v1"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StartupPrefix:
    name: str
    model: str
    request_file: Path
    request_sha256: str
    derivation: str = _DERIVATION
    cache_namespace: str = "default"
    require_persistence: bool = False
    source_file: Path | None = None


def _strict_keys(value: dict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def load_startup_prefixes(paths) -> tuple[StartupPrefix, ...]:
    """Load one or more versioned prefix documents without model I/O."""
    result: list[StartupPrefix] = []
    names: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read startup-prefix file {path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"startup-prefix file {path} must be a JSON object")
        _strict_keys(document, {"schema", "prefixes"}, str(path))
        if document.get("schema") != _SCHEMA:
            raise ValueError(
                f"startup-prefix file {path} schema must be {_SCHEMA!r}")
        entries = document.get("prefixes")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"startup-prefix file {path} needs a non-empty prefixes list")
        for index, value in enumerate(entries):
            context = f"{path}:prefixes[{index}]"
            if not isinstance(value, dict):
                raise ValueError(f"{context} must be an object")
            _strict_keys(value, {
                "name", "model", "request_file", "request_sha256",
                "derivation", "cache_namespace", "require_persistence",
            }, context)
            name = value.get("name")
            model = value.get("model")
            request_file = value.get("request_file")
            digest = value.get("request_sha256")
            derivation = value.get("derivation", _DERIVATION)
            namespace = value.get("cache_namespace", "default")
            require_persistence = value.get("require_persistence", False)
            if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
                raise ValueError(f"{context}.name is not a safe identifier")
            if name in names:
                raise ValueError(f"duplicate startup-prefix name: {name}")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"{context}.model must be a non-empty string")
            if not isinstance(request_file, str) or not request_file:
                raise ValueError(f"{context}.request_file must be a non-empty string")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"{context}.request_sha256 must be lowercase SHA-256")
            if derivation != _DERIVATION:
                raise ValueError(f"{context}.derivation must be {_DERIVATION!r}")
            if (not isinstance(namespace, str) or not namespace
                    or len(namespace) > 128):
                raise ValueError(f"{context}.cache_namespace is invalid")
            if not isinstance(require_persistence, bool):
                raise ValueError(f"{context}.require_persistence must be boolean")
            request_path = Path(request_file).expanduser()
            if not request_path.is_absolute():
                request_path = path.parent / request_path
            names.add(name)
            result.append(StartupPrefix(
                name=name,
                model=model.strip(),
                request_file=request_path.resolve(),
                request_sha256=digest,
                derivation=derivation,
                cache_namespace=namespace,
                require_persistence=require_persistence,
                source_file=path,
            ))
    return tuple(result)


def _read_request(entry: StartupPrefix) -> tuple[bytes, dict]:
    raw = entry.request_file.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != entry.request_sha256:
        raise ValueError(
            f"startup prefix {entry.name!r} request identity mismatch: "
            f"{actual} != {entry.request_sha256}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"startup prefix {entry.name!r} request must be an object")
    return raw, value


def _leading_static_request(request: dict) -> tuple[dict, list, list]:
    items = request.get("input")
    if not isinstance(items, list):
        raise ValueError("K3 startup prefix requires list-valued Responses input")
    leading = []
    for item in items:
        if not isinstance(item, dict):
            break
        role = str(item.get("role", item.get("type", "")))
        if role not in ("system", "developer"):
            break
        leading.append(item)
    dynamic = items[len(leading):]
    if not leading:
        raise ValueError("K3 startup prefix has no leading system/developer items")
    if not dynamic:
        raise ValueError("K3 startup prefix has no dynamic request suffix")
    static = dict(request)
    static["input"] = leading
    return static, leading, dynamic


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _token_hash(values) -> str:
    return hashlib.sha256(json.dumps(
        list(values), separators=(",", ":")
    ).encode()).hexdigest()


def _longest_common_prefix(left, right) -> tuple[int, ...]:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return tuple(left[:index])


def _render_responses_request(engine, model_dir: Path, request: dict, mode: str):
    from .server import _prepare_chat_prompt
    from .toolcalls import (
        merge_leading_system_messages,
        normalize_messages,
        responses_input_to_messages,
    )

    messages = responses_input_to_messages(
        request.get("input", ""), request.get("instructions"))
    messages, image_sources = normalize_messages(messages)
    if image_sources:
        raise ValueError("startup prefix does not support image requests")
    messages = merge_leading_system_messages(messages)
    raw_tools = request.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ValueError("startup prefix tools must be a list")
    tools = [
        {"type": "function", "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {},
        }} if tool.get("type") == "function" else tool
        for tool in raw_tools
    ]
    return _prepare_chat_prompt(
        engine, model_dir, messages, "low", tools, raw_tools, mode, 1,
        enable_thinking=None, reasoning_requested=False,
    )[0]


def prepare_k3_static_prefix(
    engine, model_dir: Path, request: dict, mode: str,
    *, cache_namespace: str = "default",
):
    """Render and prove a static boundary without using dynamic content."""
    if getattr(engine.cfg, "model_type", "") != "kimi_k3":
        raise ValueError("kimi-k3-responses-static-v1 requires a Kimi K3 engine")
    static_request, leading, dynamic = _leading_static_request(request)
    full_prompt = _render_responses_request(engine, model_dir, request, mode)
    static_prompt = _render_responses_request(engine, model_dir, static_request, mode)
    marker = "assistant:"
    if not str(static_prompt).endswith(marker):
        raise ValueError(
            "K3 startup prefix expected the generic fallback generation marker")
    static_stem = str(static_prompt)[:-len(marker)]
    encodings = []
    for suffix in ("user: A", "user: Z"):
        encoded = engine.tokenizer.encode(static_stem + suffix)
        encodings.append(tuple(getattr(encoded, "ids", encoded)))
    prefix_ids = _longest_common_prefix(*encodings)
    full_ids = tuple(full_prompt.token_ids)
    if not prefix_ids or len(prefix_ids) >= len(full_ids):
        raise ValueError("derived startup prefix is empty or consumes the request")
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("derived startup prefix is not an exact source-request prefix")

    from .server import PreparedPrompt

    prompt = PreparedPrompt(
        static_stem, prefix_ids, cache_namespace=cache_namespace)
    metadata = {
        "derivation": _DERIVATION,
        "tokens": len(prefix_ids),
        "token_sha256": _token_hash(prefix_ids),
        "source_prompt_tokens": len(full_ids),
        "source_suffix_tokens": len(full_ids) - len(prefix_ids),
        "tools": len(request.get("tools") or []),
        "tools_sha256": _json_hash(request.get("tools") or []),
        "leading_messages": len(leading),
        "leading_messages_sha256": _json_hash(leading),
        "dynamic_messages": len(dynamic),
        "dynamic_messages_sha256": _json_hash(dynamic),
        "dynamic_content_used_in_seed": False,
    }
    return prompt, metadata


def run_startup_prefix(engine, model_dir: Path, mode: str,
                       entry: StartupPrefix) -> dict:
    """Seed or re-affirm one exact prefix and return content-free telemetry."""
    raw, request = _read_request(entry)
    if entry.require_persistence and getattr(
            engine, "_hot_kv_persist", None) is None:
        raise ValueError(
            f"startup prefix {entry.name!r} requires durable hot-KV, but "
            "no persistence directory is configured")
    prompt, metadata = prepare_k3_static_prefix(
        engine, model_dir, request, mode,
        cache_namespace=entry.cache_namespace)
    prior_slot = next((
        slot for slot in getattr(engine, "_hot_prompt_slots", ())
        if tuple(slot.tokens) == tuple(prompt.token_ids)
        and slot.cache_namespace == entry.cache_namespace
    ), None)

    from .sampler import SamplingParams

    started = time.perf_counter()
    generate = getattr(engine, "generate_with_memory_retry", engine.generate)
    result = generate(prompt, max_tokens=1, sampling=SamplingParams(temperature=0))
    wall = time.perf_counter() - started
    slot = next((
        item for item in reversed(getattr(engine, "_hot_prompt_slots", ()))
        if tuple(item.tokens) == tuple(prompt.token_ids)
        and item.cache_namespace == entry.cache_namespace
    ), None)
    if slot is None:
        raise RuntimeError(
            f"startup prefix {entry.name!r} was not retained by hot prompt KV")
    if entry.require_persistence and not tuple(slot.segment_chain):
        raise RuntimeError(
            f"startup prefix {entry.name!r} did not produce a durable checkpoint")
    if entry.require_persistence:
        durable_match = engine._hot_kv_persist.find_best_match(
            tuple(prompt.token_ids),
            int(engine.rc.hot_prompt_kv_chunk_size),
            cache_namespace=entry.cache_namespace,
        )
        if durable_match is None:
            raise RuntimeError(
                f"startup prefix {entry.name!r} exceeded retention or was not "
                "readable from the durable journal")
    stats = result.get("path_stats") or {}
    return {
        "name": entry.name,
        "model": entry.model,
        "mode": mode,
        "request_file": str(entry.request_file),
        "request_sha256": entry.request_sha256,
        "request_bytes": len(raw),
        "cache_namespace": entry.cache_namespace,
        "require_persistence": entry.require_persistence,
        "prefix": metadata,
        "wall_seconds": wall,
        "first_token_seconds": result.get("first_token_s"),
        "prefill_seconds": result.get("prefill_s"),
        "cache_source": stats.get("prompt_cache_source"),
        "cache_prefix_tokens": stats.get("prompt_cache_prefix_tokens", 0),
        "restored_before_prewarm": prior_slot is not None,
        "durable_segment_count": len(tuple(slot.segment_chain)),
        "persist_write_seconds": stats.get("hot_prompt_kv_persist_write_s", 0.0),
        "true_peak_metal_bytes": result.get("true_peak_metal_bytes"),
    }


def atomic_write_result(path: str | Path, value: dict) -> None:
    """Publish startup telemetry without leaving a valid-looking torn file."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
