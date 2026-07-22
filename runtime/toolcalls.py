"""OpenAI-compatible tool calling for the vOOM server (Phase 11 extension).

Three concerns, all template/parse-level — no engine changes:
- normalize_messages: accept OpenAI content-parts arrays (text/image_url).
  Image parts are preserved for a registered Qwen3-VL tower; text-only models
  return a clear 400 instead of silently dropping images.
- render-side: gpt-oss's official harmony template accepts `tools` natively;
  every other model gets a hermes-style system preamble asking for
  <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
- parse_tool_calls: extract tool calls from generated text for both the
  harmony commentary channel (`to=functions.NAME ... {json}`) and
  hermes-style <tool_call> blocks, returning OpenAI tool_calls dicts.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass


VISION_SPAN = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_SPAN = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass(frozen=True)
class VideoFrames:
    frames: tuple
    fps: float
    frame_indices: tuple[int, ...]
    duration_seconds: float

    @property
    def is_video(self) -> bool:
        return True


_SEARCH_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Deterministic capability capsules for high-value tool-domain paraphrases.
# This is deliberately compact and dependency-free: loading a second embedding
# model beside a 4B LLM on a 16-GB unified-memory host would erase much of the
# KV headroom this router is meant to recover. Both schema documents and model-
# authored queries are expanded symmetrically, so `bash` can retrieve a tool
# named `execute_command` even when the original description never says bash.
_TOOL_CAPABILITY_GROUPS = tuple(frozenset(group.split()) for group in (
    "bash sh shell cli terminal command commands exec execute subprocess",
    "file files filesystem folder folders directory directories dir path paths workspace",
    "browser web webpage url urls navigate navigation tab tabs chrome",
    "search find lookup discover discovery retrieve retrieval",
    "http https api network curl fetch download upload endpoint",
    "code repository repo git github commit branch pull merge diff patch",
    "email mail gmail outlook inbox",
    "calendar event events meeting meetings schedule scheduling appointment",
    "database databases db sql table tables row rows record records",
    "image images picture pictures photo photos screenshot screenshots vision",
    "document documents doc docs pdf spreadsheet spreadsheets excel sheet sheets",
))


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json_loads(value):
    return json.loads(value, parse_constant=_reject_json_constant)


def _search_words(value) -> list[str]:
    """Small dependency-free tokenizer for side-quest tool retrieval.

    Splitting camelCase as well as punctuation makes names such as
    ``mastra_workspace_execute_command`` match ordinary requests like
    "execute a command in the workspace". This is only a router over tool
    schemas; it is not used for model tokenization.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [w.lower() for w in _SEARCH_WORD_RE.findall(text) if len(w) > 1]


def _capability_words(value) -> list[str]:
    """Tokenize and enrich one tool/query into a semantic alias capsule."""
    words = _search_words(value)
    present = set(words)
    enriched = list(words)
    for group in _TOOL_CAPABILITY_GROUPS:
        if present & group:
            enriched.extend(sorted(group - present))
    return enriched


def _message_search_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", part.get("output", "")))
            for part in content if isinstance(part, dict)
        )
    calls = message.get("tool_calls") or []
    call_names = " ".join(
        str((call.get("function") or {}).get("name", ""))
        for call in calls if isinstance(call, dict)
    )
    return f"{content or ''} {call_names}"


def tool_search_capsule(tool: dict, *, max_chars: int = 6000) -> str:
    """Create a bounded semantic passage for one tool without examples/defaults.

    The passage is used only by the offline embedding encoder. It deliberately
    includes humanized name components, purpose, input field paths/types, and
    the same capability aliases as the lexical ranker. Arbitrary schema data is
    bounded so one pathological catalog entry cannot dominate encoder work.
    """
    fn = tool.get("function", tool) if isinstance(tool, dict) else {}
    if not isinstance(fn, dict):
        fn = {}
    name = str(fn.get("name", ""))
    description = str(fn.get("description", ""))[:2000]
    parameters = fn.get("parameters", fn.get("input_schema", {}))
    fields: list[str] = []

    def walk(schema, prefix: str = "", depth: int = 0) -> None:
        if len(fields) >= 80 or depth > 5 or not isinstance(schema, dict):
            return
        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = set(schema.get("required") or ())
            for key in sorted(properties, key=str):
                if len(fields) >= 80:
                    break
                child = properties[key]
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, dict):
                    kind = child.get("type", "value")
                    if isinstance(kind, list):
                        kind = "/".join(map(str, kind))
                    detail = str(child.get("description", ""))[:240]
                    enum = child.get("enum")
                    choices = ""
                    if isinstance(enum, list) and len(enum) <= 12:
                        choices = " choices " + ", ".join(map(str, enum))[:300]
                    fields.append(
                        f"{path}: {kind}{' required' if key in required else ''}"
                        f"{choices}{' - ' + detail if detail else ''}")
                    walk(child, path, depth + 1)
        for union in ("anyOf", "oneOf", "allOf"):
            variants = schema.get(union)
            if isinstance(variants, list):
                for child in variants[:8]:
                    walk(child, prefix, depth + 1)
        items = schema.get("items")
        if isinstance(items, dict):
            walk(items, f"{prefix}[]" if prefix else "item", depth + 1)

    walk(parameters)
    # Generic argument names such as ``query``, ``message``, or ``path`` are not
    # sufficient evidence that the *tool* belongs to a capability family. Use
    # only its identity/purpose to seed explicit aliases; BGE still sees the
    # structural fields and can infer their softer semantic relationships.
    alias_source = {"name": name, "description": description}
    aliases = sorted(
        set(_capability_words(alias_source)) - set(_search_words(alias_source)))
    human_name = " ".join(_search_words(name))
    parts = [
        f"Tool: {name}",
        f"Capability name: {human_name}",
        f"Purpose: {description}",
    ]
    if aliases:
        parts.append("Alternative capability terms: " + " ".join(aliases))
    if fields:
        parts.append("Inputs:\n" + "\n".join(fields))
    return "\n".join(parts)[:max_chars]


def semantic_tool_query(messages: list[dict], *, max_chars: int = 4000) -> str:
    """Use the most recent user-authored intent as the embedding query.

    Hidden discovery appends the model-authored catalog query as a user turn,
    so it naturally wins here. Avoid embedding a 20K system prompt or stale
    turns whose unrelated vocabulary would dilute the requested capability.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _message_search_text(message).strip()
            if text:
                return text[:max_chars]
    for message in reversed(messages):
        text = _message_search_text(message).strip()
        if text:
            return text[:max_chars]
    return ""


def pinned_tool_indices(tools: list[dict], messages: list[dict]) -> list[int]:
    """Tools that a fast-mode shortlist is not allowed to remove.

    Pin an exact tool name mentioned anywhere in the transcript and every tool
    already present in assistant call history.  The caller may exceed its soft
    shortlist limit when the transcript itself requires more tools than fit.
    """
    # System prompts often discuss ordinary words that happen to be very short
    # tool names (for example ``go``). Only a user-authored, identifier-bounded
    # name counts as an explicit request; call history is pinned independently.
    user_transcript = "\n".join(
        _message_search_text(m) for m in messages if m.get("role") == "user"
    )
    historical_names = {
        str((call.get("function") or {}).get("name", ""))
        for message in messages
        for call in (message.get("tool_calls") or [])
        if isinstance(call, dict)
    }
    pinned = []
    for i, tool in enumerate(tools):
        fn = tool.get("function", tool)
        name = str(fn.get("name", ""))
        if any(character in name for character in "_.-"):
            mentioned = bool(name) and re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])",
                user_transcript, flags=re.IGNORECASE) is not None
        else:
            # A bare function name may be an ordinary word (`make`, `go`,
            # `open`). Require explicit tool-selection syntax instead of
            # hard-pinning it from prose such as "make sure to paginate".
            mentioned = bool(name) and re.search(
                rf"(?:\b(?:use|call|invoke)\s+(?:the\s+)?|"
                rf"\b(?:tool|function)\s+(?:named\s+)?|`)"
                rf"{re.escape(name)}(?:`|\b)",
                user_transcript, flags=re.IGNORECASE) is not None
        if name in historical_names or mentioned:
            pinned.append(i)
    return pinned


def explicit_tool_namespaces(
    tools: list[dict], messages: list[dict],
) -> tuple[str, ...]:
    """Return explicitly user-named ``plugin__namespace__`` providers.

    Hidden catalog search deliberately ranks against a model-authored compact
    query instead of a huge transcript. A small router can omit the provider
    word while paraphrasing the capability (for example, "list files, movies,
    and TV" for an explicit Plex request), which lets a generic workspace tool
    outrank the intended plugin. Preserve only namespace tokens that are both
    present in the offered catalog and explicitly named by a user; no fuzzy
    provider inference is performed.
    """
    user_text = "\n".join(
        _message_search_text(message)
        for message in messages if message.get("role") == "user")
    offered = set()
    for tool in tools:
        fn = tool.get("function", tool)
        name = str(fn.get("name", "")) if isinstance(fn, dict) else ""
        match = re.match(r"^plugin__([A-Za-z0-9-]+)__", name)
        if match:
            offered.add(match.group(1).lower())
    return tuple(sorted(
        namespace for namespace in offered
        if re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(namespace)}"
            rf"(?![A-Za-z0-9_-])",
            user_text, flags=re.IGNORECASE)))


def _lexical_tool_scores(
        tools: list[dict], messages: list[dict]) -> tuple[list[str], list[float]]:
    """Return the pre-existing deterministic BM25-like score per tool."""
    if not tools:
        return [], []

    docs: list[Counter] = []
    names: list[str] = []
    name_words: list[set[str]] = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = str(fn.get("name", ""))
        names.append(name)
        # The large 4x name boost is for literal identity words only. Expanding
        # `root_folder` into every filesystem alias made each synthetic alias
        # look like another exact function-name match. Likewise, schema fields
        # remain searchable but do not seed capability expansion: a generic
        # `path` parameter does not turn a Plex query tool into a filesystem
        # executor. This mirrors tool_search_capsule's semantic boundary.
        name_words.append(set(_search_words(name)))
        identity = {
            "name": name,
            "description": fn.get("description", ""),
        }
        parameters = fn.get("parameters", fn.get("input_schema", {}))
        docs.append(Counter([
            *_capability_words(identity), *_search_words(parameters)]))

    df = Counter()
    for doc in docs:
        df.update(doc.keys())
    n_docs = len(docs)

    weighted_query = Counter()
    transcript_parts = []
    historical_names = set()
    for message in messages:
        text = _message_search_text(message)
        transcript_parts.append(text.lower())
        role = message.get("role")
        weight = 6 if role == "user" else (1 if role == "system" else 2)
        for word in _capability_words(text):
            weighted_query[word] += weight
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                historical_names.add(str((call.get("function") or {}).get("name", "")))
    transcript = "\n".join(transcript_parts)

    scores = []
    for name, nws, doc in zip(names, name_words, docs):
        score = 0.0
        for word, q_weight in weighted_query.items():
            tf = min(doc.get(word, 0), 4)
            if not tf:
                continue
            idf = math.log((n_docs + 1) / (df[word] + 1)) + 1.0
            score += q_weight * idf * (1.0 + math.log(tf))
            if word in nws:
                score += 4.0 * q_weight * idf
        normalized_name = name.lower()
        if normalized_name and normalized_name in transcript:
            score += 100_000.0
        if name in historical_names:
            score += 1_000_000.0
        scores.append(score)
    return names, scores


def rank_tool_indices(
        tools: list[dict], messages: list[dict], *,
        use_embeddings: bool = False, return_metadata: bool = False):
    """Rank tool schemas for the explicitly lossy fast-mode shortlist.

    The base is deterministic BM25-like lexical/alias retrieval. When requested
    and an offline cache is complete, normalized BGE cosine similarity is fused
    with that base score. Names explicitly mentioned in the transcript and
    tools present in call history get strong boosts;
    ``pinned_tool_indices`` is the separate hard-availability rule.
    Canonical function-name order breaks ties.  Request order is deliberately
    not a ranking signal: clients commonly rebuild the same tool catalog from
    maps/plug-ins in a different order, and a tie at the shortlist boundary
    must not silently select a different set merely because of that permutation.

    It deliberately returns a ranking rather than silently dropping anything;
    lossless endpoints never call it, and fast endpoints report the requested
    and selected counts in response metadata.
    """
    names, scores = _lexical_tool_scores(tools, messages)
    metadata = {
        "tool_retrieval_profile": "hybrid-lexical-capability-v1",
        "tool_embedding_status": "disabled",
    }
    if use_embeddings and tools:
        try:
            from .tool_embeddings import (
                EmbeddingConfig, ToolEmbeddingError, embeddings_enabled,
                hybrid_scores)
            if embeddings_enabled():
                scores, embedding_meta = hybrid_scores(
                    [tool_search_capsule(tool) for tool in tools],
                    semantic_tool_query(messages), scores,
                    config=EmbeddingConfig.from_env())
                metadata.update(embedding_meta)
                if embedding_meta.get("tool_embedding_status") == "hybrid":
                    metadata["tool_retrieval_profile"] = (
                        "hybrid-bge-lexical-capability-v1")
            else:
                if os.environ.get(
                        "VMODEL_TOOL_EMBEDDINGS_REQUIRED", "0") == "1":
                    raise ToolEmbeddingError(
                        "verified offline tool embedding cache is unavailable")
                metadata["tool_embedding_status"] = "unavailable"
        except (OSError, ValueError, ToolEmbeddingError) as error:
            if os.environ.get("VMODEL_TOOL_EMBEDDINGS_REQUIRED", "0") == "1":
                raise
            metadata.update({
                "tool_embedding_status": "fallback",
                "tool_embedding_fallback": str(error)[:160],
            })

    scored = [(-score, names[i], i) for i, score in enumerate(scores)]
    scored.sort()
    ranking = [i for _, _, i in scored]
    return (ranking, metadata) if return_metadata else ranking


def canonical_tool_indices(tools: list[dict]) -> list[int]:
    """Return a deterministic prompt order for one function-tool catalog.

    Transformer KV is causal, so independently cached tool-schema KV blocks
    cannot be concatenated or reordered exactly.  Sorting the *prompt copy*
    instead makes equal catalogs render to the same token sequence and lets the
    ordinary exact prefix cache do the safe reuse.  The server retains request
    order separately for wire responses.

    Function names are the stable protocol identity.  Duplicate names are
    ambiguous for execution and would make canonical tie-breaking depend on
    request order, so reject them rather than silently choosing one schema.
    """
    named: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool {index} must be an object")
        fn = tool.get("function", tool)
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError("every tool must have a non-empty function name")
        if name in seen:
            raise ValueError(f"duplicate tool function name: {name!r}")
        seen.add(name)
        named.append((name, index))
    named.sort()
    return [index for _, index in named]


_SCHEMA_PROSE_KEYS = {"description", "title", "$comment", "examples", "example", "default"}
# These JSON-Schema objects are maps keyed by user-controlled property/definition
# names.  A key literally named ``title`` or ``description`` is data here, not an
# annotation, and must survive compaction.  Their VALUES are schemas, so prose
# pruning resumes one level below the map.
_SCHEMA_NAMED_MAP_KEYS = {
    "properties", "patternProperties", "$defs", "definitions",
    "dependentSchemas", "dependentRequired", "dependencies", "mapping",
}


def effective_tool_prompt_schema(tool: dict) -> dict:
    """Detached, prose-preserving tool copy with real optionality applied.

    Provider adapters may put every property in JSON Schema ``required`` and
    separately use ``x-optional`` to carry the actual callable contract.  The
    grammar path already honors that extension.  Prompting the model with the
    contradictory wire form made Qwen spend hundreds of tokens debating how
    to fill 25 supposedly-required nullable fields instead of calling Plex.
    Preserve every description/enum while applying the same effective schema
    the validator sees; the original wire tool remains untouched.
    """
    fn = tool.get("function")
    wrapped = fn is not None
    source = fn if wrapped else tool
    normalized_fn = deepcopy(source)
    parameter_key = (
        "parameters" if "parameters" in normalized_fn else "input_schema")
    if parameter_key in normalized_fn:
        from .structured import effective_tool_schema

        normalized_fn[parameter_key] = effective_tool_schema(
            normalized_fn[parameter_key] or {"type": "object"})
    if wrapped:
        return {**tool, "function": normalized_fn}
    return normalized_fn


def compact_tool_schema(tool: dict) -> dict:
    """Side-quest prompt copy of a tool with nested schema prose removed.

    The function name and top-level description are preserved, as are every
    structural constraint used to validate a call (types, properties,
    required, enums, unions, numeric/string bounds, and additionalProperties).
    Only parameter-level prose/examples/default annotations are removed. The
    caller still retains the original schema for execution and response echoing.

    This intentionally lossy transform is never used by the lossless endpoint.
    It is also canonical: changes confined to stripped annotations no longer
    invalidate an otherwise identical prompt cache entry.
    """
    normalized = effective_tool_prompt_schema(tool)
    fn = normalized.get("function")
    wrapped = fn is not None
    source = fn if wrapped else normalized

    def prune(value, *, named_map: bool = False):
        if isinstance(value, dict):
            if named_map:
                return {k: prune(v) for k, v in value.items()}
            out = {}
            for k, v in value.items():
                if k in _SCHEMA_PROSE_KEYS:
                    continue
                # enum/const contain arbitrary JSON DATA, not nested schemas.
                # A literal object may legally have a field named ``title`` or
                # ``default``; pruning that field changes the validation set.
                out[k] = (deepcopy(v) if k in ("enum", "const") else
                          prune(v, named_map=k in _SCHEMA_NAMED_MAP_KEYS))
            return out
        if isinstance(value, list):
            return [prune(v) for v in value]
        return value

    compact_fn = dict(source)
    parameter_key = "parameters" if "parameters" in compact_fn else "input_schema"
    if parameter_key in compact_fn:
        compact_fn[parameter_key] = prune(compact_fn[parameter_key])
    if wrapped:
        return {**tool, "function": compact_fn}
    return compact_fn


def expand_image_pad_tokens(token_ids: list[int], image_token_id: int,
                            merged_patch_counts: list[int]) -> list[int]:
    """Expand one image placeholder per image to its merged-patch token count.

    Kept here rather than in the MLX vision module so count/error behavior has a
    dependency-free regression test. A mismatch is rejected instead of silently
    dropping an image or indexing beyond the available grids.
    """
    out: list[int] = []
    image_index = 0
    for token in token_ids:
        if token != image_token_id:
            out.append(token)
            continue
        if image_index >= len(merged_patch_counts):
            raise ValueError("prompt has more image placeholders than supplied images")
        count = int(merged_patch_counts[image_index])
        if count <= 0:
            raise ValueError("image merged-patch token count must be positive")
        out.extend([token] * count)
        image_index += 1
    if image_index != len(merged_patch_counts):
        raise ValueError("supplied images exceed prompt image placeholders")
    return out


def normalize_messages(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Flatten content-parts to plain strings across all three supported
    protocols' text/image block shapes, not just OpenAI chat/completions':
      text:  {"type": "text", ...}            (OpenAI chat, Anthropic)
             {"type": "input_text", ...}       (OpenAI Responses input)
             {"type": "output_text", ...}      (OpenAI Responses history)
      image: {"type": "image_url", "image_url": {"url": ...}}   (OpenAI chat)
             {"type": "input_image", "image_url": ...}          (OpenAI Responses)
             {"type": "image", "source": {"type": "base64"|"url", ...}}  (Anthropic)
    Image parts are replaced IN PLACE by the Qwen vision-token span (so
    templates keep image position) and their sources returned in order.
    Tool-result messages pass through."""
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    out, images = [], []
    for message_index, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"message {message_index} must be an object")
        if not isinstance(m.get("role", "user"), str):
            raise ValueError(f"message {message_index} role must be a string")
        c = m.get("content")
        if isinstance(c, list):
            texts = []
            for part_index, part in enumerate(c):
                if not isinstance(part, dict):
                    raise ValueError(
                        f"message {message_index} content block {part_index} "
                        "must be an object")
                t = part.get("type")
                if t in ("text", "input_text", "output_text"):
                    value = part.get("text", "")
                    if not isinstance(value, str):
                        raise ValueError(
                            f"message {message_index} text block {part_index} "
                            "must contain a string")
                    texts.append(value)
                elif t == "refusal":
                    value = part.get("refusal", "")
                    if not isinstance(value, str):
                        raise ValueError(
                            f"message {message_index} refusal block {part_index} "
                            "must contain a string")
                    texts.append(value)
                elif t in ("image_url", "input_image"):
                    url = part.get("image_url", {})
                    url = url.get("url") if isinstance(url, dict) else url
                    if not isinstance(url, str) or not url:
                        raise ValueError(
                            f"message {message_index} image block {part_index} "
                            "has no image URL")
                    images.append(url or part.get("image", ""))
                    texts.append(VISION_SPAN)
                elif t == "image":
                    src = part.get("source", {}) or {}
                    if not isinstance(src, dict):
                        raise ValueError(
                            f"message {message_index} image source must be an object")
                    if src.get("type") == "base64":
                        data = src.get("data", "")
                        if not isinstance(data, str) or not data:
                            raise ValueError(
                                f"message {message_index} image has no base64 data")
                        images.append(
                            f"data:{src.get('media_type', 'image/png')};base64,{data}")
                    else:
                        url = src.get("url", "")
                        if not isinstance(url, str) or not url:
                            raise ValueError(
                                f"message {message_index} image has no URL")
                        images.append(url)
                    texts.append(VISION_SPAN)
                elif t in ("video_url", "input_video"):
                    url = part.get("video_url", {})
                    url = url.get("url") if isinstance(url, dict) else url
                    if not isinstance(url, str) or not url:
                        raise ValueError(
                            f"message {message_index} video block {part_index} "
                            "has no video URL")
                    images.append({"type": "video", "source": url})
                    texts.append(VIDEO_SPAN)
                elif t == "video":
                    src = part.get("source", {}) or {}
                    if not isinstance(src, dict):
                        raise ValueError(
                            f"message {message_index} video source must be an object")
                    if src.get("type") == "base64":
                        data = src.get("data", "")
                        if not isinstance(data, str) or not data:
                            raise ValueError(
                                f"message {message_index} video has no base64 data")
                        images.append({"type": "video", "source":
                            f"data:{src.get('media_type', 'video/mp4')};base64,{data}"})
                    else:
                        url = src.get("url", "")
                        if not isinstance(url, str) or not url:
                            raise ValueError(
                                f"message {message_index} video has no URL")
                        images.append({"type": "video", "source": url})
                    texts.append(VIDEO_SPAN)
                else:
                    raise ValueError(
                        f"unsupported content block type at message "
                        f"{message_index}[{part_index}]: {t!r}")
            m = {**m, "content": "".join(texts)}
        elif c is not None and not isinstance(c, str):
            raise ValueError(
                f"message {message_index} content must be a string or block array")
        out.append(m)
    return out, images


def responses_input_to_messages(input_val, instructions: str | None = None) -> list[dict]:
    """OpenAI Responses API `input` (a plain string, OR a list of items —
    message-like {"role","content"} items, "function_call" items = a prior
    assistant tool call, "function_call_output" items = a tool result) ->
    canonical chat-style messages compatible with normalize_messages() and
    parse_tool_calls()'s round trip."""
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses instructions must be a string")
    if isinstance(input_val, str):
        msgs = [{"role": "user", "content": input_val}]
    elif isinstance(input_val, list):
        msgs = []
        for item_index, item in enumerate(input_val):
            if not isinstance(item, dict):
                raise ValueError(f"Responses input item {item_index} must be an object")
            t = item.get("type")
            if t == "function_call":
                call = {"id": item.get("call_id", ""), "type": "function",
                        "function": {"name": item.get("name", ""),
                                     "arguments": item.get("arguments", "{}")}}
                # A Responses output can contain a plain assistant message and
                # one or more sibling function_call items from the SAME model
                # turn. Rehydrating each sibling as a new assistant message
                # inserts a false ChatML turn boundary, changes the prompt, and
                # defeats exact post-generation prefix reuse.
                if msgs and msgs[-1].get("role") == "assistant":
                    msgs[-1].setdefault("tool_calls", []).append(call)
                else:
                    msgs.append({"role": "assistant", "content": None,
                                 "tool_calls": [call]})
            elif t == "function_call_output":
                out = item.get("output")
                if isinstance(out, list):
                    validated_output = []
                    for block_index, block in enumerate(out):
                        if not isinstance(block, dict):
                            raise ValueError(
                                f"Responses function output block {block_index} "
                                "must be an object")
                        block_type = block.get("type")
                        if block_type in ("input_text", "text", "output_text"):
                            text = block.get("text")
                            if not isinstance(text, str):
                                raise ValueError(
                                    f"Responses function output text block "
                                    f"{block_index} must contain a string")
                            validated_output.append({"type": "text", "text": text})
                        elif block_type in ("input_image", "image_url"):
                            validated_output.append(block)
                        else:
                            raise ValueError(
                                f"unsupported Responses function output block "
                                f"type at index {block_index}: {block_type!r}")
                    out = validated_output
                elif not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)
                msgs.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": out})
            elif t in (None, "message"):
                msgs.append({"role": item.get("role", "user"), "content": item.get("content")})
            else:
                raise ValueError(
                    f"unsupported Responses input item type at index "
                    f"{item_index}: {t!r}")
    else:
        raise ValueError("Responses input must be a string or an array")
    if instructions:
        msgs = [{"role": "system", "content": instructions}] + msgs
    return msgs


def merge_leading_system_messages(messages: list[dict]) -> list[dict]:
    """Collapse a leading run of role="system" messages into one.

    Real chat templates (Qwen's included) hard-reject any system message
    that isn't first in the conversation
    (`raise_exception('System message must be at the beginning.')`), but
    real clients routinely produce more than one leading system turn:
    a Responses API caller combining top-level `instructions` with an
    explicit system item in `input`, or an agent harness appending a
    second system turn for separate instructions (2026-07-20,
    live-confirmed with a real Codex/Kai request -- two leading system
    items, a main system prompt and a distinct "WORKING_MEMORY_SYSTEM_
    INSTRUCTION" turn, no top-level `instructions` involved at all).
    Applied once, uniformly, after building canonical messages for any
    of the three protocols, rather than special-cased per producer."""
    if not messages or messages[0].get("role") != "system":
        return messages
    end = 1
    while end < len(messages) and messages[end].get("role") == "system":
        end += 1
    if end == 1:
        return messages
    merged_content = None
    for message in messages[:end]:
        content = message.get("content")
        parts = content if isinstance(content, list) else (
            [{"type": "text", "text": content}] if content else [])
        merged_content = parts if merged_content is None else merged_content + parts
    if merged_content and all(
            isinstance(part, dict) and part.get("type") == "text"
            for part in merged_content):
        merged_content = "\n\n".join(part["text"] for part in merged_content)
    return [{**messages[0], "content": merged_content}] + messages[end:]


def anthropic_messages_to_canonical(messages: list[dict], system=None) -> list[dict]:
    """Anthropic Messages API `messages` (each message's `content` may be a
    plain string OR a list mixing text/image/tool_use/tool_result blocks) ->
    canonical chat-style messages. `system` is Anthropic's separate
    top-level field (string, or a list of {"type":"text","text":...} blocks
    per the 2026 API), not part of `messages`."""
    if not isinstance(messages, list):
        raise ValueError("Anthropic messages must be an array")
    msgs = []
    if system is not None:
        if isinstance(system, str):
            sys_text = system
        elif isinstance(system, list):
            texts = []
            for block_index, block in enumerate(system):
                if not isinstance(block, dict) or block.get("type") != "text":
                    raise ValueError(
                        "Anthropic system blocks must all be text blocks")
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"Anthropic system text block {block_index} must "
                        "contain a string")
                texts.append(text)
            sys_text = "".join(texts)
        else:
            raise ValueError("Anthropic system must be a string or text-block array")
        msgs.append({"role": "system", "content": sys_text})

    for message_index, m in enumerate(messages):
        if not isinstance(m, dict):
            raise ValueError(f"Anthropic message {message_index} must be an object")
        role, content = m.get("role", "user"), m.get("content")
        if role not in ("user", "assistant"):
            raise ValueError(
                f"Anthropic message {message_index} role must be user or assistant")
        if not isinstance(content, list):
            msgs.append({"role": role, "content": content})
            continue
        tool_calls, tool_results, kept = [], [], []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValueError(
                    f"Anthropic message {message_index} block {block_index} "
                    "must be an object")
            bt = block.get("type")
            if bt == "tool_use":
                name = block.get("name")
                tool_input = block.get("input", {})
                if not isinstance(name, str) or not name:
                    raise ValueError("Anthropic tool_use name must be a non-empty string")
                if not isinstance(tool_input, dict):
                    raise ValueError("Anthropic tool_use input must be an object")
                tool_calls.append({"id": block.get("id", ""), "type": "function",
                                   "function": {"name": name,
                                               "arguments": json.dumps(tool_input)}})
            elif bt == "tool_result":
                c = block.get("content", "")
                if isinstance(c, list):
                    validated_content = []
                    for result_index, value in enumerate(c):
                        if not isinstance(value, dict):
                            raise ValueError(
                                f"Anthropic tool_result block {result_index} "
                                "must be an object")
                        if value.get("type") == "text":
                            text = value.get("text")
                            if not isinstance(text, str):
                                raise ValueError(
                                    f"Anthropic tool_result text block "
                                    f"{result_index} must contain a string")
                            validated_content.append(value)
                        elif value.get("type") == "image":
                            validated_content.append(value)
                        else:
                            raise ValueError(
                                f"unsupported Anthropic tool_result block type "
                                f"at index {result_index}: {value.get('type')!r}")
                    c = validated_content
                elif not isinstance(c, str):
                    raise ValueError(
                        "Anthropic tool_result content must be a string or block array")
                is_error = block.get("is_error", False)
                if not isinstance(is_error, bool):
                    raise ValueError("Anthropic tool_result is_error must be a boolean")
                if is_error:
                    if isinstance(c, list):
                        c = [{"type": "text", "text": "[tool error] "}, *c]
                    else:
                        c = f"[tool error] {c}"
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": c})
            else:
                kept.append(block)  # text/image blocks: let normalize_messages flatten these
        if tool_calls:
            msgs.append({"role": "assistant", "content": kept or None, "tool_calls": tool_calls})
        elif kept:
            msgs.append({"role": role, "content": kept})
        msgs.extend(tool_results)
    return msgs


def _tool_call_id(call: dict) -> str:
    return str(call.get("id") or call.get("call_id") or "")


def _merge_assistant_content(left, right):
    if left in (None, "", []):
        return right
    if right in (None, "", []):
        return left
    if isinstance(left, list) and isinstance(right, list):
        return [*left, *right]
    if isinstance(left, str) and isinstance(right, str):
        return left + right
    return [left, right]


def canonicalize_tool_history(messages: list[dict]) -> list[dict]:
    """Normalize split/out-of-order tool history without changing semantics.

    Parallel tool results are ordered by their assistant turn's declared call
    IDs, not nondeterministic completion order. Adjacent assistant output items
    split by Responses-style harnesses are merged. A contiguous result batch
    immediately preceding its matching assistant call is repaired as the same
    turn. One id-less call/result pair is repaired deterministically; ambiguous,
    duplicate, or orphan IDs fail closed rather than contaminating the prompt.
    """
    coalesced: list[dict] = []
    for message_index, original in enumerate(messages):
        if not isinstance(original, dict):
            raise ValueError(f"message {message_index} must be an object")
        message = dict(original)
        raw_calls = message.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, list):
                raise ValueError(
                    f"message {message_index} tool_calls must be an array")
            if raw_calls and message.get("role") != "assistant":
                raise ValueError("tool_calls require role='assistant'")
            normalized_calls = []
            for call_index, call in enumerate(raw_calls):
                if not isinstance(call, dict):
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "must be an object")
                if call.get("type") not in (None, "function"):
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "must have type='function'")
                source = call.get("function", call)
                if not isinstance(source, dict):
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "function must be an object")
                name = source.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "must have a non-empty function name")
                arguments = source.get("arguments", "{}")
                try:
                    parsed_arguments = (_strict_json_loads(arguments)
                                        if isinstance(arguments, str)
                                        else arguments)
                    canonical_arguments = json.dumps(
                        parsed_arguments, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "arguments must contain valid JSON") from error
                raw_id = (call.get("id") if "id" in call
                          else call.get("call_id"))
                if raw_id is not None and not isinstance(raw_id, str):
                    raise ValueError(
                        f"message {message_index} tool call {call_index} "
                        "id must be a string")
                normalized_calls.append({
                    **call,
                    "id": raw_id or "",
                    "type": "function",
                    "function": {**source, "name": name,
                                 "arguments": canonical_arguments},
                })
            message["tool_calls"] = normalized_calls
        if (coalesced and message.get("role") == "assistant"
                and coalesced[-1].get("role") == "assistant"
                and (message.get("tool_calls") or coalesced[-1].get("tool_calls"))):
            previous = coalesced[-1]
            previous["content"] = _merge_assistant_content(
                previous.get("content"), message.get("content"))
            previous["tool_calls"] = [
                *(previous.get("tool_calls") or []),
                *(message.get("tool_calls") or []),
            ]
            continue
        coalesced.append(message)

    # Repair the only unambiguous missing-ID shape. Prefer an adjacent result's
    # explicit ID so the prompt preserves the harness's externally-visible
    # identity; otherwise assign a stable prompt-local ID. Parallel missing IDs
    # cannot be associated after nondeterministic completion and must fail.
    for index, message in enumerate(coalesced):
        calls = message.get("tool_calls") or []
        missing = [call for call in calls if not _tool_call_id(call)]
        if not missing:
            continue
        if len(calls) != 1:
            raise ValueError("parallel assistant tool calls require non-empty ids")
        adjacent_ids = set()
        left = index - 1
        while left >= 0 and coalesced[left].get("role") == "tool":
            result_id = coalesced[left].get("tool_call_id")
            if result_id:
                if not isinstance(result_id, str):
                    raise ValueError("tool result id must be a string")
                adjacent_ids.add(result_id)
            left -= 1
        right = index + 1
        while right < len(coalesced) and coalesced[right].get("role") == "tool":
            result_id = coalesced[right].get("tool_call_id")
            if result_id:
                if not isinstance(result_id, str):
                    raise ValueError("tool result id must be a string")
                adjacent_ids.add(result_id)
            right += 1
        if len(adjacent_ids) > 1:
            raise ValueError("id-less tool call has ambiguous adjacent result ids")
        calls[0]["id"] = (next(iter(adjacent_ids)) if adjacent_ids
                          else f"call_repaired_{index}")

    # Associate an id-less result only with one adjacent single-call turn.
    for index, message in enumerate(coalesced):
        if message.get("role") != "tool":
            continue
        result_id = message.get("tool_call_id")
        if result_id is not None and not isinstance(result_id, str):
            raise ValueError("tool result id must be a string")
        if result_id:
            continue
        candidates = set()
        left = index - 1
        while left >= 0 and coalesced[left].get("role") == "tool":
            left -= 1
        if left >= 0:
            calls = coalesced[left].get("tool_calls") or []
            if coalesced[left].get("role") == "assistant" and len(calls) == 1:
                candidates.add(_tool_call_id(calls[0]))
        right = index + 1
        while right < len(coalesced) and coalesced[right].get("role") == "tool":
            right += 1
        if right < len(coalesced):
            calls = coalesced[right].get("tool_calls") or []
            if coalesced[right].get("role") == "assistant" and len(calls) == 1:
                candidates.add(_tool_call_id(calls[0]))
        candidates.discard("")
        if len(candidates) != 1:
            raise ValueError("id-less tool result cannot be associated unambiguously")
        message["tool_call_id"] = next(iter(candidates))

    declared_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in coalesced:
        for call in message.get("tool_calls") or []:
            call_id = _tool_call_id(call)
            if call_id in declared_ids:
                raise ValueError(f"duplicate assistant tool call id: {call_id!r}")
            declared_ids.add(call_id)
        if message.get("role") == "tool":
            result_id = message.get("tool_call_id") or ""
            if result_id in result_ids:
                raise ValueError(f"duplicate tool result id: {result_id!r}")
            result_ids.add(result_id)
    orphaned = sorted(result_ids - declared_ids)
    if orphaned:
        raise ValueError(f"orphan tool result id: {orphaned[0]!r}")

    def ordered_results(assistant: dict, results: list[dict]) -> list[dict]:
        calls = assistant.get("tool_calls") or []
        call_ids = [_tool_call_id(call) for call in calls]
        nonempty = [value for value in call_ids if value]
        duplicate_calls = sorted({value for value in nonempty
                                  if nonempty.count(value) > 1})
        if duplicate_calls:
            raise ValueError(
                f"duplicate assistant tool call id: {duplicate_calls[0]!r}")
        rank = {value: index for index, value in enumerate(call_ids) if value}
        seen_results: set[str] = set()
        indexed = []
        for request_index, result in enumerate(results):
            result_id = str(result.get("tool_call_id") or "")
            if result_id and result_id in seen_results:
                raise ValueError(f"duplicate tool result id: {result_id!r}")
            if result_id:
                seen_results.add(result_id)
            indexed.append((rank.get(result_id, len(rank) + request_index),
                            request_index, result))
        return [result for _rank, _request_index, result in sorted(indexed)]

    normalized: list[dict] = []
    index = 0
    while index < len(coalesced):
        message = coalesced[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            end = index + 1
            while end < len(coalesced) and coalesced[end].get("role") == "tool":
                end += 1
            normalized.append(message)
            normalized.extend(ordered_results(message, coalesced[index + 1:end]))
            index = end
            continue
        if message.get("role") == "tool":
            end = index
            while end < len(coalesced) and coalesced[end].get("role") == "tool":
                end += 1
            if end < len(coalesced):
                assistant = coalesced[end]
                call_ids = {_tool_call_id(call)
                            for call in assistant.get("tool_calls") or []}
                result_ids = {str(result.get("tool_call_id") or "")
                              for result in coalesced[index:end]}
                if (assistant.get("role") == "assistant" and call_ids
                        and result_ids and "" not in result_ids
                        and result_ids <= call_ids):
                    normalized.append(assistant)
                    normalized.extend(ordered_results(
                        assistant, coalesced[index:end]))
                    index = end + 1
                    continue
        normalized.append(message)
        index += 1
    return normalized


def load_image(src: str):
    """Bounded data URI, local path, or HTTP(S) URL -> eager PIL image."""
    import base64
    import binascii
    import io
    import os
    from pathlib import Path

    from PIL import Image

    if not isinstance(src, str) or not src:
        raise ValueError("image source must be a non-empty string")
    try:
        max_bytes_mb = int(os.environ.get("VMODEL_MAX_IMAGE_BYTES_MB", "25"))
        max_pixels = int(os.environ.get(
            "VMODEL_MAX_SOURCE_IMAGE_PIXELS", "64000000"))
    except ValueError as error:
        raise ValueError("image size limits must be integers") from error
    if max_bytes_mb <= 0 or max_pixels <= 0:
        raise ValueError("image size limits must be positive")
    max_bytes = max_bytes_mb * 1024 * 1024

    if src.startswith("data:"):
        header, separator, payload = src.partition(",")
        if (not separator or not header.lower().startswith("data:image/")
                or ";base64" not in header.lower()):
            raise ValueError("image data URI must be image/*;base64")
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("image data URI contains invalid base64") from error
    elif src.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(src, timeout=20) as r:
            declared = r.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise ValueError(
                        f"remote image exceeds {max_bytes_mb} MiB limit")
            data = r.read(max_bytes + 1)
    else:
        path = Path(src).expanduser()
        if not path.is_file():
            raise ValueError(f"local image does not exist: {path}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"local image exceeds {max_bytes_mb} MiB limit")
        data = path.read_bytes()

    if len(data) > max_bytes:
        raise ValueError(f"image exceeds {max_bytes_mb} MiB limit")
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError(
                f"source image exceeds {max_pixels:,}-pixel limit")
        image.load()
        return image
    except ValueError:
        raise
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError(f"invalid or unsupported image: {error}") from error


def load_video(src: str) -> VideoFrames:
    """Bounded data URI, local path, or HTTP(S) video -> sampled PIL frames."""
    import base64
    import binascii
    import io
    import os
    from pathlib import Path

    import numpy as np

    if not isinstance(src, str) or not src:
        raise ValueError("video source must be a non-empty string")
    try:
        max_bytes_mb = int(os.environ.get("VMODEL_MAX_VIDEO_BYTES_MB", "64"))
        max_frames = int(os.environ.get("VMODEL_VIDEO_MAX_FRAMES", "16"))
        max_source_frames = int(os.environ.get(
            "VMODEL_VIDEO_MAX_SOURCE_FRAMES", "3600"))
        max_pixels = int(os.environ.get(
            "VMODEL_MAX_SOURCE_VIDEO_FRAME_PIXELS", "64000000"))
        max_duration = float(os.environ.get(
            "VMODEL_VIDEO_MAX_DURATION_SECONDS", "60"))
        sample_fps = float(os.environ.get("VMODEL_VIDEO_SAMPLE_FPS", "2"))
    except ValueError as error:
        raise ValueError("video limits must be numeric") from error
    if (max_bytes_mb <= 0 or max_frames < 2 or max_source_frames < max_frames
            or max_pixels <= 0 or max_duration <= 0 or sample_fps <= 0):
        raise ValueError("video limits must be positive and internally consistent")
    max_bytes = max_bytes_mb * 1024 * 1024

    if src.startswith("data:"):
        header, separator, payload = src.partition(",")
        if (not separator or not header.lower().startswith("data:video/")
                or ";base64" not in header.lower()):
            raise ValueError("video data URI must be video/*;base64")
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("video data URI contains invalid base64") from error
    elif src.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(src, timeout=20) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) > max_bytes:
                        raise ValueError(
                            f"remote video exceeds {max_bytes_mb} MiB limit")
                except ValueError as error:
                    if "exceeds" in str(error):
                        raise
            data = response.read(max_bytes + 1)
    else:
        path = Path(src).expanduser()
        if not path.is_file():
            raise ValueError(f"local video does not exist: {path}")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"local video exceeds {max_bytes_mb} MiB limit")
        data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"video exceeds {max_bytes_mb} MiB limit")

    try:
        import av
    except ImportError as error:
        raise ValueError("video decoding requires `pip install av`") from error

    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                raise ValueError("video contains no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.base_rate or 24.0)
            total_frames = int(stream.frames or 0)
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            if duration > max_duration:
                raise ValueError(
                    f"video duration {duration:.1f}s exceeds {max_duration:g}s limit")

            if total_frames > 0:
                if total_frames > max_source_frames:
                    raise ValueError(
                        f"video has {total_frames} frames; source limit is "
                        f"{max_source_frames}")
                desired = min(max_frames, total_frames,
                              max(4, int(max(duration, total_frames / fps)
                                         * sample_fps)))
                wanted = np.linspace(
                    0, total_frames - 1, desired).round().astype(int).tolist()
                wanted_set = set(wanted)
                decoded = []
                for index, frame in enumerate(container.decode(stream)):
                    if index in wanted_set:
                        decoded.append((index, frame.to_image().convert("RGB")))
                    if index >= wanted[-1]:
                        break
            else:
                all_frames = []
                for index, frame in enumerate(container.decode(stream)):
                    if index >= max_source_frames:
                        raise ValueError(
                            f"video exceeds {max_source_frames} decoded-frame limit")
                    all_frames.append((index, frame.to_image().convert("RGB")))
                if not all_frames:
                    raise ValueError("video contains no decodable frames")
                desired = min(max_frames, len(all_frames), max(
                    4, int(len(all_frames) / fps * sample_fps)))
                positions = np.linspace(
                    0, len(all_frames) - 1, desired).round().astype(int)
                decoded = [all_frames[index] for index in positions]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid or unsupported video: {error}") from error

    if not decoded:
        raise ValueError("video contains no sampled frames")
    indices = [index for index, _frame in decoded]
    frames = [frame for _index, frame in decoded]
    # The official processor's four-frame minimum applies to sampling when the
    # source actually contains at least four frames. A shorter clip is padded
    # only to the temporal patch size (two for Qwen3-VL), not unconditionally
    # to four frames.
    if len(frames) % 2:
        frames.append(frames[-1].copy())
        indices.append(indices[-1])
    for frame in frames:
        width, height = frame.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError(
                f"video frame exceeds {max_pixels:,}-pixel source limit")
    return VideoFrames(
        tuple(frames), fps, tuple(indices),
        duration or (indices[-1] / fps if fps else 0.0))


def tools_preamble(tools: list[dict]) -> str:
    """Hermes-style tool instructions for models without native template
    support. The JSON schemas are passed through verbatim."""
    fns = [t.get("function", t) for t in tools]
    return (
        "You have access to the following tools. To call one, respond with\n"
        "<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": {...}}\n</tool_call>\n"
        "Available tools (JSON Schema):\n"
        + "\n".join(json.dumps(f) for f in fns)
    )


_HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# harmony: '<|channel|>commentary to=functions.NAME ... <|message|>{...}<|call|>'
# decoded text may keep or strip the special-token glyphs — match both.
# The trailing alternation must stop at the NEXT call too (lookahead), not
# just <|call|> or end-of-string: with glyphs stripped and >1 call in one
# response, "$" alone forces backtracking across the whole remaining text
# and merges multiple calls into one malformed match (found 2026-07-13).
_HARMONY_RE = re.compile(
    r"(?:<\|channel\|>)?(?:commentary\s+)?to=functions\.([\w.-]+).*?"
    r"(?:<\|message\|>|json)\s*(\{.*?\})\s*"
    r"(?:<\|call\|>|(?=(?:commentary\s+)?to=functions\.)|$)",
    re.S,
)


def parse_tool_calls(text: str, model_type: str, *,
                     allowed_names=None,
                     argument_schemas: dict[str, dict] | None = None
                     ) -> tuple[str, list[dict]]:
    """Return (content_without_calls, openai_tool_calls). Empty list = plain
    text response. Arguments are validated as JSON but passed as a string,
    per the OpenAI schema."""
    calls = []
    allowed = set(allowed_names) if allowed_names is not None else None

    def mk(name: str, args_raw: str) -> bool:
        if (not isinstance(name, str) or not name
                or (allowed is not None and name not in allowed)):
            return False
        try:
            parsed_args = _strict_json_loads(args_raw)
            if argument_schemas is not None and name in argument_schemas:
                from .structured import validate_json_schema

                validate_json_schema(parsed_args, argument_schemas[name])
            args = json.dumps(parsed_args, allow_nan=False)  # canonicalize / validate
        except (TypeError, ValueError, json.JSONDecodeError):
            return False  # malformed/schema-invalid: leave text as-is for the client
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
        return True

    spans = []
    if model_type == "gpt_oss":
        for m in _HARMONY_RE.finditer(text):
            if mk(m.group(1), m.group(2)):
                spans.append(m.span())
    for m in _HERMES_RE.finditer(text):
        obj_raw = m.group(1)
        try:
            obj = _strict_json_loads(obj_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(obj, dict) and "name" in obj:
            if mk(obj["name"], json.dumps(obj.get("arguments", {}))):
                spans.append(m.span())
    # xLAM-2 and a few other function-calling specialists are trained to emit
    # a bare top-level JSON array rather than Hermes markers. Accept only when
    # the *entire* response is a non-empty array of exact call-shaped objects;
    # every name and argument object must pass the same allow-list/schema gates
    # as Hermes. Ordinary JSON answers therefore remain ordinary content.
    if not calls:
        stripped = text.strip()
        try:
            array = (_strict_json_loads(stripped)
                     if stripped.startswith("[") and stripped.endswith("]")
                     else None)
        except (TypeError, ValueError, json.JSONDecodeError):
            array = None
        if (isinstance(array, list) and array
                and all(isinstance(item, dict)
                        and set(item) == {"name", "arguments"}
                        and isinstance(item.get("arguments"), dict)
                        for item in array)):
            before = len(calls)
            valid = all(
                mk(item["name"], json.dumps(item["arguments"]))
                for item in array)
            if valid:
                start = text.find(stripped)
                spans.append((start, start + len(stripped)))
            else:
                del calls[before:]
    if not calls:
        return text, []
    content = "".join(
        ch for i, ch in enumerate(text)
        if not any(a <= i < b for a, b in spans)
    )
    return content, calls
