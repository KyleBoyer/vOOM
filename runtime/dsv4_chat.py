"""DeepSeek V4 chat encoding, delegated to the checkpoint's own encoder.

DeepSeek V4 Flash ships no ``chat_template`` in any of the three places the
server looks, so the generic ``user:``/``assistant:`` fallback would apply --
a transcript with no learned turn boundary, which is the exact failure the
comment in ``_chat_prompt`` warns about. What it ships instead is
``encoding/encoding_dsv4.py``, a self-contained reference encoder inside the
checkpoint.

We import that file rather than transcribing it. The protocol has enough
surface -- BOS placement, tool-result merging into the preceding user turn,
tool-call ordering, the thinking-mode markers, DSML argument typing -- that a
transcription is a second implementation to keep in sync, and the released one
is by definition correct. It lives in the model directory, so it is loaded per
directory and cached.

Tools are not a template variable here: ``render_tools`` produces a system
section that is prepended to the system message.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def encoder_path(model_dir: Path) -> Path:
    return Path(model_dir) / "encoding" / "encoding_dsv4.py"


def has_released_encoder(model_dir: Path) -> bool:
    return encoder_path(model_dir).is_file()


def load_encoder(model_dir: Path):
    """Import the checkpoint's encoder under a private module name.

    Named after the resolved directory so two DeepSeek V4 checkpoints in one
    process cannot alias each other, and kept out of ``encoding_dsv4`` so it
    never collides with a same-named module on sys.path.
    """
    path = encoder_path(model_dir)
    key = str(path.resolve())
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(
                f"DeepSeek V4 checkpoint has no released encoder at {path}")
        name = f"_vmodel_dsv4_encoding_{abs(hash(key)):x}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load DeepSeek V4 encoder from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _CACHE[key] = module
        return module


def cached_model_dir() -> Path | None:
    """The checkpoint whose encoder is loaded, when exactly one is.

    Response parsing needs the same directory prompt rendering used, but the
    server's tool-call parser is reached from a dozen call sites that carry
    only a model type. A chat request always renders its prompt before parsing
    the reply, so by then this cache is warm. Returns None if no encoder is
    loaded, or if two checkpoints are live and the choice would be a guess.
    """
    with _LOCK:
        if len(_CACHE) != 1:
            return None
        return Path(next(iter(_CACHE))).parent.parent


def thinking_mode(enable_thinking: bool | None, reasoning_requested: bool
                  ) -> str:
    """Map the server's two independent controls onto the released modes.

    The encoder accepts exactly ``"chat"`` and ``"thinking"``. An explicit
    ``enable_thinking`` wins; otherwise asking for a reasoning effort implies
    thinking, and the default is chat.
    """
    if enable_thinking is not None:
        return "thinking" if enable_thinking else "chat"
    return "thinking" if reasoning_requested else "chat"


def _tool_schemas(module, tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    # Accept both OpenAI-wrapped ({"type": "function", "function": {...}}) and
    # already-flat schemas; the released helper only handles the former.
    wrapped = [t for t in tools if isinstance(t, dict) and "function" in t]
    flat = [t for t in tools if isinstance(t, dict) and "function" not in t]
    out = list(module.tools_from_openai_format(wrapped)) if wrapped else []
    out.extend(flat)
    return out or None


def render_prompt(model_dir: Path, messages: list[dict],
                  tools: list[dict] | None = None, *,
                  enable_thinking: bool | None = None,
                  reasoning_requested: bool = False,
                  reasoning_effort: str | None = None,
                  add_generation_prompt: bool = True) -> str:
    """Render an OpenAI-shaped conversation into the released prompt format."""
    module = load_encoder(model_dir)
    mode = thinking_mode(enable_thinking, reasoning_requested)

    prepared = [dict(m) for m in messages]
    schemas = _tool_schemas(module, tools)
    if schemas:
        section = module.render_tools(schemas)
        for message in prepared:
            if message.get("role") == "system":
                content = message.get("content") or ""
                message["content"] = (f"{content}\n\n{section}" if content
                                      else section)
                break
        else:
            prepared.insert(0, {"role": "system", "content": section})

    effort = reasoning_effort if mode == "thinking" else None
    prompt = module.encode_messages(
        prepared, thinking_mode=mode, reasoning_effort=effort)
    if not add_generation_prompt:
        # encode_messages always ends with the assistant marker; logprob-style
        # callers that scored a fixed continuation must not get it.
        marker = module.ASSISTANT_SP_TOKEN
        cut = prompt.rfind(marker)
        if cut != -1:
            prompt = prompt[:cut]
    return prompt


def parse_tool_calls(model_dir: Path | None, text: str,
                     allowed_names: set[str] | None = None):
    """Split a completion into visible content and OpenAI-shaped tool calls.

    The released ``parse_message_from_completion_text`` asserts the text ends
    with the EOS token, which a server response never carries: generation stops
    ON eos and the detokenizer drops it. So we locate the tool-call block and
    hand only that region to the released ``parse_tool_calls``, which is the
    part with real protocol surface.

    A malformed or truncated block yields no calls and leaves the text intact,
    rather than raising -- a response cut short by ``max_tokens`` is an
    ordinary outcome, not a server error.
    """
    if model_dir is None:
        model_dir = cached_model_dir()
        if model_dir is None:
            return text, []
    module = load_encoder(model_dir)
    # Note the tag is matched WITHOUT its closing '>'. The released parser's
    # first act is to read up to the next marker and require exactly ">\n",
    # so it must be handed an index sitting on that '>' -- passing the index
    # after it makes every well-formed block look malformed.
    begin = f"<{module.dsml_token}{module.tool_calls_block_name}"
    start = text.find(begin)
    if start == -1:
        return text, []

    content = text[:start].rstrip("\n")
    try:
        _index, _stop, raw_calls = module.parse_tool_calls(start + len(begin),
                                                           text)
    except (ValueError, AssertionError, IndexError):
        return text, []

    calls = []
    for position, call in enumerate(module.tool_calls_to_openai_format(
            raw_calls)):
        function = call.get("function", {})
        name = function.get("name")
        if allowed_names and name not in allowed_names:
            continue
        calls.append({
            "id": call.get("id") or f"call_{position}",
            "type": "function",
            "function": {"name": name,
                         "arguments": function.get("arguments", "{}")},
        })
    if not calls:
        return text, []
    return content, calls
