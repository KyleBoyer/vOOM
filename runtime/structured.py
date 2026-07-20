"""JSON-schema validation and token-level grammar constraints.

XGrammar is loaded lazily: ordinary free-text generation does not require it.
Structured-output and required-tool requests fail clearly when the optional
dependency is absent instead of silently weakening the request. JSON Schema
instance validation uses the schema's declared draft through ``jsonschema``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


class StructuredDecodingUnavailable(RuntimeError):
    pass


class JSONSchemaValidationError(ValueError):
    pass


def _schema_validator(schema: dict):
    if not isinstance(schema, dict):
        raise JSONSchemaValidationError("JSON Schema must be an object")
    try:
        from jsonschema.validators import validator_for
    except ImportError as error:  # pragma: no cover - exercised on minimal installs
        raise StructuredDecodingUnavailable(
            "JSON Schema support requires `pip install jsonschema`") from error
    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except Exception as error:
        raise JSONSchemaValidationError(f"invalid JSON Schema: {error}") from error
    return validator_cls(schema)


def check_json_schema(schema: dict) -> None:
    _schema_validator(schema)


def validate_json_schema(instance, schema: dict) -> None:
    validator = _schema_validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.path)
    raise JSONSchemaValidationError(f"{path}: {error.message}")


def _function(tool: dict) -> dict:
    function = tool.get("function", tool)
    return function if isinstance(function, dict) else {}


def effective_tool_schema(schema: dict) -> dict:
    """Apply tool-protocol schema extensions used by agent harnesses.

    Some Zod-to-JSON-Schema adapters must list every property in ``required``
    for provider compatibility, then preserve the actual optionality in the
    explicit ``x-optional`` extension. Ignoring it forces local constrained
    decoding to spell every nullable/default argument, which can exhaust a
    small tool-call token budget before the closing marker. Return a detached
    standards-compliant schema with those names removed from ``required``.
    """
    if not isinstance(schema, dict):
        raise JSONSchemaValidationError("JSON Schema must be an object")
    normalized = deepcopy(schema)

    def walk(node) -> None:
        if not isinstance(node, dict):
            return
        optional = node.pop("x-optional", None)
        if optional is not None:
            properties = node.get("properties")
            if (not isinstance(optional, list)
                    or not all(isinstance(name, str) for name in optional)):
                raise JSONSchemaValidationError(
                    "x-optional must be an array of property names")
            if not isinstance(properties, dict):
                raise JSONSchemaValidationError(
                    "x-optional requires an object schema with properties")
            unknown = sorted(set(optional) - set(properties))
            if unknown:
                raise JSONSchemaValidationError(
                    f"x-optional names unknown properties: {unknown}")
            required = node.get("required")
            if required is not None:
                remaining = [name for name in required if name not in optional]
                if remaining:
                    node["required"] = remaining
                else:
                    node.pop("required", None)

        for key in ("properties", "patternProperties", "$defs", "definitions",
                    "dependentSchemas"):
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    walk(child)
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    walk(child)
        for key in ("items", "contains", "additionalProperties",
                    "propertyNames", "if", "then", "else", "not"):
            walk(node.get(key))

    walk(normalized)
    check_json_schema(normalized)
    return normalized


def tool_argument_schemas(tools: list[dict]) -> dict[str, dict]:
    schemas = {}
    for tool in tools:
        function = _function(tool)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = function.get("parameters")
        if schema is None:
            schema = function.get("input_schema")
        schema = effective_tool_schema(schema or {"type": "object"})
        schemas[name] = schema
    return schemas


def tool_call_json_schema(tools: list[dict], specific_name: str | None = None) -> dict:
    choices = []
    for name, arguments in tool_argument_schemas(tools).items():
        if specific_name is not None and name != specific_name:
            continue
        choices.append({
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": arguments,
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    if not choices:
        raise JSONSchemaValidationError("tool constraint has no matching functions")
    return choices[0] if len(choices) == 1 else {"oneOf": choices}


def _xgrammar():
    try:
        import xgrammar as xgr
    except ImportError as error:  # pragma: no cover - exercised on minimal installs
        raise StructuredDecodingUnavailable(
            "constrained decoding requires `pip install xgrammar`") from error
    return xgr


def _compiler(engine):
    compiler = getattr(engine, "_xgrammar_compiler", None)
    if compiler is not None:
        return compiler
    xgr = _xgrammar()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(engine._model_dir), local_files_only=True)
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=int(engine.cfg.vocab_size),
            stop_token_ids=list(engine.cfg.eos_token_ids))
        compiler = xgr.GrammarCompiler(info, max_threads=4, cache_enabled=True)
    except Exception as error:
        raise StructuredDecodingUnavailable(
            f"could not initialize constrained decoder: {error}") from error
    engine._xgrammar_compiler = compiler
    return compiler


def _required_tool_grammar(schema: dict, allow_parallel: bool):
    """Wrap a JSON-schema grammar in required Hermes tool-call markers."""
    xgr = _xgrammar()
    grammar = str(xgr.Grammar.from_json_schema(
        schema, any_whitespace=True, strict_mode=True))
    replaced, count = re.subn(
        r"^root ::= (.+)$", r"tool_json ::= \1", grammar,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise StructuredDecodingUnavailable(
            "XGrammar JSON schema did not expose a root rule")
    suffix = (
        '\ntool_call ::= (("<tool_call>" [ \\n\\t]* tool_json '
        '[ \\n\\t]* "</tool_call>"))\n'
    )
    if allow_parallel:
        suffix += (
            "tool_calls ::= ((tool_call) | "
            "(tool_call [ \\n\\t]* tool_calls))\n"
            "root ::= ((tool_calls))\n"
        )
    else:
        suffix += "root ::= ((tool_call))\n"
    return xgr.Grammar.from_ebnf(replaced + suffix)


@dataclass
class GrammarConstraint:
    """Stateful next-token mask consumed by one generation request."""

    matcher: object
    vocab_size: int
    profile: str
    stop_on_complete: bool = True

    def __post_init__(self):
        xgr = _xgrammar()
        self._bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        self._token_indices = mx.arange(self.vocab_size, dtype=mx.uint32)
        self.completed = False

    @classmethod
    def json(cls, engine, schema: dict | None = None, *, strict: bool = True):
        compiler = _compiler(engine)
        compiled = (compiler.compile_builtin_json_grammar()
                    if schema is None else
                    compiler.compile_json_schema(
                        schema, any_whitespace=True, strict_mode=strict))
        xgr = _xgrammar()
        return cls(
            xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True),
            int(engine.cfg.vocab_size),
            "json" if schema is None else "json_schema")

    @classmethod
    def tools(cls, engine, tools: list[dict], *, required: bool,
              specific_name: str | None = None, allow_parallel: bool = True):
        schema = tool_call_json_schema(tools, specific_name)
        compiler = _compiler(engine)
        xgr = _xgrammar()
        if required:
            grammar = _required_tool_grammar(schema, allow_parallel)
            compiled = compiler.compile_grammar(grammar)
            profile = "required_tool"
        else:
            # Auto mode permits ordinary text but dispatches into a strict
            # argument schema as soon as the model starts a tool-call marker.
            grammar = xgr.Grammar.from_structural_tag(
                [xgr.StructuralTagItem(
                    begin="<tool_call>", schema=schema, end="</tool_call>")],
                ["<tool_call>"],
            )
            compiled = compiler.compile_grammar(grammar)
            profile = "auto_tool_schema"
        return cls(
            xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=required),
            int(engine.cfg.vocab_size), profile,
            stop_on_complete=required)

    def mask_logits(self, logits: mx.array) -> mx.array:
        xgr = _xgrammar()
        if self.completed:
            return logits
        need_apply = self.matcher.fill_next_token_bitmask(self._bitmask)
        if not need_apply:
            return logits
        words = mx.array(
            self._bitmask.numpy().reshape(-1).astype(np.int32)).astype(mx.uint32)
        indices = self._token_indices
        allowed = (
            (words[(indices // 32).astype(mx.int32)] >> (indices % 32))
            & mx.array(1, dtype=mx.uint32)
        ) != 0
        return mx.where(allowed, logits.reshape(-1), float("-inf"))

    def accept_token(self, token: int) -> None:
        if not self.matcher.accept_token(int(token)):
            raise RuntimeError(
                f"constrained decoder sampled token {token} outside its grammar")
        self.completed = bool(
            self.stop_on_complete and self.matcher.is_completed())
