"""JSON-schema validation and token-level grammar constraints.

XGrammar is loaded lazily: ordinary free-text generation does not require it.
Structured-output and required-tool requests fail clearly when the optional
dependency is absent instead of silently weakening the request. JSON Schema
instance validation uses the schema's declared draft through ``jsonschema``.
"""

from __future__ import annotations

import os
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


def _grammar_compatible_schema(schema: dict) -> dict:
    """Rewrite an equivalent conditional-required shape XGrammar mishandles.

    XGrammar 0.1.35 accepts an object with base ``required`` fields plus an
    ``anyOf`` whose branches contain only more ``required`` fields, but its
    compiled grammar can terminate without the base fields.  The ordinary
    JSON-Schema validator catches that after generation, so execution remains
    safe, but a required tool request degrades into rejected text. Distribute
    the parent object into each branch before compilation. This is equivalent
    for this exact schema shape and makes every branch carry its full required
    set explicitly. Keep the wire and validation schemas unchanged.
    """
    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        node = {key: rewrite(child) for key, child in value.items()}
        branches = node.get("anyOf")
        properties = node.get("properties")
        if (node.get("type") == "object" and isinstance(properties, dict)
                and isinstance(branches, list) and branches
                and all(isinstance(branch, dict)
                        and set(branch) == {"required"}
                        and isinstance(branch.get("required"), list)
                        and branch["required"]
                        and all(isinstance(name, str)
                                for name in branch["required"])
                        for branch in branches)):
            base = {key: child for key, child in node.items() if key != "anyOf"}
            inherited = list(base.get("required") or [])
            expanded = []
            for branch in branches:
                variant = deepcopy(base)
                variant["required"] = list(dict.fromkeys(
                    [*inherited, *branch["required"]]))
                expanded.append(variant)
            return {"anyOf": expanded}
        return node

    compatible = rewrite(schema)
    check_json_schema(compatible)
    return compatible


def _xgrammar():
    try:
        import xgrammar as xgr
    except ImportError as error:  # pragma: no cover - exercised on minimal installs
        raise StructuredDecodingUnavailable(
            "constrained decoding requires `pip install xgrammar`") from error
    return xgr


def _relax_layer_type_validation() -> None:
    """Drop transformers' ``validate_layer_type`` class validator.

    Newer transformers versions added a strict enum check on
    ``PretrainedConfig.layer_types`` against a hardcoded whitelist (see
    ``transformers/configuration_utils.py``). Third-party remote-code archs
    that predate this check (e.g. Jet-Nemotron's real released config, which
    uses "jet"/"swa"/"attn") fail ``AutoConfig``/``AutoTokenizer.from_pretrained``
    with a ``StrictDataclassClassValidationError`` even though ``layer_types``
    is never consulted by transformers' own generic code -- only by that
    model's own remote modeling file, which reads the raw strings directly.
    Validation-only, so this cannot change any model's actual computation.
    """
    from transformers.configuration_utils import PretrainedConfig

    validators = getattr(PretrainedConfig, "__class_validators__", None)
    if not validators or not any(
            v.__name__ == "validate_layer_type" for v in validators):
        return
    PretrainedConfig.__class_validators__ = [
        v for v in validators if v.__name__ != "validate_layer_type"]


def _compiler(engine):
    compiler = getattr(engine, "_xgrammar_compiler", None)
    if compiler is not None:
        return compiler
    xgr = _xgrammar()
    try:
        from transformers import AutoTokenizer

        _relax_layer_type_validation()
        tokenizer = AutoTokenizer.from_pretrained(
            str(engine._model_dir), local_files_only=True)
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=int(engine.cfg.vocab_size),
            stop_token_ids=list(engine.cfg.eos_token_ids))
        max_threads = int(os.environ.get("VMODEL_XGRAMMAR_MAX_THREADS", "4"))
        if not 1 <= max_threads <= 64:
            raise ValueError(
                "VMODEL_XGRAMMAR_MAX_THREADS must be between 1 and 64")
        compiler = xgr.GrammarCompiler(
            info, max_threads=max_threads, cache_enabled=True)
    except Exception as error:
        raise StructuredDecodingUnavailable(
            f"could not initialize constrained decoder: {error}") from error
    engine._xgrammar_compiler = compiler
    return compiler


def _required_tool_grammar(schema: dict, allow_parallel: bool):
    """Wrap canonical JSON in deterministic Hermes tool-call markers.

    The previous grammar allowed arbitrary whitespace before/inside/after a
    call.  Its parsed API object was canonical, but the retained KV represented
    the model's arbitrary original spacing, so the next request's structured
    history often diverged.  XGrammar's no-whitespace profile is actually its
    canonical JSON profile (`, ` and `: ` separators); make the wrapper exact
    as well so a structured round trip is token-identical.
    """
    xgr = _xgrammar()
    grammar = str(xgr.Grammar.from_json_schema(
        schema, any_whitespace=False, strict_mode=True))
    replaced, count = re.subn(
        r"^root ::= (.+)$", r"tool_json ::= \1", grammar,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise StructuredDecodingUnavailable(
            "XGrammar JSON schema did not expose a root rule")
    suffix = (
        '\ntool_call ::= (("<tool_call>" tool_json "</tool_call>"))\n'
    )
    if allow_parallel:
        suffix += (
            "tool_calls ::= ((tool_call) | (tool_call tool_calls))\n"
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
        self._dead_end = False

    @classmethod
    def json(cls, engine, schema: dict | None = None, *, strict: bool = True,
             canonical_whitespace: bool = False):
        """`canonical_whitespace=True` compiles with `any_whitespace=False`
        (the same no-extra-whitespace convention `.tools()`'s required-tool
        grammar already uses) instead of the default `any_whitespace=True`.

        This is a real formatting change, not just an internal detail: with
        `any_whitespace=True`, xgrammar's `find_jump_forward_string()`
        degenerates to almost nothing (measured: just `'{'` for a real
        3-field schema) because arbitrary whitespace could legally precede
        every key/punctuation mark, so nothing past the opening brace is
        actually determined. `any_whitespace=False` collapses that
        uncertainty and forces the canonical `{"key": ` span too (measured:
        `'{"title": "'` for the same schema) -- directly raising F98's
        forced fraction for JSON-schema-constrained requests, the same way
        the tool-call grammar already benefits. Gated to the lossy profile
        (server.py only sets this alongside `grammar_jump_forward_lossy`)
        because it changes emitted JSON whitespace byte-for-byte versus
        what the model would naturally produce unconstrained -- a real
        lossy tradeoff on output formatting, not merely an implementation
        detail, even though JSON semantics/content are unaffected either way.
        """
        compiler = _compiler(engine)
        any_whitespace = not canonical_whitespace
        compiled = (compiler.compile_builtin_json_grammar()
                    if schema is None else
                    compiler.compile_json_schema(
                        schema, any_whitespace=any_whitespace, strict_mode=strict))
        xgr = _xgrammar()
        return cls(
            xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True),
            int(engine.cfg.vocab_size),
            "json" if schema is None else "json_schema")

    @classmethod
    def tools(cls, engine, tools: list[dict], *, required: bool,
              specific_name: str | None = None, allow_parallel: bool = True):
        schema = _grammar_compatible_schema(
            tool_call_json_schema(tools, specific_name))
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
            self._dead_end = False
            return logits
        words = mx.array(
            self._bitmask.numpy().reshape(-1).astype(np.int32)).astype(mx.uint32)
        indices = self._token_indices
        allowed = (
            (words[(indices // 32).astype(mx.int32)] >> (indices % 32))
            & mx.array(1, dtype=mx.uint32)
        ) != 0
        # An under-tuned/untrained model (e.g. no native tool-call special
        # tokens) can commit to a tool-call span it cannot complete validly,
        # walking the grammar into a state with genuinely zero legal next
        # tokens. Every position is then -inf and any sampler (greedy argmax
        # included) returns an arbitrary index that accept_token() below
        # will reject regardless of which one it picked. Recording that
        # here lets accept_token() log this specific, expected case at a
        # lower level of concern than an ordinary rejection (see there).
        self._dead_end = not bool(mx.any(allowed).item())
        return mx.where(allowed, logits.reshape(-1), float("-inf"))

    def accept_token(self, token: int) -> None:
        if not self.matcher.accept_token(int(token)):
            # _compiler() (above) builds xgrammar's own tokenizer/vocabulary
            # view via `transformers.AutoTokenizer`, separate from the raw
            # `tokenizers.Tokenizer` this engine samples/decodes with
            # elsewhere -- live-confirmed 2026-07-22 that the two disagree
            # for OLMoE-1B-7B (whose vocabulary includes an unusual
            # non-BOS/EOS/pad special token at id 0, "|||IP_ADDRESS|||"):
            # mask_logits above reported real tokens as legal, greedy argmax
            # picked one of them, and the grammar matcher still rejected it.
            # A dead grammar state (see mask_logits) hits this same path
            # for a different, better-understood reason: an under-tuned
            # model with no native tool-call tokens can commit to a
            # tool-call span it can never complete validly. Neither case is
            # recoverable by resampling mid-generation -- stop the response
            # cleanly with whatever valid content exists so far rather than
            # failing the whole request over one token choice.
            if not self._dead_end:
                print(
                    f"[structured] constrained decoder rejected token {token} "
                    f"despite mask_logits allowing it (profile={self.profile}); "
                    "stopping generation for this request rather than failing it",
                    flush=True,
                )
            self.completed = True
            return
        self.completed = bool(
            self.stop_on_complete and self.matcher.is_completed())

    def forced_run(self, limit: int, encode=None) -> list[int]:
        """Grammar fast-forward (jump-forward decoding, token-level exact
        variant): return the run of tokens the grammar FORCES next.

        ``encode`` (optional, LOSSY-profile only): a ``str -> list[int]``
        tokenizer callback enabling SGLang-style STRING-level jump-forward
        via ``matcher.find_jump_forward_string()``. Measured 2026-07-23 on a
        real required-tool grammar: the exact token-level check below almost
        never fires (1 forced token in a 39-token call) because the grammar
        is byte-level -- in a forced-STRING region many BPE tokens are still
        legal (every token spelling a prefix of the forced text), so
        "exactly one legal token" is rare even where the text is fully
        determined. The string-level variant commits the canonical
        tokenization of the forced string instead (59 forced chars at the
        same tool-call state). That can DIFFER from the tokenization the
        model itself would have picked through per-token masked argmax
        (identical rendered text, different token ids), so it is gated to
        the fast/lossy profile and never the lossless target.

        Whenever the grammar allows exactly one legal next token, the
        constrained sampler's masked argmax is guaranteed to pick it
        regardless of model logits (every other position is -inf) -- so the
        token's identity needs no model forward pass at all, only its KV/
        recurrent-state update, which the caller batches into one
        multi-position sweep. This is byte-identical to the plain
        constrained path BY CONSTRUCTION (unlike SGLang-style string-level
        jump-forward, which can change tokenization): each committed token
        is precisely the one the plain per-token loop would have sampled.

        Each returned token has already been accepted into the matcher.
        Stops at `limit`, at grammar completion (setting self.completed,
        mirroring accept_token), at a dead-end/ambiguous state, or when the
        mask allows more than one token (the model must genuinely choose).
        """
        # is_terminated() guard: the auto tool profile terminates via an
        # accepted stop token WITHOUT setting self.completed
        # (stop_on_complete=False), and xgrammar hard-fails
        # find_jump_forward_string()/further stepping on a terminated
        # matcher (live-crashed 2026-07-23 on the real Plex capture; the
        # required-tool synthetic test never hit it because that profile
        # terminates without a stop token and sets completed instead).
        if self.completed or self.matcher.is_terminated():
            return []
        if encode is not None:
            jump = self.matcher.find_jump_forward_string()
            if len(jump) >= 2:
                forced: list[int] = []
                for token in list(encode(jump))[:limit]:
                    if not self.matcher.accept_token(int(token)):
                        # A canonical tokenization of grammar-forced text
                        # should always be accepted; fail closed to the
                        # per-token path on any disagreement.
                        break
                    forced.append(int(token))
                    if self.matcher.is_terminated() or self.matcher.is_completed():
                        if self.stop_on_complete and self.matcher.is_completed():
                            self.completed = True
                        break
                if forced:
                    return forced
        forced = []
        while len(forced) < limit:
            if not self.matcher.fill_next_token_bitmask(self._bitmask):
                break  # no mask needed -> everything legal -> not forced
            words = self._bitmask.numpy().reshape(-1).astype(np.int32)
            bits = np.unpackbits(
                words.astype("<i4").view(np.uint8), bitorder="little")
            allowed = np.flatnonzero(bits[:self.vocab_size])
            if allowed.size != 1:
                break  # free choice (or dead end -- normal path handles it)
            token = int(allowed[0])
            if not self.matcher.accept_token(token):
                # Should be impossible (the matcher itself reported the
                # token as the only legal one); fail closed to the ordinary
                # per-token path rather than trusting a contradicted state.
                break
            forced.append(token)
            if self.matcher.is_terminated() or self.matcher.is_completed():
                if self.stop_on_complete and self.matcher.is_completed():
                    self.completed = True
                break
        return forced
