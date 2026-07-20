"""Phase 11: OpenAI-compatible HTTP endpoint over the paged runtime.

    python -m runtime.server --port 8077

Endpoints — routing is by PATH SHAPE, not by URL prefix: every path below
also works with the leading /v1 omitted (2026-07-13, user request), since
different client SDKs default to different base URLs (the Anthropic SDK
always sends /v1/messages regardless of configured base_url; the OpenAI
SDK sends whatever base_url + path you configure).
  GET  /v1/models                  — registry (local models + NAS GLM)
  POST /v1/completions             — OpenAI legacy completions: {model, prompt, max_tokens, stream?}
  POST /v1/chat/completions        — OpenAI chat: {model, messages, max_tokens, stream?, tools?, stop?}
  POST /v1/responses               — OpenAI Responses API: {model, input, instructions?, max_output_tokens?}
                                      full tool-calling (function_call/function_call_output round
                                      trip), image/video input, streaming (typed SSE events), reasoning param
  POST /v1/messages                — Anthropic Messages API: {model, messages, max_tokens, system?, stop_sequences?}
                                      full tool_use/tool_result round trip, image/video (base64/url
                                      media blocks), streaming (typed SSE events), thinking param
Response schemas for /v1/responses and /v1/messages are verified against the
installed `openai`/`anthropic` SDKs' own Pydantic models
(tests/test_multi_protocol_clients.py, tests/test_protocol_features.py), not
hand-guessed from memory.

Mode control (GOAL/Sub-Goal vs Side-Quest), per request — via the `model` id,
not a header (2026-07-13, user request: a non-standard header doesn't fit
naturally into any of the three protocols; the `model` field does, since
every one of them already has it as a plain string):
  model = "<name>"          -> lossless (default; weights served bit-exact
                                as released — fp16/bf16 streams, MXFP4-as-
                                released for gpt-oss. The main goal path.)
  model = "lossy-<name>"    -> fast: dense models get the native-context side-quest
                                treatment — 4-bit quantize-on-load resident
                                cache (measured: 7B 41 s/token -> 15.4 tok/s).
                                Supported OLMoE models prefer a derived
                                expert-only MXFP4 artifact plus exact-candidate
                                head reranking when one is available.
  model = "lossy-long-<name>" -> Qwen2-only fast-long profile: the same
                                transforms plus experimental static YaRN
                                (2x/65,536 positions by default).
  GET /v1/models advertises `<name>` and `lossy-<name>` for every local model,
  plus `lossy-long-<name>` for Qwen2 checkpoints, so a client can discover the
  convention without prior knowledge of it. The `X-VModel-Mode` header /
  `vmodel_mode` body field from before this convention still work as a
  higher-precedence override, for compatibility.

Model ids: HF-style ("openai/gpt-oss-120b", "Qwen/Qwen2.5-72B", ...) mapped to
local stores; unknown ids are snapshot-downloaded to models/<name> and served
with default optimizations (Llama/Qwen/OLMoE/gpt-oss/GLM families supported).
The selected `lossy-`/`lossy-long-` prefix is stripped before resolution.

Auto-download is ASYNC, not a blocking client-facing call (2026-07-13, fixing
a live-confirmed gap: a synchronous snapshot_download() inside the locked
request handler once hung a client connection for 90+ seconds with zero
progress visibility on a real stalled fetch). A request for an unrecognized
model id kicks off a background download (DownloadManager) and returns
immediately:
  HTTP 202  {"vmodel_download_status": "downloading", "elapsed_seconds": ...}
  HTTP 422  {"vmodel_download_status": "failed", "error": "<clear reason>"}
Retry the same request once it's ready (or poll GET /v1/models, which lists
any in-flight/failed download alongside the normal registry). The config is
validated (ModelConfig.from_dir) before a download is marked ready, so an
unsupported architecture (e.g. a GPT-2-family config.json using `n_head`
instead of this codebase's expected `num_attention_heads`) fails with a
clear message instead of a raw KeyError traceback on the next request.

Auto-downloaded models start out served as raw safetensors. The original
second-request AUTO-PACK path is disabled by default after a durability audit:
its daemon thread deleted source shards/intermediate tensors before a
transactional initial generation existed, and a resident lazy reader could still
need those files. Set `VMODEL_ENABLE_UNSAFE_AUTOPACK=1` only for deliberate
development experiments. Normal automatic packing returns after F31 provides a
non-destructive build, verify, atomic flip under INFER_LOCK, and post-flip source
reclamation. If the experimental path is enabled, responses and GET /v1/models
carry informational fields while it runs:
  "vmodel_pack_status": "packing", "vmodel_pack_progress_pct": 42.0,
  "vmodel_pack_eta_seconds": 118
Once packing finishes, the fields disappear and the next engine load picks up
the packed store. This paragraph describes the disabled experimental path, not a
current safety guarantee.

One engine is resident at a time (16 GB machine); switching model or mode swaps
engines (close + clear Metal cache). Prompt-KV persistence keeps repeat system
prompts cheap across requests.

Sampling defaults to greedy so deterministic lossless A/B gates keep their
existing behavior. Explicit positive `temperature` requests execute real
categorical sampling with functional `top_p`, `top_k`, and optional `seed`
controls. Speculative decoders fall back to the target for stochastic requests;
their exact verification contracts remain greedy-only. `stop` is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


LOSSY_PREFIX = "lossy-"  # see split_model_mode() — the advertised mode-switch convention
LOSSY_LONG_PREFIX = "lossy-long-"


class RequestValidationError(ValueError):
    """A request/profile combination the client can correct (HTTP 400)."""


class PreparedPrompt(str):
    """Rendered prompt carrying the exact token IDs already validated.

    ``_prepare_chat_prompt`` must tokenize once to enforce the context limit.
    The generation engine used to tokenize the same large tool manifest again;
    retaining the immutable IDs on this string avoids that duplicate hot-path
    work while preserving ordinary ``str`` behavior for every protocol helper.
    """

    def __new__(cls, text: str, token_ids, tool_capsules=(),
                cache_namespace: str = "default", force_paged_kv: bool = False):
        instance = super().__new__(cls, text)
        instance.token_ids = tuple(token_ids)
        # Optional (content-id, token-start, token-end) records used only by the
        # explicitly lossy dense-model PIC path. Ordinary PreparedPrompt callers
        # and every lossless request retain the original two-argument behavior.
        instance.tool_capsules = tuple(tool_capsules)
        # Hidden tool routing renders two deliberately different prompt
        # lineages.  Carry the phase into the engine so its bounded hot-KV LRU
        # can prefer evicting a transient decision branch over an expensive
        # execution branch, while the durable segment DAG retains both.
        instance.cache_namespace = str(cache_namespace or "default")
        instance.force_paged_kv = bool(force_paged_kv)
        return instance


_PREPARED_PROMPT_TOKEN_CACHE_SLOTS = 8
_PREPARED_PROMPT_TOKEN_CACHE_MIN_TOKENS = 1024


def _prepared_prompt_ids(engine, prompt: str):
    """Tokenize an exact rendered prompt once per resident engine.

    The cache key is the complete string, not a digest, so a collision cannot
    substitute token IDs.  Keeping it engine-local also prevents IDs from one
    tokenizer/model surviving an engine swap.  Small prompts are cheaper to
    encode and must not evict the large tool manifests this path is for.
    """
    cache = getattr(engine, "_prepared_prompt_token_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(engine, "_prepared_prompt_token_cache", cache)
    cached = cache.pop(prompt, None)
    if cached is not None:
        cache[prompt] = cached
        return cached[0], cached[1], True

    encoded = engine.tokenizer.encode(prompt)
    token_ids = tuple(encoded.ids)
    offsets = tuple(getattr(encoded, "offsets", ()))
    if len(token_ids) >= _PREPARED_PROMPT_TOKEN_CACHE_MIN_TOKENS:
        cache[prompt] = (token_ids, offsets)
        while len(cache) > _PREPARED_PROMPT_TOKEN_CACHE_SLOTS:
            cache.popitem(last=False)
    return token_ids, offsets, False


class _TokenOffsetIndex:
    """Exact first-match boundary queries over monotonic tokenizer offsets."""

    _BLOCK = 256

    def __init__(self, offsets, prompt_length: int):
        starts = []
        ends = []
        first_nonempty = {}
        previous_start = -1
        for index, offset in enumerate(offsets):
            if (not isinstance(offset, (list, tuple)) or len(offset) != 2
                    or any(isinstance(value, bool)
                           or not isinstance(value, int) for value in offset)):
                raise ValueError("token offsets must be integer pairs")
            start, end = offset
            if (start < previous_start or start < 0 or end < start
                    or end > prompt_length):
                raise ValueError("token offsets must be monotonic and in range")
            starts.append(start)
            ends.append(end)
            if end > start and start not in first_nonempty:
                first_nonempty[start] = index
            previous_start = start
        self.starts = starts
        self.ends = ends
        self.first_nonempty = first_nonempty
        self.block_max_ends = [
            max(ends[index:index + self._BLOCK], default=-1)
            for index in range(0, len(ends), self._BLOCK)
        ]

    def token_start(self, char_start: int) -> int | None:
        return self.first_nonempty.get(char_start)

    def token_end(self, char_end: int, token_start: int) -> int | None:
        """First index >= start whose offset contains ``char_end``."""
        limit = bisect_left(
            self.starts, char_end, lo=token_start)
        index = token_start
        while index < limit and index % self._BLOCK:
            if self.ends[index] >= char_end:
                return index + 1
            index += 1
        while index + self._BLOCK <= limit:
            block = index // self._BLOCK
            if self.block_max_ends[block] >= char_end:
                block_end = index + self._BLOCK
                while index < block_end:
                    if self.ends[index] >= char_end:
                        return index + 1
                    index += 1
                return None
            index += self._BLOCK
        while index < limit:
            if self.ends[index] >= char_end:
                return index + 1
            index += 1
        return None


def _tool_capsule_spans(prompt: str, prompt_tools: list[dict], token_ids,
                        offsets) -> tuple[tuple[str, int, int], ...]:
    """Locate independently token-aligned compact tool objects in a prompt.

    Qwen's native template and the native-template fallback emit one compact
    JSON object per line. Qwen's tokenizer
    makes ``{"` the first token and folds the trailing newline into the final
    ``}\n`` token, yielding clean non-overlapping spans. Templates that join a
    tool to array punctuation or otherwise lack exact token boundaries simply
    return no spans; the experimental PIC path then stays disabled.
    """
    if not prompt_tools or len(offsets) != len(token_ids):
        return ()
    from jinja2.utils import htmlsafe_json_dumps

    try:
        offset_index = _TokenOffsetIndex(offsets, len(prompt))
    except ValueError:
        return ()

    # Native Qwen prose itself says ``<tools></tools>`` before the real catalog.
    # Select the final paired block, not that explanatory literal.
    search_end = prompt.rfind("</tools>")
    search_start = prompt.rfind("<tools>", 0, search_end)
    if search_start < 0 or search_end < 0:
        return ()
    search_start += len("<tools>")
    spans = []
    previous_token_end = 0
    for tool in prompt_tools:
        serialized = str(htmlsafe_json_dumps(
            tool, dumps=json.dumps, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True))
        char_start = prompt.find(serialized, search_start, search_end)
        if char_start < 0:
            return ()
        char_end = char_start + len(serialized)
        token_start = offset_index.token_start(char_start)
        token_end = (
            offset_index.token_end(char_end, token_start)
            if token_start is not None else None)
        if (token_start is None or token_end is None
                or token_start < previous_token_end):
            return ()
        identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        spans.append((identity, token_start, token_end))
        previous_token_end = token_end
        search_start = char_end
    return tuple(spans)


def _voom_quantization_metadata(model_dir: Path) -> dict | None:
    """Read the converter provenance marker, if present and usable."""
    try:
        marker = json.loads((model_dir / "config.json").read_text()).get(
            "voom_quantization")
    except (OSError, ValueError):
        return None
    return marker if isinstance(marker, dict) and marker.get("profile") else None


def _is_voom_lossy_checkpoint(model_dir: Path) -> bool:
    """Whether this artifact was derived by vOOM's lossy converter.

    Generic quantization metadata is insufficient: some publishers release a
    quantized checkpoint as the canonical artifact (gpt-oss is one example),
    and serving that artifact unchanged is still the lossless goal. The
    converter writes an explicit provenance marker, so only locally derived
    artifacts are forced onto the side-quest namespace.
    """
    return _voom_quantization_metadata(model_dir) is not None


def _derived_artifacts_for(source: Path) -> list[Path]:
    """Find complete converter outputs adjacent to one registered source."""
    try:
        source = source.resolve()
        candidates = source.parent.glob(f"{source.name}-mlx-*")
    except OSError:
        return []
    found = []
    for candidate in candidates:
        marker = _voom_quantization_metadata(candidate)
        try:
            same_source = marker and Path(marker.get("source", "")).resolve() == source
        except OSError:
            same_source = False
        if (same_source and (candidate / "model.safetensors.index.json").is_file()
                and not (candidate / ".quantize-incomplete.json").exists()):
            found.append(candidate)
    return sorted(found)


def _preferred_fast_artifact(source: Path) -> Path:
    """Use a complete expert-MXFP4 sibling for supported MoE sources."""
    if _is_voom_lossy_checkpoint(source):
        return source
    for candidate in _derived_artifacts_for(source):
        try:
            config = json.loads((candidate / "config.json").read_text())
        except (OSError, ValueError):
            continue
        quant = config.get("quantization", {})
        marker = config.get("voom_quantization", {})
        try:
            bits = int(quant.get("bits", 0))
        except (TypeError, ValueError):
            continue
        if (config.get("model_type") in ("olmoe", "qwen3_5_moe")
                and marker.get("profile") == "experts"
                and quant.get("mode") == "mxfp4"
                and bits == 4):
            print(f"[server] fast artifact: {source} -> {candidate}", flush=True)
            return candidate
    return source


def _qwen_family_key(name: str) -> str:
    """Checkpoint name with parameter count and local MLX suffix removed."""
    base = re.sub(r"(?i)-mlx(?:-.*)?$", "", name)
    return re.sub(r"(?i)(?<=-)\d+(?:\.\d+)?b(?=-|$)", "{size}", base).lower()


def _complete_local_checkpoint(path: Path) -> bool:
    return bool(
        path.is_dir()
        and (path / "config.json").is_file()
        and ((path / "model.safetensors.index.json").is_file()
             or any(path.glob("*.safetensors")))
    )


def _speculative_draft_for(target: Path, cfg) -> Path | None:
    """Find a local smaller MXFP4 Qwen draft; never downloads implicitly.

    ``VMODEL_SPECULATIVE_DRAFT`` accepts a local path, ``auto`` (the default),
    or ``off``/``0``. Automatic discovery is deliberately strict: same named
    Qwen family/instruction variant, dense architecture, smaller hidden size,
    and the locally validated 1.5B MXFP4 checkpoint adjacent to the target.
    """
    override = os.environ.get("VMODEL_SPECULATIVE_DRAFT", "auto").strip()
    if override.lower() in ("0", "off", "false", "none", "disabled"):
        return None
    if override and override.lower() != "auto":
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        candidate = candidate.resolve()
        if not _complete_local_checkpoint(candidate):
            raise RequestValidationError(
                f"VMODEL_SPECULATIVE_DRAFT is not a complete local checkpoint: {candidate}")
        return candidate

    try:
        target_hidden = int(cfg.hidden_size)
        target_family = _qwen_family_key(target.name)
        siblings = list(target.parent.iterdir())
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    found: list[tuple[int, Path]] = []
    for candidate in siblings:
        if candidate == target or not _complete_local_checkpoint(candidate):
            continue
        if _qwen_family_key(candidate.name) != target_family:
            continue
        if not re.search(r"(?i)(?:^|-)1\.5b(?:-|$)", candidate.name):
            continue
        try:
            config = json.loads((candidate / "config.json").read_text())
            hidden = int(config.get("hidden_size", 0))
            quant = config.get("quantization") or {}
            bits = int(quant.get("bits", 0))
        except (OSError, TypeError, ValueError):
            continue
        if (config.get("model_type") != "qwen2"
                or config.get("vision_config")
                or int(config.get("num_experts", 0) or 0)
                or not 0 < hidden < target_hidden
                or quant.get("mode") != "mxfp4"
                or bits != 4):
            continue
        found.append((hidden, candidate.resolve()))
    # Hidden size is only a deterministic tie-breaker if multiple copies of the
    # validated 1.5B profile are present. Other sizes require an explicit path
    # until their acceptance/throughput clears the same real-model gate.
    return min(found, default=(0, None), key=lambda item: item[0])[1]


def _dspark_draft_for(target: Path, cfg) -> Path | None:
    """Find a complete, shape-compatible local Qwen3 DSpark checkpoint.

    Discovery never downloads and deliberately requires the released block-7
    schema validated by the local exact-token/throughput gate.  A custom local
    checkpoint can still be selected explicitly for experimentation.
    """
    override = os.environ.get("VMODEL_DSPARK_DRAFT", "auto").strip()
    if override.lower() in ("0", "off", "false", "none", "disabled"):
        return None
    if override and override.lower() != "auto":
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if not _complete_local_checkpoint(candidate):
            raise RequestValidationError(
                f"VMODEL_DSPARK_DRAFT is not a complete local checkpoint: {candidate}")
        return candidate

    try:
        target_hidden = int(cfg.hidden_size)
        target_vocab = int(cfg.vocab_size)
        target_layers = int(cfg.num_hidden_layers)
        siblings = list(target.parent.iterdir())
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    found = []
    for candidate in siblings:
        if candidate == target or not _complete_local_checkpoint(candidate):
            continue
        try:
            config = json.loads((candidate / "config.json").read_text())
            taps = [int(layer) for layer in config.get("target_layer_ids", [])]
        except (OSError, TypeError, ValueError):
            continue
        architectures = config.get("architectures") or []
        if (config.get("model_type") != "qwen3"
                or "Qwen3DSparkModel" not in architectures
                or int(config.get("hidden_size", 0) or 0) != target_hidden
                or int(config.get("vocab_size", 0) or 0) != target_vocab
                or int(config.get("num_target_layers", 0) or 0) != target_layers
                or int(config.get("block_size", 0) or 0) != 7
                or not taps or min(taps) < 0 or max(taps) >= target_layers):
            continue
        found.append(candidate.resolve())
    return min(found, default=None, key=lambda path: path.name.lower())


def _registry() -> dict[str, Path]:
    """Discover base models; advertising adds lossless/fast and Qwen long IDs."""
    from .local_config import get_storage_config

    reg: dict[str, Path] = {}
    for d in sorted((ROOT / "models").iterdir()):
        if (d.is_dir() and not d.name.startswith("tool-embed-")
                and (d / "config.json").exists()):
            reg[d.name] = d
    storage = get_storage_config()
    for store in storage.stores:
        for mount in sorted(Path(storage.volumes_root).glob(f"{store.name}*")):
            base = mount / store.models_subdir if store.models_subdir else mount
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if (d.is_dir() and not d.name.startswith("tool-embed-")
                        and d.name not in reg and (d / "config.json").exists()):
                    reg[d.name] = d
            break  # first healthy mount for this store wins (avoids scanning cycled duplicates)
    # Converter outputs live beside their source so multi-gigabyte artifacts do
    # not need to be copied or manually symlinked into every registry root.
    # Only explicit, complete provenance-matched outputs are added.
    for source in list(reg.values()):
        if _is_voom_lossy_checkpoint(source):
            continue
        for artifact in _derived_artifacts_for(source):
            reg.setdefault(artifact.name, artifact)
    return reg


def _advertised_model_ids() -> list[str]:
    reg = _registry()
    base = [name for name, path in reg.items()
            if not _is_voom_lossy_checkpoint(path)]
    fast = list(reg)
    long_qwen = []
    for name in fast:
        try:
            if json.loads((reg[name] / "config.json").read_text()).get("model_type") == "qwen2":
                long_qwen.append(LOSSY_LONG_PREFIX + name)
        except (OSError, ValueError):
            pass
    return base + [LOSSY_PREFIX + name for name in fast] + long_qwen


_MIN_FREE_BYTES_FOR_AUTO_DOWNLOAD = 5_000_000_000  # 5 GB; see _resolve() note
_DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
_DEFAULT_REQUEST_READ_TIMEOUT_SECONDS = 30.0
_DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS = 30.0
# API clients may omit an output budget and expect generation to stop at the
# model's learned EOS. The engine still needs a finite runaway guard; 4096 is a
# safety ceiling, not a planned response length, and can be tuned explicitly.
_DEFAULT_OMITTED_MAX_OUTPUT_TOKENS = 4096
# Current two-stage auto-pack deletes raw shards and intermediate .vt files
# before a transactional generation is committed.  It is therefore disabled by
# default: a daemon-thread crash or concurrent lazy read can otherwise destroy a
# usable model.  This escape hatch is intentionally named UNSAFE until F31 covers
# initial builds; deliberate operators can still exercise the old path.
_ENABLE_UNSAFE_AUTOPACK = os.environ.get("VMODEL_ENABLE_UNSAFE_AUTOPACK") == "1"


class ModelDownloading(Exception):
    """Raised by _resolve() when a model isn't local yet but a background
    download has been kicked off (or is already in flight). 2026-07-13:
    replaces the old behavior of running snapshot_download() INLINE inside
    the locked request handler — live-tested and confirmed to silently
    hang the client connection with ZERO progress visibility (one real
    pull, yujiepan/qwen2.5-tiny-random, froze mid-download for 90+ seconds
    with the server process's CPU time static, i.e. genuinely blocked on
    network I/O, no timeout, no way for the client to tell). Callers should
    catch this and return an immediate, clear "still downloading" response
    instead of blocking on it."""

    def __init__(self, model_id: str, status: dict):
        self.model_id, self.status = model_id, status


class ModelDownloadFailed(Exception):
    """Raised by _resolve() when a background download or its post-download
    config validation failed. Carries a clear, actionable message rather
    than letting a raw exception (e.g. the KeyError from an unsupported
    config.json architecture) surface as an opaque 500 on some LATER,
    unrelated-looking request."""

    def __init__(self, model_id: str, error: str):
        self.model_id, self.error = model_id, error
        super().__init__(error)


class DownloadManager:
    """Tracks in-flight/ready/failed HF snapshot downloads in a background
    thread, keyed by the exact model_id string requested, so a client
    request for an unrecognized model never blocks the shared INFER_LOCK on
    a network fetch. Also validates the config loads under a supported
    architecture BEFORE marking a download ready, so an unsupported
    checkpoint (e.g. a GPT-2-family config using `n_head` instead of the
    Llama-style `num_attention_heads` this codebase's parser expects —
    confirmed live 2026-07-13 with hf-internal-testing/tiny-random-gpt2)
    fails here with a clear message instead of as a raw KeyError traceback
    on the next inference call."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status: dict[str, dict] = {}

    def status(self, model_id: str) -> dict | None:
        with self._lock:
            st = self._status.get(model_id)
            return dict(st) if st is not None else None

    def pending_entries(self) -> list[dict]:
        """Non-ready models, for GET /v1/models to advertise download
        progress instead of a client having to guess why a model 404s."""
        with self._lock:
            out = []
            for mid, st in self._status.items():
                if st["state"] == "ready":
                    continue
                entry = {"id": mid, "object": "model", "owned_by": "vmodel",
                         "vmodel_download_status": st["state"]}
                if st["error"]:
                    entry["vmodel_download_error"] = st["error"]
                out.append(entry)
            return out

    def start(self, model_id: str, base: str, target: Path):
        with self._lock:
            existing = self._status.get(model_id)
            if existing is not None and existing["state"] == "downloading":
                return  # already in flight; don't start a second fetch
            self._status[model_id] = {"state": "downloading", "error": None, "started_at": time.time()}

        def _run():
            try:
                import shutil

                free = shutil.disk_usage(ROOT).free
                if free < _MIN_FREE_BYTES_FOR_AUTO_DOWNLOAD:
                    raise RuntimeError(
                        f"only {free / 1e9:.1f} GB free, need at least "
                        f"{_MIN_FREE_BYTES_FOR_AUTO_DOWNLOAD / 1e9:.0f} GB free")
                from huggingface_hub import snapshot_download

                print(f"[server] pulling {model_id} from HF -> {target}", flush=True)
                snapshot_download(model_id, local_dir=str(target),
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "merges.txt", "vocab*"],
                                  ignore_patterns=["original/*", "metal/*"])
                from .config import ModelConfig

                try:
                    ModelConfig.from_dir(target)
                except KeyError as e:
                    raise RuntimeError(
                        f"downloaded '{model_id}' but its config.json is missing the "
                        f"expected key {e} -- this checkpoint's architecture isn't one "
                        f"this runtime's config parser supports yet (it expects "
                        f"Llama-style key names; e.g. GPT-2-family configs using `n_head` "
                        f"instead of `num_attention_heads` hit exactly this). Files are on "
                        f"disk at {target} but will not be served.") from e
                with self._lock:
                    self._status[model_id] = {"state": "ready", "error": None,
                                              "started_at": self._status[model_id]["started_at"]}
                print(f"[server] {model_id} ready -> {target}", flush=True)
            except Exception as e:
                # 2026-07-13: remove any partial/invalid directory now, not
                # just mark the in-memory status failed — DOWNLOADS state is
                # per-process, so a leftover dir containing a config.json
                # would look like a perfectly normal LOCAL model to
                # _registry()'s plain filesystem scan on the NEXT server
                # start, silently reintroducing this exact bug across a
                # restart (found live while testing this fix).
                try:
                    import shutil as _shutil

                    if target.exists():
                        _shutil.rmtree(target)
                except Exception:
                    pass
                with self._lock:
                    started = self._status.get(model_id, {}).get("started_at", time.time())
                    self._status[model_id] = {"state": "failed", "error": f"{type(e).__name__}: {e}",
                                              "started_at": started}
                print(f"[server] download FAILED for {model_id}: {type(e).__name__}: {e}", flush=True)

        threading.Thread(target=_run, daemon=True, name=f"download-{base}").start()


DOWNLOADS = DownloadManager()


def split_model_mode(model_id: str) -> tuple[str, str | None]:
    """Model-ID PREFIX convention for picking GOAL/Sub-Goal (lossless) vs
    Side-Quest (fast) mode: "lossy-SmolLM2-135M" -> ("SmolLM2-135M", "fast").
    No prefix -> ("SmolLM2-135M", None), meaning "use the default"
    (lossless) unless overridden some other way.

    2026-07-13 (user request, revised from an earlier `:suffix` design
    after feedback that a non-standard header wasn't ideal): the `model`
    field is the ONE thing every supported protocol (OpenAI chat/
    completions and Responses, Anthropic Messages) already has as a plain
    string, so a naming convention on it needs zero protocol-specific
    plumbing — and unlike a suffix, a PREFIX lets `GET /v1/models` advertise
    every mode as its own first-class, independently-selectable model id
    (see _advertised_model_ids()) rather than requiring the client to already
    know an undocumented suffix syntax. The `X-VModel-Mode` header /
    `vmodel_mode` body field still work as a higher-precedence override for
    backward compatibility, but are no longer the primary, advertised
    mechanism."""
    if model_id.lower().startswith(LOSSY_LONG_PREFIX):
        return model_id[len(LOSSY_LONG_PREFIX):], "fast-long"
    if model_id.lower().startswith(LOSSY_PREFIX):
        return model_id[len(LOSSY_PREFIX):], "fast"
    return model_id, None


def _resolve(model_id: str) -> Path:
    """Unknown model ids are downloaded via DOWNLOADS (background thread,
    non-blocking) rather than inline. 2026-07-13: this used to call
    snapshot_download() directly, synchronously, inside the locked request
    handler — live-tested and confirmed to (a) hang the client connection
    with zero progress visibility for as long as the fetch takes, and
    (b) surface config-parsing failures for unsupported architectures as a
    raw KeyError traceback. Raises ModelDownloading / ModelDownloadFailed
    for the caller to turn into a clear, immediate JSON response instead of
    blocking on either case.

    DOWNLOADS.status() is checked BEFORE the registry scan, not after —
    live-tested and confirmed the other order is a real bug: the raw files
    land on disk (config.json included) within ~1-2s, well before this
    manager's post-download validation finishes, so a second request in
    that window would find the model "local" via the registry's plain
    dir/config.json scan and skip validation/failure state entirely,
    leaking a raw KeyError straight through. Checking DOWNLOADS state first
    means once a model_id has gone through this manager, ITS state machine
    (not a filesystem race) decides what the caller sees."""
    st = DOWNLOADS.status(model_id)
    if st is not None:
        if st["state"] == "downloading":
            raise ModelDownloading(model_id, st)
        if st["state"] == "failed":
            raise ModelDownloadFailed(model_id, st["error"])
        # state == "ready": this model_id went through an auto-download; every
        # resolve of it (even long after "ready") passes through here, which is
        # the natural hook for PackManager's request-count-based auto-pack
        # trigger — fall through to the registry lookup below either way.
        PACKS.note_request(model_id, ROOT / "models" / model_id.split("/")[-1])

    reg = _registry()
    if model_id in reg:
        return reg[model_id]
    base = model_id.split("/")[-1]
    if base in reg:
        return reg[base]

    target = ROOT / "models" / base
    DOWNLOADS.start(model_id, base, target)
    raise ModelDownloading(model_id, DOWNLOADS.status(model_id))


def _dense_fast_resident_bytes(cfg) -> int:
    """Conservative Q4 cache estimate used to decide whether to pin embeddings.

    Untied BF16 embeddings cannot consume a lazy predicted token through the
    row-paged sidecar, so dense pipelining requires the full input matrix. Only
    opt in when it and an MXFP4 body/head fit with substantial cache headroom.
    """
    try:
        hidden = int(cfg.hidden_size)
        heads = int(cfg.num_attention_heads)
        kv_heads = int(cfg.num_key_value_heads)
        head_dim = int(cfg.head_dim)
        intermediate = int(cfg.intermediate_size)
        layers = int(cfg.num_hidden_layers)
        vocab = int(cfg.vocab_size)
    except (AttributeError, TypeError, ValueError):
        return 2**63 - 1
    packed = 4 / 8 + 1 / 32  # MXFP4 payload plus one uint8 scale/group
    attn = (
        hidden * head_dim * (heads + 2 * kv_heads)
        + head_dim * heads * hidden
    )
    mlp = 3 * hidden * intermediate
    norms_and_biases = layers * (
        2 * hidden
        + (hidden + 2 * kv_heads * head_dim
           if getattr(cfg, "attention_bias", False) else 0)
    ) * 2
    embedding = vocab * hidden * 2  # deliberately stays BF16 for indexed lookup
    lm_head = 0 if cfg.tie_word_embeddings else vocab * hidden * packed
    return int(embedding + lm_head + layers * (attn + mlp) * packed
               + norms_and_biases)


def _dense_lossless_resident_bytes(cfg) -> int:
    """Conservative exact-weight footprint for dense resident scheduling."""
    try:
        hidden = int(cfg.hidden_size)
        heads = int(cfg.num_attention_heads)
        kv_heads = int(cfg.num_key_value_heads)
        head_dim = int(cfg.head_dim)
        intermediate = int(cfg.intermediate_size)
        layers = int(cfg.num_hidden_layers)
        vocab = int(cfg.vocab_size)
    except (AttributeError, TypeError, ValueError):
        return 2**63 - 1
    attn = (
        hidden * head_dim * (heads + 2 * kv_heads)
        + head_dim * heads * hidden
    )
    mlp = 3 * hidden * intermediate
    per_layer_small = 2 * hidden
    if getattr(cfg, "attention_bias", False):
        per_layer_small += hidden + 2 * kv_heads * head_dim
    embeddings = vocab * hidden
    lm_head = 0 if cfg.tie_word_embeddings else vocab * hidden
    # Include final norm plus 5% for architecture-specific small tensors the
    # common Llama/Qwen shape model does not enumerate (for example Q/K norms).
    exact = 2 * (
        embeddings + lm_head + layers * (attn + mlp + per_layer_small)
        + hidden
    )
    return int(exact * 1.05)


def _checkpoint_payload_bytes(model_dir: Path) -> int:
    """Best available exact tensor-payload size without materializing weights."""
    index_path = model_dir / "model.safetensors.index.json"
    try:
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            total = index.get("metadata", {}).get("total_size")
            if isinstance(total, int) and total > 0:
                return total
            shards = set(index.get("weight_map", {}).values())
            if shards:
                return sum((model_dir / shard).stat().st_size for shard in shards)
        return sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
    except (OSError, TypeError, ValueError):
        return 0


class EngineManager:
    """One resident engine; (model_dir, mode) keyed swap."""

    def __init__(self):
        self._key = None
        self._engine = None
        self._lock = threading.Lock()

    def close(self):
        """Release the resident engine and flush its durable learned state."""
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
                self._key = None

    def get(self, model_dir: Path, mode: str):
        import mlx.core as mx

        from .config import ModelConfig
        from .engine import RuntimeConfig, StreamingEngine
        from .path_resolver import resolve_model_dir

        yarn_factor = 0.0
        if mode == "fast-long":
            try:
                yarn_factor = float(os.environ.get("VMODEL_FAST_LONG_YARN_FACTOR", "2"))
            except ValueError as e:
                raise RequestValidationError(
                    "VMODEL_FAST_LONG_YARN_FACTOR must be numeric") from e
            if not math.isfinite(yarn_factor) or yarn_factor <= 1:
                raise RequestValidationError(
                    "VMODEL_FAST_LONG_YARN_FACTOR must be finite and greater than 1")
        key = (str(model_dir), mode, yarn_factor.hex())
        # A healthy resident engine owns all state it needs. Return before
        # touching model storage so a transient NAS disconnect cannot stall or
        # fail an otherwise cache-hot request.
        with self._lock:
            if self._key == key and self._engine is not None:
                return self._engine

        model_dir = resolve_model_dir(model_dir)
        key = (str(model_dir), mode, yarn_factor.hex())
        # Validate the requested profile before evicting a healthy resident
        # engine. ModelConfig.from_dir is read-only and has the same remount
        # retry behavior as WeightStore.
        cfg_probe = ModelConfig.from_dir(model_dir)
        mtype = cfg_probe.model_type
        untied = not cfg_probe.tie_word_embeddings
        if mode == "lossless" and _is_voom_lossy_checkpoint(model_dir):
            raise RequestValidationError(
                "this checkpoint is a vOOM-derived lossy artifact; request it "
                f"as {LOSSY_PREFIX}{model_dir.name} (fast mode), or use its "
                "original source checkpoint for lossless mode")
        if mode == "fast-long" and mtype != "qwen2":
            raise RequestValidationError(
                "fast-long is currently supported only for Qwen2 checkpoints")
        with self._lock:
            # Recheck after the read-only probe for callers that use
            # EngineManager outside the server's coarser INFER_LOCK.
            if self._key == key and self._engine is not None:
                return self._engine
            if self._engine is not None:
                print(f"[server] swapping engine {self._key} -> {key}", flush=True)
                self._engine.close()
                self._engine = None
                self._key = None
                mx.clear_cache()
            else:
                self._key = None

            # Memory-only chunking fixes the local long-prefill scratch spike:
            # Qwen2.5-1.5B measured 5.09GB at 32K and 7.01GB at 64K with exact
            # tokens. It is deliberately separate from checkpoint persistence;
            # the old coupled path wrote a growing full KV after every chunk.
            # 4096 is the conservative dense-model default. GLM overrides it
            # below with 64 plus F68 adaptation; neither value is itself an
            # expert live-memory proof. Short prompts remain one ordinary sweep.
            # 2026-07-15: F37's disk-store LRU budget (default 2 GB) was never
            # actually configurable at the server layer -- prompt_kv_max_mb is
            # a real RuntimeConfig field, but nothing here read an env var for
            # it, so an operator with a large fast local drive (e.g. a 4 TB
            # NVMe) had no way to raise it without editing this file directly.
            try:
                prompt_kv_max_mb = int(os.environ.get("VMODEL_PROMPT_KV_MAX_MB", "2000"))
                prompt_kv_min_tokens = int(os.environ.get(
                    "VMODEL_PROMPT_KV_MIN_TOKENS", "2048"))
                prompt_kv_journal_chunk_size = int(os.environ.get(
                    "VMODEL_PROMPT_KV_JOURNAL_CHUNK_SIZE", "512"))
            except ValueError as e:
                raise ValueError(
                    "VMODEL prompt KV limits must be integers") from e
            if prompt_kv_max_mb < 0:
                raise ValueError("VMODEL_PROMPT_KV_MAX_MB must be >= 0 (0 = unbounded)")
            if prompt_kv_min_tokens < 0:
                raise ValueError("VMODEL_PROMPT_KV_MIN_TOKENS must be >= 0")
            if prompt_kv_journal_chunk_size <= 0:
                raise ValueError(
                    "VMODEL_PROMPT_KV_JOURNAL_CHUNK_SIZE must be positive")
            raw_hash_value = os.environ.get(
                "VMODEL_REQUIRE_RAW_WEIGHT_HASHES", "0")
            if raw_hash_value not in ("0", "1"):
                raise ValueError(
                    "VMODEL_REQUIRE_RAW_WEIGHT_HASHES must be 0 or 1")
            rc = RuntimeConfig(prefetch_depth=2, pin_lm_head=True, embed_rows=untied,
                               prompt_kv_dir=str(ROOT / ".kv_prompts"),
                               prompt_kv_max_mb=prompt_kv_max_mb,
                               prompt_kv_min_tokens=prompt_kv_min_tokens,
                               prompt_kv_journal_chunk_size=(
                                   prompt_kv_journal_chunk_size),
                               require_raw_weight_hashes=(
                                   raw_hash_value == "1"),
                               prefill_chunk_size=4096,
                               prefill_checkpoint_every=0)
            if mtype == "gpt_oss":
                rc.max_weight_cache_mb, rc.pin_first_layers = 6500, 36
            elif mtype == "glm_moe_dsa":
                rc.max_weight_cache_mb = 5000
                if mode in ("fast", "fast-long"):
                    # Side-quest GLM used to advertise a lossy model id while
                    # retaining the exact same BF16 weight path as lossless.
                    # Quantize cache pages (or preserve QTensor pages from a
                    # pre-quantized MLX checkpoint) so the mode is real.
                    rc.quant_bits = 4
                    # A decode position routes to exactly eight experts, so Q4
                    # pages fit comfortably as one compute batch. Large prefills
                    # retain q=1 because their union can approach all 256 experts.
                    rc.decode_expert_fetch_batch = int(
                        os.environ.get("VMODEL_FAST_DECODE_EXPERT_BATCH", "8"))
                    if rc.decode_expert_fetch_batch <= 0:
                        raise ValueError(
                            "VMODEL_FAST_DECODE_EXPERT_BATCH must be positive")
                # F37 v6 is durable, but real GLM prompt-journal restore has not
                # passed the same released-DSA >index_topk end-to-end gate that
                # still constrains this public profile. Keep it off until that
                # model-scale correctness proof exists; direct experiments may
                # opt in explicitly.
                rc.prompt_kv_dir = ""
                # Released DSA semantics for multi-position prefill beyond
                # index_topk are still under F22/F33.  Direct experiments may
                # opt into longer contexts, but the public server must fail
                # closed instead of silently serving dense, non-released math.
                rc.context_bound = int(cfg_probe.index_topk or 2048)
                # 2026-07-14: a direct-engine (non-server) test against the REAL
                # GLM-5.2 checkpoint with the fixed prefill_chunk_size=4096
                # default (bypassing context_bound on purpose, to see what F22/
                # F33 unlocking longer contexts would eventually hit) forced a
                # single MoE layer's per-chunk expert union toward all 256
                # routed experts at once -- a governor.reserve() request for
                # ~22GB. First fix attempt (adaptive chunk controller, F68,
                # small initial chunk=64) REDUCED but did NOT solve this: a
                # re-test still hit a 16GB reservation / 9.8GB metal. Root
                # cause is a coupon-collector effect -- 256 routed experts, 8
                # active/token, cold per-layer cache -- so a chunk's expert
                # union approaches the full 256 regardless of chunk size; it
                # does not shrink proportionally. Cache-only sub-fetching was
                # then found insufficient: a returned full-union dict kept all
                # evicted arrays strongly referenced. F74-v2 therefore bounds
                # one complete fetch + expert compute + mx.eval + release
                # lifetime in runtime/glm.py. This matters even BELOW DSA's
                # index_topk: a normal 64-position prefill can route to most
                # experts, so context_bound is a DSA correctness gate, not an
                # expert-memory safety proof. Keep the adaptive controller too
                # (throughput:
                # smaller reservations, fewer wasted disk reads) but do not
                # rely on it alone. F74-v2 remains quarantined pending the same
                # real-scale peak test that exposed the original problem.
                rc.prefill_chunk_size = 64
                rc.adaptive_chunk_size = True
                # Let F68 resample F16's live admission boundary each chunk.
                # A fixed target measured on one 16-GB host needlessly throttles
                # machines with more headroom and can be unsafe on a busy one.
                rc.adaptive_chunk_safe_bytes = 0
                # Multi-position prefill remains at the fail-closed q=1
                # fallback. Fast-mode single-position decode has a separately
                # bounded q=8 path above; lossless keeps q=1 everywhere.
                rc.expert_fetch_batch = 1
            elif mtype == "kimi_k25":
                # 2026-07-19: K2.5's vocab_size (163840) x hidden_size (7168)
                # combination makes its untied lm_head unusually large
                # (~2.35 GB bf16) -- confirmed via direct measurement to be
                # the dominant contributor to the memory-governor rejections
                # every real K2.5 request hit that day (the pinned
                # "incoming" reserve of ~4.7GB closely matched embed_tokens
                # + lm_head both bf16-resident). Stream the lm_head instead
                # of pinning it (F02's StreamedLMHead -- bit-identical, not
                # an approximation, see runtime/lm_head_stream.py); embed_rows
                # already streams the embedding table via the untied-model
                # default above. Kimi Linear does NOT need this override --
                # its much smaller hidden_size keeps its lm_head modest.
                rc.pin_lm_head = False
                rc.stream_lm_head = True
                # A 4 GB cache still admitted enough concurrent demand/prefetch
                # growth to trigger real swap-outs before F42 reached its next
                # large reservation. Lossless K2.5 already computes experts at
                # the minimum q=1 lifetime, so the remaining exact paging lever
                # is lower residency, not a fictional q<1. Keep 1.5 GB of LRU
                # pages (with pre-fetch eviction before known-size trunk pages)
                # and disable speculative prefetch; demand misses still
                # stream the same released tensors and preserve arithmetic.
                # The live governor may shrink this further toward its 1.5 GB
                # floor when system headroom requires it.
                rc.max_weight_cache_mb = 1500
                rc.prefetch_depth = 0
            elif mtype == "qwen3_5_moe":
                # Qwen3.6-35B-A3B retains Qwen3.5's architecture id. Thirty
                # DeltaNet layers carry fixed recurrent+conv state that the
                # token-indexed prompt-KV journal cannot serialize or trim.
                # F37's token-indexed prompt journal cannot represent the
                # recurrent half. The in-memory hot endpoint can: it transfers
                # the complete KVCache + KDAStateCache object and only accepts
                # exact endpoints/extensions (never a trimmed branch).
                rc.prompt_kv_dir = ""
                rc.hot_prompt_kv = True
                try:
                    rc.hot_prompt_kv_slots = int(os.environ.get(
                        "VMODEL_QWEN35_HOT_KV_SLOTS", "2"))
                    rc.hot_prompt_kv_min_tokens = int(os.environ.get(
                        "VMODEL_QWEN35_HOT_KV_MIN_TOKENS", "16"))
                except ValueError as error:
                    raise ValueError(
                        "VMODEL_QWEN35_HOT_KV settings must be integers") from error
                if rc.hot_prompt_kv_slots <= 0:
                    raise ValueError(
                        "VMODEL_QWEN35_HOT_KV_SLOTS must be positive")
                if rc.hot_prompt_kv_min_tokens < 0:
                    raise ValueError(
                        "VMODEL_QWEN35_HOT_KV_MIN_TOKENS must be non-negative")
                rc.hot_prompt_kv_persist_dir = os.environ.get(
                    "VMODEL_QWEN35_HOT_KV_PERSIST_DIR",
                    str(ROOT / ".kv_hybrid"))
                try:
                    rc.max_weight_cache_mb = int(os.environ.get(
                        "VMODEL_QWEN35_WEIGHT_CACHE_MB", "6000"))
                except ValueError as error:
                    raise ValueError(
                        "VMODEL_QWEN35_WEIGHT_CACHE_MB must be an integer") from error
                if not 1500 <= rc.max_weight_cache_mb <= 7000:
                    raise ValueError(
                        "VMODEL_QWEN35_WEIGHT_CACHE_MB must be in [1500, 7000]")
                rc.fast_dirs = (str(
                    Path.home() / "vmodel_fast_tier" / model_dir.name),)
                rc.prefill_chunk_size = 512
                rc.hot_prompt_kv_chunk_size = rc.prefill_chunk_size
                # 2026-07-20: this was q=1 (one expert fetched at a time,
                # serially) copied from GLM-5.2's fail-closed prefill
                # fallback -- but that caution was calibrated to GLM's
                # ~75.5MB-per-expert danger zone (a real 16-22GB reservation
                # incident, see the glm_moe_dsa block above), not
                # re-evaluated for this model. Qwen3.6's routed experts are
                # ~6.3MB each (same figure the decode comment below already
                # relies on) -- even a full 256-expert worst-case union is
                # ~1.6GB, nowhere near GLM's regime. q=1 also serialized one
                # governor-reservation-plus-lock round trip per expert, which
                # is cheap next to a slow USB drive's own per-read latency
                # but became the dominant cost once this repo's storage
                # moved to a real NVMe (~3 GB/s measured, see
                # runtime/disk_bench.py) -- live-confirmed: a 134-tool,
                # ~30K-token prompt prefilled at ~13 tok/s, and process
                # sampling showed real time going to thread-sync waits
                # around each single-expert fetch, not to disk reads
                # themselves. Raised to 8 to match decode's already-proven
                # value below and let get_many's shard-grouped batching
                # actually engage during prefill too. This does NOT bypass
                # memory safety: governor.admissible_units() (called inside
                # _iter_expert_batches) still adaptively clamps the live
                # batch size down to whatever current headroom allows,
                # exactly as it already does for decode's batch=8 -- this
                # only raises the ceiling attempted when there's room, never
                # forces a bigger batch under pressure.
                rc.expert_fetch_batch = 8
                # One decode position activates exactly eight ~6.3 MB experts
                # per layer. Fetching them as one archive-coalesced batch is
                # comfortably bounded and avoids eight serialized disk waits;
                # multi-position prefill now batches the same width above.
                rc.decode_expert_fetch_batch = 8
                if mode in ("fast", "fast-long"):
                    # Initial side-quest profile: quantize only expert MLP
                    # matrices. DeltaNet, gated full attention, routers, shared
                    # scalar gates, embeddings, and the exact head remain BF16.
                    # This is explicitly separate from lossless mode and must
                    # pass a real-model quality gate before stronger transforms.
                    rc.quant_bits = 4
                    rc.quant_mode = "mxfp4"
                    rc.quant_group_size = 32
                    rc.quant_min_dim = 0
                    rc.quant_attention = False
                    rc.quant_router = False
                    rc.quant_lm_head = False
                    # NOT rc.max_weight_cache_mb = 6000 here: that stomped the
                    # VMODEL_QWEN35_WEIGHT_CACHE_MB-configured value set above
                    # right before every fast/fast-long request regardless of
                    # what an operator asked for (2026-07-20, live-confirmed --
                    # a tool-heavy real request needed a smaller resident
                    # budget to leave headroom for the expert-fetch reserve,
                    # and this line silently overrode that knob back to 6000
                    # every time).
            else:
                rc.max_weight_cache_mb = 6000
                if mtype == "kimi_linear":
                    # KDA state is a recurrent fold, not token-indexed KV.
                    # Engine validation now fails closed if a caller tries to
                    # persist only the ordinary KV half of this hybrid state.
                    rc.prompt_kv_dir = ""
                if mode in ("fast", "fast-long"):  # side-quest: lossy 4-bit resident cache
                    rc.quant_bits = 4
                    # Local Qwen A/B: MXFP4 retained coherent math/chat/code
                    # behavior, used ~48MB less, and decoded ~2.4% faster than
                    # affine Q4. Q2/Q3 were faster but failed basic quality probes.
                    rc.quant_mode = "mxfp4"
                    rc.quant_group_size = 32
                    rc.quant_min_dim = 0
                    rc.max_weight_cache_mb = 7000
                    if (not getattr(cfg_probe, "num_experts", 0)
                            and not getattr(cfg_probe, "vision_config", None)):
                        default_fast_cache_floor = (
                            "600" if os.environ.get(
                                "VMODEL_FAST_TOOL_GATEWAY", "0") == "1"
                            else "1500")
                        try:
                            rc.min_weight_cache_mb = int(os.environ.get(
                                "VMODEL_FAST_WEIGHT_CACHE_FLOOR_MB",
                                default_fast_cache_floor))
                        except ValueError as error:
                            raise ValueError(
                                "VMODEL_FAST_WEIGHT_CACHE_FLOOR_MB must be an integer"
                            ) from error
                        if rc.min_weight_cache_mb <= 0:
                            raise ValueError(
                                "VMODEL_FAST_WEIGHT_CACHE_FLOOR_MB must be positive")
                    if getattr(cfg_probe, "num_experts", 0):
                        # Real OLMoE A/B (64 experts, top-8): quantizing only
                        # experts retained much longer BF16 token prefixes and
                        # exact prime-list behavior at ~20 tok/s versus ~21 for
                        # quantize-everything and 6-9 BF16. Attention, routing,
                        # and the untied head add ~1.1 GB but remain safely under
                        # the target bound. Routing and argmax are both
                        # discontinuous, so those tiny savings are a poor
                        # quality trade. Attention gets a separate MXFP8-only
                        # resident transform below; MXFP4 attention remains
                        # rejected by the quality gate.
                        rc.quant_attention = False
                        rc.quant_router = False
                        rc.quant_lm_head = False
                        rc.resident_moe_decode = cfg_probe.model_type == "olmoe"
                        rc.fused_swiglu = rc.resident_moe_decode
                        # The untied 50K-token BF16 head is ~206 MB read per
                        # decode step. Quantized candidate search followed by
                        # exact BF16 gather_mm scoring of the top 32 retains the
                        # candidate winner. It remains explicitly lossy/greedy-
                        # only because candidate recall is empirical, not proved.
                        rc.rerank_lm_head = (
                            rc.resident_moe_decode
                            and not cfg_probe.tie_word_embeddings
                        )
                        # Resident OLMoE feeds the lazy predicted token directly
                        # into the next embedding gather. A row-paged sidecar
                        # must first convert that token to Python and breaks the
                        # overlap. Its embedding is small enough to pin; large
                        # streamed MoEs (GLM) retain row paging above.
                        if rc.resident_moe_decode:
                            rc.embed_rows = False
                            # A separate stepped-cache class avoids hot-path
                            # branching and won from the smallest controlled
                            # gate (+1.7% at 32 prompt tokens) through 3.5K
                            # context (+90% decode), with identical tokens.
                            rc.stepped_kv_threshold = 1
                            # Match stock MLX-LM's portable-checkpoint schedule:
                            # chunks of at most 2048, with the final prompt token
                            # evaluated separately to produce endpoint logits.
                            rc.prefill_chunk_size = 2048
                            rc.prefill_last_token_separate = True
                            try:
                                disk_quant = json.loads(
                                    (model_dir / "config.json").read_text()).get(
                                        "quantization", {})
                            except (OSError, ValueError):
                                disk_quant = {}
                            try:
                                disk_bits = int(disk_quant.get("bits", 0))
                            except (TypeError, ValueError):
                                disk_bits = 0
                            expert_mxfp8 = (
                                disk_quant.get("mode") == "mxfp8"
                                and disk_bits == 8
                            )
                            if expert_mxfp8:
                                # Expert-only MXFP8 quality profile: measured
                                # 7.67 GB true Metal peak and 180.6 tok/s with
                                # exact-candidate head reranking. A 9 GB budget lets
                                # the 7.60 GB artifact pass the resident path's
                                # 15% admission discount. Runtime safety remains
                                # governed by MLX's recommended working set and
                                # live system-available unified-memory headroom.
                                rc.max_weight_cache_mb = 9000
                            else:
                                # Affine Q2 candidate search retained 3,268/
                                # 3,268 BF16-head IDs across the task gate and a
                                # diverse held-out prompt corpus. Controlled
                                # 512-token ABBA improved 1.63% over the prior
                                # MXFP4 candidate head (288.6 vs 284.0 tok/s).
                                # Apply only to this validated hybrid; the
                                # higher-fidelity MXFP8 control stays MXFP4.
                                rc.rerank_lm_head_mode = "affine"
                                rc.rerank_lm_head_bits = 2
                                rc.rerank_lm_head_group_size = 64
                                # Hybrid profile measured +19% decode throughput
                                # over expert-MXFP4 alone. On 3,072 held-out local
                                # code/prose tokens it changed NLL 3.515->3.544,
                                # while a 30-item restricted-choice gate stayed
                                # 23/30. Keep MXFP8 experts' attention BF16 so that
                                # artifact remains the higher-fidelity control.
                                rc.resident_attention_mode = "mxfp8"
                                rc.resident_attention_bits = 8
                                rc.resident_attention_group_size = 32
                    else:
                        if cfg_probe.vision_config:
                            # Qwen3-VL-2B real-model gate: uniform MXFP4 kept
                            # color/shape/count but reversed two-image ordering
                            # ("Green, Blue" -> "Blue, Green"). Quantizing only
                            # the text MLP restored every accepted multimodal
                            # answer while retaining ~2.5x faster short decode
                            # and 2.71GB active memory. Uniform MXFP8 was no
                            # faster and used another ~0.51GB, so it is dominated.
                            # The vision tower itself is cached as released BF16.
                            rc.quant_attention = False
                            rc.quant_lm_head = False
                            rc.quantize_tied_lm_head = False
                            try:
                                rc.vision_max_patches = int(os.environ.get(
                                    "VMODEL_FAST_VISION_MAX_PATCHES", "1024"))
                            except ValueError as e:
                                raise RequestValidationError(
                                    "VMODEL_FAST_VISION_MAX_PATCHES must be an integer") from e
                            if not 256 <= rc.vision_max_patches <= 4096:
                                raise RequestValidationError(
                                    "VMODEL_FAST_VISION_MAX_PATCHES must be in [256, 4096]")
                        else:
                            rc.quantize_tied_lm_head = True
                        rc.resident_fast_decode = True
                        # Exact scheduling-only optimization over the already
                        # selected lossy weights. Qwen-1.5B gained 5.2%/3.2%
                        # total at 128/512 positions; Qwen-7B gained 4.8%/1.9%,
                        # with identical IDs. At 2K the 7B gain fell below 1%
                        # and peak rose, so keep the side-quest bound tighter.
                        rc.resident_fast_prefill_limit = 512
                        rc.fused_swiglu = True
                        # Dense Qwen A/B: short 1.5B decode slightly favors
                        # concatenation, while stepped KV wins beyond 512
                        # requested positions (+12% around 2K on both 1.5B/7B).
                        rc.stepped_kv_threshold = 512
                    if (rc.embed_rows and not cfg_probe.vision_config
                            and not getattr(cfg_probe, "num_experts", 0)
                            and _dense_fast_resident_bytes(cfg_probe)
                            <= int(rc.max_weight_cache_mb * 1_000_000 * 0.85)):
                        # Qwen2.5-7B proof: 4.85GB actual cache under this 7GB
                        # budget and 84.9 tok/s warm. Pinning its 1.09GB BF16
                        # embedding unlocks lazy-token pipelining; row paging
                        # would force every predicted id back through Python.
                        rc.embed_rows = False
                    # A 44K-token Qwen KV is ~1.26 GB. F37 synchronously wrote
                    # it before first-token delivery and again after generation;
                    # the 2 GB LRU then evicted the first copy. Keep the state
                    # already resident in this single-owner engine instead.
                    # Reuse only complete 4096-token prefill chunks so warm and
                    # cold suffixes retain the same kernel/chunk boundaries.
                    rc.prompt_kv_dir = ""
                    # Vision generation owns a separate KV path today. Keeping
                    # a text hot-KV while it allocates another full state would
                    # violate the 16-GB single-owner bound, so vision stays cold.
                    rc.hot_prompt_kv = not bool(cfg_probe.vision_config)
                    rc.hot_prompt_kv_chunk_size = rc.prefill_chunk_size
                    # 2026-07-15: default (1 slot) preserves the original
                    # single-slot behavior, which a live test against a real
                    # harness (kai-desktop) showed fails on completely
                    # ordinary traffic -- any interleaved request (a
                    # title-generation call, a working-memory update) evicts
                    # the main conversation's retained state before its own
                    # next turn can ever reuse it. Configurable rather than
                    # silently bumped, since each slot holds a full KV state
                    # proportional to its own context length (~1.26 GB for a
                    # real 44K-token Qwen2.5-1.5B conversation) -- a real
                    # tradeoff against live unified-memory headroom, not a free
                    # win, so the operator should size it deliberately.
                    try:
                        rc.hot_prompt_kv_slots = int(
                            os.environ.get("VMODEL_HOT_PROMPT_KV_SLOTS", "1"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_SLOTS must be an integer") from e
                    if rc.hot_prompt_kv_slots <= 0:
                        raise ValueError("VMODEL_HOT_PROMPT_KV_SLOTS must be positive")
                    # 2026-07-15 later: raising hot_prompt_kv_slots is still a
                    # guess at "how many interleaved calls happen per turn" --
                    # a live test against kai-desktop showed that count is
                    # itself variable (one interleaved call between one pair
                    # of real turns, two between the next), so any fixed slot
                    # count can still be exceeded and evict the expensive
                    # conversation slot. Refuse to RETAIN (not: refuse to
                    # match against) a slot for any prompt shorter than this
                    # many tokens -- tiny title-gen/working-memory calls never
                    # occupy a slot at all, so they can never evict one.
                    try:
                        rc.hot_prompt_kv_min_tokens = int(
                            os.environ.get("VMODEL_HOT_PROMPT_KV_MIN_TOKENS", "2048"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS must be an integer") from e
                    if rc.hot_prompt_kv_min_tokens < 0:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS must be >= 0")
                    try:
                        rc.hot_prompt_kv_min_available_mb = int(os.environ.get(
                            "VMODEL_HOT_PROMPT_KV_MIN_AVAILABLE_MB", "0"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_MIN_AVAILABLE_MB must be an "
                            "integer") from e
                    if rc.hot_prompt_kv_min_available_mb < 0:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_MIN_AVAILABLE_MB must be >= 0")
                    if os.environ.get("VMODEL_FAST_TOOL_GATEWAY", "0") == "1":
                        try:
                            gateway_kv_min = int(os.environ.get(
                                "VMODEL_FAST_TOOL_GATEWAY_KV_MIN_TOKENS", "1024"))
                            gateway_kv_chunk = int(os.environ.get(
                                "VMODEL_FAST_TOOL_GATEWAY_KV_CHUNK_SIZE", "512"))
                        except ValueError as e:
                            raise ValueError(
                                "VMODEL fast tool gateway KV settings must be "
                                "integers") from e
                        if gateway_kv_min < 0:
                            raise ValueError(
                                "VMODEL_FAST_TOOL_GATEWAY_KV_MIN_TOKENS must be >= 0")
                        if (not 256 <= gateway_kv_chunk <= 4096
                                or gateway_kv_chunk & (gateway_kv_chunk - 1)):
                            raise ValueError(
                                "VMODEL_FAST_TOOL_GATEWAY_KV_CHUNK_SIZE must be "
                                "a power of two in [256, 4096]")
                        # The virtual-only real harness prompt is ~1.9K tokens,
                        # just below the ordinary 2K anti-thrash threshold. Keep
                        # this smaller threshold scoped to the explicit gateway
                        # profile so its stable router prefix becomes warm while
                        # the normal server still rejects tiny helper slots.
                        rc.hot_prompt_kv_min_tokens = min(
                            rc.hot_prompt_kv_min_tokens, gateway_kv_min)
                        # Agent transcripts usually diverge at the previous
                        # generation marker, so branch reuse floors to this
                        # boundary. A 2K boundary left 1-2K old tokens to prefill
                        # on nearly every follow-up. The gateway pays more small
                        # durable segments on its first call in exchange for a
                        # <=511-token alignment loss thereafter.
                        rc.prefill_chunk_size = gateway_kv_chunk
                        rc.hot_prompt_kv_chunk_size = gateway_kv_chunk
                    try:
                        adaptive_spill_default = (
                            "256" if os.environ.get(
                                "VMODEL_FAST_TOOL_GATEWAY", "0") == "1"
                            else "0")
                        rc.adaptive_kv_spill_mb = int(os.environ.get(
                            "VMODEL_FAST_KV_ADAPTIVE_SPILL_MB",
                            adaptive_spill_default))
                        rc.adaptive_kv_spill_prefill_chunk_size = int(
                            os.environ.get(
                                "VMODEL_FAST_KV_ADAPTIVE_PREFILL_CHUNK_SIZE",
                                "512"))
                    except ValueError as error:
                        raise ValueError(
                            "VMODEL_FAST_KV_ADAPTIVE spill settings must be "
                            "integers") from error
                    if rc.adaptive_kv_spill_mb < 0:
                        raise ValueError(
                            "VMODEL_FAST_KV_ADAPTIVE_SPILL_MB must be >= 0")
                    if not 1 <= rc.adaptive_kv_spill_prefill_chunk_size <= 4096:
                        raise ValueError(
                            "VMODEL_FAST_KV_ADAPTIVE_PREFILL_CHUNK_SIZE must "
                            "be in [1, 4096]")
                    tool_pic_value = os.environ.get(
                        "VMODEL_FAST_TOOL_PIC", "1")
                    if tool_pic_value not in ("0", "1"):
                        raise ValueError("VMODEL_FAST_TOOL_PIC must be 0 or 1")
                    # Position-independent tool KV is a separately gated lossy
                    # profile. Canonical exact-prefix reuse still wins whenever
                    # applicable. Text Qwen/OLMoE use the hot LRU; Qwen3-VL
                    # uses its single-owner multimodal prompt cache.
                    rc.tool_pic = bool(
                        tool_pic_value == "1"
                        and (rc.hot_prompt_kv or cfg_probe.vision_config)
                        and mtype in (
                            "qwen2", "qwen3", "olmoe", "qwen3_vl"))
                    shared_pic_value = os.environ.get(
                        "VMODEL_FAST_TOOL_PIC_SHARED_PAGES", "0")
                    if shared_pic_value not in ("0", "1"):
                        raise ValueError(
                            "VMODEL_FAST_TOOL_PIC_SHARED_PAGES must be 0 or 1")
                    # Experimental text-only MiniPIC path. Keep it separately
                    # opt-in until the real-model memory/latency gates below the
                    # unit suite prove it beats the established private-copy PIC.
                    rc.tool_pic_shared_pages = bool(
                        shared_pic_value == "1"
                        and mtype == "qwen2"
                        # Pool/scatter overhead dominated the locally gated
                        # 1.5B/Qwen3/OLMoE paths. At 7B-class width, copied KV is
                        # large enough for sharing to clear both prefill and
                        # whole-request >1% gates.
                        and cfg_probe.hidden_size >= 3000
                        and not cfg_probe.vision_config
                        and not cfg_probe.num_experts)
                    try:
                        rc.tool_pic_repair_tokens = int(os.environ.get(
                            "VMODEL_FAST_TOOL_PIC_REPAIR_TOKENS", "4"))
                        rc.tool_pic_min_savings = int(os.environ.get(
                            "VMODEL_FAST_TOOL_PIC_MIN_SAVINGS", "128"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL_FAST_TOOL_PIC limits must be integers") from e
                    if rc.tool_pic_repair_tokens < 0:
                        raise ValueError(
                            "VMODEL_FAST_TOOL_PIC_REPAIR_TOKENS must be >= 0")
                    if rc.tool_pic_min_savings < 0:
                        raise ValueError(
                            "VMODEL_FAST_TOOL_PIC_MIN_SAVINGS must be >= 0")
                    # 2026-07-15: disk backing for the in-memory LRU above, so
                    # a conversation survives a server restart instead of
                    # paying a full cold prefill again (user-requested).
                    # "" (default) disables it -- pure in-memory, matching
                    # every prior behavior exactly. See
                    # runtime/hot_kv_persist.py for why this is deliberately
                    # not a reuse of F37's own disk store.
                    rc.hot_prompt_kv_persist_dir = os.environ.get(
                        "VMODEL_HOT_PROMPT_KV_PERSIST_DIR", "")
                    # Disk retention budget, deliberately larger than and
                    # decoupled from hot_prompt_kv_slots -- disk is meant to
                    # hold more forked/historical conversation points than
                    # memory ever needs to (see hot_kv_persist.py's gc()).
                    try:
                        rc.hot_prompt_kv_persist_max_checkpoints = int(
                            os.environ.get(
                                "VMODEL_HOT_PROMPT_KV_PERSIST_MAX_CHECKPOINTS", "64"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_PERSIST_MAX_CHECKPOINTS "
                            "must be an integer") from e
                    if rc.hot_prompt_kv_persist_max_checkpoints < 0:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_PERSIST_MAX_CHECKPOINTS must be >= 0")
                    # Exact BF16 KV can dominate dense Qwen at agent-harness
                    # prompt sizes even though the selected weight profile is
                    # lossy.  Opt-in disk paging retains every KV bit while
                    # bounding resident KV.  It is deliberately incompatible
                    # with the in-memory hot/PIC paths: paged attention must
                    # reload the complete history and cannot safely retain or
                    # relocate those cache objects between requests yet.
                    try:
                        fast_kv_max_mb = int(os.environ.get(
                            "VMODEL_FAST_KV_MAX_MB", "0"))
                    except ValueError as error:
                        raise ValueError(
                            "VMODEL_FAST_KV_MAX_MB must be an integer") from error
                    if fast_kv_max_mb < 0:
                        raise ValueError(
                            "VMODEL_FAST_KV_MAX_MB must be non-negative")
                    if (fast_kv_max_mb
                            and mtype in ("qwen2", "qwen3")
                            and not cfg_probe.vision_config
                            and not cfg_probe.num_experts):
                        try:
                            paged_chunk = int(os.environ.get(
                                "VMODEL_FAST_KV_PREFILL_CHUNK_SIZE", "512"))
                            paged_mlx_cache_mb = int(os.environ.get(
                                "VMODEL_FAST_KV_MLX_CACHE_MB", "64"))
                        except ValueError as error:
                            raise ValueError(
                                "VMODEL_FAST_KV prefill/cache limits must be "
                                "integers") from error
                        if not 1 <= paged_chunk <= 4096:
                            raise ValueError(
                                "VMODEL_FAST_KV_PREFILL_CHUNK_SIZE must be in "
                                "[1, 4096]")
                        if not 1 <= paged_mlx_cache_mb <= 1024:
                            raise ValueError(
                                "VMODEL_FAST_KV_MLX_CACHE_MB must be in [1, 1024]")
                        compress_value = os.environ.get(
                            "VMODEL_FAST_KV_SPILL_COMPRESS", "0")
                        if compress_value not in ("0", "1"):
                            raise ValueError(
                                "VMODEL_FAST_KV_SPILL_COMPRESS must be 0 or 1")
                        rc.max_kv_mb = fast_kv_max_mb
                        rc.release_paged_kv_after_generate = True
                        rc.mlx_cache_limit_mb = paged_mlx_cache_mb
                        # Paged retention alone did not bound the 4,096-token
                        # activation/reload transient: real 28K Qwen3 prefill
                        # still fell below 4 GB available. Progressive 512-token
                        # sweeps keep that separately bounded and expose much
                        # finer-grained progress events.
                        rc.prefill_chunk_size = paged_chunk
                        rc.kv_spill_dir = os.environ.get(
                            "VMODEL_FAST_KV_SPILL_DIR",
                            str(ROOT / ".kv_spill"),
                        )
                        rc.kv_spill_compress = compress_value == "1"
                        rc.hot_prompt_kv = False
                        rc.hot_prompt_kv_persist_dir = ""
                        rc.tool_pic = False
                        rc.tool_pic_shared_pages = False
                    if mode == "fast-long":
                        rc.qwen_yarn_factor = yarn_factor
                if (mode == "lossless"
                        and not getattr(cfg_probe, "num_experts", 0)
                        and not getattr(cfg_probe, "vision_config", None)
                        and _dense_lossless_resident_bytes(cfg_probe)
                        <= int(rc.max_weight_cache_mb * 1_000_000 * 0.85)):
                    # Exact BF16 Qwen2.5-1.5B: 3.09 GB resident, 58.1 ->
                    # 106.5 tok/s (+83.4%) across 1,404 byte-identical tokens.
                    # This changes scheduling only: weights and block math stay
                    # untouched, and the 15% admission margin keeps every layer
                    # resident before the lazy/pipelined branch can activate.
                    rc.resident_fast_decode = True
                    rc.resident_fast_prefill_limit = 2048
                    rc.embed_rows = False
                    # Exact stepped KV regresses short resident requests, but
                    # clears the project threshold at 2K: Qwen-1.5B gained
                    # 3.5% decode (~2.1% total) and SmolLM gained 4.7% decode
                    # (~3.6% total), with identical IDs. Streamed Qwen-7B was
                    # neutral, so this stays inside the resident admission arm.
                    rc.stepped_kv_threshold = 2048
                    # Keep one exact in-memory endpoint ahead of the durable
                    # F37 disk snapshots. Repeated 1.9K/4K prompts were 6.3%/
                    # 2.1% faster from memory with identical IDs; disk remains
                    # the restart/overflow fallback. The minimum prevents tiny
                    # helper calls from evicting an expensive conversation.
                    rc.hot_prompt_kv = True
                    rc.hot_prompt_kv_chunk_size = rc.prefill_chunk_size
                    try:
                        rc.hot_prompt_kv_slots = int(os.environ.get(
                            "VMODEL_HOT_PROMPT_KV_SLOTS", "1"))
                        rc.hot_prompt_kv_min_tokens = int(os.environ.get(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS", "2048"))
                    except ValueError as e:
                        raise ValueError(
                            "VMODEL hot prompt KV settings must be integers") from e
                    if rc.hot_prompt_kv_slots <= 0:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_SLOTS must be positive")
                    if rc.hot_prompt_kv_min_tokens < 0:
                        raise ValueError(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS must be >= 0")
                if (mode == "lossless"
                        and getattr(cfg_probe, "vision_config", None)
                        and not getattr(cfg_probe, "num_experts", 0)
                        and _checkpoint_payload_bytes(model_dir)
                        <= int(rc.max_weight_cache_mb * 1_000_000 * 0.85)):
                    # Qwen3-VL owns a custom M-RoPE decode loop, but once its
                    # complete released checkpoint fits it can use the same
                    # lazy predicted-token pipeline. Five-pair exact BF16 ABBA:
                    # 56/56 IDs each run, 1.202s -> 0.614s median decode
                    # (1.96x) with all 55 continuation steps exercised.
                    rc.resident_fast_decode = True
                    rc.resident_fast_prefill_limit = 2048
                    rc.embed_rows = False
            draft_dir = None
            speculative_k = 6
            speculative_prompt_limit = 2048
            speculative_draft_cache_mb = 1200
            speculative_target_prefetch_workers = 2
            speculative_target_prefetch_depth = 4
            speculative_prompt_cache_min_tokens = 2048
            dspark_dir = None
            dspark_cap = 4
            dspark_prompt_limit = 2048
            dspark_confidence_threshold = 0.0
            dspark_prompt_cache_min_tokens = 2048
            dspark_target_prefetch_workers = 2
            dspark_target_prefetch_depth = 4
            dspark_target_cache_mb = max(
                6000,
                math.ceil(_dense_lossless_resident_bytes(cfg_probe) * 1.07 / 1_000_000),
            )
            qwen2_adaptive_candidate = False
            qwen2_streamed_rc = None
            qwen2_resident_cache_mb = 0
            olmoe_adaptive_candidate = False
            olmoe_streamed_rc = None
            olmoe_resident_required = 0
            qwen3_adaptive_candidate = False

            if mode == "lossless" and mtype == "olmoe":
                payload = _checkpoint_payload_bytes(model_dir)
                if payload > 0:
                    # Keep released routing, expert matmuls, and summation order
                    # untouched; only make the cache large enough to avoid a
                    # cyclic reread of exact BF16 pages. A local 64-token gate
                    # matched every ID and reduced 9.30s -> 4.81s (1.93x).
                    olmoe_adaptive_candidate = True
                    olmoe_streamed_rc = replace(rc)
                    olmoe_resident_required = payload
                    rc.max_weight_cache_mb = max(
                        rc.max_weight_cache_mb,
                        math.ceil(payload * 1.07 / 1_000_000),
                    )
            # Keep this lossless-only. The streamed BF16 target amortizes one
            # layer read across a verify window; an already-resident Qwen-7B
            # MXFP4 target has no such bill. Real A/B: ordinary fast mode was
            # 86.6 tok/s versus 22.6 with speculation (0.395x), exact IDs.
            if (mode == "lossless"
                    and mtype == "qwen2"
                    and not getattr(cfg_probe, "num_experts", 0)
                    and not getattr(cfg_probe, "vision_config", None)
                    and not rc.resident_fast_decode):
                draft_dir = _speculative_draft_for(model_dir, cfg_probe)
                try:
                    speculative_k = int(os.environ.get(
                        "VMODEL_SPECULATIVE_K", "6"))
                    speculative_prompt_limit = int(os.environ.get(
                        "VMODEL_SPECULATIVE_MAX_PROMPT_TOKENS", "2048"))
                    speculative_draft_cache_mb = int(os.environ.get(
                        "VMODEL_SPECULATIVE_DRAFT_CACHE_MB", "1200"))
                    speculative_target_prefetch_workers = int(os.environ.get(
                        "VMODEL_SPECULATIVE_TARGET_PREFETCH_WORKERS", "2"))
                    speculative_target_prefetch_depth = int(os.environ.get(
                        "VMODEL_SPECULATIVE_TARGET_PREFETCH_DEPTH", "4"))
                    speculative_prompt_cache_min_tokens = int(os.environ.get(
                        "VMODEL_SPECULATIVE_PROMPT_CACHE_MIN_TOKENS", "2048"))
                except ValueError as e:
                    raise RequestValidationError(
                        "VMODEL speculative limits must be integers") from e
                if (speculative_k <= 0 or speculative_prompt_limit <= 0
                        or speculative_draft_cache_mb <= 0
                        or speculative_target_prefetch_workers <= 0
                        or speculative_target_prefetch_depth <= 0):
                    raise RequestValidationError(
                        "VMODEL speculative limits must be positive")
                if speculative_prompt_cache_min_tokens < 0:
                    raise RequestValidationError(
                        "VMODEL_SPECULATIVE_PROMPT_CACHE_MIN_TOKENS must be >= 0")
                if draft_dir is not None:
                    # Raw local Qwen-7B target, exact 1/2/2/1 A/B: two
                    # prefetch workers overlapped independent layer reads and
                    # improved speculative decode 8.54% (4.78 -> 5.19 tok/s),
                    # with identical IDs. Three workers regressed. With two
                    # workers, depth four added another 4.87% over depth two;
                    # depth six was neutral and depth eight regressed.
                    rc.prefetch_workers = speculative_target_prefetch_workers
                    rc.prefetch_depth = speculative_target_prefetch_depth

                # A fixed 6 GB cache is the right fallback on the constrained
                # target host, but it needlessly streams a complete exact
                # target on roomier machines. Probe a model-sized allowance
                # against the same live governor used for Qwen3 below. Keep a
                # pristine copy of the established streamed/speculative
                # profile so a rejected admission is behaviorally unchanged.
                override = os.environ.get(
                    "VMODEL_SPECULATIVE_DRAFT", "auto").strip().lower()
                force_speculation = override not in (
                    "", "auto", "0", "off", "false", "none", "disabled")
                if not force_speculation:
                    qwen2_adaptive_candidate = True
                    qwen2_streamed_rc = replace(rc)
                    qwen2_resident_cache_mb = max(
                        6000,
                        math.ceil(
                            _dense_lossless_resident_bytes(cfg_probe)
                            * 1.07 / 1_000_000),
                    )
                    rc.max_weight_cache_mb = qwen2_resident_cache_mb
                    # The admitted target must populate every layer once.
                    # Qwen-7B's validated two-worker/depth-four schedule also
                    # improves this cold residency fill; after that no reads
                    # remain on the decode path.
                    rc.prefetch_workers = speculative_target_prefetch_workers
                    rc.prefetch_depth = speculative_target_prefetch_depth

            # Qwen3's released DSpark drafter is a different architecture from
            # the autoregressive Qwen2 draft above.  The local Qwen3-4B/block-7
            # gate matched 128/128 target IDs and reduced aggregate wall time
            # 29.70s -> 12.21s at a 9.24 GB observed peak. Cap four beat caps
            # 2/3/5/6/7 by
            # more than the retention threshold on the streamed target.
            if (mode == "lossless"
                    and mtype == "qwen3"
                    and not getattr(cfg_probe, "num_experts", 0)
                    and not getattr(cfg_probe, "vision_config", None)
                    and not rc.resident_fast_decode):
                qwen3_adaptive_candidate = True
                dspark_dir = _dspark_draft_for(model_dir, cfg_probe)
                try:
                    dspark_cap = int(os.environ.get(
                        "VMODEL_DSPARK_MAX_DRAFT_TOKENS", "4"))
                    dspark_prompt_limit = int(os.environ.get(
                        "VMODEL_DSPARK_MAX_PROMPT_TOKENS", "2048"))
                    dspark_confidence_threshold = float(os.environ.get(
                        "VMODEL_DSPARK_CONFIDENCE_THRESHOLD", "0"))
                    dspark_prompt_cache_min_tokens = int(os.environ.get(
                        "VMODEL_DSPARK_PROMPT_CACHE_MIN_TOKENS", "2048"))
                    dspark_target_prefetch_workers = int(os.environ.get(
                        "VMODEL_DSPARK_TARGET_PREFETCH_WORKERS", "2"))
                    dspark_target_prefetch_depth = int(os.environ.get(
                        "VMODEL_DSPARK_TARGET_PREFETCH_DEPTH", "4"))
                    cache_override = os.environ.get(
                        "VMODEL_DSPARK_TARGET_CACHE_MB", "").strip()
                    if cache_override:
                        dspark_target_cache_mb = int(cache_override)
                except ValueError as e:
                    raise RequestValidationError(
                        "VMODEL DSpark limits must be numeric") from e
                if not 1 <= dspark_cap <= 7 or dspark_prompt_limit <= 0:
                    raise RequestValidationError(
                        "VMODEL DSpark cap must be in [1, 7] and prompt limit positive")
                if (not math.isfinite(dspark_confidence_threshold)
                        or not 0.0 <= dspark_confidence_threshold <= 1.0):
                    raise RequestValidationError(
                        "VMODEL_DSPARK_CONFIDENCE_THRESHOLD must be finite and in [0, 1]")
                if dspark_prompt_cache_min_tokens < 0:
                    raise RequestValidationError(
                        "VMODEL_DSPARK_PROMPT_CACHE_MIN_TOKENS must be >= 0")
                if (dspark_target_prefetch_workers <= 0
                        or dspark_target_prefetch_depth <= 0
                        or dspark_target_cache_mb <= 0):
                    raise RequestValidationError(
                        "VMODEL DSpark target cache/prefetch settings must be positive")
                # Qwen3-4B/block-7 exact A/B: two workers + depth four reduced
                # a representative streamed DSpark request 4.70s -> 3.25s.
                # Three workers regressed to 3.35s and raised peak memory. The
                # target-only resident alternative benefits from the same cold
                # load schedule, including when DSpark discovery is disabled.
                rc.prefetch_workers = dspark_target_prefetch_workers
                rc.prefetch_depth = dspark_target_prefetch_depth
                # Configured from the model's estimated exact footprint, then
                # fitted after construction against sampled live headroom.
                rc.max_weight_cache_mb = dspark_target_cache_mb
                # Keep exact long prompt endpoints in memory ahead of F37's
                # durable snapshot. A 2,048-token repeat measured ~0.29ms
                # end-to-end for a one-token request versus ~18.95ms from disk,
                # with identical IDs. Tiny helper prompts must not evict it.
                rc.hot_prompt_kv = True
                rc.hot_prompt_kv_chunk_size = rc.prefill_chunk_size
                try:
                    rc.hot_prompt_kv_slots = int(os.environ.get(
                        "VMODEL_HOT_PROMPT_KV_SLOTS", "1"))
                    rc.hot_prompt_kv_min_tokens = int(os.environ.get(
                        "VMODEL_HOT_PROMPT_KV_MIN_TOKENS", "2048"))
                except ValueError as e:
                    raise RequestValidationError(
                        "VMODEL hot prompt KV settings must be integers") from e
                if rc.hot_prompt_kv_slots <= 0:
                    raise RequestValidationError(
                        "VMODEL_HOT_PROMPT_KV_SLOTS must be positive")
                if rc.hot_prompt_kv_min_tokens < 0:
                    raise RequestValidationError(
                        "VMODEL_HOT_PROMPT_KV_MIN_TOKENS must be >= 0")

            target_engine = StreamingEngine(model_dir, rc)
            self._engine = target_engine
            if qwen2_adaptive_candidate:
                fitted_cache = target_engine.cache.max_bytes
                if target_engine.governor is not None:
                    fitted_cache = (
                        target_engine.governor.fit_cache_to_live_headroom())
                resident_required = _dense_lossless_resident_bytes(cfg_probe)

                # Qwen2.5-7B is untied and the streamed profile row-pages its
                # embedding. Resident pipelining needs the next predicted token
                # to feed a device-side gather, so admission requires a clean
                # reconstruction with the exact embedding pinned instead.
                target_engine.close()
                self._engine = None
                target_engine = None
                mx.clear_cache()

                if fitted_cache >= resident_required:
                    try:
                        hot_slots = int(os.environ.get(
                            "VMODEL_HOT_PROMPT_KV_SLOTS", "1"))
                        hot_min_tokens = int(os.environ.get(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS", "2048"))
                    except ValueError as e:
                        raise RequestValidationError(
                            "VMODEL hot prompt KV settings must be integers") from e
                    if hot_slots <= 0:
                        raise RequestValidationError(
                            "VMODEL_HOT_PROMPT_KV_SLOTS must be positive")
                    if hot_min_tokens < 0:
                        raise RequestValidationError(
                            "VMODEL_HOT_PROMPT_KV_MIN_TOKENS must be >= 0")

                    rc = replace(
                        qwen2_streamed_rc,
                        max_weight_cache_mb=qwen2_resident_cache_mb,
                        embed_rows=False,
                        resident_fast_decode=True,
                        resident_fast_prefill_limit=2048,
                        stepped_kv_threshold=2048,
                        hot_prompt_kv=True,
                        hot_prompt_kv_chunk_size=(
                            qwen2_streamed_rc.prefill_chunk_size),
                        hot_prompt_kv_slots=hot_slots,
                        hot_prompt_kv_min_tokens=hot_min_tokens,
                    )
                    target_engine = StreamingEngine(model_dir, rc)
                    self._engine = target_engine
                    final_fitted = target_engine.cache.max_bytes
                    if target_engine.governor is not None:
                        final_fitted = (
                            target_engine.governor.fit_cache_to_live_headroom())
                    if final_fitted >= resident_required:
                        draft_dir = None
                        print(
                            f"[server] exact resident target admitted: "
                            f"target={model_dir.name} "
                            f"cache={final_fitted / 1e9:.2f}GB "
                            f"required={resident_required / 1e9:.2f}GB; "
                            f"autoregressive draft not needed",
                            flush=True,
                        )
                    else:
                        # Availability can change between the lightweight probe
                        # and reconstruction. Fail back to the known 6 GB target
                        # plus verified draft instead of leaving a non-resident
                        # target configured for the resident-only fast loop.
                        target_engine.close()
                        self._engine = None
                        target_engine = None
                        mx.clear_cache()
                        rc = qwen2_streamed_rc
                        target_engine = StreamingEngine(model_dir, rc)
                        self._engine = target_engine
                        print(
                            f"[server] exact resident target admission changed; "
                            f"using streamed target cache="
                            f"{target_engine.cache.max_bytes / 1e9:.2f}GB",
                            flush=True,
                        )
                else:
                    rc = qwen2_streamed_rc
                    target_engine = StreamingEngine(model_dir, rc)
                    self._engine = target_engine
                    print(
                        f"[server] exact resident target not admitted: "
                        f"target={model_dir.name} "
                        f"cache={fitted_cache / 1e9:.2f}GB "
                        f"required={resident_required / 1e9:.2f}GB; "
                        f"using streamed target",
                        flush=True,
                    )
            if olmoe_adaptive_candidate:
                fitted_cache = target_engine.cache.max_bytes
                if target_engine.governor is not None:
                    fitted_cache = (
                        target_engine.governor.fit_cache_to_live_headroom())
                if fitted_cache >= olmoe_resident_required:
                    print(
                        f"[server] exact OLMoE cache admitted: "
                        f"target={model_dir.name} "
                        f"cache={fitted_cache / 1e9:.2f}GB "
                        f"required={olmoe_resident_required / 1e9:.2f}GB",
                        flush=True,
                    )
                else:
                    # Preserve the established streamed profile when current
                    # unified-memory headroom cannot hold every exact page.
                    target_engine.close()
                    self._engine = None
                    target_engine = None
                    mx.clear_cache()
                    rc = olmoe_streamed_rc
                    target_engine = StreamingEngine(model_dir, rc)
                    self._engine = target_engine
                    print(
                        f"[server] exact OLMoE cache not admitted: "
                        f"target={model_dir.name} "
                        f"cache={fitted_cache / 1e9:.2f}GB "
                        f"required={olmoe_resident_required / 1e9:.2f}GB; "
                        f"using streamed target",
                        flush=True,
                    )
            if qwen3_adaptive_candidate:
                fitted_cache = target_engine.cache.max_bytes
                if target_engine.governor is not None:
                    fitted_cache = (
                        target_engine.governor.fit_cache_to_live_headroom())
                resident_required = _dense_lossless_resident_bytes(cfg_probe)
                override = os.environ.get(
                    "VMODEL_DSPARK_DRAFT", "auto").strip().lower()
                force_dspark = override not in (
                    "", "auto", "0", "off", "false", "none", "disabled")
                if fitted_cache >= resident_required and not force_dspark:
                    # Full exact target residency is both faster and smaller
                    # than Qwen3-4B's unusually large 2.6GB BF16 drafter on a
                    # roomy host: 128/128 IDs, 3.28s aggregate, 8.06GB peak.
                    # Keep DSpark as the streamed/explicit fallback only.
                    dspark_dir = None
                    rc.resident_fast_decode = True
                    rc.resident_fast_prefill_limit = 2048
                    rc.stepped_kv_threshold = 2048
                    print(
                        f"[server] exact resident target admitted: "
                        f"target={model_dir.name} cache={fitted_cache / 1e9:.2f}GB "
                        f"required={resident_required / 1e9:.2f}GB; "
                        f"DSpark not needed",
                        flush=True,
                    )
            if dspark_dir is not None:
                from .dspark import DSparkSpeculativeEngine

                try:
                    self._engine = DSparkSpeculativeEngine(
                        target_engine, dspark_dir,
                        max_draft_tokens=dspark_cap,
                        max_prompt_tokens=dspark_prompt_limit,
                        confidence_threshold=dspark_confidence_threshold,
                        prompt_cache_min_tokens=(
                            dspark_prompt_cache_min_tokens),
                    )
                    print(
                        f"[server] exact DSpark speculation: "
                        f"target={model_dir.name} draft={dspark_dir.name} "
                        f"cap={dspark_cap} prompt_limit={dspark_prompt_limit} "
                        f"confidence={dspark_confidence_threshold:g} "
                        f"prefetch={dspark_target_prefetch_workers}/"
                        f"{dspark_target_prefetch_depth} "
                        f"target_cache={rc.max_weight_cache_mb}MB-configured",
                        flush=True,
                    )
                except Exception as error:
                    explicit = os.environ.get(
                        "VMODEL_DSPARK_DRAFT", "auto").strip().lower() not in (
                            "", "auto")
                    if explicit:
                        target_engine.close()
                        self._engine = None
                        raise RequestValidationError(
                            f"could not initialize DSpark draft {dspark_dir}: "
                            f"{error}") from error
                    self._engine = target_engine
                    mx.clear_cache()
                    print(
                        f"[server] DSpark draft skipped ({dspark_dir}): {error}",
                        flush=True,
                    )
            elif draft_dir is not None:
                from .speculative import SpeculativeEngine

                # This is a per-engine cache ceiling, not a fixed total-memory
                # admission cap. Both engines retain live pressure governors;
                # the draft needs only enough budget for its ~0.8 GB MXFP4
                # checkpoint while the shared unified-memory ceiling continues
                # to follow sampled system availability.
                draft_rc = RuntimeConfig(
                    max_weight_cache_mb=speculative_draft_cache_mb,
                    pin_embeddings=True,
                    pin_lm_head=True,
                    prefetch_depth=0,
                    resident_fast_decode=True,
                    resident_fast_prefill_limit=speculative_prompt_limit,
                    stepped_kv_threshold=512,
                    fused_swiglu=True,
                    governor=rc.governor,
                )
                draft_engine = None
                try:
                    draft_engine = StreamingEngine(draft_dir, draft_rc)
                    self._engine = SpeculativeEngine(
                        target_engine, draft_engine,
                        k=speculative_k,
                        max_prompt_tokens=speculative_prompt_limit,
                        prompt_cache_min_tokens=(
                            speculative_prompt_cache_min_tokens),
                    )
                    print(
                        f"[server] exact speculation: target={model_dir.name} "
                        f"draft={draft_dir.name} k={speculative_k} "
                        f"prompt_limit={speculative_prompt_limit}",
                        flush=True,
                    )
                except Exception as error:
                    if draft_engine is not None:
                        draft_engine.close()
                    explicit = os.environ.get(
                        "VMODEL_SPECULATIVE_DRAFT", "auto").strip().lower() not in (
                            "", "auto")
                    if explicit:
                        target_engine.close()
                        self._engine = None
                        raise RequestValidationError(
                            f"could not initialize speculative draft {draft_dir}: {error}") from error
                    print(
                        f"[server] speculative draft skipped ({draft_dir}): {error}",
                        flush=True,
                    )
            self._key = key
            return self._engine

    def invalidate(self, model_dir: Path):
        """Force-close the resident engine if it's currently serving
        model_dir (any mode), so the NEXT request reopens fresh and picks
        up an on-disk change — specifically, PackManager just converted raw
        safetensors to vpack2 in place; WeightStore auto-detects vpack2 on
        open, but an already-open engine object never re-checks on its own.
        Acquires INFER_LOCK first: closing an engine out from under an
        in-flight generate() call would corrupt state, so this simply waits
        for the current request (if any) to finish first."""
        import mlx.core as mx

        with INFER_LOCK, self._lock:
            if self._key is not None and self._key[0] == str(model_dir):
                print(f"[server] invalidating resident engine for {model_dir} (freshly packed)", flush=True)
                self._engine.close()
                self._engine = None
                self._key = None
                mx.clear_cache()


MANAGER = EngineManager()
# 2026-07-12 audit: ThreadingHTTPServer would otherwise run CONCURRENT Metal
# inference (competing for the same sampled unified-memory headroom) and could swap/close an engine out
# from under an in-flight request. One inference at a time, engine held.
INFER_LOCK = threading.Lock()


class PackManager:
    """Auto-packs a raw-safetensors auto-downloaded model into the
    zstd-compressed, heat-ordered vpack2 format (F06/F20) once it has been
    requested more than once — a one-off typo/try shouldn't pay the packing
    cost, but a model that's clearly being used repeatedly should get the
    same disk-read optimizations as a deliberately-provisioned model,
    without any extra client action (2026-07-13, user request: fully
    "auto", triggered by a second request coming through — regardless of
    whether it's the bare or `lossy-` form, since _resolve() is always
    called with the mode-prefix already stripped — and purely informational
    while it runs: the raw-safetensors path already serves fine, so packing
    never blocks or degrades a response, only adds status fields to it).

    Only engages for models that came through DOWNLOADS (i.e. genuinely
    auto-downloaded) — deliberately-provisioned models (GLM-5.2, gpt-oss,
    ...) are packed through the existing manual ops workflow and must not
    be touched here."""

    def __init__(self):
        self._lock = threading.Lock()
        self._request_counts: dict[str, int] = {}
        self._status: dict[str, dict] = {}

    def status(self, model_id: str) -> dict | None:
        with self._lock:
            st = self._status.get(model_id)
            return dict(st) if st is not None else None

    def status_fields(self, model_id: str) -> dict:
        """Small dict to merge into a response's usage/metadata — empty
        once packing is done, so a client sees this only while it's live."""
        st = self.status(model_id)
        if st is None or st["state"] == "packed":
            return {}
        fields = {"vmodel_pack_status": st["state"]}
        if st["state"] == "packing" and st["tensors_total"]:
            fields["vmodel_pack_progress_pct"] = round(100 * st["tensors_done"] / st["tensors_total"], 1)
            elapsed = time.time() - st["started_at"]
            if st["tensors_done"] > 0:
                eta = elapsed / st["tensors_done"] * (st["tensors_total"] - st["tensors_done"])
                fields["vmodel_pack_eta_seconds"] = round(eta, 1)
        if st["error"]:
            fields["vmodel_pack_error"] = st["error"]
        return fields

    def pending_entries(self) -> list[dict]:
        """Non-packed models, for GET /v1/models — same style as
        DOWNLOADS.pending_entries()."""
        with self._lock:
            items = list(self._status.items())
        out = []
        for mid, st in items:
            if st["state"] == "packed":
                continue
            entry = {"id": mid, "object": "model", "owned_by": "vmodel"}
            entry.update(self.status_fields(mid))
            out.append(entry)
        return out

    def note_request(self, model_id: str, target: Path):
        """Call on every _resolve() of an auto-downloaded, ready model.
        Starts packing automatically on the SECOND distinct request."""
        if not _ENABLE_UNSAFE_AUTOPACK:
            return
        with self._lock:
            if model_id in self._status:
                return  # already packing/packed/failed -- nothing to do
            count = self._request_counts.get(model_id, 0) + 1
            self._request_counts[model_id] = count
            if count < 2:
                return
        if (target / "weights.vpack2.index.json").exists() or (target / "weights.vpack").exists():
            return  # already packed, or mid-pack from a previous process -- don't double-pack
        self._start(model_id, target)

    def _start(self, model_id: str, target: Path):
        with self._lock:
            if model_id in self._status:
                return
            self._status[model_id] = {"state": "packing", "error": None, "started_at": time.time(),
                                      "tensors_done": 0, "tensors_total": 0}

        def _progress(done: int, total: int):
            with self._lock:
                st = self._status.get(model_id)
                if st is not None:
                    st["tensors_done"], st["tensors_total"] = done, total

        def _run():
            try:
                import shutil

                free = shutil.disk_usage(ROOT).free
                if free < _MIN_FREE_BYTES_FOR_AUTO_DOWNLOAD:
                    raise RuntimeError(
                        f"only {free / 1e9:.1f} GB free, need at least "
                        f"{_MIN_FREE_BYTES_FOR_AUTO_DOWNLOAD / 1e9:.0f} GB free to pack")
                from formats.packed import pack_model
                from formats.packed2 import build_from_vpack

                print(f"[server] auto-packing {model_id} -> {target}", flush=True)
                pack_model(target, delete_shards=True, progress=_progress)
                build_from_vpack(target, consume_source=True, progress=_progress)
                with self._lock:
                    self._status[model_id]["state"] = "packed"
                print(f"[server] {model_id} auto-pack complete -> {target}", flush=True)
                MANAGER.invalidate(target)
            except Exception as e:
                with self._lock:
                    self._status[model_id]["state"] = "failed"
                    self._status[model_id]["error"] = f"{type(e).__name__}: {e}"
                print(f"[server] auto-pack FAILED for {model_id}: {type(e).__name__}: {e}", flush=True)

        threading.Thread(target=_run, daemon=True, name=f"pack-{target.name}").start()


PACKS = PackManager()


@lru_cache(maxsize=32)
def _compiled_template(template_text: str, compact_json: bool):
    """Compile each immutable checkpoint template/profile once per process."""
    # 2026-07-19: Kimi K2.5's real chat_template.jinja uses {% break %}
    # (a for-loop early-exit), which needs jinja2's loopcontrols extension
    # enabled -- without it, Environment/Template can't even PARSE the
    # template (not just fail to render it). Harmless for every other
    # checkpoint's template, which simply doesn't use the tag.
    # Real chat templates (Qwen's included) call a `raise_exception(msg)`
    # global to reject malformed conversations (e.g. a non-leading system
    # message) -- this is the same global name HF's own
    # `apply_chat_template` injects. Leaving it undefined doesn't skip
    # those checks; Jinja still evaluates the call and raises its own
    # opaque `UndefinedError: 'raise_exception' is undefined`, which hid
    # the template's actual, informative message behind a 500 traceback
    # (2026-07-20, live-confirmed). Routing it through
    # RequestValidationError surfaces the template's real message as a
    # clean 400 instead.
    def _raise_exception(msg):
        raise RequestValidationError(str(msg))

    if not compact_json:
        from jinja2 import Template

        tmpl = Template(template_text, extensions=["jinja2.ext.loopcontrols"])
        tmpl.globals["raise_exception"] = _raise_exception
        return tmpl

    from jinja2 import Environment
    from jinja2.utils import htmlsafe_json_dumps

    env = Environment(autoescape=False, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = _raise_exception
    # Preserve Jinja's tojson escaping contract while changing only key order
    # and insignificant JSON whitespace. Raw json.dumps would allow strings
    # such as ``</tools>`` to escape a template's tool delimiter.
    # Match Jinja's public filter signature. Compact mode intentionally ignores
    # a template's indent request so equivalent schemas remain canonical.
    env.filters["tojson"] = lambda value, indent=None: htmlsafe_json_dumps(
        value, dumps=json.dumps, ensure_ascii=False,
        separators=(",", ":"), sort_keys=True)
    return env.from_string(template_text)


def _render_template(template_text: str, *, compact_json: bool = False, **context) -> str:
    """Render a checkpoint template, optionally with canonical compact JSON.

    Fast-mode tool schemas are large enough that Jinja's default whitespace in
    every JSON object costs thousands of model tokens. This only changes the
    side-quest representation; lossless mode retains the checkpoint template's
    ordinary filter byte-for-byte.
    """
    return _compiled_template(template_text, compact_json).render(**context)


def _messages_for_native_template(messages: list[dict]) -> list[dict]:
    """Return a prompt-only copy with tool arguments restored to JSON values.

    OpenAI-compatible history stores function arguments as a JSON *string*.
    Native Qwen/Harmony templates call ``arguments | tojson`` and expect the
    parsed object; passing the wire string double-encodes it and changes the
    learned tool-call transcript. Generic fallback rendering intentionally keeps
    its existing raw-string path, so normalization happens only at template use.
    """
    normalized = []
    for message in messages:
        copied = dict(message)
        calls = message.get("tool_calls") or []
        if calls:
            copied_calls = []
            for call in calls:
                copied_call = dict(call)
                wrapped = isinstance(call.get("function"), dict)
                source = call["function"] if wrapped else call
                copied_function = dict(source)
                arguments = copied_function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        copied_function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        # Preserve malformed historical text rather than
                        # silently rewriting or dropping it.
                        pass
                if wrapped:
                    copied_call["function"] = copied_function
                else:
                    copied_call = copied_function
                copied_calls.append(copied_call)
            copied["tool_calls"] = copied_calls
        normalized.append(copied)
    return normalized


def _messages_with_canonical_hermes_tool_history(
        messages: list[dict]) -> list[dict]:
    """Serialize structured history in the exact fast tool-call protocol.

    Qwen3.5/3.6's released template renders historical calls as nested XML,
    while vOOM's schema constraint and wire parser use Hermes JSON.  Mixing
    those protocols means a valid call can never be an exact prefix of the
    following tool-result turn, which is especially expensive for the hybrid
    recurrent cache (it cannot rewind to a merely similar prefix).  Fast mode
    deliberately uses canonical Hermes for both generation and history;
    lossless mode continues to render the released template unchanged.
    """
    normalized = []
    for message in messages:
        copied = dict(message)
        calls = copied.pop("tool_calls", None) or []
        if calls:
            blocks = []
            for call in calls:
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # canonicalize_tool_history validates this before prompt
                        # rendering; retain a fail-closed fallback for direct
                        # private-helper callers.
                        arguments = arguments
                payload = {"name": function.get("name", ""),
                           "arguments": arguments}
                blocks.append(
                    "<tool_call>"
                    + json.dumps(payload, ensure_ascii=False,
                                 separators=(", ", ": "), allow_nan=False)
                    + "</tool_call>")
            content = copied.get("content")
            copied["content"] = (content if isinstance(content, str) else "") + "".join(blocks)
        normalized.append(copied)
    return normalized


def _messages_with_effort_instruction(
        messages: list[dict], reasoning: str) -> list[dict]:
    instructions = {
        "none": "Answer directly. Do not produce a separate reasoning section.",
        "minimal": "Answer directly after only a minimal internal check.",
        "low": "Use a brief internal check, then answer directly.",
        "medium": "Reason carefully and verify the key steps before answering.",
        "high": "Reason thoroughly, check alternatives, and verify the result before answering.",
        "xhigh": "Use exhaustive reasoning, test assumptions, and verify the result before answering.",
    }
    directive = instructions[reasoning]
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system" and isinstance(
            copied[0].get("content"), str):
        copied[0]["content"] = directive + "\n\n" + copied[0]["content"]
    else:
        copied.insert(0, {"role": "system", "content": directive})
    return copied


@lru_cache(maxsize=32)
def _template_consumes_tools(template_text: str) -> bool:
    """Whether a native Jinja template actually reads the ``tools`` input."""
    from jinja2 import Environment, meta

    return "tools" in meta.find_undeclared_variables(
        Environment(extensions=["jinja2.ext.loopcontrols"]).parse(template_text))


def _tools_system_preamble(tools: list[dict], *, compact_json: bool) -> str:
    """Canonical tool block for native templates that ignore ``tools``.

    Some instruct checkpoints ship a chat template but no tool branch. Passing
    a Jinja context variable to those templates silently drops the catalog.
    Keep using the learned role delimiters while injecting an explicit system
    message whose JSON-line spans can also participate in tool PIC.
    """
    if compact_json:
        from jinja2.utils import htmlsafe_json_dumps

        serialized = [str(htmlsafe_json_dumps(
            tool, dumps=json.dumps, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True)) for tool in tools]
    else:
        serialized = [json.dumps(tool, ensure_ascii=False) for tool in tools]
    return (
        "You have access to the following tools. To call one, respond with\n"
        "<tool_call>\n{\"name\": \"<function-name>\", "
        "\"arguments\": {...}}\n</tool_call>\n"
        "<tools>\n" + "\n".join(serialized) + "\n</tools>"
    )


def _prepend_system_content(messages: list[dict], content: str) -> list[dict]:
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system" and isinstance(
            copied[0].get("content"), str):
        copied[0]["content"] = content + "\n\n" + copied[0]["content"]
    else:
        copied.insert(0, {"role": "system", "content": content})
    return copied


def _chat_prompt(engine, model_dir: Path, messages: list[dict], reasoning: str,
                 tools: list[dict] | None = None, *, compact_json: bool = False,
                 enable_thinking: bool | None = None,
                 reasoning_requested: bool = False,
                 canonical_hermes_tools: bool = False) -> str:
    # 2026-07-14: a standalone chat_template.json file is one HF convention,
    # but the more common one is a `chat_template` field embedded directly in
    # tokenizer_config.json -- checking ONLY for the standalone file meant
    # any model shipped the common way (e.g. Qwen2.5) silently fell through
    # to the generic role-tagged fallback below, even though a real chat
    # template (with real turn-boundary tokens like <|im_end|> the model
    # actually learned to emit as a stop signal) was available. Found live:
    # this silently produced a chat template with NO learned stop boundary,
    # so a real instruct model just free-ran past where it should have
    # stopped, repeating "user: ...\nassistant: ..." until max_tokens cut it
    # off -- and that garbage then got fed back as history on the next turn,
    # compounding token usage and repetition further.
    template_text = None
    if (model_dir / "chat_template.jinja").exists():
        # GLM-5.2 and gpt-oss both ship their learned protocol this way. The
        # previous gpt-oss-only special case silently sent GLM a generic
        # ``user: ...`` transcript and dropped reasoning controls entirely.
        template_text = (model_dir / "chat_template.jinja").read_text()
    elif (model_dir / "chat_template.json").exists():
        template_text = json.loads((model_dir / "chat_template.json").read_text())["chat_template"]
    else:
        tok_cfg_path = model_dir / "tokenizer_config.json"
        if tok_cfg_path.exists():
            template_text = json.loads(tok_cfg_path.read_text()).get("chat_template")
    # Native reasoning templates consume one or both variables below. For an
    # ordinary instruct/base template, make an explicit API effort request
    # functional with a concise system directive instead of silently ignoring
    # it. The directive asks for internal verification, not exposed chain of
    # thought, and is absent when the caller did not request an effort level.
    if (reasoning_requested
            and (not template_text
                 or ("reasoning_effort" not in template_text
                     and "enable_thinking" not in template_text))):
        messages = _messages_with_effort_instruction(messages, reasoning)
    if template_text:
        if tools and canonical_hermes_tools:
            messages = _messages_with_canonical_hermes_tool_history(messages)
            messages = _prepend_system_content(
                messages,
                _tools_system_preamble(tools, compact_json=compact_json))
            template_tools = None
        else:
            template_tools = tools or None
        if tools and not canonical_hermes_tools and not _template_consumes_tools(template_text):
            messages = _prepend_system_content(
                messages,
                _tools_system_preamble(tools, compact_json=compact_json))
        context = {
            "messages": _messages_for_native_template(messages),
            "add_generation_prompt": True,
            "reasoning_effort": reasoning,
            "tools": template_tools,
        }
        if enable_thinking is not None:
            context["enable_thinking"] = enable_thinking
        if (engine.cfg.model_type == "qwen3_5_moe"
                and enable_thinking is False):
            # The released template inserts an explicit empty
            # <think></think> prefix before a no-thinking answer, but normally
            # removes it when that answer is rendered as history. That makes a
            # follow-up token stream diverge immediately before the assistant
            # content and defeats exact recurrent endpoint reuse. Preserve the
            # marker that was genuinely part of the previous model input.
            context["preserve_thinking"] = True
        if engine.cfg.model_type == "gpt_oss":
            import datetime

            context["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
        return _render_template(
            template_text, compact_json=compact_json, **context)
    # generic fallback: simple role-tagged transcript (fine for base models);
    # tools become a hermes-style system preamble the parser understands
    if tools:
        from .toolcalls import tools_preamble

        messages = [{"role": "system", "content": tools_preamble(tools)}] + list(messages)
    # 2026-07-13: this used to do `f"{m['role']}: {m['content']}\n"` for EVERY
    # message unconditionally — an assistant message carrying `tool_calls`
    # (content is None/absent, per the OpenAI convention) rendered literally
    # as "assistant: None", which is not a real multi-turn tool round trip,
    # just silently broken history. Serialize tool_calls back into the same
    # <tool_call>{...}</tool_call> form tools_preamble asks the model to
    # produce, so re-fed history is at least self-consistent.
    return _render_fallback_transcript(messages)


def _active_context_limit(engine) -> int:
    """Return the strictest positive model or runtime correctness limit.

    ``max_position_embeddings``/YaRN describes the RoPE capacity.  A smaller
    ``RuntimeConfig.context_bound`` can additionally be a correctness contract
    (currently GLM's DSA/indexer gate), so public request validation must honor
    both before streaming headers are sent.
    """
    model_limit = int(getattr(
        engine, "effective_max_position_embeddings",
        engine.cfg.max_position_embeddings,
    ) or 0)
    runtime_limit = int(getattr(getattr(engine, "rc", None), "context_bound", 0) or 0)
    limits = [limit for limit in (model_limit, runtime_limit) if limit > 0]
    return min(limits) if limits else 0


def _positive_token_limit(value, field: str) -> int:
    """Parse a request token budget without accepting bool/fraction/zero."""
    if isinstance(value, bool):
        raise RequestValidationError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RequestValidationError(f"{field} must be a positive integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise RequestValidationError(f"{field} must be a positive integer")
    if parsed <= 0:
        raise RequestValidationError(f"{field} must be a positive integer")
    return parsed


def _omitted_output_token_limit() -> int:
    value = os.environ.get(
        "VMODEL_OMITTED_MAX_OUTPUT_TOKENS",
        str(_DEFAULT_OMITTED_MAX_OUTPUT_TOKENS))
    return _positive_token_limit(value, "VMODEL_OMITTED_MAX_OUTPUT_TOKENS")


def _tool_request_controls(route: str, req: dict, tools: list[dict]):
    """Validate supported tool schemas/controls and return effective settings.

    Automatic calls use a triggered schema grammar; required/specific choices
    use a root grammar that permits only one or more schema-valid tool calls.
    Disabling tools and single-vs-parallel output filtering remain explicit.
    """
    names = []
    specific_name = None
    for index, tool in enumerate(tools):
        if route == "/chat/completions":
            if tool.get("type") != "function" or not isinstance(
                    tool.get("function"), dict):
                raise RequestValidationError(
                    f"tools[{index}] must be an OpenAI function tool")
            function = tool["function"]
            name = function.get("name")
            parameters = function.get("parameters")
            description = function.get("description")
        elif route == "/responses":
            if tool.get("type") != "function":
                raise RequestValidationError(
                    f"tools[{index}] must be a Responses function tool")
            function = tool
            name = tool.get("name")
            parameters = tool.get("parameters")
            description = tool.get("description")
        elif route == "/messages":
            if tool.get("type") not in (None, "custom"):
                raise RequestValidationError(
                    f"tools[{index}] must be an Anthropic client tool")
            function = tool
            name = tool.get("name")
            parameters = tool.get("input_schema")
            description = tool.get("description")
            if "input_schema" not in tool:
                raise RequestValidationError(
                    f"tools[{index}].input_schema must be an object")
        else:
            # Legacy text completions do not define a tools surface.
            if tools:
                raise RequestValidationError(
                    "tools are not supported by the completions endpoint")
            return [], "none", False

        if not isinstance(name, str) or not name:
            raise RequestValidationError(
                f"tools[{index}] function name must be a non-empty string")
        if route == "/messages" and not isinstance(parameters, dict):
            raise RequestValidationError(
                f"tools[{index}].input_schema must be an object")
        if (route != "/messages" and parameters is not None
                and not isinstance(parameters, dict)):
            raise RequestValidationError(
                f"tools[{index}].parameters must be an object or null")
        if description is not None and not isinstance(description, str):
            raise RequestValidationError(
                f"tools[{index}].description must be a string or null")
        if parameters is not None:
            try:
                from .structured import check_json_schema

                check_json_schema(parameters)
            except (RuntimeError, ValueError) as error:
                raise RequestValidationError(
                    f"tools[{index}] has invalid JSON Schema: {error}") from error
        names.append(name)

    if len(names) != len(set(names)):
        duplicate = next(name for name in names if names.count(name) > 1)
        raise RequestValidationError(f"duplicate tool function name: {duplicate!r}")

    if route == "/messages":
        raw_choice = req.get("tool_choice")
        if raw_choice is None:
            choice = "auto" if tools else "none"
            disable_parallel = False
        else:
            if not isinstance(raw_choice, dict):
                raise RequestValidationError(
                    "Anthropic tool_choice must be an object")
            choice = raw_choice.get("type")
            if choice not in ("auto", "none", "any", "tool"):
                raise RequestValidationError(
                    "Anthropic tool_choice.type must be auto|none|any|tool")
            disable_parallel = raw_choice.get("disable_parallel_tool_use", False)
            if not isinstance(disable_parallel, bool):
                raise RequestValidationError(
                    "tool_choice.disable_parallel_tool_use must be a boolean")
            if choice == "tool":
                chosen_name = raw_choice.get("name")
                if not isinstance(chosen_name, str) or not chosen_name:
                    raise RequestValidationError(
                        "tool_choice.name must be a non-empty string")
                if chosen_name not in names:
                    raise RequestValidationError(
                        f"tool_choice names unknown tool {chosen_name!r}")
                specific_name = chosen_name
        allow_parallel = not disable_parallel
        required = choice in ("any", "tool")
        if choice == "any":
            choice = "required"
        elif choice == "tool":
            choice = f"specific:{specific_name}"
    else:
        raw_parallel = req.get("parallel_tool_calls", True)
        if not isinstance(raw_parallel, bool):
            raise RequestValidationError("parallel_tool_calls must be a boolean")
        allow_parallel = raw_parallel
        raw_choice = req.get("tool_choice")
        if raw_choice is None:
            choice = "auto" if tools else "none"
            required = False
        elif isinstance(raw_choice, str):
            if raw_choice not in ("auto", "none", "required"):
                raise RequestValidationError(
                    "tool_choice must be none|auto|required or a function object")
            choice = raw_choice
            required = choice == "required"
        elif isinstance(raw_choice, dict):
            if route == "/chat/completions":
                function = raw_choice.get("function")
                chosen_name = function.get("name") if isinstance(function, dict) else None
                valid_shape = raw_choice.get("type") == "function"
            else:
                chosen_name = raw_choice.get("name")
                valid_shape = raw_choice.get("type") == "function"
            if not valid_shape or not isinstance(chosen_name, str) or not chosen_name:
                raise RequestValidationError(
                    "tool_choice function object must name a non-empty function")
            if chosen_name not in names:
                raise RequestValidationError(
                    f"tool_choice names unknown tool {chosen_name!r}")
            specific_name = chosen_name
            choice = f"specific:{chosen_name}"
            required = True
        else:
            raise RequestValidationError(
                "tool_choice must be none|auto|required or a function object")

    if required and not tools:
        raise RequestValidationError(
            "required/specific tool_choice requires at least one tool")
    effective = [] if choice == "none" else list(tools)
    if specific_name is not None:
        effective = [tool for tool, name in zip(tools, names)
                     if name == specific_name]
    return effective, choice, allow_parallel


def _request_sampling(route: str, req: dict):
    """Validate protocol sampling controls and return executable parameters."""
    from .sampler import SamplingParams

    for field in ("temperature", "top_p"):
        value = req.get(field)
        if value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise RequestValidationError(f"{field} must be a finite number")
        upper = 1.0 if field == "top_p" or route == "/messages" else 2.0
        if not 0.0 <= float(value) <= upper:
            raise RequestValidationError(f"{field} must be between 0 and {upper:g}")

    top_k = req.get("top_k", 0)
    if top_k is None:
        top_k = 0
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise RequestValidationError("top_k must be a non-negative integer")
    seed = req.get("seed")
    explicit_sampling = any(req.get(field) is not None for field in (
        "temperature", "top_p", "top_k", "seed"))
    temperature = req.get("temperature")
    if temperature is None:
        temperature = 1.0 if explicit_sampling else 0.0
    try:
        return SamplingParams(
            temperature=float(temperature),
            top_p=float(1.0 if req.get("top_p") is None else req["top_p"]),
            top_k=top_k,
            seed=seed,
        )
    except ValueError as error:
        raise RequestValidationError(str(error)) from error


_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _request_reasoning_controls(route: str, req: dict):
    """Map protocol-native effort controls to template/prompt behavior."""
    if route == "/chat/completions":
        raw = req.get("reasoning_effort")
        requested = raw is not None
        effort = raw or "low"
        enabled = (effort not in {"none", "minimal", "low"}
                   if requested else None)
        budget = None
    elif route == "/responses":
        reasoning = req.get("reasoning") or {}
        raw = reasoning.get("effort")
        requested = raw is not None
        effort = raw or "low"
        enabled = (effort not in {"none", "minimal", "low"}
                   if requested else None)
        budget = None
    elif route == "/messages":
        thinking = req.get("thinking") or {}
        kind = thinking.get("type")
        requested = kind is not None
        if kind in ("enabled", "adaptive"):
            effort = "high" if kind == "enabled" else "medium"
            enabled = True
        elif kind == "disabled":
            effort, enabled = "low", False
        else:
            effort, enabled = "low", None
        budget = thinking.get("budget_tokens")
    else:
        return "low", None, False, None

    if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
        raise RequestValidationError(
            "reasoning effort must be one of none|minimal|low|medium|high|xhigh")
    return effort, enabled, requested, budget


def _structured_output_request(route: str, req: dict):
    """Return ``(kind, schema, strict)`` for protocol structured output."""
    config = None
    if route == "/chat/completions":
        config = req.get("response_format")
    elif route == "/responses":
        text = req.get("text")
        if text is not None and not isinstance(text, dict):
            raise RequestValidationError("Responses text must be an object")
        config = (text or {}).get("format")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise RequestValidationError("structured output format must be an object")
    kind = config.get("type", "text")
    if kind == "text":
        return None
    if kind == "json_object":
        return "json", None, True
    if kind != "json_schema":
        raise RequestValidationError(
            "structured output type must be text|json_object|json_schema")

    payload = config.get("json_schema") if route == "/chat/completions" else config
    if not isinstance(payload, dict):
        raise RequestValidationError("json_schema format must be an object")
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        raise RequestValidationError("json_schema.schema must be an object")
    strict = payload.get("strict", True)
    if not isinstance(strict, bool):
        raise RequestValidationError("json_schema.strict must be a boolean")
    try:
        from .structured import check_json_schema

        check_json_schema(schema)
    except (RuntimeError, ValueError) as error:
        raise RequestValidationError(str(error)) from error
    return "json_schema", schema, strict


def _configure_constraint(engine, structured_output, tools: list[dict],
                          tool_choice: str, allow_parallel: bool):
    from .structured import (GrammarConstraint, JSONSchemaValidationError,
                             StructuredDecodingUnavailable)

    if structured_output is not None and tools and tool_choice != "none":
        raise RequestValidationError(
            "structured text output cannot be combined with enabled tools")
    try:
        if structured_output is not None:
            kind, schema, strict = structured_output
            return GrammarConstraint.json(
                engine, (schema if kind == "json_schema" else
                         {"type": "object", "additionalProperties": True}),
                strict=(strict if kind == "json_schema" else False))
        if tools and tool_choice != "none":
            specific = (tool_choice.split(":", 1)[1]
                        if tool_choice.startswith("specific:") else None)
            return GrammarConstraint.tools(
                engine, tools,
                required=(tool_choice == "required" or specific is not None),
                specific_name=specific,
                allow_parallel=allow_parallel and specific is None)
    except (JSONSchemaValidationError, StructuredDecodingUnavailable,
            RuntimeError, ValueError) as error:
        raise RequestValidationError(str(error)) from error
    return None


def _messages_for_structured_output(messages: list[dict], structured_output):
    if structured_output is None:
        return messages
    kind, schema, _strict = structured_output
    directive = "Return only one valid JSON value"
    if kind == "json_schema":
        directive += " matching this JSON Schema: " + json.dumps(
            schema, ensure_ascii=False, separators=(",", ":"))
    directive += ". Do not wrap it in Markdown fences or add commentary."
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system" and isinstance(
            copied[0].get("content"), str):
        copied[0]["content"] = copied[0]["content"] + "\n\n" + directive
    else:
        copied.insert(0, {"role": "system", "content": directive})
    return copied


def _validate_generation_controls(route: str, req: dict):
    """Reject unsupported wire controls and validate implemented controls."""
    sampling = _request_sampling(route, req)
    _request_reasoning_controls(route, req)
    _structured_output_request(route, req)

    if route in ("/chat/completions", "/completions"):
        n = req.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise RequestValidationError("n must be 1; multiple choices are unsupported")

        logprobs = req.get("logprobs")
        if logprobs not in (None, False, 0):
            raise RequestValidationError("logprobs are not supported")
        top_logprobs = req.get("top_logprobs")
        if top_logprobs not in (None, 0):
            raise RequestValidationError("top_logprobs are not supported")

        for field in ("presence_penalty", "frequency_penalty"):
            value = req.get(field)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))):
                raise RequestValidationError(f"{field} must be a number")
            if value not in (None, 0, 0.0):
                raise RequestValidationError(f"{field} is not supported")
        logit_bias = req.get("logit_bias")
        if logit_bias not in (None, {}):
            raise RequestValidationError("logit_bias is not supported")

    if route == "/chat/completions":
        reasoning_effort = req.get("reasoning_effort")
        if reasoning_effort is not None and not isinstance(reasoning_effort, str):
            raise RequestValidationError("reasoning_effort must be a string")
        if "functions" in req or "function_call" in req:
            raise RequestValidationError(
                "legacy functions/function_call are unsupported; use tools/tool_choice")
    elif route == "/completions":
        best_of = req.get("best_of", 1)
        if isinstance(best_of, bool) or not isinstance(best_of, int) or best_of != 1:
            raise RequestValidationError("best_of must be 1")
        if req.get("echo") not in (None, False):
            raise RequestValidationError("echo is not supported")
        if req.get("suffix") not in (None, ""):
            raise RequestValidationError("suffix is not supported")
    elif route == "/responses":
        if req.get("top_logprobs") not in (None, 0):
            raise RequestValidationError("top_logprobs are not supported")
        for field in ("previous_response_id", "conversation", "prompt"):
            if req.get(field) is not None:
                raise RequestValidationError(
                    f"Responses {field} is not supported by this stateless server")
        if req.get("background") not in (None, False):
            raise RequestValidationError("background Responses are not supported")
        if req.get("include") not in (None, []):
            raise RequestValidationError("Responses include expansions are not supported")
        if req.get("truncation") not in (None, "disabled"):
            raise RequestValidationError(
                "automatic truncation is not supported; shorten the input explicitly")
        text_config = req.get("text")
        if text_config is not None:
            if not isinstance(text_config, dict):
                raise RequestValidationError("Responses text must be an object")
            if text_config.get("verbosity") is not None:
                raise RequestValidationError("Responses text.verbosity is not supported")
        reasoning = req.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise RequestValidationError("reasoning must be an object")
            effort = reasoning.get("effort")
            if effort is not None and not isinstance(effort, str):
                raise RequestValidationError("reasoning.effort must be a string")
    elif route == "/messages":
        thinking = req.get("thinking")
        if thinking:
            if not isinstance(thinking, dict):
                raise RequestValidationError("thinking must be an object")
            thinking_type = thinking.get("type")
            if thinking_type not in ("disabled", "enabled", "adaptive"):
                raise RequestValidationError(
                    "thinking.type must be disabled|enabled|adaptive")
            if thinking_type == "enabled":
                budget = thinking.get("budget_tokens")
                if (isinstance(budget, bool) or not isinstance(budget, int)
                        or budget <= 0):
                    raise RequestValidationError(
                        "thinking.budget_tokens must be a positive integer")
    return sampling


def _parse_request_tool_calls(text: str, tools: list[dict], model_type: str,
                              allow_parallel: bool = True):
    if not tools:
        return text, []
    from .toolcalls import parse_tool_calls

    allowed_names = {
        function["name"]
        for tool in tools
        if isinstance((function := tool.get("function", tool)), dict)
        and isinstance(function.get("name"), str)
    }
    from .structured import tool_argument_schemas

    schemas = tool_argument_schemas(tools)
    content, calls = parse_tool_calls(
        text, model_type, allowed_names=allowed_names,
        argument_schemas=schemas)
    if not allow_parallel and len(calls) > 1:
        print(f"[server] parallel_tool_calls=false: keeping first of {len(calls)} calls",
              flush=True)
        calls = calls[:1]
    return content, calls


def _validate_context_budget(engine, prompt_tokens: int, max_output_tokens: int,
                             *, prompt_label: str, output_label: str,
                             hint: str = "") -> int:
    context_limit = _active_context_limit(engine)
    if context_limit and prompt_tokens + max_output_tokens > context_limit:
        raise RequestValidationError(
            f"{prompt_label}({prompt_tokens})+{output_label}({max_output_tokens}) "
            f"exceeds active context limit={context_limit} "
            f"({getattr(engine, 'rope_profile', 'released')}).{hint}")
    return context_limit


def _prepare_vision_prompt(engine, prompt: str, images):
    """Run allocation-free vision preflight and expose failures as HTTP 400."""
    from .qwen3vl import prepare_vl_prompt

    try:
        return prepare_vl_prompt(engine, prompt, images)
    except ValueError as error:
        raise RequestValidationError(str(error)) from error


def _load_vision_images(sources):
    from .toolcalls import load_image, load_video

    images = []
    for index, source in enumerate(sources, start=1):
        kind = ("video" if isinstance(source, dict)
                and source.get("type") == "video" else "image")
        try:
            if kind == "video":
                images.append(load_video(source.get("source")))
            else:
                images.append(load_image(source))
        except (OSError, ValueError) as error:
            raise RequestValidationError(
                f"invalid {kind} {index}: {error}") from error
    return images


def _fast_dense_resident_kv_projection(
        engine, mode: str, prompt_tokens: int, max_output_tokens: int):
    """Project the exact resident BF16 KV payload for dense text Qwen.

    Weight quantization does not change K/V activations.  Qwen3-4B's geometry
    is 147,456 resident bytes per position, so a harness prompt that merely
    looks like a moderate 28K context asks this 16-GB host to retain ~4.17 GB
    of KV before layer weights and scratch.  Keep this arithmetic independent
    of MLX so request validation and its regression tests remain CPU-only.
    """
    if mode not in ("fast", "fast-long"):
        return None
    cfg = getattr(engine, "cfg", None)
    rc = getattr(engine, "rc", None)
    if (cfg is None or rc is None
            or getattr(cfg, "model_type", "") not in ("qwen2", "qwen3")
            or getattr(cfg, "vision_config", None)
            or getattr(cfg, "num_experts", 0)):
        return None
    if getattr(rc, "max_kv_mb", 0):
        # Exact paging has its own explicit resident-byte budget.
        return None
    try:
        limit_mb = int(os.environ.get(
            "VMODEL_FAST_RESIDENT_KV_LIMIT_MB", "3000"))
    except ValueError as error:
        raise RequestValidationError(
            "VMODEL_FAST_RESIDENT_KV_LIMIT_MB must be an integer") from error
    if limit_mb < 0:
        raise RequestValidationError(
            "VMODEL_FAST_RESIDENT_KV_LIMIT_MB must be non-negative")
    if limit_mb == 0:
        return None
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", 0) or 0)
    if min(layers, kv_heads, head_dim) <= 0:
        return None
    bytes_per_token = layers * 2 * kv_heads * head_dim * 2
    # Reserve the KV that must exist before the first output token, not a
    # hypothetical fully-consumed output ceiling. The engine grows exact KV
    # incrementally and re-runs live admission during decode. Reserving all
    # 4,096 optional output positions up front collapsed the weight cache for
    # short tool calls that actually emitted only a handful of tokens.
    positions = prompt_tokens
    declared_positions = prompt_tokens + max_output_tokens
    projection = {
        "bytes_per_token": bytes_per_token,
        "positions": positions,
        "declared_positions": declared_positions,
        "projected_bytes": positions * bytes_per_token,
        "declared_projected_bytes": declared_positions * bytes_per_token,
        "limit_bytes": limit_mb * 1_000_000,
        "active_metal_bytes": 0,
        "retained_prompt_kv_bytes": 0,
        "orphan_prompt_kv_bytes": 0,
        "evictable_prompt_kv_bytes": 0,
        "active_after_prompt_kv_eviction_bytes": 0,
        "dynamic_projected_bytes": 0,
        "dynamic_ceiling_bytes": 0,
        "admission_margin_bytes": 400_000_000,
    }
    snapshot_fn = getattr(engine, "prompt_cache_memory_snapshot", None)
    if snapshot_fn is not None:
        snapshot = snapshot_fn()
        active = max(0, int(snapshot.get("active_metal_bytes", 0) or 0))
        retained = max(0, int(
            snapshot.get("retained_prompt_kv_bytes", 0) or 0))
        orphan = max(0, int(
            snapshot.get("orphan_prompt_kv_bytes", 0) or 0))
        evictable = max(0, int(
            snapshot.get("evictable_prompt_kv_bytes", retained + orphan) or 0))
        ceiling = max(0, int(snapshot.get("metal_ceiling_bytes", 0) or 0))
        active_after = max(0, active - min(active, evictable))
        projection.update({
            "active_metal_bytes": active,
            "retained_prompt_kv_bytes": retained,
            "orphan_prompt_kv_bytes": orphan,
            "evictable_prompt_kv_bytes": evictable,
            "active_after_prompt_kv_eviction_bytes": active_after,
            "dynamic_projected_bytes": (
                active_after + projection["projected_bytes"]
                + projection["admission_margin_bytes"]),
            "dynamic_ceiling_bytes": ceiling,
        })
    return projection


def _validate_fast_dense_resident_kv(
        engine, mode: str, prompt_tokens: int, max_output_tokens: int):
    projection = _fast_dense_resident_kv_projection(
        engine, mode, prompt_tokens, max_output_tokens)
    if (projection is not None
            and projection["projected_bytes"] > projection["limit_bytes"]):
        adaptive_spill_mb = int(getattr(
            getattr(engine, "rc", None), "adaptive_kv_spill_mb", 0) or 0)
        if adaptive_spill_mb:
            projection["adaptive_spill_required"] = 1
            projection["adaptive_spill_mb"] = adaptive_spill_mb
            return projection
        projected_mb = math.ceil(projection["projected_bytes"] / 1_000_000)
        limit_mb = projection["limit_bytes"] // 1_000_000
        raise RequestValidationError(
            "resident BF16 KV projection "
            f"({projection['positions']} positions, {projected_mb} MB) exceeds "
            f"the dense-Qwen safety limit ({limit_mb} MB). "
            "Reduce the rendered input: for large tool catalogs, enable "
            "VMODEL_FAST_TOOL_GATEWAY=1 or set "
            "VMODEL_FAST_TOOL_LIMIT=32 or lower. Exact dense-Qwen disk-paged "
            "KV is experimental and quarantined on this 16-GB host because "
            "the real 28k-token gate exceeded the swap-out limit. "
            "Set VMODEL_FAST_RESIDENT_KV_LIMIT_MB=0 only after an independent "
            "memory/swap gate.")
    if (projection is not None
            and projection["dynamic_ceiling_bytes"]
            and projection["dynamic_projected_bytes"]
            > projection["dynamic_ceiling_bytes"]):
        adaptive_spill_mb = int(getattr(
            getattr(engine, "rc", None), "adaptive_kv_spill_mb", 0) or 0)
        if adaptive_spill_mb:
            projection["adaptive_spill_required"] = 1
            projection["adaptive_spill_mb"] = adaptive_spill_mb
            return projection
        projected_mb = math.ceil(
            projection["dynamic_projected_bytes"] / 1_000_000)
        ceiling_mb = projection["dynamic_ceiling_bytes"] // 1_000_000
        evictable_mb = projection["evictable_prompt_kv_bytes"] // 1_000_000
        raise RequestValidationError(
            "live dense-Qwen Metal projection remains unsafe even after "
            f"evicting {evictable_mb} MB of retained prompt KV "
            f"({projected_mb} MB projected including margin; "
            f"{ceiling_mb} MB live ceiling). Reduce context/output or free "
            "unified memory; the request was rejected before generation.")
    return projection


_HIDDEN_TOOL_SEARCH_NAME = "vmodel_search_tools"
_HIDDEN_TOOL_ENABLE_NAME = "vmodel_enable_tools"
_HIDDEN_TOOL_ABSTAIN_NAME = "vmodel_no_suitable_tool"

# Hidden gateway activation is process-local routing metadata, not model state:
# a bounded hash-keyed LRU remembers only real function names.  Raw messages and
# schemas remain in the caller request/durable KV journal and are never copied
# into this table.  The durable per-phase KV checkpoints are the restart tier;
# this table merely keeps an execution catalog stable while the server lives.
_GATEWAY_ACTIVATION_LOCK = threading.Lock()
_GATEWAY_ACTIVATIONS: OrderedDict[str, tuple[str, ...]] = OrderedDict()
_GATEWAY_ACTIVATION_LIMIT = 256

_HIDDEN_GATEWAY_DECISION_POLICY = (
    "Private tool-routing phase: decide whether the latest request needs "
    "information or an action outside the conversation. If it needs the "
    "filesystem, current working directory, shell/CLI, browser/web, network, "
    "account, calendar, email, database, application, or any other external "
    "state, call one private catalog function immediately as your first "
    "output. Call vmodel_enable_tools when the previously enabled real tools "
    "can continue the same operation (including pagination or corrected "
    "arguments). Call vmodel_search_tools when a different capability may be "
    "needed or the previous tool was unsuitable. Never "
    "guess external state and never say that you will check, run, inspect, or "
    "use a tool later: call the search function now. Answer directly only when "
    "the answer follows from the conversation or stable general knowledge. If "
    "the latest message is a tool result, answer from that result unless a new "
    "external action is still required. This routing phase is hidden from the "
    "caller."
)

_HIDDEN_GATEWAY_REAL_TOOL_POLICY = (
    "Private tool-execution phase: catalog search has already selected the "
    "most relevant real tools. Call exactly one real tool now when any provided "
    "tool can perform the requested action or inspection. If and only if none "
    "of the provided real tools is suitable, call vmodel_no_suitable_tool. Do "
    "not answer with a plan, a promise to act later, guessed external state, "
    "or an unrelated tool call."
)
_HIDDEN_GATEWAY_ABSTAIN_TEXT = (
    "I couldn't find a suitable available tool for this request."
)

_GATEWAY_CONFIRMATION_RE = re.compile(
    r"^(?:do it|go ahead|proceed|please do|run it|check it|try it|yes|ok(?:ay)?)"
    r"[\s.!?]*$", re.IGNORECASE)
_GATEWAY_COMMITMENT_RE = re.compile(
    r"\b(?:i(?:'ll| will)|let me|i(?:'m| am) going to)\b.{0,100}"
    r"\b(?:check|verify|inspect|run|execute|list|open|browse|search|query|use)\b",
    re.IGNORECASE | re.DOTALL)
_GATEWAY_EXPLICIT_TOOL_RE = re.compile(
    r"\b(?:use|call|invoke)\s+(?:an?\s+)?(?:available\s+)?tool\b|"
    r"\btool\s+call\b", re.IGNORECASE)
_GATEWAY_ALWAYS_ACTION_RE = re.compile(
    r"^(?:(?:please|actually|now)\s+)*"
    r"(?:run|execute|download|upload|install|deploy|scan)\b|"
    r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:run|execute|download|upload|install|deploy|scan)\b",
    re.IGNORECASE)
_GATEWAY_VERIFY_REAL_RE = re.compile(
    r"^(?:(?:please|actually|now)\s+)*(?:check|verify|inspect)\b.*"
    r"(?:for real|actually|current|live|on disk|in the workspace)\b",
    re.IGNORECASE)
_GATEWAY_ACTION_VERB_RE = re.compile(
    r"\b(?:check|verify|inspect|list|open|browse|search|fetch|query|read|write|"
    r"edit|create|delete|send|schedule|build|test)\b",
    re.IGNORECASE)
_GATEWAY_EXTERNAL_RESOURCE_RE = re.compile(
    r"\b(?:files?|folders?|directories?|filesystem|paths?|workspace|"
    r"repositories|repository|repo|terminal|shell|commands?|browser|web|urls?|"
    r"websites?|email|inbox|calendar|database|spreadsheet|documents?|pdf|"
    r"images?|video|audio|codebase|source code|git|github|plex|server|"
    r"application|packages?)\b",
    re.IGNORECASE)
_GATEWAY_EXTERNAL_STATE_RE = re.compile(
    r"\b(?:cwd|pwd|working directory|current (?:folder|directory)|"
    r"what (?:folder|directory) (?:are|am) (?:we|i) in|"
    r"top[- ]level (?:folder|directory)|"
    r"largest (?:top[- ]level )?(?:folder|directory)|"
    r"smallest (?:top[- ]level )?(?:folder|directory)|"
    r"files? (?:in|inside|under) (?:this|the|our|my) |"
    r"(?:this|our|my) (?:workspace|repository|repo|filesystem|folder|directory)|"
    r"https?://|browser tab|terminal|shell command)\b",
    re.IGNORECASE)


def _gateway_message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part.get("output", "")))
            for part in content if isinstance(part, dict)
        )
    return ""


def _hidden_gateway_force_reason(messages: list[dict]) -> str | None:
    """Return a high-confidence reason to require hidden tool discovery.

    This is intentionally narrower than semantic tool retrieval. It only
    prevents the small serving model from hallucinating or promising action on
    unmistakable external-state requests. Ambiguous/general-knowledge turns
    still go through the model's ordinary auto decision.
    """
    if not messages:
        return None
    # A function result is followed by an answer turn, not another forced call.
    if messages[-1].get("role") == "tool":
        return None
    user_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if messages[index].get("role") == "user"),
        None,
    )
    if user_index is None:
        return None
    user_text = " ".join(
        _gateway_message_text(messages[user_index]).split())
    if not user_text:
        return None
    if _GATEWAY_EXPLICIT_TOOL_RE.search(user_text):
        return "explicit-tool-request"
    if (_GATEWAY_ALWAYS_ACTION_RE.search(user_text)
            or _GATEWAY_VERIFY_REAL_RE.search(user_text)
            or (_GATEWAY_ACTION_VERB_RE.search(user_text)
                and _GATEWAY_EXTERNAL_RESOURCE_RE.search(user_text))):
        return "external-action-imperative"
    if _GATEWAY_EXTERNAL_STATE_RE.search(user_text):
        return "external-state-inspection"
    if _GATEWAY_CONFIRMATION_RE.fullmatch(user_text):
        previous_assistant = next(
            (_gateway_message_text(messages[index])
             for index in range(user_index - 1, -1, -1)
             if messages[index].get("role") == "assistant"),
            "",
        )
        if _GATEWAY_COMMITMENT_RE.search(previous_assistant):
            return "confirmed-deferred-action"
    return None


def _hidden_gateway_decision_choice(
        tool_choice: str, force_reason: str | None) -> str:
    """Constrain only high-confidence action turns to hidden discovery."""
    return (
        f"specific:{_HIDDEN_TOOL_SEARCH_NAME}"
        if force_reason is not None else tool_choice
    )


def _hidden_tool_search_pair():
    """Return wrapped/raw copies of the gateway-only catalog search tool."""
    raw = {
        "type": "function",
        "name": _HIDDEN_TOOL_SEARCH_NAME,
        "description": (
            "Search the hidden tool catalog when the request needs workspace, "
            "browser, network, account, or other external action. For ordinary "
            "questions that can be answered directly, do not call this tool. "
            "If you call it, the tool call must be your first output, with no "
            "prose before the call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short capability-oriented search query.",
                },
                "max_results": {
                    "type": "integer", "minimum": 1, "maximum": 64,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
    wrapped = {"type": "function", "function": {
        "name": raw["name"],
        "description": raw["description"],
        "parameters": raw["parameters"],
    }}
    return wrapped, raw


def _hidden_tool_enable_pair():
    """Return the stable gateway-only reuse-current-catalog function."""
    raw = {
        "type": "function",
        "name": _HIDDEN_TOOL_ENABLE_NAME,
        "description": (
            "Reuse the real tools already enabled for this conversation when "
            "they can continue the same operation, such as pagination, retrying "
            "with corrected arguments, or interpreting a prior tool result. "
            "Use vmodel_search_tools instead when a different capability or "
            "replacement tool may be needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }
    wrapped = {"type": "function", "function": {
        "name": raw["name"],
        "description": raw["description"],
        "parameters": raw["parameters"],
    }}
    return wrapped, raw


def _hidden_gateway_virtual_pairs():
    """The decision catalog is exactly these two schemas on every turn."""
    search, search_raw = _hidden_tool_search_pair()
    enable, enable_raw = _hidden_tool_enable_pair()
    return [search, enable], [search_raw, enable_raw]


def _tool_function_name(tool: dict) -> str:
    function = tool.get("function", tool) if isinstance(tool, dict) else {}
    return str(function.get("name", "")) if isinstance(function, dict) else ""


def _hidden_gateway_conversation_key(
        model_id: str, tools: list[dict], messages: list[dict]) -> str:
    """Hash the stable conversation anchor and full catalog identity.

    Appended assistant/tool/user turns deliberately do not change this key.
    The first user turn plus any leading system/developer turns separates
    concurrent chats without requiring a harness-supplied conversation id.
    """
    from .toolcalls import canonical_tool_indices, compact_tool_schema

    anchor = []
    for message in messages:
        role = str(message.get("role", ""))
        if role in ("system", "developer"):
            anchor.append(message)
            continue
        if role == "user":
            anchor.append(message)
        break
    order = canonical_tool_indices(tools)
    catalog = [compact_tool_schema(tools[index]) for index in order]
    payload = {
        "model": model_id,
        "anchor": anchor,
        "catalog": catalog,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _hidden_gateway_activation_get(key: str, tools: list[dict]) -> tuple[str, ...]:
    available = {_tool_function_name(tool) for tool in tools}
    with _GATEWAY_ACTIVATION_LOCK:
        names = _GATEWAY_ACTIVATIONS.get(key, ())
        if names:
            _GATEWAY_ACTIVATIONS.move_to_end(key)
    return tuple(name for name in names if name in available)


def _hidden_gateway_activation_put(key: str, tools: list[dict]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(
        name for tool in tools if (name := _tool_function_name(tool))))
    with _GATEWAY_ACTIVATION_LOCK:
        _GATEWAY_ACTIVATIONS[key] = names
        _GATEWAY_ACTIVATIONS.move_to_end(key)
        while len(_GATEWAY_ACTIVATIONS) > _GATEWAY_ACTIVATION_LIMIT:
            _GATEWAY_ACTIVATIONS.popitem(last=False)
    return names


def _hidden_gateway_activation_clear() -> None:
    """Test/process lifecycle helper; durable KV is intentionally untouched."""
    with _GATEWAY_ACTIVATION_LOCK:
        _GATEWAY_ACTIVATIONS.clear()


def _hidden_tool_abstain_pair():
    """Return the gateway-only safe alternative to an irrelevant real call."""
    raw = {
        "type": "function",
        "name": _HIDDEN_TOOL_ABSTAIN_NAME,
        "description": (
            "Use only when none of the retrieved real tools can perform the "
            "requested action. Do not use this when any real tool is relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why none of the retrieved tools is suitable.",
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    }
    wrapped = {"type": "function", "function": {
        "name": raw["name"],
        "description": raw["description"],
        "parameters": raw["parameters"],
    }}
    return wrapped, raw


def _hidden_tool_gateway_enabled(mode: str, tool_count: int, tool_choice: str) -> bool:
    value = os.environ.get("VMODEL_FAST_TOOL_GATEWAY", "0")
    if value not in ("0", "1"):
        raise RequestValidationError("VMODEL_FAST_TOOL_GATEWAY must be 0 or 1")
    if value == "0" or mode not in ("fast", "fast-long") or tool_count <= 0:
        return False
    # A named function is already the smallest correct catalog. `none` has no
    # prompt tools after request normalization. Auto/required benefit from the
    # hidden discovery round.
    return tool_choice in ("auto", "required")


def _hidden_gateway_search_result_limit(activated_limit: int, search_cap: int,
                                        requested) -> int:
    """Bound model-authored search breadth without forcing it to take the cap."""
    if isinstance(requested, bool) or not isinstance(requested, int):
        requested = activated_limit
    return min(activated_limit, search_cap, max(1, requested))


def _hidden_gateway_catalogs(tools, raw_tools, messages, query: str | None = None,
                             limit: int = 32, *, activated_names=(),
                             expansion_limit: int = 4,
                             max_activated: int = 64):
    """Build the real-tool subset before/after the hidden discovery round."""
    from .toolcalls import pinned_tool_indices, rank_tool_indices

    if limit <= 0:
        raise RequestValidationError("hidden tool gateway limit must be positive")
    if expansion_limit <= 0:
        raise RequestValidationError(
            "hidden tool gateway expansion limit must be positive")
    if not limit <= max_activated <= 64:
        raise RequestValidationError(
            "hidden tool gateway activated limit must be between the search "
            "limit and 64")
    # The hidden query is authored by the serving model specifically for this
    # catalog lookup. Rank against that intent alone: re-inserting a ~20K system
    # prompt here lets generic framework vocabulary swamp both BM25 and vector
    # retrieval. Transcript requirements are preserved separately as hard pins.
    routed_messages = (
        [{"role": "user", "content": query}] if query else list(messages))
    pinned = pinned_tool_indices(tools, messages)
    name_to_index = {
        _tool_function_name(tool): index for index, tool in enumerate(tools)}
    activated = [
        name_to_index[name] for name in dict.fromkeys(activated_names)
        if name in name_to_index
    ]
    selected = set(activated) | set(pinned)
    retrieval_metadata = {
        "tool_retrieval_profile": "not_queried",
        "tool_embedding_status": "not_queried",
    }
    if query:
        ranking, retrieval_metadata = rank_tool_indices(
            tools, routed_messages, use_embeddings=True, return_metadata=True)
        top_index = ranking[0] if ranking else None
        top_already_activated = top_index in set(activated)
        if activated and top_already_activated:
            # Same capability/page/corrected-arguments path: preserve the exact
            # schema set and therefore the execution KV prefix.
            activation_profile = "stable-hit"
        else:
            target = max(limit, len(selected)) if not activated else min(
                max_activated, max(len(selected), len(activated) + expansion_limit))
            for index in ranking:
                if len(selected) >= target:
                    break
                selected.add(index)
            activation_profile = "initial" if not activated else "expanded"
        retrieval_metadata = {
            **retrieval_metadata,
            "gateway_activation_profile": activation_profile,
            "gateway_activation_previous_tools": len(activated),
            "gateway_activation_top_tool_reused": int(top_already_activated),
        }
    else:
        retrieval_metadata = {
            **retrieval_metadata,
            "gateway_activation_profile": (
                "loaded" if activated else "not_queried"),
            "gateway_activation_previous_tools": len(activated),
            "gateway_activation_top_tool_reused": 0,
        }
    # Prompt rendering canonicalizes by function name. Returning the same order
    # here makes state/metadata deterministic as well and keeps raw/wrapped
    # catalogs aligned.
    indices = sorted(selected, key=lambda index: _tool_function_name(tools[index]))
    return (
        [tools[index] for index in indices],
        [raw_tools[index] for index in indices],
        len(pinned),
        retrieval_metadata,
    )


def _prepare_chat_prompt(engine, model_dir: Path, messages: list[dict], reasoning: str,
                         tools: list[dict], raw_tools: list[dict], mode: str,
                         max_output_tokens: int, *,
                         enable_thinking: bool | None = None,
                         reasoning_requested: bool = False,
                         cache_namespace: str = "default"):
    """Apply explicitly side-quest-only tool compaction/retrieval and render.

    Fast mode keeps all tools by default but strips parameter-level prose and
    emits canonical compact JSON. ``VMODEL_FAST_TOOL_LIMIT=N`` additionally
    enables deterministic lexical top-N retrieval; zero (the default) disables
    retrieval. Lossless mode performs neither transform.

    Returns ``(prompt, prompt_tokens, prompt_tools, response_tools, metadata)``.
    The original response schemas remain available even when their prompt copy
    is compacted. Any tool shortlist is reported rather than silently hidden.
    """
    from .toolcalls import (canonical_tool_indices, canonicalize_tool_history,
                            compact_tool_schema, pinned_tool_indices,
                            rank_tool_indices)

    try:
        messages = canonicalize_tool_history(messages)
    except ValueError as error:
        raise RequestValidationError(str(error)) from error

    requested = len(tools)
    selected_indices = list(range(requested))
    fast_mode = mode in ("fast", "fast-long")
    compacted = fast_mode and bool(tools)
    limit = 0
    pinned_indices: list[int] = []
    retrieval_metadata = {}
    if tools:
        try:
            canonical_tool_indices(tools)
        except ValueError as e:
            raise RequestValidationError(str(e)) from e
    if fast_mode:
        # Exact "tool capsules" are catalog versions, not detached KV blocks:
        # each schema token causally attends to every preceding schema.  A
        # canonical prompt-only order collapses permutations of the same set to
        # one byte/token sequence, while the request-order copies below remain
        # untouched for protocol responses and execution.
        try:
            limit = int(os.environ.get("VMODEL_FAST_TOOL_LIMIT", "0"))
        except ValueError as e:
            raise RequestValidationError(
                "VMODEL_FAST_TOOL_LIMIT must be a non-negative integer") from e
        if limit < 0:
            raise RequestValidationError(
                "VMODEL_FAST_TOOL_LIMIT must be a non-negative integer")
        if limit and requested > limit:
            ranking, retrieval_metadata = rank_tool_indices(
                tools, messages, use_embeddings=True, return_metadata=True)
            pinned_indices = pinned_tool_indices(tools, messages)
            selected = set(pinned_indices)
            # The configured limit is a soft budget. Transcript-required tools
            # overflow it rather than vanishing from a later agent loop.
            budget = max(limit, len(selected))
            for index in ranking:
                if len(selected) >= budget:
                    break
                selected.add(index)
            selected_indices = sorted(selected)

    selected_tools = [tools[i] for i in selected_indices]
    selected_raw = [raw_tools[i] for i in selected_indices]
    if compacted:
        prompt_order = canonical_tool_indices(selected_tools)
        prompt_tools = [compact_tool_schema(selected_tools[i]) for i in prompt_order]
    else:
        prompt_tools = selected_tools
    effective_enable_thinking = enable_thinking
    if effective_enable_thinking is None and fast_mode:
        effective_enable_thinking = False
    canonical_hermes_tools = bool(
        compacted and engine.cfg.model_type == "qwen3_5_moe")
    prompt = _chat_prompt(
        engine, model_dir, messages, reasoning, tools=prompt_tools,
        compact_json=compacted,
        # Fast mode defaults to no hidden-thinking, but an explicit medium/high
        # API request overrides that default and is passed to native templates.
        enable_thinking=effective_enable_thinking,
        reasoning_requested=reasoning_requested,
        canonical_hermes_tools=canonical_hermes_tools)
    prompt_ids, prompt_offsets, prompt_token_cache_hit = _prepared_prompt_ids(
        engine, prompt)
    tool_capsules = (
        _tool_capsule_spans(
            prompt, prompt_tools, prompt_ids, prompt_offsets)
        if compacted else ())
    prompt_tokens = len(prompt_ids)

    resident_kv = _validate_fast_dense_resident_kv(
        engine, mode, prompt_tokens, max_output_tokens)

    # Never silently claim semantics beyond the active RoPE profile. Native
    # fast stays at the checkpoint limit; fast-long has a separately named,
    # cache-incompatible experimental YaRN profile.
    hint = ((" Use model id lossy-long-<name> for experimental Qwen2 YaRN, "
             "or set VMODEL_FAST_TOOL_LIMIT=32 (or lower).")
            if fast_mode and tools else "")
    context_limit = _validate_context_budget(
        engine, prompt_tokens, max_output_tokens,
        prompt_label="rendered prompt", output_label="max_output_tokens", hint=hint)

    metadata = {
        "requested": requested,
        "selected": len(selected_tools),
        "lossy_shortlist": len(selected_tools) != requested,
        "shortlist_soft_limit": limit,
        "pinned": len(pinned_indices),
        "tool_retrieval_profile": (
            retrieval_metadata.get(
                "tool_retrieval_profile", "hybrid-lexical-capability-v1")
            if limit else None),
        "schema_profile": "compact-no-nested-prose" if compacted else "released",
        "tool_order_profile": "canonical-name-v1" if compacted else "request-order",
        "tool_protocol_profile": (
            "canonical-hermes-v1" if canonical_hermes_tools else "released"),
        "tool_catalog_id": (
            hashlib.sha256(json.dumps(
                prompt_tools, ensure_ascii=False, separators=(",", ":"),
                sort_keys=True).encode("utf-8")).hexdigest()[:16]
            if prompt_tools else None
        ),
        "tool_capsule_spans": len(tool_capsules),
        "prompt_token_cache_hit": int(prompt_token_cache_hit),
        "thinking_profile": (
            "enabled" if effective_enable_thinking is True else
            "disabled" if effective_enable_thinking is False else
            "template-default"),
        "reasoning_effort": reasoning,
        "reasoning_requested": int(reasoning_requested),
        "context_profile": getattr(engine, "rope_profile", "released"),
        "context_limit": context_limit,
        "resident_kv_bytes_per_token": (
            resident_kv["bytes_per_token"] if resident_kv else 0),
        "resident_kv_projected_bytes": (
            resident_kv["projected_bytes"] if resident_kv else 0),
        "resident_kv_declared_projected_bytes": (
            resident_kv["declared_projected_bytes"] if resident_kv else 0),
        "resident_kv_limit_bytes": (
            resident_kv["limit_bytes"] if resident_kv else 0),
        "resident_kv_active_metal_bytes": (
            resident_kv["active_metal_bytes"] if resident_kv else 0),
        "resident_kv_retained_cache_bytes": (
            resident_kv["retained_prompt_kv_bytes"] if resident_kv else 0),
        "resident_kv_evictable_cache_bytes": (
            resident_kv["evictable_prompt_kv_bytes"] if resident_kv else 0),
        "resident_kv_dynamic_projected_bytes": (
            resident_kv["dynamic_projected_bytes"] if resident_kv else 0),
        "resident_kv_dynamic_ceiling_bytes": (
            resident_kv["dynamic_ceiling_bytes"] if resident_kv else 0),
        "resident_kv_paged": int(bool(
            getattr(getattr(engine, "rc", None), "max_kv_mb", 0)
            or (resident_kv or {}).get("adaptive_spill_required", 0))),
        **retrieval_metadata,
    }
    schema_chars = sum(len(json.dumps(t, ensure_ascii=False)) for t in prompt_tools)
    print(
        f"[server] rendered prompt={prompt_tokens} tokens, tools="
        f"{len(selected_tools)}/{requested}, tool_schema_chars={schema_chars}, "
        f"schema_profile={metadata['schema_profile']}, "
        f"tool_order={metadata['tool_order_profile']}, "
        f"prompt_token_cache_hit={metadata['prompt_token_cache_hit']}",
        flush=True,
    )
    return (PreparedPrompt(
                prompt, prompt_ids, tool_capsules,
                cache_namespace=cache_namespace,
                force_paged_kv=bool(
                    (resident_kv or {}).get("adaptive_spill_required", 0))),
            prompt_tokens, selected_tools,
            selected_raw, metadata)


def _render_fallback_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "assistant" and m.get("tool_calls"):
            calls_text = "".join(
                f'<tool_call>\n{{"name": "{c["function"]["name"]}", '
                f'"arguments": {c["function"]["arguments"]}}}\n</tool_call>\n'
                for c in m["tool_calls"])
            lines.append(f"assistant: {content or ''}{calls_text}")
        elif role == "tool":
            call_id = m.get("tool_call_id", "")
            lines.append(f"tool ({call_id}): {content}\n" if call_id else f"tool: {content}\n")
        else:
            lines.append(f"{role}: {content}\n")
    text = "".join(lines)
    return text + "assistant:"



# 2026-07-14: tool-call markers are plain text conventions embedded in normal
# generated output (hermes-style <tool_call>...</tool_call>, or gpt-oss's
# harmony channel), not a separate structured output head -- so a streaming
# handler can't tell a token isn't the start of a marker until later tokens
# confirm or refute it. _safe_emit_len finds how much of a growing buffer is
# UNAMBIGUOUSLY not part of any marker start, so that prefix can be streamed
# as real content immediately while only the genuinely ambiguous tail (up to
# a full marker's length, inclusive) is held back pending disambiguation.
_HOLDBACK_MARKERS = {
    "gpt_oss": ("<tool_call>", "<|channel|>", "to=functions.", "commentary"),
}
_DEFAULT_HOLDBACK_MARKERS = ("<tool_call>",)

# Hidden discovery is a start-of-output protocol: the virtual search call must
# be the first non-whitespace output.  These are therefore deliberately the
# complete starts of tool-call syntax rather than the broader fragments used
# by the ordinary anywhere-in-output holdback above.  In particular, an
# ordinary gpt-oss ``<|channel|>final`` response becomes direct as soon as the
# channel name diverges from these prefixes.
_HIDDEN_DECISION_MARKERS = {
    "gpt_oss": (
        "<tool_call>",
        "<|channel|>commentary to=functions.",
        "<|channel|>to=functions.",
        "commentary to=functions.",
        "to=functions.",
    ),
}


def _hidden_decision_markers(model_type: str) -> tuple[str, ...]:
    return _HIDDEN_DECISION_MARKERS.get(
        model_type, _DEFAULT_HOLDBACK_MARKERS)


def _hidden_marker_candidate(text: str, model_type: str) -> str:
    candidate = text.lstrip()
    if model_type == "gpt_oss":
        # Harmony permits one-or-more spaces between ``commentary`` and
        # ``to=functions``. Collapse them for prefix classification while
        # preserving the original bytes for eventual streaming.
        candidate = re.sub(r"\s+", " ", candidate)
    return candidate


def _safe_emit_len(pending: str, markers: tuple[str, ...]) -> int:
    if not markers or not pending:
        return len(pending)
    # A decode piece can finish a marker and contain trailing JSON in the same
    # string (for example ``<tool_call>{``).  Full markers must win over the
    # partial-suffix test below or that complete marker is incorrectly emitted.
    full_starts = [i for marker in markers if (i := pending.find(marker)) >= 0]
    if full_starts:
        return min(full_starts)
    # Window must cover a FULL marker match, not just len(marker)-1: a
    # window one character too narrow lets a completely-matched marker (the
    # exact bug this window exists to prevent) slide out of range and get
    # reported as "safe" once enough trailing characters accumulate after it.
    max_overlap = max(len(m) for m in markers)
    start = max(0, len(pending) - max_overlap)
    for i in range(start, len(pending)):
        suffix = pending[i:]
        if any(m.startswith(suffix) for m in markers):
            return i
    return len(pending)


class _MarkerHoldback:
    """Bounded text holdback until a tool-call marker is confirmed."""

    def __init__(self, markers: tuple[str, ...]):
        self.markers = markers
        self.pending = ""
        self.streamed = ""
        self.holding = False

    def feed(self, text: str) -> str:
        self.pending += text
        if self.holding:
            return ""
        safe_len = _safe_emit_len(self.pending, self.markers)
        safe, self.pending = self.pending[:safe_len], self.pending[safe_len:]
        self.streamed += safe
        if self.pending and any(self.pending.startswith(marker) for marker in self.markers):
            self.holding = True
        return safe

    def final_remainder(self, parsed_content: str) -> str:
        if not parsed_content.startswith(self.streamed):
            raise RuntimeError("streamed text is not a prefix of parsed content")
        return parsed_content[len(self.streamed):]


class _HiddenDecisionStream:
    """Stream a hidden-gateway decision once it cannot start with a call.

    Only the leading whitespace/marker-prefix ambiguity is buffered. Once the
    prefix diverges, the decision is permanently direct and its safe text is
    forwarded incrementally. A second bounded holdback prevents a late marker
    from leaking if a model violates the start-only hidden-search protocol;
    the caller can then remove that virtual call before flushing the tail.
    """

    def __init__(self, model_type: str, emit):
        self.model_type = model_type
        self.start_markers = _hidden_decision_markers(model_type)
        self.emit = emit
        self.branch = "undecided"
        self.prefix_pending = ""
        self.direct_holdback = _MarkerHoldback(
            _HOLDBACK_MARKERS.get(model_type, _DEFAULT_HOLDBACK_MARKERS))
        self.late_marker_detected = False
        self.finished = False

    def _feed_direct(self, text: str) -> None:
        safe = self.direct_holdback.feed(text)
        if self.direct_holdback.holding:
            self.late_marker_detected = True
        if safe:
            self.emit(safe)

    def feed(self, text: str) -> None:
        if self.finished or not text:
            return
        if self.branch == "tool":
            return
        if self.branch == "direct":
            self._feed_direct(text)
            return

        self.prefix_pending += text
        candidate = _hidden_marker_candidate(
            self.prefix_pending, self.model_type)
        if not candidate:
            return
        # Check a completed marker before the inverse prefix relationship:
        # equality satisfies both, but is already conclusively the tool path.
        if any(candidate.startswith(marker) for marker in self.start_markers):
            self.branch = "tool"
            return
        if any(marker.startswith(candidate) for marker in self.start_markers):
            return

        self.branch = "direct"
        pending, self.prefix_pending = self.prefix_pending, ""
        self._feed_direct(pending)

    def finish_direct(self, parsed_content: str) -> None:
        """Flush direct content after all hidden calls have been removed."""
        if self.finished:
            return
        if self.branch == "tool":
            raise RuntimeError("cannot finish a tool-prefixed decision as direct")
        if self.branch == "undecided":
            self.branch = "direct"
            pending, self.prefix_pending = self.prefix_pending, ""
            self._feed_direct(pending)
        remainder = self.direct_holdback.final_remainder(parsed_content)
        if remainder:
            self.emit(remainder)
        self.finished = True


def _cache_phase_telemetry(name: str, phase_result: dict) -> dict:
    """Stable per-inference cache accounting for multi-phase Responses calls."""
    stats = phase_result.get("path_stats") or {}
    phase_prompt_tokens = int(phase_result.get("prompt_tokens", 0) or 0)
    cached = min(phase_prompt_tokens, int(
        stats.get("prompt_cache_prefix_tokens", 0) or 0))
    pic_reused = int(stats.get("tool_pic_reused_tokens", 0) or 0)
    return {
        "phase": name,
        "cache_namespace": stats.get("prompt_cache_namespace", name),
        "input_tokens": phase_prompt_tokens,
        "cached_tokens": cached,
        "cache_write_tokens": min(phase_prompt_tokens, int(
            stats.get("prompt_cache_write_tokens", 0) or 0)),
        "effective_reused_tokens": min(
            phase_prompt_tokens, cached + pic_reused),
        "cache_source": stats.get("prompt_cache_source", "cold"),
        "exact_hit": int(stats.get("prompt_cache_exact_hit", 0) or 0),
        "tool_pic_reused_tokens": pic_reused,
        "tool_pic_selected_tokens": int(
            stats.get("tool_pic_selected_tokens", 0) or 0),
        "suffix_prefill_seconds": round(float(
            phase_result.get("prefill_s", 0.0) or 0.0), 4),
        "cache_lookup_seconds": round(float(
            stats.get("prompt_cache_lookup_s", 0.0) or 0.0), 4),
        "admission_evicted_slots": int(stats.get(
            "hot_prompt_admission_evicted_slots", 0) or 0),
        "admission_evicted_bytes": int(stats.get(
            "hot_prompt_admission_evicted_bytes", 0) or 0),
        "admission_system_available_bytes": int(stats.get(
            "hot_prompt_admission_system_available_bytes", 0) or 0),
        "admission_system_floor_bytes": int(stats.get(
            "hot_prompt_admission_system_floor_bytes", 0) or 0),
        "admission_governor_reservations": int(stats.get(
            "hot_prompt_admission_governor_reservations", 0) or 0),
    }


def _log_path_stats(result: dict, prompt_tokens: int) -> None:
    """F37 prompt-prefix caching (runtime/kv_store.py) has been enabled by
    default for every non-GLM model this whole time, but engine.generate()'s
    path_stats (prompt_cache_prefix_tokens, prompt_cache_exact_hit, etc.) was
    computed and then silently discarded -- server.py never read it, so
    there was no way to tell whether a real conversation's repeated system
    prompt/tool manifest was actually getting reused or not (2026-07-14,
    found investigating why a 131-tool harness request was so slow)."""
    stats = result.get("path_stats") or {}
    if stats.get("speculative_used"):
        proposed = int(stats.get("speculative_proposed", 0) or 0)
        accepted = int(stats.get("speculative_accepted", 0) or 0)
        acceptance = 100.0 * accepted / proposed if proposed else 0.0
        print(
            f"[server] speculation: accepted {accepted}/{proposed} "
            f"({acceptance:.0f}%), target_sweeps="
            f"{int(stats.get('speculative_target_sweeps', 0) or 0)}, "
            f"k={int(stats.get('speculative_k', 0) or 0)}, "
            f"draft_oov_fallbacks="
            f"{int(stats.get('speculative_draft_oov_fallbacks', 0) or 0)}",
            flush=True,
        )
    elif stats.get("speculative_enabled"):
        print(
            f"[server] speculation fallback: "
            f"{stats.get('speculative_fallback_reason', 'unknown')}",
            flush=True,
        )
    hit = stats.get("prompt_cache_exact_hit")
    prefix = stats.get("prompt_cache_prefix_tokens", 0)
    source = stats.get("prompt_cache_source", "unknown")
    namespace = stats.get("prompt_cache_namespace", "default")
    admission_evicted = int(
        stats.get("hot_prompt_admission_evicted_slots", 0) or 0)
    admission_bytes = int(
        stats.get("hot_prompt_admission_evicted_bytes", 0) or 0)
    lookup_s = float(stats.get("prompt_cache_lookup_s", 0.0) or 0.0)
    if stats.get("tool_pic"):
        print(
            f"[server] tool-pic[{namespace}]: exact_prefix={prefix}/{prompt_tokens}, "
            f"capsule_reused={int(stats.get('tool_pic_reused_tokens', 0) or 0)}, "
            f"selected={int(stats.get('tool_pic_selected_tokens', 0) or 0)}, "
            f"repaired={int(stats.get('tool_pic_repaired_tokens', 0) or 0)}, "
            f"prefill={result.get('prefill_s', 0.0):.3f}s, "
            f"projected={int(stats.get('tool_pic_projected_bytes', 0) or 0) / 1e9:.2f}GB, "
            f"engine_total={result.get('total_s', 0.0):.3f}s",
            flush=True,
        )
    elif prefix or hit:
        pct = (100.0 * prefix / prompt_tokens) if prompt_tokens else 0.0
        print(
            f"[server] prompt-cache[{namespace}]: reused {prefix}/{prompt_tokens} prefix "
            f"tokens ({pct:.0f}%) from {source} in {lookup_s:.3f}s"
            f"{' (exact hit, zero sweeps)' if hit else ''}; "
            f"suffix_prefill={result.get('prefill_s', 0.0):.3f}s, "
            f"engine_total={result.get('total_s', 0.0):.3f}s",
            flush=True,
        )
    else:
        print(
            f"[server] prompt-cache[{namespace}]: no prefix match ({prompt_tokens} tokens "
            f"prefilled cold); lookup={lookup_s:.3f}s, "
            f"prefill={result.get('prefill_s', 0.0):.3f}s, "
            f"snapshot_writes="
            f"{float(stats.get('prompt_snapshot_write_s', 0.0) or 0.0) + float(stats.get('postgen_snapshot_write_s', 0.0) or 0.0):.3f}s, "
            f"engine_total={result.get('total_s', 0.0):.3f}s",
            flush=True,
        )
    if admission_evicted:
        print(
            f"[server] prompt-cache[{namespace}] admission evicted "
            f"{admission_evicted} unmatched slot(s), "
            f"{admission_bytes / 1e9:.2f}GB resident KV",
            flush=True,
        )
    cache_hits = int(stats.get("expert_cache_hits", 0) or 0)
    cache_misses = int(stats.get("expert_cache_misses", 0) or 0)
    if cache_hits or cache_misses:
        total = cache_hits + cache_misses
        print(
            f"[server] expert-I/O: {cache_hits}/{total} resident hits "
            f"({100.0 * cache_hits / total:.0f}%), "
            f"store={int(stats.get('weight_store_bytes_read', 0) or 0) / 1e9:.2f}GB "
            f"(prefill={int(stats.get('prefill_weight_store_bytes_read', 0) or 0) / 1e9:.2f}, "
            f"decode={int(stats.get('decode_weight_store_bytes_read', 0) or 0) / 1e9:.2f}; "
            f"fast-tier={int(stats.get('weight_fast_tier_bytes', 0) or 0) / 1e9:.2f}), "
            f"evictions={int(stats.get('weight_cache_evictions', 0) or 0)}, "
            f"resident={int(stats.get('weight_cache_resident_bytes', 0) or 0) / 1e9:.2f}/"
            f"{int(stats.get('weight_cache_budget_bytes', 0) or 0) / 1e9:.2f}GB, "
            f"transient(layer/token)="
            f"{int(stats.get('layer_transient_bytes', 0) or 0) / 1e9:.2f}/"
            f"{int(stats.get('token_transient_bytes', 0) or 0) / 1e9:.2f}GB, "
            f"governor_reservations="
            f"{int(stats.get('governor_reservations', 0) or 0)}",
            flush=True,
        )


def _vision_protocol_timing(result: dict) -> dict:
    """Stable protocol fields backed by generic vision ``path_stats``."""
    stats = result.get("path_stats") or {}

    def metric(path_key: str, result_key: str | None = None, default=0):
        if path_key in stats:
            return stats[path_key]
        return result.get(result_key or path_key, default)

    return {
        "vision_cache_hits": int(metric("vision_cache_hits") or 0),
        "vision_cache_misses": int(metric("vision_cache_misses") or 0),
        "vision_prompt_cache_tower_skipped": int(metric(
            "vision_prompt_cache_tower_skipped") or 0),
        "vision_prompt_cache_prefix_tokens": int(metric(
            "prompt_cache_prefix_tokens",
            "vision_prompt_cache_prefix_tokens") or 0),
        "vision_prompt_cache_exact_hit": int(metric(
            "prompt_cache_exact_hit",
            "vision_prompt_cache_exact_hit") or 0),
        "vision_prompt_cache_stored": int(metric(
            "vision_prompt_cache_stored") or 0),
        "cache_source": str(metric(
            "prompt_cache_source", default="none") or "none"),
        "tool_pic": int(metric("tool_pic", "vision_tool_pic") or 0),
        "tool_pic_reused_tokens": int(metric(
            "tool_pic_reused_tokens",
            "vision_tool_pic_reused_tokens") or 0),
        "tool_pic_selected_tokens": int(metric(
            "tool_pic_selected_tokens",
            "vision_tool_pic_selected_tokens") or 0),
        "tool_pic_repaired_tokens": int(metric(
            "tool_pic_repaired_tokens",
            "vision_tool_pic_repaired_tokens") or 0),
        "tool_pic_memory_admitted": int(metric(
            "tool_pic_memory_admitted",
            "vision_tool_pic_memory_admitted") or 0),
        "tool_pic_projected_bytes": int(metric(
            "tool_pic_projected_bytes",
            "vision_tool_pic_projected_bytes") or 0),
        "tool_pic_system_available_bytes": int(metric(
            "tool_pic_system_available_bytes") or 0),
        "tool_pic_system_floor_bytes": int(metric(
            "tool_pic_system_floor_bytes") or 0),
        "tool_pic_system_memory_admitted": int(metric(
            "tool_pic_system_memory_admitted") or 0),
        "prompt_state_approximate": int(metric(
            "prompt_state_approximate") or 0),
    }


def _execution_profile_fields(engine) -> dict[str, str]:
    """Non-sensitive artifact identity for API reproducibility telemetry."""
    model_dir = Path(getattr(engine, "_model_dir", ""))
    store = getattr(engine, "store", None)
    rc = getattr(engine, "rc", None)
    marker = _voom_quantization_metadata(model_dir) if model_dir.name else None
    quantization = getattr(store, "quantization", {}) or {}
    try:
        bits = int(quantization.get("bits", 0))
        group = int(quantization.get("group_size", 0))
    except (AttributeError, TypeError, ValueError):
        bits = group = 0
    mode = (quantization.get("mode", "affine")
            if isinstance(quantization, dict) else "affine")
    if marker:
        profile = str(marker.get("profile", "derived"))
        weight_profile = f"{profile}-{mode}-q{bits}-g{group}"
    elif getattr(store, "on_disk_quantized", False):
        weight_profile = f"published-{mode}-q{bits}-g{group}"
    elif rc is not None and getattr(rc, "quant_bits", 0):
        weight_profile = (
            f"runtime-{rc.quant_mode}-q{rc.quant_bits}-g{rc.quant_group_size}")
    else:
        weight_profile = "released"
    if rc is not None and getattr(rc, "rerank_lm_head", False):
        weight_profile += (
            f"+head-{rc.rerank_lm_head_mode}"
            f"-q{rc.rerank_lm_head_bits}"
            f"-g{rc.rerank_lm_head_group_size}"
            f"-rerank{rc.rerank_lm_head_candidates}"
        )
    if rc is not None and getattr(rc, "resident_attention_mode", ""):
        weight_profile += (
            f"+attn-{rc.resident_attention_mode}"
            f"-q{rc.resident_attention_bits}"
            f"-g{rc.resident_attention_group_size}"
        )
    expert_top_k_by_layer = (
        tuple(getattr(rc, "expert_top_k_by_layer", ()))
        if rc is not None else ())
    if expert_top_k_by_layer:
        weight_profile += "+olmoe-topk-" + ".".join(
            str(top_k) for top_k in expert_top_k_by_layer)
    fields = {
        "vmodel_checkpoint": model_dir.name or "unknown",
        "vmodel_weight_profile": weight_profile,
    }
    draft_dir = getattr(engine, "_speculative_draft_dir", None)
    if draft_dir is not None:
        kind = getattr(engine, "_speculative_kind", "autoregressive")
        fields["vmodel_decode_profile"] = (
            f"exact-dspark-k{getattr(engine, '_speculative_k', 0)}"
            if kind == "dspark" else
            f"exact-speculative-k{getattr(engine, '_speculative_k', 0)}")
        fields["vmodel_draft_checkpoint"] = Path(draft_dir).name
    return fields


def _responses_output_items(text: str, tools: list[dict], model_type: str,
                            message_id: str, *,
                            message_status: str = "completed",
                            allow_parallel: bool = True) -> tuple[str, list[dict]]:
    """Build Responses output without losing text surrounding tool calls."""
    content, calls = _parse_request_tool_calls(
        text, tools, model_type, allow_parallel)
    output = []
    if content or not calls:
        output.append({
            "id": message_id, "type": "message", "role": "assistant",
            "status": message_status, "content": [{
                "type": "output_text", "text": content, "annotations": [],
            }],
        })
    output.extend({
        "id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
        "call_id": call["id"], "name": call["function"]["name"],
        "arguments": call["function"]["arguments"], "status": "completed",
    } for call in calls)
    return content, output


def _openai_finish_reason(result: dict, *, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"
    return "length" if result.get("termination_reason") == "length" else "stop"


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self) -> str:
        """Protocol dispatch is by ENDPOINT SHAPE (/chat/completions,
        /completions, /responses, /messages), not by URL prefix — accept
        both the versioned OpenAI/Anthropic convention (/v1/...) and the
        bare path, since different client SDKs default to different base
        URLs (2026-07-13, user request)."""
        # BaseHTTPRequestHandler exposes the request target, including its
        # query string. Protocol dispatch depends only on the path component.
        p = self.path.partition("?")[0]
        return p[3:] if p.startswith("/v1/") else p

    def do_GET(self):
        if self._route() == "/models":
            data = [{"id": k, "object": "model", "owned_by": "vmodel"} for k in _advertised_model_ids()]
            data += DOWNLOADS.pending_entries()  # in-flight/failed downloads, so status is discoverable here too
            data += PACKS.pending_entries()  # in-flight/failed auto-pack jobs, same style
            self._json(200, {"object": "list", "data": data})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        self._t0 = time.time()
        parsed = self._read_json_request()
        if parsed is None:
            return
        self._parsed_request = parsed
        try:
            (self._normalized_messages, self._image_sources,
             self._preloaded_images) = self._preflight_nested_request(parsed[1])
        except RequestValidationError as error:
            return self._json(400, {"error": str(error)})
        if os.environ.get("VMODEL_CAPTURE_REQUESTS"):
            try:
                cap_dir = ROOT / "logs" / "captured_requests"
                cap_dir.mkdir(parents=True, exist_ok=True)
                cap_path = cap_dir / (
                    f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json")
                cap_path.write_bytes(parsed[0])
                print(f"[server] captured raw request body -> {cap_path}", flush=True)
            except OSError as error:
                return self._json(500, {"error": f"request capture failed: {error}"})
        try:
            write_timeout = float(os.environ.get(
                "VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS",
                str(_DEFAULT_RESPONSE_WRITE_TIMEOUT_SECONDS)))
        except ValueError:
            return self._json(500, {"error": (
                "server configuration VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS "
                "must be a number")})
        if not math.isfinite(write_timeout) or write_timeout <= 0:
            return self._json(500, {"error": (
                "server configuration VMODEL_RESPONSE_WRITE_TIMEOUT_SECONDS "
                "must be finite and positive")})

        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(write_timeout)
            with INFER_LOCK:
                return self._do_post_locked()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            # No response can be recovered once a streamed socket has failed;
            # close it and, critically, leave the lock context immediately.
            self.close_connection = True
            return None
        finally:
            try:
                self.connection.settimeout(previous_timeout)
            except OSError:
                pass

    def _preflight_nested_request(self, req: dict):
        """Validate nested protocol items and fetch images outside INFER_LOCK."""
        from .toolcalls import (anthropic_messages_to_canonical,
                                canonicalize_tool_history,
                                merge_leading_system_messages,
                                normalize_messages,
                                responses_input_to_messages)

        route = self._route()
        try:
            if route == "/chat/completions":
                messages, image_sources = normalize_messages(
                    req.get("messages", []))
            elif route == "/responses":
                messages = responses_input_to_messages(
                    req.get("input", ""), req.get("instructions"))
                messages, image_sources = normalize_messages(messages)
            elif route == "/messages":
                messages = anthropic_messages_to_canonical(
                    req.get("messages", []), req.get("system"))
                messages, image_sources = normalize_messages(messages)
            else:
                messages, image_sources = [], []
            messages = merge_leading_system_messages(messages)
            messages = canonicalize_tool_history(messages)
        except ValueError as error:
            raise RequestValidationError(str(error)) from error
        images = _load_vision_images(image_sources) if image_sources else []
        return messages, image_sources, images

    def _read_json_request(self):
        """Receive and decode a bounded request before taking INFER_LOCK.

        ThreadingHTTPServer gives each connection its own thread, but inference
        is intentionally serialized.  Reading a client-controlled byte count
        while holding that global lock let one slow/incomplete upload stall an
        unrelated generation.  Keep socket I/O outside the critical section
        and put an explicit deadline on it.
        """
        route = self._route()
        if route not in ("/completions", "/chat/completions",
                         "/responses", "/messages"):
            # The declared body has not been consumed, so this connection
            # cannot safely carry another keep-alive request.
            self.close_connection = True
            self._json(404, {"error": "not found"})
            return None

        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except ValueError:
            self._json(400, {"error": "Content-Length must be an integer"})
            return None
        if length <= 0:
            self._json(400, {"error": "request body must be a non-empty JSON object"})
            return None
        try:
            max_body_mb = int(os.environ.get(
                "VMODEL_MAX_REQUEST_BODY_MB",
                str(_DEFAULT_MAX_REQUEST_BODY_BYTES // (1024 * 1024))))
        except ValueError:
            self._json(500, {"error":
                "server configuration VMODEL_MAX_REQUEST_BODY_MB must be an integer"})
            return None
        if max_body_mb <= 0:
            self._json(500, {"error":
                "server configuration VMODEL_MAX_REQUEST_BODY_MB must be positive"})
            return None
        max_body = max_body_mb * 1024 * 1024
        if length > max_body:
            self.close_connection = True
            self._json(413, {"error": (
                f"request body is {length} bytes; limit is {max_body} bytes "
                "(VMODEL_MAX_REQUEST_BODY_MB)")})
            return None

        try:
            timeout = float(os.environ.get(
                "VMODEL_REQUEST_READ_TIMEOUT_SECONDS",
                str(_DEFAULT_REQUEST_READ_TIMEOUT_SECONDS)))
        except ValueError:
            self._json(500, {"error": (
                "server configuration VMODEL_REQUEST_READ_TIMEOUT_SECONDS "
                "must be a number")})
            return None
        if not math.isfinite(timeout) or timeout <= 0:
            self._json(500, {"error": (
                "server configuration VMODEL_REQUEST_READ_TIMEOUT_SECONDS "
                "must be finite and positive")})
            return None

        previous_timeout = self.connection.gettimeout()
        try:
            self.connection.settimeout(timeout)
            raw = self.rfile.read(length)
        except TimeoutError:
            self.close_connection = True
            self._json(408, {"error": (
                f"request body was not received within {timeout:g} seconds "
                "(VMODEL_REQUEST_READ_TIMEOUT_SECONDS)")})
            return None
        finally:
            self.connection.settimeout(previous_timeout)
        if len(raw) != length:
            self.close_connection = True
            self._json(400, {"error": (
                f"incomplete request body: expected {length} bytes, received {len(raw)}")})
            return None
        try:
            def reject_constant(value):
                raise ValueError(
                    f"non-finite JSON constant is not allowed: {value}")

            req = json.loads(raw, parse_constant=reject_constant)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            detail = getattr(error, "msg", str(error))
            self._json(400, {"error": f"invalid JSON request body: {detail}"})
            return None
        if not isinstance(req, dict):
            self._json(400, {"error": "request body must be a JSON object"})
            return None
        return raw, req, length

    def _do_post_locked(self):
        raw, req, length = self._parsed_request
        try:
            route = self._route()
            requested_model = req.get("model", "SmolLM2-135M")
            if not isinstance(requested_model, str) or not requested_model:
                return self._json(400, {"error": "model must be a non-empty string"})
            requested_tools = req.get("tools")
            if requested_tools is None:
                requested_tools = []
            elif (not isinstance(requested_tools, list)
                  or any(not isinstance(tool, dict) for tool in requested_tools)):
                return self._json(400, {"error": "tools must be an array of objects"})
            effective_tools, requested_tool_choice, allow_parallel_tool_calls = \
                _tool_request_controls(route, req, requested_tools)
            sampling = _validate_generation_controls(route, req)
            self._sampling = sampling
            self._structured_output = _structured_output_request(route, req)
            self._constraint = None
            (self._reasoning_effort, self._enable_thinking,
             self._reasoning_requested,
             self._reasoning_budget) = _request_reasoning_controls(route, req)
            if route in ("/chat/completions", "/messages"):
                messages = req.get("messages", [])
                if (not isinstance(messages, list)
                        or any(not isinstance(message, dict) for message in messages)):
                    return self._json(400, {
                        "error": "messages must be an array of objects"})
                if route == "/messages":
                    # Validate optional fields before the protocol-required
                    # max_tokens field.  This keeps malformed values
                    # diagnosable even when a minimal validation probe omits
                    # max_tokens, and ensures request-shape errors are settled
                    # before any model lookup or generation admission.
                    stop_sequences = req.get("stop_sequences")
                    if stop_sequences is not None and (
                            not isinstance(stop_sequences, list)
                            or not all(isinstance(value, str)
                                       for value in stop_sequences)):
                        return self._json(400, {
                            "error": "stop_sequences must be a list of strings"})
                    thinking = req.get("thinking")
                    if thinking is not None and not isinstance(thinking, dict):
                        return self._json(400, {
                            "error": "thinking must be an object"})
            elif route == "/responses":
                input_value = req.get("input", "")
                if not isinstance(input_value, (str, list)):
                    return self._json(400, {
                        "error": "Responses input must be a string or an array"})
            elif not isinstance(req.get("prompt", ""), str):
                return self._json(400, {
                    "error": "prompt must be a string for this completions endpoint"})
            model_id, suffix_mode = split_model_mode(requested_model)
            requested_mode = (self.headers.get("X-VModel-Mode")
                              or req.get("vmodel_mode")
                              or suffix_mode or "lossless")
            if not isinstance(requested_mode, str):
                return self._json(400, {"error": "vmodel_mode must be a string"})
            mode = requested_mode.lower()
            if mode not in ("lossless", "fast", "fast-long"):
                return self._json(
                    400, {"error": "vmodel_mode must be lossless|fast|fast-long"})
            token_budget_defaulted = False
            if route == "/responses":
                token_field = "max_output_tokens"
                token_value = req.get("max_output_tokens", req.get("max_tokens"))
            elif route == "/chat/completions":
                if (req.get("max_completion_tokens") is not None
                        and req.get("max_tokens") is not None):
                    raise RequestValidationError(
                        "provide only one of max_completion_tokens or max_tokens")
                token_field = ("max_completion_tokens"
                               if req.get("max_completion_tokens") is not None
                               else "max_tokens")
                token_value = (req.get("max_completion_tokens")
                               if token_field == "max_completion_tokens"
                               else req.get("max_tokens"))
            elif route == "/messages":
                token_field = "max_tokens"
                token_value = req.get("max_tokens")
                if token_value is None:
                    raise RequestValidationError(
                        "Anthropic Messages requires max_tokens")
            else:
                token_field = "max_tokens"
                token_value = req.get("max_tokens")
            if token_value is None:
                max_tokens = _omitted_output_token_limit()
                token_budget_defaulted = True
            else:
                max_tokens = _positive_token_limit(token_value, token_field)
            self._output_token_budget_source = (
                "eos_safety_ceiling" if token_budget_defaulted else "request")
            self._max_output_tokens = max_tokens
            stream_value = req.get("stream", False)
            if not isinstance(stream_value, bool):
                return self._json(400, {"error": "stream must be a boolean"})
            stream = stream_value
            include_stream_usage = False
            if route in ("/chat/completions", "/completions"):
                stream_options = req.get("stream_options")
                if stream_options is not None:
                    if not isinstance(stream_options, dict):
                        return self._json(400, {
                            "error": "stream_options must be an object"})
                    for field in ("include_usage", "include_obfuscation"):
                        if (stream_options.get(field) is not None
                                and not isinstance(stream_options[field], bool)):
                            return self._json(400, {
                                "error": f"stream_options.{field} must be a boolean"})
                    if stream_options.get("include_obfuscation"):
                        return self._json(400, {
                            "error": "stream obfuscation is not supported"})
                    if stream_options and not stream:
                        return self._json(400, {
                            "error": "stream_options require stream=true"})
                    include_stream_usage = bool(
                        stream_options.get("include_usage", False))
            stop_req = req.get("stop")  # OpenAI: string or list[str] or omitted
            if stop_req is None:
                stop = []
            elif isinstance(stop_req, str):
                stop = [stop_req]
            elif (isinstance(stop_req, list)
                  and all(isinstance(value, str) for value in stop_req)):
                stop = stop_req
            else:
                return self._json(400, {
                    "error": "stop must be a string or a list of strings"})
            if route == "/responses":
                reasoning = req.get("reasoning")
                if reasoning is not None and not isinstance(reasoning, dict):
                    return self._json(400, {"error": "reasoning must be an object"})

            normalized_messages = self._normalized_messages
            image_srcs = self._image_sources
            # Diagnostic logging (2026-07-14): the stock BaseHTTPRequestHandler
            # log_message() hook only fires AFTER send_response(), i.e. after a
            # request has already finished -- a request that hangs mid-processing
            # (e.g. a large tool-injected prompt on a small model) never logs
            # anything at all until it completes, which looks identical to the
            # server being stuck versus just slow. Log what came IN, up front.
            _msgs = req.get("messages") or req.get("input") or []
            _tools = req.get("tools") or []

            def _clen(m):
                c = m.get("content", "") if isinstance(m, dict) else m
                if isinstance(c, str):
                    return len(c)
                if isinstance(c, list):
                    return sum(len(str(b.get("text", b))) if isinstance(b, dict) else len(str(b)) for b in c)
                return len(str(c))

            prompt_chars = sum(_clen(m) for m in _msgs) if isinstance(_msgs, list) else len(str(_msgs))
            system = req.get("system")  # Anthropic Messages puts the system prompt out-of-band
            if isinstance(system, str):
                prompt_chars += len(system)
            elif isinstance(system, list):
                prompt_chars += sum(_clen(b) for b in system)
            print(
                f"[server] <- {self._route()} model={model_id} mode={mode} "
                f"messages={len(_msgs) if isinstance(_msgs, list) else '?'} tools={len(_tools)} "
                f"prompt_chars~{prompt_chars} max_tokens={max_tokens}"
                f"{'(eos-ceiling)' if token_budget_defaulted else ''} stream={stream} "
                f"body_bytes={length}",
                flush=True,
            )
            # Greedy remains the omitted-control default. Explicit sampling
            # values were validated above and are passed to every text/vision
            # generation path; response telemetry reports the applied profile.
            requested_temperature = req.get("temperature")
            requested_top_p = req.get("top_p")
            requested_top_k = req.get("top_k")
            requested_seed = req.get("seed")

            try:
                model_dir = _resolve(model_id)
            except ModelDownloading as e:
                elapsed = time.time() - e.status["started_at"]
                return self._json(202, {
                    "error": (f"model '{model_id}' is not local yet -- a background "
                             f"download started {elapsed:.0f}s ago; retry in a few "
                             f"seconds. Poll GET /v1/models to see live status."),
                    "vmodel_download_status": "downloading",
                    "elapsed_seconds": round(elapsed, 1),
                })
            except ModelDownloadFailed as e:
                return self._json(422, {
                    "error": f"model '{model_id}' could not be downloaded or loaded: {e.error}",
                    "vmodel_download_status": "failed",
                })
            if mode == "fast":
                model_dir = _preferred_fast_artifact(model_dir)
            engine = MANAGER.get(model_dir, mode)

            tools = effective_tools
            tool_selection = None
            rendered_prompt_tokens = None
            if route == "/responses":
                return self._do_responses(
                    req, model_id, model_dir, engine, mode, max_tokens, stream, stop,
                    tools, requested_tools, requested_tool_choice,
                    allow_parallel_tool_calls, normalized_messages, image_srcs)
            if route == "/messages":
                return self._do_anthropic_messages(
                    req, model_id, model_dir, engine, mode, max_tokens, stream, stop,
                    tools, requested_tool_choice, allow_parallel_tool_calls,
                    normalized_messages, image_srcs)
            if route == "/chat/completions":
                msgs = _messages_for_structured_output(
                    normalized_messages, self._structured_output)
                if image_srcs and not engine.cfg.vision_config:
                    return self._json(400, {"error": (
                        f"model '{model_id}' has no vision tower — use a "
                        "Qwen3-VL model (e.g. Qwen3-VL-8B-Instruct) for image input")})
                prompt, rendered_prompt_tokens, tools, _raw_tools, tool_selection = \
                    _prepare_chat_prompt(
                        engine, model_dir, msgs, self._reasoning_effort,
                        tools, tools, mode, max_tokens,
                        enable_thinking=self._enable_thinking,
                        reasoning_requested=self._reasoning_requested)
                self._constraint = _configure_constraint(
                    engine, self._structured_output, tools,
                    requested_tool_choice, allow_parallel_tool_calls)
                kind = "chat.completion"
                if image_srcs:
                    # 2026-07-13: this used to ignore `stream` entirely and always
                    # return a full non-streaming JSON response, even when the
                    # client asked for an SSE stream — generate_vl already accepts
                    # an on_token callback (same as the text path), so there was no
                    # engine-side reason for the gap, only missing server wiring.
                    from .qwen3vl import generate_vl
                    images = self._preloaded_images
                    prepared_vl = _prepare_vision_prompt(engine, prompt, images)
                    rendered_prompt_tokens = len(prepared_vl["tokens"])
                    _validate_context_budget(
                        engine, rendered_prompt_tokens, max_tokens,
                        prompt_label="expanded vision prompt", output_label="max_tokens")
                    rid = f"vmdl-{uuid.uuid4().hex[:12]}"
                    t0 = time.time()
                    buffer_for_tools = bool(tools)

                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()

                        markers = (_HOLDBACK_MARKERS.get(
                            engine.cfg.model_type, _DEFAULT_HOLDBACK_MARKERS)
                            if buffer_for_tools else ())
                        holdback = _MarkerHoldback(markers) if buffer_for_tools else None

                        def write_vision_chunk(delta, finish_reason=None):
                            chunk = {"id": rid, "object": kind + ".chunk",
                                     "model": model_id, "choices": [{
                                         "index": 0, "delta": delta,
                                     "finish_reason": finish_reason}],
                                     **_execution_profile_fields(engine)}
                            if include_stream_usage:
                                chunk["usage"] = None
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()

                        def emit(tok: str):
                            if holdback is None:
                                write_vision_chunk({"content": tok})
                                return
                            safe = holdback.feed(tok)
                            if safe:
                                write_vision_chunk({"content": safe})
                            else:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()

                        def vision_progress(progress):
                            if progress.get("phase") == "vision":
                                detail = (f"vision {progress['completed_images']}/"
                                          f"{progress['total_images']}")
                            else:
                                detail = (f"prefill {progress.get('completed_tokens', 0)}/"
                                          f"{progress.get('total_tokens', 0)}")
                            self.wfile.write(f": {detail}\n\n".encode())
                            self.wfile.flush()

                        write_vision_chunk({"role": "assistant"})
                        result = generate_vl(
                            engine, prompt, images, max_tokens, on_token=emit, stop=stop,
                            on_progress=vision_progress,
                            prepared=prepared_vl, sampling=self._sampling,
                            constraint=self._constraint)
                        _log_path_stats(result, result.get("prompt_tokens", 0))
                        finish_reason = _openai_finish_reason(result)
                        if buffer_for_tools:
                            content, calls = _parse_request_tool_calls(
                                result["text"], tools, engine.cfg.model_type,
                                allow_parallel_tool_calls)
                            remainder = holdback.final_remainder(content)
                            if remainder:
                                write_vision_chunk({"content": remainder})
                            if calls:
                                write_vision_chunk({"tool_calls": [
                                    {**call, "index": index}
                                    for index, call in enumerate(calls)]})
                                finish_reason = _openai_finish_reason(
                                    result, has_tool_calls=True)
                        write_vision_chunk({}, finish_reason)
                        if include_stream_usage:
                            prompt_count = int(result.get(
                                "prompt_tokens", rendered_prompt_tokens) or 0)
                            completion_count = len(result["tokens"])
                            usage_chunk = {
                                "id": rid, "object": kind + ".chunk",
                                "model": model_id, "choices": [],
                                "usage": {
                                    "prompt_tokens": prompt_count,
                                    "completion_tokens": completion_count,
                                    "total_tokens": prompt_count + completion_count,
                                },
                                "vmodel_timing": {
                                    "vision_seconds": round(float(
                                        result.get("vision_s", 0.0)), 4),
                                    **_vision_protocol_timing(result),
                                },
                                **_execution_profile_fields(engine),
                            }
                            self.wfile.write(
                                f"data: {json.dumps(usage_chunk)}\n\n".encode())
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return

                    result = generate_vl(
                        engine, prompt, images, max_tokens, stop=stop,
                        prepared=prepared_vl, sampling=self._sampling,
                        constraint=self._constraint)
                    _log_path_stats(result, result.get("prompt_tokens", 0))
                    message = {"role": "assistant", "content": result["text"]}
                    finish = _openai_finish_reason(result)
                    if tools:
                        content, calls = _parse_request_tool_calls(
                            result["text"], tools, engine.cfg.model_type,
                            allow_parallel_tool_calls)
                        if calls:
                            message = {"role": "assistant", "content": content or None,
                                       "tool_calls": calls}
                            finish = _openai_finish_reason(result, has_tool_calls=True)
                    prompt_tokens = result.get("prompt_tokens", 0)
                    completion_tokens = len(result["tokens"])
                    return self._json(200, {
                        "id": rid, "object": kind, "created": int(t0), "model": model_id,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "vmodel_timing": {
                            "vision_seconds": round(float(
                                result.get("vision_s", 0.0)), 4),
                            "resident_pipelined_decode_steps": int(result.get(
                                "resident_pipelined_decode_steps", 0) or 0),
                            **_vision_protocol_timing(result),
                        },
                        "usage": {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": completion_tokens,
                                  "total_tokens": prompt_tokens + completion_tokens,
                                  "prefill_seconds": round(result["prefill_s"], 2),
                                  "resident_pipelined_decode_steps": int(
                                      result.get("resident_pipelined_decode_steps", 0)),
                                  "vmodel_mode": mode,
                                  "vmodel_sampling": self._sampling.profile,
                                  "requested_temperature": requested_temperature,
                                  "requested_top_p": requested_top_p,
                                  "requested_top_k": requested_top_k,
                                  "requested_seed": requested_seed,
                                  "vmodel_reasoning_effort": self._reasoning_effort,
                                  "vmodel_thinking_enabled": self._enable_thinking,
                                  **_execution_profile_fields(engine),
                                  **({"vmodel_tool_selection": tool_selection}
                                     if tool_selection else {}),
                                  **PACKS.status_fields(model_id)}})
            elif route == "/completions":
                prompt = req.get("prompt", "")
                rendered_prompt_tokens = len(engine.tokenizer.encode(prompt).ids)
                _validate_context_budget(
                    engine, rendered_prompt_tokens, max_tokens,
                    prompt_label="prompt", output_label="max_tokens")
                kind = "text_completion"
            else:
                return self._json(404, {"error": "not found"})

            rid = f"vmdl-{uuid.uuid4().hex[:12]}"
            t0 = time.time()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                buffer_for_tools = bool(tools) and kind == "chat.completion"
                markers = (_HOLDBACK_MARKERS.get(
                    engine.cfg.model_type, _DEFAULT_HOLDBACK_MARKERS)
                    if buffer_for_tools else ())
                holdback = _MarkerHoldback(markers) if buffer_for_tools else None

                def write_stream_chunk(value, finish_reason=None):
                    choice = {"index": 0, "finish_reason": finish_reason}
                    if kind == "chat.completion":
                        choice["delta"] = value
                    else:
                        choice["text"] = value
                    chunk = {"id": rid, "object": kind + ".chunk", "model": model_id,
                             "choices": [choice], **_execution_profile_fields(engine)}
                    if include_stream_usage:
                        chunk["usage"] = None
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                def emit(tok: str):
                    if holdback is None:
                        write_stream_chunk(
                            {"content": tok} if kind == "chat.completion" else tok)
                        return
                    safe = holdback.feed(tok)
                    if safe:
                        write_stream_chunk({"content": safe})
                    else:
                        # A heartbeat per held token is cheap and bounds idle
                        # time even when decode itself takes many seconds/token.
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()

                def on_progress(progress):
                    done = int(progress.get("completed_tokens", 0))
                    total = int(progress.get("total_tokens", 0))
                    self.wfile.write(f": prefill {done}/{total}\n\n".encode())
                    self.wfile.flush()

                if kind == "chat.completion":
                    write_stream_chunk({"role": "assistant"})
                result = engine.generate(
                    prompt, max_tokens, on_token=emit, stop=stop,
                    on_progress=on_progress, sampling=self._sampling,
                    constraint=self._constraint)
                _log_path_stats(result, result.get("prompt_tokens", 0))
                finish_reason = _openai_finish_reason(result)
                if buffer_for_tools:
                    content, calls = _parse_request_tool_calls(
                        result["text"], tools, engine.cfg.model_type,
                        allow_parallel_tool_calls)
                    remainder = holdback.final_remainder(content)
                    if remainder:
                        write_stream_chunk({"content": remainder})
                    if calls:
                        write_stream_chunk({"tool_calls": [
                            {**call, "index": index}
                            for index, call in enumerate(calls)]})
                        finish_reason = _openai_finish_reason(
                            result, has_tool_calls=True)
                # Always send a terminal choice before [DONE], including the
                # no-tools path. Some clients wait for finish_reason rather than
                # treating socket/data termination as the semantic finish.
                write_stream_chunk(
                    {} if kind == "chat.completion" else "", finish_reason)
                if include_stream_usage:
                    prompt_count = (rendered_prompt_tokens
                                    if rendered_prompt_tokens is not None
                                    else int(result.get("prompt_tokens", 0) or 0))
                    completion_count = len(result["tokens"])
                    usage_chunk = {
                        "id": rid, "object": kind + ".chunk", "model": model_id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_count,
                            "completion_tokens": completion_count,
                            "total_tokens": prompt_count + completion_count,
                        },
                        **_execution_profile_fields(engine),
                    }
                    self.wfile.write(
                        f"data: {json.dumps(usage_chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            result = engine.generate(
                prompt, max_tokens, stop=stop, sampling=self._sampling,
                constraint=self._constraint)
            _log_path_stats(result, result.get("prompt_tokens", 0))
            text = result["text"]
            if kind == "chat.completion":
                message = {"role": "assistant", "content": text}
                finish = _openai_finish_reason(result)
                if tools:
                    content, calls = _parse_request_tool_calls(
                        text, tools, engine.cfg.model_type,
                        allow_parallel_tool_calls)
                    if calls:
                        message = {"role": "assistant", "content": content or None,
                                   "tool_calls": calls}
                        finish = _openai_finish_reason(result, has_tool_calls=True)
                choice = {"index": 0, "message": message, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text,
                          "finish_reason": _openai_finish_reason(result)}
            prompt_tokens = (rendered_prompt_tokens if rendered_prompt_tokens is not None
                             else len(engine.tokenizer.encode(prompt).ids))
            completion_tokens = len(result["tokens"])
            self._json(200, {
                "id": rid, "object": kind, "created": int(t0), "model": model_id,
                "choices": [choice],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens,
                          "prefill_seconds": round(result["prefill_s"], 2),
                          "tokens_per_second": round(result["tok_per_s"], 3),
                          "vmodel_mode": mode,
                          "vmodel_sampling": self._sampling.profile,
                          "requested_temperature": requested_temperature,
                          "requested_top_p": requested_top_p,
                          "requested_top_k": requested_top_k,
                          "requested_seed": requested_seed,
                          "vmodel_reasoning_effort": self._reasoning_effort,
                          "vmodel_thinking_enabled": self._enable_thinking,
                          **_execution_profile_fields(engine),
                          **({"vmodel_tool_selection": tool_selection}
                             if tool_selection else {}),
                          **PACKS.status_fields(model_id)},
            })
        except RequestValidationError as e:
            return self._json(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
            return None
        except Exception as e:  # surface errors as JSON, keep the server alive
            import traceback

            traceback.print_exc()
            if isinstance(e, MemoryError):
                # A failed long prefill may own several GiB of KV through
                # engine.last_kv.  Release that request before an agent harness
                # retries, while preserving unrelated hot-prefix slots.
                try:
                    failed_engine = locals().get("engine")
                    if failed_engine is not None:
                        failed_engine.discard_failed_request_state()
                        import mlx.core as mx
                        mx.clear_cache()
                except Exception:
                    traceback.print_exc()
            try:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def _do_responses(self, req: dict, model_id: str, model_dir: Path, engine, mode: str,
                      max_output_tokens: int, stream: bool, stop: list,
                      effective_raw_tools: list[dict],
                      requested_raw_tools: list[dict], tool_choice: str,
                      allow_parallel_tool_calls: bool, msgs: list[dict],
                      image_srcs: list[str]) -> None:
        """OpenAI Responses API (POST /responses or /v1/responses). Schema
        verified against the installed `openai` SDK's Pydantic models
        (openai.types.responses.Response/ResponseStreamEvent) rather than
        from memory. 2026-07-13: extended from a text-only first pass to
        full tool-calling (function_call/function_call_output round trip),
        vision, streaming (typed SSE events), and reasoning/sampling-param
        honesty — the same standard as /v1/chat/completions."""
        instructions = req.get("instructions")
        requested_temperature = req.get("temperature")
        requested_top_p = req.get("top_p")
        requested_top_k = req.get("top_k")
        requested_seed = req.get("seed")
        progress_events = req.get("vmodel_progress_events", False)
        if not isinstance(progress_events, bool):
            raise RequestValidationError(
                "vmodel_progress_events must be a boolean")

        # Responses API tools are FLAT ({"type":"function","name",...}); convert
        # to the {"type":"function","function":{...}} shape tools_preamble/
        # parse_tool_calls already use (same convention as chat/completions).
        raw_tools = effective_raw_tools
        tools = [
            {"type": "function", "function": {
                "name": t.get("name"), "description": t.get("description", ""),
                "parameters": t.get("parameters") or {}}}
            if t.get("type") == "function" else t
            for t in raw_tools
        ]

        if image_srcs and not engine.cfg.vision_config:
            return self._json(400, {"error": (
                f"model '{model_id}' has no vision tower — use a "
                "Qwen3-VL model (e.g. Qwen3-VL-8B-Instruct) for image input")})

        msgs = _messages_for_structured_output(msgs, self._structured_output)
        all_tools = list(tools)
        all_raw_tools = list(raw_tools)
        gateway_enabled = (
            _hidden_tool_gateway_enabled(mode, len(all_tools), tool_choice)
            and not image_srcs
        )
        gateway_initial_tools = []
        gateway_initial_raw = []
        gateway_pinned = 0
        gateway_force_reason = None
        gateway_activation_key = ""
        gateway_activated_names: tuple[str, ...] = ()
        gateway_expansion_limit = 4
        gateway_max_activated = 64
        gateway_search_results = 4
        gateway_virtual_tools = []
        gateway_virtual_raw = []
        if gateway_enabled:
            try:
                gateway_limit = int(os.environ.get(
                    "VMODEL_FAST_TOOL_GATEWAY_LIMIT", "32"))
                gateway_expansion_limit = int(os.environ.get(
                    "VMODEL_FAST_TOOL_GATEWAY_EXPANSION_LIMIT", "4"))
                gateway_max_activated = int(os.environ.get(
                    "VMODEL_FAST_TOOL_GATEWAY_MAX_ACTIVATED", "64"))
                gateway_search_results = int(os.environ.get(
                    "VMODEL_FAST_TOOL_GATEWAY_SEARCH_RESULTS", "4"))
            except ValueError as error:
                raise RequestValidationError(
                    "VMODEL fast tool gateway limits must be integers") from error
            if not 1 <= gateway_limit <= 64:
                raise RequestValidationError(
                    "VMODEL_FAST_TOOL_GATEWAY_LIMIT must be in [1, 64]")
            if not 1 <= gateway_expansion_limit <= 16:
                raise RequestValidationError(
                    "VMODEL_FAST_TOOL_GATEWAY_EXPANSION_LIMIT must be in [1, 16]")
            if not gateway_limit <= gateway_max_activated <= 64:
                raise RequestValidationError(
                    "VMODEL_FAST_TOOL_GATEWAY_MAX_ACTIVATED must be between "
                    "VMODEL_FAST_TOOL_GATEWAY_LIMIT and 64")
            if not 1 <= gateway_search_results <= min(16, gateway_limit):
                raise RequestValidationError(
                    "VMODEL_FAST_TOOL_GATEWAY_SEARCH_RESULTS must be between "
                    "1 and min(16, VMODEL_FAST_TOOL_GATEWAY_LIMIT)")
            gateway_activation_key = _hidden_gateway_conversation_key(
                model_id, all_tools, msgs)
            gateway_activated_names = _hidden_gateway_activation_get(
                gateway_activation_key, all_tools)
            (gateway_initial_tools, gateway_initial_raw,
             gateway_pinned, gateway_initial_retrieval) = _hidden_gateway_catalogs(
                all_tools, all_raw_tools, msgs, limit=gateway_limit,
                activated_names=gateway_activated_names,
                expansion_limit=gateway_expansion_limit,
                max_activated=gateway_max_activated)
            gateway_force_reason = (
                "client-required" if tool_choice == "required"
                else _hidden_gateway_force_reason(msgs)
            )
            gateway_virtual_tools, gateway_virtual_raw = (
                _hidden_gateway_virtual_pairs())
            prompt_catalog = gateway_virtual_tools
            prompt_raw_catalog = gateway_virtual_raw
            decision_messages = _prepend_system_content(
                msgs, _HIDDEN_GATEWAY_DECISION_POLICY)
        else:
            gateway_limit = 0
            gateway_initial_retrieval = {}
            prompt_catalog = all_tools
            prompt_raw_catalog = all_raw_tools
            decision_messages = msgs

        prompt, prompt_tokens, prompt_tools, selected_raw_tools, tool_selection = \
            _prepare_chat_prompt(
                engine, model_dir, decision_messages, self._reasoning_effort,
                prompt_catalog,
                prompt_raw_catalog, mode, max_output_tokens,
                enable_thinking=self._enable_thinking,
                reasoning_requested=self._reasoning_requested,
                cache_namespace=(
                    "gateway_decision" if gateway_enabled else "default"))
        gateway_decision_choice = (
            f"specific:{_HIDDEN_TOOL_SEARCH_NAME}"
            if gateway_enabled and gateway_force_reason is not None
            else tool_choice
        )
        self._constraint = _configure_constraint(
            engine, self._structured_output, prompt_tools, gateway_decision_choice,
            False if gateway_enabled else allow_parallel_tool_calls)
        gateway_constraint = self._constraint
        # API response parsing never admits the gateway-only virtual function.
        # It may parse any caller-supplied real tool, while constrained decoding
        # ensures the model itself only sees/calls the phase's selected subset.
        response_parse_tools = all_tools
        response_raw_tools = (
            requested_raw_tools if gateway_enabled or tool_choice == "none"
            else selected_raw_tools)
        wire_tool_choice = (
            {"type": "function", "name": tool_choice.split(":", 1)[1]}
            if tool_choice.startswith("specific:") else tool_choice)
        rid = f"resp_{uuid.uuid4().hex[:24]}"
        response_message_id = f"msg_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        def build(text: str, n_tokens: int, result: dict | None = None) -> dict:
            result = result or {}
            path_stats = result.get("path_stats") or {}
            phases = result.get("vmodel_cache_phases") or [
                _cache_phase_telemetry("request", {
                    **result,
                    "prompt_tokens": int(
                        result.get("prompt_tokens", prompt_tokens) or 0),
                })
            ]
            usage_input_tokens = sum(int(
                phase.get("input_tokens", 0) or 0) for phase in phases)
            cached_tokens = sum(int(
                phase.get("cached_tokens", 0) or 0) for phase in phases)
            cache_write_tokens = sum(int(
                phase.get("cache_write_tokens", 0) or 0) for phase in phases)
            incomplete = result.get("termination_reason") == "length"
            content, output = _responses_output_items(
                text, response_parse_tools, engine.cfg.model_type, response_message_id,
                message_status="incomplete" if incomplete else "completed",
                allow_parallel=allow_parallel_tool_calls)
            output_text = content
            return {
                "id": rid, "object": "response", "created_at": created_at, "model": model_id,
                "status": "incomplete" if incomplete else "completed", "error": None,
                "incomplete_details": (
                    {"reason": "max_output_tokens"} if incomplete else None),
                "instructions": instructions, "metadata": {},
                "parallel_tool_calls": allow_parallel_tool_calls,
                "temperature": requested_temperature, "top_p": requested_top_p,
                "tool_choice": wire_tool_choice, "tools": response_raw_tools,
                "output": output, "output_text": output_text,
                "usage": {
                    "input_tokens": usage_input_tokens,
                    "input_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                    },
                    "output_tokens": n_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": usage_input_tokens + n_tokens,
                },
                "vmodel_cache_phases": phases,
                "vmodel_sampling": self._sampling.profile,
                "vmodel_max_output_tokens": max_output_tokens,
                "vmodel_output_budget_source": self._output_token_budget_source,
                "vmodel_top_k": requested_top_k,
                "vmodel_seed": requested_seed,
                "vmodel_reasoning_effort": self._reasoning_effort,
                "vmodel_thinking_enabled": self._enable_thinking,
                "vmodel_constraint": (
                    (result.get("path_stats") or {}).get(
                        "constraint_profile", result.get("constraint_profile", "none"))),
                "vmodel_tool_selection": tool_selection,
                **_execution_profile_fields(engine),
                "vmodel_timing": {
                    "vision_seconds": round(float(result.get("vision_s", 0.0)), 4),
                    "engine_prompt_tokenize_seconds": round(
                        float(path_stats.get("prompt_tokenize_s", 0.0)), 4),
                    "cache_lookup_seconds": round(
                        float(path_stats.get("prompt_cache_lookup_s", 0.0)), 4),
                    "suffix_prefill_seconds": round(float(result.get("prefill_s", 0.0)), 4),
                    "first_token_seconds": round(float(result.get("first_token_s", 0.0)), 4),
                    "decode_seconds": round(float(result.get("decode_s", 0.0)), 4),
                    "resident_pipelined_decode_steps": int(
                        result.get("resident_pipelined_decode_steps", 0) or 0),
                    "prompt_cache_write_seconds": round(float(
                        path_stats.get("prompt_snapshot_write_s", 0.0)), 4),
                    "postgen_cache_write_seconds": round(float(
                        path_stats.get("postgen_snapshot_write_s", 0.0)), 4),
                    "total_engine_seconds": round(float(result.get("total_s", 0.0)), 4),
                    "rope_profile": path_stats.get("rope_profile", "released"),
                    **_vision_protocol_timing(result),
                },
                **PACKS.status_fields(model_id),
            }

        def run_hidden_gateway(on_token=None, on_progress=None):
            """Run one hidden discovery decision and at most one real-tool pass."""
            nonlocal prompt, prompt_tokens, prompt_tools, selected_raw_tools
            nonlocal tool_selection

            decision_stream = (
                _HiddenDecisionStream(engine.cfg.model_type, on_token)
                if on_token is not None else None
            )
            decision = engine.generate(
                prompt, max_output_tokens, stop=stop,
                on_token=(decision_stream.feed if decision_stream is not None else None),
                on_progress=on_progress, sampling=self._sampling,
                constraint=gateway_constraint)
            decision_cache_phase = _cache_phase_telemetry(
                "gateway_decision", decision)
            decision_content, calls = _parse_request_tool_calls(
                decision["text"], prompt_tools, engine.cfg.model_type,
                allow_parallel=False)
            gateway_call = next(
                (call for call in calls
                 if call["function"]["name"] in (
                     _HIDDEN_TOOL_SEARCH_NAME, _HIDDEN_TOOL_ENABLE_NAME)),
                None,
            )
            # A hidden catalog action is valid only at the first non-whitespace
            # output.
            # Once direct text was streamed, retracting it would violate SSE's
            # append-only contract. Suppress only the late virtual call, while
            # preserving any real caller tool marker for the ordinary parser.
            late_gateway_suppressed = bool(
                gateway_call is not None
                and decision_stream is not None
                and decision_stream.branch == "direct"
            )
            if late_gateway_suppressed:
                visible_text, _hidden_calls = _parse_request_tool_calls(
                    decision["text"], gateway_virtual_tools,
                    engine.cfg.model_type,
                    allow_parallel=False)
                decision["text"] = visible_text
                decision_content, _visible_calls = _parse_request_tool_calls(
                    visible_text, response_parse_tools, engine.cfg.model_type,
                    allow_parallel=allow_parallel_tool_calls)
                gateway_call = None
            decision_branch = (
                decision_stream.branch
                if decision_stream is not None
                else ("tool" if gateway_call is not None else "direct")
            )
            if decision_branch == "undecided":
                decision_branch = "direct"
            decision_meta = {
                "requested": len(all_tools),
                "selected": len(gateway_initial_tools),
                "lossy_shortlist": len(gateway_initial_tools) != len(all_tools),
                "shortlist_soft_limit": gateway_limit,
                "gateway_search_result_cap": gateway_search_results,
                "pinned": gateway_pinned,
                "hidden_tool_gateway": True,
                "gateway_phase": "direct",
                "gateway_decision_prompt_tokens": int(
                    decision.get("prompt_tokens", prompt_tokens)),
                "gateway_decision_output_tokens": len(decision.get("tokens", ())),
                "gateway_search_rounds": 0,
                "gateway_search_forced": int(gateway_force_reason is not None),
                "gateway_force_reason": gateway_force_reason,
                "gateway_decision_branch": decision_branch,
                "gateway_direct_streaming": bool(
                    decision_stream is not None
                    and decision_stream.branch == "direct"),
                # Keep the old field for wire/log compatibility while exposing
                # the action-neutral name to new harnesses.
                "gateway_late_search_suppressed": int(
                    late_gateway_suppressed),
                "gateway_late_catalog_action_suppressed": int(
                    late_gateway_suppressed),
                "tool_retrieval_profile": "hybrid-lexical-capability-v1",
                **gateway_initial_retrieval,
            }
            if gateway_call is None:
                tool_selection = {**tool_selection, **decision_meta}
                decision["vmodel_cache_phases"] = [decision_cache_phase]
                if decision_stream is not None:
                    if decision_stream.branch == "tool":
                        # A leading marker was conclusive, but it was a real
                        # caller tool (or malformed marker), not hidden search.
                        # Replay through the ordinary marker-aware streamer.
                        if decision["text"]:
                            on_token(decision["text"])
                        decision["first_token_s"] = decision.get(
                            "total_s", decision.get("first_token_s", 0.0))
                    else:
                        decision_stream.finish_direct(decision_content)
                return decision

            gateway_call_name = gateway_call["function"]["name"]
            try:
                arguments = json.loads(gateway_call["function"]["arguments"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "hidden tool gateway produced invalid catalog arguments") from error
            if gateway_call_name == _HIDDEN_TOOL_ENABLE_NAME and gateway_initial_tools:
                query = ""
                routed_limit = gateway_limit
                selected_tools = gateway_initial_tools
                selected_raw = gateway_initial_raw
                pinned = gateway_pinned
                retrieval_meta = {
                    **gateway_initial_retrieval,
                    "gateway_activation_profile": "enabled",
                }
            else:
                if gateway_call_name == _HIDDEN_TOOL_SEARCH_NAME:
                    query = str(arguments.get("query", "")).strip()
                else:
                    # An initial/expired activation cannot be enabled. Preserve
                    # the model's action decision and perform a normal semantic
                    # lookup against the latest user intent.
                    from .toolcalls import semantic_tool_query

                    query = semantic_tool_query(msgs)
                if not query:
                    raise RuntimeError(
                        "hidden tool gateway produced an empty search query")
                # `gateway_limit` bounds the activated catalog, not the amount
                # one semantic lookup should eagerly materialize. Add only the
                # best few results; a later decision pass can expand by another
                # bounded batch when the first tool is unsuitable or a new
                # capability/page is required.
                routed_limit = _hidden_gateway_search_result_limit(
                    gateway_limit, gateway_search_results,
                    arguments.get("max_results", gateway_limit))
                (selected_tools, selected_raw, pinned,
                 retrieval_meta) = _hidden_gateway_catalogs(
                    all_tools, all_raw_tools, msgs, query=query,
                    limit=routed_limit,
                    activated_names=gateway_activated_names,
                    expansion_limit=gateway_expansion_limit,
                    max_activated=gateway_max_activated)
            _hidden_gateway_activation_put(
                gateway_activation_key, selected_tools)
            call_id = f"vmodel_gateway_{uuid.uuid4().hex[:12]}"
            gateway_call_arguments = (
                {} if gateway_call_name == _HIDDEN_TOOL_ENABLE_NAME else
                {"query": query, "max_results": routed_limit})
            internal_messages = list(msgs) + [
                {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {
                            "name": gateway_call_name,
                            "arguments": json.dumps(
                                gateway_call_arguments,
                                ensure_ascii=False, separators=(",", ":")),
                        },
                    }],
                },
                {
                    "role": "tool", "tool_call_id": call_id,
                    "name": gateway_call_name,
                    "content": json.dumps({
                        "status": "enabled",
                        "tools": [
                            tool["function"]["name"] for tool in selected_tools
                        ],
                    }, ensure_ascii=False, separators=(",", ":")),
                },
            ]
            execution_messages = _prepend_system_content(
                internal_messages, _HIDDEN_GATEWAY_REAL_TOOL_POLICY)
            abstain_tool, abstain_raw = _hidden_tool_abstain_pair()
            execution_tools = [*selected_tools, abstain_tool]
            execution_raw = [*selected_raw, abstain_raw]
            prompt, prompt_tokens, prompt_tools, selected_raw_tools, phase_meta = \
                _prepare_chat_prompt(
                    engine, model_dir, execution_messages, self._reasoning_effort,
                    execution_tools, execution_raw, mode, max_output_tokens,
                    enable_thinking=self._enable_thinking,
                    reasoning_requested=self._reasoning_requested,
                    cache_namespace="gateway_execution")
            self._constraint = _configure_constraint(
                engine, self._structured_output, prompt_tools, "required",
                False)
            result = engine.generate(
                prompt, max_output_tokens, stop=stop,
                on_progress=on_progress, sampling=self._sampling,
                constraint=self._constraint)
            execution_cache_phase = _cache_phase_telemetry(
                "gateway_execution", result)
            _execution_content, execution_calls = _parse_request_tool_calls(
                result["text"], prompt_tools, engine.cfg.model_type,
                allow_parallel=False)
            abstain_call = next(
                (call for call in execution_calls
                 if call["function"]["name"] == _HIDDEN_TOOL_ABSTAIN_NAME),
                None,
            )
            real_calls = [
                call for call in execution_calls
                if call["function"]["name"] != _HIDDEN_TOOL_ABSTAIN_NAME
            ]
            if abstain_call is not None:
                execution_outcome = "no_suitable_tool"
                result["text"] = _HIDDEN_GATEWAY_ABSTAIN_TEXT
            elif real_calls:
                execution_outcome = "real_tool"
            else:
                # Required structured generation can still hit a very small
                # output limit before closing its marker. Never expose that
                # partial virtual/real call as ordinary assistant prose.
                execution_outcome = "invalid_or_incomplete_tool_call"
                result["text"] = _HIDDEN_GATEWAY_ABSTAIN_TEXT
            if on_token is not None and result["text"]:
                # The execution choice was buffered because the abstention is
                # gateway-private. Replay only the public real call or the
                # converted ordinary explanation through the outer streamer.
                on_token(result["text"])
            hidden_total = float(decision.get("total_s", 0.0))
            result["prefill_s"] = (
                float(result.get("prefill_s", 0.0))
                + float(decision.get("prefill_s", 0.0)))
            result["decode_s"] = (
                float(result.get("decode_s", 0.0))
                + float(decision.get("decode_s", 0.0)))
            result["first_token_s"] = (
                float(result.get("first_token_s", 0.0)) + hidden_total)
            result["total_s"] = float(result.get("total_s", 0.0)) + hidden_total
            result["vmodel_cache_phases"] = [
                decision_cache_phase, execution_cache_phase]
            tool_selection = {
                **phase_meta,
                **decision_meta,
                "selected": len(selected_tools),
                "lossy_shortlist": len(selected_tools) != len(all_tools),
                "pinned": pinned,
                "gateway_phase": (
                    "enable" if gateway_call_name == _HIDDEN_TOOL_ENABLE_NAME
                    else "search"),
                "gateway_search_rounds": int(
                    gateway_call_name == _HIDDEN_TOOL_SEARCH_NAME),
                "gateway_enable_rounds": int(
                    gateway_call_name == _HIDDEN_TOOL_ENABLE_NAME),
                "gateway_catalog_action": gateway_call_name,
                "gateway_activated_tools": len(selected_tools),
                "gateway_execution_choice_required": True,
                "gateway_real_tool_required": False,
                "gateway_abstention_available": True,
                "gateway_execution_outcome": execution_outcome,
                "gateway_query_sha256": (
                    hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
                    if query else None),
                "gateway_requested_results": routed_limit,
                **retrieval_meta,
            }
            return result

        if image_srcs:
            from .qwen3vl import generate_vl

            images = self._preloaded_images
            prepared_vl = _prepare_vision_prompt(engine, prompt, images)
            prompt_tokens = len(prepared_vl["tokens"])
            _validate_context_budget(
                engine, prompt_tokens, max_output_tokens,
                prompt_label="expanded vision prompt", output_label="max_output_tokens")
            if stream:
                return self._stream_responses(
                    prompt, max_output_tokens, stop, engine, prompt_tools,
                    build, rid, model_id, created_at, instructions,
                    requested_temperature, requested_top_p, response_raw_tools,
                    response_message_id, wire_tool_choice, allow_parallel_tool_calls,
                    generate_fn=lambda on_token, on_progress: generate_vl(
                        engine, prompt, images, max_output_tokens,
                        on_token=on_token, stop=stop, on_progress=on_progress,
                        prepared=prepared_vl, sampling=self._sampling,
                        constraint=self._constraint),
                    progress_events=progress_events,
                )
            result = generate_vl(
                engine, prompt, images, max_output_tokens, stop=stop,
                prepared=prepared_vl, sampling=self._sampling,
                constraint=self._constraint)
            _log_path_stats(result, result.get("prompt_tokens", 0))
            return self._json(200, build(result["text"], len(result["tokens"]), result))

        if stream:
            return self._stream_responses(
                                          prompt, max_output_tokens, stop, engine,
                                          (response_parse_tools if gateway_enabled
                                           else prompt_tools),
                                          build, rid, model_id, created_at, instructions,
                                          requested_temperature, requested_top_p,
                                          response_raw_tools, response_message_id,
                                          wire_tool_choice, allow_parallel_tool_calls,
                                          generate_fn=(run_hidden_gateway
                                                       if gateway_enabled else None),
                                          progress_events=progress_events)

        result = (
            run_hidden_gateway()
            if gateway_enabled else
            engine.generate(
                prompt, max_output_tokens, stop=stop, sampling=self._sampling,
                constraint=self._constraint)
        )
        _log_path_stats(result, result.get("prompt_tokens", 0))
        self._json(200, build(result["text"], len(result["tokens"]), result))

    def _stream_responses(self, prompt, max_tokens, stop, engine, tools, build, rid, model_id,
                          created_at, instructions, temperature, top_p, raw_tools,
                          response_message_id, tool_choice, allow_parallel_tool_calls,
                          generate_fn=None, progress_events=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = [0]

        def emit(event_type: str, **fields):
            seq[0] += 1
            payload = {"type": event_type, "sequence_number": seq[0], **fields}
            print(f"[server] -> sse #{seq[0]} {event_type}", flush=True)
            try:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print(f"[server] !! client disconnected mid-stream at event "
                      f"#{seq[0]} ({event_type}) -- {seq[0]} of an expected 6-8 "
                      f"events for this response went out before the socket closed",
                      flush=True)
                raise

        in_progress = {"id": rid, "object": "response", "created_at": created_at,
                       "model": model_id, "status": "in_progress", "output": [],
                       "instructions": instructions, "tool_choice": tool_choice,
                       "tools": raw_tools,
                       "parallel_tool_calls": allow_parallel_tool_calls,
                       "temperature": temperature, "top_p": top_p}
        emit("response.created", response=in_progress)
        emit("response.in_progress", response=in_progress)

        msg_id = response_message_id
        # 2026-07-14, second pass: tool calls must still be parsed from the
        # WHOLE text at once (this runtime's tool-call marker -- hermes-style
        # <tool_call>...</tool_call>, or gpt-oss's harmony channel -- is a
        # plain text convention, not a separate structured output head), but
        # that no longer means withholding every token as text arrives. Only
        # the SUFFIX of the buffer that could still grow into a marker needs
        # holding back (_safe_emit_len); everything before that is provably
        # not part of a marker and streams as a real delta immediately, same
        # as the no-tools path. Two earlier, narrower attempts at this were
        # each real bugs found live against a Mastra/@vercel/ai-sdk harness:
        # (1) withholding EVERY event (not just deltas) during a long
        # buffered wait produced total SSE silence indistinguishable from a
        # hung server, well before a client's own idle/chunk timeout even
        # start counting down real progress -- fixed by a periodic raw SSE
        # comment (spec-legal, ignored by conformant parsers) whenever no
        # real delta went out. (2) a version that buffered the ENTIRE
        # response and only replayed it as one lump delta at the very end
        # still rendered nothing client-side whenever the client disconnected
        # before completion (a real risk here: a 131-tool/~6600-char prompt
        # measured 326s to fail after only 20 of 64 requested tokens) --
        # this version streams genuinely safe text as it's produced, so a
        # client that gives up mid-generation has still SEEN whatever text
        # came before any (still-withheld) suspected tool-call span.
        buffer_for_tools = bool(tools)
        markers = (_HOLDBACK_MARKERS.get(engine.cfg.model_type, _DEFAULT_HOLDBACK_MARKERS)
                  if buffer_for_tools else ())

        msg_item_added = [False]

        def ensure_msg_item_added():
            if not msg_item_added[0]:
                msg_item_added[0] = True
                emit("response.output_item.added", output_index=0,
                    item={"id": msg_id, "type": "message", "role": "assistant",
                          "status": "in_progress", "content": []})
                emit("response.content_part.added", item_id=msg_id, output_index=0, content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []})

        if not buffer_for_tools:
            ensure_msg_item_added()

        token_count = [0]
        holdback = _MarkerHoldback(markers) if buffer_for_tools else None

        def on_token(tok):
            if not buffer_for_tools:
                emit("response.output_text.delta", item_id=msg_id, output_index=0,
                    content_index=0, delta=tok, logprobs=[])
                return
            token_count[0] += 1
            safe_text = holdback.feed(tok)
            if safe_text:
                ensure_msg_item_added()
                emit("response.output_text.delta", item_id=msg_id, output_index=0,
                    content_index=0, delta=safe_text, logprobs=[])
            else:
                try:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    print(f"[server] !! client disconnected mid-stream during buffered "
                          f"generation (keepalive at token {token_count[0]}, "
                          f"{len(holdback.streamed)} chars already streamed safely)", flush=True)
                    raise

        def on_progress(progress):
            # SSE comments are ignored by conforming Responses clients but
            # reset proxy/client idle timers during a long cold prefill, before
            # the first generated token exists.
            try:
                if progress.get("phase") == "vision":
                    done = int(progress.get("completed_images", 0))
                    total = int(progress.get("total_images", 0))
                    label = "vision"
                else:
                    done = int(progress.get("completed_tokens", 0))
                    total = int(progress.get("total_tokens", 0))
                    label = "prefill"
                if progress_events:
                    emit(
                        f"response.vmodel.{label}_progress",
                        phase=label,
                        completed=done,
                        total=total,
                        fraction=(done / total if total else 0.0),
                        cache_source=progress.get("cache_source", "cold"),
                    )
                else:
                    self.wfile.write(f": {label} {done}/{total}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("[server] !! client disconnected during prefill", flush=True)
                raise

        try:
            result = (
                generate_fn(on_token, on_progress)
                if generate_fn is not None else
                engine.generate(
                    prompt, max_tokens, on_token=on_token, stop=stop,
                    on_progress=on_progress, sampling=self._sampling,
                    constraint=self._constraint)
            )
        except (BrokenPipeError, ConnectionResetError):
            raise
        except Exception as error:
            # Headers and response.created are already on the wire. Attempting
            # to append an HTTP 500 JSON document to this SSE stream makes
            # Responses clients see an abrupt/truncated transport and retry the
            # identical multi-GiB request indefinitely. Terminate the protocol
            # correctly with response.failed instead.
            import traceback

            traceback.print_exc()
            if isinstance(error, MemoryError):
                try:
                    engine.discard_failed_request_state()
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    traceback.print_exc()
            failed = {
                **in_progress,
                "status": "failed",
                "error": {
                    "code": "server_memory_error" if isinstance(
                        error, MemoryError) else "server_error",
                    "message": f"{type(error).__name__}: {error}",
                },
            }
            emit("response.failed", response=failed)
            return
        _log_path_stats(result, result.get("prompt_tokens", 0))
        final = build(result["text"], len(result["tokens"]), result)
        message_item = next(
            (item for item in final["output"] if item["type"] == "message"), None)
        call_items = [item for item in final["output"] if item["type"] == "function_call"]

        if not call_items:
            # No real tool call after all: flush whatever was still held back.
            out_item = message_item
            text = out_item["content"][0]["text"] if out_item is not None else ""
            remainder = holdback.final_remainder(text) if holdback is not None else ""
            if remainder:
                ensure_msg_item_added()
                emit("response.output_text.delta", item_id=msg_id, output_index=0,
                    content_index=0, delta=remainder, logprobs=[])
            ensure_msg_item_added()  # covers an entirely-empty response too
            emit("response.output_text.done", item_id=msg_id, output_index=0,
                content_index=0, text=text, logprobs=[])
            emit("response.content_part.done", item_id=msg_id, output_index=0, content_index=0,
                part={"type": "output_text", "text": text, "annotations": []})
            emit("response.output_item.done", output_index=0, item=out_item)
        else:
            # A real tool call was recognized. Close out whatever plain text
            # was already safely streamed for msg_id (may be none, if the
            # marker started right away), then emit the function_call item(s).
            if message_item is not None:
                text = message_item["content"][0]["text"]
                remainder = holdback.final_remainder(text)
                if remainder:
                    ensure_msg_item_added()
                    emit("response.output_text.delta", item_id=msg_id, output_index=0,
                        content_index=0, delta=remainder, logprobs=[])
                holdback.streamed = text
            if msg_item_added[0]:
                emit("response.output_text.done", item_id=msg_id, output_index=0,
                    content_index=0, text=holdback.streamed, logprobs=[])
                emit("response.content_part.done", item_id=msg_id, output_index=0, content_index=0,
                    part={"type": "output_text", "text": holdback.streamed, "annotations": []})
                emit("response.output_item.done", output_index=0,
                    item=message_item)
            base_index = 1 if msg_item_added[0] else 0
            for offset, item in enumerate(call_items):
                i = base_index + offset
                added = {**item, "status": "in_progress", "arguments": ""}
                emit("response.output_item.added", output_index=i, item=added)
                emit("response.function_call_arguments.delta", item_id=item["id"],
                    output_index=i, delta=item["arguments"])
                emit("response.function_call_arguments.done", item_id=item["id"],
                    output_index=i, arguments=item["arguments"])
                emit("response.output_item.done", output_index=i, item=item)

        emit("response.incomplete" if final["status"] == "incomplete"
             else "response.completed", response=final)

    def _do_anthropic_messages(self, req: dict, model_id: str, model_dir: Path, engine, mode: str,
                               max_tokens: int, stream: bool, stop: list,
                               raw_tools: list[dict],
                               tool_choice: str, allow_parallel_tool_calls: bool,
                               msgs: list[dict],
                               image_srcs: list[str]) -> None:
        """Anthropic Messages API (POST /messages or /v1/messages). Schema
        verified against the installed `anthropic` SDK's Pydantic models
        (anthropic.types.Message/RawMessageStreamEvent) rather than from
        memory. `system` is a separate top-level field, not part of
        `messages`. 2026-07-13: extended from a text-only first pass to
        full tool_use/tool_result round trip, vision (base64/url image
        blocks), streaming (typed SSE events), and reasoning/sampling-param
        honesty."""
        import json as _json

        stop_sequences = req.get("stop_sequences") or []
        all_stop = list(stop) + list(stop_sequences)
        requested_temperature = req.get("temperature")
        requested_top_p = req.get("top_p")
        requested_top_k = req.get("top_k")
        requested_seed = req.get("seed")

        tools = [{"type": "function", "function": {
                     "name": t.get("name"), "description": t.get("description", ""),
                     "parameters": t.get("input_schema") or {}}}
                for t in raw_tools]

        if image_srcs and not engine.cfg.vision_config:
            return self._json(400, {"error": (
                f"model '{model_id}' has no vision tower — use a "
                "Qwen3-VL model (e.g. Qwen3-VL-8B-Instruct) for image input")})

        prompt, prompt_tokens, tools, raw_tools, tool_selection = _prepare_chat_prompt(
            engine, model_dir, msgs, self._reasoning_effort, tools, raw_tools,
            mode, max_tokens, enable_thinking=self._enable_thinking,
            reasoning_requested=self._reasoning_requested)
        self._constraint = _configure_constraint(
            engine, self._structured_output, tools,
            tool_choice, allow_parallel_tool_calls)

        def build_content(text: str, termination_reason: str):
            content, calls = _parse_request_tool_calls(
                text, tools, engine.cfg.model_type, allow_parallel_tool_calls)
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for c in calls:
                blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}",
                              "name": c["function"]["name"],
                              "input": _json.loads(c["function"]["arguments"])})
            stop_reason = ("tool_use" if calls else
                          "stop_sequence" if termination_reason == "stop_sequence" else
                          "max_tokens" if termination_reason == "length" else "end_turn")
            return blocks, stop_reason

        if image_srcs:
            from .qwen3vl import generate_vl

            images = self._preloaded_images
            prepared_vl = _prepare_vision_prompt(engine, prompt, images)
            prompt_tokens = len(prepared_vl["tokens"])
            _validate_context_budget(
                engine, prompt_tokens, max_tokens,
                prompt_label="expanded vision prompt", output_label="max_tokens")
            if stream:
                return self._stream_anthropic_messages(
                    prompt, max_tokens, all_stop, engine, tools,
                    build_content, model_id, prompt_tokens,
                    generate_fn=lambda on_token, on_progress: generate_vl(
                        engine, prompt, images, max_tokens,
                        on_token=on_token, stop=all_stop,
                        on_progress=on_progress, prepared=prepared_vl,
                        sampling=self._sampling, constraint=self._constraint),
                )
            result = generate_vl(
                engine, prompt, images, max_tokens, stop=all_stop,
                prepared=prepared_vl, sampling=self._sampling,
                constraint=self._constraint)
            _log_path_stats(result, result.get("prompt_tokens", 0))
            blocks, stop_reason = build_content(
                result["text"], result.get("termination_reason", "eos"))
            return self._json(200, {
                "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
                "model": model_id, "content": blocks, "stop_reason": stop_reason,
                "stop_sequence": (result.get("stop_sequence")
                                  if stop_reason == "stop_sequence" else None),
                "usage": {"input_tokens": prompt_tokens, "output_tokens": len(result["tokens"])},
                "vmodel_timing": {
                    "vision_seconds": round(float(result.get("vision_s", 0.0)), 4),
                    "resident_pipelined_decode_steps": int(
                        result.get("resident_pipelined_decode_steps", 0) or 0),
                    **_vision_protocol_timing(result),
                },
                "vmodel_sampling": self._sampling.profile,
                "requested_temperature": requested_temperature,
                "requested_top_p": requested_top_p,
                "requested_top_k": requested_top_k,
                "requested_seed": requested_seed,
                "vmodel_reasoning_effort": self._reasoning_effort,
                "vmodel_thinking_enabled": self._enable_thinking,
                "vmodel_constraint": result.get("constraint_profile", "none"),
                "vmodel_tool_selection": tool_selection,
                **_execution_profile_fields(engine),
                **PACKS.status_fields(model_id),
            })

        if stream:
            return self._stream_anthropic_messages(prompt, max_tokens, all_stop, engine, tools,
                                                    build_content, model_id, prompt_tokens)

        result = engine.generate(
            prompt, max_tokens, stop=all_stop, sampling=self._sampling,
            constraint=self._constraint)
        _log_path_stats(result, result.get("prompt_tokens", 0))
        blocks, stop_reason = build_content(
            result["text"], result.get("termination_reason", "eos"))
        self._json(200, {
            "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
            "model": model_id, "content": blocks, "stop_reason": stop_reason,
            "stop_sequence": (result.get("stop_sequence")
                              if stop_reason == "stop_sequence" else None),
            "usage": {"input_tokens": prompt_tokens, "output_tokens": len(result["tokens"])},
            "vmodel_timing": {
                "vision_seconds": round(float(result.get("vision_s", 0.0)), 4),
                "resident_pipelined_decode_steps": int(
                    result.get("resident_pipelined_decode_steps", 0) or 0),
            },
            "vmodel_sampling": self._sampling.profile,
            "requested_temperature": requested_temperature,
            "requested_top_p": requested_top_p,
            "requested_top_k": requested_top_k,
            "requested_seed": requested_seed,
            "vmodel_reasoning_effort": self._reasoning_effort,
            "vmodel_thinking_enabled": self._enable_thinking,
            "vmodel_constraint": (
                (result.get("path_stats") or {}).get(
                    "constraint_profile", result.get("constraint_profile", "none"))),
            "vmodel_tool_selection": tool_selection,
            **_execution_profile_fields(engine),
            **PACKS.status_fields(model_id),
        })

    def _stream_anthropic_messages(self, prompt, max_tokens, all_stop, engine, tools,
                                   build_content, model_id, prompt_tokens,
                                   generate_fn=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(event_type: str, data: dict):
            payload = {"type": event_type, **data}
            self.wfile.write(f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        emit("message_start", {"message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model_id,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            **_execution_profile_fields(engine)}})
        emit("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})

        # Stream text up to a possible tool marker, then parse/replay the
        # withheld suffix after generation. This preserves ordinary text and
        # text surrounding tool calls instead of dropping it merely because a
        # tools array was supplied.
        buffer_for_tools = bool(tools)
        markers = (_HOLDBACK_MARKERS.get(
            engine.cfg.model_type, _DEFAULT_HOLDBACK_MARKERS)
            if buffer_for_tools else ())
        holdback = _MarkerHoldback(markers) if buffer_for_tools else None

        def on_token(tok):
            if holdback is None:
                emit("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": tok}})
                return
            safe = holdback.feed(tok)
            if safe:
                emit("content_block_delta", {
                    "index": 0, "delta": {"type": "text_delta", "text": safe}})
            else:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()

        def on_progress(progress):
            if progress.get("phase") == "vision":
                done = int(progress.get("completed_images", 0))
                total = int(progress.get("total_images", 0))
                label = "vision"
            else:
                done = int(progress.get("completed_tokens", 0))
                total = int(progress.get("total_tokens", 0))
                label = "prefill"
            self.wfile.write(f": {label} {done}/{total}\n\n".encode())
            self.wfile.flush()

        result = (
            generate_fn(on_token, on_progress)
            if generate_fn is not None else
            engine.generate(
                prompt, max_tokens, on_token=on_token, stop=all_stop,
                on_progress=on_progress, sampling=self._sampling,
                constraint=self._constraint)
        )
        _log_path_stats(result, result.get("prompt_tokens", 0))
        blocks, stop_reason = build_content(
            result["text"], result.get("termination_reason", "eos"))
        text_content = "".join(
            block["text"] for block in blocks if block["type"] == "text")
        if holdback is not None:
            remainder = holdback.final_remainder(text_content)
            if remainder:
                emit("content_block_delta", {
                    "index": 0, "delta": {"type": "text_delta", "text": remainder}})
        emit("content_block_stop", {"index": 0})

        idx = 1
        for block in blocks:
            if block["type"] == "tool_use":
                emit("content_block_start", {"index": idx, "content_block": {
                    "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}})
                emit("content_block_delta", {"index": idx, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
                emit("content_block_stop", {"index": idx})
                idx += 1

        emit("message_delta", {"delta": {
                                   "stop_reason": stop_reason,
                                   "stop_sequence": (result.get("stop_sequence")
                                                     if stop_reason == "stop_sequence"
                                                     else None)},
                               "usage": {"output_tokens": len(result["tokens"])},
                               **({"vmodel_timing": {
                                       "vision_seconds": round(float(
                                           result.get("vision_s", 0.0)), 4),
                                       **_vision_protocol_timing(result),
                                   }} if "vision_s" in result else {})})
        emit("message_stop", {})

    def log_message(self, fmt, *args):
        elapsed = f" ({time.time() - self._t0:.2f}s)" if hasattr(self, "_t0") else ""
        print(f"[server] -> {args[0] if args else ''}{elapsed}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()
    print(f"[server] vOOM endpoint on http://127.0.0.1:{args.port}  models: {list(_registry())}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        with INFER_LOCK:
            MANAGER.close()


if __name__ == "__main__":
    main()
