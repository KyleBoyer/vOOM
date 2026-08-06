"""Optional fully-resident MLX-LM backend for safely fitting text models.

vOOM remains the out-of-core engine.  This module is imported only after a
pure admission policy proves that a locally-derived, fully-quantized dense
Qwen checkpoint fits below both the Metal ceiling and current system headroom.
Missing/incompatible MLX-LM therefore never makes the core runtime unusable.
"""

from __future__ import annotations

import copy
import gc
import importlib
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import mlx.core as mx
import psutil
from tokenizers import Tokenizer

from .incremental_decode import IncrementalDetokenizer
from .sampler import SamplingParams, sample


_METAL_HARD_CEILING_BYTES = 8_500_000_000
# F201: lfm2 (LFM2.5-2.6B) joins the allowlist. Its hot path is the generic
# one: the two Qwen-specific hooks (_qwen35_layerwise_forward and
# _qwen35_lossy_suffix_prefill) are reached only when the opt-in lossy suffix
# prefill is configured, and that mode already fails closed on any non-Qwen
# checkpoint. Native MTP is likewise gated on model_type. Admission still
# requires the locally-derived all-MXFP4 artifact and the memory proof below.
_SUPPORTED_AUTO_MODEL_TYPES = frozenset({"qwen3_5", "lfm2"})


def _parse_lossy_suffix_prefill(raw: str | None) -> tuple[int, int] | None:
    """Parse the explicit depth-adaptive prefill schedule.

    The schedule is deliberately request-independent: ``D:S`` runs the full
    prompt through the first D layers, retains the most recent S hidden
    positions, then runs that suffix through the remaining layers.  It is a
    lossy side-quest mode and never activates from ``auto``.
    """
    value = str(raw or "").strip().lower()
    if value in ("", "0", "off"):
        return None
    fields = value.split(":")
    if len(fields) != 2:
        raise ValueError(
            "VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL must be off or "
            "EARLY_LAYERS:SUFFIX_TOKENS")
    try:
        early_layers, suffix_tokens = map(int, fields)
    except ValueError as error:
        raise ValueError(
            "VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL fields must be integers"
        ) from error
    if early_layers <= 0 or suffix_tokens <= 0:
        raise ValueError(
            "VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL fields must be positive")
    return early_layers, suffix_tokens


@dataclass(frozen=True)
class ResidentBackendDecision:
    backend: str
    requested: str
    reason: str
    payload_bytes: int = 0
    estimated_metal_bytes: int = 0
    available_bytes: int = 0
    metal_ceiling_bytes: int = 0

    @property
    def admitted(self) -> bool:
        return self.backend == "mlx-lm"


def _positive_env_mb(
    env: Mapping[str, str], name: str, default: int, *, maximum: int | None = None,
) -> int:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer number of MB") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _checkpoint_payload_bytes(model_dir: Path) -> int:
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


def choose_resident_backend(
    model_dir: str | Path,
    cfg,
    mode: str,
    *,
    requires_vision: bool = False,
    execution_profile: str = "",
    available_bytes: int | None = None,
    env: Mapping[str, str] | None = None,
) -> ResidentBackendDecision:
    """Choose MLX-LM only inside the measured, memory-safe envelope.

    ``auto`` is deliberately narrower than MLX-LM's full architecture list:
    the measured 4B equivalence/speed gate used dense Qwen3.5 with every model
    tensor prequantized to MLX MXFP4.  New architectures expand this allowlist
    only after their own same-artifact greedy gate.
    """
    env = os.environ if env is None else env
    requested = str(env.get("VMODEL_RESIDENT_BACKEND", "auto")).strip().lower()
    if requested not in ("auto", "voom", "mlx-lm"):
        raise ValueError(
            "VMODEL_RESIDENT_BACKEND must be auto, voom, or mlx-lm")
    if requested == "voom":
        return ResidentBackendDecision("voom", requested, "operator_forced_voom")
    if requires_vision:
        return ResidentBackendDecision(
            "voom", requested, "vision_requires_voom")
    if execution_profile:
        return ResidentBackendDecision(
            "voom", requested, "layer_profile_requires_voom")
    if mode not in ("fast", "fast-long"):
        return ResidentBackendDecision(
            "voom", requested, "auto_route_requires_fast_mode")
    if getattr(cfg, "model_type", "") not in _SUPPORTED_AUTO_MODEL_TYPES:
        return ResidentBackendDecision(
            "voom", requested, "architecture_not_measured")
    if int(getattr(cfg, "num_experts", 0) or 0):
        return ResidentBackendDecision(
            "voom", requested, "moe_checkpoint_is_out_of_core")

    model_dir = Path(model_dir)
    try:
        raw_config = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return ResidentBackendDecision(
            "voom", requested, "missing_or_invalid_config")
    provenance = raw_config.get("voom_quantization")
    quantization = (
        raw_config.get("quantization_config")
        or raw_config.get("quantization")
        or {})
    if not (
        isinstance(provenance, dict)
        and provenance.get("profile") == "all"
        and isinstance(quantization, dict)
        and int(quantization.get("bits", 0) or 0) == 4
        and str(quantization.get("mode", "")).lower() == "mxfp4"
    ):
        return ResidentBackendDecision(
            "voom", requested, "checkpoint_not_measured_all_mxfp4_profile")

    payload_bytes = _checkpoint_payload_bytes(model_dir)
    if payload_bytes <= 0:
        return ResidentBackendDecision(
            "voom", requested, "no_raw_safetensor_payload")

    max_model_mb = _positive_env_mb(
        env, "VMODEL_MLX_LM_MAX_MODEL_MB", 7_500)
    transient_mb = _positive_env_mb(
        env, "VMODEL_MLX_LM_TRANSIENT_RESERVE_MB", 1_200)
    system_reserve_mb = _positive_env_mb(
        env, "VMODEL_MLX_LM_SYSTEM_RESERVE_MB", 2_000)
    metal_ceiling_mb = _positive_env_mb(
        env, "VMODEL_MLX_LM_METAL_CEILING_MB", 8_300, maximum=8_500)
    metal_ceiling_bytes = min(
        metal_ceiling_mb * 1_000_000, _METAL_HARD_CEILING_BYTES)
    estimated_metal_bytes = payload_bytes + transient_mb * 1_000_000
    available_bytes = int(
        psutil.virtual_memory().available
        if available_bytes is None else available_bytes)

    base = dict(
        payload_bytes=payload_bytes,
        estimated_metal_bytes=estimated_metal_bytes,
        available_bytes=available_bytes,
        metal_ceiling_bytes=metal_ceiling_bytes,
    )
    if payload_bytes > max_model_mb * 1_000_000:
        return ResidentBackendDecision(
            "voom", requested, "checkpoint_exceeds_resident_model_limit", **base)
    if estimated_metal_bytes > metal_ceiling_bytes:
        return ResidentBackendDecision(
            "voom", requested, "checkpoint_plus_transient_exceeds_metal_ceiling",
            **base)
    if payload_bytes + system_reserve_mb * 1_000_000 > available_bytes:
        return ResidentBackendDecision(
            "voom", requested, "insufficient_current_system_headroom", **base)
    return ResidentBackendDecision(
        "mlx-lm", requested, "measured_resident_qwen_profile_admitted", **base)


def _apply_transformers_compat() -> None:
    """Patch the installed Transformers string-registration regression.

    mlx-lm 0.31.3 registers ``"NewlineTokenizer"`` as a key.  The installed
    Transformers implementation unconditionally reads ``key.__module__`` even
    after recognizing that the key has no ``__name__``.  Delegate every normal
    class key to the original implementation and handle only string keys with
    the same `_extra_content` update the original method intended.
    """
    from transformers.models.auto.auto_factory import _LazyAutoMapping

    current = _LazyAutoMapping.register
    if getattr(current, "_vmodel_string_key_compat", False):
        return
    original = current

    def register(self, key, value, exist_ok=False):
        if isinstance(key, str):
            self._extra_content[key] = value
            return
        return original(self, key, value, exist_ok=exist_ok)

    register._vmodel_string_key_compat = True
    register._vmodel_original = original
    _LazyAutoMapping.register = register


def import_mlx_lm():
    """Import the optional backend after applying the pinned narrow shim."""
    _apply_transformers_compat()
    return importlib.import_module("mlx_lm")


def _walk_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_arrays(item)


def _prompt_cache_nbytes(prompt_cache) -> int:
    total = 0
    for entry in prompt_cache or ():
        # MLX-LM cache classes own their accounting.  Prefer it over walking
        # ``state`` views: KVCache.state may manufacture short-lived array
        # views whose Python ids are reused, which made an id-deduplicating
        # walker materially undercount long Qwen3.5 attention caches.
        try:
            total += int(entry.nbytes)
            continue
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            pass
        total += sum(
            int(array.nbytes)
            for array in _walk_arrays(getattr(entry, "state", ())))
    return total


def _unique_retained_array_bytes(*values) -> int:
    """Count unique arrays owned by retained prompt/generation artifacts."""
    seen: set[int] = set()
    total = 0

    def add(array):
        nonlocal total
        if isinstance(array, mx.array) and id(array) not in seen:
            seen.add(id(array))
            total += int(array.nbytes)

    def visit(value):
        if isinstance(value, mx.array):
            add(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            cache_values = getattr(value, "cache", None)
            if isinstance(cache_values, list):
                visit(cache_values)
            child_caches = getattr(value, "caches", None)
            if isinstance(child_caches, (tuple, list)):
                visit(child_caches)
            add(getattr(value, "keys", None))
            add(getattr(value, "values", None))

    for value in values:
        visit(value)
    return total


def _qwen35_request_incremental_bytes(
    cfg, positions: int, *, transient_bytes: int = 1_200_000_000,
) -> int:
    """Conservative incremental resident bytes for one Qwen3.5 request."""
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    interval = int(getattr(cfg, "full_attention_interval", 0) or 0)
    kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", 0) or 0)
    if min(layers, interval, kv_heads, head_dim) <= 0:
        return transient_bytes
    full_attention_layers = (layers + interval - 1) // interval
    kv_bytes_per_token = (
        full_attention_layers * 2 * kv_heads * head_dim * 2)
    # DeltaNet recurrent+conv state is fixed rather than token-growing.  The
    # real 9B cache was about 59 MB above the exact attention projection;
    # reserve 64 MB for that state plus allocator rounding.
    recurrent_state_reserve = 64_000_000
    return (
        max(0, int(positions)) * kv_bytes_per_token
        + recurrent_state_reserve + transient_bytes)


def _exact_extension_prefix(
    cached_token_ids: list[int] | tuple[int, ...] | None,
    prompt_ids: list[int] | tuple[int, ...],
) -> int:
    """Return a reusable cache length only for a strict token extension."""
    cached = tuple(cached_token_ids or ())
    current = tuple(prompt_ids)
    if cached and len(current) > len(cached) and current[:len(cached)] == cached:
        return len(cached)
    return 0


def _exact_prompt_cache_match(
    cached_token_ids: list[int] | tuple[int, ...] | None,
    prompt_ids: list[int] | tuple[int, ...],
) -> tuple[int, str]:
    """Match only an identical prompt endpoint or its strict extension.

    Qwen3.5's DeltaNet state is a recurrent fold, so an arbitrary LCP cannot
    be rewound safely.  Exact endpoint reuse and forward-only extension are
    both released-model-correct and independent of prompt contents.
    """
    cached = tuple(cached_token_ids or ())
    current = tuple(prompt_ids)
    if not cached:
        return 0, "miss"
    if current == cached:
        return len(cached), "exact"
    prefix = _exact_extension_prefix(cached, current)
    if prefix:
        return prefix, "extension"
    return 0, "miss"


def _fork_prompt_cache(prompt_cache):
    """Fork MLX-LM cache wrappers without copying evaluated array payloads.

    Qwen3.5 ``ArraysCache`` updates replace recurrent arrays, while
    ``KVCache`` appends only after its numeric offset.  A shallow wrapper fork
    therefore gives copy-on-write recurrent state and a stable attention
    prefix: decode may write spare KV capacity after the saved endpoint, but
    the retained fork never exposes positions beyond its own offset.
    """
    forked = []
    for entry in prompt_cache or ():
        clone = copy.copy(entry)
        cache_list = getattr(entry, "cache", None)
        if isinstance(cache_list, list):
            clone.cache = list(cache_list)
        child_caches = getattr(entry, "caches", None)
        if isinstance(child_caches, (tuple, list)):
            children = _fork_prompt_cache(child_caches)
            clone.caches = (
                tuple(children)
                if isinstance(child_caches, tuple) else children)
        forked.append(clone)
    return forked


class ResidentMLXLMEngine:
    """Server-compatible text engine backed by a fully resident MLX-LM model."""

    backend_name = "mlx-lm"

    def __init__(
        self, model_dir: str | Path, cfg, rc,
        decision: ResidentBackendDecision,
    ):
        if not decision.admitted:
            raise ValueError("ResidentMLXLMEngine requires an admitted decision")
        self._model_dir = Path(model_dir)
        self.cfg = cfg
        self.rc = rc
        self.effective_max_position_embeddings = int(
            getattr(cfg, "max_position_embeddings", 0) or 0)
        self.rope_profile = "released"
        self._resident_backend_decision = decision
        self._xgrammar_compiler = None
        self._prepared_prompt_token_cache = None
        self.last_kv = None
        self.last_mtp_kv = None
        self._last_cache_token_ids: tuple[int, ...] = ()
        self._last_prompt_logits = None
        self._last_prompt_hidden = None
        self._last_generation_tokens: tuple[int, ...] = ()
        self._last_generation_step_logits: tuple = ()
        self._last_generation_checkpoints: tuple = ()
        self._generation_sampled_tokens = 0
        self._native_mtp = None
        self._persistent_prompt_store = None
        self._lossy_suffix_prefill = _parse_lossy_suffix_prefill(
            os.environ.get("VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL"))

        tokenizer_path = self._model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"MLX-LM resident backend requires tokenizer.json: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        raw = json.loads((self._model_dir / "config.json").read_text())
        quantization = (
            raw.get("quantization_config") or raw.get("quantization") or {})
        self.store = SimpleNamespace(
            quantization=quantization,
            on_disk_quantized=bool(quantization),
            integrity_mode="raw-safetensors",
        )

        before = mx.get_active_memory()
        started = time.perf_counter()
        mlx_lm = import_mlx_lm()
        self.model, _ = mlx_lm.load(
            str(self._model_dir), lazy=False)
        if self._lossy_suffix_prefill is not None:
            early_layers, _suffix_tokens = self._lossy_suffix_prefill
            layers = getattr(
                getattr(
                    getattr(self.model, "language_model", None),
                    "model", None),
                "layers", ())
            if (getattr(cfg, "model_type", "") != "qwen3_5"
                    or not layers):
                raise ValueError(
                    "VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL currently requires "
                    "a dense Qwen3.5 checkpoint")
            if early_layers >= len(layers):
                raise ValueError(
                    "VMODEL_MLX_LM_LOSSY_SUFFIX_PREFILL EARLY_LAYERS must "
                    f"be less than the model depth ({len(layers)})")
        native_mtp_setting = os.environ.get(
            "VMODEL_MLX_LM_NATIVE_MTP", "0")
        if native_mtp_setting not in ("0", "1"):
            raise ValueError(
                "VMODEL_MLX_LM_NATIVE_MTP must be 0 or 1")
        if native_mtp_setting == "1":
            from .resident_qwen_mtp import ResidentQwenMTP

            self._native_mtp = ResidentQwenMTP(
                self.model, self._model_dir, quantization)
        persistent_prompt_dir = os.environ.get(
            "VMODEL_MLX_LM_PERSISTENT_PROMPT_CACHE_DIR", "").strip()
        if persistent_prompt_dir:
            from .kv_store import model_fingerprint
            from .resident_prompt_store import ResidentPromptStore

            persistent_max_mb = _positive_env_mb(
                os.environ,
                "VMODEL_MLX_LM_PERSISTENT_PROMPT_CACHE_MAX_MB",
                4_000,
            )
            version = {}
            for distribution in ("mlx", "mlx-lm"):
                try:
                    version[distribution] = importlib.metadata.version(
                        distribution)
                except importlib.metadata.PackageNotFoundError:
                    version[distribution] = "unknown"
            arithmetic = json.dumps({
                "backend": "resident-mlx-lm",
                "native_mtp_loaded": native_mtp_setting,
                "lossy_suffix_prefill": self._lossy_suffix_prefill,
                "persistent_prompt_format": "v2",
                "versions": version,
            }, sort_keys=True, separators=(",", ":"))
            fingerprint = model_fingerprint(
                self._model_dir,
                compressed_mla=False,
                quant=json.dumps(
                    quantization, sort_keys=True, separators=(",", ":")),
                arithmetic=arithmetic,
            )
            self._persistent_prompt_store = ResidentPromptStore(
                persistent_prompt_dir,
                fingerprint,
                max_bytes=persistent_max_mb * 1_000_000,
            )
        self._load_s = time.perf_counter() - started
        self._model_active_bytes = max(0, mx.get_active_memory() - before)
        if mx.get_active_memory() > decision.metal_ceiling_bytes:
            active = mx.get_active_memory()
            self.close()
            raise MemoryError(
                "MLX-LM load crossed resident Metal ceiling: "
                f"{active / 1e9:.2f}GB > "
                f"{decision.metal_ceiling_bytes / 1e9:.2f}GB")

    def close(self) -> None:
        native_mtp = getattr(self, "_native_mtp", None)
        if native_mtp is not None:
            self._native_mtp = None
            native_mtp.close()
        self._persistent_prompt_store = None
        model = getattr(self, "model", None)
        if model is not None:
            self.model = None
            del model
        self.last_kv = None
        self.last_mtp_kv = None
        self._last_cache_token_ids = ()
        self._last_prompt_logits = None
        self._last_prompt_hidden = None
        self._last_generation_tokens = ()
        self._last_generation_step_logits = ()
        self._last_generation_checkpoints = ()
        gc.collect()
        mx.clear_cache()

    def discard_failed_request_state(self) -> None:
        self.last_kv = None
        self.last_mtp_kv = None
        self._last_cache_token_ids = ()
        self._last_prompt_logits = None
        self._last_prompt_hidden = None
        self._last_generation_tokens = ()
        self._last_generation_step_logits = ()
        self._last_generation_checkpoints = ()
        mx.clear_cache()

    def report(self) -> str:
        decision = self._resident_backend_decision
        return (
            "backend=mlx-lm "
            f"load={self._load_s:.3f}s "
            f"payload={decision.payload_bytes / 1e9:.2f}GB "
            f"model_active={self._model_active_bytes / 1e9:.2f}GB "
            f"estimated_peak={decision.estimated_metal_bytes / 1e9:.2f}GB")

    def prompt_cache_memory_snapshot(self) -> dict[str, int]:
        """Expose live resident state to the server's pre-generation gate.

        A previous request's MLX-LM prompt cache is disposable.  Reporting it
        separately lets the shared admission arithmetic subtract that cache
        before projecting the next prompt instead of pessimistically counting
        both contexts at once.
        """
        retained = _unique_retained_array_bytes(
            self.last_kv,
            self.last_mtp_kv,
            self._last_prompt_logits,
            self._last_prompt_hidden,
            self._last_generation_step_logits,
            [cache for _position, cache
             in self._last_generation_checkpoints],
        )
        active = max(0, int(mx.get_active_memory()))
        return {
            "active_metal_bytes": active,
            "retained_prompt_kv_bytes": retained,
            "orphan_prompt_kv_bytes": 0,
            "evictable_prompt_kv_bytes": retained,
            "metal_ceiling_bytes": int(
                self._resident_backend_decision.metal_ceiling_bytes),
        }

    def _make_prompt_cache(self):
        cache_module = importlib.import_module("mlx_lm.models.cache")
        return cache_module.make_prompt_cache(self.model)

    def _qwen35_layerwise_forward(self, inputs, cache):
        """Forward Qwen3.5 with a mask derived from each layer's own cache.

        Released MLX-LM shares masks created from layer 0 and layer 3 because
        ordinary forwards keep all hybrid-cache lengths equal.  Depth-adaptive
        prefill intentionally gives early and late layers different history
        lengths, so decode must derive the otherwise-identical mask locally.
        """
        base = importlib.import_module("mlx_lm.models.base")
        text_model = self.model.language_model
        core = text_model.model
        hidden = core.embed_tokens(inputs)
        for layer, layer_cache in zip(core.layers, cache):
            mask = (
                base.create_ssm_mask(hidden, layer_cache)
                if layer.is_linear
                else base.create_attention_mask(hidden, layer_cache))
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        hidden = core.norm(hidden)
        if text_model.args.tie_word_embeddings:
            return core.embed_tokens.as_linear(hidden)
        return text_model.lm_head(hidden)

    def _qwen35_lossy_suffix_prefill(
        self, prompt, prompt_cache, early_layers: int, suffix_tokens: int,
        prefill_step_size: int, progress,
    ):
        """Run request-independent shallow-full/deep-suffix prompt encoding."""
        base = importlib.import_module("mlx_lm.models.base")
        text_model = self.model.language_model
        core = text_model.model
        total = int(prompt.shape[0])
        processed = 0
        boundary = None

        while processed < total:
            count = min(prefill_step_size, total - processed)
            hidden = core.embed_tokens(
                prompt[processed:processed + count][None])
            for layer_index in range(early_layers):
                layer = core.layers[layer_index]
                layer_cache = prompt_cache[layer_index]
                mask = (
                    base.create_ssm_mask(hidden, layer_cache)
                    if layer.is_linear
                    else base.create_attention_mask(hidden, layer_cache))
                hidden = layer(hidden, mask=mask, cache=layer_cache)
            tail = hidden[:, -min(suffix_tokens, count):, :]
            boundary = (
                tail if boundary is None
                else mx.concatenate([boundary, tail], axis=1)[
                    :, -suffix_tokens:, :])
            mx.eval(
                boundary,
                [entry.state for entry in prompt_cache[:early_layers]])
            processed += count
            progress(processed, total)
            mx.clear_cache()

        hidden = boundary
        for layer_index in range(early_layers, len(core.layers)):
            layer = core.layers[layer_index]
            layer_cache = prompt_cache[layer_index]
            mask = (
                base.create_ssm_mask(hidden, layer_cache)
                if layer.is_linear
                else base.create_attention_mask(hidden, layer_cache))
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        hidden = core.norm(hidden[:, -1:, :])
        logits = (
            core.embed_tokens.as_linear(hidden)
            if text_model.args.tie_word_embeddings
            else text_model.lm_head(hidden))
        raw_values = logits[:, -1, :].reshape(-1)
        mx.eval(
            raw_values,
            [entry.state for entry in prompt_cache])
        return raw_values

    def _direct_tokens(
        self, prompt_ids: list[int], max_tokens: int, sampling: SamplingParams,
        constraint, generated: list[int], prompt_cache, progress,
        prefill_step_size: int, prefix_len: int = 0,
        exact_prompt_logits=None, exact_prompt_hidden=None,
        prompt_mtp_cache=None, native_mtp=None, mtp_stats=None,
        cached_generation_tokens=(), cached_generation_step_logits=(),
        cached_generation_checkpoints=(),
        generation_step_logits=None, generation_logit_limit: int = 0,
        logit_chain_stats=None,
        on_generation_checkpoint=None,
        on_prompt_endpoint=None,
        lossy_suffix_prefill=None,
        model_forward=None,
    ):
        model_forward = model_forward or self.model
        prompt = mx.array(prompt_ids, dtype=mx.int32)
        total = len(prompt_ids)
        processed = prefix_len
        progress(processed, total)
        use_mtp = (
            native_mtp is not None
            and prompt_mtp_cache is not None
            and constraint is None
            and max_tokens > 1)

        # A cached MTP endpoint for N prompt tokens contains transitions
        # h[0]+tok[1] ... h[N-2]+tok[N-1].  Before forwarding a strict
        # extension, add its first new transition from the retained h[N-1].
        if use_mtp and prefix_len and total > prefix_len:
            if exact_prompt_hidden is None:
                raise RuntimeError(
                    "native MTP extension requires retained prompt hidden state")
            native_mtp.advance(
                exact_prompt_hidden,
                prompt[prefix_len:prefix_len + 1][None],
                prompt_mtp_cache)
            mx.eval([entry.state for entry in prompt_mtp_cache])

        if exact_prompt_logits is not None:
            raw_values = exact_prompt_logits
            endpoint_hidden = exact_prompt_hidden
        elif lossy_suffix_prefill is not None:
            early_layers, suffix_tokens = lossy_suffix_prefill
            raw_values = self._qwen35_lossy_suffix_prefill(
                prompt, prompt_cache, early_layers, suffix_tokens,
                prefill_step_size, progress)
            endpoint_hidden = None
            if on_prompt_endpoint is not None:
                on_prompt_endpoint(
                    prompt_cache, raw_values, None, None)
        else:
            while total - processed > 1:
                count = min(prefill_step_size, (total - processed) - 1)
                chunk = prompt[processed:processed + count][None]
                if use_mtp:
                    _logits, hidden = native_mtp.trunk_forward(
                        chunk, prompt_cache)
                    native_mtp.advance(
                        hidden,
                        prompt[
                            processed + 1:processed + count + 1][None],
                        prompt_mtp_cache)
                    mx.eval(
                        [entry.state for entry in prompt_cache],
                        [entry.state for entry in prompt_mtp_cache])
                else:
                    model_forward(chunk, cache=prompt_cache)
                    mx.eval([entry.state for entry in prompt_cache])
                processed += count
                progress(processed, total)
                mx.clear_cache()
            current = prompt[processed:]
            if use_mtp:
                logits, endpoint_hidden = native_mtp.trunk_forward(
                    current[None], prompt_cache)
            else:
                logits = model_forward(current[None], cache=prompt_cache)
                endpoint_hidden = None
            raw_values = logits[:, -1, :].reshape(-1)
            # Materialize the raw endpoint distribution, exact trunk hidden,
            # and all cache state before retaining shallow array snapshots.
            evaluation = [
                raw_values,
                [entry.state for entry in prompt_cache],
            ]
            if endpoint_hidden is not None:
                evaluation.append(endpoint_hidden)
            mx.eval(*evaluation)
            if on_prompt_endpoint is not None:
                on_prompt_endpoint(
                    prompt_cache, raw_values, endpoint_hidden,
                    prompt_mtp_cache if use_mtp else None)

        progress(total, total)
        values = raw_values
        if constraint is not None:
            values = constraint.mask_logits(values)
        if (generation_step_logits is not None
                and len(generation_step_logits) < generation_logit_limit):
            generation_step_logits.append(raw_values)
        token = sample(
            values, sampling, history=[*prompt_ids, *generated])
        yield token

        if not use_mtp:
            chain_stats = (
                logit_chain_stats if logit_chain_stats is not None else {})
            chain_limit = min(
                len(cached_generation_tokens),
                len(cached_generation_step_logits))
            chain_position = 1
            matched_prefix = 0
            while chain_position < chain_limit:
                if (len(generated) < chain_position
                        or int(generated[chain_position - 1])
                        != int(cached_generation_tokens[
                            chain_position - 1])):
                    chain_stats["diverged_at"] = chain_position - 1
                    matched_prefix = chain_position - 1
                    break
                matched_prefix = chain_position
                raw_values = cached_generation_step_logits[chain_position]
                values = raw_values
                if constraint is not None:
                    values = constraint.mask_logits(values)
                if (generation_step_logits is not None
                        and len(generation_step_logits)
                        < generation_logit_limit):
                    generation_step_logits.append(raw_values)
                token = sample(
                    values, sampling, history=[*prompt_ids, *generated])
                chain_stats["reused_step_logits"] = (
                    int(chain_stats.get("reused_step_logits", 0)) + 1)
                yield token
                chain_position += 1
            else:
                matched_prefix = min(
                    len(generated), len(cached_generation_tokens))

            # The target cache is still at the prompt endpoint while cached
            # logits are sampled.  If the branch diverges or exhausts the
            # retained chain, restore the closest exact recurrent checkpoint
            # and catch up only its unmatched tail one position at a time.
            # Sequential catch-up preserves the established MXFP4 GEMV
            # numerics; a multi-position refeed is not byte-identical.
            catchup_start = 0
            eligible_checkpoints = [
                (int(position), cache)
                for position, cache in cached_generation_checkpoints
                if int(position) <= matched_prefix
            ]
            if eligible_checkpoints:
                catchup_start, checkpoint = max(
                    eligible_checkpoints, key=lambda item: item[0])
                prompt_cache[:] = _fork_prompt_cache(checkpoint)
                chain_stats["checkpoint_restored_tokens"] = catchup_start
            raw_values = None
            for position, catchup in enumerate(
                    generated[catchup_start:], start=catchup_start + 1):
                logits = model_forward(
                    mx.array([[int(catchup)]], dtype=mx.int32),
                    cache=prompt_cache)
                raw_values = logits[:, -1, :].reshape(-1)
                chain_stats["catchup_sweeps"] = (
                    int(chain_stats.get("catchup_sweeps", 0)) + 1)
                if on_generation_checkpoint is not None:
                    on_generation_checkpoint(position, prompt_cache)
            if raw_values is None:
                raise RuntimeError("logit-chain catch-up had no generated token")
            values = raw_values
            if constraint is not None:
                values = constraint.mask_logits(values)
            if (generation_step_logits is not None
                    and len(generation_step_logits) < generation_logit_limit):
                generation_step_logits.append(raw_values)
            token = sample(
                values, sampling, history=[*prompt_ids, *generated])
            yield token
            current = mx.array([token], dtype=mx.int32)
            while True:
                logits = model_forward(
                    current[None], cache=prompt_cache)[:, -1, :]
                raw_values = logits.reshape(-1)
                if on_generation_checkpoint is not None:
                    on_generation_checkpoint(len(generated), prompt_cache)
                values = raw_values
                if constraint is not None:
                    values = constraint.mask_logits(values)
                if (generation_step_logits is not None
                        and len(generation_step_logits)
                        < generation_logit_limit):
                    generation_step_logits.append(raw_values)
                token = sample(
                    values, sampling, history=[*prompt_ids, *generated])
                yield token
                current = mx.array([token], dtype=mx.int32)

        if endpoint_hidden is None:
            raise RuntimeError(
                "native MTP requires the exact pre-norm prompt hidden state")
        stats = mtp_stats if mtp_stats is not None else {}
        stats.update({
            "enabled": 1,
            "used": 0,
            "proposed": 0,
            "accepted": 0,
            "target_sweeps": 0,
            "rejection_refeeds": 0,
            "draft_head_calls": 0,
            "sampling_proof": (
                "greedy-target-match"
                if sampling.is_greedy
                else "independent-target-draw"),
        })

        def propose(hidden, next_tokens):
            stats["draft_head_calls"] += 1
            ids = mx.array([next_tokens], dtype=mx.int32)
            logits = native_mtp.draft_logits(
                hidden, ids, prompt_mtp_cache)[:, -1, :].reshape(-1)
            return sample(
                logits, sampling, history=[*prompt_ids, *generated])

        catchup = int(token)
        draft = propose(endpoint_hidden, [catchup])
        while True:
            stats["used"] = 1
            stats["proposed"] += 1
            checkpoint = _fork_prompt_cache(prompt_cache)
            verification = mx.array(
                [[catchup, int(draft)]], dtype=mx.int32)
            logits, hidden = native_mtp.trunk_forward(
                verification, prompt_cache, confirmed_prefix=1)
            stats["target_sweeps"] += 1
            target = sample(
                logits[:, 0, :].reshape(-1), sampling,
                history=[*prompt_ids, *generated])

            if int(target) == int(draft):
                stats["accepted"] += 1
                bonus = sample(
                    logits[:, 1, :].reshape(-1), sampling,
                    history=[*prompt_ids, *generated, int(draft)])
                yield int(draft)
                yield int(bonus)
                # Commit both released MTP transitions in one two-position
                # head call and use its final distribution as the next draft.
                draft = propose(
                    hidden, [int(draft), int(bonus)])
                catchup = int(bonus)
            else:
                # ArraysCache updates replace arrays and KVCache appends beyond
                # its numeric offset, so a shallow wrapper fork is an exact
                # copy-on-write checkpoint for this hybrid trunk.
                prompt_cache[:] = checkpoint
                _confirmed_logits, confirmed_hidden = (
                    native_mtp.trunk_forward(
                        mx.array([[catchup]], dtype=mx.int32),
                        prompt_cache))
                mx.eval(
                    confirmed_hidden,
                    [entry.state for entry in prompt_cache])
                stats["target_sweeps"] += 1
                stats["rejection_refeeds"] += 1
                yield int(target)
                draft = propose(confirmed_hidden, [int(target)])
                catchup = int(target)

    @staticmethod
    def _stop_match(text: str, stop: list[str]):
        matches = [
            (text.find(value), index, value)
            for index, value in enumerate(stop)
            if value and text.find(value) >= 0
        ]
        return min(matches) if matches else None

    def generate(
        self, prompt: str, max_tokens: int = 64, on_token=None, stop=None,
        on_progress=None, sampling: SamplingParams | None = None,
        constraint=None,
    ) -> dict:
        request_started = time.perf_counter()
        sampling = sampling or SamplingParams()
        sampling.seed_rng()
        stop = list(stop or ())
        prepared = getattr(prompt, "token_ids", None)
        prompt_ids = (
            list(prepared) if prepared is not None
            else self.tokenizer.encode(str(prompt)).ids)
        if not prompt_ids:
            raise ValueError("prompt must encode to at least one token")
        if (self.effective_max_position_embeddings
                and len(prompt_ids) + max_tokens
                > self.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(prompt_ids)})+max_tokens({max_tokens}) exceeds "
                f"active context limit={self.effective_max_position_embeddings}")

        default_prefill_step = (
            512 if self._resident_backend_decision.payload_bytes
            >= 6_000_000_000 else 2048)
        try:
            prefill_step_size = int(os.environ.get(
                "VMODEL_MLX_LM_PREFILL_STEP_SIZE",
                str(default_prefill_step)))
        except ValueError as error:
            raise ValueError(
                "VMODEL_MLX_LM_PREFILL_STEP_SIZE must be an integer") from error
        if prefill_step_size <= 0:
            raise ValueError(
                "VMODEL_MLX_LM_PREFILL_STEP_SIZE must be positive")
        cache_setting = os.environ.get(
            "VMODEL_MLX_LM_PROMPT_CACHE", "1")
        if cache_setting not in ("0", "1"):
            raise ValueError("VMODEL_MLX_LM_PROMPT_CACHE must be 0 or 1")
        logit_chain_setting = os.environ.get(
            "VMODEL_MLX_LM_LOGIT_CHAIN", "1")
        if logit_chain_setting not in ("0", "1"):
            raise ValueError("VMODEL_MLX_LM_LOGIT_CHAIN must be 0 or 1")
        try:
            logit_chain_limit = int(os.environ.get(
                "VMODEL_MLX_LM_LOGIT_CHAIN_MAX_TOKENS", "128"))
        except ValueError as error:
            raise ValueError(
                "VMODEL_MLX_LM_LOGIT_CHAIN_MAX_TOKENS must be an integer"
            ) from error
        if not 1 <= logit_chain_limit <= 512:
            raise ValueError(
                "VMODEL_MLX_LM_LOGIT_CHAIN_MAX_TOKENS must be 1..512")
        try:
            logit_checkpoint_stride = int(os.environ.get(
                "VMODEL_MLX_LM_LOGIT_CHECKPOINT_STRIDE", "4"))
            logit_checkpoint_limit = int(os.environ.get(
                "VMODEL_MLX_LM_LOGIT_CHECKPOINT_MAX", "8"))
        except ValueError as error:
            raise ValueError(
                "resident logit checkpoint controls must be integers"
            ) from error
        if not 1 <= logit_checkpoint_stride <= 128:
            raise ValueError(
                "VMODEL_MLX_LM_LOGIT_CHECKPOINT_STRIDE must be 1..128")
        if not 0 <= logit_checkpoint_limit <= 32:
            raise ValueError(
                "VMODEL_MLX_LM_LOGIT_CHECKPOINT_MAX must be 0..32")
        mtp_decode_default = "1" if self._native_mtp is not None else "0"
        mtp_decode_setting = os.environ.get(
            "VMODEL_MLX_LM_NATIVE_MTP_DECODE", mtp_decode_default)
        if mtp_decode_setting not in ("0", "1"):
            raise ValueError(
                "VMODEL_MLX_LM_NATIVE_MTP_DECODE must be 0 or 1")
        native_mtp = (
            self._native_mtp
            if mtp_decode_setting == "1"
            and self._native_mtp is not None
            and self._lossy_suffix_prefill is None
            and constraint is None
            and max_tokens > 1
            else None)
        persistent_stats = {
            "hit": 0,
            "saved": 0,
            "load_s": 0.0,
            "save_s": 0.0,
            "cache_bytes": 0,
            "logits_bytes": 0,
            "generation_logits": 0,
            "error": 0,
        }
        persistent_store = (
            self._persistent_prompt_store
            if cache_setting == "1" and native_mtp is None else None)
        in_memory_prefix, in_memory_match = _exact_prompt_cache_match(
            self._last_cache_token_ids, prompt_ids)
        in_memory_exact = (
            in_memory_match == "exact"
            and in_memory_prefix
            and self.last_kv is not None
            and self._last_prompt_logits is not None)
        if persistent_store is not None and not in_memory_exact:
            try:
                loaded = persistent_store.load(prompt_ids)
            except Exception:
                # Persistence is an optional acceleration layer.  A storage
                # or format failure must fall back to released model math.
                loaded = None
                persistent_stats["error"] = 1
            if loaded is not None:
                (
                    loaded_cache,
                    loaded_logits,
                    loaded_generation_tokens,
                    loaded_generation_logits,
                    loaded_stats,
                ) = loaded
                persistent_stats.update(loaded_stats)
                self.last_kv = loaded_cache
                self.last_mtp_kv = None
                self._last_cache_token_ids = tuple(prompt_ids)
                self._last_prompt_logits = loaded_logits
                self._last_prompt_hidden = None
                self._last_generation_tokens = loaded_generation_tokens
                self._last_generation_step_logits = loaded_generation_logits
                self._last_generation_checkpoints = ()
        prefix_len = 0
        cache_match = "miss"
        matched_cache = None
        matched_mtp_cache = None
        exact_prompt_logits = None
        exact_prompt_hidden = None
        cached_generation_tokens = ()
        cached_generation_step_logits = ()
        cached_generation_checkpoints = ()
        full_candidate_tokens = len(self._last_cache_token_ids)
        full_candidate_lcp = 0
        for old, new in zip(self._last_cache_token_ids, prompt_ids):
            if old != new:
                break
            full_candidate_lcp += 1
        if cache_setting == "1":
            if self.last_kv is not None:
                full_prefix, full_match = _exact_prompt_cache_match(
                    self._last_cache_token_ids, prompt_ids)
                if (native_mtp is not None and full_prefix
                        and (self.last_mtp_kv is None
                             or self._last_prompt_hidden is None)):
                    full_prefix, full_match = 0, "miss"
                if (self._lossy_suffix_prefill is not None
                        and full_match == "extension"):
                    # Different-depth caches cannot yet advance a strict
                    # extension without recomputing the retained boundary.
                    full_prefix, full_match = 0, "miss"
                if full_match != "exact" or self._last_prompt_logits is not None:
                    prefix_len = full_prefix
                    cache_match = full_match
                    matched_cache = self.last_kv if full_prefix else None
                    matched_mtp_cache = (
                        self.last_mtp_kv if full_prefix else None)
                    exact_prompt_logits = (
                        self._last_prompt_logits
                        if full_match == "exact" else None)
                    exact_prompt_hidden = (
                        self._last_prompt_hidden if full_prefix else None)
                    if (logit_chain_setting == "1"
                            and full_match == "exact"):
                        cached_generation_tokens = (
                            self._last_generation_tokens)
                        cached_generation_step_logits = (
                            self._last_generation_step_logits)
                        cached_generation_checkpoints = (
                            self._last_generation_checkpoints)
        if not prefix_len:
            self.last_kv = None
            self.last_mtp_kv = None
            self._last_cache_token_ids = ()
            self._last_prompt_logits = None
            self._last_prompt_hidden = None
            self._last_generation_tokens = ()
            self._last_generation_step_logits = ()
            self._last_generation_checkpoints = ()
            gc.collect()
            mx.clear_cache()
        request_reserve_mb = _positive_env_mb(
            os.environ, "VMODEL_MLX_LM_REQUEST_SYSTEM_RESERVE_MB", 1_200)
        incremental_bytes = _qwen35_request_incremental_bytes(
            self.cfg, len(prompt_ids) + max_tokens - prefix_len)
        if native_mtp is not None:
            incremental_bytes += (
                native_mtp.cache_bytes_per_token
                * max(0, len(prompt_ids) + max_tokens - prefix_len))
        current_available = int(psutil.virtual_memory().available)
        required_available = (
            incremental_bytes + request_reserve_mb * 1_000_000)
        if current_available < required_available:
            raise MemoryError(
                "MLX-LM resident request lacks live system headroom: "
                f"available={current_available / 1e9:.2f}GB < "
                f"request+reserve={required_available / 1e9:.2f}GB "
                f"(positions={len(prompt_ids) + max_tokens}); use vOOM "
                "or free unified memory")

        def check_metal(phase: str, done: int, total: int):
            live_peak = max(
                int(mx.get_active_memory()), int(mx.get_peak_memory()))
            ceiling = int(
                self._resident_backend_decision.metal_ceiling_bytes)
            if ceiling and live_peak > ceiling:
                raise MemoryError(
                    f"MLX-LM resident {phase} crossed the Metal ceiling: "
                    f"{live_peak / 1e9:.2f}GB > {ceiling / 1e9:.2f}GB "
                    f"at {done}/{total} tokens; reduce "
                    "VMODEL_MLX_LM_PREFILL_STEP_SIZE or use vOOM")

        def progress(done: int, total: int):
            check_metal("prefill", done, total)
            if on_progress is not None:
                on_progress({
                    "phase": "prefill",
                    "completed_tokens": int(done),
                    "total_tokens": int(total),
                    "cache_source": (
                        "persistent-prompt-exact"
                        if persistent_stats["hit"]
                        and cache_match == "exact"
                        else f"hot-prompt-{cache_match}"
                        if prefix_len else "cold"),
                })

        prompt_cache = (
            _fork_prompt_cache(matched_cache)
            if prefix_len else self._make_prompt_cache())
        prompt_mtp_cache = (
            _fork_prompt_cache(matched_mtp_cache)
            if native_mtp is not None and prefix_len
            else native_mtp.make_cache()
            if native_mtp is not None else None)
        retained_prompt_cache = (
            self.last_kv if prefix_len else None)
        retained_prompt_mtp_cache = (
            self.last_mtp_kv if prefix_len else None)
        retained_prompt_logits = (
            self._last_prompt_logits if prefix_len else None)
        retained_prompt_hidden = (
            self._last_prompt_hidden if prefix_len else None)

        def capture_prompt_endpoint(cache, logits, hidden, mtp_cache):
            nonlocal retained_prompt_cache, retained_prompt_mtp_cache
            nonlocal retained_prompt_logits, retained_prompt_hidden
            retained_prompt_cache = _fork_prompt_cache(cache)
            retained_prompt_mtp_cache = (
                _fork_prompt_cache(mtp_cache)
                if mtp_cache is not None else None)
            retained_prompt_logits = logits
            retained_prompt_hidden = hidden

        generated: list[int] = []
        stream_decoder = (
            IncrementalDetokenizer(self.tokenizer, stop)
            if on_token is not None else None)
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()

        # Use one direct loop for cold and cached requests.  Besides keeping
        # sampling/constraint behavior identical across both paths, this gives
        # us the exact raw distribution and cache state at the prompt endpoint
        # before decode advances either one.
        mtp_stats: dict = {}
        logit_chain_stats = {
            "eligible": int(bool(cached_generation_step_logits)),
            "candidate_tokens": len(cached_generation_tokens),
            "candidate_step_logits": len(
                cached_generation_step_logits),
            "reused_step_logits": 0,
            "catchup_sweeps": 0,
            "candidate_checkpoints": len(
                cached_generation_checkpoints),
            "checkpoint_restored_tokens": 0,
        }
        generation_step_logits = (
            [] if logit_chain_setting == "1" else None)
        generation_checkpoints: list = []

        def capture_generation_checkpoint(position, cache):
            if (logit_chain_setting != "1"
                    or native_mtp is not None
                    or logit_checkpoint_limit == 0
                    or position % logit_checkpoint_stride
                    or len(generation_checkpoints)
                    >= logit_checkpoint_limit):
                return
            mx.eval([entry.state for entry in cache])
            generation_checkpoints.append(
                (int(position), _fork_prompt_cache(cache)))

        token_iterator = self._direct_tokens(
            prompt_ids, max_tokens, sampling, constraint, generated,
            prompt_cache, progress, prefill_step_size, prefix_len,
            exact_prompt_logits=exact_prompt_logits,
            exact_prompt_hidden=exact_prompt_hidden,
            prompt_mtp_cache=prompt_mtp_cache,
            native_mtp=native_mtp,
            mtp_stats=mtp_stats,
            cached_generation_tokens=(
                cached_generation_tokens
                if native_mtp is None else ()),
            cached_generation_step_logits=(
                cached_generation_step_logits
                if native_mtp is None else ()),
            cached_generation_checkpoints=(
                cached_generation_checkpoints
                if native_mtp is None else ()),
            generation_step_logits=generation_step_logits,
            generation_logit_limit=logit_chain_limit,
            logit_chain_stats=logit_chain_stats,
            on_generation_checkpoint=capture_generation_checkpoint,
            on_prompt_endpoint=(
                None if cache_match == "exact"
                else capture_prompt_endpoint),
            lossy_suffix_prefill=(
                self._lossy_suffix_prefill
                if cache_match != "exact" else None),
            model_forward=(
                self._qwen35_layerwise_forward
                if self._lossy_suffix_prefill is not None
                else self.model))
        base_execution_path = (
            "mlx_lm_prompt_exact"
            if cache_match == "exact"
            else "mlx_lm_prompt_extension"
            if cache_match == "extension"
            else "mlx_lm_direct")
        execution_path = (
            f"{base_execution_path}_native_mtp"
            if native_mtp is not None
            else f"{base_execution_path}_lossy_suffix_prefill"
            if self._lossy_suffix_prefill is not None
            else base_execution_path)

        first_token_s = 0.0
        prefill_s = 0.0
        decode_intervals: list[float] = []
        last_boundary = request_started
        stop_text = None
        matched_stop_sequence = None
        termination_reason = "length"
        self._generation_sampled_tokens = 0

        try:
            for token in token_iterator:
                now = time.perf_counter()
                interval = now - last_boundary
                last_boundary = now
                if not generated:
                    first_token_s = prefill_s = interval
                else:
                    decode_intervals.append(interval)
                generated.append(int(token))
                self._generation_sampled_tokens = len(generated)
                check_metal(
                    "decode", len(generated), max_tokens)
                if constraint is not None:
                    constraint.accept_token(int(token))

                decoded = self.tokenizer.decode(generated)
                match = self._stop_match(decoded, stop)
                if match is not None:
                    cut, _order, matched_stop_sequence = match
                    stop_text = decoded[:cut]
                    termination_reason = "stop"
                elif int(token) in tuple(self.cfg.eos_token_ids):
                    termination_reason = "eos"
                elif constraint is not None and constraint.completed:
                    termination_reason = "constraint_complete"

                if stop_text is None and stream_decoder is not None:
                    delta = stream_decoder.push(generated)
                    if delta:
                        on_token(delta)
                if (stop_text is not None
                        or termination_reason != "length"
                        or len(generated) >= max_tokens):
                    break
        finally:
            close = getattr(token_iterator, "close", None)
            if callable(close):
                close()

        final_text = (
            stop_text if stop_text is not None
            else self.tokenizer.decode(generated))
        if stream_decoder is not None:
            delta = stream_decoder.finish(generated, final_text=final_text)
            if delta:
                on_token(delta)

        decode_s = sum(decode_intervals)
        kv_bytes = (
            _prompt_cache_nbytes(prompt_cache)
            + _prompt_cache_nbytes(prompt_mtp_cache))
        # Retain the state and raw logits exactly at the prompt endpoint, not
        # the post-generation endpoint.  Identical prompts can resample from
        # the same distribution at any temperature; strict extensions advance
        # the saved recurrent fold.  Arbitrary LCP branches remain ineligible.
        self.last_kv = retained_prompt_cache
        self.last_mtp_kv = retained_prompt_mtp_cache
        self._last_cache_token_ids = tuple(prompt_ids)
        self._last_prompt_logits = retained_prompt_logits
        self._last_prompt_hidden = retained_prompt_hidden
        if (cache_setting == "1" and logit_chain_setting == "1"
                and native_mtp is None):
            retained_steps = min(
                len(generated), len(generation_step_logits))
            self._last_generation_tokens = tuple(
                generated[:retained_steps])
            self._last_generation_step_logits = tuple(
                generation_step_logits[:retained_steps])
            if generation_checkpoints:
                self._last_generation_checkpoints = tuple(
                    generation_checkpoints)
            elif (cached_generation_checkpoints
                    and tuple(generated[:retained_steps])
                    == tuple(cached_generation_tokens[:retained_steps])):
                self._last_generation_checkpoints = tuple(
                    cached_generation_checkpoints)
            else:
                self._last_generation_checkpoints = ()
        else:
            self._last_generation_tokens = ()
            self._last_generation_step_logits = ()
            self._last_generation_checkpoints = ()
        if (persistent_store is not None
                and not persistent_stats["hit"]
                and not in_memory_exact
                and retained_prompt_cache is not None
                and retained_prompt_logits is not None):
            try:
                saved_stats = persistent_store.save(
                    prompt_ids,
                    retained_prompt_cache,
                    retained_prompt_logits,
                    self._last_generation_tokens,
                    self._last_generation_step_logits,
                )
                persistent_stats.update(saved_stats)
            except Exception:
                # Preserve a correct response even when the opt-in cache
                # directory becomes unavailable or the entry is unwriteable.
                persistent_stats["error"] = 1
        total_s = time.perf_counter() - request_started
        true_peak = max(
            active_before, mx.get_active_memory(), mx.get_peak_memory())
        result = {
            "text": final_text,
            "tokens": generated,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "first_token_s": first_token_s,
            "total_s": total_s,
            "tok_per_s": (
                len(decode_intervals) / decode_s if decode_s else 0.0),
            "kv_bytes": kv_bytes,
            "kv_positions": len(prompt_ids) + max(0, len(generated) - 1),
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": termination_reason,
            "true_peak_metal_bytes": true_peak,
            "prompt_tokens": len(prompt_ids),
            "path_stats": {
                "execution_backend": "mlx-lm",
                "execution_path": execution_path,
                "prompt_cache_prefix_tokens": prefix_len,
                "prompt_cache_source": (
                    "persistent-prompt-exact"
                    if persistent_stats["hit"]
                    and cache_match == "exact"
                    else f"hot-prompt-{cache_match}"
                    if prefix_len else "cold"),
                "resident_persistent_prompt_cache_enabled": int(
                    persistent_store is not None),
                "resident_persistent_prompt_cache_hit": int(
                    persistent_stats["hit"]),
                "resident_persistent_prompt_cache_saved": int(
                    persistent_stats["saved"]),
                "resident_persistent_prompt_cache_error": int(
                    persistent_stats["error"]),
                "resident_persistent_prompt_cache_load_s": float(
                    persistent_stats["load_s"]),
                "resident_persistent_prompt_cache_save_s": float(
                    persistent_stats["save_s"]),
                "resident_persistent_prompt_cache_bytes": int(
                    persistent_stats["cache_bytes"]),
                "resident_persistent_prompt_logits_bytes": int(
                    persistent_stats["logits_bytes"]),
                "resident_persistent_generation_logits": int(
                    persistent_stats["generation_logits"]),
                "retained_prompt_kv_bytes": _prompt_cache_nbytes(
                    retained_prompt_cache)
                + _prompt_cache_nbytes(retained_prompt_mtp_cache),
                "prompt_cache_full_candidate_tokens": full_candidate_tokens,
                "prompt_cache_full_candidate_lcp_tokens": full_candidate_lcp,
                "constraint_profile": getattr(constraint, "profile", "none"),
                "resident_model_payload_bytes": (
                    self._resident_backend_decision.payload_bytes),
                "resident_model_estimated_metal_bytes": (
                    self._resident_backend_decision.estimated_metal_bytes),
                "resident_model_load_s": self._load_s,
                "prefill_step_size": prefill_step_size,
                "lossy_suffix_prefill_enabled": int(
                    self._lossy_suffix_prefill is not None),
                "lossy_suffix_prefill_early_layers": int(
                    self._lossy_suffix_prefill[0]
                    if self._lossy_suffix_prefill is not None else 0),
                "lossy_suffix_prefill_tokens": int(
                    self._lossy_suffix_prefill[1]
                    if self._lossy_suffix_prefill is not None else 0),
                "prompt_state_approximate": int(
                    self._lossy_suffix_prefill is not None),
                "request_incremental_projection_bytes": incremental_bytes,
                "request_system_available_bytes": current_available,
                "request_system_required_bytes": required_available,
                "qwen_native_mtp_loaded": int(self._native_mtp is not None),
                "qwen_native_mtp_enabled": int(native_mtp is not None),
                "qwen_native_mtp_used": int(mtp_stats.get("used", 0)),
                "qwen_native_mtp_proposed": int(
                    mtp_stats.get("proposed", 0)),
                "qwen_native_mtp_accepted": int(
                    mtp_stats.get("accepted", 0)),
                "qwen_native_mtp_accept_rate": (
                    int(mtp_stats.get("accepted", 0))
                    / int(mtp_stats.get("proposed", 1))
                    if int(mtp_stats.get("proposed", 0)) else 0.0),
                "qwen_native_mtp_target_sweeps": int(
                    mtp_stats.get("target_sweeps", 0)),
                "qwen_native_mtp_rejection_refeeds": int(
                    mtp_stats.get("rejection_refeeds", 0)),
                "qwen_native_mtp_draft_head_calls": int(
                    mtp_stats.get("draft_head_calls", 0)),
                "qwen_native_mtp_sampling_proof": mtp_stats.get(
                    "sampling_proof",
                    "fallback-constrained-or-short"
                    if self._native_mtp is not None else "not-loaded"),
                "resident_logit_chain_enabled": int(
                    logit_chain_setting == "1"),
                "resident_logit_chain_eligible": int(
                    logit_chain_stats["eligible"]),
                "resident_logit_chain_candidate_tokens": int(
                    logit_chain_stats["candidate_tokens"]),
                "resident_logit_chain_reused_step_logits": int(
                    logit_chain_stats["reused_step_logits"]),
                "resident_logit_chain_catchup_sweeps": int(
                    logit_chain_stats["catchup_sweeps"]),
                "resident_logit_chain_candidate_checkpoints": int(
                    logit_chain_stats["candidate_checkpoints"]),
                "resident_logit_chain_checkpoint_restored_tokens": int(
                    logit_chain_stats["checkpoint_restored_tokens"]),
                "resident_logit_chain_diverged_at": (
                    logit_chain_stats.get("diverged_at")),
                "resident_logit_chain_retained_step_logits": len(
                    self._last_generation_step_logits),
                "resident_logit_chain_retained_checkpoints": len(
                    self._last_generation_checkpoints),
            },
        }
        if cache_setting == "0":
            self.last_kv = None
            self.last_mtp_kv = None
            self._last_cache_token_ids = ()
            self._last_prompt_logits = None
            self._last_prompt_hidden = None
            self._last_generation_tokens = ()
            self._last_generation_step_logits = ()
            self._last_generation_checkpoints = ()
            del prompt_cache
            if prompt_mtp_cache is not None:
                del prompt_mtp_cache
            mx.clear_cache()
        return result
