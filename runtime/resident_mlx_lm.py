"""Optional fully-resident MLX-LM backend for safely fitting text models.

vOOM remains the out-of-core engine.  This module is imported only after a
pure admission policy proves that a locally-derived, fully-quantized dense
Qwen checkpoint fits below both the Metal ceiling and current system headroom.
Missing/incompatible MLX-LM therefore never makes the core runtime unusable.
"""

from __future__ import annotations

import gc
import importlib
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
_SUPPORTED_AUTO_MODEL_TYPES = frozenset({"qwen3_5"})


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
        self._last_cache_token_ids: tuple[int, ...] = ()
        self._generation_sampled_tokens = 0

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
        model = getattr(self, "model", None)
        if model is not None:
            self.model = None
            del model
        self.last_kv = None
        self._last_cache_token_ids = ()
        gc.collect()
        mx.clear_cache()

    def discard_failed_request_state(self) -> None:
        self.last_kv = None
        self._last_cache_token_ids = ()
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
        retained = _prompt_cache_nbytes(self.last_kv)
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

    def _direct_tokens(
        self, prompt_ids: list[int], max_tokens: int, sampling: SamplingParams,
        constraint, generated: list[int], prompt_cache, progress,
        prefill_step_size: int, prefix_len: int = 0,
    ):
        prompt = mx.array(prompt_ids, dtype=mx.int32)
        total = len(prompt_ids)
        processed = prefix_len
        progress(processed, total)
        while total - processed > 1:
            count = min(prefill_step_size, (total - processed) - 1)
            self.model(prompt[processed:processed + count][None],
                       cache=prompt_cache)
            mx.eval([entry.state for entry in prompt_cache])
            processed += count
            progress(processed, total)
            mx.clear_cache()

        current = prompt[processed:]
        while True:
            logits = self.model(current[None], cache=prompt_cache)[:, -1, :]
            values = logits.reshape(-1)
            if constraint is not None:
                values = constraint.mask_logits(values)
            token = sample(
                values, sampling, history=[*prompt_ids, *generated])
            if not generated:
                progress(total, total)
            yield token
            current = mx.array([token], dtype=mx.int32)

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
        prefix_len = (
            _exact_extension_prefix(self._last_cache_token_ids, prompt_ids)
            if cache_setting == "1" and self.last_kv is not None else 0)
        if not prefix_len:
            self.last_kv = None
            self._last_cache_token_ids = ()
            gc.collect()
            mx.clear_cache()
        request_reserve_mb = _positive_env_mb(
            os.environ, "VMODEL_MLX_LM_REQUEST_SYSTEM_RESERVE_MB", 1_200)
        incremental_bytes = _qwen35_request_incremental_bytes(
            self.cfg, len(prompt_ids) + max_tokens - prefix_len)
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
                        "hot-exact-extension" if prefix_len else "cold"),
                })

        prompt_cache = (
            self.last_kv if prefix_len else self._make_prompt_cache())
        self.last_kv = prompt_cache
        generated: list[int] = []
        stream_decoder = (
            IncrementalDetokenizer(self.tokenizer, stop)
            if on_token is not None else None)
        active_before = mx.get_active_memory()
        mx.reset_peak_memory()

        direct = (
            constraint is not None
            or sampling.repetition_penalty != 1.0
            or prefix_len > 0)
        if direct:
            token_iterator = self._direct_tokens(
                prompt_ids, max_tokens, sampling, constraint, generated,
                prompt_cache, progress, prefill_step_size, prefix_len)
            execution_path = (
                "mlx_lm_exact_extension"
                if prefix_len else "mlx_lm_direct_constrained")
        else:
            generate_module = importlib.import_module("mlx_lm.generate")

            if sampling.is_greedy:
                sampler = lambda values: mx.argmax(values, axis=-1)
            else:
                sampler = lambda values: mx.array(sample(values, sampling))
            token_iterator = (
                int(token)
                for token, _logprobs in generate_module.generate_step(
                    mx.array(prompt_ids, dtype=mx.int32),
                    self.model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=prompt_cache,
                    prefill_step_size=prefill_step_size,
                    prompt_progress_callback=progress,
                )
            )
            execution_path = "mlx_lm_generate_step"

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

        total_s = time.perf_counter() - request_started
        decode_s = sum(decode_intervals)
        kv_bytes = _prompt_cache_nbytes(prompt_cache)
        # Generation caches all prompt tokens and all generated tokens except
        # the final yielded token.  That exact endpoint can seed a later strict
        # extension; no trim, LCP branch, or approximate recurrent state is
        # accepted.
        self._last_cache_token_ids = tuple(
            [*prompt_ids, *generated[:-1]])
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
                    "hot-exact-extension" if prefix_len else "cold"),
                "constraint_profile": getattr(constraint, "profile", "none"),
                "resident_model_payload_bytes": (
                    self._resident_backend_decision.payload_bytes),
                "resident_model_estimated_metal_bytes": (
                    self._resident_backend_decision.estimated_metal_bytes),
                "resident_model_load_s": self._load_s,
                "prefill_step_size": prefill_step_size,
                "request_incremental_projection_bytes": incremental_bytes,
                "request_system_available_bytes": current_available,
                "request_system_required_bytes": required_available,
            },
        }
        if cache_setting == "0":
            self.last_kv = None
            self._last_cache_token_ids = ()
            del prompt_cache
            mx.clear_cache()
        return result
