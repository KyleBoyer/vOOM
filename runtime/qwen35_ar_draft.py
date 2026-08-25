"""Resident autoregressive Qwen3.5 draft source for exact target verification.

The draft model is allowed to be quantized or otherwise approximate.  It never
owns a released answer: :class:`runtime.qwen35_mtp.QwenMTPSpeculativeEngine`
verifies every proposal against the configured target and applies the exact
``p/q`` correction for stochastic requests.  This adapter only manages the
smaller model's hybrid recurrent/attention cache.

MLX-LM's Qwen3.5 cache wrappers are copy-on-write under the shallow fork used
by the resident backend: ArraysCache replaces recurrent arrays and KVCache
advances a private numeric offset.  Each proposal round therefore runs on a
disposable fork.  Once the target has selected a prefix, this adapter replays
only the exact target-fed input tokens from the retained round base.  Rejected
draft suffixes can never leak into the next proposal distribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

import mlx.core as mx

from .config import ModelConfig
from .resident_mlx_lm import (
    ResidentMLXLMEngine,
    _fork_prompt_cache,
    _prompt_cache_nbytes,
    choose_resident_backend,
)


_MAX_RETAINED_DRAFT_PAYLOAD_BYTES = 600_000_000


def _canonical_tokenizer_core(path: Path) -> tuple[str, dict]:
    """Fingerprint the token-id semantics needed by an ID-fed draft model.

    Qwen3.5 and Qwen3.8 expose a different number of *registered* added tokens,
    while retaining the same complete vocabulary, merges, normalization, and
    byte-level encoding.  The adapter consumes target token IDs directly, so
    registration metadata is intentionally excluded; the embedding/output row
    assigned to every ID and the ordinary text tokenization must match.
    """
    try:
        raw = json.loads(path.read_text())
        model = raw["model"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid tokenizer artifact {path}: {error}") from error
    merges = []
    for index, merge in enumerate(model.get("merges") or ()):
        if isinstance(merge, str):
            pair = merge.split(" ")
        elif isinstance(merge, list):
            pair = merge
        else:
            raise ValueError(
                f"invalid tokenizer merge {index} in {path}: {merge!r}")
        if len(pair) != 2 or not all(isinstance(token, str) for token in pair):
            raise ValueError(
                f"invalid tokenizer merge {index} in {path}: {merge!r}")
        merges.append(tuple(pair))
    core = {
        "model": {
            **{
                key: model.get(key)
                for key in (
                    "type", "vocab", "unk_token", "byte_fallback",
                    "fuse_unk", "ignore_merges",
                )
            },
            # tokenizers has used both legacy string pairs ("a b") and
            # structured pairs (["a", "b"]) for the same ordered BPE merge.
            # Normalize only that serialization detail; token IDs, pair
            # contents, ordering, and every surrounding tokenizer component
            # remain fail-closed equality gates.
            "merges": merges,
        },
        "normalizer": raw.get("normalizer"),
        "pre_tokenizer": raw.get("pre_tokenizer"),
        "decoder": raw.get("decoder"),
    }
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), core


def validate_qwen_ar_draft_compatibility(
    target_dir: str | Path,
    target_cfg: ModelConfig,
    draft_dir: str | Path,
    draft_cfg: ModelConfig,
) -> str:
    """Fail closed unless target IDs have identical draft-side semantics."""
    target_dir = Path(target_dir)
    draft_dir = Path(draft_dir)
    if target_cfg.model_type != "qwen3_5" or target_cfg.num_experts:
        raise ValueError("resident Qwen AR draft requires a dense qwen3_5 target")
    if draft_cfg.model_type != "qwen3_5" or draft_cfg.num_experts:
        raise ValueError("resident Qwen AR draft must be a dense qwen3_5 model")
    if draft_cfg.vocab_size != target_cfg.vocab_size:
        raise ValueError(
            "resident Qwen AR draft vocabulary width differs from target: "
            f"{draft_cfg.vocab_size} != {target_cfg.vocab_size}")
    if draft_cfg.max_position_embeddings < target_cfg.max_position_embeddings:
        raise ValueError(
            "resident Qwen AR draft context limit is shorter than target: "
            f"{draft_cfg.max_position_embeddings} < "
            f"{target_cfg.max_position_embeddings}")
    target_fingerprint, target_core = _canonical_tokenizer_core(
        target_dir / "tokenizer.json")
    draft_fingerprint, draft_core = _canonical_tokenizer_core(
        draft_dir / "tokenizer.json")
    if target_core != draft_core:
        raise ValueError(
            "resident Qwen AR draft tokenizer ID semantics differ from target")
    if target_fingerprint != draft_fingerprint:
        raise RuntimeError("canonical tokenizer comparison was not deterministic")
    return target_fingerprint


class ResidentQwenARDrafter:
    """Proposal-only adapter around a fully resident MLX-LM Qwen model."""

    proposal_source = "A"
    request_weight_representation = "resident-ar-quantized"

    def __init__(
        self,
        backend: ResidentMLXLMEngine,
        *,
        identity: str,
        prefill_step_size: int = 512,
        backend_loader: Callable[[], ResidentMLXLMEngine] | None = None,
        unload_between_requests: bool = False,
        weight_representation: str = "resident-ar-quantized",
        retain_for_target_verification: bool = False,
        resident_payload_bytes: int = 0,
    ):
        if isinstance(prefill_step_size, bool) or not isinstance(
                prefill_step_size, int) or not 1 <= prefill_step_size <= 4096:
            raise ValueError("Qwen AR draft prefill step size must be in [1, 4096]")
        if backend is None and backend_loader is None:
            raise ValueError("Qwen AR draft requires a backend or loader")
        if backend is not None and getattr(backend, "model", None) is None:
            raise ValueError("Qwen AR draft backend has no loaded model")
        self.backend = backend
        self.model = None if backend is None else backend.model
        self._backend_loader = backend_loader
        self._unload_between_requests = bool(unload_between_requests)
        self.retain_for_target_verification = bool(
            retain_for_target_verification)
        self.resident_payload_bytes = int(resident_payload_bytes)
        if self.retain_for_target_verification and (
                self.resident_payload_bytes <= 0
                or self.resident_payload_bytes
                > _MAX_RETAINED_DRAFT_PAYLOAD_BYTES):
            raise ValueError(
                "retained Qwen AR draft payload must be in (0, 600MB]")
        self.staged_load = backend is None
        self.request_weight_representation = str(weight_representation)
        self.identity = str(identity)
        self.prefill_step_size = int(prefill_step_size)
        self._committed_cache = None
        self._committed_ids: tuple[int, ...] = ()
        self._round_base = None
        self._round_anchor_ids: tuple[int, ...] = ()
        self._working_cache = None
        self._working_inputs: list[int] = []
        self._request_prompt_tokens = 0
        self._request_prefill_s = 0.0
        self._round_sync_s = 0.0
        self._proposal_s = 0.0
        self._commit_replay_s = 0.0
        self._proposal_steps = 0
        self._commit_replay_steps = 0
        self._round_sync_steps = 0
        self._peak_cache_bytes = 0
        self._backend_load_s = 0.0
        self._backend_loads = 0
        self._backend_unload_s = 0.0
        self._backend_unloads = 0
        self._verification_suspends = 0
        self._verification_suspend_s = 0.0
        self._verification_released_active_bytes = 0
        self._verification_retained_rounds = 0
        self._verification_retained_active_bytes_peak = 0

    @property
    def weights_loaded(self) -> bool:
        return self.backend is not None

    def ensure_loaded(self) -> None:
        if self.backend is not None:
            return
        loader = self._backend_loader
        if loader is None:
            raise RuntimeError("Qwen AR draft backend loader is unavailable")
        started = time.perf_counter()
        backend = loader()
        if getattr(backend, "model", None) is None:
            backend.close()
            raise ValueError("Qwen AR draft loader returned no model")
        self.backend = backend
        self.model = backend.model
        self._backend_load_s += time.perf_counter() - started
        self._backend_loads += 1

    @classmethod
    def from_model_dir(
        cls,
        draft_dir: str | Path,
        *,
        target_dir: str | Path,
        target_cfg: ModelConfig,
        prefill_step_size: int = 512,
        retain_for_target_verification: bool = False,
    ) -> "ResidentQwenARDrafter":
        draft_dir = Path(draft_dir).expanduser().resolve()
        draft_cfg = ModelConfig.from_dir(draft_dir)
        tokenizer_fingerprint = validate_qwen_ar_draft_compatibility(
            target_dir, target_cfg, draft_dir, draft_cfg)

        # This is an explicitly requested auxiliary backend even when the
        # operator correctly keeps the 27B target on vOOM.  Reuse the resident
        # backend's measured all-MXFP4, system-headroom, and Metal-ceiling gate
        # rather than inventing a looser sidecar admission rule.
        admission_env = dict(os.environ)
        admission_env["VMODEL_RESIDENT_BACKEND"] = "mlx-lm"
        decision = choose_resident_backend(
            draft_dir,
            draft_cfg,
            "fast",
            requires_vision=False,
            execution_profile="",
            allow_lossy_draft=True,
            env=admission_env,
        )
        if not decision.admitted:
            raise MemoryError(
                "resident Qwen AR draft was not admitted: "
                f"{decision.reason} payload={decision.payload_bytes} "
                f"estimated_metal={decision.estimated_metal_bytes} "
                f"available={decision.available_bytes}")
        if (retain_for_target_verification
                and int(decision.payload_bytes)
                > _MAX_RETAINED_DRAFT_PAYLOAD_BYTES):
            raise MemoryError(
                "retained Qwen AR draft exceeds the 600MB payload ceiling: "
                f"payload={decision.payload_bytes}")
        identity = (
            f"qwen-ar-{draft_dir.name}-"
            f"tok-{tokenizer_fingerprint[:12]}-"
            f"payload-{decision.payload_bytes}"
            f"{'-retain' if retain_for_target_verification else ''}"
        )
        quantization = json.loads((draft_dir / "config.json").read_text()).get(
            "quantization_config", {})
        weight_representation = (
            f"resident-ar-{quantization.get('mode', 'unknown')}"
            f"{quantization.get('bits', 'unknown')}"
        )

        def load_backend() -> ResidentMLXLMEngine:
            fresh = choose_resident_backend(
                draft_dir,
                draft_cfg,
                "fast",
                requires_vision=False,
                execution_profile="",
                allow_lossy_draft=True,
                env=admission_env,
            )
            if not fresh.admitted:
                raise MemoryError(
                    "resident Qwen AR draft was not admitted after target "
                    f"prefill: {fresh.reason} payload={fresh.payload_bytes} "
                    f"estimated_metal={fresh.estimated_metal_bytes} "
                    f"available={fresh.available_bytes}")
            return ResidentMLXLMEngine(
                draft_dir,
                draft_cfg,
                SimpleNamespace(),
                fresh,
                auxiliary=True,
            )

        return cls(
            None,
            identity=identity,
            prefill_step_size=prefill_step_size,
            backend_loader=load_backend,
            unload_between_requests=True,
            weight_representation=weight_representation,
            retain_for_target_verification=retain_for_target_verification,
            resident_payload_bytes=int(decision.payload_bytes),
        )

    @staticmethod
    def _cache_state(cache) -> list:
        return [entry.state for entry in cache]

    def _record_peak(self, *caches) -> None:
        self._peak_cache_bytes = max(
            self._peak_cache_bytes,
            sum(_prompt_cache_nbytes(cache) for cache in caches if cache),
        )

    def _forward(self, cache, token_ids: Sequence[int]) -> mx.array:
        if not token_ids:
            raise ValueError("resident Qwen AR draft cannot forward zero tokens")
        ids = mx.array([list(map(int, token_ids))], dtype=mx.int32)
        logits = self.model(ids, cache=cache)
        row = logits[:, -1, :].reshape(-1)
        mx.eval(row, self._cache_state(cache))
        return row

    def begin_request(self, prompt_ids: Sequence[int]) -> None:
        """Build an exact draft endpoint for every target prompt token."""
        self._clear_request_state()
        self.ensure_loaded()
        self._request_prompt_tokens = 0
        self._request_prefill_s = 0.0
        self._round_sync_s = 0.0
        self._proposal_s = 0.0
        self._commit_replay_s = 0.0
        self._proposal_steps = 0
        self._commit_replay_steps = 0
        self._round_sync_steps = 0
        self._peak_cache_bytes = 0
        self._verification_retained_rounds = 0
        self._verification_retained_active_bytes_peak = 0
        normalized = tuple(map(int, prompt_ids))
        if not normalized:
            raise ValueError("resident Qwen AR draft requires a nonempty prompt")
        limit = int(getattr(self.backend, "effective_max_position_embeddings", 0))
        if limit and len(normalized) > limit:
            raise ValueError(
                f"Qwen AR draft prompt({len(normalized)}) exceeds limit={limit}")
        cache = self.backend._make_prompt_cache()
        started = time.perf_counter()
        for start in range(0, len(normalized), self.prefill_step_size):
            self._forward(
                cache, normalized[start:start + self.prefill_step_size])
            mx.clear_cache()
        self._request_prefill_s = time.perf_counter() - started
        self._request_prompt_tokens = len(normalized)
        self._committed_cache = cache
        self._committed_ids = normalized
        self._record_peak(cache)

    def begin_round(self, all_tokens: Sequence[int]) -> None:
        """Synchronize plain target steps, then fork one disposable proposal path."""
        if self.backend is None:
            raise RuntimeError(
                "Qwen AR draft weights must be restored before begin_round")
        if self._committed_cache is None:
            raise RuntimeError("Qwen AR draft request was not initialized")
        normalized = tuple(map(int, all_tokens))
        if not normalized:
            raise ValueError("Qwen AR draft round has no pending token")
        desired = normalized[:-1]
        if desired[:len(self._committed_ids)] != self._committed_ids:
            raise RuntimeError(
                "Qwen AR draft target history diverged before its committed endpoint")
        started = time.perf_counter()
        missing = desired[len(self._committed_ids):]
        for token in missing:
            self._forward(self._committed_cache, [token])
            self._round_sync_steps += 1
        self._round_sync_s += time.perf_counter() - started
        self._committed_ids = desired
        self._round_base = _fork_prompt_cache(self._committed_cache)
        self._round_anchor_ids = desired
        self._working_cache = _fork_prompt_cache(self._committed_cache)
        self._working_inputs = []
        self._record_peak(self._round_base, self._working_cache)

    def draft_step(
        self, h_last: mx.array, last_token: int, _mtp_kv, _offset: int,
        _weights=None,
    ) -> tuple[mx.array, mx.array]:
        if self._working_cache is None:
            raise RuntimeError("Qwen AR draft step has no active round")
        started = time.perf_counter()
        logits = self._forward(self._working_cache, [int(last_token)])
        self._proposal_s += time.perf_counter() - started
        self._proposal_steps += 1
        self._working_inputs.append(int(last_token))
        self._record_peak(self._round_base, self._working_cache)
        # The native-MTP adapter threads a hidden value between steps.  A full
        # AR model owns that state inside its private cache, so the target hidden
        # is deliberately returned unchanged and ignored on the next call.
        return logits, h_last

    def draft_logits(self, *args, **kwargs) -> mx.array:
        logits, _hidden = self.draft_step(*args, **kwargs)
        return logits

    def draft_token(self, *args, **kwargs) -> int:
        return int(mx.argmax(self.draft_logits(*args, **kwargs)))

    def commit_target_inputs(self, input_tokens: Sequence[int]) -> None:
        """Install only the exact input prefix selected by the target verifier."""
        if self.backend is None:
            raise RuntimeError(
                "Qwen AR draft weights must be restored before commit")
        if self._round_base is None:
            raise RuntimeError("Qwen AR draft commit has no active round")
        committed = tuple(map(int, input_tokens))
        common = min(len(committed), len(self._working_inputs))
        if committed[:common] != tuple(self._working_inputs[:common]):
            raise RuntimeError(
                "Qwen AR target committed a non-prefix of the proposed inputs")

        # Drop the speculative endpoint before replaying the authoritative
        # prefix, bounding live recurrent-state copies to the round base plus
        # one committed path.  Replay is cheap on the resident 4B and avoids
        # retaining one ~recurrent-state snapshot per proposal depth.
        self._working_cache = None
        self._working_inputs = []
        cache = _fork_prompt_cache(self._round_base)
        started = time.perf_counter()
        for token in committed:
            self._forward(cache, [token])
            self._commit_replay_steps += 1
        self._commit_replay_s += time.perf_counter() - started
        self._committed_cache = cache
        self._committed_ids = self._round_anchor_ids + committed
        self._round_base = None
        self._round_anchor_ids = ()
        self._record_peak(cache)
        mx.clear_cache()

    def suspend_for_target_verification(self) -> dict[str, int | float]:
        """Release only draft weights while retaining its exact private state.

        The target verifier streams a complete 27B body and owns the released
        distribution.  Keeping a 4B proposal model resident during that sweep
        wastes headroom and, on the 16 GB host, can cross the hard available-
        memory floor.  Prompt/KV/Delta caches are owned by this adapter rather
        than ``ResidentMLXLMEngine.close()``, so they remain exact across the
        weight-only suspension.  The speculative working fork is no longer
        needed after its proposal IDs have been recorded; dropping it here
        leaves only the immutable round base needed for authoritative replay.
        """
        backend = self.backend
        if backend is None:
            return {"suspended": 0, "released_active_bytes": 0, "seconds": 0.0}
        if self.retain_for_target_verification:
            # The disposable proposal fork is no longer useful, but a bounded
            # <=600MB draft may keep its weights and authoritative committed
            # state resident. The target remains exact and its governor sees
            # these referenced MLX bytes before admitting every target page.
            self._working_cache = None
            active = int(mx.get_active_memory())
            self._verification_retained_rounds += 1
            self._verification_retained_active_bytes_peak = max(
                self._verification_retained_active_bytes_peak, active)
            mx.clear_cache()
            return {
                "suspended": 0,
                "released_active_bytes": 0,
                "retained": 1,
                "retained_active_bytes": active,
                "seconds": 0.0,
            }
        started = time.perf_counter()
        before = int(mx.get_active_memory())
        self._working_cache = None
        self.backend = None
        self.model = None
        backend.close()
        after = int(mx.get_active_memory())
        elapsed = time.perf_counter() - started
        released = max(0, before - after)
        self._backend_unload_s += elapsed
        self._backend_unloads += 1
        self._verification_suspends += 1
        self._verification_suspend_s += elapsed
        self._verification_released_active_bytes += released
        return {
            "suspended": 1,
            "released_active_bytes": released,
            "seconds": elapsed,
        }

    def telemetry_snapshot(self) -> dict[str, int | float | str]:
        return {
            "identity": self.identity,
            "request_prompt_tokens": self._request_prompt_tokens,
            "request_prefill_s": self._request_prefill_s,
            "round_sync_s": self._round_sync_s,
            "round_sync_steps": self._round_sync_steps,
            "proposal_s": self._proposal_s,
            "proposal_steps": self._proposal_steps,
            "commit_replay_s": self._commit_replay_s,
            "commit_replay_steps": self._commit_replay_steps,
            "peak_cache_bytes": self._peak_cache_bytes,
            "committed_tokens": len(self._committed_ids),
            "backend_load_s": self._backend_load_s,
            "backend_loads": self._backend_loads,
            "backend_unload_s": self._backend_unload_s,
            "backend_unloads": self._backend_unloads,
            "verification_suspends": self._verification_suspends,
            "verification_suspend_s": self._verification_suspend_s,
            "verification_released_active_bytes": (
                self._verification_released_active_bytes),
            "verification_retain_enabled": int(
                self.retain_for_target_verification),
            "verification_retained_rounds": (
                self._verification_retained_rounds),
            "verification_retained_active_bytes_peak": (
                self._verification_retained_active_bytes_peak),
        }

    def _clear_request_state(self) -> None:
        self._committed_cache = None
        self._committed_ids = ()
        self._round_base = None
        self._round_anchor_ids = ()
        self._working_cache = None
        self._working_inputs = []

    def end_request(self, *, clear_device_cache: bool = True) -> None:
        self._clear_request_state()
        if self._unload_between_requests and self.backend is not None:
            started = time.perf_counter()
            backend = self.backend
            self.backend = None
            self.model = None
            backend.close()
            self._backend_unload_s += time.perf_counter() - started
            self._backend_unloads += 1
        if clear_device_cache:
            mx.clear_cache()

    def close(self) -> None:
        self.end_request()
        backend = self.backend
        self.backend = None
        self.model = None
        self._backend_loader = None
        if backend is not None:
            backend.close()
