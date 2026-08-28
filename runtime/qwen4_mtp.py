"""Released Qwen3.8 Flash-Next Lightning-MTP draft adapter.

The target model remains the only verifier and distribution authority.  This
module only evaluates the checkpoint's top-level ``mtp.`` block to propose a
token.  The one physical MTP layer may be reused recurrently by a future
speculative controller, but this adapter deliberately does not implement that
controller or alter server defaults.

The input fusion follows Qwen's released architecture and the independently
implemented MLX reference in oMLX commit
``85708e4b9a585df42241c826b6be2b4dba018406``: project the next-token
embedding and each of the four normalized target hidden streams with separate
shared matrices, add them streamwise, run one QSA+MoE decoder layer, then use
the MTP hyper-connection mixer and the target's shared output head.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import time
from typing import Sequence

import mlx.core as mx
import numpy as np

from . import layer_runner, quant
from .kv_cache import KVCache
from .lm_head_stream import StreamedLMHead
from .qwen4_exp import (
    hyper_connection_mix,
    qwen4_rms_norm,
    run_qwen4_block,
)
from .qwen4_exp_state import Qwen4ExpStateCache
from .sampler import (
    SamplingParams,
    filtered_probabilities,
    sample,
    sample_probabilities,
    speculative_residual_probabilities,
)
from .speculative import ngram_propose


_REQUIRED_NON_EXPERT_NAMES = frozenset({
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
})


class Qwen4MTPDrafter:
    """One released-BF16 Lightning-MTP proposal step.

    The fused 512-expert tensors are never part of ``_page_names``.  WeightStore
    exposes each selected expert as three exact virtual row ranges, so a draft
    step reads only the ten routed experts instead of materializing the 5.03 GB
    fused expert bodies.
    """

    _CACHE_KEY = "qwen4_mtp:released-bf16"

    def __init__(self, engine):
        if engine.cfg.model_type != "qwen4_exp":
            raise ValueError("Qwen4 MTP requires a qwen4_exp target")
        if int(engine.cfg.qwen4_hc_count) <= 0:
            raise ValueError("Qwen4 MTP requires hyper-connection streams")
        self.engine = engine
        names = tuple(engine.store.names_with_prefix("mtp."))
        self._page_names = tuple(
            name for name in names
            if not name.startswith("mtp.layers.0.mlp.experts.")
        )
        missing = sorted(_REQUIRED_NON_EXPERT_NAMES - set(self._page_names))
        if missing:
            raise ValueError(
                f"Qwen4 MTP checkpoint is incomplete: {missing[:3]}")
        self._mtp_cfg = replace(
            engine.cfg,
            num_hidden_layers=1,
            layer_types=("full_attention",),
            qwen4_ple_layers=(),
        )
        self._resident_expert_pages: set[tuple[str, tuple[str, ...]]] = set()
        self.proposal_steps = 0
        self.proposal_expert_pages = 0
        self.proposal_expert_bytes = 0

    @property
    def non_expert_storage_bytes(self) -> int:
        return int(self.engine.store.storage_bytes(self._page_names))

    def new_cache(self) -> KVCache:
        cache = KVCache(1)
        cache.qwen4_cache = Qwen4ExpStateCache(1)
        return cache

    def _weights(self) -> dict:
        weights = self.engine.cache.get(
            self._CACHE_KEY,
            list(self._page_names),
            apply_transform=False,
        )
        invalid = [
            name for name, value in weights.items()
            if not isinstance(value, mx.array) or value.dtype != mx.bfloat16
        ]
        if invalid:
            raise ValueError(
                "Qwen4 MTP released page contains non-BF16 tensors: "
                f"{invalid[:3]}")
        return weights

    def _embed_tokens(self, tokens: Sequence[int]) -> mx.array:
        values = [int(token) for token in tokens]
        if getattr(self.engine, "_embed_rows", None) is not None:
            return self.engine._embed_rows.lookup(values)
        return layer_runner.embed(
            mx.array(values), self.engine._embed_weight())

    def fuse_inputs(
        self,
        token_embeddings: mx.array,
        hidden: mx.array,
        weights: dict,
    ) -> mx.array:
        """Fuse one or more token embeddings with four target HC streams."""
        cfg = self.engine.cfg
        hidden_width = int(cfg.qwen4_hc_count) * int(cfg.hidden_size)
        if hidden.ndim != 3 or int(hidden.shape[-1]) != hidden_width:
            raise ValueError(
                "Qwen4 MTP hidden state must be [batch, positions, "
                f"{hidden_width}]")
        if (
            token_embeddings.ndim != 3
            or tuple(token_embeddings.shape[:-1]) != tuple(hidden.shape[:-1])
            or int(token_embeddings.shape[-1]) != int(cfg.hidden_size)
        ):
            raise ValueError("Qwen4 MTP embedding/hidden geometry mismatch")
        embedding = qwen4_rms_norm(
            token_embeddings,
            weights["mtp.pre_fc_norm_embedding.weight"],
            cfg.rms_norm_eps,
        )
        projected_embedding = quant.matmul(
            embedding, weights["mtp.fc_embedding.weight"])
        hidden_streams = qwen4_rms_norm(
            hidden,
            weights["mtp.pre_fc_norm_hidden.weight"],
            cfg.rms_norm_eps,
        ).reshape(
            *hidden.shape[:-1], cfg.qwen4_hc_count, cfg.hidden_size)
        projected_hidden = quant.matmul(
            hidden_streams, weights["mtp.fc_hidden.weight"])
        return (projected_embedding[..., None, :] + projected_hidden).reshape(
            hidden.shape)

    def _iter_expert_batches(
        self,
        _layer: int,
        expert_ids: list[int],
        positions: dict[int, list[int]] | None = None,
    ):
        del positions
        ids = sorted({int(expert) for expert in expert_ids})
        if not ids:
            return iter(())
        items = []
        for expert in ids:
            prefix = f"mtp.layers.0.mlp.experts.{expert}."
            names = tuple(self.engine.store.names_with_prefix(prefix))
            if len(names) != 3:
                raise ValueError(
                    f"Qwen4 MTP expert {expert} has {len(names)} matrices")
            key = f"qwen4_mtp.expert.{expert}"
            items.append((key, list(names)))
            self._resident_expert_pages.add((key, names))
        before = int(getattr(self.engine.cache, "total_bytes", 0))
        pages = self.engine.cache.get_many(items)
        after = int(getattr(self.engine.cache, "total_bytes", before))
        self.proposal_expert_pages += len(items)
        self.proposal_expert_bytes += max(0, after - before)
        return iter(((ids, {
            expert: pages[f"qwen4_mtp.expert.{expert}"]
            for expert in ids
        }),))

    def draft_step(
        self,
        hidden: mx.array,
        last_token: int,
        mtp_cache: KVCache,
        offset: int,
        *,
        weights: dict | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Return ``(proposal_logits, post_mtp_hc_hidden)`` for one step."""
        if not isinstance(
            getattr(mtp_cache, "qwen4_cache", None), Qwen4ExpStateCache
        ):
            raise ValueError("Qwen4 MTP cache is missing QSA auxiliary state")
        if int(offset) < 0:
            raise ValueError("Qwen4 MTP offset must be non-negative")
        w = self._weights() if weights is None else weights
        token_embeddings = self._embed_tokens((last_token,))
        fused = self.fuse_inputs(token_embeddings, hidden, w)
        post_hidden = run_qwen4_block(
            fused,
            (int(last_token),),
            w,
            "mtp.layers.0",
            self._mtp_cfg,
            mtp_cache,
            0,
            int(offset),
            None,
            row_store=None,
            iter_expert_batches=self._iter_expert_batches,
        )
        mixed = hyper_connection_mix(
            post_hidden,
            w,
            "mtp.hyper_connection_mixer",
            self._mtp_cfg,
            inject=False,
        )
        head = self.engine._lm_head_weight()
        if isinstance(head, StreamedLMHead):
            logits = head.logits(mixed)[0, 0]
        else:
            logits = quant.matmul(mixed, head)[0, 0]
        mx.eval(logits, post_hidden)
        self.proposal_steps += 1
        return logits, post_hidden

    def release_round_weights(self) -> dict[str, int]:
        """Drop draft-only pages before the target verifier streams its trunk."""
        released = 0
        for key, names in self._resident_expert_pages:
            released += int(self.engine.cache.discard(key, names))
        self._resident_expert_pages.clear()
        released += int(self.engine.cache.discard(
            self._CACHE_KEY, self._page_names))
        mx.clear_cache()
        return {
            "released_pages": released,
            "proposal_steps": int(self.proposal_steps),
            "proposal_expert_pages": int(self.proposal_expert_pages),
            "proposal_expert_bytes": int(self.proposal_expert_bytes),
        }


class _Qwen4MTPBootstrapPrompt(str):
    """Prepared-prompt view retaining a paged endpoint after bootstrap."""

    def __new__(cls, prompt, token_ids):
        instance = super().__new__(cls, str(prompt))
        instance.token_ids = tuple(int(value) for value in token_ids)
        instance.tool_capsules = tuple(getattr(prompt, "tool_capsules", ()))
        instance.cache_namespace = str(
            getattr(prompt, "cache_namespace", "default") or "default")
        instance.force_paged_kv = bool(
            getattr(prompt, "force_paged_kv", False))
        instance.stable_boundary_tokens = int(
            getattr(prompt, "stable_boundary_tokens", 0) or 0)
        instance.rerank_capture_shape = dict(
            getattr(prompt, "rerank_capture_shape", {}) or {})
        instance.disable_hot_prompt_kv = bool(
            getattr(prompt, "disable_hot_prompt_kv", False))
        instance.retain_paged_kv_after_generate = True
        return instance


def _detached_probabilities(
    logits: mx.array,
    sampling: SamplingParams,
    history: Sequence[int],
) -> mx.array:
    """Materialize q without retaining the released MTP weight graph.

    A Qwen4 vocabulary row is about one MiB in FP32.  Copying at most seven
    such rows through host memory is cheap compared with one target sweep and
    lets ``release_round_weights`` actually free every draft-only page before
    verification starts.
    """
    probabilities = filtered_probabilities(
        logits, sampling, history=history).reshape(-1)
    mx.eval(probabilities)
    host = np.array(probabilities, dtype=np.float32, copy=True)
    detached = mx.array(host)
    mx.eval(detached)
    return detached


def _verify_stochastic_token(
    proposal: int,
    draft_probabilities: mx.array,
    target_logits: mx.array,
    sampling: SamplingParams,
    history: Sequence[int],
) -> tuple[bool, int, mx.array, float]:
    """Exact Leviathan p/q verification for one Qwen4 MTP proposal."""
    target_probabilities = filtered_probabilities(
        target_logits, sampling, history=history).reshape(-1)
    draft_probabilities = draft_probabilities.reshape(-1)
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError(
            "Qwen4 MTP target/draft vocabulary mismatch: "
            f"target={target_probabilities.shape}, "
            f"draft={draft_probabilities.shape}")
    overlap = mx.sum(mx.minimum(
        target_probabilities, draft_probabilities))
    mx.eval(target_probabilities, draft_probabilities, overlap)
    p_token = float(target_probabilities[int(proposal)].item())
    q_token = float(draft_probabilities[int(proposal)].item())
    ratio = 1.0 if q_token <= 0.0 else min(1.0, p_token / q_token)
    if float(mx.random.uniform().item()) <= ratio:
        return (
            True, int(proposal), target_probabilities, float(overlap.item()))
    replacement = sample_probabilities(
        speculative_residual_probabilities(
            target_probabilities, draft_probabilities))
    return (
        False, int(replacement), target_probabilities, float(overlap.item()))


class Qwen4MTPSpeculativeEngine:
    """Exact-target Lightning-MTP serving prototype.

    The server exposes it only through an explicit Qwen4 profile knob.  The
    full target remains authoritative for both greedy and categorical
    sampling, including rejection correction and exact recurrent rollback.
    Unsupported request features fall back before any bootstrap work.
    """

    def __init__(
        self,
        target,
        *,
        depth: int = 4,
        min_draft_probability: float = 0.0,
        ngram_first: bool = False,
        ngram_min_ngram: int = 2,
        ngram_max_ngram: int = 6,
        ngram_max_draft_tokens: int = 7,
        q_calibration_scales: Sequence[float] = (),
        drafter=None,
    ):
        if target.cfg.model_type != "qwen4_exp":
            raise ValueError("Qwen4 MTP requires a qwen4_exp target")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 7:
            raise ValueError("Qwen4 MTP depth must be in [1, 7]")
        if isinstance(min_draft_probability, bool):
            raise ValueError(
                "Qwen4 MTP minimum draft probability must be in [0, 1]")
        try:
            min_draft_probability = float(min_draft_probability)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Qwen4 MTP minimum draft probability must be in [0, 1]"
            ) from error
        if not math.isfinite(min_draft_probability) or not (
                0.0 <= min_draft_probability <= 1.0):
            raise ValueError(
                "Qwen4 MTP minimum draft probability must be in [0, 1]")
        if not isinstance(ngram_first, bool):
            raise TypeError("Qwen4 MTP ngram_first must be bool")
        if (isinstance(ngram_min_ngram, bool)
                or not isinstance(ngram_min_ngram, int)
                or ngram_min_ngram <= 0):
            raise ValueError("Qwen4 MTP minimum n-gram must be positive")
        if (isinstance(ngram_max_ngram, bool)
                or not isinstance(ngram_max_ngram, int)
                or ngram_max_ngram < ngram_min_ngram):
            raise ValueError(
                "Qwen4 MTP maximum n-gram must be >= minimum n-gram")
        if (isinstance(ngram_max_draft_tokens, bool)
                or not isinstance(ngram_max_draft_tokens, int)
                or not 2 <= ngram_max_draft_tokens <= 7):
            raise ValueError(
                "Qwen4 MTP n-gram draft tokens must be in [2, 7]")
        if ngram_first and min_draft_probability > 0.0:
            raise ValueError(
                "Qwen4 MTP n-gram-first and adaptive width are mutually "
                "exclusive experiments")
        normalized_calibration_scales = []
        for value in q_calibration_scales:
            if isinstance(value, bool):
                raise ValueError(
                    "Qwen4 MTP q-calibration scales must be finite values "
                    "in [0.25, 4]")
            try:
                scale = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Qwen4 MTP q-calibration scales must be finite values "
                    "in [0.25, 4]") from error
            if not math.isfinite(scale) or not 0.25 <= scale <= 4.0:
                raise ValueError(
                    "Qwen4 MTP q-calibration scales must be finite values "
                    "in [0.25, 4]")
            if scale in normalized_calibration_scales:
                raise ValueError(
                    "Qwen4 MTP q-calibration scales must be unique")
            normalized_calibration_scales.append(scale)
        if len(normalized_calibration_scales) > 9:
            raise ValueError(
                "Qwen4 MTP q-calibration accepts at most 9 scales")
        self.target = target
        self.depth = depth
        self.min_draft_probability = min_draft_probability
        self.ngram_first = ngram_first
        self.ngram_min_ngram = ngram_min_ngram
        self.ngram_max_ngram = ngram_max_ngram
        self.ngram_max_draft_tokens = ngram_max_draft_tokens
        self.q_calibration_scales = tuple(normalized_calibration_scales)
        self.drafter = drafter or Qwen4MTPDrafter(target)
        self.mtp_engine_identity = (
            f"qwen4-mtp-depth{depth}-qmin{min_draft_probability:g}"
            + (f"-ngram-first-k{ngram_max_draft_tokens}"
               if ngram_first else "")
            + ("-qcal-" + "_".join(
                f"{scale:g}" for scale in self.q_calibration_scales)
               if self.q_calibration_scales else "")
            + "-exact-target")

    def __getattr__(self, name):
        return getattr(self.target, name)

    def close(self) -> None:
        self.target.close()

    def _release_idle_phase_head(self, result: dict) -> None:
        """Drop the exact head after response state/logits are committed.

        The next request restores it at its first post-prefill projection.
        Keeping 1.27 GB resident while the server is idle only reduces the
        admission headroom for that request and cannot save any work before
        the projection point.
        """
        release = getattr(self.target, "_suspend_qwen4_phase_lm_head", None)
        released = int(release() or 0) if callable(release) else 0
        stats = result.setdefault("path_stats", {})
        stats["qwen4_mtp_idle_head_release_calls"] = int(callable(release))
        stats["qwen4_mtp_idle_head_release_bytes"] = released

    def _fallback(
        self, reason, prompt, max_tokens, on_token, stop, on_progress,
        sampling, constraint,
    ):
        generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        kwargs = {
            "on_token": on_token,
            "stop": stop,
            "on_progress": on_progress,
            "sampling": sampling,
        }
        if constraint is not None:
            kwargs["constraint"] = constraint
        result = generate(prompt, max_tokens, **kwargs)
        result.setdefault("path_stats", {}).update({
            "qwen4_mtp_enabled": 1,
            "qwen4_mtp_used": 0,
            "qwen4_mtp_min_draft_probability": self.min_draft_probability,
            "qwen4_mtp_adaptive_width_enabled": int(
                self.min_draft_probability > 0.0),
            "qwen4_mtp_ngram_first_enabled": int(self.ngram_first),
            "qwen4_mtp_fallback_reason": reason,
            "qwen4_mtp_engine_identity": self.mtp_engine_identity,
        })
        self._release_idle_phase_head(result)
        return result

    def generate(
        self,
        prompt,
        max_tokens: int = 64,
        on_token=None,
        stop=None,
        on_progress=None,
        sampling: SamplingParams | None = None,
        constraint=None,
    ) -> dict:
        sampling = sampling or SamplingParams()
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if sampling.repetition_penalty != 1.0:
            return self._fallback(
                "repetition-penalty", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if constraint is not None and not all(callable(getattr(
                constraint, name, None)) for name in (
                    "mask_logits", "accept_token", "fork")):
            return self._fallback(
                "unsupported-constraint", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)

        target = self.target
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (
            list(prepared_ids)
            if prepared_ids is not None
            else list(target.tokenizer.encode(prompt).ids)
        )
        if (
            target.effective_max_position_embeddings
            and len(ids) + max_tokens > target.effective_max_position_embeddings
        ):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={target.effective_max_position_embeddings}")

        from .engine import _cache_io_snapshot, _record_cache_io_delta

        request_started = time.perf_counter()
        request_cache_before = _cache_io_snapshot(target)
        stop = stop or []
        bootstrap_generate = getattr(
            target, "generate_with_memory_retry", target.generate)
        bootstrap_kwargs = {
            "on_token": None,
            "stop": stop,
            "on_progress": on_progress,
            "sampling": sampling,
        }
        if constraint is not None:
            bootstrap_kwargs["constraint"] = constraint
        bootstrap = bootstrap_generate(
            _Qwen4MTPBootstrapPrompt(prompt, ids), 1, **bootstrap_kwargs)
        path_stats = dict(bootstrap.get("path_stats") or {})
        if max_tokens == 1 or bootstrap.get("termination_reason") != "length":
            path_stats.update({
                "qwen4_mtp_enabled": 1,
                "qwen4_mtp_used": 0,
                "qwen4_mtp_min_draft_probability": (
                    self.min_draft_probability),
                "qwen4_mtp_adaptive_width_enabled": int(
                    self.min_draft_probability > 0.0),
                "qwen4_mtp_ngram_first_enabled": int(self.ngram_first),
                "qwen4_mtp_fallback_reason": (
                    "single-token-budget" if max_tokens == 1
                    else "terminal-first-token"),
                "qwen4_mtp_engine_identity": self.mtp_engine_identity,
            })
            bootstrap["path_stats"] = path_stats
            if on_token is not None and bootstrap.get("text"):
                on_token(bootstrap["text"])
            self._release_idle_phase_head(bootstrap)
            return bootstrap

        kv = getattr(target, "last_kv", None)
        if kv is None or getattr(kv, "kda_cache", None) is None or getattr(
                kv, "qwen4_cache", None) is None:
            raise RuntimeError(
                "Qwen4 MTP bootstrap omitted target recurrent state")
        mtp_kv = self.drafter.new_cache()
        eos = set(target.cfg.eos_token_ids)
        catchup_token = int(bootstrap["tokens"][0])
        emitted = [catchup_token]
        all_tokens = list(ids) + emitted
        h_last = target._h_last

        # A 49K prompt's complete HC hidden window is about one GiB and is no
        # longer needed after bootstrap.  BF16 -> host FP32 -> BF16 preserves
        # every value while severing its lazy weight graph before MTP pages are
        # admitted.
        endpoint_fp32 = h_last.astype(mx.float32)
        mx.eval(endpoint_fp32)
        endpoint_host = np.array(endpoint_fp32, dtype=np.float32, copy=True)
        h_last = mx.array(endpoint_host).astype(h_last.dtype)
        mx.eval(h_last)
        target._h_last = h_last
        target._h_window = h_last
        del endpoint_fp32, endpoint_host
        mx.clear_cache()

        endpoint_slot = None
        hot_slots = getattr(target, "_hot_prompt_slots", None)
        if hot_slots is not None:
            for index, slot in enumerate(hot_slots):
                if getattr(slot, "kv", None) is kv:
                    endpoint_slot = hot_slots.pop(index)
                    break
        last_emitted_logits = (
            getattr(endpoint_slot, "logits", None)
            if endpoint_slot is not None else None)

        stop_text = None
        matched_stop_sequence = None

        def stop_match(text: str):
            matches = [
                (text.find(value), index, value)
                for index, value in enumerate(stop)
                if value and text.find(value) != -1
            ]
            return min(matches) if matches else None

        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(target.tokenizer, stop)
            delta = stream_decoder.push(emitted)
            if delta:
                on_token(delta)

        proposed = 0
        accepted = 0
        rounds = 0
        target_sweeps = 0
        rollbacks = 0
        kda_restores = 0
        qwen4_restores = 0
        draft_s = 0.0
        verifier_s = 0.0
        stochastic_overlap_sum = 0.0
        stochastic_verified = 0
        q_calibration_sums = [0.0] * len(self.q_calibration_scales)
        q_calibration_counts = [0] * len(self.q_calibration_scales)
        q_calibration_step_sums = [
            [0.0] * len(self.q_calibration_scales)
            for _ in range(self.depth)
        ]
        q_calibration_step_counts = [
            [0] * len(self.q_calibration_scales)
            for _ in range(self.depth)
        ]
        selected_probabilities = []
        truncation_probabilities = []
        round_widths = []
        proposal_sources = []
        ngram_attempts = 0
        ngram_matches = 0
        ngram_proposed = 0
        ngram_accepted = 0
        ngram_rejected = 0
        ngram_native_draft_bypasses = 0
        outcomes = []
        decode_started = time.perf_counter()
        while (
            len(emitted) < max_tokens
            and emitted[-1] not in eos
            and stop_text is None
            and not bool(constraint is not None and constraint.completed)
        ):
            remaining = max_tokens - len(emitted)
            round_depth = min(self.depth, remaining)
            if round_depth <= 0:
                break
            rounds += 1
            round_start_offset = int(kv.offset)
            target_start_lengths = kv.layer_lengths()
            mtp_start_lengths = mtp_kv.layer_lengths()
            draft_constraint = constraint.fork() if constraint is not None else None
            draft_tokens = []
            draft_probabilities = []
            draft_calibration_probabilities = []
            draft_hidden = h_last
            current = catchup_token
            weights = None
            draft_started = time.perf_counter()
            ngram_tokens = []
            if self.ngram_first:
                ngram_attempts += 1
                ngram_width = min(
                    self.ngram_max_draft_tokens,
                    max(0, remaining - 1),
                )
                if ngram_width >= 2:
                    ngram_tokens = [int(token) for token in ngram_propose(
                        all_tokens,
                        ngram_width,
                        self.ngram_max_ngram,
                        self.ngram_min_ngram,
                    )]
                    # A one-token match cannot amortize the target sweep and
                    # is intentionally treated as a miss.
                    if len(ngram_tokens) < 2:
                        ngram_tokens = []
                if ngram_tokens:
                    ngram_matches += 1
                    ngram_native_draft_bypasses += 1
                    ngram_proposed += len(ngram_tokens)

            proposal_source = "N" if ngram_tokens else "M"
            proposal_sources.append(proposal_source)
            try:
                weights = None if ngram_tokens else self.drafter._weights()
                if ngram_tokens:
                    for token in ngram_tokens:
                        step_probabilities = None
                        if not sampling.is_greedy:
                            step_probabilities = mx.zeros(
                                (int(target.cfg.vocab_size),),
                                dtype=mx.float32,
                            ).at[token].add(1.0)
                            mx.eval(step_probabilities)
                        draft_tokens.append(token)
                        draft_probabilities.append(step_probabilities)
                        proposed += 1
                        selected_probabilities.append(1.0)
                else:
                    for step in range(round_depth):
                        logits, draft_hidden = self.drafter.draft_step(
                            draft_hidden,
                            current,
                            mtp_kv,
                            round_start_offset + step,
                            weights=weights,
                        )
                        if draft_constraint is not None:
                            logits = draft_constraint.mask_logits(logits)
                        if sampling.is_greedy:
                            token = int(mx.argmax(logits).item())
                            step_probabilities = None
                            selected_probability = None
                            if self.min_draft_probability > 0.0:
                                selected_probability = float(mx.softmax(
                                    logits.astype(mx.float32),
                                    axis=-1,
                                ).reshape(-1)[token].item())
                        else:
                            step_probabilities = _detached_probabilities(
                                logits,
                                sampling,
                                all_tokens + draft_tokens,
                            )
                            step_calibration_probabilities = []
                            for scale in self.q_calibration_scales:
                                if scale == 1.0:
                                    calibrated = step_probabilities
                                else:
                                    calibrated = _detached_probabilities(
                                        logits,
                                        replace(
                                            sampling,
                                            temperature=(
                                                float(sampling.temperature)
                                                * scale),
                                        ),
                                        all_tokens + draft_tokens,
                                    )
                                step_calibration_probabilities.append(
                                    calibrated)
                            token = sample_probabilities(step_probabilities)
                            selected_probability = float(
                                step_probabilities.reshape(-1)[token].item())
                        draft_tokens.append(token)
                        draft_probabilities.append(step_probabilities)
                        draft_calibration_probabilities.append(
                            tuple(step_calibration_probabilities)
                            if not sampling.is_greedy else ())
                        proposed += 1
                        if selected_probability is not None:
                            selected_probabilities.append(selected_probability)
                        if draft_constraint is not None:
                            draft_constraint.accept_token(token)
                        current = token
                        if token in eos or bool(
                            draft_constraint is not None
                            and draft_constraint.completed
                        ):
                            break
                        if (
                            self.min_draft_probability > 0.0
                            and selected_probability is not None
                            and selected_probability < self.min_draft_probability
                            and step + 1 < round_depth
                        ):
                            truncation_probabilities.append(selected_probability)
                            break
            finally:
                weights = None
                release_info = (
                    {"proposal_expert_pages": 0, "proposal_expert_bytes": 0}
                    if ngram_tokens
                    else self.drafter.release_round_weights()
                )
            draft_s += time.perf_counter() - draft_started
            if not draft_tokens:
                raise RuntimeError("Qwen4 MTP produced an empty draft round")
            round_widths.append(len(draft_tokens))

            verify_tokens = [catchup_token, *draft_tokens]
            verifier_started = time.perf_counter()
            logits_window = target.forward_tokens_serial_positions(
                verify_tokens,
                kv,
                capture_kda_endpoints=True,
                capture_qwen4_endpoints=True,
            )
            verifier_s += time.perf_counter() - verifier_started
            target_sweeps += 1
            target_end_lengths = kv.layer_lengths()
            target_growth = tuple(
                end - start for start, end in zip(
                    target_start_lengths, target_end_lengths, strict=True)
            )

            new_tokens = []
            new_logits = []
            rejected = False
            accepted_prefix = 0
            for index, draft_token in enumerate(draft_tokens):
                target_logits = logits_window[index]
                if constraint is not None:
                    target_logits = target._constraint_logits(
                        target_logits,
                        constraint,
                        hidden=target._h_window[:, index:index + 1],
                    )
                if sampling.is_greedy:
                    draft_accepted = int(mx.argmax(
                        target_logits).item()) == draft_token
                    true_token = (
                        draft_token if draft_accepted
                        else int(mx.argmax(target_logits).item())
                    )
                else:
                    step_probabilities = draft_probabilities[index]
                    if step_probabilities is None:
                        raise RuntimeError(
                            "stochastic Qwen4 MTP proposal omitted q")
                    (
                        draft_accepted,
                        true_token,
                        _target_probabilities,
                        overlap,
                    ) = _verify_stochastic_token(
                        draft_token,
                        step_probabilities,
                        target_logits,
                        sampling,
                        all_tokens + new_tokens,
                    )
                    stochastic_overlap_sum += overlap
                    stochastic_verified += 1
                    if proposal_source == "M":
                        for scale_index, calibrated in enumerate(
                                draft_calibration_probabilities[index]):
                            calibrated_overlap = mx.sum(mx.minimum(
                                _target_probabilities, calibrated))
                            mx.eval(calibrated_overlap)
                            value = float(calibrated_overlap.item())
                            q_calibration_sums[scale_index] += value
                            q_calibration_counts[scale_index] += 1
                            q_calibration_step_sums[index][scale_index] += value
                            q_calibration_step_counts[index][scale_index] += 1
                if not draft_accepted:
                    if proposal_source == "N":
                        ngram_rejected += 1
                    new_tokens.append(true_token)
                    new_logits.append(target_logits)
                    if constraint is not None:
                        constraint.accept_token(true_token)
                    rejected = True
                    break
                accepted += 1
                accepted_prefix += 1
                if proposal_source == "N":
                    ngram_accepted += 1
                new_tokens.append(draft_token)
                new_logits.append(target_logits)
                if constraint is not None:
                    constraint.accept_token(draft_token)
                if (
                    draft_token in eos
                    or len(emitted) + len(new_tokens) >= max_tokens
                    or bool(constraint is not None and constraint.completed)
                ):
                    break

            if (
                not rejected
                and accepted_prefix == len(draft_tokens)
                and len(emitted) + len(new_tokens) < max_tokens
                and new_tokens[-1] not in eos
                and not bool(constraint is not None and constraint.completed)
            ):
                bonus_logits = logits_window[len(draft_tokens)]
                if constraint is not None:
                    bonus_logits = target._constraint_logits(
                        bonus_logits,
                        constraint,
                        hidden=target._h_window[
                            :, len(draft_tokens):len(draft_tokens) + 1],
                    )
                bonus = sample(
                    bonus_logits,
                    sampling,
                    history=all_tokens + new_tokens,
                )
                new_tokens.append(bonus)
                new_logits.append(bonus_logits)
                if constraint is not None:
                    constraint.accept_token(bonus)

            emitted_before = len(emitted)
            for token, token_logits in zip(new_tokens, new_logits, strict=True):
                emitted.append(int(token))
                all_tokens.append(int(token))
                last_emitted_logits = token_logits
                if stop:
                    decoded = target.tokenizer.decode(emitted)
                    match = stop_match(decoded)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                if stream_decoder is not None and stop_text is None:
                    delta = stream_decoder.push(emitted)
                    if delta:
                        on_token(delta)
                if (
                    stop_text is not None
                    or token in eos
                    or len(emitted) >= max_tokens
                    or bool(constraint is not None and constraint.completed)
                ):
                    break

            emitted_this_round = len(emitted) - emitted_before
            target_fed = min(len(verify_tokens), max(1, emitted_this_round))
            if target_fed < len(verify_tokens):
                kda_endpoint = target.consume_serial_kda_endpoint(target_fed)
                qwen4_endpoint = target.consume_serial_qwen4_endpoint(target_fed)
                if kda_endpoint is None or qwen4_endpoint is None:
                    raise RuntimeError(
                        "Qwen4 verifier omitted an exact strict-prefix endpoint")
                retained_lengths = tuple(
                    start + (target_fed if growth else 0)
                    for start, growth in zip(
                        target_start_lengths, target_growth, strict=True)
                )
                kv.trim_layer_lengths(retained_lengths)
                kv.kda_cache = kda_endpoint
                kv.qwen4_cache.restore_recurrent_prefix(
                    qwen4_endpoint, round_start_offset + target_fed)
                h_last = target._h_window[
                    :, target_fed - 1:target_fed, :]
                target._h_last = h_last
                rollbacks += 1
                kda_restores += 1
                qwen4_restores += 1
            else:
                target.consume_serial_kda_endpoint(None)
                target.consume_serial_qwen4_endpoint(None)
                h_last = target._h_last

            mtp_end_lengths = mtp_kv.layer_lengths()
            mtp_growth = tuple(
                end - start for start, end in zip(
                    mtp_start_lengths, mtp_end_lengths, strict=True)
            )
            committed_mtp = min(len(draft_tokens), target_fed)
            if committed_mtp < len(draft_tokens):
                mtp_kv.trim_layer_lengths(tuple(
                    start + (committed_mtp if growth else 0)
                    for start, growth in zip(
                        mtp_start_lengths, mtp_growth, strict=True)
                ))
                mtp_kv.qwen4_cache.trim(
                    mtp_start_lengths[0] + committed_mtp)

            outcomes.append(
                ("N:" if proposal_source == "N" else "")
                + (f"A{accepted_prefix}" if not rejected
                   else ("R" if accepted_prefix == 0
                         else f"A{accepted_prefix}R")))
            catchup_token = emitted[-1]

        final_text = (
            stop_text if stop_text is not None
            else target.tokenizer.decode(emitted)
        )
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_started
        endpoint = len(ids) + len(emitted) - 1
        if kv.offset > endpoint:
            # This should be reachable only when a stop string truncates the
            # visible round after endpoint selection.
            raise RuntimeError(
                "Qwen4 MTP stop handling left target state beyond endpoint")

        path_stats.update({
            "qwen4_mtp_enabled": 1,
            "qwen4_mtp_used": int(proposed > 0),
            "qwen4_mtp_depth": self.depth,
            "qwen4_mtp_min_draft_probability": self.min_draft_probability,
            "qwen4_mtp_adaptive_width_enabled": int(
                self.min_draft_probability > 0.0),
            "qwen4_mtp_adaptive_truncations": len(
                truncation_probabilities),
            "qwen4_mtp_round_widths": ",".join(
                str(width) for width in round_widths),
            "qwen4_mtp_selected_probability_min": (
                min(selected_probabilities) if selected_probabilities else 0.0),
            "qwen4_mtp_selected_probability_mean": (
                sum(selected_probabilities) / len(selected_probabilities)
                if selected_probabilities else 0.0),
            "qwen4_mtp_selected_probability_max": (
                max(selected_probabilities) if selected_probabilities else 0.0),
            "qwen4_mtp_truncation_probabilities": ",".join(
                f"{probability:.8g}"
                for probability in truncation_probabilities),
            "qwen4_mtp_ngram_first_enabled": int(self.ngram_first),
            "qwen4_mtp_ngram_first_attempts": ngram_attempts,
            "qwen4_mtp_ngram_first_matches": ngram_matches,
            "qwen4_mtp_ngram_first_proposed": ngram_proposed,
            "qwen4_mtp_ngram_first_accepted": ngram_accepted,
            "qwen4_mtp_ngram_first_rejected": ngram_rejected,
            "qwen4_mtp_ngram_first_native_draft_bypasses": (
                ngram_native_draft_bypasses),
            "qwen4_mtp_proposal_sources": ",".join(proposal_sources),
            "qwen4_mtp_engine_identity": self.mtp_engine_identity,
            "qwen4_mtp_rounds": rounds,
            "qwen4_mtp_proposed": proposed,
            "qwen4_mtp_accepted": accepted,
            "qwen4_mtp_accept_rate": accepted / proposed if proposed else 0.0,
            "qwen4_mtp_stochastic": int(not sampling.is_greedy),
            "qwen4_mtp_stochastic_verified": stochastic_verified,
            "qwen4_mtp_expected_acceptance": (
                stochastic_overlap_sum / stochastic_verified
                if stochastic_verified else 0.0),
            "qwen4_mtp_q_calibration_rows": max(
                q_calibration_counts, default=0),
            "qwen4_mtp_q_calibration": json.dumps({
                "schema": "voom.qwen4-mtp-q-calibration.v1",
                "scope": "observed-native-draft-path",
                "scales": list(self.q_calibration_scales),
                "overall": [
                    (total / count if count else None)
                    for total, count in zip(
                        q_calibration_sums,
                        q_calibration_counts,
                        strict=True,
                    )
                ],
                "counts": q_calibration_counts,
                "by_step": [
                    [
                        (total / count if count else None)
                        for total, count in zip(
                            step_sums,
                            step_counts,
                            strict=True,
                        )
                    ]
                    for step_sums, step_counts in zip(
                        q_calibration_step_sums,
                        q_calibration_step_counts,
                        strict=True,
                    )
                ],
                "by_step_counts": q_calibration_step_counts,
            }, separators=(",", ":"), sort_keys=True),
            "qwen4_mtp_target_sweeps": target_sweeps,
            "qwen4_mtp_plain_equivalent_target_sweeps": max(
                0, len(emitted) - 1),
            "qwen4_mtp_target_sweeps_avoided": max(
                0, len(emitted) - 1 - target_sweeps),
            "qwen4_mtp_target_tokens_per_sweep": (
                (len(emitted) - 1) / target_sweeps if target_sweeps else 0.0),
            "qwen4_mtp_target_prefix_rollbacks": rollbacks,
            "qwen4_mtp_kda_endpoint_restores": kda_restores,
            "qwen4_mtp_aux_endpoint_restores": qwen4_restores,
            "qwen4_mtp_draft_s": draft_s,
            "qwen4_mtp_verifier_s": verifier_s,
            "qwen4_mtp_round_outcomes": ",".join(outcomes),
            "qwen4_mtp_non_expert_storage_bytes": (
                self.drafter.non_expert_storage_bytes),
            "qwen4_mtp_proposal_expert_pages": int(
                release_info.get("proposal_expert_pages", 0)),
            "qwen4_mtp_proposal_expert_bytes": int(
                release_info.get("proposal_expert_bytes", 0)),
            "qwen4_mtp_constraint_verified": int(constraint is not None),
            "qwen4_serial_verify_union_layers": int(getattr(
                target, "_qwen4_serial_verify_union_layers", 0)),
            "qwen4_serial_verify_expert_slots": int(getattr(
                target, "_qwen4_serial_verify_expert_slots", 0)),
            "qwen4_serial_verify_union_experts": int(getattr(
                target, "_qwen4_serial_verify_union_experts", 0)),
            "qwen4_serial_verify_expert_pages_avoided": int(getattr(
                target, "_qwen4_serial_verify_expert_pages_avoided", 0)),
            "qwen4_serial_verify_union_fetch_s": float(getattr(
                target, "_qwen4_serial_verify_union_fetch_s", 0.0)),
            "qwen4_serial_verify_page_prepare_s": float(getattr(
                target, "_qwen4_serial_verify_page_prepare_s", 0.0)),
            "qwen4_serial_verify_weight_wait_s": float(getattr(
                target, "_qwen4_serial_verify_weight_wait_s", 0.0)),
            "qwen4_serial_verify_reserve_s": float(getattr(
                target, "_qwen4_serial_verify_reserve_s", 0.0)),
            "qwen4_serial_verify_linear_compute_s": float(getattr(
                target, "_qwen4_serial_verify_linear_compute_s", 0.0)),
            "qwen4_serial_verify_full_compute_s": float(getattr(
                target, "_qwen4_serial_verify_full_compute_s", 0.0)),
            "qwen4_serial_verify_linear_layers": int(getattr(
                target, "_qwen4_serial_verify_linear_layers", 0)),
            "qwen4_serial_verify_full_layers": int(getattr(
                target, "_qwen4_serial_verify_full_layers", 0)),
            "qwen4_serial_verify_head_s": float(getattr(
                target, "_qwen4_serial_verify_head_s", 0.0)),
            "qwen4_serial_verify_pipelined_expert_layers": int(getattr(
                target,
                "_qwen4_serial_verify_pipelined_expert_layers",
                0,
            )),
            "expert_compute_batches": int(getattr(
                target, "_expert_compute_batches", 0)),
            "max_experts_per_compute_batch": int(getattr(
                target, "_max_experts_per_compute_batch", 0)),
            "adaptive_expert_batch_clamps": int(getattr(
                target, "_adaptive_expert_batch_clamps", 0)),
            "min_adaptive_expert_batch": int(getattr(
                target, "_min_adaptive_expert_batch", 0)),
            "expert_batch_prefetch": int(getattr(
                target, "_expert_batch_executor", None) is not None),
            "expert_batch_prefetch_submitted": int(getattr(
                target, "_expert_batch_prefetch_submitted", 0)),
            "expert_batch_prefetch_wait_s": float(getattr(
                target, "_expert_batch_prefetch_wait_s", 0.0)),
            "expert_batch_prefetch_hidden_s": float(getattr(
                target, "_expert_batch_prefetch_hidden_s", 0.0)),
            "expert_shared_overlap_layers": int(getattr(
                target, "_expert_shared_overlap_layers", 0)),
        })
        request_cache_after = _cache_io_snapshot(target)
        _record_cache_io_delta(
            target, request_cache_before, path_stats,
            after=request_cache_after)

        if endpoint_slot is not None and last_emitted_logits is not None:
            mx.eval(last_emitted_logits)
            kv.kda_cache.synchronize()
            kv.qwen4_cache.synchronize()
            endpoint_slot.tokens = tuple(ids + emitted[:-1])
            endpoint_slot.kv = kv
            endpoint_slot.logits = last_emitted_logits
            endpoint_slot.prompt_length = len(ids)
            endpoint_slot.segment_chain = ()
            target._append_hot_prompt_slot(endpoint_slot)

        target.last_kv = kv
        if target.governor is not None:
            target._true_peak_metal_bytes = max(
                target._true_peak_metal_bytes,
                target.governor.request_peak(),
                mx.get_active_memory(),
            )
        total_s = time.perf_counter() - request_started
        grammar_completed = bool(
            constraint is not None and constraint.completed)
        result = {
            "text": final_text,
            "tokens": emitted,
            "prefill_s": float(bootstrap.get("prefill_s", 0.0)),
            "decode_s": decode_s,
            "first_token_s": float(bootstrap.get(
                "first_token_s", bootstrap.get("prefill_s", 0.0))),
            "total_s": total_s,
            "tok_per_s": (
                (len(emitted) - 1) / decode_s if len(emitted) > 1 else 0.0),
            "kv_bytes": kv.nbytes(),
            "kv_positions": kv.offset,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": (
                "stop_sequence" if stop_text is not None else
                "grammar" if grammar_completed else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": target._true_peak_metal_bytes,
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
        }
        profiler = getattr(target, "_request_profiler", None)
        execution_profile = (
            profiler.result(total_s) if profiler is not None else None)
        if execution_profile is not None:
            result["execution_profile"] = execution_profile
        self._release_idle_phase_head(result)
        return result
