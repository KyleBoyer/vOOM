"""Speculative decoding (Phase 9), greedy target-verifying design.

Exactness is an implementation-path property, not a blanket label: each path must
pass target-only greedy token A/B, and only an actual exact logit tie is exempt.
The GLM MTP path remains provisional under F23.

Why it matters for a paged runtime: a streamed target pays fixed/shared bytes once
per forward call, but a MoE verifier also reads the union of routed experts over
k+1 positions. Verification is therefore one target call, not necessarily one
token's disk bill. Every accepted token remains target-verified, but speed must be
reported as committed tokens per physical target byte (F01/F23).

Bookkeeping invariants (greedy):
  all_tokens = prompt + emitted; the final element is never yet fed to either model.
  target KV always holds len(all_tokens) - 1 positions after rollback.
  draft KV may lag; it is caught up by feeding the missing span before proposing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from .engine import StreamingEngine
from .sampler import SamplingParams


@dataclass
class SpecStats:
    sweeps: int = 0  # target calls; MoE store-accounted bytes depend on lane unions
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    draft_s: float = 0.0
    verify_s: float = 0.0
    draft_bytes: int = 0  # F01: store-accounted bytes while PROPOSING (MTP etc.)
    draft_oov_fallbacks: int = 0
    resident_draft_rounds: int = 0
    resident_draft_tokens: int = 0
    rounds: list = field(default_factory=list)  # F01: (k_eff, m_accepted, verify_bytes) per round

    def summary(self) -> str:
        acc = self.accepted / self.proposed * 100 if self.proposed else 0.0
        per_sweep = self.emitted / self.sweeps if self.sweeps else 0.0
        return (
            f"speculative: {self.emitted} tokens in {self.sweeps} target sweeps "
            f"({per_sweep:.2f} tok/sweep), acceptance {self.accepted}/{self.proposed} "
            f"({acc:.0f}%), draft {self.draft_s:.1f}s, verify {self.verify_s:.1f}s"
        )

    def bytes_summary(self) -> str:
        """F01: per-round store-accounted bytes.

        Raw safetensors currently report requested logical tensor payload, not
        observed physical I/O; fit physical-byte models only after F66 adds OS/
        device counters.
        """
        if not self.rounds:
            return "byte telemetry: no rounds recorded"
        plain = [b for k, _, b in self.rounds if k == 0]
        by_k: dict[int, list[int]] = {}
        committed_bytes = self.draft_bytes + sum(b for _, _, b in self.rounds)
        committed = sum(m + 1 for _, m, _ in self.rounds)
        for k, _, b in self.rounds:
            if k > 0:
                by_k.setdefault(k, []).append(b)
        lines = [f"byte telemetry: {committed} committed in "
                 f"{committed_bytes / 1e9:.1f} GB verify+draft "
                 f"({committed_bytes / max(committed, 1) / 1e9:.2f} GB/token); "
                 f"draft reads {self.draft_bytes / 1e9:.2f} GB"]
        p_avg = sum(plain) / len(plain) if plain else 0
        if plain:
            lines.append(f"  plain rounds: {len(plain)} x {p_avg / 1e9:.2f} GB/sweep")
        for k in sorted(by_k):
            v = by_k[k]
            ratio = (sum(v) / len(v) / p_avg) if p_avg else float("nan")
            lines.append(f"  k={k} rounds: {len(v)} x {sum(v) / len(v) / 1e9:.2f} GB/sweep"
                         f" = x{ratio:.2f} plain" if p_avg else
                         f"  k={k} rounds: {len(v)} x {sum(v) / len(v) / 1e9:.2f} GB/sweep")
        return "\n".join(lines)


def ngram_propose(tokens: list[int], k: int, max_ngram: int = 6, min_ngram: int = 2) -> list[int]:
    """F11 zero-model proposal: find the longest recent n-gram suffix that occurred
    earlier in the context and propose the tokens that followed it. Free (no draft
    model, no RAM); acceptance is high exactly when text is repetitive — code,
    lists, quotations — and zero cost when there is no match."""
    L = len(tokens)
    for n in range(min(max_ngram, L - 1), min_ngram - 1, -1):
        suffix = tokens[-n:]
        for i in range(L - n - 1, -1, -1):
            if tokens[i : i + n] == suffix:
                cont = tokens[i + n : i + n + k]
                if cont:
                    return cont
    return []


def serial_position_verifier_supported(cfg) -> bool:
    """Whether a verify window can retain one-token arithmetic layer-major."""
    return (
        not int(getattr(cfg, "num_experts", 0) or 0)
        or getattr(cfg, "model_type", None) in (
            "glm_moe_dsa",
            "glm5_next",
            "kimi_k25",
            "glm4_moe_lite",
            "kimi_linear",
            "kimi_k3",
        )
    )


def _glm5_kda_rollback_factors_required(cfg, verify_width: int) -> bool:
    """Only a speculative suffix needs a fork/factor rollback boundary.

    Width one is the controller's ordinary autoregressive fallback. Its one
    canonical KDA update is retained directly and has no rejected lane to
    reconstruct.
    """
    return (
        getattr(cfg, "model_type", None) == "glm5_next"
        and int(verify_width) > 1
    )


def _native_mtp_strict_prefix_cache(
        cache, ids: list[int], *, target_stepped: bool,
        draft_stepped: bool):
    """Return a compatible immutable native-MTP strict-prefix boundary."""
    if cache is None:
        return None
    cached_key = cache[0]
    cached_ids, cached_target_stepped, cached_draft_stepped = cached_key
    if (cached_target_stepped != target_stepped
            or cached_draft_stepped != draft_stepped
            or len(cached_ids) >= len(ids)
            or tuple(ids[:len(cached_ids)]) != cached_ids):
        return None
    return cache


class SpeculativeDecoder:
    def __init__(self, target: StreamingEngine, draft: StreamingEngine | None, k: int = 6,
                 min_tokens_per_sweep: float | None = None,
                 prompt_cache_min_tokens: int = 0,
                 _unsafe_allow_moe_verify: bool = False):
        """draft=None -> prompt-lookup mode (F11): proposals come from n-gram
        matches in the running context instead of a draft model.

        _unsafe_allow_moe_verify: F113 (2026-07-25) escape hatch for tests
        ONLY -- bypasses the MoE/hybrid-target construction guard below.
        Real usage against real weights has a CONFIRMED correctness bug at
        this configuration (see the guard's own comment); the only
        legitimate caller is tests/test_f32_rollback.py, which exercises
        accept/reject/KV-rollback BOOKKEEPING against a tiny synthetic
        fixture with forced/fake draft outputs -- small enough that the
        batched-GEMM numerical divergence this guard protects against does
        not manifest, so the bookkeeping arithmetic itself is still validly
        tested. Never pass this for a real checkpoint.

        F01 acceptance controller: a multi-position MoE verify sweep can cost far
        more routed-expert bytes than a plain token. The present ~2.0-2.6x range is
        inferred from aggregate runs, not per-round physical-byte telemetry, so the
        threshold is a conservative heuristic rather than a proved break-even. It
        periodically re-probes after falling back to plain decoding."""
        self.target, self.draft, self.k = target, draft, k
        if draft != "mtp" and getattr(target.cfg, "model_type", None) in (
                "qwen3_5", "qwen3_5_moe", "kimi_linear", "kimi_k3"):
            # F94: KVCache.trim() has no kda_cache branch, so a partially
            # accepted draft-model/n-gram round would silently roll back
            # only the ordinary KV, leaving the DeltaNet/KDA recurrent state
            # polluted by the rejected suffix with no error raised. GLM's
            # own MTP path (draft == "mtp") is exempt: its target never has
            # kda_cache. Fail closed rather than silently corrupt output --
            # see runtime/qwen35_mtp.py's QwenMTPSpeculativeEngine for the
            # real fix this gap needs (fork/restore at a clean round
            # boundary), which is architecturally separate from this class.
            raise ValueError(
                f"SpeculativeDecoder does not support recurrent-state "
                f"targets ({target.cfg.model_type}) with draft={draft!r} -- "
                f"KVCache.trim() cannot roll back kda_cache on partial "
                f"rejection. Use QwenMTPSpeculativeEngine instead.")
        target_glm_family = getattr(target.cfg, "model_type", None) in (
            "glm_moe_dsa", "glm5_next", "kimi_k25", "glm4_moe_lite")
        # F128: kimi_k3 gets the SAME exemption treatment as glm_family
        # below -- forward_tokens_serial_positions gained a real kimi_k3_
        # family per-position dispatch (AttnRes-aware, batching only the
        # per-position-independent _apply_attn_res mixing, never attention/
        # MLP), verified near-identical (float32-rounding-only diff, <1e-3
        # max abs logit diff) against forward_tokens() on real downloaded
        # K3 weights (tests/test_f128_k3_verify_positions_oracle.py).
        target_kimi_k3_family = getattr(target.cfg, "model_type", None) == "kimi_k3"
        # F116 follow-on (2026-07-27, closing the previously-noted gap):
        # kimi_linear's own kimi_family per-position dispatch in
        # forward_tokens_serial_positions (reusing
        # _kimi_linear_attention_residual/_kimi_linear_mlp_residual,
        # already in place since F113's Kimi K3-readiness pass) had never
        # actually been exempted from THIS guard, even though the
        # underlying per-position dispatch it protects against already
        # existed -- an oversight, not a deliberate exclusion. Verified the
        # same way glm_family was: byte-identical greedy argmax tokens
        # (all 6 positions) against true sequential decode, on the real
        # Kimi-Linear-48B-A3B-Instruct checkpoint's full 27-layer stack
        # (tests/test_f116_kimi_linear_verify_positions_oracle.py).
        target_kimi_linear_family = getattr(target.cfg, "model_type", None) == "kimi_linear"
        if not _unsafe_allow_moe_verify and not target_glm_family and not target_kimi_k3_family and not target_kimi_linear_family and (
                getattr(target.cfg, "num_experts", 0) or getattr(
                    target.cfg, "model_type", None) in (
                    "gpt_oss", "qwen3_5", "qwen3_5_moe", "kimi_linear", "kimi_k3")):
            # F113 (2026-07-25): this class's multi-position verify sweep
            # uses forward_tokens_serial_positions() when eligible, but that
            # method explicitly refuses most MoE/hybrid targets (their
            # layer_runner.run_block-based per-layer call has no MoE/hybrid
            # dispatch awareness) -- which forces verification onto plain
            # forward_tokens() instead. That method's own docstring already
            # documents the mechanism: "Batched (L, hidden) GEMMs can choose
            # different reduction kernels from ordinary one-token greedy
            # decode and were observed to move Qwen-7B tokens during
            # speculative verification." Confirmed empirically for
            # draft="mtp" against the real GLM-4.7-Flash checkpoint: a real
            # 6-position batched forward_tokens() verify call produced a
            # DIFFERENT argmax at position 5 (token 15332) than true
            # sequential single-position decoding (token 17664) using the
            # identical real weights -- a real, reproducible correctness
            # violation, not a performance question. Fail closed for every
            # draft mode this class supports (not just "mtp") since the
            # unsafe verify path is the same regardless of where proposals
            # come from.
            #
            # glm_moe_dsa/kimi_k25/glm4_moe_lite/kimi_linear/kimi_k3 are all
            # EXEMPT from this guard: forward_tokens_serial_positions has a
            # real MoE/hybrid-aware per-position dispatch for each of these
            # families, individually verified against true sequential
            # decode on real weights (byte-identical greedy tokens for
            # glm_moe_dsa/kimi_k25's Kimi-K2.5 checkpoint and kimi_linear's
            # own 48B checkpoint; <1e-3 max abs logit diff for kimi_k3,
            # whose real checkpoint's much larger per-layer scale made
            # exact byte-identical float comparison impractical to test at
            # -- see each family's own test file) -- the underlying bug is
            # fixed for these architecture families specifically, not just
            # gated around.
            raise ValueError(
                f"SpeculativeDecoder does not support MoE/hybrid targets "
                f"({target.cfg.model_type}) -- their multi-position verify "
                f"sweep cannot use the numerically-safe "
                f"forward_tokens_serial_positions() path (excluded there "
                f"too), and the batched forward_tokens() fallback has been "
                f"confirmed to diverge from true sequential decode for "
                f"these architectures.")
        if prompt_cache_min_tokens < 0:
            raise ValueError("prompt_cache_min_tokens must be >= 0")
        self.prompt_cache_min_tokens = prompt_cache_min_tokens
        self._prompt_cache = None
        # Native-MTP prompt reuse needs two immutable boundaries: the target's
        # hybrid KV/KDA/DSA state and the appended draft block's dense MLA
        # state. It cannot reuse ``_prompt_cache`` above: that cache transfers
        # mutable ownership and later trims the target KV, while a recurrent
        # GLM target cannot reconstruct an earlier KDA state by slicing.
        self._mtp_prompt_cache = None
        cfg = target.cfg
        # MoE verify cost grows with drafted positions. Aggregate post-fix runs
        # motivated the rough ~0.7 increment, but F01 per-round physical-byte
        # telemetry is still required; scale with k instead of using a flat gate.
        self.min_tps = min_tokens_per_sweep if min_tokens_per_sweep is not None else (
            max(2.0, 0.7 * (k + 1)) if cfg.num_experts else 1.15
        )
        self._recent: list[int] = []  # committed tokens of the last speculative rounds
        self._cooldown = 0  # legacy F01 field (superseded by F48 _choose_k)
        self.PROBE_EVERY = 6
        self.controller_disabled_rounds = 0
        # F48 controller state: decayed per-position acceptance estimate
        self._acc_num = 0.0
        self._acc_den = 0.0
        self._round_ix = 0
        # F48 v2: self-tuned marginal verify cost per drafted position, fitted
        # from THIS run's F01 telemetry (init 0.7 = conservative prior; GLM
        # measured 0.46-0.57, OLMoE 0.30-0.36 on 2026-07-12)
        self._c_est = 0.7
        self._plain_bytes = []
        self.mtp = None
        if draft == "mtp":  # F23: the target's own MTP block is the draft source
            from .glm_mtp import MTPDrafter

            self.mtp = MTPDrafter(target)
            self._mtp_kv = self._new_mtp_kv()
            self.draft = draft = None
        if draft is not None:
            probe = "The quick brown fox, 123!"
            if target.tokenizer.encode(probe).ids != draft.tokenizer.encode(probe).ids:
                raise ValueError("target and draft tokenizers disagree — incompatible pair")

    def _new_mtp_kv(self):
        from .kv_cache import KVCache

        if self.mtp is None:
            return None
        # F65: the appended draft block sits one past the target trunk.  Its
        # GLM-5.3 attention is dense MLA and intentionally does not allocate a
        # second DSA indexer; target verification still uses released DSA.
        kv = KVCache(self.mtp.mtp_layer + 1)
        kv.compressed_mla = True
        return kv

    def clear_prompt_cache(self) -> None:
        self._prompt_cache = None
        self._mtp_prompt_cache = None

    def _choose_k(self) -> int:
        """F48 (MoE-SpeQ-derived local heuristic): adaptive draft length.

        A MoE verify sweep is approximated as 1 + C*k*miss plain-sweep byte units,
        where C ~= 0.7 is fitted from aggregate runs rather than measured per-round
        expert unions, and `miss` is the cumulative cache miss rate.
        Expected commits follow the geometric acceptance model
        E[m+1] = (1 - a^(k+1)) / (1 - a) with `a` the decayed per-position
        acceptance. Pick k maximizing committed-per-byte (k=0 = plain round).
        Dense targets always use full k (verify adds ~no weight bytes).
        v1 caveat: miss rate is cumulative (prefill-biased high early), which
        errs conservative."""
        if not self.target.cfg.num_experts:
            return self.k
        h, ms = self.target.expert_hits, self.target.expert_misses
        miss = ms / (h + ms) if (h + ms) else 1.0
        if self._acc_den <= 1e-6:
            return self.k  # no acceptance signal yet: probe at full k
        a = min(max(self._acc_num / self._acc_den, 0.02), 0.98)
        self._round_ix += 1
        best_k, best_v = 0, 1.0  # plain round baseline: 1 token / 1 sweep-unit
        for kk in range(1, self.k + 1):
            exp_committed = (1 - a ** (kk + 1)) / (1 - a)
            v = exp_committed / (1.0 + self._c_est * kk * miss)
            if v > best_v + 1e-9:
                best_k, best_v = kk, v
        if best_k == 0 and self._round_ix % self.PROBE_EVERY == 0:
            return 1  # periodic probe keeps the acceptance estimate fresh
        return best_k

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None, *,
                 encoded_ids: list[int] | None = None, constraint=None) -> dict:
        tgt, drf, k = self.target, self.draft, self.k
        stats = SpecStats()
        if tgt.cfg.model_type == "glm5_next":
            tgt._glm53_mtp_state_only_prefill_tokens = 0
        eos = set(tgt.cfg.eos_token_ids)
        stop = stop or []

        # StreamingEngine.generate normally owns request profiling, but this
        # decoder intentionally calls the target's lower-level sweeps. Create
        # the same bounded profiler here so speculative verification is not an
        # observability blind spot.
        from . import telemetry

        tgt._request_profiler = (
            telemetry.RequestProfiler(tgt.rc.execution_profile)
            if tgt.rc.execution_profile else None)
        if tgt._request_profiler is not None:
            tgt._request_profiler.set_phase("prefill")
        from .engine import _cache_io_snapshot, _record_cache_io_delta

        request_cache_before = _cache_io_snapshot(tgt)
        expert_pipeline_before = (
            int(getattr(tgt, "_expert_batch_prefetch_submitted", 0) or 0),
            float(getattr(tgt, "_expert_batch_prefetch_wait_s", 0.0) or 0.0),
            float(getattr(tgt, "_expert_batch_prefetch_hidden_s", 0.0) or 0.0),
        )

        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if constraint is not None:
            if self.mtp is None or not all(callable(getattr(
                    constraint, name, None)) for name in (
                        "mask_logits", "accept_token", "fork")):
                raise ValueError(
                    "constraint-aware generic speculation requires native MTP "
                    "and mask_logits/accept_token/fork")
        request_t0 = time.perf_counter()
        tokenize_t0 = time.perf_counter()
        ids = (list(encoded_ids) if encoded_ids is not None
               else list(tgt.tokenizer.encode(prompt).ids))
        tokenize_s = time.perf_counter() - tokenize_t0
        if drf is not None and ids != list(drf.tokenizer.encode(prompt).ids):
            raise ValueError(
                "target and draft tokenizers disagree on this prompt — incompatible pair")
        if (tgt.effective_max_position_embeddings
                and len(ids) + max_tokens > tgt.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={tgt.effective_max_position_embeddings} "
                f"({tgt.rope_profile})")
        if tgt.rc.context_bound and len(ids) + max_tokens > tgt.rc.context_bound:
            raise ValueError(
                f"context_bound={tgt.rc.context_bound} but prompt({len(ids)})"
                f"+max_tokens({max_tokens}) exceeds it")
        if (self.mtp is not None
                and tgt.cfg.model_type != "glm5_next"
                and tgt.cfg.index_topk
                and len(ids) + max_tokens > tgt.cfg.index_topk):
            raise RuntimeError(
                "MTP above index_topk is quarantined until its released dynamic "
                "DSA rule (full indexer on draft step 0, reuse on steps 1+) and "
                "rollback state pass F33"
            )
        # Speculation does not use generate()'s retained-request state. Drop any
        # previous owner before allocating two fresh KVs so a fallback request
        # cannot silently coexist with stale multi-GB prompt state.
        tgt.release_request_state()
        if drf is not None:
            drf.release_request_state()
        tgt._provisional = None
        if self.mtp is not None:
            self._mtp_kv = self._new_mtp_kv()
            self._mtp_h = None
        tgt._true_peak_metal_bytes = mx.get_active_memory()
        if tgt.governor is not None:
            tgt.governor.reset_request_peak(tgt._true_peak_metal_bytes)
        if drf is not None:
            drf._true_peak_metal_bytes = tgt._true_peak_metal_bytes
            if drf.governor is not None:
                drf.governor.reset_request_peak(drf._true_peak_metal_bytes)
        total_positions = len(ids) + max_tokens
        target_stepped = bool(
            tgt.rc.stepped_kv_threshold
            and total_positions > tgt.rc.stepped_kv_threshold)
        draft_stepped = bool(
            drf is not None and drf.rc.stepped_kv_threshold
            and total_positions > drf.rc.stepped_kv_threshold)
        prompt_cache_key = (tuple(ids), target_stepped, draft_stepped)
        cache_eligible = bool(
            drf is not None and self.prompt_cache_min_tokens
            and len(ids) >= self.prompt_cache_min_tokens)
        mtp_cache_eligible = bool(
            self.mtp is not None and self.prompt_cache_min_tokens
            and len(ids) >= self.prompt_cache_min_tokens)
        lookup_t0 = time.perf_counter()
        cached = (
            self._prompt_cache
            if cache_eligible and self._prompt_cache is not None
            and self._prompt_cache[0] == prompt_cache_key
            else None
        )
        mtp_cached = (
            self._mtp_prompt_cache
            if mtp_cache_eligible and self._mtp_prompt_cache is not None
            and self._mtp_prompt_cache[0] == prompt_cache_key
            else None
        )
        mtp_prefix_cached = None
        mtp_prefix_tokens = 0
        if (mtp_cache_eligible and mtp_cached is None
                and tgt.cfg.model_type == "glm5_next"
                and self._mtp_prompt_cache is not None):
            mtp_prefix_cached = _native_mtp_strict_prefix_cache(
                self._mtp_prompt_cache, ids,
                target_stepped=target_stepped,
                draft_stepped=draft_stepped)
            if mtp_prefix_cached is not None:
                mtp_prefix_tokens = len(mtp_prefix_cached[0][0])
        prompt_cache_lookup_s = time.perf_counter() - lookup_t0
        prompt_cache_hit = bool(
            cached is not None or mtp_cached is not None
            or mtp_prefix_cached is not None)
        prompt_cache_exact_hit = cached is not None or mtp_cached is not None
        prompt_cache_matched = (
            len(ids) if prompt_cache_exact_hit else mtp_prefix_tokens)
        mtp_target_prompt_snapshot = None
        mtp_draft_prompt_snapshot = None
        mtp_prompt_h_last = None
        if prompt_cache_hit:
            if cached is not None:
                _, t_kv, d_kv, prompt_last_logits = cached
                # Transfer ownership into this request. A failure must not leave a
                # partially advanced cache eligible for the next request.
                self._prompt_cache = None
            elif mtp_cached is not None:
                (_key, mtp_target_prompt_snapshot,
                 mtp_draft_prompt_snapshot, prompt_last_logits,
                 mtp_prompt_h_last) = mtp_cached
                # The saved boundaries remain immutable. Each request advances
                # independent COW branches, so an exception or rejected suffix
                # cannot poison the next exact hit.
                t_kv = mtp_target_prompt_snapshot.fork()
                d_kv = None
                self._mtp_kv = mtp_draft_prompt_snapshot.fork()
                self._mtp_h = mtp_prompt_h_last
                tgt._h_last = mtp_prompt_h_last
                tgt._h_window = None
            else:
                (_key, cached_target_snapshot,
                 cached_draft_snapshot, _cached_logits,
                 cached_prompt_h_last) = mtp_prefix_cached
                t_kv = cached_target_snapshot.fork()
                d_kv = None
                self._mtp_kv = cached_draft_snapshot.fork()
                self._mtp_h = cached_prompt_h_last
                tgt._h_last = cached_prompt_h_last
                tgt._h_window = None
                prompt_last_logits = None
        else:
            if cache_eligible:
                self._prompt_cache = None
            if mtp_cache_eligible:
                self._mtp_prompt_cache = None
            t_kv = tgt.new_kv(stepped=target_stepped)
            d_kv = (drf.new_kv(stepped=draft_stepped)
                    if drf is not None else None)
            prompt_last_logits = None
        draft_vocab = int(getattr(drf.cfg, "vocab_size", 0)) if drf is not None else 0
        draft_usable = bool(
            drf is not None and draft_vocab > 0
            and all(0 <= token < draft_vocab for token in ids))
        prompt_draft_compatible = draft_usable
        if drf is not None and not draft_usable:
            stats.draft_oov_fallbacks += 1

        prefill_t0 = time.perf_counter()
        if prompt_cache_exact_hit:
            prefill_s = time.perf_counter() - prefill_t0
        elif mtp_prefix_cached is not None:
            suffix_ids = ids[prompt_cache_matched:]
            hidden = tgt._embed(suffix_ids)
            hidden = tgt._layer_stationary_glm5_next_sweep(
                hidden, t_kv, offset=prompt_cache_matched,
                tile_width=tgt.rc.prefill_chunk_size)
            tgt._h_window = hidden
            tgt._h_last = hidden[:, -1:, :]
            self._mtp_h = tgt._h_last
            prompt_last_logits = tgt._final_logits(hidden)
            mx.eval(prompt_last_logits)
            # Cached MTP state ends at draft position P-2. Append pairs
            # (token_P, h_{P-1}) through (token_{N-1}, h_{N-2}).
            extension_h_source = mx.concatenate(
                [cached_prompt_h_last, hidden[:, :-1, :]], axis=1)
            self.mtp.prefill_extension(
                suffix_ids, extension_h_source, self._mtp_kv)
            mtp_target_prompt_snapshot = t_kv.fork()
            mtp_draft_prompt_snapshot = self._mtp_kv.fork()
            mtp_prompt_h_last = tgt._h_last
            prefill_s = time.perf_counter() - prefill_t0
        else:
            if tgt.cfg.model_type == "glm5_next":
                hidden = tgt._embed(ids)
                hidden = tgt._layer_stationary_glm5_next_sweep(
                    hidden, t_kv, offset=0,
                    tile_width=tgt.rc.prefill_chunk_size)
                tgt._h_window = hidden
                tgt._h_last = hidden[:, -1:, :]
                prompt_last_logits = tgt._final_logits(hidden)
            else:
                logits = tgt.forward_tokens(ids, t_kv)  # cold prompt sweep
                prompt_last_logits = logits[-1]
            mx.eval(prompt_last_logits)
            prefill_s = time.perf_counter() - prefill_t0
            if mtp_cache_eligible:
                # Capture the exact prompt endpoint before verification advances
                # any target recurrence. Tensor storage is shared COW; only the
                # small Python/cache-state shells are copied here.
                mtp_target_prompt_snapshot = t_kv.fork()
                mtp_prompt_h_last = tgt._h_last
        stats.verify_s += prefill_s
        prefill_cache_after = _cache_io_snapshot(tgt)
        # Keep the historical synthetic prefill sweep in this counter even on
        # a cache hit so ``sweeps - 1`` remains decode-target sweeps.
        stats.sweeps += 1
        first_logits = (
            tgt._constraint_logits(
                prompt_last_logits, constraint, hidden=tgt._h_last)
            if constraint is not None else prompt_last_logits)
        all_tokens = list(ids) + [int(mx.argmax(first_logits))]
        emitted = [all_tokens[-1]]
        if constraint is not None:
            constraint.accept_token(all_tokens[-1])
        first_token_s = time.perf_counter() - request_t0
        stop_text = None
        matched_stop_sequence = None

        def _stop_match(text: str):
            matches = [(text.find(value), index, value)
                       for index, value in enumerate(stop)
                       if value and text.find(value) != -1]
            return min(matches) if matches else None

        if stop:
            decoded = tgt.tokenizer.decode(emitted)
            match = _stop_match(decoded)
            if match is not None:
                cut, _order, matched_stop_sequence = match
                stop_text = decoded[:cut]
        if on_progress is not None:
            on_progress({
                "phase": "prefill",
                "completed_tokens": len(ids),
                "total_tokens": len(ids),
                "cache_source": (
                    "speculative-memory" if prompt_cache_exact_hit
                    else "speculative-prefix-memory"
                    if mtp_prefix_cached is not None
                    else "speculative-cold"),
            })
        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(tgt.tokenizer, stop)
            if stop_text is None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)
        draft_fed = prompt_cache_matched
        # Number of all_tokens already absorbed by the draft KV.
        decode_t0 = time.perf_counter()
        if tgt._request_profiler is not None:
            tgt._request_profiler.set_phase("decode")

        while (len(emitted) < max_tokens and all_tokens[-1] not in eos
               and stop_text is None
               and not bool(constraint is not None and constraint.completed)):
            # --- propose k tokens: draft model, or F11 prompt-lookup (zero-model) ---
            t0 = time.perf_counter()
            draft_cache = drf.cache if drf is not None else tgt.cache
            b_draft0 = draft_cache.stats.bytes_read  # F01 telemetry
            # A verify round commits up to k proposals plus one target token.
            # Bound k by the caller's remaining output budget so the final
            # round cannot leave target KV/routing state past the returned text.
            k = min(
                self._choose_k(),
                max(0, max_tokens - len(emitted) - 1),
            )  # F48: byte-marginal adaptive draft length
            if k == 0:  # speculation not paying this round — plain step
                self.controller_disabled_rounds += 1
                proposals = []
                stats.draft_s += 0.0
            elif self.mtp is not None:
                # F32: draft from the trunk hidden at the ACCEPTED-prefix end, not
                # the last (possibly rejected) verify position
                h_draft = self._mtp_h if getattr(self, "_mtp_h", None) is not None else tgt._h_last
                if self._mtp_kv.offset == 0 and len(all_tokens) > 2:
                    # F32: synchronize MTP state with the prompt so entry j covers
                    # absolute position j (input token_{j+1} paired with trunk h_j).
                    # Requires the full trunk window from a non-cache-hit prefill.
                    P = tgt._h_window.shape[1]
                    assert P == len(all_tokens) - 1, "MTP prefill needs the full trunk window"
                    self.mtp.prefill(all_tokens[:P], tgt._h_window, self._mtp_kv)
                if (mtp_cache_eligible and mtp_draft_prompt_snapshot is None
                        and not prompt_cache_hit):
                    # Snapshot immediately after prompt synchronization and
                    # before draft_tokens feeds the pending first output.
                    mtp_draft_prompt_snapshot = self._mtp_kv.fork()
                # invariant: MTP inputs fed so far = all_tokens[1:-1] (the last
                # committed token becomes an input only when drafting begins)
                assert self._mtp_kv.offset == len(all_tokens) - 2, \
                    f"MTP KV desync: {self._mtp_kv.offset} vs {len(all_tokens) - 2}"
                draft_constraint = (
                    constraint.fork() if constraint is not None else None)
                draft_args = (
                    {"constraint": draft_constraint}
                    if draft_constraint is not None else {})
                proposals = self.mtp.draft_tokens(
                    h_draft, all_tokens[-1], k, self._mtp_kv,
                    offset=len(all_tokens) - 2,
                    **draft_args,
                )
            elif drf is None:
                proposals = ngram_propose(all_tokens, k)
            elif not draft_usable:
                # A target may emit one of its tokenizer's added IDs even when
                # the smaller Qwen draft has no corresponding embedding row.
                # Once that happens the draft can no longer represent the exact
                # prefix, so use target-only rounds for the rest of this request.
                proposals = []
            else:
                catchup = all_tokens[draft_fed:-1]
                if (any(token < 0 or token >= draft_vocab for token in catchup)
                        or not 0 <= all_tokens[-1] < draft_vocab):
                    draft_usable = False
                    stats.draft_oov_fallbacks += 1
                    proposals = []
                elif catchup:
                    drf.forward_tokens(catchup, d_kv)
                    draft_fed += len(catchup)
                    proposals = []
                else:
                    proposals = []
                if draft_usable:
                    cur = all_tokens[-1]
                    resident = drf.draft_tokens_resident(cur, k, d_kv)
                    if resident is not None:
                        proposals.extend(resident)
                        draft_fed += len(resident)
                        stats.resident_draft_rounds += 1
                        stats.resident_draft_tokens += len(resident)
                    else:
                        for _ in range(k):
                            dl = drf.forward_tokens([cur], d_kv)
                            draft_fed += 1  # cur is now absorbed in draft KV
                            cur = int(mx.argmax(dl[-1]))
                            proposals.append(cur)
            stats.draft_s += time.perf_counter() - t0
            stats.draft_bytes += draft_cache.stats.bytes_read - b_draft0
            k_eff = len(proposals)  # lookup may propose fewer (or zero) tokens

            # --- target verifies all proposals in ONE sweep ---
            base = t_kv.offset  # == len(all_tokens) - 1
            t0 = time.perf_counter()
            b_verify0 = tgt.cache.stats.bytes_read  # F01 telemetry
            tgt.begin_provisional()  # F55: routing stats commit post-acceptance
            verify_tokens = [all_tokens[-1]] + proposals
            glm5_recurrent = tgt.cfg.model_type == "glm5_next"
            glm5_factor_commit = _glm5_kda_rollback_factors_required(
                tgt.cfg, len(verify_tokens))
            glm5_kda_base = (
                t_kv.kda_cache.fork()
                if glm5_factor_commit else None)
            try:
                # F113 (2026-07-25): glm_moe_dsa/kimi_k25/glm4_moe_lite now
                # have real MoE-aware serial-position support (see
                # forward_tokens_serial_positions's own glm_family branch) --
                # route them there too, not just num_experts==0 targets,
                # since the whole reason this dispatch exists is to avoid
                # the batched-GEMM verify divergence confirmed for these
                # architectures.
                if (
                    serial_position_verifier_supported(tgt.cfg)
                    and len(verify_tokens) > 1
                ):
                    logits = tgt.forward_tokens_serial_positions(
                        verify_tokens, t_kv,
                        capture_kda_factors=glm5_recurrent)
                else:
                    logits = tgt.forward_tokens(verify_tokens, t_kv)
            except BaseException:
                # Do not leak a provisional routing buffer into the next request
                # if verification is interrupted or fails.
                tgt._provisional = None
                raise
            stats.verify_s += time.perf_counter() - t0
            round_bytes = tgt.cache.stats.bytes_read - b_verify0
            stats.sweeps += 1
            if constraint is None:
                greedy = [int(g) for g in mx.argmax(logits, axis=-1)]
                m = 0
                while m < k_eff and greedy[m] == proposals[m]:
                    m += 1
                new = proposals[:m] + [
                    greedy[m if m < k_eff else k_eff]]
            else:
                # The target window is computed in one authoritative sweep,
                # but grammar state is inherently sequential. Mask each row at
                # its matching retained hidden position and advance only over
                # tokens that will actually be committed. A rejected proposal
                # never mutates the live grammar past the correction token.
                m = 0
                new = []
                while m < k_eff:
                    target_logits = tgt._constraint_logits(
                        logits[m], constraint,
                        hidden=tgt._h_window[:, m:m + 1, :],
                    )
                    true_token = int(mx.argmax(target_logits))
                    if true_token != proposals[m]:
                        new.append(true_token)
                        constraint.accept_token(true_token)
                        break
                    new.append(proposals[m])
                    constraint.accept_token(proposals[m])
                    m += 1
                    if constraint.completed:
                        break
                if (m == k_eff and not constraint.completed):
                    bonus_logits = tgt._constraint_logits(
                        logits[k_eff], constraint,
                        hidden=tgt._h_window[:, k_eff:k_eff + 1, :],
                    )
                    bonus = int(mx.argmax(bonus_logits))
                    new.append(bonus)
                    constraint.accept_token(bonus)
            stats.proposed += k_eff
            stats.accepted += m
            stats.rounds.append((k_eff, m, round_bytes))  # F01 per-round physical bytes
            # F48 v2: fit the marginal cost from live telemetry
            if k_eff == 0:
                self._plain_bytes.append(round_bytes)
            elif self._plain_bytes:
                p_avg = sum(self._plain_bytes) / len(self._plain_bytes)
                if p_avg > 0:
                    c_obs = max(0.05, (round_bytes / p_avg - 1.0) / k_eff)
                    self._c_est = 0.7 * self._c_est + 0.3 * c_obs
            tgt.commit_provisional(m + 1)  # F55: only committed positions teach heat/predictor

            # --- rollback speculative KV entries past the accepted prefix ---
            t_kv.trim(base + 1 + m)
            if glm5_factor_commit:
                factors = tgt.consume_serial_kda_factors()
                if factors is None or glm5_kda_base is None:
                    raise RuntimeError(
                        "GLM-5.3 verifier did not retain KDA rollback factors")
                if m + 1 < len(verify_tokens):
                    t_kv.kda_cache = factors.commit_prefix(
                        glm5_kda_base, m + 1)
            valid = len(all_tokens) + m  # fed-and-valid positions incl. all_tokens[-1]
            if d_kv is not None:
                if d_kv.offset > valid:
                    d_kv.trim(valid)
                draft_fed = min(draft_fed, valid)
            if self.mtp is not None:
                if self._mtp_kv.offset > valid - 1:
                    self._mtp_kv.trim(valid - 1)  # MTP inputs lag tokens by one
                elif self._mtp_kv.offset == valid - 2:
                    # F32: two round shapes leave the MTP KV one input short
                    # (both caught live by the desync assertion):
                    #   cooldown round (k_eff=0): MTP never fed — sync with the
                    #     just-verified token + the hidden that predicted it;
                    #   full-accept round (m == k_eff): the k-th draft committed
                    #     but was never fed back as an input — sync with it +
                    #     the trunk hidden at the preceding position.
                    # The sync prediction is discarded; cost is one layer call.
                    if k_eff == 0:
                        h_sync, tok_sync = self._mtp_h, all_tokens[-1]
                    else:
                        h_sync, tok_sync = tgt._h_window[:, m - 1 : m, :], proposals[m - 1]
                    self.mtp.draft_tokens(h_sync, tok_sync, 1, self._mtp_kv,
                                          offset=valid - 2)
                # F32: next round drafts from the trunk state at window index m
                # (the hidden that predicted the last committed token)
                self._mtp_h = tgt._h_window[:, m : m + 1, :]
                assert t_kv.offset == base + 1 + m, \
                    f"target KV desync: {t_kv.offset} vs {base + 1 + m}"

            if k_eff > 0:  # F48: update the decayed per-position acceptance.
                # CENSORED estimator (2026-07-12 audit): positions after the
                # first mismatch were never tested — only m successes plus AT
                # MOST one observed failure count as trials. The old
                # denominator (k_eff) treated the untested suffix as failures,
                # biasing acceptance low and under-speculating.
                trials = m + (1 if m < k_eff else 0)
                self._acc_num = 0.8 * self._acc_num + m
                self._acc_den = 0.8 * self._acc_den + trials

            for tok in new:
                all_tokens.append(tok)
                emitted.append(tok)
                if stop:
                    decoded = tgt.tokenizer.decode(emitted)
                    match = _stop_match(decoded)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                if stream_decoder is not None and stop_text is None:
                    delta = stream_decoder.push(emitted)
                    if delta:
                        on_token(delta)
                if (stop_text is not None or tok in eos
                        or len(emitted) >= max_tokens):
                    break
            # A stop/EOS can land before the end of a fully accepted proposal
            # window. Keep only the inputs preceding the final returned token,
            # matching ordinary generation's endpoint-KV contract.
            endpoint = len(ids) + len(emitted) - 1
            if t_kv.offset > endpoint:
                t_kv.trim(endpoint)
            if d_kv is not None and d_kv.offset > endpoint:
                d_kv.trim(endpoint)
                draft_fed = min(draft_fed, endpoint)
            if self.mtp is not None and self._mtp_kv.offset > endpoint - 1:
                self._mtp_kv.trim(endpoint - 1)
            stats.emitted = len(emitted)

        stats.emitted = len(emitted)
        final_text = (stop_text if stop_text is not None
                      else tgt.tokenizer.decode(emitted))
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_t0
        endpoint_kv_bytes = t_kv.nbytes()
        endpoint_kv_positions = t_kv.offset
        prompt_cache_stored = False
        if (cache_eligible and prompt_draft_compatible
                and t_kv.offset >= len(ids)
                and d_kv is not None and d_kv.offset >= len(ids)):
            t_kv.trim(len(ids))
            d_kv.trim(len(ids))
            mx.eval(prompt_last_logits)
            self._prompt_cache = (
                prompt_cache_key, t_kv, d_kv, prompt_last_logits)
            prompt_cache_stored = True
        elif (mtp_cache_eligible and not prompt_cache_exact_hit
                and mtp_target_prompt_snapshot is not None
                and mtp_draft_prompt_snapshot is not None
                and mtp_prompt_h_last is not None):
            self._mtp_prompt_cache = (
                prompt_cache_key,
                mtp_target_prompt_snapshot,
                mtp_draft_prompt_snapshot,
                prompt_last_logits,
                mtp_prompt_h_last,
            )
            prompt_cache_stored = True
        total_s = time.perf_counter() - request_t0
        tgt._note_true_peak()
        if drf is not None:
            drf._note_true_peak()
        if tgt.governor is not None:
            tgt._true_peak_metal_bytes = max(
                tgt._true_peak_metal_bytes,
                tgt.governor.request_peak(),
                mx.get_active_memory(),
            )
        if drf is not None:
            tgt._true_peak_metal_bytes = max(
                tgt._true_peak_metal_bytes,
                drf._true_peak_metal_bytes,
                (drf.governor.request_peak()
                 if drf.governor is not None else 0),
            )
        path_stats = {
            "prompt_cache_exact_hit": int(prompt_cache_exact_hit),
            "prompt_cache_prefix_tokens": prompt_cache_matched,
            "prompt_cache_source": (
                "speculative-memory" if prompt_cache_exact_hit
                else "speculative-prefix-memory"
                if mtp_prefix_cached is not None
                else "speculative-cold"),
            "prompt_cache_kind": (
                "native-mtp" if mtp_cache_eligible else
                "draft-model" if cache_eligible else "disabled"),
            "prompt_cache_lookup_s": prompt_cache_lookup_s,
            "prompt_cache_write_tokens": (
                len(ids) if prompt_cache_stored else 0),
            "prompt_cache_extension_tokens": (
                len(ids) - prompt_cache_matched
                if mtp_prefix_cached is not None else 0),
            "prompt_cache_resident_bytes": (
                mtp_target_prompt_snapshot.nbytes()
                + mtp_draft_prompt_snapshot.nbytes()
                if mtp_cache_eligible
                and mtp_target_prompt_snapshot is not None
                and mtp_draft_prompt_snapshot is not None
                else 0),
            "prompt_tokenize_s": tokenize_s,
            "prompt_snapshot_write_s": 0.0,
            "postgen_snapshot_write_s": 0.0,
            "rope_profile": tgt.rope_profile,
            "effective_context_limit": tgt.effective_max_position_embeddings,
            "speculative_used": 1,
            "speculative_k": self.k,
            "speculative_target_sweeps": max(0, stats.sweeps - 1),
            "speculative_proposed": stats.proposed,
            "speculative_accepted": stats.accepted,
            "speculative_constraint_verified": int(constraint is not None),
            "speculative_acceptance_rate": (
                stats.accepted / stats.proposed if stats.proposed else 0.0),
            "speculative_draft_s": stats.draft_s,
            "speculative_verify_s": stats.verify_s,
            "speculative_verify_decode_s": max(
                0.0, stats.verify_s - prefill_s),
            "speculative_draft_bytes": stats.draft_bytes,
            "speculative_rounds": [list(values) for values in stats.rounds],
            "speculative_draft_oov_fallbacks": stats.draft_oov_fallbacks,
            "speculative_resident_draft_rounds": stats.resident_draft_rounds,
            "speculative_resident_draft_tokens": stats.resident_draft_tokens,
            # F48 (2026-07-25): the adaptive controller's own disabled-round
            # count was tracked (self.controller_disabled_rounds) but never
            # surfaced anywhere a caller could observe it -- no way to tell
            # "the controller correctly backed off N times" from "the
            # controller never backs off" without reading the instance
            # attribute directly. Real gap found while investigating whether
            # GLM's native MTP path was slower because its cost/benefit
            # controller wasn't working -- it was working (backed off half
            # the rounds in that test), but only instance-attribute
            # inspection could show that.
            "speculative_controller_disabled_rounds": self.controller_disabled_rounds,
        }
        request_cache_after = _cache_io_snapshot(tgt)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats,
            after=request_cache_after)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats,
            prefix="prefill_", after=prefill_cache_after)
        _record_cache_io_delta(
            tgt, prefill_cache_after, path_stats,
            prefix="decode_", after=request_cache_after)
        expert_pipeline_after = (
            int(getattr(tgt, "_expert_batch_prefetch_submitted", 0) or 0),
            float(getattr(tgt, "_expert_batch_prefetch_wait_s", 0.0) or 0.0),
            float(getattr(tgt, "_expert_batch_prefetch_hidden_s", 0.0) or 0.0),
        )
        path_stats["expert_batch_prefetch_submitted"] = max(
            0, expert_pipeline_after[0] - expert_pipeline_before[0])
        path_stats["expert_batch_prefetch_wait_s"] = max(
            0.0, expert_pipeline_after[1] - expert_pipeline_before[1])
        path_stats["expert_batch_prefetch_hidden_s"] = max(
            0.0, expert_pipeline_after[2] - expert_pipeline_before[2])
        execution_profile = (
            tgt._request_profiler.result(total_s)
            if tgt._request_profiler is not None else None)
        result = {
            "text": final_text,
            "tokens": emitted,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "first_token_s": first_token_s,
            "total_s": total_s,
            "tok_per_s": ((len(emitted) - 1) / decode_s if len(emitted) > 1 else 0.0),
            "kv_bytes": endpoint_kv_bytes,
            "kv_positions": endpoint_kv_positions,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": (
                "stop_sequence" if stop_text is not None else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": tgt._true_peak_metal_bytes,
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
            "stats": stats,
        }
        if execution_profile is not None:
            result["execution_profile"] = execution_profile
        return result


class SpeculativeEngine:
    """Serving adapter for an exact target plus a smaller proposal model.

    The adapter deliberately uses speculation only for the request shape proved
    locally: greedy generation, optional string stops, and a bounded prompt.
    Target-verified tokens may be returned or streamed through the same stateful
    detokenizer as ordinary generation. Every other shape delegates to the
    ordinary target engine.
    Attribute access is delegated so protocol rendering and telemetry continue
    to see the target checkpoint, tokenizer, config, and execution profile.
    """

    def __init__(self, target: StreamingEngine, draft: StreamingEngine, *,
                 k: int = 6, max_prompt_tokens: int = 2048,
                 prompt_cache_min_tokens: int = 2048):
        if k <= 0:
            raise ValueError("speculative k must be positive")
        if max_prompt_tokens <= 0:
            raise ValueError("speculative max_prompt_tokens must be positive")
        if prompt_cache_min_tokens < 0:
            raise ValueError("prompt_cache_min_tokens must be >= 0")
        self.target = target
        self.draft = draft
        self.decoder = SpeculativeDecoder(
            target, draft, k=k,
            prompt_cache_min_tokens=prompt_cache_min_tokens)
        self.max_prompt_tokens = max_prompt_tokens
        self._speculative_k = k
        self._speculative_draft_dir = Path(draft._model_dir)
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt: str, max_tokens: int,
                         on_token=None, stop=None, on_progress=None,
                         sampling: SamplingParams | None = None,
                         constraint=None) -> dict:
        kwargs = {"on_token": on_token, "stop": stop,
                  "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        # Prefer the target's OWN fail-slow prefill retry (a genuine bound
        # method on the plain StreamingEngine, not a delegated wrapper
        # attribute -- see _engine_generate's docstring in server.py for
        # the equivalent mistake this class's own __getattr__ used to let
        # the server-side caller make one level up).
        target_generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = target_generate(prompt, max_tokens, **kwargs)
        path_stats = result.setdefault("path_stats", {})
        path_stats.update({
            "speculative_enabled": 1,
            "speculative_used": 0,
            "speculative_fallback_reason": reason,
            "speculative_k": self._speculative_k,
        })
        return result

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        request_t0 = time.perf_counter()
        if constraint is not None:
            return self._target_generate(
                "constrained-decoding", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if sampling is not None and not sampling.is_greedy:
            return self._target_generate(
                "stochastic-sampling", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (list(prepared_ids) if prepared_ids is not None
               else list(self.target.tokenizer.encode(prompt).ids))
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, on_token, stop, on_progress,
                sampling, constraint)
        draft_ids = list(self.draft.tokenizer.encode(prompt).ids)
        if ids != draft_ids:
            return self._target_generate(
                "tokenizer-mismatch", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        draft_vocab = int(getattr(self.draft.cfg, "vocab_size", 0))
        if not draft_vocab or any(token < 0 or token >= draft_vocab for token in ids):
            return self._target_generate(
                "draft-vocab", prompt, max_tokens, on_token, stop, on_progress,
                sampling, constraint)

        decode_start = time.perf_counter()
        result = self.decoder.generate(
            prompt, max_tokens, on_token=on_token, stop=stop,
            on_progress=on_progress, encoded_ids=ids)
        adapter_prefix_s = decode_start - request_t0
        result["first_token_s"] += adapter_prefix_s
        result["total_s"] = time.perf_counter() - request_t0
        result["path_stats"]["prompt_tokenize_s"] = (
            float(result["path_stats"].get("prompt_tokenize_s", 0.0))
            + adapter_prefix_s)
        result["path_stats"]["speculative_enabled"] = 1
        return result

    def release_request_state(self):
        self.decoder.clear_prompt_cache()
        self.target.release_request_state()
        self.draft.release_request_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decoder.clear_prompt_cache()
        try:
            self.draft.close()
        finally:
            self.target.close()


class NativeMTPEngine:
    """Explicit target-verified adapter for an appended native NextN block."""

    def __init__(self, target: StreamingEngine, *, k: int = 3,
                 max_prompt_tokens: int = 2048):
        if not 1 <= k <= 5:
            raise ValueError("native MTP depth must be in [1, 5]")
        if not 1 <= max_prompt_tokens <= 65_536:
            raise ValueError(
                "native MTP prompt limit must be in [1, 65536]")
        if target.cfg.model_type != "glm5_next":
            raise ValueError("NativeMTPEngine currently requires GLM-5.3")
        self.target = target
        self.decoder = SpeculativeDecoder(
            target, "mtp", k=k, prompt_cache_min_tokens=1)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self._speculative_k = int(k)
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt, max_tokens: int,
                         on_token=None, stop=None, on_progress=None,
                         sampling: SamplingParams | None = None,
                         constraint=None) -> dict:
        kwargs = {
            "on_token": on_token,
            "stop": stop,
            "on_progress": on_progress,
        }
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = generate(prompt, max_tokens, **kwargs)
        result.setdefault("path_stats", {}).update({
            "glm53_mtp_enabled": 1,
            "glm53_mtp_used": 0,
            "glm53_mtp_fallback_reason": reason,
            "glm53_mtp_depth": self._speculative_k,
            "glm53_mtp_max_prompt_tokens": self.max_prompt_tokens,
            "glm53_mtp_constraint_verified": 0,
        })
        return result

    def generate(self, prompt, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        if constraint is not None and not all(callable(getattr(
                constraint, name, None)) for name in (
                    "mask_logits", "accept_token", "fork")):
            return self._target_generate(
                "unsupported-constraint", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if sampling is not None and not sampling.is_greedy:
            return self._target_generate(
                "stochastic-sampling", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if max_tokens <= 1:
            return self._target_generate(
                "output-budget", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (
            list(prepared_ids) if prepared_ids is not None
            else list(self.target.tokenizer.encode(prompt).ids))
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)

        result = self.decoder.generate(
            prompt, max_tokens, on_token=on_token, stop=stop,
            on_progress=on_progress, encoded_ids=ids,
            constraint=constraint)
        result.setdefault("path_stats", {}).update({
            "glm53_mtp_enabled": 1,
            "glm53_mtp_used": 1,
            "glm53_mtp_fallback_reason": "",
            "glm53_mtp_depth": self._speculative_k,
            "glm53_mtp_max_prompt_tokens": self.max_prompt_tokens,
            "glm53_mtp_constraint_verified": int(constraint is not None),
            "glm53_mtp_state_only_prefill_tokens": int(getattr(
                self.target, "_glm53_mtp_state_only_prefill_tokens", 0)),
        })
        return result

    def release_request_state(self):
        self.decoder.clear_prompt_cache()
        self.target.release_request_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decoder.clear_prompt_cache()
        self.target.close()
