"""F11: prompt-lookup (n-gram) speculative decoding for qwen3_5's hybrid
DeltaNet/full-attention layers.

Mirrors qwen35_mtp.py's QwenMTPSpeculativeEngine exactly for the
safety-critical part: the generic SpeculativeDecoder/KVCache.trim() path
is NOT safe here (trim() has no kda_cache branch at all -- a partial
reject would silently corrupt the DeltaNet recurrent state, see
qwen35_mtp.py's module docstring for the full explanation and the
fork()/restore checkpoint pattern this reuses).

Unlike MTP (a real small transformer-layer forward pass per draft), n-gram
proposal is free: it looks up the longest recent repeated substring in the
token history and reuses its known continuation (runtime/speculative.py's
existing ngram_propose, F11's own zero-model mechanism), costing nothing
extra when there is no match, and able to propose SEVERAL tokens at once
(unlike MTP's fixed k=1) -- which matters because this project's own
decode profiling (2026-07-23, STATUS.md) found lossy Qwen3.5-9B decode is
compute-bound with the model fully cache-resident: batching k>1 verified
positions into one forward sweep costs close to the SAME per-layer compute
as a single position, the actual mechanism a speculative scheme needs to
win in a compute-bound regime -- unlike MTP's k=1, where the extra
drafter-layer cost ate the gain (measured this session: a wash, sometimes
slightly slower). This project's agentic/tool-calling workload (repeated
tool-schema text, echoed function signatures/arguments) is exactly the
high-n-gram-overlap regime prompt-lookup decoding is documented to help
most (see docs/future_lossless_techniques.md F11).
"""

from __future__ import annotations

import time

import mlx.core as mx

from .kv_cache import KVCache
from .sampler import SamplingParams
from .speculative import ngram_propose


class QwenNgramSpeculativeEngine:
    """Serving adapter, mirroring QwenMTPSpeculativeEngine's shape exactly.
    Falls back to the plain target engine for any request shape the
    greedy-only verified scheme doesn't cover."""

    def __init__(self, target, max_prompt_tokens: int = 8192,
                 k: int = 8, max_ngram: int = 6, min_ngram: int = 2):
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if min_ngram <= 0 or max_ngram < min_ngram:
            raise ValueError("max_ngram must be >= min_ngram > 0")
        self.target = target
        self.max_prompt_tokens = max_prompt_tokens
        self.k = k
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt, max_tokens, on_token,
                          stop, on_progress, sampling, constraint) -> dict:
        kwargs = {"on_token": on_token, "stop": stop, "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        target_generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = target_generate(prompt, max_tokens, **kwargs)
        path_stats = result.setdefault("path_stats", {})
        path_stats.update({
            "qwen_ngram_enabled": 1,
            "qwen_ngram_used": 0,
            "qwen_ngram_fallback_reason": reason,
        })
        return result

    def generate(self, prompt, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        if sampling is not None and not sampling.is_greedy:
            return self._target_generate(
                "stochastic-sampling", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        tgt = self.target
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (list(prepared_ids) if prepared_ids is not None
               else list(tgt.tokenizer.encode(prompt).ids))
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if (tgt.effective_max_position_embeddings
                and len(ids) + max_tokens > tgt.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={tgt.effective_max_position_embeddings} "
                f"({tgt.rope_profile})")

        request_t0 = time.perf_counter()
        tgt.release_request_state()
        eos = set(tgt.cfg.eos_token_ids)
        stop = stop or []
        def _verify_forward(tokens, active_kv):
            # Let small multi-position verify/refeed sweeps use the resident
            # fast path (engine._sweep's fast_decode_eligible) -- they are
            # decode-shaped work, and paying the ordinary per-layer sync
            # loop for them inverts speculation's economics. Scoped to each
            # call so the hint can never leak into prefill or a later
            # plain request.
            tgt._speculative_verify_hint = True
            try:
                return tgt.forward_tokens(tokens, active_kv)
            finally:
                tgt._speculative_verify_hint = False

        kv = tgt.new_kv()
        prefill_t0 = time.perf_counter()
        try:
            logits = tgt.forward_tokens(ids, kv)
            prompt_last_logits = logits[-1]
            mx.eval(prompt_last_logits)
        except MemoryError:
            tgt.release_request_state()
            return self._target_generate(
                "memory-pressure-fallback", prompt, max_tokens, on_token,
                stop, on_progress, sampling, constraint)
        prefill_s = time.perf_counter() - prefill_t0

        proposed = 0
        accepted = 0
        forced_committed = 0
        rounds = 0
        sweeps = 1  # count the prefill sweep, matching SpeculativeDecoder's convention

        # Grammar-aware speculation (2026-07-23, v2): constraints are no
        # longer a fallback. The matcher advances ACCEPT-AS-YOU-VERIFY --
        # only tokens that pass masked-argmax verification are ever accepted
        # into the grammar, so no matcher rollback is needed on rejection --
        # and F98's forced runs fold into each round as a zero-verification
        # prefix (grammar-deterministic tokens need feeding, not deciding).
        grammar_completed = False
        encode_cb = (
            (lambda text: tgt.tokenizer.encode(text).ids)
            if getattr(tgt.rc, "grammar_jump_forward_lossy", False) else None)

        def _masked(position_logits):
            if constraint is not None and not grammar_completed:
                return constraint.mask_logits(position_logits)
            return position_logits

        catchup_tok = int(mx.argmax(_masked(prompt_last_logits)))
        if constraint is not None:
            constraint.accept_token(catchup_tok)
            grammar_completed = bool(constraint.completed)
        all_tokens = list(ids) + [catchup_tok]
        emitted = [catchup_tok]
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
                "phase": "prefill", "completed_tokens": len(ids),
                "total_tokens": len(ids), "cache_source": "qwen-ngram-cold",
            })
        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(tgt.tokenizer, stop)
            if stop_text is None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)

        decode_t0 = time.perf_counter()

        def _commit(tok: int) -> bool:
            """Append one committed token; True when generation must end."""
            nonlocal stop_text, matched_stop_sequence
            all_tokens.append(tok)
            emitted.append(tok)
            if stop:
                decoded = tgt.tokenizer.decode(emitted)
                match = _stop_match(decoded)
                if match is not None:
                    cut, _order, matched_stop_sequence = match
                    stop_text = decoded[:cut]
                    return True  # never streamed, matching the plain path
            if stream_decoder is not None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)
            return tok in eos or len(emitted) >= max_tokens

        while (len(emitted) < max_tokens and catchup_tok not in eos
               and stop_text is None and not grammar_completed):
            round_start_offset = kv.offset
            rounds += 1

            # Checkpoint before ANY speculative advance -- cheap (no array
            # copies, see kda_state.py). Also the terminal-repair anchor:
            # DeltaNet state cannot be trimmed like ordinary KV, so any
            # round that ends generation mid-feed restores this fork and
            # re-feeds exactly the committed tokens (final-token-never-fed
            # contract), instead of leaving overfed recurrent state behind.
            kda_checkpoint = (
                kv.kda_cache.fork() if getattr(kv, "kda_cache", None) is not None
                else None)

            # F98 folded in: grammar-forced tokens are deterministic --
            # commit them immediately and let this round's single sweep
            # carry their state updates as a prefix.
            forced: list[int] = []
            if constraint is not None and not grammar_completed:
                forced = constraint.forced_run(
                    max(0, max_tokens - len(emitted)), encode=encode_cb)
                grammar_completed = bool(constraint.completed)
            forced_committed += len(forced)
            committed_round: list[int] = []
            terminal = False
            for tok in forced:
                committed_round.append(tok)
                if _commit(tok):
                    terminal = True
                    break
            if grammar_completed:
                terminal = True

            proposals: list[int] = []
            bonus_tok = None
            m = 0
            if not terminal:
                k = min(self.k, max(0, max_tokens - len(emitted) - 1))
                proposals = (
                    ngram_propose(all_tokens, k, self.max_ngram, self.min_ngram)
                    if k > 0 else [])
                k_eff = len(proposals)
                proposed += k_eff

                verify_tokens = [catchup_tok] + forced + proposals
                spec_logits = _verify_forward(verify_tokens, kv)
                sweeps += 1

                # logits[len(forced) + i] is the distribution for
                # proposals[i]'s slot; the matcher advances only over tokens
                # that verification actually commits (accept-as-you-verify).
                base = len(forced)
                while True:
                    true_tok = int(mx.argmax(_masked(spec_logits[base + m])))
                    if m < k_eff and true_tok == proposals[m]:
                        if constraint is not None:
                            constraint.accept_token(true_tok)
                            if constraint.completed:
                                grammar_completed = True
                                m += 1
                                break
                        m += 1
                        continue
                    bonus_tok = true_tok
                    if constraint is not None:
                        constraint.accept_token(bonus_tok)
                        grammar_completed = bool(constraint.completed)
                    break
                accepted += m

                if m < k_eff:
                    # Rejected tail: restore recurrent state to the
                    # pre-round fork, roll ordinary KV back, and re-feed
                    # exactly the accepted prefix so both halves of the
                    # hybrid state reflect precisely the committed tokens.
                    # bonus_tok stays valid: it came from the same pass,
                    # and re-feeding the identical prefix reproduces that
                    # exact trunk state by construction.
                    if kda_checkpoint is not None:
                        kv.kda_cache = kda_checkpoint
                    kv.trim(round_start_offset)
                    refeed = [catchup_tok] + forced + proposals[:m]
                    refeed_logits = _verify_forward(refeed, kv)
                    mx.eval(refeed_logits)
                    sweeps += 1

                for tok in proposals[:m]:
                    committed_round.append(tok)
                    if _commit(tok):
                        terminal = True
                        break
                if not terminal and bonus_tok is not None:
                    committed_round.append(bonus_tok)
                    if _commit(bonus_tok):
                        terminal = True
                    catchup_tok = bonus_tok
                elif not terminal and grammar_completed:
                    terminal = True

            if terminal or grammar_completed:
                # Terminal repair (final-token-never-fed contract): kv must
                # end holding prompt + emitted[:-1]. Whatever this round fed
                # (nothing yet, or catchup+forced+proposals) is replaced by
                # exactly [catchup] + committed_round[:-1] from the fork.
                if kda_checkpoint is not None:
                    kv.kda_cache = kda_checkpoint
                if kv.offset > round_start_offset:
                    kv.trim(round_start_offset)
                repair = [emitted[len(emitted) - len(committed_round) - 1]] \
                    + committed_round[:-1] if committed_round else []
                if repair:
                    repair_logits = _verify_forward(repair, kv)
                    mx.eval(repair_logits)
                    sweeps += 1
                break

        final_text = stop_text if stop_text is not None else tgt.tokenizer.decode(emitted)
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_t0
        endpoint = len(ids) + len(emitted) - 1
        if kv.offset > endpoint:
            kv.trim(endpoint)
        total_s = time.perf_counter() - request_t0
        path_stats = {
            "prompt_tokenize_s": 0.0,
            "rope_profile": tgt.rope_profile,
            "effective_context_limit": tgt.effective_max_position_embeddings,
            "qwen_ngram_enabled": 1,
            "qwen_ngram_used": 1,
            "qwen_ngram_target_sweeps": max(0, sweeps - 1),
            "qwen_ngram_rounds": rounds,
            "qwen_ngram_proposed": proposed,
            "qwen_ngram_accepted": accepted,
            "grammar_fast_forward_tokens": forced_committed,
        }
        return {
            "text": final_text,
            "tokens": emitted,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "first_token_s": first_token_s,
            "total_s": total_s,
            "tok_per_s": ((len(emitted) - 1) / decode_s if len(emitted) > 1 else 0.0),
            "kv_bytes": kv.nbytes(),
            "kv_positions": kv.offset,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": (
                "stop_sequence" if stop_text is not None else
                "eos" if emitted[-1] in eos else
                "grammar" if grammar_completed else "length"),
            "true_peak_metal_bytes": mx.get_active_memory(),
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
        }
