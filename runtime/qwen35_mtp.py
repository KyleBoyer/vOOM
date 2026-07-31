"""F94: Qwen3.5/3.6 native MTP (multi-token prediction) as a verified draft
source for speculative decoding.

Checkpoint block under the top-level ``mtp.`` prefix (DeepSeek-V3 style,
mtp_num_hidden_layers=1, applied ONCE -- unlike GLM-5.2's MTP, which chains
iteratively for up to 5 drafts, Qwen3.6's real checkpoints only ever draft
ONE token ahead):

    e   = pre_fc_norm_embedding( embed(token_t) )
    hn  = pre_fc_norm_hidden( h )                  # h = trunk hidden at t-1
    x   = fc( [ e | hn ] )                          # 2*hidden -> hidden
    h'  = mtp.layers.0( x )                          # ordinary decoder layer:
                                                      # full attention (gated
                                                      # QK-RMSNorm, partial
                                                      # RoPE) + dense MLP --
                                                      # NOT DeltaNet, confirmed
                                                      # by tensor shapes
                                                      # matching qwen35.py's
                                                      # _full_attention/_swiglu
                                                      # exactly.
    logit = lm_head( mtp.norm(h') )                  # lm_head is SHARED with
                                                      # the trunk (no separate
                                                      # mtp.lm_head tensor on
                                                      # disk; mtp_use_dedicated_
                                                      # embeddings=False confirms
                                                      # the embedding table is
                                                      # shared too).
    draft = argmax(logit)

Why the recurrent-state rollback problem this blocked on is tractable here:
the TRUNK's DeltaNet/gated-linear-attention layers (KDAStateCache) update
destructively each token, so a rejected draft can't simply be "trimmed" the
way ordinary KV can (see runtime/kv_cache.py::KVCache.trim, which has no
kda_cache branch -- a real, separate, NOT-currently-reachable gap in the
generic suffix_decoding/speculative.py draft paths, guarded against
elsewhere rather than fixed here). But Qwen3.6 only ever drafts ONE token
(mtp_num_hidden_layers=1), so the rollback problem collapses to picking
between exactly two states -- no chain of K checkpoints, no compact
transition-factor math (contra the general case solved by SpecLA, arXiv
2607.16673). KDAStateCache.fork() (kda_state.py) is already a cheap,
existing snapshot primitive (list-shallow-copy, no array copies) with one
prior caller (kv_cache.py's fork_hybrid_kv_endpoint, used by qwen3vl.py).

Round structure (matching speculative.py's own "1 + k" verify convention,
k=1 here): each round feeds ``[catchup_token, draft_token]`` through one
layer-major sweep. Every position retains the ordinary one-token arithmetic
shape inside that sweep, so MoE routing never widens into a two-position union
and MXFP4 reductions remain identical to sequential decode. The verifier also
retains the recurrent KDA endpoint after the catchup position. A rejection can
therefore trim ordinary KV to that midpoint and install the retained KDA
endpoint directly; it never restores the pre-round state and re-feeds the
catchup token through another full weight sweep. This is the difference
between speculation whose misses cost an extra streamed pass and speculation
whose misses merely fail to save one.

This module mirrors runtime/glm_mtp.py's structure and safety framing:
greedy drafts require target argmax equality, while stochastic drafts use the
standard exact p/q acceptance ratio and normalized positive-part (p-q)
rejection residual. This is deliberately a NEW,
SIMPLER engine (not a SpeculativeDecoder extension) because k is always 1
here -- SpeculativeDecoder's adaptive-k controller, F48 byte-fitted
telemetry, and multi-round bookkeeping are irrelevant complexity for a fixed
k=1 target, and this keeps the already-provisional GLM path untouched.
"""

from __future__ import annotations

import math
import time

import mlx.core as mx

from . import quant
from .kv_cache import KVCache
from .qwen35 import _full_attention, _moe, _swiglu, final_logits, qwen35_rms_norm
from .sampler import (SamplingParams, filtered_probabilities, sample,
                      sample_probabilities,
                      speculative_residual_probabilities)


def _adaptive_mtp_should_disable(
    proposed: int, accepted: int, probe_rounds: int,
) -> bool:
    """True only when a complete initial probe accepted no draft tokens."""
    return proposed >= probe_rounds and accepted == 0


def _adaptive_stochastic_mtp_should_disable(
    proposed: int,
    expected_acceptance_sum: float,
    probe_rounds: int,
    min_expected_acceptance: float = 0.05,
) -> bool:
    """Use exact p/q overlap, not random outcomes, to stop a bad draft."""
    return (
        proposed >= probe_rounds
        and expected_acceptance_sum / proposed < min_expected_acceptance
    )


def _verify_stochastic_mtp_token(
    proposal: int,
    draft_probabilities: mx.array,
    target_logits: mx.array,
    sampling: SamplingParams,
    history: list[int],
) -> tuple[bool, int, mx.array]:
    """Leviathan rejection correction for Qwen's one-token MTP draft.

    Returns ``(accepted, committed_token, target_distribution)``.  The
    target distribution is exposed so callers can retain the exact
    authoritative position telemetry without recomputing its filters.
    """
    target_probabilities = filtered_probabilities(
        target_logits, sampling, history=history).reshape(-1)
    draft_probabilities = draft_probabilities.reshape(-1)
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError(
            "Qwen MTP target/draft vocabulary mismatch: "
            f"target={target_probabilities.shape}, "
            f"draft={draft_probabilities.shape}")
    mx.eval(target_probabilities, draft_probabilities)
    p_token = float(target_probabilities[proposal].item())
    q_token = float(draft_probabilities[proposal].item())
    ratio = 1.0 if q_token <= 0 else min(1.0, p_token / q_token)
    if float(mx.random.uniform().item()) <= ratio:
        return True, int(proposal), target_probabilities
    replacement = sample_probabilities(
        speculative_residual_probabilities(
            target_probabilities, draft_probabilities))
    return False, replacement, target_probabilities


def _flat_top_k_draft_probabilities(
    draft_logits: mx.array,
    sampling: SamplingParams,
    history: list[int],
    top_k: int,
    constraint=None,
) -> mx.array:
    """Build q, tolerating disjoint sparse-head and grammar support.

    q may propose outside the target constraint without changing the target
    distribution: such proposals have p=0 and speculative rejection samples
    the normalized positive part of p-q.
    """
    # final_logits preserves the leading singleton batch dimension, whereas
    # the serial target verifier returns a rank-1 row. Normalize both the
    # proposal filter and q to one vocabulary axis before indexing tokens.
    values = draft_logits.reshape(-1)
    authoritative = (
        constraint.mask_logits(values)
        if constraint is not None else values
    )
    calibrated = filtered_probabilities(
        authoritative, sampling, history=history)

    def flattened(probabilities: mx.array) -> tuple[mx.array, mx.array]:
        selected_k = min(top_k, int(probabilities.size))
        ranked = mx.argsort(probabilities)[::-1]
        support = ranked[:selected_k]
        positive = (probabilities[support] > 0).astype(mx.float32)
        return support, positive

    support, positive = flattened(calibrated)
    support_mass = mx.sum(positive)
    mx.eval(support_mass)
    support_mass_value = float(support_mass.item())
    if (
        (not math.isfinite(support_mass_value) or support_mass_value <= 0)
        and constraint is not None
    ):
        calibrated = filtered_probabilities(
            values, sampling, history=history)
        support, positive = flattened(calibrated)
        support_mass = mx.sum(positive)
        mx.eval(support_mass)
        support_mass_value = float(support_mass.item())
    if not math.isfinite(support_mass_value) or support_mass_value <= 0:
        # A numerically empty draft is still not a serving failure. Uniform q
        # is valid for Leviathan verification and will almost certainly be
        # rejected; the adaptive probe then disables the unhelpful drafter.
        # This keeps the target distribution exact while guaranteeing forward
        # progress for unfamiliar head/constraint combinations.
        return mx.full(
            calibrated.shape, 1.0 / int(calibrated.size), dtype=mx.float32)
    # MLX's indexed ``.at[vector].add`` currently treats this one-dimensional
    # scatter like a reduction on this platform (two distinct indices yielded
    # [1, 0, ...], not [0.5, 0.5, ...]). put_along_axis expresses the required
    # one-to-one write and is covered by the distribution test below.
    return mx.put_along_axis(
        mx.zeros(calibrated.shape, dtype=mx.float32),
        support,
        positive / support_mass,
        axis=0,
    )


class QwenMTPDrafter:
    def __init__(self, engine):
        self.engine = engine
        names = engine.store.names_with_prefix("mtp.")
        if not names:
            raise ValueError(
                "QwenMTPDrafter requires a checkpoint with mtp.* weights")
        # MoE checkpoints (e.g. Qwen3.6-35B-A3B) shape the MTP layer's MLP
        # exactly like a trunk MoE layer (mtp.layers.0.mlp.{gate,
        # shared_expert,shared_expert_gate,experts.N.*}); dense checkpoints
        # (Qwen3.6-27B) use a plain SwiGLU (mtp.layers.0.mlp.{gate,up,down}_proj).
        # Real tensor names confirmed directly against both released
        # checkpoints' safetensors indices, not inferred.
        self._page_names = [n for n in names if ".mlp.experts." not in n]

    def _weights(self) -> dict:
        return self.engine.cache.get("qwen35_mtp", self._page_names)

    def _get_experts(self, layer: int, expert_ids: list[int],
                      positions: dict[int, list[int]] | None = None) -> dict[int, dict]:
        """engine._get_experts hardcodes 'model.layers.{layer}.{prefix}.' --
        wrong location for mtp.layers.0's experts, so this is a small local
        duplicate (same reasoning as kimi_linear.py's _route_experts: a
        different weight prefix, not different math). No heat/prefetch
        tracking or governor.reserve() -- a single draft touches at most
        topk+1 tiny expert pages, not worth that machinery."""
        items = [
            (f"mtp_expert.{e}",
             self.engine.store.names_with_prefix(f"mtp.layers.0.mlp.experts.{e}."))
            for e in expert_ids
        ]
        pages = self.engine.cache.get_many(items)
        return {e: pages[f"mtp_expert.{e}"] for e in expert_ids}

    def draft_logits(
        self, h_last: mx.array, last_token: int, mtp_kv, offset: int,
    ) -> mx.array:
        """h_last: (1, 1, hidden) trunk hidden (pre final-norm) at position
        offset-1 (i.e. the state that produced last_token). Returns the full
        draft-logit vector for position offset+1. `offset` is the ABSOLUTE
        sequence position of last_token (matching the trunk's own kv.offset
        convention, not a decode-session-local counter) -- RoPE inside this
        MTP layer must see real positions or acceptance rate silently
        degrades (never correctness: every draft is exactly re-verified
        against the trunk regardless of how it was positioned). mtp_kv
        accumulates the MTP block's own (ordinary, non-recurrent) KV --
        it is plain attention, so it never needs rollback: an unaccepted
        draft's MTP-KV entry is harmless history for future drafts, not a
        source of wrong output the way trunk kda_cache pollution would be."""
        eng = self.engine
        cfg = eng.cfg
        w = self._weights()
        e = eng._embed([last_token])  # (1, 1, hidden), row-paged when enabled
        e = qwen35_rms_norm(e, w["mtp.pre_fc_norm_embedding.weight"], cfg.rms_norm_eps)
        hn = qwen35_rms_norm(h_last, w["mtp.pre_fc_norm_hidden.weight"], cfg.rms_norm_eps)
        x = quant.matmul(mx.concatenate([e, hn], axis=-1), w["mtp.fc.weight"])
        # Standard decoder-layer residual wiring (same shape as
        # run_qwen35_block's full_attention branch) -- mtp.layers.0 isn't
        # part of cfg.layer_types, so this reimplements that one block's
        # wiring directly rather than routing through run_qwen35_block's
        # layer_types[layer] dispatch.
        residual = x
        h = qwen35_rms_norm(x, w["mtp.layers.0.input_layernorm.weight"], cfg.rms_norm_eps)
        attn = _full_attention(h, w, "mtp.layers.0", cfg, mtp_kv, 0, offset)
        x = residual + attn
        residual = x
        h = qwen35_rms_norm(
            x, w["mtp.layers.0.post_attention_layernorm.weight"], cfg.rms_norm_eps)
        if not cfg.num_experts:
            x = residual + _swiglu(h, w, "mtp.layers.0.mlp")
        else:
            x = residual + _moe(
                h, w, "mtp.layers.0", cfg, 0, self._get_experts)
        logits = final_logits(
            x, w["mtp.norm.weight"], eng._lm_head_weight(), cfg.rms_norm_eps)
        mx.eval(logits)
        return logits[-1]

    def draft_token(
        self, h_last: mx.array, last_token: int, mtp_kv, offset: int,
    ) -> int:
        return int(mx.argmax(self.draft_logits(
            h_last, last_token, mtp_kv, offset)))


class QwenMTPSpeculativeEngine:
    """Serving adapter, mirroring SpeculativeEngine's shape: falls back to
    the plain target engine for any request shape the (target-exact, k=1)
    verified-draft scheme doesn't cover. Attribute access delegates to the
    target so protocol rendering/telemetry see the real checkpoint,
    tokenizer, config, and execution profile."""

    def __init__(
        self, target, max_prompt_tokens: int = 32768,
        min_output_tokens: int = 32, adaptive_stop: bool = True,
        adaptive_probe_rounds: int = 3, plain_warmup_tokens: int = 3,
        stochastic_draft_top_k: int = 4,
    ):
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if min_output_tokens <= 1:
            raise ValueError("min_output_tokens must be greater than one")
        if adaptive_probe_rounds <= 0:
            raise ValueError("adaptive_probe_rounds must be positive")
        if plain_warmup_tokens < 0:
            raise ValueError("plain_warmup_tokens must be non-negative")
        if stochastic_draft_top_k <= 0:
            raise ValueError("stochastic_draft_top_k must be positive")
        self.target = target
        self.drafter = QwenMTPDrafter(target)
        self.max_prompt_tokens = max_prompt_tokens
        self.min_output_tokens = min_output_tokens
        self.adaptive_stop = bool(adaptive_stop)
        self.adaptive_probe_rounds = int(adaptive_probe_rounds)
        self.plain_warmup_tokens = int(plain_warmup_tokens)
        self.stochastic_draft_top_k = int(stochastic_draft_top_k)
        if (
            getattr(target.cfg, "num_experts", 0)
            and not callable(getattr(
                target, "forward_tokens_serial_positions", None))
        ):
            raise ValueError(
                "MoE Qwen MTP requires serial-position target verification")

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt, max_tokens, on_token,
                          stop, on_progress, sampling, constraint) -> dict:
        kwargs = {"on_token": on_token, "stop": stop, "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        # Prefer the target's OWN fail-slow prefill retry (a genuine bound
        # method on the plain StreamingEngine, not a delegated wrapper
        # attribute -- see _engine_generate's docstring in server.py for the
        # bug this class used to trigger via the opposite mistake).
        target_generate = getattr(
            self.target, "generate_with_memory_retry", self.target.generate)
        result = target_generate(prompt, max_tokens, **kwargs)
        path_stats = result.setdefault("path_stats", {})
        path_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": 0,
            "qwen_mtp_fallback_reason": reason,
        })
        return result

    def generate(self, prompt, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        sampling = sampling or SamplingParams()
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
        if max_tokens < self.min_output_tokens:
            return self._target_generate(
                "short-output-budget", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
        if (tgt.effective_max_position_embeddings
                and len(ids) + max_tokens > tgt.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={tgt.effective_max_position_embeddings} "
                f"({tgt.rope_profile})")

        # Let the target's ordinary generation path do exactly one thing:
        # safe prefill plus sampling the first (still-unfed) token. This keeps
        # all of StreamingEngine's chunking, layer-stationary scheduling,
        # memory retries, exact hot-prefix reuse, and request telemetry. The
        # old MTP adapter called forward_tokens(ids, kv) directly, bypassing
        # every one of those mechanisms and therefore had to fall back above
        # an arbitrary 8K prompt limit. After this bootstrap, ``last_kv`` is at
        # the same endpoint the MTP loop expects: prompt fully fed, first token
        # sampled but not fed.
        from .engine import _cache_io_snapshot, _record_cache_io_delta

        request_t0 = time.perf_counter()
        request_cache_before = _cache_io_snapshot(tgt)
        eos = set(tgt.cfg.eos_token_ids)
        stop = stop or []
        bootstrap_generate = getattr(
            tgt, "generate_with_memory_retry", tgt.generate)
        bootstrap_kwargs = {
            "on_token": None,
            "stop": stop,
            "on_progress": on_progress,
            "sampling": sampling,
        }
        if constraint is not None:
            bootstrap_kwargs["constraint"] = constraint
        bootstrap = bootstrap_generate(prompt, 1, **bootstrap_kwargs)
        bootstrap_stats = dict(bootstrap.get("path_stats") or {})
        bootstrap_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": 0,
        })
        if (max_tokens == 1
                or bootstrap.get("termination_reason") != "length"):
            reason = (
                "single-token-budget" if max_tokens == 1
                else f"terminal-first-token:{bootstrap.get('termination_reason')}")
            bootstrap_stats["qwen_mtp_fallback_reason"] = reason
            bootstrap["path_stats"] = bootstrap_stats
            if on_token is not None and bootstrap.get("text"):
                on_token(bootstrap["text"])
            return bootstrap

        kv = getattr(tgt, "last_kv", None)
        if kv is None:
            raise RuntimeError(
                "Qwen MTP bootstrap did not retain its exact target KV endpoint")
        decode_cache_before = _cache_io_snapshot(tgt)
        mtp_kv = KVCache(1)
        proposed = 0
        accepted = 0
        target_decode_sweeps = 0
        plain_decode_sweeps = 0
        warmup_decode_sweeps = 0
        warmup_remaining = self.plain_warmup_tokens
        adaptive_disabled = False
        adaptive_probe_limit = self.adaptive_probe_rounds
        serial_verify_rounds = 0
        kda_endpoint_restores = 0
        refeed_sweeps_saved = 0
        grammar_forced_tokens = 0
        grammar_forced_sweeps = 0
        stochastic_expected_acceptance_sum = 0.0
        round_outcomes: list[str] = []

        # Invariant (matching speculative.py's documented one): all_tokens =
        # prompt + emitted; catchup_tok = all_tokens[-1] is sampled but not
        # yet fed to kv (kv.offset == len(all_tokens) - 1) until this
        # round's combined forward call feeds it.
        catchup_tok = int(bootstrap["tokens"][0])
        h_last = tgt._h_last
        all_tokens = list(ids) + [catchup_tok]
        emitted = [catchup_tok]
        prefill_s = float(bootstrap.get("prefill_s", 0.0))
        first_token_s = float(bootstrap.get("first_token_s", prefill_s))
        stop_text = (
            bootstrap.get("text")
            if bootstrap.get("termination_reason") == "stop_sequence"
            else None)
        matched_stop_sequence = bootstrap.get("stop_sequence")
        grammar_completed = bool(
            constraint is not None and constraint.completed)

        # StreamingEngine retained a prompt endpoint after the one-token
        # bootstrap. Transfer that slot before mutating its KV; otherwise the
        # slot's token tuple would still describe the prompt while its aliased
        # recurrent state advances through speculative decode. Stable-boundary
        # slots own a separate fork and remain untouched (they are the useful
        # next-turn cache for recurrent chat templates).
        endpoint_slot = None
        hot_slots = getattr(tgt, "_hot_prompt_slots", None)
        if hot_slots is not None:
            for index, slot in enumerate(hot_slots):
                if getattr(slot, "kv", None) is kv:
                    endpoint_slot = hot_slots.pop(index)
                    break
        last_emitted_logits = (
            getattr(endpoint_slot, "logits", None)
            if endpoint_slot is not None else None)

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
        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(tgt.tokenizer, stop)
            if stop_text is None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)

        decode_t0 = time.perf_counter()
        while (len(emitted) < max_tokens and catchup_tok not in eos
               and not grammar_completed
               and stop_text is None):
            accepted_round = False
            retained_midpoint = None
            retained_midpoint_lengths = None
            # Grammar-deterministic spans need target state updates, but no
            # target decisions.  Preserve StreamingEngine's jump-forward win
            # inside MTP by folding catchup + the whole forced run into one
            # layer-major sweep, then sample only the first ambiguous token.
            # This is capability-driven for every schema/topic; it contains
            # no request text, tool name, or captured-shape predicate.
            forced_tokens: list[int] = []
            if (
                constraint is not None
                and getattr(
                    getattr(tgt, "rc", None),
                    "grammar_jump_forward_lossy", False)
            ):
                forced_tokens = constraint.forced_run(
                    max(0, max_tokens - len(emitted)),
                    encode=lambda text: tgt.tokenizer.encode(text).ids,
                )
                grammar_completed = bool(constraint.completed)
            if forced_tokens:
                terminal_forced = grammar_completed
                committed_forced: list[int] = []
                for token in forced_tokens:
                    committed_forced.append(int(token))
                    if stop:
                        match = _stop_match(
                            tgt.tokenizer.decode(
                                emitted + committed_forced))
                        if match is not None:
                            terminal_forced = True
                    if (
                        token in eos
                        or len(emitted) + len(committed_forced) >= max_tokens
                    ):
                        terminal_forced = True
                    if terminal_forced:
                        break

                # Retain the final-token-never-fed endpoint on terminal
                # forced spans.  Otherwise feed every forced token so the
                # last position's logits can decide the next free token.
                feed_forced = (
                    committed_forced[:-1]
                    if terminal_forced else committed_forced
                )
                verify_tokens = [catchup_tok] + feed_forced
                serial_verify = callable(getattr(
                    tgt, "forward_tokens_serial_positions", None))
                if serial_verify:
                    forced_logits = tgt.forward_tokens_serial_positions(
                        verify_tokens, kv, capture_kda_endpoints=False)
                else:
                    forced_logits = tgt.forward_tokens(verify_tokens, kv)
                target_decode_sweeps += 1
                plain_decode_sweeps += 1
                grammar_forced_sweeps += 1
                grammar_forced_tokens += len(committed_forced)
                round_outcomes.append(f"F{len(committed_forced)}")
                mx.eval(forced_logits)
                h_last = tgt._h_last
                if terminal_forced:
                    new_tokens = committed_forced
                    new_token_logits = [
                        forced_logits[index]
                        for index in range(len(committed_forced))
                    ]
                    next_catchup_tok = committed_forced[-1]
                else:
                    authoritative_logits = constraint.mask_logits(
                        forced_logits[-1])
                    next_free = sample(
                        authoritative_logits,
                        sampling,
                        history=all_tokens + committed_forced,
                    )
                    constraint.accept_token(next_free)
                    grammar_completed = bool(constraint.completed)
                    new_tokens = committed_forced + [next_free]
                    new_token_logits = [
                        forced_logits[index]
                        for index in range(len(committed_forced))
                    ] + [authoritative_logits]
                    next_catchup_tok = next_free
            elif warmup_remaining or adaptive_disabled:
                plain_logits = tgt.forward_tokens([catchup_tok], kv)
                target_decode_sweeps += 1
                plain_decode_sweeps += 1
                if warmup_remaining:
                    warmup_remaining -= 1
                    warmup_decode_sweeps += 1
                mx.eval(plain_logits)
                authoritative_logits = (
                    constraint.mask_logits(plain_logits[-1])
                    if constraint is not None else plain_logits[-1]
                )
                next_plain = sample(
                    authoritative_logits, sampling, history=all_tokens)
                if constraint is not None:
                    constraint.accept_token(next_plain)
                    grammar_completed = bool(constraint.completed)
                new_tokens = [next_plain]
                new_token_logits = [authoritative_logits]
                h_last = tgt._h_last
                next_catchup_tok = new_tokens[0]
            else:
                # Position of catchup_tok, matching kv.offset before this
                # round's combined call feeds it.
                round_start_offset = kv.offset
                layer_lengths_fn = getattr(kv, "layer_lengths", None)
                round_start_layer_lengths = (
                    layer_lengths_fn() if callable(layer_lengths_fn) else None)
                # GLM's identical prefill-sync convention (glm_mtp.py:53-69,
                # "entry i covers position i") confirms the MTP entry's RoPE
                # position matches h_last's OWN position (round_start_offset-1,
                # the state that produced catchup_tok) -- not catchup_tok's
                # position. Only affects acceptance rate, never correctness.
                draft_probabilities = None
                if sampling.is_greedy:
                    draft_tok = self.drafter.draft_token(
                        h_last, catchup_tok, mtp_kv,
                        round_start_offset - 1)
                else:
                    draft_logits = self.drafter.draft_logits(
                        h_last, catchup_tok, mtp_kv,
                        round_start_offset - 1)
                    # A released MTP head can rank the right continuation but
                    # be poorly calibrated as a sampler.  Flatten q over its
                    # constrained top ranks instead of trusting raw
                    # probabilities. Leviathan correction is exact for ANY q:
                    # accept with min(1,p(d)/q(d)), otherwise sample
                    # normalize((p-q)+). This preserves the target while
                    # covering near-miss ranks that a delta proposal loses.
                    draft_probabilities = _flat_top_k_draft_probabilities(
                        draft_logits,
                        sampling,
                        all_tokens,
                        self.stochastic_draft_top_k,
                        constraint,
                    )
                    draft_tok = sample_probabilities(draft_probabilities)
                proposed += 1

                verify_tokens = [catchup_tok, draft_tok]
                serial_verify = callable(getattr(
                    tgt, "forward_tokens_serial_positions", None))
                capture_endpoint = bool(
                    serial_verify
                    and getattr(kv, "kda_cache", None) is not None
                )
                if serial_verify:
                    spec_logits = tgt.forward_tokens_serial_positions(
                        verify_tokens,
                        kv,
                        capture_kda_endpoints=capture_endpoint,
                    )
                    serial_verify_rounds += 1
                else:
                    # Compatibility fallback for old dense adapters. MoE
                    # construction fails closed above when the exact verifier
                    # is unavailable.
                    spec_logits = tgt.forward_tokens(verify_tokens, kv)
                target_decode_sweeps += 1
                authoritative_logits = (
                    constraint.mask_logits(spec_logits[0])
                    if constraint is not None else spec_logits[0]
                )
                if sampling.is_greedy:
                    draft_accepted = (
                        int(mx.argmax(authoritative_logits)) == draft_tok)
                    true_tok = (
                        draft_tok if draft_accepted
                        else int(mx.argmax(authoritative_logits))
                    )
                else:
                    if draft_probabilities is None:
                        raise RuntimeError(
                            "stochastic Qwen MTP proposal omitted q distribution")
                    draft_accepted, true_tok, _target_probabilities = (
                        _verify_stochastic_mtp_token(
                            draft_tok,
                            draft_probabilities,
                            authoritative_logits,
                            sampling,
                            all_tokens,
                        )
                    )
                    overlap = mx.sum(mx.minimum(
                        _target_probabilities, draft_probabilities))
                    mx.eval(overlap)
                    stochastic_expected_acceptance_sum += float(
                        overlap.item())
                retained_midpoint = (
                    tgt.consume_serial_kda_endpoint(1)
                    if capture_endpoint else None
                )
                if round_start_layer_lengths is not None:
                    serial_end_layer_lengths = kv.layer_lengths()
                    layer_growth = tuple(
                        end - start for start, end in zip(
                            round_start_layer_lengths,
                            serial_end_layer_lengths,
                            strict=True,
                        )
                    )
                    if any(growth not in (0, 2) for growth in layer_growth):
                        raise RuntimeError(
                            "serial Qwen verifier changed an attention layer "
                            "by an unexpected position count: "
                            f"{layer_growth}")
                    retained_midpoint_lengths = tuple(
                        start + (1 if growth else 0)
                        for start, growth in zip(
                            round_start_layer_lengths, layer_growth, strict=True)
                    )
                if draft_accepted:
                    # Accept: kv already reflects [..., catchup_tok, draft_tok]
                    # from the single combined call above -- draft_tok is
                    # committed, and spec_logits[1] is a genuinely free second
                    # token from the SAME pass.
                    accepted += 1
                    round_outcomes.append("A")
                    accepted_round = True
                    if constraint is not None:
                        constraint.accept_token(draft_tok)
                        grammar_completed = bool(constraint.completed)
                    if grammar_completed:
                        new_tokens = [draft_tok]
                        new_token_logits = [authoritative_logits]
                        next_catchup_tok = draft_tok
                    else:
                        bonus_logits = (
                            constraint.mask_logits(spec_logits[1])
                            if constraint is not None else spec_logits[1]
                        )
                        bonus_tok = sample(
                            bonus_logits,
                            sampling,
                            history=all_tokens + [draft_tok],
                        )
                        if constraint is not None:
                            constraint.accept_token(bonus_tok)
                            grammar_completed = bool(constraint.completed)
                        new_tokens = [draft_tok, bonus_tok]
                        new_token_logits = [authoritative_logits, bonus_logits]
                        next_catchup_tok = bonus_tok
                    h_last = tgt._h_last
                else:
                    # Position-zero logits already sampled the authoritative
                    # next token. Keep the exact midpoint instead of restoring
                    # and refeeding through another complete streamed sweep.
                    midpoint_hidden = tgt._h_window[:, :1, :]
                    round_outcomes.append("R")
                    if capture_endpoint:
                        if retained_midpoint is None:
                            raise RuntimeError(
                                "serial Qwen verifier did not retain its KDA "
                                "midpoint")
                        kda_endpoint_restores += 1
                    # Mixed-depth endpoint-packed prefill gives upper full-
                    # attention layers a compact suffix whose local length is
                    # much smaller than the cache's aggregate offset. A global
                    # trim therefore leaves the rejected draft resident in
                    # those layers. Restore each layer to its checkpoint-local
                    # midpoint instead (KVCache explicitly exposes this exact
                    # rollback primitive for mixed-depth speculation).
                    if retained_midpoint_lengths is not None:
                        kv.trim_layer_lengths(retained_midpoint_lengths)
                    else:
                        kv.trim(round_start_offset + 1)
                    if capture_endpoint:
                        kv.kda_cache = retained_midpoint
                    if constraint is not None:
                        constraint.accept_token(true_tok)
                        grammar_completed = bool(constraint.completed)
                    new_tokens = [true_tok]
                    new_token_logits = [authoritative_logits]
                    h_last = midpoint_hidden
                    tgt._h_last = midpoint_hidden
                    next_catchup_tok = true_tok
                    refeed_sweeps_saved += int(serial_verify)

            emitted_before_round = len(emitted)
            for tok, token_logits in zip(
                    new_tokens, new_token_logits, strict=True):
                all_tokens.append(tok)
                emitted.append(tok)
                last_emitted_logits = token_logits
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
            terminal_round = (
                stop_text is not None
                or emitted[-1] in eos
                or grammar_completed
                or len(emitted) >= max_tokens)
            if (
                accepted_round
                and terminal_round
                and len(emitted) - emitted_before_round == 1
                and retained_midpoint is not None
            ):
                # The accepted draft itself became terminal. It is the final
                # returned (unfed) token, so recurrent state must end after the
                # catchup position, not after that draft.
                kv.kda_cache = retained_midpoint
                if retained_midpoint_lengths is not None:
                    kv.trim_layer_lengths(retained_midpoint_lengths)
                else:
                    kv.trim(round_start_offset + 1)
                tgt._h_last = tgt._h_window[:, :1, :]
                kda_endpoint_restores += 1
            # An accepted pair may encounter EOS/stop on its first token, in
            # which case ``next_catchup_tok`` is the unused bonus token. Never
            # continue from that uncommitted value (the old behavior leaked
            # post-EOS template tokens such as ``user`` into the response).
            catchup_tok = emitted[-1] if terminal_round else next_catchup_tok
            if terminal_round:
                break
            # Retained per-layer/KDA midpoints changed the break-even math:
            # every rejected draft now emits one authoritative token in the
            # same target sweep the ordinary path required, while every
            # accepted draft emits two and saves a complete target sweep.
            # The released draft head is only one extra layer, so any observed
            # acceptance is useful; disable only after an all-reject probe.
            # This still bounds truly incompatible prompt/template domains
            # without throwing away later wins after a mixed A/R prefix.
            if not adaptive_disabled and self.adaptive_stop:
                adaptive_disabled = (
                    _adaptive_mtp_should_disable(
                        proposed, accepted, adaptive_probe_limit)
                    if sampling.is_greedy else
                    _adaptive_stochastic_mtp_should_disable(
                        proposed,
                        stochastic_expected_acceptance_sum,
                        adaptive_probe_limit,
                    )
                )

        final_text = stop_text if stop_text is not None else tgt.tokenizer.decode(emitted)
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_t0
        # A stop/EOS can land before the end of a fully-accepted round's
        # bonus token, mirroring speculative.py's own endpoint-KV contract.
        endpoint = len(ids) + len(emitted) - 1
        if kv.offset > endpoint:
            kv.trim(endpoint)
        total_s = time.perf_counter() - request_t0
        path_stats = bootstrap_stats
        path_stats.update({
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": int(proposed > 0),
            "qwen_mtp_target_sweeps": target_decode_sweeps,
            "qwen_mtp_proposed": proposed,
            "qwen_mtp_accepted": accepted,
            "qwen_mtp_accept_rate": (
                accepted / proposed if proposed else 0.0),
            "qwen_mtp_decode_tokens": max(0, len(emitted) - 1),
            "qwen_mtp_adaptive_disabled": int(adaptive_disabled),
            "qwen_mtp_probe_rounds": min(
                proposed, adaptive_probe_limit),
            "qwen_mtp_plain_decode_sweeps": plain_decode_sweeps,
            "qwen_mtp_warmup_decode_sweeps": warmup_decode_sweeps,
            "qwen_mtp_serial_verify_rounds": serial_verify_rounds,
            "qwen_mtp_kda_endpoint_restores": kda_endpoint_restores,
            "qwen_mtp_refeed_sweeps_saved": refeed_sweeps_saved,
            "qwen_mtp_constraint_verified": int(constraint is not None),
            "qwen_mtp_stochastic": int(not sampling.is_greedy),
            "qwen_mtp_stochastic_draft_argmax": int(
                not sampling.is_greedy
                and self.stochastic_draft_top_k == 1),
            "qwen_mtp_stochastic_draft_top_k": (
                self.stochastic_draft_top_k
                if not sampling.is_greedy else 0),
            "qwen_mtp_stochastic_expected_acceptance": (
                stochastic_expected_acceptance_sum / proposed
                if not sampling.is_greedy and proposed else 0.0),
            "qwen_mtp_grammar_forced_tokens": grammar_forced_tokens,
            "qwen_mtp_grammar_forced_sweeps": grammar_forced_sweeps,
            "qwen_mtp_round_outcomes": "".join(round_outcomes),
        })
        if not proposed:
            path_stats["qwen_mtp_fallback_reason"] = (
                "terminal-during-plain-warmup")

        request_cache_after = _cache_io_snapshot(tgt)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats, after=request_cache_after)
        _record_cache_io_delta(
            tgt, request_cache_before, path_stats, prefix="prefill_",
            after=decode_cache_before)
        _record_cache_io_delta(
            tgt, decode_cache_before, path_stats, prefix="decode_",
            after=request_cache_after)

        # Restore a truthful full endpoint slot only when the bootstrap slot
        # itself owned this KV. A separately forked stable-boundary slot is
        # already the correct reusable artifact and was intentionally left in
        # the LRU above.
        if endpoint_slot is not None and last_emitted_logits is not None:
            mx.eval(last_emitted_logits)
            recurrent_state = getattr(kv, "kda_cache", None)
            if recurrent_state is not None:
                recurrent_state.synchronize()
            endpoint_slot.tokens = tuple(ids + emitted[:-1])
            endpoint_slot.kv = kv
            endpoint_slot.logits = last_emitted_logits
            endpoint_slot.prompt_length = len(ids)
            endpoint_slot.segment_chain = ()
            tgt._append_hot_prompt_slot(endpoint_slot)

        if tgt.governor is not None:
            tgt._true_peak_metal_bytes = max(
                tgt._true_peak_metal_bytes,
                tgt.governor.request_peak(),
                mx.get_active_memory(),
            )
        profiler = getattr(tgt, "_request_profiler", None)
        execution_profile = (
            profiler.result(total_s) if profiler is not None else None)
        result = {
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
                "grammar" if grammar_completed else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": tgt._true_peak_metal_bytes,
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
        }
        if execution_profile is not None:
            result["execution_profile"] = execution_profile
        return result
