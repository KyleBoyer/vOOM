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
prior caller (qwen3vl.py's _fork_hybrid_prompt_endpoint).

Round structure (matching speculative.py's own "1 + k" verify-batch
convention exactly, k=1 here): each round feeds
``[catchup_token, draft_token]`` as ONE combined forward call (not two
separate ones -- separate calls would silently double-process every
accepted token across rounds, since the token committed as this round's
draft becomes the NEXT round's catchup token). The checkpoint is taken
once, at the CLEAN boundary before the round's combined call -- exactly the
state as of the end of the previous round -- rather than attempting a
mid-call snapshot inside a layer-major forward loop (which would need new
plumbing there). The combined call itself goes through
StreamingEngine.forward_tokens (which uses _sweep -> run_qwen35_block, the
same dispatch ordinary decode already uses); forward_tokens_serial_positions
is NOT usable for qwen3_5 targets -- it calls layer_runner.run_block, a
plain dense-transformer block with no awareness of the hybrid layer_types
here (engine.py now explicitly excludes qwen3_5/qwen3_5_moe/kimi_linear from
it, a real pre-existing gap this work surfaced). On reject, restoring that
checkpoint
and re-feeding just the catchup token costs one extra single-position
forward pass -- the same worst case any speculative scheme pays on a miss,
never more.

This module mirrors runtime/glm_mtp.py's structure and safety framing:
target-only exact-argmax verification is the ENTIRE correctness mechanism,
same as GLM's (still-provisional, F23) MTP path. This is deliberately a NEW,
SIMPLER engine (not a SpeculativeDecoder extension) because k is always 1
here -- SpeculativeDecoder's adaptive-k controller, F48 byte-fitted
telemetry, and multi-round bookkeeping are irrelevant complexity for a fixed
k=1 target, and this keeps the already-provisional GLM path untouched.
"""

from __future__ import annotations

import time

import mlx.core as mx

from . import quant
from .kv_cache import KVCache
from .qwen35 import _full_attention, _moe, _swiglu, final_logits, qwen35_rms_norm
from .sampler import SamplingParams


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

    def draft_token(self, h_last: mx.array, last_token: int, mtp_kv, offset: int) -> int:
        """h_last: (1, 1, hidden) trunk hidden (pre final-norm) at position
        offset-1 (i.e. the state that produced last_token). Returns a single
        draft token id for position offset+1. `offset` is the ABSOLUTE
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
        return int(mx.argmax(logits))


class QwenMTPSpeculativeEngine:
    """Serving adapter, mirroring SpeculativeEngine's shape: falls back to
    the plain target engine for any request shape the (greedy-only, k=1)
    verified-draft scheme doesn't cover. Attribute access delegates to the
    target so protocol rendering/telemetry see the real checkpoint,
    tokenizer, config, and execution profile."""

    def __init__(self, target, max_prompt_tokens: int = 8192):
        if max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        self.target = target
        self.drafter = QwenMTPDrafter(target)
        self.max_prompt_tokens = max_prompt_tokens

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt, max_tokens, on_token,
                          stop, on_progress, sampling, constraint) -> dict:
        kwargs = {"on_token": on_token, "stop": stop, "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        result = self.target.generate(prompt, max_tokens, **kwargs)
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
        if constraint is not None:
            return self._target_generate(
                "constrained-decoding", prompt, max_tokens, on_token, stop,
                on_progress, sampling, constraint)
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
        tokenize_s = 0.0
        tgt.release_request_state()
        eos = set(tgt.cfg.eos_token_ids)
        stop = stop or []

        kv = tgt.new_kv()
        mtp_kv = KVCache(1)
        prefill_t0 = time.perf_counter()
        logits = tgt.forward_tokens(ids, kv)
        prompt_last_logits = logits[-1]
        mx.eval(prompt_last_logits)
        prefill_s = time.perf_counter() - prefill_t0

        proposed = 0
        accepted = 0
        sweeps = 1  # count the prefill sweep, matching SpeculativeDecoder's convention

        # Invariant (matching speculative.py's documented one): all_tokens =
        # prompt + emitted; catchup_tok = all_tokens[-1] is sampled but not
        # yet fed to kv (kv.offset == len(all_tokens) - 1) until this
        # round's combined forward call feeds it.
        catchup_tok = int(mx.argmax(prompt_last_logits))
        h_last = tgt._h_last
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
                "total_tokens": len(ids), "cache_source": "qwen-mtp-cold",
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
        while (len(emitted) < max_tokens and catchup_tok not in eos
               and stop_text is None):
            # Position of catchup_tok, matching kv.offset before this round's
            # combined call feeds it.
            round_start_offset = kv.offset
            # GLM's identical prefill-sync convention (glm_mtp.py:53-69,
            # "entry i covers position i") confirms the MTP entry's RoPE
            # position matches h_last's OWN position (round_start_offset-1,
            # the state that produced catchup_tok) -- not catchup_tok's
            # position. Only affects acceptance rate, never correctness.
            draft_tok = self.drafter.draft_token(
                h_last, catchup_tok, mtp_kv, round_start_offset - 1)
            proposed += 1

            # Checkpoint: state reflects exactly the PREVIOUS round's
            # committed endpoint -- the one clean rollback point this scheme
            # needs, taken before catchup_tok is fed at all.
            kda_checkpoint = (
                kv.kda_cache.fork() if getattr(kv, "kda_cache", None) is not None
                else None)

            # forward_tokens_serial_positions is NOT usable here: it calls
            # layer_runner.run_block, a plain dense-transformer block with no
            # awareness of qwen3_5's hybrid DeltaNet/full-attention
            # layer_types (engine.py excludes qwen3_5/qwen3_5_moe/
            # kimi_linear from it for exactly this reason). forward_tokens
            # (via _sweep, which DOES dispatch correctly to
            # run_qwen35_block) is the correct path for both dense and MoE
            # qwen3_5 targets.
            verify_tokens = [catchup_tok, draft_tok]
            spec_logits = tgt.forward_tokens(verify_tokens, kv)
            sweeps += 1
            true_tok = int(mx.argmax(spec_logits[0]))

            if true_tok == draft_tok:
                # Accept: kv already reflects [..., catchup_tok, draft_tok]
                # from the single combined call above -- draft_tok is
                # committed, and spec_logits[1] is a genuinely free second
                # token from the SAME pass (this is the actual speedup: one
                # combined forward call emits two verified tokens).
                accepted += 1
                bonus_tok = int(mx.argmax(spec_logits[1]))
                new_tokens = [draft_tok, bonus_tok]
                h_last = tgt._h_last
                next_catchup_tok = bonus_tok
            else:
                # Reject: discard the speculative advance through draft_tok
                # entirely -- restore kda_cache to the pre-round checkpoint
                # and roll ordinary KV back to round_start_offset (removing
                # BOTH fed positions), then re-feed catchup_tok alone. This
                # costs one extra single-position forward pass versus plain
                # decoding for this one round -- the standard cost of a
                # rejected speculative round, never more.
                if kda_checkpoint is not None:
                    kv.kda_cache = kda_checkpoint
                kv.trim(round_start_offset)
                catchup_logits = tgt.forward_tokens([catchup_tok], kv)
                mx.eval(catchup_logits)
                new_tokens = [true_tok]
                h_last = tgt._h_last
                next_catchup_tok = true_tok

            for tok in new_tokens:
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
            catchup_tok = next_catchup_tok

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
        path_stats = {
            "prompt_tokenize_s": tokenize_s,
            "rope_profile": tgt.rope_profile,
            "effective_context_limit": tgt.effective_max_position_embeddings,
            "qwen_mtp_enabled": 1,
            "qwen_mtp_used": 1,
            "qwen_mtp_target_sweeps": max(0, sweeps - 1),
            "qwen_mtp_proposed": proposed,
            "qwen_mtp_accepted": accepted,
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
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": mx.get_active_memory(),
            "path_stats": path_stats,
            "prompt_tokens": len(ids),
        }
