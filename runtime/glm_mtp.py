"""F23: GLM-5.2 MTP (multi-token prediction) block as a verified draft source.

Checkpoint block at model.layers.78 (DeepSeek-V3 style, num_nextn_predict_layers=1,
applied ITERATIVELY):

    x_t   = eh_proj( [ enorm(embed(token_t)) | hnorm(h) ] )      # 2*6144 -> 6144
    h     = glm_block_78(x_t, mtp_kv, pos_t)                     # full MLA+MoE block
    logit = lm_head( shared_head.norm(h) )                       # lm_head is SHARED
    token_{t+1} = argmax -> next draft; iterate for k drafts

h starts as the TRUNK's last-position hidden state (pre final-norm) and evolves
through the MTP block across iterations. Cost per draft ≈ one layer (attention
~330 MB + 9 experts ~680 MB) instead of a 79 GB full sweep; per F01, a 5-draft
verify sweep costs ~x2.6 one token's bytes, so break-even is ~31% acceptance.

Wiring: SpeculativeDecoder gains an 'mtp' proposal mode that calls draft_tokens()
with the trunk state the engine already computes; the intended safety mechanism
is unchanged exact-target verification. This implementation remains provisional
until target-only token A/B, rollback, and long-DSA gates pass; do not infer
losslessness from the design label alone. MTP KV must be trimmed on rollback like
any draft KV.
"""

from __future__ import annotations

import math

import mlx.core as mx

from . import quant
from .glm import run_glm_block
from .layer_runner import _linear

MTP_LAYER = 78  # model.layers.78 on the RELEASED GLM-5.2 checkpoint specifically —
# do not import this as a general constant; MTPDrafter derives the real index
# from config so architecture-faithful fixtures with fewer trunk layers work.


class MTPDrafter:
    def __init__(self, engine):
        self.engine = engine
        # F65: the MTP block always sits ONE PAST the trunk (checkpoint convention
        # confirmed on the release: num_hidden_layers=78, MTP at layers.78).
        # Deriving this from config (instead of the hardcoded 78) is what lets a
        # tiny fixture with e.g. 4 trunk layers exercise the real MTP code path.
        self.mtp_layer = engine.cfg.num_hidden_layers
        if engine.cfg.model_type == "glm_moe_dsa" and engine.cfg.num_hidden_layers == 78:
            assert self.mtp_layer == MTP_LAYER, "released-checkpoint MTP layer drifted from 78"
        names = engine.store.names_with_prefix(f"model.layers.{self.mtp_layer}.")
        self._page_names = [n for n in names if ".mlp.experts." not in n]
        self._capture_logit_margin = False
        self._min_logit_margin = 0.0
        self.reset_confidence_telemetry()

    def configure_confidence(
        self, *, capture_logit_margin: bool = False,
        min_logit_margin: float = 0.0,
    ) -> None:
        """Configure content-blind confidence telemetry/width truncation.

        A positive margin threshold may withhold a weak candidate after its
        native MTP state update but before widening the authoritative target
        verifier.  The target still emits every token; this can change only
        speculative work, never target acceptance or sampling arithmetic.
        """
        if not isinstance(capture_logit_margin, bool):
            raise TypeError("capture_logit_margin must be bool")
        try:
            margin = float(min_logit_margin)
        except (TypeError, ValueError) as error:
            raise ValueError("MTP minimum logit margin must be finite") from error
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("MTP minimum logit margin must be finite and nonnegative")
        self._capture_logit_margin = capture_logit_margin
        self._min_logit_margin = margin

    def reset_confidence_telemetry(self) -> None:
        self._proposal_logit_margins: list[float] = []
        self._synchronization_logit_margins: list[float] = []
        self._confidence_withheld = 0
        self._confidence_synchronization = False

    def confidence_telemetry(self) -> dict[str, int | float | str]:
        margins = self._proposal_logit_margins
        synchronization = self._synchronization_logit_margins
        return {
            "glm53_mtp_confidence_enabled": int(
                self._capture_logit_margin or self._min_logit_margin > 0.0),
            "glm53_mtp_min_logit_margin": float(self._min_logit_margin),
            "glm53_mtp_confidence_candidates": len(margins),
            "glm53_mtp_confidence_withheld": int(self._confidence_withheld),
            "glm53_mtp_logit_margin_min": min(margins) if margins else 0.0,
            "glm53_mtp_logit_margin_mean": (
                sum(margins) / len(margins) if margins else 0.0),
            "glm53_mtp_logit_margin_max": max(margins) if margins else 0.0,
            "glm53_mtp_logit_margins": ",".join(
                f"{value:.8g}" for value in margins),
            "glm53_mtp_sync_confidence_candidates": len(synchronization),
            "glm53_mtp_sync_logit_margins": ",".join(
                f"{value:.8g}" for value in synchronization),
        }

    def _weights(self) -> dict:
        return self.engine.cache.get(f"layer.{self.mtp_layer}", self._page_names)

    def prefill(self, tokens: list[int], h_window: mx.array, mtp_kv) -> None:
        """F32: synchronize MTP attention state with the prompt BEFORE the first
        proposal, so MTP KV positions are absolute (entry i covers position i).
        Pairs embed(token_{i+1}) with trunk state h_i for i in [0, len-2], one
        multi-position block call."""
        eng = self.engine
        cfg = eng.cfg
        w = self._weights()
        p = f"model.layers.{self.mtp_layer}"
        e = eng._embed(list(tokens[1:]))  # (1, L-1, hidden)
        if e.shape[1]:
            # Released DeepSeek/GLM MTP masks inputs_embeds to zero at absolute
            # position 0 before enorm. The first synchronized pair occupies that
            # position even though its token value is tokens[1].
            e = mx.concatenate([mx.zeros_like(e[:, :1, :]), e[:, 1:, :]], axis=1)
        e = mx.fast.rms_norm(e, w[f"{p}.enorm.weight"], cfg.rms_norm_eps)
        h_source = h_window[:, :-1, :]
        if cfg.model_type == "glm5_next":
            # GLM-5.3 exports the trunk's post-final-norm ``h_nextn`` to the
            # draft graph.  The ordinary target keeps its pre-norm window for
            # target rollback, so apply that released boundary explicitly.
            h_source = mx.fast.rms_norm(
                h_source, eng._norm_w, cfg.rms_norm_eps)
        hn = mx.fast.rms_norm(
            h_source, w[f"{p}.hnorm.weight"], cfg.rms_norm_eps)
        x = _linear(mx.concatenate([e, hn], axis=-1), w, f"{p}.eh_proj")
        if cfg.model_type == "glm5_next":
            # Only the MTP attention cache crosses the prompt/decode boundary.
            # The prompt rows' Q/O projections and MoE outputs are dead: the
            # first proposal is conditioned on the target's released h_last,
            # while later proposal rows are computed normally by draft_tokens.
            # Populate precisely the compressed latent that the full block's
            # attention would install, byte-for-byte, without allocating a
            # quadratic dense prompt attention matrix or streaming 288 expert
            # pages whose outputs have no consumer.
            attn_input = mx.fast.rms_norm(
                x, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
            latent = _linear(
                attn_input, w, f"{p}.self_attn.kv_a_proj_with_mqa")
            latent = mx.fast.rms_norm(
                latent, w[f"{p}.self_attn.kv_a_layernorm.weight"],
                cfg.mla_latent_norm_eps)
            mtp_kv.update_latent(self.mtp_layer, latent)
            mx.eval(mtp_kv.keys[self.mtp_layer])
            eng._glm53_mtp_state_only_prefill_tokens = int(getattr(
                eng, "_glm53_mtp_state_only_prefill_tokens", 0)) + int(
                    latent.shape[1])
            h = latent
        else:
            h = run_glm_block(
                x, w, p, cfg, mtp_kv, self.mtp_layer, 0,
                eng._get_experts,
                iter_expert_batches=eng._iter_expert_batches,
            )
        mx.eval(h)

    def prefill_extension(
            self, tokens: list[int], h_source: mx.array, mtp_kv) -> None:
        """Append exact GLM-5.3 MTP prompt state for a strict extension.

        A cached prompt of length ``P`` already owns draft positions
        ``0..P-2``. New target tokens ``tokens[P:N]`` pair with trunk hidden
        rows ``h[P-1:N-1]`` and therefore append exactly ``N-P`` draft-cache
        rows. Unlike the checkpoint's absolute position-zero pair, none of
        these extension embeddings is masked to zero.
        """
        eng = self.engine
        cfg = eng.cfg
        if cfg.model_type != "glm5_next":
            raise ValueError(
                "MTP strict-extension prefill currently requires GLM-5.3")
        if len(tokens) != int(h_source.shape[1]):
            raise ValueError(
                "MTP extension tokens and target hidden rows must align")
        if not tokens:
            return
        w = self._weights()
        p = f"model.layers.{self.mtp_layer}"
        e = eng._embed(list(tokens))
        e = mx.fast.rms_norm(
            e, w[f"{p}.enorm.weight"], cfg.rms_norm_eps)
        h_source = mx.fast.rms_norm(
            h_source, eng._norm_w, cfg.rms_norm_eps)
        hn = mx.fast.rms_norm(
            h_source, w[f"{p}.hnorm.weight"], cfg.rms_norm_eps)
        x = _linear(mx.concatenate([e, hn], axis=-1), w, f"{p}.eh_proj")
        attn_input = mx.fast.rms_norm(
            x, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
        latent = _linear(
            attn_input, w, f"{p}.self_attn.kv_a_proj_with_mqa")
        latent = mx.fast.rms_norm(
            latent, w[f"{p}.self_attn.kv_a_layernorm.weight"],
            cfg.mla_latent_norm_eps)
        mtp_kv.update_latent(self.mtp_layer, latent)
        mx.eval(mtp_kv.keys[self.mtp_layer])
        eng._glm53_mtp_state_only_prefill_tokens = int(getattr(
            eng, "_glm53_mtp_state_only_prefill_tokens", 0)) + int(
                latent.shape[1])

    def draft_tokens(
            self, h_last: mx.array, last_token: int, k: int, mtp_kv,
            offset: int, *, constraint=None) -> list[int]:
        """h_last: (1, 1, hidden) trunk hidden (pre final-norm) at the last position.
        Returns up to k draft tokens; mtp_kv accumulates the MTP block's KV (caller
        trims on rollback, mirroring target-KV rollback)."""
        eng = self.engine
        cfg = eng.cfg
        w = self._weights()
        p = f"model.layers.{self.mtp_layer}"
        drafts: list[int] = []
        h = h_last
        if cfg.model_type == "glm5_next":
            h = mx.fast.rms_norm(h, eng._norm_w, cfg.rms_norm_eps)
        tok = last_token
        for i in range(k):
            e = eng._embed([tok])  # (1,1,hidden) — row-paged when enabled
            if offset + i == 0:
                e = mx.zeros_like(e)
            e = mx.fast.rms_norm(e, w[f"{p}.enorm.weight"], cfg.rms_norm_eps)
            hn = mx.fast.rms_norm(h, w[f"{p}.hnorm.weight"], cfg.rms_norm_eps)
            x = _linear(mx.concatenate([e, hn], axis=-1), w, f"{p}.eh_proj")
            if cfg.model_type == "glm5_next":
                from .glm5_next import run_glm5_next_mtp_block

                h = run_glm5_next_mtp_block(
                    x, w, p, cfg, mtp_kv, self.mtp_layer, offset + i,
                    eng._get_experts,
                    iter_expert_batches=eng._iter_expert_batches,
                )
            else:
                h = run_glm_block(
                    x, w, p, cfg, mtp_kv, self.mtp_layer, offset + i,
                    eng._get_experts,
                    iter_expert_batches=eng._iter_expert_batches,
                )
            g = mx.fast.rms_norm(h, w[f"{p}.shared_head.norm.weight"], cfg.rms_norm_eps)
            head = eng._lm_head_weight()
            from .lm_head_stream import StreamedLMHead

            logits = (
                head.logits(g)[0, -1]
                if isinstance(head, StreamedLMHead)
                else quant.matmul(g, head)[0, -1])
            if constraint is not None:
                logits = constraint.mask_logits(logits)
            mx.eval(logits)
            tok = int(mx.argmax(logits))
            if self._capture_logit_margin or self._min_logit_margin > 0.0:
                top_two = mx.topk(logits, 2)
                mx.eval(top_two)
                margin = float(
                    (mx.max(top_two) - mx.min(top_two)).item())
                if self._confidence_synchronization:
                    self._synchronization_logit_margins.append(margin)
                else:
                    self._proposal_logit_margins.append(margin)
                if (not self._confidence_synchronization
                        and margin < self._min_logit_margin):
                    self._confidence_withheld += 1
                    break
            drafts.append(tok)
            if constraint is not None:
                constraint.accept_token(tok)
            # Released deepseek_mtp recycles the post-final-norm hidden state.
            # Reusing pre-norm h (the old code) makes proposal 2+ follow a
            # different recurrence even if proposal 1 happens to match.
            h = g
            if (tok in cfg.eos_token_ids
                    or bool(constraint is not None and constraint.completed)):
                break
        return drafts
