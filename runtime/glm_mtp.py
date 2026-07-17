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
        hn = mx.fast.rms_norm(h_window[:, :-1, :], w[f"{p}.hnorm.weight"], cfg.rms_norm_eps)
        x = _linear(mx.concatenate([e, hn], axis=-1), w, f"{p}.eh_proj")
        h = run_glm_block(
            x, w, p, cfg, mtp_kv, self.mtp_layer, 0, eng._get_experts,
            iter_expert_batches=eng._iter_expert_batches,
        )
        mx.eval(h)

    def draft_tokens(self, h_last: mx.array, last_token: int, k: int, mtp_kv, offset: int) -> list[int]:
        """h_last: (1, 1, hidden) trunk hidden (pre final-norm) at the last position.
        Returns up to k draft tokens; mtp_kv accumulates the MTP block's KV (caller
        trims on rollback, mirroring target-KV rollback)."""
        eng = self.engine
        cfg = eng.cfg
        w = self._weights()
        p = f"model.layers.{self.mtp_layer}"
        drafts: list[int] = []
        h = h_last
        tok = last_token
        for i in range(k):
            e = eng._embed([tok])  # (1,1,hidden) — row-paged when enabled
            if offset + i == 0:
                e = mx.zeros_like(e)
            e = mx.fast.rms_norm(e, w[f"{p}.enorm.weight"], cfg.rms_norm_eps)
            hn = mx.fast.rms_norm(h, w[f"{p}.hnorm.weight"], cfg.rms_norm_eps)
            x = _linear(mx.concatenate([e, hn], axis=-1), w, f"{p}.eh_proj")
            h = run_glm_block(
                x, w, p, cfg, mtp_kv, self.mtp_layer, offset + i, eng._get_experts,
                iter_expert_batches=eng._iter_expert_batches,
            )
            g = mx.fast.rms_norm(h, w[f"{p}.shared_head.norm.weight"], cfg.rms_norm_eps)
            logits = quant.matmul(g, eng._lm_head_weight())[0, -1]
            mx.eval(logits)
            tok = int(mx.argmax(logits))
            drafts.append(tok)
            # Released deepseek_mtp recycles the post-final-norm hidden state.
            # Reusing pre-norm h (the old code) makes proposal 2+ follow a
            # different recurrence even if proposal 1 happens to match.
            h = g
            if tok in cfg.eos_token_ids:
                break
        return drafts
