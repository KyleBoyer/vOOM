"""F62: DSpark drafter (DeepSpec, arXiv:2607.05147) for the "qwen3" family of
standalone drafter checkpoints (e.g. deepseek-ai/dspark_qwen3_4b_block7).

Architecture, confirmed by direct inspection of the checkpoint's config.json
and safetensors tensor names/shapes, and cross-checked against the reference
MLX port (github.com/ARahim3/mlx-dspark, MIT-licensed) — this is an
independent re-implementation adapted to this runtime's own conventions
(mx.fast.rope matching runtime/layer_runner.py's
existing Qwen convention instead of the reference port's mlx_vlm dependency;
plain mx.nn.Module classes so `load_weights` matches the checkpoint's tensor
names 1:1, same as the reference), NOT a vendored copy:

- `embed_tokens`/`lm_head`: the drafter's OWN vocabulary projection (separate
  weights from the target, untied).
- `fc`: a (hidden*n_taps -> hidden) linear that fuses N concatenated target
  hidden-state taps (captured via runtime/engine.py's `tap_layers`, F62 prep)
  into the drafter's own hidden space, followed by `hidden_norm`.
- `layers.0..4`: 5 standard Qwen3-style transformer blocks (GQA, QK-norm,
  SwiGLU MLP) — EXCEPT self-attention is CROSS-attention: Q comes from the
  drafter's own "block" positions (1 real anchor token + (block_size-1) MASK
  tokens), K/V come from concat([cached fused-target-context, block]). The
  context cache grows only with committed tokens (never rolled back — unlike
  the target's own KV cache) and is genuinely bidirectional within the block
  (no causal mask): every block position (including the mask positions)
  attends to the whole context AND every other block position. This is what
  lets one forward pass predict block_size-1 tokens in parallel.
- `markov_head`: rank-256 previous-token bigram-style logit bias,
  `bias(prev, v) = markov_w2(markov_w1[prev])[v]`, applied SEQUENTIALLY
  (each drafted position's bias depends on the ACTUAL previous drafted
  token, not the parallel block prediction) on top of the block's base
  logits.
- `confidence_head`: predicts per-position conditional survival probability
  from [hidden ; markov_w1[prev_token]] — used to adaptively truncate the
  draft length; NOT used for correctness (target still verifies everything
  that IS drafted).

Safety property unchanged from every other drafter in this runtime: the
target ALWAYS verifies every proposed token exactly (F32's rollback
invariant). A wrong or low-quality drafter can only reduce acceptance rate
(speed), never change the emitted token stream — see
experiments/dspark_control.py for the strict token-identity gate.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .sampler import SamplingParams


@dataclass
class DSparkConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int
    mask_token_id: int
    target_layer_ids: list[int]
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "DSparkConfig":
        c = json.loads(Path(path).read_text())
        if "speculators_config" in c or "speculators_model_type" in c:
            raise ValueError(
                f"{path}: vLLM 'speculators' packaging (speculators_config present) — "
                f"different tensor/config schema, not supported by this loader. Needs a "
                f"DeepSpec-native standalone drafter (e.g. deepseek-ai/dspark_*_block7).")
        if c.get("model_type") != "qwen3":
            raise ValueError(f"{path}: only the qwen3-family DSpark drafter is implemented "
                             f"here (got model_type={c.get('model_type')!r})")
        rp = c.get("rope_parameters") or {}
        return cls(
            hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            block_size=c["block_size"], mask_token_id=c["mask_token_id"],
            target_layer_ids=list(c["target_layer_ids"]),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
            attention_bias=c.get("attention_bias", False),
            markov_rank=c.get("markov_rank", 256),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
        )


class CtxCache:
    """Per-layer cache of the drafter's cross-attention K/V, projected from
    committed target-hidden-state fusions. Append-only — grows with commits,
    never trimmed (unlike the target's own KV cache; there is no spec-decode
    rollback concept on the CONTEXT side, only on the target's own state)."""

    def __init__(self):
        self.k: mx.array | None = None
        self.v: mx.array | None = None

    def append(self, k: mx.array, v: mx.array) -> None:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)

    def trim_to(self, length: int) -> None:
        """Keep exactly the committed prefix after an early stop/EOS."""
        if length < 0:
            raise ValueError("DSpark context length must be non-negative")
        if self.k is not None and length < self.k.shape[2]:
            self.k = self.k[:, :, :length, :]
            self.v = self.v[:, :, :length, :]

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class DSparkAttention(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta

        h, b = cfg.hidden_size, cfg.attention_bias
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=b)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def _rope(self, x: mx.array, offset) -> mx.array:
        # matches runtime/layer_runner.py's existing Qwen convention exactly
        return mx.fast.rope(x, self.head_dim, traditional=False, base=self.rope_theta,
                            scale=1.0, offset=offset)

    def _kv(self, x: mx.array):
        b, s, _ = x.shape
        k = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        return self.k_norm(k), v

    def update_ctx(self, fused_new: mx.array, ctx_offset: int, cache: CtxCache) -> None:
        """fused_new: (1, S_new, hidden), the fc+hidden_norm output for newly
        committed positions. Projects to K/V, ropes K at its absolute
        position, appends to the persistent context cache. V is not roped."""
        k, v = self._kv(fused_new)
        cache.append(self._rope(k, offset=ctx_offset), v)

    def attend(self, hidden: mx.array, block_offset: int, cache: CtxCache) -> mx.array:
        """hidden: (1, block_size, hidden) — the block's own positions. Q from
        the block; K/V from concat([cached context, block's own K/V]).
        Bidirectional within the block (no causal mask): every position,
        including MASK positions, sees the whole context and every other
        block position — this is what makes one forward predict the whole
        block in parallel."""
        b, q_len, _ = hidden.shape
        q = self.q_proj(hidden).reshape(b, q_len, self.n_heads, self.head_dim)
        q = self._rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=block_offset)

        k_blk, v_blk = self._kv(hidden)
        k_blk = self._rope(k_blk, offset=block_offset)
        k = mx.concatenate([cache.k, k_blk], axis=2) if cache.k is not None else k_blk
        v = mx.concatenate([cache.v, v_blk], axis=2) if cache.v is not None else v_blk

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(b, q_len, -1)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.self_attn = DSparkAttention(cfg)
        self.mlp = DSparkMLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, hidden: mx.array, block_offset: int, cache: CtxCache) -> mx.array:
        h = self.input_layernorm(hidden)
        h = self.self_attn.attend(h, block_offset, cache)
        hidden = hidden + h
        h = self.post_attention_layernorm(hidden)
        h = self.mlp(h)
        return hidden + h


class VanillaMarkov(nn.Module):
    """Rank-256 previous-token bigram bias: bias(prev, v) = w2(w1[prev])[v]."""

    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.markov_w1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank)
        self.markov_w2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, prev_token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(prev_token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class DSparkDrafter(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.config = cfg
        self.block_size = cfg.block_size
        self.mask_token_id = cfg.mask_token_id

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size,
                            cfg.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = [DSparkDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.markov_head = VanillaMarkov(cfg) if cfg.markov_rank > 0 else None
        self.confidence_head = None
        if cfg.enable_confidence_head:
            in_dim = cfg.hidden_size + (cfg.markov_rank if cfg.confidence_head_with_markov else 0)
            self.confidence_head = ConfidenceHead(in_dim)

    @classmethod
    def load(cls, model_dir: str | Path) -> "DSparkDrafter":
        model_dir = Path(model_dir)
        cfg = DSparkConfig.from_json(model_dir / "config.json")
        model = cls(cfg)
        model.load_weights(str(model_dir / "model.safetensors"))
        mx.eval(model.parameters())
        return model

    def make_ctx_cache(self) -> list[CtxCache]:
        return [CtxCache() for _ in self.layers]

    def fuse_target(self, target_hidden_cat: mx.array) -> mx.array:
        """target_hidden_cat: (1, S, n_taps*hidden), the concatenation (in
        target_layer_ids order) of the target's tapped hidden states."""
        return self.hidden_norm(self.fc(target_hidden_cat))

    def update_context(self, target_hidden_cat: mx.array, ctx_offset: int,
                       ctx_caches: list[CtxCache]) -> None:
        fused = self.fuse_target(target_hidden_cat)
        for layer, cache in zip(self.layers, ctx_caches):
            layer.self_attn.update_ctx(fused, ctx_offset, cache)

    def draft_block(self, pending_token: int, block_offset: int,
                    ctx_caches: list[CtxCache], cap: int):
        """pending_token: the last COMMITTED token. Returns base_logits
        (cap, vocab) for the next `cap` positions after pending_token —
        position i's logits come from the block's position i (0-indexed),
        which attends bidirectionally to the whole context + block."""
        block_ids = [pending_token] + [self.mask_token_id] * (self.block_size - 1)
        h = self.embed_tokens(mx.array([block_ids]))
        for layer, cache in zip(self.layers, ctx_caches):
            h = layer(h, block_offset, cache)
        h = self.norm(h)
        head_hidden = h[:, :cap, :]
        return self.lm_head(head_hidden)[0], head_hidden[0]  # (cap, vocab), (cap, hidden)

    def sample_block_greedy(self, base_logits: mx.array, first_prev_token: int) -> list[int]:
        """Sequential Markov-bias application: position i's bias depends on
        the ACTUAL token drafted at i-1 (or first_prev_token for i=0)."""
        cap = base_logits.shape[0]
        if self.markov_head is None:
            return [int(x) for x in mx.argmax(base_logits, axis=-1).tolist()]
        tokens = []
        prev = first_prev_token
        for i in range(cap):
            step_logits = base_logits[i] + self.markov_head.step_bias(mx.array([prev]))[0]
            nxt = int(mx.argmax(step_logits).item())
            tokens.append(nxt)
            prev = nxt
        return tokens

    def sample_block(self, base_logits: mx.array,
                     first_prev_token: int) -> mx.array:
        """Greedy block proposal kept on-device until one final materialization.

        ``sample_block_greedy`` is retained as the small, convenient probe API.
        The serving path uses this method so the sequential Markov correction
        builds one lazy graph instead of forcing one CPU/GPU synchronization
        per proposed token.
        """
        if self.markov_head is None:
            return mx.argmax(base_logits, axis=-1)
        tokens = []
        prev = mx.array([first_prev_token])
        for i in range(base_logits.shape[0]):
            step_logits = base_logits[i] + self.markov_head.step_bias(prev)[0]
            nxt = mx.argmax(step_logits, axis=-1, keepdims=True)
            tokens.append(nxt)
            prev = nxt
        return mx.concatenate(tokens)

    def confidence_survival(self, head_hidden: mx.array, prev_token_ids: mx.array):
        """Per-position sigmoid(confidence logit) — conditional survival
        probability estimate, NOT used for correctness (truncation-only)."""
        if self.confidence_head is None:
            return None
        if self.config.confidence_head_with_markov:
            feats = mx.concatenate(
                [head_hidden, self.markov_head.prev_embeddings(prev_token_ids)], axis=-1)
        else:
            feats = head_hidden
        return mx.sigmoid(self.confidence_head(feats))


@dataclass
class DSparkStats:
    target_sweeps: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0
    draft_s: float = 0.0
    verify_s: float = 0.0
    context_s: float = 0.0
    rounds: list[tuple[int, int]] = field(default_factory=list)


class DSparkSpeculativeDecoder:
    """Exact greedy target verification for a resident DSpark block drafter.

    The target KV and drafter context both hold every committed input up to,
    but never including, ``pending``.  A round verifies
    ``[pending] + proposals`` with ordinary one-token arithmetic shapes, keeps
    ``pending + accepted proposals`` in both states, and emits the accepted
    prefix plus one target token.  Thus draft quality affects only speed.
    """

    def __init__(self, target, drafter: DSparkDrafter, *,
                 max_draft_tokens: int = 4,
                 confidence_threshold: float = 0.0,
                 prompt_cache_min_tokens: int = 2048):
        cfg = drafter.config
        if max_draft_tokens <= 0:
            raise ValueError("DSpark max_draft_tokens must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("DSpark confidence_threshold must be in [0, 1]")
        if prompt_cache_min_tokens < 0:
            raise ValueError("DSpark prompt_cache_min_tokens must be >= 0")
        if target.cfg.model_type != "qwen3":
            raise ValueError(
                f"DSpark Qwen3 checkpoint needs a qwen3 target, got "
                f"{target.cfg.model_type!r}")
        if target.cfg.hidden_size != cfg.hidden_size:
            raise ValueError(
                f"DSpark/target hidden mismatch: {cfg.hidden_size} vs "
                f"{target.cfg.hidden_size}")
        if target.cfg.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"DSpark/target vocab mismatch: {cfg.vocab_size} vs "
                f"{target.cfg.vocab_size}")
        if not cfg.target_layer_ids or max(cfg.target_layer_ids) >= target.cfg.num_hidden_layers:
            raise ValueError("DSpark target_layer_ids are incompatible with the target")
        self.target = target
        self.drafter = drafter
        self.max_draft_tokens = min(max_draft_tokens, cfg.block_size)
        self.confidence_threshold = confidence_threshold
        self.prompt_cache_min_tokens = prompt_cache_min_tokens
        self._prompt_cache = None

    def clear_prompt_cache(self) -> None:
        self._prompt_cache = None

    @staticmethod
    def _stop_match(text: str, stops: list[str]):
        matches = [(text.find(value), index, value)
                   for index, value in enumerate(stops)
                   if value and text.find(value) != -1]
        return min(matches) if matches else None

    def _propose(self, pending: int, offset: int,
                 ctx_caches: list[CtxCache], cap: int):
        base_logits, head_hidden = self.drafter.draft_block(
            pending, offset, ctx_caches, cap)
        draft_arr = self.drafter.sample_block(base_logits, pending)
        mx.eval(draft_arr)
        proposals = [int(token) for token in draft_arr.tolist()]
        if (self.confidence_threshold > 0.0
                and self.drafter.confidence_head is not None):
            prev = mx.array([pending] + proposals[:-1])
            confidence = self.drafter.confidence_survival(head_hidden, prev)
            mx.eval(confidence)
            survival = 1.0
            keep = 0
            for value in confidence.tolist():
                survival *= float(value)
                if survival < self.confidence_threshold:
                    break
                keep += 1
            # A zero-token round pays a target sweep without amortizing it.
            # Confidence schedules width, never whether to make progress.
            proposals = proposals[:max(1, keep)]
        return proposals

    def _tapped_context(self) -> mx.array:
        target = self.target
        ids = self.drafter.config.target_layer_ids
        missing = [layer for layer in ids if layer not in target._tap_hidden]
        if missing:
            raise RuntimeError(f"target did not capture DSpark tap layers {missing}")
        return mx.concatenate([target._tap_hidden[layer] for layer in ids], axis=-1)

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None, *, encoded_ids=None) -> dict:
        target = self.target
        stops = list(stop or [])
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        request_t0 = time.perf_counter()
        tokenize_t0 = time.perf_counter()
        ids = (list(encoded_ids) if encoded_ids is not None
               else list(target.tokenizer.encode(prompt).ids))
        tokenize_s = time.perf_counter() - tokenize_t0
        if not ids:
            raise ValueError("DSpark generation needs a non-empty prompt")
        if (target.effective_max_position_embeddings
                and len(ids) + max_tokens > target.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(ids)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={target.effective_max_position_embeddings}")
        if target.rc.context_bound and len(ids) + max_tokens > target.rc.context_bound:
            raise ValueError(
                f"context_bound={target.rc.context_bound} but prompt({len(ids)})"
                f"+max_tokens({max_tokens}) exceeds it")

        target.release_request_state()
        target._provisional = None
        target._true_peak_metal_bytes = mx.get_active_memory()
        if target.governor is not None:
            target.governor.reset_request_peak(target._true_peak_metal_bytes)
        use_stepped = bool(
            target.rc.stepped_kv_threshold
            and len(ids) + max_tokens > target.rc.stepped_kv_threshold)
        prompt_cache_key = (tuple(ids), use_stepped)
        cache_eligible = bool(
            self.prompt_cache_min_tokens
            and len(ids) >= self.prompt_cache_min_tokens)
        lookup_t0 = time.perf_counter()
        cached = (
            self._prompt_cache
            if cache_eligible and self._prompt_cache is not None
            and self._prompt_cache[0] == prompt_cache_key
            else None
        )
        prompt_cache_lookup_s = time.perf_counter() - lookup_t0
        prompt_cache_hit = cached is not None
        if prompt_cache_hit:
            _, target_kv, ctx_caches, prompt_last_logits = cached
            # Transfer sole ownership into the request.  A failed request must
            # not leave partially advanced state eligible for a later repeat.
            self._prompt_cache = None
        else:
            if cache_eligible:
                # Release a divergent cached prompt before allocating another
                # multi-hundred-MB target/context pair.
                self._prompt_cache = None
            target_kv = target.new_kv(stepped=use_stepped)
            ctx_caches = self.drafter.make_ctx_cache()
            prompt_last_logits = None
        taps = set(self.drafter.config.target_layer_ids)
        stats = DSparkStats()

        prefill_t0 = time.perf_counter()
        if prompt_cache_hit:
            # Keep the historical synthetic prefill sweep so target_sweeps-1
            # remains decode sweeps in telemetry.
            stats.target_sweeps += 1
        else:
            logits = target.forward_tokens(ids, target_kv, tap_layers=taps)
            prompt_last_logits = logits[-1]
            mx.eval(prompt_last_logits)
            stats.target_sweeps += 1
            context = self._tapped_context()
            ctx_t0 = time.perf_counter()
            self.drafter.update_context(context, 0, ctx_caches)
            mx.async_eval([cache.k for cache in ctx_caches])
            stats.context_s += time.perf_counter() - ctx_t0
            target._tap_hidden = {}
        pending = int(mx.argmax(prompt_last_logits))
        emitted = [pending]
        prefill_s = time.perf_counter() - prefill_t0
        first_token_s = time.perf_counter() - request_t0
        stop_text = None
        matched_stop_sequence = None
        if stops:
            match = self._stop_match(target.tokenizer.decode(emitted), stops)
            if match is not None:
                cut, _order, matched_stop_sequence = match
                stop_text = target.tokenizer.decode(emitted)[:cut]

        stream_decoder = None
        if on_token is not None:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(target.tokenizer, stops)
            if stop_text is None:
                delta = stream_decoder.push(emitted)
                if delta:
                    on_token(delta)
        if on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": len(ids),
                         "total_tokens": len(ids), "cache_source": (
                             "dspark-memory" if prompt_cache_hit else
                             "dspark-cold")})

        eos = set(target.cfg.eos_token_ids)
        decode_t0 = time.perf_counter()
        while (len(emitted) < max_tokens and pending not in eos
               and stop_text is None):
            # Every round emits at most cap accepted drafts plus one target
            # token.  Bounding cap keeps endpoint KV/context exact at the
            # caller's output limit without a speculative overrun.
            cap = min(
                self.max_draft_tokens,
                max_tokens - len(emitted) - 1,
            )
            base = target_kv.offset
            if any(cache.length != base for cache in ctx_caches):
                raise RuntimeError(
                    "DSpark context/target KV desync before proposal: "
                    f"kv={base}, ctx={[cache.length for cache in ctx_caches]}")

            if cap > 0:
                draft_t0 = time.perf_counter()
                proposals = self._propose(pending, base, ctx_caches, cap)
                stats.draft_s += time.perf_counter() - draft_t0
            else:
                # One output slot remains: verify the pending token alone and
                # emit its target successor.  Breaking here would silently
                # return max_tokens-1 whenever the prior round filled exactly
                # to that boundary.
                proposals = []

            verify_t0 = time.perf_counter()
            verify_ids = [pending] + proposals
            verified = target.forward_tokens_serial_positions(
                verify_ids, target_kv, tap_layers=taps)
            predictions = [int(token) for token in
                           mx.argmax(verified, axis=-1).reshape(-1).tolist()]
            stats.verify_s += time.perf_counter() - verify_t0
            stats.target_sweeps += 1

            accepted = 0
            while (accepted < len(proposals)
                   and proposals[accepted] == predictions[accepted]):
                accepted += 1
            committed = proposals[:accepted] + [predictions[accepted]]
            stats.proposed += len(proposals)
            stats.accepted += accepted
            stats.rounds.append((len(proposals), accepted))

            # Only the anchor and accepted proposal prefix are committed input
            # positions.  The target token is the next pending output and has
            # not yet entered either state.
            tapped = self._tapped_context()[:, :accepted + 1, :]
            target_kv.trim(base + accepted + 1)
            ctx_t0 = time.perf_counter()
            self.drafter.update_context(tapped, base, ctx_caches)
            mx.async_eval([cache.k for cache in ctx_caches])
            stats.context_s += time.perf_counter() - ctx_t0
            target._tap_hidden = {}

            for token in committed:
                emitted.append(token)
                pending = token
                if stops:
                    decoded = target.tokenizer.decode(emitted)
                    match = self._stop_match(decoded, stops)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                if stream_decoder is not None and stop_text is None:
                    delta = stream_decoder.push(emitted)
                    if delta:
                        on_token(delta)
                if (stop_text is not None or token in eos
                        or len(emitted) >= max_tokens):
                    break

            endpoint = len(ids) + len(emitted) - 1
            if target_kv.offset > endpoint:
                target_kv.trim(endpoint)
            for cache in ctx_caches:
                cache.trim_to(endpoint)

        stats.emitted = len(emitted)
        final_text = (stop_text if stop_text is not None
                      else target.tokenizer.decode(emitted))
        if stream_decoder is not None:
            delta = stream_decoder.finish(emitted, final_text=final_text)
            if delta:
                on_token(delta)
        decode_s = time.perf_counter() - decode_t0
        endpoint_kv_bytes = target_kv.nbytes()
        endpoint_kv_positions = target_kv.offset
        prompt_cache_stored = False
        if (cache_eligible and target_kv.offset >= len(ids)
                and all(cache.length >= len(ids) for cache in ctx_caches)):
            target_kv.trim(len(ids))
            for cache in ctx_caches:
                cache.trim_to(len(ids))
            mx.eval(prompt_last_logits)
            self._prompt_cache = (
                prompt_cache_key, target_kv, ctx_caches,
                prompt_last_logits)
            prompt_cache_stored = True
        total_s = time.perf_counter() - request_t0
        target._note_true_peak()
        if target.governor is not None:
            target._true_peak_metal_bytes = max(
                target._true_peak_metal_bytes,
                target.governor.request_peak(),
                mx.get_active_memory(),
            )
        return {
            "text": final_text,
            "tokens": emitted,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "first_token_s": first_token_s,
            "total_s": total_s,
            "tok_per_s": ((len(emitted) - 1) / decode_s
                          if len(emitted) > 1 else 0.0),
            "kv_bytes": endpoint_kv_bytes,
            "kv_positions": endpoint_kv_positions,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": (
                "stop_sequence" if stop_text is not None else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": target._true_peak_metal_bytes,
            "prompt_tokens": len(ids),
            "stats": stats,
            "path_stats": {
                "prompt_cache_exact_hit": int(prompt_cache_hit),
                "prompt_cache_prefix_tokens": (
                    len(ids) if prompt_cache_hit else 0),
                "prompt_cache_source": (
                    "dspark-memory" if prompt_cache_hit else "dspark-cold"),
                "prompt_cache_lookup_s": prompt_cache_lookup_s,
                "prompt_cache_write_tokens": (
                    len(ids) if prompt_cache_stored else 0),
                "prompt_tokenize_s": tokenize_s,
                "prompt_snapshot_write_s": 0.0,
                "postgen_snapshot_write_s": 0.0,
                "rope_profile": target.rope_profile,
                "effective_context_limit": target.effective_max_position_embeddings,
                "speculative_enabled": 1,
                "speculative_used": 1,
                "speculative_kind": "dspark",
                "speculative_k": self.max_draft_tokens,
                "speculative_target_sweeps": max(0, stats.target_sweeps - 1),
                "speculative_proposed": stats.proposed,
                "speculative_accepted": stats.accepted,
                "speculative_draft_oov_fallbacks": 0,
                "dspark_block_size": self.drafter.block_size,
                "dspark_confidence_threshold": self.confidence_threshold,
                "dspark_context_s": stats.context_s,
            },
        }


class DSparkSpeculativeEngine:
    """Serving adapter for a streamed Qwen3 target and DSpark checkpoint."""

    def __init__(self, target, draft_dir: str | Path, *,
                 max_draft_tokens: int = 4,
                 max_prompt_tokens: int = 2048,
                 confidence_threshold: float = 0.0,
                 prompt_cache_min_tokens: int = 2048):
        if max_prompt_tokens <= 0:
            raise ValueError("DSpark max_prompt_tokens must be positive")
        self.target = target
        draft_dir = Path(draft_dir)
        if target.governor is not None:
            # The drafter is a persistent resident allocation outside the
            # target WeightCache.  Admit it against the same sampled
            # device/system ceiling before materializing its BF16 tensors.
            try:
                draft_bytes = (draft_dir / "model.safetensors").stat().st_size
            except OSError:
                draft_bytes = 0
            if draft_bytes:
                target.governor.reserve(draft_bytes)
        self.drafter = DSparkDrafter.load(draft_dir)
        if target.governor is not None:
            # A large configured target cache is a performance ceiling, not a
            # fixed machine cap.  Fit its *additional* residency around the
            # now-live drafter; the governor may restore it as headroom grows.
            target.governor.fit_cache_to_live_headroom()
        self.decoder = DSparkSpeculativeDecoder(
            target, self.drafter,
            max_draft_tokens=max_draft_tokens,
            confidence_threshold=confidence_threshold,
            prompt_cache_min_tokens=prompt_cache_min_tokens,
        )
        self.max_prompt_tokens = max_prompt_tokens
        self._speculative_k = self.decoder.max_draft_tokens
        self._speculative_draft_dir = draft_dir
        self._speculative_kind = "dspark"
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt: str, max_tokens: int,
                         on_token=None, stop=None, on_progress=None,
                         sampling: SamplingParams | None = None,
                         constraint=None):
        self.decoder.clear_prompt_cache()
        kwargs = {"on_token": on_token, "stop": stop,
                  "on_progress": on_progress}
        if sampling is not None:
            kwargs["sampling"] = sampling
        if constraint is not None:
            kwargs["constraint"] = constraint
        result = self.target.generate(prompt, max_tokens, **kwargs)
        result.setdefault("path_stats", {}).update({
            "speculative_enabled": 1,
            "speculative_used": 0,
            "speculative_kind": "dspark",
            "speculative_fallback_reason": reason,
            "speculative_k": self._speculative_k,
        })
        return result

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None,
                 sampling: SamplingParams | None = None,
                 constraint=None):
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
        return self.decoder.generate(
            prompt, max_tokens, on_token=on_token, stop=stop,
            on_progress=on_progress, encoded_ids=ids)

    def release_request_state(self):
        self.decoder.clear_prompt_cache()
        self.target.release_request_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decoder.clear_prompt_cache()
        self.decoder = None
        self.drafter = None
        try:
            self.target.close()
        finally:
            mx.clear_cache()
