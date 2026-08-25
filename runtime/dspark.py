"""F62/F190: DSpark drafter (DeepSpec, arXiv:2607.05147) for standalone
Qwen3 checkpoints and Inferact's public Kimi K3 DSpark checkpoint.

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

K3 uses the same block/Markov/confidence contract with five target taps, but
its five draft layers use compressed multi-head latent attention (MLA).  The
port retains only normalized KV latents plus RoPE keys in its persistent
context cache and expands K/V inside each block, matching the released
checkpoint's tensor names and DeepSeek-YaRN parameters.  K3's checkpoint has
no private LM head, so the drafter shares the target's exact streamed output
projection.  A lossy draft representation is safe: every proposed token still
passes the released K3 target verifier below.

Safety design unchanged from every other drafter in this runtime: the target
computes the authoritative distribution for every proposed position.  With a
correct architecture-specific recurrent rollback, a wrong or low-quality
drafter can only reduce acceptance rate (speed), never change the emitted token
stream.  K3 has released-weight rollback gates; Qwen3.8 stays experimental
until its longer forced-rejection recurrent oracle passes.  See
experiments/dspark_control.py for the strict token-identity gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .sampler import (
    SamplingParams,
    filtered_probabilities,
    sample_probabilities,
    speculative_residual_probabilities,
)


def _target_io_snapshot(target):
    """Snapshot target-page I/O when the wrapped engine exposes accounting.

    DSpark/DFlash verification runs outside ``StreamingEngine.generate`` after
    the bootstrap prefill.  Without a wrapper-level snapshot, the bootstrap's
    path statistics are returned unchanged and every authoritative decode
    sweep is incorrectly reported as zero target-weight traffic.  Tiny oracle
    targets deliberately omit the cache/store surface and remain unaffected.
    """
    if not all(hasattr(target, name) for name in (
        "cache", "store", "expert_hits", "expert_misses",
    )):
        return None
    from .engine import _cache_io_snapshot

    return _cache_io_snapshot(target)


def _merge_target_io_stats(
    target, before, prefill_after, request_after, path_stats: dict,
) -> None:
    """Replace bootstrap-only counters with whole speculative-request deltas."""
    if before is None or prefill_after is None or request_after is None:
        return
    from .engine import _record_cache_io_delta

    _record_cache_io_delta(
        target, before, path_stats, after=request_after)
    _record_cache_io_delta(
        target, before, path_stats, prefix="prefill_", after=prefill_after)
    _record_cache_io_delta(
        target, prefill_after, path_stats, prefix="decode_",
        after=request_after)


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
    model_type: str = "qwen3"
    target_model_type: str = ""
    share_target_embed: bool = False
    share_target_lm_head: bool = False
    logits_start: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    target_hidden_size: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    max_position_embeddings: int = 0
    rope_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_model_type:
            self.target_model_type = (
                "kimi_k3" if self.model_type == "k3_dspark" else "qwen3")
        if self.model_type == "k3_dspark":
            self.share_target_lm_head = True

    @classmethod
    def from_json(cls, path: str | Path) -> "DSparkConfig":
        c = json.loads(Path(path).read_text())
        if "speculators_config" in c or "speculators_model_type" in c:
            raise ValueError(
                f"{path}: vLLM 'speculators' packaging (speculators_config present) — "
                f"different tensor/config schema, not supported by this loader. Needs a "
                f"DeepSpec-native standalone drafter (e.g. deepseek-ai/dspark_*_block7).")
        model_type = c.get("model_type")
        dflash = c.get("dflash_config") or {}
        specforge = bool(
            model_type == "qwen3"
            and dflash.get("projector_type") == "dspark")
        if model_type not in ("qwen3", "k3_dspark"):
            raise ValueError(
                f"{path}: unsupported DSpark model_type={model_type!r}")
        if specforge:
            model_type = "qwen3_specforge"
        rp = c.get("rope_parameters") or {}
        if model_type == "k3_dspark":
            head_dim = (
                int(c["qk_nope_head_dim"])
                + int(c["qk_rope_head_dim"])
            )
        else:
            head_dim = c.get(
                "head_dim",
                c["hidden_size"] // c["num_attention_heads"],
            )
        return cls(
            hidden_size=c["hidden_size"], vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            head_dim=head_dim,
            # Inferact's first public K3 DSpark config omitted block_size even
            # though the released checkpoint/model card fixes it at seven.
            block_size=c.get("block_size", 7),
            mask_token_id=(
                dflash["mask_token_id"] if specforge else c["mask_token_id"]),
            target_layer_ids=list(
                dflash["target_layer_ids"] if specforge
                else c["target_layer_ids"]),
            model_type=model_type,
            target_model_type=((
                "qwen3_5"
                if (specforge and int(c.get("num_target_layers", 0)) == 64
                    and int(c.get("vocab_size", 0)) == 248_320)
                else "qwen3") if specforge else (
                "kimi_k3" if model_type == "k3_dspark" else "qwen3")),
            share_target_embed=specforge,
            share_target_lm_head=specforge or model_type == "k3_dspark",
            # RadixArk's SpecForge head predicts from its anchor slot. The
            # inherited base-DFlash example used slot one and measured about
            # 1.35 accepted tokens versus 3.42 from slot zero.
            logits_start=(0 if specforge else int(c.get("logits_start", 0))),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
            attention_bias=c.get("attention_bias", False),
            markov_rank=c.get("markov_rank", 256),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
            target_hidden_size=c.get(
                "target_hidden_size", c["hidden_size"]),
            q_lora_rank=c.get("q_lora_rank", 0) or 0,
            kv_lora_rank=c.get("kv_lora_rank", 0) or 0,
            qk_nope_head_dim=c.get("qk_nope_head_dim", 0) or 0,
            qk_rope_head_dim=c.get("qk_rope_head_dim", 0) or 0,
            v_head_dim=c.get("v_head_dim", 0) or 0,
            max_position_embeddings=c.get("max_position_embeddings", 0) or 0,
            rope_parameters=dict(rp),
        )


class CtxCache:
    """Per-layer cache of the drafter's cross-attention K/V, projected from
    committed target-hidden-state fusions. Append-only — grows with commits,
    never trimmed (unlike the target's own KV cache; there is no spec-decode
    rollback concept on the CONTEXT side, only on the target's own state)."""

    def __init__(self):
        self.k: mx.array | None = None
        self.v: mx.array | None = None
        self.position_start: int | None = None
        self.position_end: int = 0

    def append(
        self, k: mx.array, v: mx.array, *, position_start: int | None = None,
    ) -> None:
        position_start = (
            self.position_end if position_start is None
            else int(position_start))
        if self.k is None:
            self.k, self.v = k, v
            self.position_start = position_start
        else:
            if position_start != self.position_end:
                raise ValueError(
                    "DSpark context append is not position-contiguous: "
                    f"expected {self.position_end}, got {position_start}")
            self.k = mx.concatenate([self.k, k], axis=2)
            self.v = mx.concatenate([self.v, v], axis=2)
        self.position_end = position_start + int(k.shape[2])

    def fork(self) -> "CtxCache":
        result = CtxCache()
        result.k = self.k
        result.v = self.v
        result.position_start = self.position_start
        result.position_end = self.position_end
        return result

    def trim_to(self, length: int) -> None:
        """Keep context entries strictly before absolute ``length``."""
        if length < 0:
            raise ValueError("DSpark context length must be non-negative")
        start = int(self.position_start or 0)
        keep = max(0, int(length) - start)
        if self.k is not None and keep < self.k.shape[2]:
            self.k = self.k[:, :, :keep, :]
            self.v = self.v[:, :, :keep, :]
            self.position_end = start + keep

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class DSparkTapCollector:
    """Turn target residual-stream taps into the drafter's context cache.

    The normal lossless prefill presents the same contiguous position range at
    every tapped target layer.  The explicit Huihui fast profile is different:
    its early layers see the complete prompt while later layers see only the
    final endpoint-packed suffix.  The deepest tap is authoritative about the
    positions common to every feature source; earlier full-prompt taps are
    sliced to that range before fusion.  Non-contiguous endpoint packing fails
    closed for now because the drafter RoPE API below deliberately accepts one
    contiguous absolute range at a time.
    """

    def __init__(
        self, drafter: "DSparkDrafter | None", caches: list[CtxCache], *,
        position_floor: int = 0,
        tap_layers: tuple[int, ...] | list[int] | None = None,
        defer_updates: bool = False,
    ):
        self.drafter = drafter
        self.caches = caches
        if int(position_floor) < 0:
            raise ValueError("DSpark context position floor must be non-negative")
        self.position_floor = int(position_floor)
        if tap_layers is None:
            if drafter is None:
                raise ValueError(
                    "deferred DSpark tap collection requires tap_layers")
            tap_layers = drafter.config.target_layer_ids
        self.tap_layers = tuple(int(v) for v in tap_layers)
        if not self.tap_layers:
            raise ValueError("DSpark tap collection requires at least one layer")
        self.defer_updates = bool(defer_updates)
        if drafter is None and not self.defer_updates:
            raise ValueError("immediate DSpark tap collection needs a drafter")
        self._wanted = set(self.tap_layers)
        self._seen: dict[int, tuple[int, int, mx.array]] = {}
        self._deferred_chunks: list[tuple[int, mx.array]] = []
        self.deferred_bytes_peak = 0
        self._deferred_end: int | None = None
        self.updates = 0
        self.positions = 0
        self.update_s = 0.0
        self.attempts = 0

    def begin_attempt(self) -> None:
        """Discard partial tap context before a memory-retry replay.

        ``generate_with_memory_retry`` can restart Qwen prefill after a governor
        refusal.  Target KV is discarded by the engine, so the companion draft
        context must restart at the same boundary rather than append a second
        copy of whatever layers completed before the refusal.
        """
        self.attempts += 1
        self._seen.clear()
        self._deferred_chunks.clear()
        self.deferred_bytes_peak = 0
        self._deferred_end = None
        self.updates = 0
        self.positions = 0
        self.update_s = 0.0
        for cache in self.caches:
            cache.k = None
            cache.v = None
            cache.position_start = None
            cache.position_end = 0

    @staticmethod
    def _contiguous_start(
        hidden: mx.array, position_start: int | None,
        positions: mx.array | None,
    ) -> int:
        width = int(hidden.shape[1])
        if positions is None:
            if position_start is None:
                raise ValueError("DSpark tap omitted its absolute position")
            return int(position_start)
        mx.eval(positions)
        values = [int(value) for value in positions.tolist()]
        if len(values) != width or any(
            value != values[0] + index
            for index, value in enumerate(values)
        ):
            raise ValueError(
                "DSpark tap collection currently requires contiguous "
                "endpoint positions")
        return values[0]

    def observe(
        self, layer: int, hidden: mx.array, *,
        position_start: int | None = None,
        positions: mx.array | None = None,
    ) -> None:
        layer = int(layer)
        if layer not in self._wanted:
            return
        start = self._contiguous_start(hidden, position_start, positions)
        self._seen[layer] = (start, start + int(hidden.shape[1]), hidden)
        if not all(value in self._seen for value in self.tap_layers):
            return

        canonical_start, canonical_end, _ = self._seen[self.tap_layers[-1]]
        if canonical_end <= self.position_floor:
            self._seen.clear()
            return
        canonical_start = max(canonical_start, self.position_floor)
        aligned = []
        for tap in self.tap_layers:
            start_i, end_i, hidden_i = self._seen[tap]
            if start_i > canonical_start or end_i < canonical_end:
                raise ValueError(
                    "DSpark target taps do not share the deepest contiguous "
                    f"range: layer {tap}=[{start_i},{end_i}) versus "
                    f"[{canonical_start},{canonical_end})")
            left = canonical_start - start_i
            aligned.append(hidden_i[:, left:left + canonical_end - canonical_start, :])

        fused = mx.concatenate(aligned, axis=-1)
        if self.defer_updates:
            if (self._deferred_end is not None
                    and canonical_start != self._deferred_end):
                raise ValueError(
                    "deferred DSpark taps are not contiguous: "
                    f"expected {self._deferred_end}, got {canonical_start}")
            # Materialize now so the retained bounded window does not keep the
            # target layers' larger lazy graphs alive until the sidecar loads.
            mx.eval(fused)
            self._deferred_chunks.append((canonical_start, fused))
            self._deferred_end = canonical_end
            retained = sum(
                int(value.nbytes) for _offset, value in self._deferred_chunks)
            self.deferred_bytes_peak = max(
                self.deferred_bytes_peak, retained)
        else:
            started = time.perf_counter()
            self.drafter.update_context(
                fused,
                canonical_start,
                self.caches,
            )
            mx.eval(*[
                value
                for cache in self.caches
                for value in (cache.k, cache.v)
                if value is not None
            ])
            self.update_s += time.perf_counter() - started
            self.updates += 1
        self.positions += canonical_end - canonical_start
        self._seen.clear()

    def finish(self, expected_end: int) -> None:
        if self._seen:
            raise RuntimeError(
                "target prefill ended with an incomplete DSpark tap set: "
                f"{sorted(self._seen)}")
        if self.defer_updates:
            if self._deferred_end != int(expected_end):
                raise RuntimeError(
                    "deferred DSpark taps did not reach the target prompt "
                    "endpoint")
            return
        if not self.caches or any(
                cache.position_end != int(expected_end) for cache in self.caches):
            raise RuntimeError(
                "DSpark context did not reach the target prompt endpoint")

    def materialize(self, drafter: "DSparkDrafter") -> None:
        """Project a bounded retained tap window after target prefill.

        DFlash2's quantized sidecar is intentionally absent while the larger
        target streams its prompt layers. Only fused released target residuals
        are retained, then projected into request-local draft K/V afterward.
        """
        if not self.defer_updates:
            raise RuntimeError("immediate DSpark collector cannot materialize")
        if any(cache.length for cache in self.caches):
            raise RuntimeError("deferred DSpark context cache is not empty")
        started = time.perf_counter()
        for offset, fused in self._deferred_chunks:
            drafter.update_context(fused, offset, self.caches)
            mx.eval(*[
                value
                for cache in self.caches
                for value in (cache.k, cache.v)
                if value is not None
            ])
            self.updates += 1
        self.update_s += time.perf_counter() - started
        if not self.caches or any(
                cache.position_end != self._deferred_end
                for cache in self.caches):
            raise RuntimeError(
                "materialized DSpark context missed the retained endpoint")
        self._deferred_chunks.clear()


class _DSparkBootstrapPrompt(str):
    """Prepared-prompt-compatible view with an isolated hot-cache lineage."""

    def __new__(cls, prompt, token_ids):
        instance = super().__new__(cls, str(prompt))
        instance.token_ids = tuple(int(value) for value in token_ids)
        instance.tool_capsules = tuple(
            getattr(prompt, "tool_capsules", ()))
        base_namespace = str(
            getattr(prompt, "cache_namespace", "default") or "default")
        digest = hashlib.sha256(
            b"".join(int(value).to_bytes(4, "little") for value in token_ids)
        ).hexdigest()[:16]
        instance.cache_namespace = f"dspark:{base_namespace}:{digest}"
        instance.force_paged_kv = bool(
            getattr(prompt, "force_paged_kv", False))
        # A target-only checkpoint cannot restore the companion DSpark context
        # after a process restart. Keep the complete target+draft endpoint in
        # DSpark's own in-process COW cache below, but never admit this bootstrap
        # into the ordinary target slot/journal tiers where the absent draft
        # state could turn a future apparent hit into an invalid partial prefill.
        instance.disable_hot_prompt_kv = True
        # The target's ordinary server-only paged profile drops its cache
        # before returning. DSpark immediately transfers that exact endpoint
        # into its verifier, so retain it for this internal bootstrap only.
        instance.retain_paged_kv_after_generate = True
        instance.stable_boundary_tokens = 0
        instance.rerank_capture_shape = dict(
            getattr(prompt, "rerank_capture_shape", {}) or {})
        return instance


class DSparkAttention(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta
        self._rope_freqs = None
        self._rope_scale = 1.0
        rope_type = cfg.rope_parameters.get(
            "rope_type", cfg.rope_parameters.get("type", "default"))
        if rope_type == "yarn":
            from .rope import yarn_parameters

            rp = cfg.rope_parameters
            denominators, rope_scale = yarn_parameters(
                self.head_dim,
                self.rope_theta,
                float(rp["factor"]),
                int(rp["original_max_position_embeddings"]),
                beta_fast=float(rp.get("beta_fast", 32.0)),
                beta_slow=float(rp.get("beta_slow", 1.0)),
            )
            self._rope_freqs = mx.array(denominators, dtype=mx.float32)
            self._rope_scale = float(rope_scale)

        h, b = cfg.hidden_size, cfg.attention_bias
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=b)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=b)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=b)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def _rope(self, x: mx.array, offset) -> mx.array:
        # matches runtime/layer_runner.py's existing Qwen convention exactly
        if self._rope_freqs is None:
            value = mx.fast.rope(
                x, self.head_dim, traditional=False, base=self.rope_theta,
                scale=1.0, offset=offset)
        else:
            value = mx.fast.rope(
                x, self.head_dim, traditional=False, base=None,
                freqs=self._rope_freqs, scale=1.0, offset=offset)
        return value * self._rope_scale

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
        cache.append(
            self._rope(k, offset=ctx_offset), v,
            position_start=ctx_offset)

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


class K3DSparkAttention(nn.Module):
    """The released K3 draft's bidirectional-in-block MLA.

    Context cache entries remain in Moonshot's compressed
    ``[normalized c_kv, roped k_pe]`` representation.  Only the seven draft
    block rows are expanded through ``kv_b_proj`` on each proposal, so the
    persistent draft cache is 576 BF16 values/token/layer rather than
    64*(128+128).
    """

    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.nope_dim = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.scale = (self.nope_dim + self.rope_dim) ** -0.5
        self.rope_theta = cfg.rope_theta
        self.rope_parameters = cfg.rope_parameters

        h = cfg.hidden_size
        self.q_a_proj = nn.Linear(h, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(
            self.q_lora_rank, eps=cfg.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.n_heads * (self.nope_dim + self.rope_dim),
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            h, self.kv_lora_rank + self.rope_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(
            self.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.nope_dim + self.v_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.v_dim, h, bias=False)

        self._rope_freqs = None
        self._rope_scale = 1.0
        rope_type = self.rope_parameters.get(
            "rope_type", self.rope_parameters.get("type", "default"))
        if rope_type == "yarn":
            from .rope import yarn_parameters

            rp = self.rope_parameters
            denominators, rope_scale = yarn_parameters(
                self.rope_dim,
                self.rope_theta,
                float(rp["factor"]),
                int(rp["original_max_position_embeddings"]),
                beta_fast=float(rp.get("beta_fast", 32.0)),
                beta_slow=float(rp.get("beta_slow", 1.0)),
                mscale=float(rp.get("mscale", 1.0)),
                mscale_all_dim=float(rp.get("mscale_all_dim", 0.0)),
            )
            self._rope_freqs = mx.array(
                denominators, dtype=mx.float32)
            self._rope_scale = float(rope_scale)
            mscale_all_dim = float(rp.get("mscale_all_dim", 0.0))
            if mscale_all_dim:
                factor = float(rp["factor"])
                mscale = (
                    1.0 if factor <= 1
                    else 0.1 * mscale_all_dim * math.log(factor) + 1.0
                )
                self.scale *= mscale * mscale

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._rope_freqs is None:
            value = mx.fast.rope(
                x, self.rope_dim, traditional=False,
                base=self.rope_theta, scale=1.0, offset=offset)
        else:
            value = mx.fast.rope(
                x, self.rope_dim, traditional=False, base=None,
                freqs=self._rope_freqs, scale=1.0, offset=offset)
        return value * self._rope_scale

    def _latent(self, x: mx.array, offset: int):
        b, length, _ = x.shape
        value = self.kv_a_proj_with_mqa(x)
        c_kv = self.kv_a_layernorm(value[..., :self.kv_lora_rank])
        k_pe = value[..., self.kv_lora_rank:]
        k_pe = k_pe.reshape(
            b, length, 1, self.rope_dim
        ).transpose(0, 2, 1, 3)
        return c_kv, self._rope(k_pe, offset)

    def update_ctx(
        self, fused_new: mx.array, ctx_offset: int, cache: CtxCache
    ) -> None:
        c_kv, k_pe = self._latent(fused_new, ctx_offset)
        # CtxCache's names are deliberately representation-agnostic: K3's
        # ``k`` is the compressed content latent and ``v`` is shared k_pe.
        cache.append(
            c_kv[:, None, :, :], k_pe,
            position_start=ctx_offset)

    def attend(
        self, hidden: mx.array, block_offset: int, cache: CtxCache
    ) -> mx.array:
        b, q_len, _ = hidden.shape
        q_a = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(q_a).reshape(
            b, q_len, self.n_heads, self.nope_dim + self.rope_dim
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = q[..., :self.nope_dim], q[..., self.nope_dim:]
        q = mx.concatenate([q_nope, self._rope(q_pe, block_offset)], axis=-1)

        c_blk, k_pe_blk = self._latent(hidden, block_offset)
        c_ctx = (
            cache.k[:, 0, :, :] if cache.k is not None
            else mx.zeros((b, 0, self.kv_lora_rank), dtype=c_blk.dtype)
        )
        k_pe_ctx = (
            cache.v if cache.v is not None
            else mx.zeros((b, 1, 0, self.rope_dim), dtype=k_pe_blk.dtype)
        )
        c_all = mx.concatenate([c_ctx, c_blk], axis=1)
        k_pe_all = mx.concatenate([k_pe_ctx, k_pe_blk], axis=2)
        kv = self.kv_b_proj(c_all).reshape(
            b, c_all.shape[1], self.n_heads, self.nope_dim + self.v_dim
        ).transpose(0, 2, 1, 3)
        k_nope, values = kv[..., :self.nope_dim], kv[..., self.nope_dim:]
        keys = mx.concatenate(
            [
                k_nope,
                mx.broadcast_to(
                    k_pe_all,
                    (b, self.n_heads, c_all.shape[1], self.rope_dim),
                ),
            ],
            axis=-1,
        )
        out = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(
            b, q_len, self.n_heads * self.v_dim)
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


class K3DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: DSparkConfig):
        super().__init__()
        self.self_attn = K3DSparkAttention(cfg)
        self.mlp = DSparkMLP(cfg)
        self.input_layernorm = nn.RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self, hidden: mx.array, block_offset: int, cache: CtxCache
    ) -> mx.array:
        hidden = hidden + self.self_attn.attend(
            self.input_layernorm(hidden), block_offset, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


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

        self.embed_tokens = (
            None if cfg.share_target_embed
            else nn.Embedding(cfg.vocab_size, cfg.hidden_size))
        self._target_embed_provider = None
        self._target_lm_head_provider = None
        if cfg.model_type == "k3_dspark":
            self.context_proj = nn.Linear(
                len(cfg.target_layer_ids) * cfg.target_hidden_size,
                cfg.hidden_size,
                bias=False,
            )
            self.context_norm = nn.RMSNorm(
                cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.layers = [
                K3DSparkDecoderLayer(cfg)
                for _ in range(cfg.num_hidden_layers)
            ]
            self.final_norm = nn.RMSNorm(
                cfg.hidden_size, eps=cfg.rms_norm_eps)
        else:
            self.fc = nn.Linear(
                len(cfg.target_layer_ids) * cfg.hidden_size,
                cfg.hidden_size,
                bias=False,
            )
            self.hidden_norm = nn.RMSNorm(
                cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.layers = [
                DSparkDecoderLayer(cfg)
                for _ in range(cfg.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(
                cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.lm_head = (
                None if cfg.share_target_lm_head
                else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False))

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
        raw = json.loads((model_dir / "config.json").read_text())
        path = model_dir / "model.safetensors"
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            physical_names = set(
                json.loads(index_path.read_text())["weight_map"])
        elif path.is_file():
            physical_names = set(mx.load(str(path)))
        else:
            physical_names = set()
        quantization = raw.get("quantization") or raw.get(
            "quantization_config")
        if quantization:
            nn.quantize(
                model,
                group_size=int(quantization["group_size"]),
                bits=int(quantization["bits"]),
                mode=quantization.get("mode", "affine"),
                # Select from the checkpoint's physical .scales keys.  The
                # draft's Markov table is an Embedding and is quantized by the
                # portable converter, while embed_tokens stays exact.
                class_predicate=lambda module_path, _module: (
                    f"{module_path}.scales" in physical_names),
            )
        if path.is_file():
            model.load_weights(str(path))
        else:
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"no DSpark safetensors checkpoint in {model_dir}")
            index = json.loads(index_path.read_text())
            weights = []
            for name in sorted(set(index["weight_map"].values())):
                weights.extend(mx.load(str(model_dir / name)).items())
            model.load_weights(weights)
        mx.eval(model.parameters())
        return model

    def make_ctx_cache(self) -> list[CtxCache]:
        return [CtxCache() for _ in self.layers]

    def fuse_target(self, target_hidden_cat: mx.array) -> mx.array:
        """target_hidden_cat: (1, S, n_taps*hidden), the concatenation (in
        target_layer_ids order) of the target's tapped hidden states."""
        if self.config.model_type == "k3_dspark":
            return self.context_norm(self.context_proj(target_hidden_cat))
        return self.hidden_norm(self.fc(target_hidden_cat))

    def bind_target_lm_head(self, provider) -> None:
        if not self.config.share_target_lm_head:
            raise ValueError("this DSpark checkpoint owns its LM head")
        self._target_lm_head_provider = provider

    def bind_target_embed(self, provider) -> None:
        if not self.config.share_target_embed:
            raise ValueError("this DSpark checkpoint owns its embedding")
        self._target_embed_provider = provider

    def _embed_block(self, token_ids: list[int]) -> mx.array:
        if self.embed_tokens is not None:
            return self.embed_tokens(mx.array([token_ids]))
        if self._target_embed_provider is None:
            raise RuntimeError("DSpark requires the target token embedding")
        value = self._target_embed_provider(token_ids)
        if value.ndim != 3 or value.shape[:2] != (1, len(token_ids)):
            raise RuntimeError(
                "DSpark target embedding provider returned an invalid shape: "
                f"{tuple(value.shape)}")
        return value

    def _project_logits(self, hidden: mx.array) -> mx.array:
        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(hidden)
        if self._target_lm_head_provider is None:
            raise RuntimeError("DSpark requires the target LM head")
        head = self._target_lm_head_provider()
        from .lm_head_stream import StreamedLMHead

        if isinstance(head, StreamedLMHead):
            return head.logits(hidden)
        from .layer_runner import _linear

        return _linear(
            hidden, {"draft_target_head.weight": head},
            "draft_target_head")

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
        h = self._embed_block(block_ids)
        for layer, cache in zip(self.layers, ctx_caches):
            h = layer(h, block_offset, cache)
        h = (
            self.final_norm(h)
            if self.config.model_type == "k3_dspark"
            else self.norm(h)
        )
        start = int(self.config.logits_start)
        head_hidden = h[:, start:start + cap, :]
        return self._project_logits(head_hidden)[0], head_hidden[0]

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

    def sample_block_exact(
        self,
        base_logits: mx.array,
        first_prev_token: int,
        sampling: SamplingParams,
        history: list[int],
    ) -> tuple[list[int], list[mx.array]]:
        """Sample proposals and retain their exact filtered ``q`` vectors."""
        proposals: list[int] = []
        distributions: list[mx.array] = []
        prev = first_prev_token
        for index in range(base_logits.shape[0]):
            logits = base_logits[index]
            if self.markov_head is not None:
                logits = logits + self.markov_head.step_bias(
                    mx.array([prev]))[0]
            q = filtered_probabilities(
                logits, sampling, history=history + proposals)
            token = sample_probabilities(q)
            proposals.append(token)
            distributions.append(q)
            prev = token
        return proposals, distributions

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
    kda_factor_bytes_peak: int = 0
    kda_factor_restores: int = 0
    kda_factor_restore_s: float = 0.0
    kda_refeed_sweeps: int = 0
    qwen_kda_endpoint_restores: int = 0
    tree_nodes_verified: int = 0
    tree_paths_committed: int = 0
    tree_factor_bytes_peak: int = 0
    tree_factor_commit_s: float = 0.0
    tree_rounds: list[tuple[int, int]] = field(default_factory=list)
    expert_prefetch_router_bytes: int = 0
    expert_prefetch_planned_bytes: int = 0
    expert_prefetch_pages: int = 0
    expert_prefetch_predicted: int = 0
    expert_prefetch_authoritative: int = 0
    expert_prefetch_matched: int = 0
    sidecar_round_loads: int = 0
    sidecar_round_releases: int = 0
    sidecar_load_s: float = 0.0
    sidecar_release_s: float = 0.0
    sidecar_loaded_bytes: int = 0
    sidecar_released_active_bytes: int = 0
    deferred_context_positions: int = 0
    deferred_context_bytes_peak: int = 0
    draft_context_suspend_rounds: int = 0
    draft_context_restore_rounds: int = 0
    draft_context_suspend_s: float = 0.0
    draft_context_restore_s: float = 0.0
    draft_context_suspended_bytes: int = 0
    draft_context_released_active_bytes: int = 0
    rounds: list[tuple[int, int]] = field(default_factory=list)
    round_draft_s: list[float] = field(default_factory=list)
    round_verify_s: list[float] = field(default_factory=list)
    round_context_s: list[float] = field(default_factory=list)


class DSparkSpeculativeDecoder:
    """Exact target verification for a resident DSpark block drafter.

    The target KV and drafter context both hold every committed input up to,
    but never including, ``pending``.  A round verifies
    ``[pending] + proposals`` with ordinary one-token arithmetic shapes, keeps
    ``pending + accepted proposals`` in both states, and emits the accepted
    prefix plus one target token.  Greedy acceptance is equality; stochastic
    requests use standard rejection sampling with normalized ``(p-q)+``
    correction.  Thus draft quality affects only speed.
    """

    def __init__(self, target, drafter: DSparkDrafter, *,
                 max_draft_tokens: int = 4,
                 confidence_threshold: float = 0.0,
                 prompt_cache_min_tokens: int = 2048,
                 context_window_tokens: int = 0,
                 expert_prefetcher=None,
                 expert_prefetch_depth: int = 0,
                 native_kda_factor_commit: bool = False,
                 drafter_loader=None,
                 release_between_sweeps: bool = False,
                 drafter_storage_bytes: int = 0,
                 drafter_load_margin_bytes: int = 0):
        cfg = drafter.config
        if max_draft_tokens <= 0:
            raise ValueError("DSpark max_draft_tokens must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("DSpark confidence_threshold must be in [0, 1]")
        if prompt_cache_min_tokens < 0:
            raise ValueError("DSpark prompt_cache_min_tokens must be >= 0")
        if context_window_tokens < 0:
            raise ValueError("DSpark context_window_tokens must be >= 0")
        draft_model_type = getattr(cfg, "model_type", "qwen3")
        expected_target = getattr(cfg, "target_model_type", "qwen3")
        if target.cfg.model_type != expected_target:
            raise ValueError(
                f"DSpark {draft_model_type} checkpoint needs a "
                f"{expected_target} target, got {target.cfg.model_type!r}")
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
        self._cfg = cfg
        self._drafter_loader = drafter_loader
        self.release_between_sweeps = bool(release_between_sweeps)
        self.drafter_storage_bytes = max(0, int(drafter_storage_bytes))
        self.drafter_load_margin_bytes = max(
            0, int(drafter_load_margin_bytes))
        self.max_draft_tokens = min(
            max_draft_tokens,
            cfg.block_size - int(getattr(cfg, "logits_start", 0)))
        self.confidence_threshold = confidence_threshold
        self.prompt_cache_min_tokens = prompt_cache_min_tokens
        self.context_window_tokens = int(context_window_tokens)
        self._prompt_cache = None
        self.expert_prefetcher = expert_prefetcher
        self.expert_prefetch_depth = max(0, int(expert_prefetch_depth))
        self._active_expert_prefetch_plan = None
        # Experimental only: changes float32 reduction order.  The plain MLX
        # factor replay remains the lossless default until a released-model
        # greedy token-identity gate admits this kernel.
        self.native_kda_factor_commit = bool(native_kda_factor_commit)
        # Large proposal models can retain a bounded target-tap window during
        # prompt streaming and project it only after the target sweep.
        self.defer_prompt_context_updates = False

    def _new_ctx_caches(self) -> list[CtxCache]:
        return self._ensure_drafter().make_ctx_cache()

    def _note_verified_round(self, proposed: int, accepted: int) -> None:
        """Optional proposal-policy feedback after target verification.

        The base decoder is intentionally inert.  Specialized draft sources
        can use this target-owned evidence to change scheduling in a later
        round without influencing the token distribution being served.
        """
        del proposed, accepted

    def _draft_context_required(self) -> bool:
        """Whether the next proposal consumes the learned draft K/V cache."""
        return True

    def _suspend_draft_context(
        self, ctx_caches: list[CtxCache], stats: DSparkStats,
    ):
        """Optionally move draft-only context out of the target Metal sweep.

        The base decoder keeps its historical resident-context behavior.
        Large specialized drafters can override this hook, but the suspended
        representation must preserve every bit because it is proposal state
        reused by later rounds.
        """
        del ctx_caches, stats
        return None

    def _restore_draft_context(
        self, ctx_caches: list[CtxCache], snapshot, stats: DSparkStats,
    ) -> None:
        del ctx_caches, snapshot, stats

    def _discard_suspended_draft_context(
        self, ctx_caches: list[CtxCache], snapshot,
    ) -> None:
        del ctx_caches, snapshot

    def _ensure_drafter(self, stats: DSparkStats | None = None):
        if self.drafter is not None:
            return self.drafter
        if not callable(self._drafter_loader):
            raise RuntimeError("DSpark drafter was released without a loader")
        started = time.perf_counter()
        drafter = self._drafter_loader()
        if drafter.config != self._cfg:
            raise RuntimeError("reloaded DSpark sidecar config changed")
        self.drafter = drafter
        if stats is not None:
            stats.sidecar_round_loads += 1
            stats.sidecar_load_s += time.perf_counter() - started
            stats.sidecar_loaded_bytes += self.drafter_storage_bytes
        return drafter

    def _release_drafter(
        self, stats: DSparkStats | None = None, *, force: bool = False,
    ) -> int:
        """End the draft-weight lifetime before a target verification sweep.

        Context K/V belongs to the request and remains exact.  Only the
        lossy proposal model is dropped; the released target therefore never
        overlaps its largest streamed layer with the sidecar allocation.
        """
        if (not self.release_between_sweeps and not force) or self.drafter is None:
            return 0
        active_before = int(mx.get_active_memory())
        started = time.perf_counter()
        drafter = self.drafter
        self.drafter = None
        del drafter
        mx.clear_cache()
        released = max(0, active_before - int(mx.get_active_memory()))
        if stats is not None:
            stats.sidecar_round_releases += 1
            stats.sidecar_release_s += time.perf_counter() - started
            stats.sidecar_released_active_bytes += released
        return released

    def clear_prompt_cache(self) -> None:
        self._prompt_cache = None

    @staticmethod
    def _stop_match(text: str, stops: list[str]):
        matches = [(text.find(value), index, value)
                   for index, value in enumerate(stops)
                   if value and text.find(value) != -1]
        return min(matches) if matches else None

    def _propose(
        self,
        pending: int,
        offset: int,
        ctx_caches: list[CtxCache],
        cap: int,
        sampling: SamplingParams,
        history: list[int],
    ):
        drafter = self._ensure_drafter()
        base_logits, head_hidden = drafter.draft_block(
            pending, offset, ctx_caches, cap)
        if self.expert_prefetcher is not None:
            plan = self.expert_prefetcher.build(head_hidden[:cap])
            self._active_expert_prefetch_plan = plan
            self.target._dspark_expert_prefetch_plan = plan
            self.target._dspark_expert_prefetch_depth = (
                self.expert_prefetch_depth)
        if sampling.is_greedy:
            draft_arr = drafter.sample_block(base_logits, pending)
            mx.eval(draft_arr)
            proposals = [int(token) for token in draft_arr.tolist()]
            distributions = None
        else:
            proposals, distributions = drafter.sample_block_exact(
                base_logits, pending, sampling, history)
        if (self.confidence_threshold > 0.0
                and drafter.confidence_head is not None):
            prev = mx.array([pending] + proposals[:-1])
            confidence = drafter.confidence_survival(head_hidden, prev)
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
            if distributions is not None:
                distributions = distributions[:len(proposals)]
        return proposals, distributions

    @staticmethod
    def _verify_stochastic(
        proposals: list[int],
        draft_probabilities: list[mx.array],
        verified: mx.array,
        sampling: SamplingParams,
        history: list[int],
    ) -> tuple[int, list[int]]:
        """Exact speculative rejection with target-minus-draft correction."""
        if len(proposals) != len(draft_probabilities):
            raise ValueError("one draft distribution is required per proposal")
        accepted = 0
        for index, proposal in enumerate(proposals):
            p = filtered_probabilities(
                verified[index], sampling,
                history=history + proposals[:index])
            q = draft_probabilities[index]
            mx.eval(p, q)
            p_token = float(p[proposal].item())
            q_token = float(q[proposal].item())
            ratio = 1.0 if q_token <= 0 else min(1.0, p_token / q_token)
            uniform = float(mx.random.uniform().item())
            if uniform <= ratio:
                accepted += 1
                continue
            replacement = sample_probabilities(
                speculative_residual_probabilities(p, q))
            return accepted, proposals[:accepted] + [replacement]

        bonus = filtered_probabilities(
            verified[len(proposals)], sampling,
            history=history + proposals)
        return accepted, proposals + [sample_probabilities(bonus)]

    def _tapped_context(self) -> mx.array:
        target = self.target
        ids = self._cfg.target_layer_ids
        missing = [layer for layer in ids if layer not in target._tap_hidden]
        if missing:
            raise RuntimeError(f"target did not capture DSpark tap layers {missing}")
        return mx.concatenate([target._tap_hidden[layer] for layer in ids], axis=-1)

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None,
                 stop=None, on_progress=None, *, encoded_ids=None,
                 sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        target = self.target
        request_cache_before = _target_io_snapshot(target)
        sampling = sampling or SamplingParams()
        sampling.seed_rng()
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
        hybrid_qwen = (
            getattr(self._cfg, "target_model_type", "qwen3")
            == "qwen3_5")
        bootstrap_path_stats: dict = {}
        bootstrap_first_token_s = 0.0
        if prompt_cache_hit and hybrid_qwen:
            from .kv_cache import fork_hybrid_kv_endpoint

            (_, cached_target_kv, cached_ctx, prompt_last_logits,
             prompt_last_hidden) = cached
            target_kv = fork_hybrid_kv_endpoint(cached_target_kv)
            ctx_caches = [cache.fork() for cache in cached_ctx]
            target.last_kv = target_kv
        elif prompt_cache_hit:
            (_, target_kv, ctx_caches, prompt_last_logits,
             prompt_last_hidden) = cached
            # Historical dense/K3 behavior transfers sole ownership. Hybrid
            # Qwen keeps one immutable COW prompt endpoint above instead.
            self._prompt_cache = None
        else:
            if cache_eligible:
                # Release a divergent cached prompt before allocating another
                # multi-hundred-MB target/context pair.
                self._prompt_cache = None
            target_kv = None if hybrid_qwen else target.new_kv(
                stepped=use_stepped)
            ctx_caches = self._new_ctx_caches()
            prompt_last_logits = None
            prompt_last_hidden = None
        taps = set(self._cfg.target_layer_ids)
        stats = DSparkStats()

        prefill_t0 = time.perf_counter()
        if prompt_cache_hit:
            # Keep the historical synthetic prefill sweep so target_sweeps-1
            # remains decode sweeps in telemetry.
            stats.target_sweeps += 1
        elif hybrid_qwen:
            context_floor = (
                max(0, len(ids) - self.context_window_tokens)
                if self.context_window_tokens else 0)
            collector = DSparkTapCollector(
                (None if self.defer_prompt_context_updates
                 else self._ensure_drafter()),
                ctx_caches,
                position_floor=context_floor,
                tap_layers=(
                    self._cfg.target_layer_ids
                    if self.defer_prompt_context_updates else None),
                defer_updates=self.defer_prompt_context_updates,
            )
            target.begin_dspark_tap_capture(collector)
            try:
                bootstrap_prompt = _DSparkBootstrapPrompt(prompt, ids)
                target_generate = getattr(
                    target, "generate_with_memory_retry", None)
                if target_generate is None:
                    target_generate = target.generate
                bootstrap = target_generate(
                    bootstrap_prompt,
                    1,
                    sampling=sampling,
                    on_progress=on_progress,
                    constraint=constraint,
                )
            finally:
                target.end_dspark_tap_capture(collector)
            collector.finish(len(ids))
            if self.defer_prompt_context_updates:
                stats.deferred_context_positions = collector.positions
                stats.deferred_context_bytes_peak = (
                    collector.deferred_bytes_peak)
                collector.materialize(self._ensure_drafter(stats))
            collector_update_s = collector.update_s
            # The collector otherwise owns the same large module after tap
            # collection has completed, defeating the round lifetime below.
            collector = None
            stats.context_s += collector_update_s
            stats.target_sweeps += 1
            bootstrap_path_stats = dict(bootstrap.get("path_stats", {}))
            bootstrap_first_token_s = float(
                bootstrap.get("first_token_s", 0.0))
            target_kv = target.last_kv
            if target_kv is None:
                raise RuntimeError(
                    "DSpark target bootstrap did not retain its prompt endpoint")

            endpoint_slot = None
            hot_slots = getattr(target, "_hot_prompt_slots", None)
            if hot_slots is not None:
                for index, slot in enumerate(hot_slots):
                    if getattr(slot, "kv", None) is target_kv:
                        endpoint_slot = hot_slots.pop(index)
                        break
            prompt_last_logits = (
                getattr(endpoint_slot, "logits", None)
                if endpoint_slot is not None else None)
            if prompt_last_logits is None:
                prompt_last_logits = target._final_logits(target._h_last)
                mx.eval(prompt_last_logits)
            prompt_last_hidden = target._h_last
            mx.eval(prompt_last_hidden)
            pending = int(bootstrap["tokens"][0])

            if cache_eligible:
                from .kv_cache import fork_hybrid_kv_endpoint

                self._prompt_cache = (
                    prompt_cache_key,
                    target_kv,
                    ctx_caches,
                    prompt_last_logits,
                    prompt_last_hidden,
                )
                target_kv = fork_hybrid_kv_endpoint(target_kv)
                ctx_caches = [cache.fork() for cache in ctx_caches]
                target.last_kv = target_kv
        else:
            logits = target.forward_tokens(ids, target_kv, tap_layers=taps)
            prompt_last_logits = logits[-1]
            prompt_last_hidden = getattr(target, "_h_last", None)
            mx.eval(*(
                (prompt_last_logits, prompt_last_hidden)
                if prompt_last_hidden is not None else (prompt_last_logits,)))
            stats.target_sweeps += 1
            context = self._tapped_context()
            ctx_t0 = time.perf_counter()
            self._ensure_drafter(stats).update_context(
                context, 0, ctx_caches)
            mx.async_eval([cache.k for cache in ctx_caches])
            stats.context_s += time.perf_counter() - ctx_t0
            target._tap_hidden = {}
        if not (hybrid_qwen and not prompt_cache_hit):
            pending_logits = prompt_last_logits
            if constraint is not None:
                constraint_logits = getattr(
                    target, "_constraint_logits", None)
                if callable(constraint_logits) and prompt_last_hidden is not None:
                    pending_logits = constraint_logits(
                        pending_logits, constraint, hidden=prompt_last_hidden)
                else:
                    pending_logits = constraint.mask_logits(pending_logits)
            pending = sample_probabilities(filtered_probabilities(
                pending_logits, sampling, history=ids))
            if constraint is not None:
                constraint.accept_token(pending)
        emitted = [pending]
        prefill_s = time.perf_counter() - prefill_t0
        first_token_s = (
            bootstrap_first_token_s
            if bootstrap_first_token_s else time.perf_counter() - request_t0)
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

        prefill_cache_after = _target_io_snapshot(target)
        eos = set(target.cfg.eos_token_ids)
        decode_t0 = time.perf_counter()
        while (len(emitted) < max_tokens and pending not in eos
               and not (constraint is not None and constraint.completed)
               and stop_text is None):
            # Every round emits at most cap accepted drafts plus one target
            # token.  Bounding cap keeps endpoint KV/context exact at the
            # caller's output limit without a speculative overrun.
            cap = min(
                self.max_draft_tokens,
                max_tokens - len(emitted) - 1,
            )
            base = target_kv.offset
            round_uses_draft_context = self._draft_context_required()
            if (
                round_uses_draft_context
                and any(cache.position_end != base for cache in ctx_caches)
            ):
                raise RuntimeError(
                    "DSpark context/target KV desync before proposal: "
                    f"kv={base}, ctx_end="
                    f"{[cache.position_end for cache in ctx_caches]}")

            round_draft_s = 0.0
            tree_draft = None
            if cap > 0:
                draft_t0 = time.perf_counter()
                proposed = self._propose(
                    pending, base, ctx_caches, cap, sampling,
                    ids + emitted,
                )
                # Preserve the tiny forced-outcome test/probe API used before
                # stochastic verification existed.
                from .speculative_tree import TreeDraft

                if isinstance(proposed, TreeDraft):
                    if constraint is not None:
                        raise ValueError(
                            "proposal-tree verification does not yet support "
                            "grammar constraints")
                    if not sampling.is_greedy:
                        raise ValueError(
                            "proposal-tree verification is greedy-only")
                    tree_draft = proposed
                    proposals, draft_probabilities = [], None
                elif isinstance(proposed, tuple):
                    proposals, draft_probabilities = proposed
                else:
                    proposals, draft_probabilities = proposed, None
                round_draft_s = time.perf_counter() - draft_t0
                stats.draft_s += round_draft_s
            else:
                # One output slot remains: verify the pending token alone and
                # emit its target successor.  Breaking here would silently
                # return max_tokens-1 whenever the prior round filled exactly
                # to that boundary.
                proposals = []
                draft_probabilities = [] if not sampling.is_greedy else None

            # Proposal tensors are evaluated before this boundary.  The target
            # is the verifier and never needs draft weights, so avoid making
            # every streamed target layer compete with the resident sidecar.
            self._release_drafter(stats)
            context_snapshot = (
                self._suspend_draft_context(ctx_caches, stats)
                if round_uses_draft_context else None
            )

            verify_t0 = time.perf_counter()
            verify_ids = [pending] + proposals
            tree_verification = None
            round_layer_lengths = (
                target_kv.layer_lengths()
                if callable(getattr(target_kv, "layer_lengths", None))
                else None)
            k3_kda_checkpoint = (
                target_kv.kda_cache.fork()
                if (target.cfg.model_type == "kimi_k3"
                    and getattr(target_kv, "kda_cache", None) is not None)
                else None
            )
            qwen_kda_checkpoint = (
                target_kv.kda_cache.fork()
                if (tree_draft is None
                    and target.cfg.model_type in ("qwen3_5", "qwen3_5_moe")
                    and getattr(target_kv, "kda_cache", None) is not None)
                else None
            )
            capture_kda_factors = bool(
                k3_kda_checkpoint is not None
                and len(verify_ids) > 1
                and callable(getattr(
                    target, "consume_serial_kda_factors", None))
            )
            capture_qwen_endpoints = bool(
                qwen_kda_checkpoint is not None and len(verify_ids) > 1)
            try:
                if tree_draft is not None:
                    from .qwen35_tree_verify import verify_qwen35_tree

                    tree_verification = verify_qwen35_tree(
                        target,
                        tree_draft.tree,
                        target_kv,
                        tap_layers=(
                            taps if round_uses_draft_context else ()),
                    )
                    verified = tree_verification.logits
                elif capture_kda_factors:
                    verified = target.forward_tokens_serial_positions(
                        verify_ids,
                        target_kv,
                        tap_layers=(taps if round_uses_draft_context else ()),
                        capture_kda_factors=True,
                    )
                elif capture_qwen_endpoints:
                    verified = target.forward_tokens_serial_positions(
                        verify_ids,
                        target_kv,
                        tap_layers=(taps if round_uses_draft_context else ()),
                        capture_kda_endpoints=True,
                    )
                else:
                    verified = target.forward_tokens_serial_positions(
                        verify_ids, target_kv,
                        tap_layers=(taps if round_uses_draft_context else ()))
            except BaseException:
                target._dspark_expert_prefetch_plan = None
                if round_layer_lengths is not None:
                    target_kv.trim_layer_lengths(round_layer_lengths)
                elif target_kv.offset > base:
                    target_kv.trim(base)
                if k3_kda_checkpoint is not None:
                    target_kv.kda_cache = k3_kda_checkpoint
                    k3_kda_checkpoint.cancel_factor_capture()
                if qwen_kda_checkpoint is not None:
                    target_kv.kda_cache = qwen_kda_checkpoint
                    consume_endpoint = getattr(
                        target, "consume_serial_kda_endpoint", None)
                    if callable(consume_endpoint):
                        consume_endpoint(None)
                consume_factors = getattr(
                    target, "consume_serial_kda_factors", None)
                if consume_factors is not None:
                    consume_factors()
                raise
            plan = self._active_expert_prefetch_plan
            self._active_expert_prefetch_plan = None
            target._dspark_expert_prefetch_plan = None
            if plan is not None:
                stats.expert_prefetch_router_bytes += (
                    plan.router_storage_bytes)
                stats.expert_prefetch_planned_bytes += (
                    plan.estimated_storage_bytes)
                stats.expert_prefetch_pages += sum(
                    len(experts)
                    for experts in plan.experts_by_layer.values())
                stats.expert_prefetch_predicted += plan.predicted_experts
                stats.expert_prefetch_authoritative += (
                    plan.authoritative_experts)
                stats.expert_prefetch_matched += plan.matched_experts
            round_verify_s = time.perf_counter() - verify_t0
            stats.verify_s += round_verify_s
            stats.target_sweeps += 1

            selected_tree_path = None
            if tree_verification is not None:
                from .speculative_tree import greedy_tree_walk

                predictions = [int(token) for token in mx.argmax(
                    verified, axis=-1).reshape(-1).tolist()]
                selected_tree_path, bonus = greedy_tree_walk(
                    tree_verification.tree, predictions)
                accepted = len(selected_tree_path) - 1
                committed = [
                    tree_verification.tree.token_ids[index]
                    for index in selected_tree_path[1:]
                ] + [bonus]
            elif constraint is not None:
                from .qwen35_mtp import _authoritative_target_logits

                accepted = 0
                committed = []
                constrained_terminal = False
                for index, proposal in enumerate(proposals):
                    authoritative = _authoritative_target_logits(
                        target, verified[index], constraint, index)
                    if sampling.is_greedy:
                        target_token = int(mx.argmax(authoritative).item())
                        proposal_accepted = proposal == target_token
                    else:
                        if draft_probabilities is None:
                            raise RuntimeError(
                                "stochastic DSpark proposal omitted q")
                        p = filtered_probabilities(
                            authoritative, sampling,
                            history=ids + emitted + committed)
                        q = draft_probabilities[index]
                        mx.eval(p, q)
                        p_token = float(p[proposal].item())
                        q_token = float(q[proposal].item())
                        ratio = (
                            1.0 if q_token <= 0
                            else min(1.0, p_token / q_token))
                        proposal_accepted = (
                            float(mx.random.uniform().item()) <= ratio)
                        target_token = (
                            proposal if proposal_accepted else
                            sample_probabilities(
                                speculative_residual_probabilities(p, q)))
                    committed.append(
                        int(proposal if proposal_accepted else target_token))
                    constraint.accept_token(committed[-1])
                    if proposal_accepted:
                        accepted += 1
                    candidate = emitted + committed
                    constrained_terminal = bool(
                        constraint.completed
                        or committed[-1] in eos
                        or len(candidate) >= max_tokens
                        or (
                            stops
                            and self._stop_match(
                                target.tokenizer.decode(candidate), stops)
                            is not None))
                    if not proposal_accepted or constrained_terminal:
                        break
                if (accepted == len(proposals)
                        and len(committed) == len(proposals)
                        and not constrained_terminal):
                    bonus_logits = _authoritative_target_logits(
                        target, verified[len(proposals)], constraint,
                        len(proposals))
                    bonus = sample_probabilities(filtered_probabilities(
                        bonus_logits, sampling,
                        history=ids + emitted + committed))
                    committed.append(bonus)
                    constraint.accept_token(bonus)
            elif sampling.is_greedy:
                predictions = [int(token) for token in
                               mx.argmax(
                                   verified, axis=-1
                               ).reshape(-1).tolist()]
                accepted = 0
                while (accepted < len(proposals)
                       and proposals[accepted] == predictions[accepted]):
                    accepted += 1
                committed = proposals[:accepted] + [predictions[accepted]]
            else:
                if draft_probabilities is None:
                    raise RuntimeError(
                        "stochastic DSpark proposal omitted q distributions")
                accepted, committed = self._verify_stochastic(
                    proposals,
                    draft_probabilities,
                    verified,
                    sampling,
                    ids + emitted,
                )
            round_proposed = (
                tree_verification.tree.node_count
                if tree_verification is not None else len(proposals))
            stats.proposed += round_proposed
            stats.accepted += accepted
            stats.rounds.append((round_proposed, accepted))
            if tree_verification is not None:
                stats.tree_rounds.append((
                    max(tree_verification.tree.depths), accepted))
            stats.round_draft_s.append(round_draft_s)
            stats.round_verify_s.append(round_verify_s)
            self._note_verified_round(
                max(tree_verification.tree.depths)
                if tree_verification is not None else len(proposals),
                accepted,
            )

            # Only the anchor and accepted proposal prefix are committed input
            # positions.  The target token is the next pending output and has
            # not yet entered either state.
            planned_emit = len(committed)
            for index, token in enumerate(committed):
                candidate = emitted + committed[:index + 1]
                terminal = token in eos or len(candidate) >= max_tokens
                if stops and self._stop_match(
                        target.tokenizer.decode(candidate), stops) is not None:
                    terminal = True
                if terminal:
                    planned_emit = index + 1
                    break
            committed_inputs = min(
                (len(selected_tree_path)
                 if selected_tree_path is not None else len(verify_ids)),
                1 + max(0, planned_emit - 1))
            retain_draft_context = (
                round_uses_draft_context and self._draft_context_required()
            )
            tapped = (
                self._tapped_context()[:, :committed_inputs, :]
                if retain_draft_context and tree_verification is None else None
            )
            if tree_verification is not None:
                commit_path = selected_tree_path[:committed_inputs]
                restore_t0 = time.perf_counter()
                tree_verification.commit(
                    commit_path, target=target, kv=target_kv)
                stats.tree_factor_commit_s += (
                    time.perf_counter() - restore_t0)
                stats.tree_nodes_verified += len(
                    tree_verification.tree.token_ids)
                stats.tree_paths_committed += 1
                stats.tree_factor_bytes_peak = max(
                    stats.tree_factor_bytes_peak,
                    tree_verification.factors.nbytes(),
                )
                if retain_draft_context:
                    tapped = self._tapped_context()
            elif qwen_kda_checkpoint is not None:
                end_lengths = target_kv.layer_lengths()
                growth = tuple(
                    end - start for start, end in zip(
                        round_layer_lengths, end_lengths, strict=True))
                width = len(verify_ids)
                if any(value not in (0, width) for value in growth):
                    raise RuntimeError(
                        "DSpark Qwen verifier changed a layer by an "
                        f"unexpected count: {growth}")
                if committed_inputs < width:
                    retained = target.consume_serial_kda_endpoint(
                        committed_inputs)
                    if retained is None:
                        raise RuntimeError(
                            "DSpark Qwen verifier omitted the accepted KDA "
                            "endpoint")
                    target_kv.trim_layer_lengths(tuple(
                        start + (committed_inputs if value else 0)
                        for start, value in zip(
                            round_layer_lengths, growth, strict=True)
                    ))
                    target_kv.kda_cache = retained
                    stats.qwen_kda_endpoint_restores += 1
                else:
                    target.consume_serial_kda_endpoint(None)
                hidden_window = getattr(target, "_h_window", None)
                if hidden_window is None:
                    raise RuntimeError(
                        "DSpark Qwen verifier omitted its hidden window")
                target._h_last = hidden_window[
                    :, committed_inputs - 1:committed_inputs, :]
            else:
                target_kv.trim(base + committed_inputs)
            if k3_kda_checkpoint is not None:
                consume_factors = getattr(
                    target, "consume_serial_kda_factors", None)
                factor_window = (
                    consume_factors()
                    if capture_kda_factors
                    and consume_factors is not None
                    else None
                )
                if committed_inputs < len(verify_ids):
                    if factor_window is not None:
                        stats.kda_factor_bytes_peak = max(
                            stats.kda_factor_bytes_peak,
                            factor_window.nbytes(),
                        )
                        restore_t0 = time.perf_counter()
                        target_kv.kda_cache = factor_window.commit_prefix(
                            k3_kda_checkpoint,
                            committed_inputs,
                            native_fused=self.native_kda_factor_commit,
                        )
                        stats.kda_factor_restore_s += (
                            time.perf_counter() - restore_t0)
                        stats.kda_factor_restores += 1
                    else:
                        # Fail-slow exact fallback.  It is expected only for a
                        # custom K3 target lacking factor capture.
                        target_kv.trim(base)
                        target_kv.kda_cache = k3_kda_checkpoint
                        replay = verify_ids[:committed_inputs]
                        replay_logits = target.forward_tokens_serial_positions(
                            replay, target_kv)
                        mx.eval(replay_logits)
                        stats.kda_refeed_sweeps += 1
            if tree_verification is not None:
                # The accepted factors have already been replayed into the
                # target endpoint and the selected tap rows live in `tapped`.
                # Drop sibling K/V, node hidden rows, logits, and compact
                # factors before reloading a large draft sidecar for context
                # maintenance.  Retaining this dead verification graph made
                # the next 1.08 GB DFlash load miss admission by ~10 MB on a
                # 16 GB machine.
                del tree_verification, verified
                mx.clear_cache()
            ctx_t0 = time.perf_counter()
            if retain_draft_context:
                self._restore_draft_context(
                    ctx_caches, context_snapshot, stats)
                context_snapshot = None
                self._ensure_drafter(stats).update_context(
                    tapped, base, ctx_caches)
                mx.async_eval([cache.k for cache in ctx_caches])
            elif context_snapshot is not None:
                self._discard_suspended_draft_context(
                    ctx_caches, context_snapshot)
                context_snapshot = None
            round_context_s = time.perf_counter() - ctx_t0
            stats.context_s += round_context_s
            stats.round_context_s.append(round_context_s)
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
            if hybrid_qwen:
                if target_kv.offset != endpoint:
                    raise RuntimeError(
                        "DSpark Qwen verifier did not land on the released "
                        f"endpoint: kv={target_kv.offset}, expected={endpoint}")
            elif target_kv.offset > endpoint:
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
        prompt_cache_stored = bool(
            hybrid_qwen and cache_eligible and self._prompt_cache is not None
            and self._prompt_cache[0] == prompt_cache_key)
        if (not hybrid_qwen and cache_eligible
                and target_kv.offset >= len(ids)
                and all(cache.length >= len(ids) for cache in ctx_caches)):
            target_kv.trim(len(ids))
            for cache in ctx_caches:
                cache.trim_to(len(ids))
            mx.eval(prompt_last_logits)
            self._prompt_cache = (
                prompt_cache_key, target_kv, ctx_caches,
                prompt_last_logits, prompt_last_hidden)
            prompt_cache_stored = True
        total_s = time.perf_counter() - request_t0
        target._note_true_peak()
        if target.governor is not None:
            target._true_peak_metal_bytes = max(
                target._true_peak_metal_bytes,
                target.governor.request_peak(),
                mx.get_active_memory(),
            )
        request_cache_after = _target_io_snapshot(target)
        result = {
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
                "grammar" if constraint is not None and constraint.completed else
                "eos" if emitted[-1] in eos else "length"),
            "true_peak_metal_bytes": target._true_peak_metal_bytes,
            "prompt_tokens": len(ids),
            "stats": stats,
            "path_stats": {
                **bootstrap_path_stats,
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
                "speculative_round_proposed": [
                    proposed for proposed, _accepted in stats.rounds
                ],
                "speculative_round_accepted": [
                    accepted for _proposed, accepted in stats.rounds
                ],
                "speculative_round_draft_s": list(stats.round_draft_s),
                "speculative_round_verify_s": list(stats.round_verify_s),
                "speculative_round_context_s": list(stats.round_context_s),
                "speculative_draft_oov_fallbacks": 0,
                "dspark_block_size": self._cfg.block_size,
                "dspark_confidence_threshold": self.confidence_threshold,
                "dspark_context_s": stats.context_s,
                "dspark_context_window_tokens": self.context_window_tokens,
                "dspark_deferred_context_positions": (
                    stats.deferred_context_positions),
                "dspark_deferred_context_bytes_peak": (
                    stats.deferred_context_bytes_peak),
                "dspark_draft_context_suspend_rounds": (
                    stats.draft_context_suspend_rounds),
                "dspark_draft_context_restore_rounds": (
                    stats.draft_context_restore_rounds),
                "dspark_draft_context_suspend_s": (
                    stats.draft_context_suspend_s),
                "dspark_draft_context_restore_s": (
                    stats.draft_context_restore_s),
                "dspark_draft_context_suspended_bytes": (
                    stats.draft_context_suspended_bytes),
                "dspark_draft_context_released_active_bytes": (
                    stats.draft_context_released_active_bytes),
                "dspark_kda_factor_bytes_peak": (
                    stats.kda_factor_bytes_peak),
                "dspark_kda_factor_restores": stats.kda_factor_restores,
                "dspark_kda_factor_restore_s": stats.kda_factor_restore_s,
                "dspark_kda_factor_native_fused": int(
                    self.native_kda_factor_commit),
                "dspark_kda_refeed_sweeps": stats.kda_refeed_sweeps,
                "dspark_qwen_kda_endpoint_restores": (
                    stats.qwen_kda_endpoint_restores),
                "dspark_tree_nodes_verified": stats.tree_nodes_verified,
                "dspark_tree_paths_committed": stats.tree_paths_committed,
                "dspark_tree_factor_bytes_peak": (
                    stats.tree_factor_bytes_peak),
                "dspark_tree_factor_commit_s": stats.tree_factor_commit_s,
                "dspark_tree_round_depth": [
                    depth for depth, _accepted in stats.tree_rounds],
                "dspark_tree_round_accepted": [
                    accepted for _depth, accepted in stats.tree_rounds],
                "dspark_target_model_type": (
                    getattr(
                        self._cfg, "target_model_type", "qwen3")),
                "dspark_shared_target_embedding": int(
                    getattr(
                        self._cfg, "share_target_embed", False)),
                "dspark_shared_target_lm_head": int(
                    getattr(
                        self._cfg, "share_target_lm_head", False)),
                "dspark_sidecar_release_between_sweeps": int(
                    self.release_between_sweeps),
                "dspark_sidecar_round_loads": stats.sidecar_round_loads,
                "dspark_sidecar_round_releases": stats.sidecar_round_releases,
                "dspark_sidecar_load_s": stats.sidecar_load_s,
                "dspark_sidecar_release_s": stats.sidecar_release_s,
                "dspark_sidecar_loaded_bytes": stats.sidecar_loaded_bytes,
                "dspark_sidecar_load_margin_bytes": (
                    self.drafter_load_margin_bytes),
                "dspark_sidecar_released_active_bytes": (
                    stats.sidecar_released_active_bytes),
                "dspark_constraint_verified": int(constraint is not None),
                "dspark_context_physical_positions": (
                    ctx_caches[0].length if ctx_caches else 0),
                "dspark_context_position_start": (
                    ctx_caches[0].position_start if ctx_caches else None),
                "dspark_context_position_end": (
                    ctx_caches[0].position_end if ctx_caches else 0),
                "dspark_expert_prefetch_router_bytes": (
                    stats.expert_prefetch_router_bytes),
                "dspark_expert_prefetch_planned_bytes": (
                    stats.expert_prefetch_planned_bytes),
                "dspark_expert_prefetch_pages": (
                    stats.expert_prefetch_pages),
                "dspark_expert_prefetch_predicted": (
                    stats.expert_prefetch_predicted),
                "dspark_expert_prefetch_authoritative": (
                    stats.expert_prefetch_authoritative),
                "dspark_expert_prefetch_matched": (
                    stats.expert_prefetch_matched),
            },
        }
        _merge_target_io_stats(
            target,
            request_cache_before,
            prefill_cache_after,
            request_cache_after,
            result["path_stats"],
        )
        return result


class DSparkSpeculativeEngine:
    """Serving adapter for a streamed Qwen3/Kimi K3 target and DSpark draft."""

    def __init__(self, target, draft_dir: str | Path, *,
                 max_draft_tokens: int = 4,
                 max_prompt_tokens: int = 2048,
                 confidence_threshold: float = 0.0,
                 prompt_cache_min_tokens: int = 2048,
                 context_window_tokens: int = 0,
                 expert_prefetch_budget_mb: int = 0,
                 expert_prefetch_per_layer: int = 8,
                 expert_prefetch_min_margin: float = 0.0,
                 expert_prefetch_depth: int = 2,
                 native_kda_factor_commit: bool = False,
                 release_between_sweeps: bool = False,
                 drafter_load_margin_bytes: int = int(0.4e9)):
        if max_prompt_tokens <= 0:
            raise ValueError("DSpark max_prompt_tokens must be positive")
        self.target = target
        if drafter_load_margin_bytes < 0:
            raise ValueError("DSpark drafter load margin must be non-negative")
        self.drafter_load_margin_bytes = int(drafter_load_margin_bytes)
        draft_dir = Path(draft_dir)
        try:
            draft_bytes = (draft_dir / "model.safetensors").stat().st_size
        except OSError:
            draft_bytes = 0

        def load_drafter():
            # The next draft round is a caller-known lifetime.  Shed consumed
            # target pages first, then admit the compact sidecar against the
            # same hard system/Metal floor used by streamed target layers.
            prepare_for = getattr(target.cache, "prepare_for", None)
            if draft_bytes and callable(prepare_for):
                prepare_for(draft_bytes)
            if target.governor is not None and draft_bytes:
                target.governor.reserve(
                    draft_bytes, margin=self.drafter_load_margin_bytes)
            loaded = DSparkDrafter.load(draft_dir)
            if loaded.config.share_target_lm_head:
                loaded.bind_target_lm_head(target._lm_head_weight)
            if loaded.config.share_target_embed:
                loaded.bind_target_embed(
                    lambda token_ids: target._embed(list(token_ids)))
            if target.governor is not None:
                target.governor.fit_cache_to_live_headroom()
            return loaded

        if target.governor is not None:
            # The drafter is a persistent resident allocation outside the
            # target WeightCache.  Admit it against the same sampled
            # device/system ceiling before materializing its BF16 tensors.
            if draft_bytes:
                target.governor.reserve(draft_bytes)
        drafter = DSparkDrafter.load(draft_dir)
        if drafter.config.share_target_lm_head:
            drafter.bind_target_lm_head(target._lm_head_weight)
        if drafter.config.share_target_embed:
            drafter.bind_target_embed(
                lambda token_ids: target._embed(list(token_ids)))
        if target.governor is not None:
            # A large configured target cache is a performance ceiling, not a
            # fixed machine cap.  Fit its *additional* residency around the
            # now-live drafter; the governor may restore it as headroom grows.
            target.governor.fit_cache_to_live_headroom()
        expert_prefetcher = None
        self._expert_page_prefetcher = None
        if expert_prefetch_budget_mb:
            from .dspark_prefetch import DSparkExpertPrefetcher
            from .prefetcher import Prefetcher

            expert_prefetcher = DSparkExpertPrefetcher(
                target,
                budget_bytes=int(expert_prefetch_budget_mb) * 1_000_000,
                experts_per_layer=expert_prefetch_per_layer,
                min_margin=expert_prefetch_min_margin,
            )
            # Trunk NF12 lives on the internal device while routed experts live
            # on the external checkpoint device.  A distinct bounded worker is
            # therefore intentional multi-device concurrency, not two readers
            # contending on one disk.
            self._expert_page_prefetcher = Prefetcher(
                target.cache,
                page_size_hint=int(target._expert_fetch_page_bytes),
                workers=1,
            )
            target._dspark_expert_prefetcher = (
                self._expert_page_prefetcher)
        self.decoder = DSparkSpeculativeDecoder(
            target, drafter,
            max_draft_tokens=max_draft_tokens,
            confidence_threshold=confidence_threshold,
            prompt_cache_min_tokens=prompt_cache_min_tokens,
            context_window_tokens=context_window_tokens,
            expert_prefetcher=expert_prefetcher,
            expert_prefetch_depth=expert_prefetch_depth,
            native_kda_factor_commit=native_kda_factor_commit,
            drafter_loader=load_drafter,
            release_between_sweeps=release_between_sweeps,
            drafter_storage_bytes=draft_bytes,
            drafter_load_margin_bytes=self.drafter_load_margin_bytes,
        )
        self.max_prompt_tokens = max_prompt_tokens
        self._speculative_k = self.decoder.max_draft_tokens
        self._speculative_draft_dir = draft_dir
        self._speculative_kind = "dspark"
        self._closed = False

    @property
    def drafter(self):
        return self.decoder.drafter if self.decoder is not None else None

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _target_generate(self, reason: str, prompt: str, max_tokens: int,
                         on_token=None, stop=None, on_progress=None,
                         sampling: SamplingParams | None = None,
                         constraint=None):
        self.decoder.clear_prompt_cache()
        self.decoder._release_drafter()
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
        prepared_ids = getattr(prompt, "token_ids", None)
        ids = (list(prepared_ids) if prepared_ids is not None
               else list(self.target.tokenizer.encode(prompt).ids))
        if len(ids) > self.max_prompt_tokens:
            return self._target_generate(
                "prompt-limit", prompt, max_tokens, on_token, stop, on_progress,
                sampling, constraint)
        try:
            return self.decoder.generate(
                prompt, max_tokens, on_token=on_token, stop=stop,
                on_progress=on_progress, encoded_ids=ids, sampling=sampling,
                constraint=constraint)
        finally:
            self.decoder._release_drafter()

    def release_request_state(self):
        self.decoder.clear_prompt_cache()
        self.target.release_request_state()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.decoder.clear_prompt_cache()
        self.decoder._release_drafter()
        self.decoder = None
        try:
            if self._expert_page_prefetcher is not None:
                self._expert_page_prefetcher.close()
                self._expert_page_prefetcher = None
                self.target._dspark_expert_prefetcher = None
        finally:
            try:
                self.target.close()
            finally:
                mx.clear_cache()
