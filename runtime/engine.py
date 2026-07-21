"""StreamingEngine: ties WeightStore + WeightCache + Prefetcher + KVCache into a
generate() loop. Experiments configure it via RuntimeConfig (or a YAML file matching
the shape below) instead of re-implementing the layer sweep.

memory:
  max_weight_cache_mb: 6000
  pinned:
    embeddings: true
    lm_head: false
    first_layers: 2
    last_layers: 2
prefetch:
  depth: 2
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import psutil
import yaml
from tokenizers import Tokenizer

from . import layer_runner, telemetry
from .config import validate_expert_top_k_by_layer
from .kv_cache import KVCache
from .model_loader import WeightStore
from .prefetcher import Prefetcher
from .sampler import SamplingParams, sample
from .weight_cache import WeightCache


def _resident_adjusted_transient(
    start_active: int, end_active: int, peak_active: int,
) -> int:
    """Scratch high-water above both resident endpoints.

    Weight/expert cache growth is persistent and has its own admission path.
    Counting that growth as scratch makes the governor reserve it a second time
    on every following layer/token, evicting the very pages just admitted.
    """
    return max(0, int(peak_active) - max(
        int(start_active), int(end_active)))


def _cache_io_snapshot(engine) -> tuple[int, ...]:
    """Cumulative counters used to derive one request's physical work."""
    stats = engine.cache.stats
    governor = getattr(engine, "governor", None)
    return (
        int(stats.hits), int(stats.misses), int(stats.evictions),
        int(stats.bytes_read), int(engine.expert_hits),
        int(engine.expert_misses),
        int(getattr(governor, "reservations", 0) or 0),
        int(getattr(governor, "reservation_failures", 0) or 0),
        int(getattr(engine.store, "fast_tier_bytes", 0) or 0),
        int(getattr(engine.store, "archive_bytes", 0) or 0),
    )


def _record_cache_io_delta(
    engine, before: tuple[int, ...], stats: dict, *,
    prefix: str = "", after: tuple[int, ...] | None = None,
) -> None:
    """Expose cache/I/O evidence without confusing cumulative engine totals."""
    after = _cache_io_snapshot(engine) if after is None else after
    keys = (
        "weight_cache_hits", "weight_cache_misses",
        "weight_cache_evictions", "weight_store_bytes_read",
        "expert_cache_hits", "expert_cache_misses",
        "governor_reservations", "governor_reservation_failures",
        "weight_fast_tier_bytes", "weight_archive_bytes",
    )
    for key, start, end in zip(keys, before, after, strict=True):
        stats[prefix + key] = max(0, end - start)
    if not prefix:
        stats["weight_cache_resident_bytes"] = int(engine.cache.total_bytes)
        stats["weight_cache_budget_bytes"] = int(engine.cache.max_bytes)
        stats["layer_transient_bytes"] = int(engine._layer_transient)
        stats["token_transient_bytes"] = int(engine._token_transient)


def _quantization_cache_identity(rc: "RuntimeConfig", store) -> str:
    """Fingerprint physical packing plus any load-time transformation."""
    runtime = (
        f"{rc.quant_mode}-q{rc.quant_bits}g{rc.quant_group_size}"
        f"d{rc.quant_min_dim}"
        f"a{int(rc.quant_attention)}m{int(rc.quant_mlp)}"
        f"r{int(rc.quant_router)}h{int(rc.quant_lm_head)}"
        if rc.quant_bits else "bf16"
    )
    if store.on_disk_quantized:
        # Selective checkpoints still contain raw matrices. WeightCache applies
        # the runtime policy to those matrices, so disk identity alone is not a
        # complete description of the KV-producing arithmetic.
        identity = f"disk-{store.quantization_identity}+load-{runtime}"
    else:
        identity = runtime
    if rc.rerank_lm_head:
        identity += (
            f"+headrerank-{rc.rerank_lm_head_mode}"
            f"q{rc.rerank_lm_head_bits}g{rc.rerank_lm_head_group_size}"
            f"k{rc.rerank_lm_head_candidates}"
        )
    if rc.resident_attention_mode:
        identity += (
            f"+residentattn-{rc.resident_attention_mode}"
            f"q{rc.resident_attention_bits}"
            f"g{rc.resident_attention_group_size}"
        )
    if rc.expert_top_k_by_layer:
        identity += "+olmoe-topk-" + ".".join(
            str(top_k) for top_k in rc.expert_top_k_by_layer)
    return identity


def _system_allocation_preserves_floor(
        incoming_bytes: int, floor_mb: int) -> tuple[bool, int, int]:
    """Sample whether one allocation leaves the configured unified-RAM floor."""
    available = int(psutil.virtual_memory().available)
    floor = max(0, int(floor_mb)) * 1_000_000
    return (floor == 0 or available - int(incoming_bytes) >= floor,
            available, floor)


def _gptoss_rope_state(cfg, *, packed: bool):
    """Initialize GPT-OSS RoPE independently of raw-MoE layout validation."""
    if not packed:
        raise RuntimeError(
            "gpt-oss requires a packed store (fused expert tensors must be "
            "unfused): run formats.packed.pack_model first"
        )
    from .gptoss import yarn_params

    return yarn_params(cfg)


@dataclass
class RuntimeConfig:
    max_weight_cache_mb: int = 6000
    mlx_cache_limit_mb: int = 1024
    # Lowest cache budget the live governor may shrink to before refusing an
    # imminent allocation.  Long dense prompts can devote several GiB to exact
    # BF16 KV, so the historical global 1.5 GB floor needlessly made otherwise
    # safe requests fail even though WeightCache supports pass-through pages.
    # Keep the conservative default; side-quest server profiles may opt into a
    # smaller floor with their own real-request gate.
    min_weight_cache_mb: int = 1500
    pin_embeddings: bool = True
    pin_lm_head: bool = False
    pin_first_layers: int = 0
    pin_last_layers: int = 0
    prefetch_depth: int = 0  # 0 disables prefetch
    prefetch_workers: int = 0  # 0 = store default (raw: 1, packed: 2)
    max_kv_mb: int = 0  # 0 = unpaged KV (all resident); >0 enables disk spilling
    adaptive_kv_spill_mb: int = 0  # 0 disables last-resort per-request paging;
    # when positive, ordinary hot KV remains preferred but an unsafe resident
    # admission falls back to this bounded exact BF16 disk-paged cache.
    adaptive_kv_spill_prefill_chunk_size: int = 512
    release_paged_kv_after_generate: bool = False  # server-only single-request
    # paging profile: drop resident pages and spill files before replying. Direct
    # experiment callers retain the historical diagnostic `last_kv` by default.
    stepped_kv_threshold: int = 0  # request positions; 0 disables long-context stepped KV
    kv_page_positions: int = 256
    kv_spill_dir: str = ".kv_spill"
    kv_spill_compress: bool = False  # F07: zstd-L1 closed KV pages before spilling (lossless;
    # bf16 round-trips byte-exact). Opt-in pending an A/B on whether the decode cost is worth
    # the disk-byte saving for THIS workload — KV activations need not compress like weights
    # (F06/warm_tier measured 1.44-1.46x there; F04's warm tier went NEGATIVE when sync
    # compression cost exceeded disk savings, so this is not assumed to win by analogy).
    quant_bits: int = 0  # 0 = keep disk precision; otherwise quantize-on-load
    quant_group_size: int = 64
    quant_mode: str = "affine"
    quant_min_dim: int = 512  # keep small projections in disk precision
    quant_attention: bool = True  # False + quant_bits -> mixed policy (attn bf16, MLP quantized)
    quant_mlp: bool = True
    quant_router: bool = True  # MoE routing is discontinuous; expert-only profiles keep it BF16
    quant_lm_head: bool = True  # untied output projection; separate from tied-head second view below
    quantize_tied_lm_head: bool = False  # keep BF16 rows for embedding lookup but use a
    # separate quantized view for the tied output projection (side-quest only)
    rerank_lm_head: bool = False  # lossy candidate search + exact BF16 rerank;
    # preserves the candidate winner for greedy decode, truncates stochastic support
    rerank_lm_head_candidates: int = 32
    rerank_lm_head_mode: str = "mxfp4"
    rerank_lm_head_bits: int = 4
    rerank_lm_head_group_size: int = 32
    resident_fast_decode: bool = False  # fully-resident dense decode may build one lazy
    # graph across all layers instead of forcing a Metal synchronization per layer
    resident_fast_prefill_limit: int = 0  # maximum exact total position for the
    # same resident lazy graph during prefill; 0 disables it
    resident_moe_decode: bool = False  # fully-resident quantized OLMoE may stack expert
    # pages once and route through gather_qmm without Python expert loops
    # Explicit lossy opt-in only. Empty preserves the checkpoint's released
    # top-k; a nonempty value must cover every OLMoE layer.
    expert_top_k_by_layer: tuple[int, ...] = ()
    resident_attention_mode: str = ""  # resident OLMoE only; e.g. MXFP8 trunk attention
    resident_attention_bits: int = 8
    resident_attention_group_size: int = 32
    fused_swiglu: bool = False  # lossy side-quest: compiled/fused activation arithmetic
    fast_dirs: tuple[str, ...] = ()  # fast-tier overlay dirs, fastest first (split placement)
    require_vpack_hashes: bool = False  # proof runs set True. False preserves
    # pre-F31 local archives but exposes path_stats=legacy-unhashed and is not L0.
    require_raw_weight_hashes: bool = False  # verify every raw safetensors shard
    # against voom.safetensors.sha256.json before accepting weights or prompt KV
    prompt_kv_dir: str = ""  # persist prefill KV per token-prefix; repeat prompts skip the prefill sweep
    prompt_kv_max_mb: int = 2000  # LRU byte budget for the prompt-KV store (0 = unbounded)
    prompt_kv_min_tokens: int = 0  # skip lookup/writes below this prompt size;
    # short misses can cost more to scan and snapshot than to recompute
    prompt_kv_journal_chunk_size: int = 512  # immutable delta positions/object
    hot_prompt_kv: bool = False  # retain one prompt/post-generation KV in memory between requests
    hot_prompt_kv_chunk_size: int = 4096  # reuse divergent prompts only at this fixed boundary
    hot_prompt_kv_slots: int = 1  # LRU capacity: how many retained KV branches survive
    # concurrently (2026-07-15). Default 1 preserves the original single-slot behavior.
    hot_prompt_kv_min_tokens: int = 0  # never RETAIN a slot for a prompt shorter than
    # this (0 = retain everything, the original behavior). Lookup/matching against
    # existing slots is unaffected -- a small request can still get a hit. This only
    # gates the SAVE side. Real harness traffic (2026-07-15) showed a variable, not
    # fixed, number of tiny non-conversational calls (title generation, working-
    # memory updates: 89 and 885 tokens, tools=0) between real conversation turns
    # (26,872-27,047 tokens, tools=131) -- one interleaved call between one pair of
    # turns, two between the next. A fixed `hot_prompt_kv_slots` count that covers
    # the worst observed case today is still just a guess an even busier harness
    # session can exceed tomorrow. Refusing to let cheap, quick-to-recompute prompts
    # occupy a slot at all removes the guess: only prompts big enough to make
    # eviction expensive ever risk evicting something.
    # A real harness that interleaves unrelated requests (e.g. a title-generation or
    # working-memory call between two turns of the same conversation) evicts a
    # single slot before the NEXT turn of the actual conversation can reuse it --
    # observed live: a title-gen request between "hello world" and "how are you"
    # meant "how are you" missed entirely (26,907 tokens prefilled cold both times).
    # Raising this lets each distinct prompt lineage (the main thread, a title-gen
    # helper, etc.) keep its own slot instead of fighting over one. Each retained
    # slot holds a full KV state proportional to its context length -- this is a
    # real memory/quality tradeoff, not a free win; size it to the actual number of
    # concurrently-live prompt lineages a caller's harness produces, not larger.
    hot_prompt_kv_min_available_mb: int = 0  # optional serving reserve above
    # the governor's hardware-derived Metal ceiling and critical reserve.
    # Durable slots are evicted from RAM first and weight-cache residency is
    # shed next. Zero avoids turning a benchmark/ops floor into an ordinary
    # request rejection; proofs may still opt into a stricter sampled floor.
    tool_pic: bool = False  # lossy Qwen/OLMoE tool-span relocation + boundary repair
    tool_pic_shared_pages: bool = False  # experimental dense-Qwen MiniPIC-style unrotated
    # K/V page sharing. Engine-local only: durable snapshots, spill, and
    # multimodal M-RoPE need separate formats/kernels and fail closed for now.
    tool_pic_repair_tokens: int = 4  # recomputed leading positions per reused tool span
    tool_pic_min_savings: int = 128  # minimum avoided positions versus exact-prefix prefill
    # Engine-local linear SuffixDecoding. This history is intentionally
    # single-tenant: target verification protects output correctness, but cache
    # hits can still reveal cross-request workload membership through timing.
    suffix_decoding: bool = False
    suffix_decoding_k: int = 6
    suffix_decoding_factor: float = 4.0
    suffix_decoding_max_depth: int = 64
    suffix_decoding_min_probability: float = 0.1
    suffix_decoding_max_cached_requests: int = 256
    suffix_decoding_max_cached_tokens: int = 32_768
    suffix_decoding_max_nodes: int = 262_144
    suffix_decoding_max_bytes: int = 96_000_000
    suffix_decoding_max_local_tokens: int = 2_048
    hot_prompt_kv_persist_dir: str = ""  # disk backing for the in-memory hot-
    # prompt-kv LRU above (2026-07-15): "" disables it -- pure in-memory,
    # does not survive a restart, the original behavior. When set, every
    # slot appended to `_hot_prompt_slots` is also written here as a parent-
    # hashed segment DAG (see runtime/hot_kv_persist.py's module docstring:
    # true delta-only writes, fork-preserving), and engine startup reloads
    # up to `hot_prompt_kv_slots` of them so a conversation can resume warm
    # across a restart instead of paying a full cold prefill again.
    hot_prompt_kv_persist_max_checkpoints: int = 64  # disk retention budget,
    # DECOUPLED from hot_prompt_kv_slots (in-memory capacity) on purpose:
    # disk is meant to hold more history/forks than memory ever needs to.
    # Oldest-by-mtime checkpoints beyond this are dropped each turn; their
    # ancestor segments are swept only once no surviving checkpoint needs
    # them.
    # Side-quest-only override for a Qwen2 checkpoint that does not itself
    # declare rope_scaling. 0/1 = released RoPE; >1 = static YaRN extrapolation.
    qwen_yarn_factor: float = 0.0
    prefill_chunk_size: int = 0  # bound prefill compute/transient memory WITHOUT writing state
    prefill_last_token_separate: bool = False  # MLX-LM-compatible endpoint schedule
    prefill_checkpoint_every: int = 0  # F60: save prompt-KV state every N prefill positions
    # (0 = off). Interrupted mega-prefills then RESUME via the existing
    # longest-prefix load. For compatibility, this also acts as the chunk size
    # when prefill_chunk_size=0. F37 v6 appends only new journal positions; the
    # checkpoint cadence remains opt-in because every endpoint still adds fsync,
    # checksum, logits, and metadata work.
    adaptive_chunk_size: bool = False  # F68: learn a safe prefill_chunk_size ONLINE from
    # observed peak-memory slope instead of a hard-coded architecture-specific constant
    # (4096 was measured on Qwen2.5-1.5B only). See runtime/adaptive_chunk.py. Overrides
    # prefill_chunk_size's fixed value per-chunk once enough chunks have run. It is
    # intended as scheduling only, but changed shapes can select different kernels;
    # every enabled shape still needs block-output and greedy-token gates.
    adaptive_chunk_safe_bytes: int = 0  # 0 = resample the governor's live ceiling per chunk;
    # a positive value is an explicit experiment/replay target
    embed_rows: bool = False  # F02: row-paged embeddings from a raw sidecar (untied models only)
    stream_lm_head: bool = False  # F02: block-streamed lm_head matmul, never materializes the
    # full (vocab, hidden) tensor (GLM: ~1.9GB). Bit-identical (only the output/vocab dim is
    # chunked, not the reduction dim). Plain safetensors checkpoints only (not vpack2/packed).
    governor: bool = True  # F16: live memory-pressure governor (safety default on)
    # Qwen3-VL preprocessing budget. 0 selects the runtime's exact global-
    # attention safety ceiling; fast mode may choose a smaller quality-gated
    # patch budget to reduce both ViT attention and multimodal prefill.
    vision_max_patches: int = 0
    warm_start: int = 0  # F19: preload this many hottest expert pages at engine-up (0=off; measured a DROP)
    mla_compressed_kv: bool = True  # F21: cache MLA latents — 49x less KV RAM on GLM, equivalence-verified
    mla_absorbed_decode: bool = False  # F21 follow-up: decode-time (L=1) attention computed directly
    # in the compressed latent space (the DeepSeek MLA "absorption" trick — algebraically fold
    # kv_b_proj into the query/output projections) instead of re-expanding K/V for every cached
    # position every step. Opt-in pending the strict equivalence gate (see tests/test_mla_absorbed.py).
    warm_mb: int = 0  # F04: compressed-RAM warm tier budget (0=off; bf16 pages only)
    final_dead_token_elim: bool = True  # F36: last layer's MLP runs only on the last prefill position
    router_lookahead: bool = False  # F45: measured NEGATIVE on local disk (pollutes LFU, competes with demand reads); retry over NAS only
    expert_predictive_prefetch: bool = False  # Markov next-layer expert hints;
    # separately gated from deterministic trunk prefetch because F45-class
    # speculative traffic regressed on the saturated local disk. Explicit opt-in.
    expert_prefetch_idle_only: bool = True  # when enabled, issue a predicted
    # expert hint only if no other prefetch is queued/active. False is an
    # intentionally aggressive experiment and must be byte/wall A/B tested.
    context_bound: int = 0  # F43: declared max positions (prompt+generation). On GLM, a bound
    # <= index_topk provably never invokes the DSA indexer, so its weights are never
    # loaded and its state never computed. Runs exceeding the bound are refused.
    expert_fetch_batch: int = 0  # F74-v2: cap the fetch+compute+release lifetime
    # (0 = unbounded, old behavior). Real-GLM incident (2026-07-14): with 256
    # routed experts/layer and 8 active/token, a cold-cache layer's expert union
    # approaches the full 256 even at SMALL prefill chunk sizes (coupon-collector
    # effect) -- F68's chunk-size throttling alone could not bound this because the
    # actual spike is the complete routed union staying strongly referenced by
    # the caller. Cache-only fetch sub-batching is insufficient: GLM must compute
    # and materialize each sub-batch before fetching the next one.
    decode_expert_fetch_batch: int = 0  # optional larger batch when routing covers
    # exactly one position; unlike prefill, decode's union is bounded by top-k

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        run = raw.get("runtime", raw)
        mem = raw.get("memory", {})
        pinned = mem.get("pinned", {})
        expert_top_k_by_layer = run.get("expert_top_k_by_layer", ())
        if not isinstance(expert_top_k_by_layer, (list, tuple)):
            raise ValueError(
                "expert_top_k_by_layer must be a YAML sequence of integers")
        return cls(
            max_weight_cache_mb=mem.get("max_weight_cache_mb", 6000),
            mlx_cache_limit_mb=mem.get("mlx_cache_limit_mb", 1024),
            min_weight_cache_mb=mem.get("min_weight_cache_mb", 1500),
            pin_embeddings=pinned.get("embeddings", True),
            pin_lm_head=pinned.get("lm_head", False),
            pin_first_layers=pinned.get("first_layers", 0),
            pin_last_layers=pinned.get("last_layers", 0),
            prefetch_depth=raw.get("prefetch", {}).get("depth", 0),
            prefetch_workers=raw.get("prefetch", {}).get("workers", 0),
            max_kv_mb=mem.get("max_kv_mb", 0),
            release_paged_kv_after_generate=run.get(
                "release_paged_kv_after_generate", False),
            stepped_kv_threshold=run.get("stepped_kv_threshold", 0),
            kv_page_positions=mem.get("kv_page_positions", 256),
            kv_spill_dir=mem.get("kv_spill_dir", run.get("kv_spill_dir", ".kv_spill")),
            kv_spill_compress=mem.get(
                "kv_spill_compress", run.get("kv_spill_compress", False)
            ),
            adaptive_kv_spill_mb=run.get("adaptive_kv_spill_mb", 0),
            adaptive_kv_spill_prefill_chunk_size=run.get(
                "adaptive_kv_spill_prefill_chunk_size", 512),
            quant_bits=raw.get("quant", {}).get("bits", 0),
            quant_group_size=raw.get("quant", {}).get("group_size", 64),
            quant_mode=raw.get("quant", {}).get("mode", "affine"),
            quant_min_dim=raw.get("quant", {}).get("min_dim", 512),
            quant_attention=raw.get("quant", {}).get("attention", True),
            quant_mlp=raw.get("quant", {}).get("mlp", True),
            quant_router=raw.get("quant", {}).get("router", True),
            quant_lm_head=raw.get("quant", {}).get("lm_head", True),
            quantize_tied_lm_head=raw.get("quant", {}).get("tied_lm_head", False),
            rerank_lm_head=run.get("rerank_lm_head", False),
            rerank_lm_head_candidates=run.get(
                "rerank_lm_head_candidates", 32),
            rerank_lm_head_mode=run.get("rerank_lm_head_mode", "mxfp4"),
            rerank_lm_head_bits=run.get("rerank_lm_head_bits", 4),
            rerank_lm_head_group_size=run.get(
                "rerank_lm_head_group_size", 32),
            resident_fast_decode=run.get("resident_fast_decode", False),
            resident_fast_prefill_limit=run.get(
                "resident_fast_prefill_limit", 0),
            resident_moe_decode=run.get("resident_moe_decode", False),
            expert_top_k_by_layer=tuple(expert_top_k_by_layer),
            resident_attention_mode=run.get("resident_attention_mode", ""),
            resident_attention_bits=run.get("resident_attention_bits", 8),
            resident_attention_group_size=run.get(
                "resident_attention_group_size", 32),
            fused_swiglu=run.get("fused_swiglu", False),
            fast_dirs=tuple(mem.get("fast_dirs", [])),
            require_vpack_hashes=run.get(
                "require_vpack_hashes", mem.get("require_vpack_hashes", False)
            ),
            require_raw_weight_hashes=run.get(
                "require_raw_weight_hashes",
                mem.get("require_raw_weight_hashes", False),
            ),
            prompt_kv_dir=run.get("prompt_kv_dir", ""),
            prompt_kv_max_mb=run.get("prompt_kv_max_mb", 2000),
            prompt_kv_min_tokens=run.get("prompt_kv_min_tokens", 0),
            prompt_kv_journal_chunk_size=run.get(
                "prompt_kv_journal_chunk_size", 512),
            hot_prompt_kv=run.get("hot_prompt_kv", False),
            hot_prompt_kv_chunk_size=run.get("hot_prompt_kv_chunk_size", 4096),
            hot_prompt_kv_slots=run.get("hot_prompt_kv_slots", 1),
            hot_prompt_kv_min_tokens=run.get("hot_prompt_kv_min_tokens", 0),
            hot_prompt_kv_min_available_mb=run.get(
                "hot_prompt_kv_min_available_mb", 0),
            tool_pic=run.get("tool_pic", False),
            tool_pic_shared_pages=run.get("tool_pic_shared_pages", False),
            tool_pic_repair_tokens=run.get("tool_pic_repair_tokens", 4),
            tool_pic_min_savings=run.get("tool_pic_min_savings", 128),
            suffix_decoding=run.get("suffix_decoding", False),
            suffix_decoding_k=run.get("suffix_decoding_k", 6),
            suffix_decoding_factor=run.get("suffix_decoding_factor", 4.0),
            suffix_decoding_max_depth=run.get(
                "suffix_decoding_max_depth", 64),
            suffix_decoding_min_probability=run.get(
                "suffix_decoding_min_probability", 0.1),
            suffix_decoding_max_cached_requests=run.get(
                "suffix_decoding_max_cached_requests", 256),
            suffix_decoding_max_cached_tokens=run.get(
                "suffix_decoding_max_cached_tokens", 32_768),
            suffix_decoding_max_nodes=run.get(
                "suffix_decoding_max_nodes", 262_144),
            suffix_decoding_max_bytes=run.get(
                "suffix_decoding_max_bytes", 96_000_000),
            suffix_decoding_max_local_tokens=run.get(
                "suffix_decoding_max_local_tokens", 2_048),
            hot_prompt_kv_persist_dir=run.get("hot_prompt_kv_persist_dir", ""),
            hot_prompt_kv_persist_max_checkpoints=run.get(
                "hot_prompt_kv_persist_max_checkpoints", 64),
            qwen_yarn_factor=run.get("qwen_yarn_factor", 0.0),
            prefill_chunk_size=run.get("prefill_chunk_size", 0),
            prefill_last_token_separate=run.get(
                "prefill_last_token_separate", False),
            prefill_checkpoint_every=run.get("prefill_checkpoint_every", 0),
            adaptive_chunk_size=run.get("adaptive_chunk_size", False),
            adaptive_chunk_safe_bytes=run.get("adaptive_chunk_safe_bytes", 0),
            embed_rows=run.get("embed_rows", False),
            stream_lm_head=run.get("stream_lm_head", False),
            governor=run.get("governor", True),
            vision_max_patches=run.get("vision_max_patches", 0),
            warm_start=run.get("warm_start", 0),
            mla_compressed_kv=run.get("mla_compressed_kv", True),
            mla_absorbed_decode=run.get("mla_absorbed_decode", False),
            warm_mb=run.get("warm_mb", 0),
            final_dead_token_elim=run.get("final_dead_token_elim", True),
            router_lookahead=run.get("router_lookahead", False),
            expert_predictive_prefetch=run.get(
                "expert_predictive_prefetch", False),
            expert_prefetch_idle_only=run.get(
                "expert_prefetch_idle_only", True),
            context_bound=run.get("context_bound", 0),
            expert_fetch_batch=run.get("expert_fetch_batch", 0),
            decode_expert_fetch_batch=run.get("decode_expert_fetch_batch", 0),
        )


def _apply_runtime_expert_top_k(rc: RuntimeConfig, cfg) -> None:
    """Validate and copy the runtime-only OLMoE routing schedule."""
    raw_schedule = rc.expert_top_k_by_layer
    if not isinstance(raw_schedule, (list, tuple)):
        raise ValueError(
            "expert_top_k_by_layer must be a list or tuple of integers")
    if raw_schedule and cfg.model_type != "olmoe":
        raise ValueError(
            "expert_top_k_by_layer is supported only for OLMoE checkpoints")
    schedule = (
        validate_expert_top_k_by_layer(cfg, raw_schedule)
        if cfg.model_type == "olmoe" else ()
    )
    rc.expert_top_k_by_layer = schedule
    if cfg.model_type == "olmoe":
        cfg.expert_top_k_by_layer = schedule


@dataclass
class _HotPromptSlot:
    """One retained in-memory prompt-KV branch. Ownership of `kv`/`logits`/
    `prompt_logits` is transferred, never cloned, matching the original
    single-slot design's own comment. A list of these is an LRU (see
    RuntimeConfig.hot_prompt_kv_slots): most-recently-(re)inserted at the
    end, least-recently-used evicted first from the front."""

    tokens: tuple[int, ...]
    kv: "KVCache"
    logits: mx.array
    prompt_length: int
    prompt_logits: mx.array
    reusable_prefix: int
    approximate: bool = False  # true only for a selectively repaired PIC prompt
    # Optional (content id, prompt-token start, prompt-token end) records used
    # by the lossy PIC path. They are included in durable checkpoint manifests
    # so the first edited catalog after a restart can reuse a warm source.
    tool_capsules: tuple[tuple[str, int, int], ...] = ()
    # Root-to-leaf disk segment ids backing this slot (runtime/hot_kv_persist.py),
    # empty when persistence is disabled or this slot has not been saved yet.
    # segment_chain[-1] is this slot's own checkpoint identity; segment_chain[:n]
    # for n = reusable_prefix // hot_prompt_kv_chunk_size is a valid PARENT for a
    # future save (see the "branch" persist_parent_chain derivation in generate()).
    segment_chain: tuple[str, ...] = ()
    # Logical prompt lineage. Hidden gateway decision/execution prompts differ
    # near the beginning even when they belong to one caller turn. Namespace
    # isolation prevents either phase from matching or displacing the other's
    # logical cache. Exact token equality within a namespace remains the final
    # correctness condition for reuse.
    cache_namespace: str = "default"


class StreamingEngine:
    def __init__(self, model_dir: str | Path, rc: RuntimeConfig | None = None):
        self.rc = rc or RuntimeConfig()
        if self.rc.mlx_cache_limit_mb <= 0:
            raise ValueError("mlx_cache_limit_mb must be positive")
        # MLX's buffer cache is NOT counted in our weight budget and can balloon
        # by gigabytes under paging churn (measured 2.3 GB), pushing the machine
        # over the macOS wired-memory line. Exact paged-KV profiles use a much
        # smaller server-configured cap than the ordinary 1-GiB default.
        mx.set_cache_limit(self.rc.mlx_cache_limit_mb * 1_000_000)
        if self.rc.stepped_kv_threshold < 0:
            raise ValueError("stepped_kv_threshold must be >= 0")
        if self.rc.min_weight_cache_mb <= 0:
            raise ValueError("min_weight_cache_mb must be positive")
        if self.rc.prefetch_workers < 0:
            raise ValueError("prefetch_workers must be >= 0")
        if self.rc.resident_fast_prefill_limit < 0:
            raise ValueError("resident_fast_prefill_limit must be >= 0")
        if self.rc.vision_max_patches < 0:
            raise ValueError("vision_max_patches must be >= 0")
        if self.rc.rerank_lm_head and self.rc.rerank_lm_head_candidates <= 0:
            raise ValueError("rerank_lm_head_candidates must be positive")
        if self.rc.adaptive_chunk_safe_bytes < 0:
            raise ValueError("adaptive_chunk_safe_bytes must be >= 0")
        if self.rc.prompt_kv_min_tokens < 0:
            raise ValueError("prompt_kv_min_tokens must be >= 0")
        if self.rc.prompt_kv_journal_chunk_size <= 0:
            raise ValueError("prompt_kv_journal_chunk_size must be positive")
        if self.rc.tool_pic_repair_tokens < 0:
            raise ValueError("tool_pic_repair_tokens must be non-negative")
        if self.rc.tool_pic_min_savings < 0:
            raise ValueError("tool_pic_min_savings must be non-negative")
        if self.rc.suffix_decoding:
            from .suffix_decoding import validate_suffix_settings

            validate_suffix_settings(
                max_depth=self.rc.suffix_decoding_max_depth,
                max_spec_tokens=self.rc.suffix_decoding_k,
                factor=self.rc.suffix_decoding_factor,
                min_probability=self.rc.suffix_decoding_min_probability,
                max_cached_requests=(
                    self.rc.suffix_decoding_max_cached_requests),
                max_cached_tokens=self.rc.suffix_decoding_max_cached_tokens,
                max_nodes=self.rc.suffix_decoding_max_nodes,
                max_bytes=self.rc.suffix_decoding_max_bytes,
                max_local_tokens=self.rc.suffix_decoding_max_local_tokens,
            )
        if self.rc.tool_pic and self.rc.max_kv_mb:
            raise ValueError("tool_pic does not support paged/spilled KV")
        if self.rc.adaptive_kv_spill_mb < 0:
            raise ValueError("adaptive_kv_spill_mb must be non-negative")
        if not 1 <= self.rc.adaptive_kv_spill_prefill_chunk_size <= 4096:
            raise ValueError(
                "adaptive_kv_spill_prefill_chunk_size must be in [1, 4096]")
        if self.rc.tool_pic_shared_pages and not self.rc.tool_pic:
            raise ValueError("tool_pic_shared_pages requires tool_pic")
        if self.rc.tool_pic_shared_pages and not self.rc.hot_prompt_kv:
            raise ValueError("tool_pic_shared_pages requires hot_prompt_kv")
        if self.rc.tool_pic_shared_pages and (
                self.rc.max_kv_mb or self.rc.prompt_kv_dir
                or self.rc.hot_prompt_kv_persist_dir):
            raise ValueError(
                "tool_pic_shared_pages is engine-local and does not yet support "
                "KV spill or durable prompt/hot-KV persistence")
        if (self.rc.adaptive_chunk_size
                and self.rc.adaptive_chunk_safe_bytes == 0
                and not self.rc.governor):
            raise ValueError(
                "adaptive_chunk_size needs the governor when "
                "adaptive_chunk_safe_bytes=0"
            )
        if self.rc.hot_prompt_kv and self.rc.hot_prompt_kv_chunk_size <= 0:
            raise ValueError("hot_prompt_kv_chunk_size must be positive when hot_prompt_kv is enabled")
        if self.rc.hot_prompt_kv and self.rc.hot_prompt_kv_slots <= 0:
            raise ValueError("hot_prompt_kv_slots must be positive when hot_prompt_kv is enabled")
        if self.rc.hot_prompt_kv_min_available_mb < 0:
            raise ValueError("hot_prompt_kv_min_available_mb must be non-negative")
        if self.rc.hot_prompt_kv:
            if self.rc.prefill_chunk_size != self.rc.hot_prompt_kv_chunk_size:
                raise ValueError(
                    "hot_prompt_kv requires prefill_chunk_size == hot_prompt_kv_chunk_size")
            if self.rc.adaptive_chunk_size or self.rc.prefill_checkpoint_every:
                raise ValueError(
                    "hot_prompt_kv requires fixed chunks and no persistent prefill checkpoints")
        if self.rc.prompt_kv_dir and self.rc.adaptive_chunk_size:
            # Adaptive boundaries depend on live memory observations and can
            # select different kernels/reduction paths on two otherwise equal
            # requests. A static fingerprint cannot certify that schedule.
            raise ValueError(
                "durable prompt KV requires a fixed prefill schedule; "
                "disable adaptive_chunk_size or prompt_kv_dir")
        self.store = WeightStore(
            model_dir,
            fast_dirs=list(self.rc.fast_dirs),
            require_vpack_hashes=self.rc.require_vpack_hashes,
            require_raw_weight_hashes=self.rc.require_raw_weight_hashes,
        )
        # WeightStore may have re-resolved a stale SMB mount from Plex to
        # Plex-N.  Every later checkpoint-relative path must follow that same
        # healthy directory; mixing the recovered weights/config with a stale
        # tokenizer, sidecar, fingerprint, or predictor path defeats F24.
        self._model_dir = self.store.dir
        self.cfg = self.store.config
        _apply_runtime_expert_top_k(self.rc, self.cfg)
        if (self.cfg.model_type in ("kimi_linear", "qwen3_5_moe", "qwen3_5")
                and self.rc.prompt_kv_dir):
            raise ValueError(
                f"{self.cfg.model_type} recurrent attention state is not "
                "supported by token-indexed prompt KV persistence; "
                "disable prompt_kv_dir")
        vision_tool_pic = bool(
            self.cfg.vision_config
            and self.cfg.model_type.startswith("qwen3_vl"))
        if self.rc.tool_pic and not self.rc.hot_prompt_kv and not vision_tool_pic:
            raise ValueError(
                "tool_pic requires hot_prompt_kv outside Qwen3-VL")
        if (self.rc.tool_pic
                and (self.cfg.model_type not in (
                    "qwen2", "qwen3", "olmoe", "qwen3_vl")
                     or (self.cfg.num_experts
                         and self.cfg.model_type != "olmoe")
                     or (self.cfg.vision_config and not vision_tool_pic))):
            raise ValueError(
                "tool_pic currently supports Qwen2/Qwen3, OLMoE, and Qwen3-VL")
        if self.rc.tool_pic_shared_pages:
            if (self.cfg.vision_config
                    or self.cfg.model_type not in ("qwen2", "qwen3")
                    or self.cfg.num_experts):
                raise ValueError(
                    "tool_pic_shared_pages currently supports dense text Qwen only")
            if (self.cfg.head_dim % 32
                    or self.cfg.num_attention_heads
                    % self.cfg.num_key_value_heads):
                raise ValueError(
                    "tool_pic_shared_pages needs head_dim divisible by 32 and "
                    "an integral GQA ratio")
            if not mx.metal.is_available():
                raise ValueError("tool_pic_shared_pages requires Apple Metal")
        if (self.rc.resident_attention_mode
                and (self.cfg.model_type != "olmoe"
                     or not self.rc.resident_moe_decode)):
            raise ValueError(
                "resident_attention_mode requires resident OLMoE decode")
        # Shape-stable prompt cache, an LRU of up to `hot_prompt_kv_slots`
        # branches. It is deliberately engine-local: unlike F37's durable store
        # it performs no serialization or device/host copy, and ownership of
        # a slot's arrays is transferred (not cloned) between requests.
        # Least-recently-used slot is index 0; most-recently-(re)inserted is
        # the last element (2026-07-15: generalized from a single slot, which
        # meant any interleaved request -- e.g. a harness's own title-
        # generation call between two turns of the same conversation --
        # evicted the main thread's state before it could ever be reused).
        self._hot_prompt_slots: list[_HotPromptSlot] = []
        self.last_kv = None
        self._position_free_pool = None
        # F37's journal owns immutable metadata indexes; retain one wrapper for
        # the engine lifetime instead of rebuilding every segment index on each
        # request. It is initialized lazily only after the admission threshold.
        self._prompt_kv_store = None
        if self.cfg.model_type in ("glm_moe_dsa", "kimi_k25"):
            # This runtime currently implements the released target's n_group=1
            # router. Silently ignoring group-restricted routing on another GLM
            # checkpoint would change the discontinuous expert choice. Kimi
            # K2.5 shares this exact noaux_tc routing math (run_glm_block
            # reused unmodified, F93) so the same guard applies.
            if self.cfg.n_group != 1 or self.cfg.topk_group != 1:
                raise NotImplementedError(
                    "group-restricted GLM-family routing is unsupported: "
                    f"n_group={self.cfg.n_group}, topk_group={self.cfg.topk_group}"
                )
            if self.cfg.index_topk and len(self.cfg.indexer_types) != self.cfg.num_hidden_layers:
                raise ValueError(
                    "GLM indexer_types must describe every trunk layer: "
                    f"{len(self.cfg.indexer_types)} != {self.cfg.num_hidden_layers}"
                )
        if (self.cfg.model_type in (
                "glm_moe_dsa", "kimi_linear", "kimi_k25", "qwen3_5_moe")
                and self.rc.expert_fetch_batch <= 0):
            # F74-v2 is a safety default for every construction path, including
            # direct experiments and YAML. Leaving zero as "unbounded" silently
            # bypassed the server's GLM-specific protection and recreated the
            # 16-22 GB union lifetime. q=1 is the fail-closed default until the
            # q=2/8 lazy-graph peak and arithmetic-order gates exist; explicit
            # validation scripts may request those larger batches. Other
            # architectures retain zero semantics. F92: Kimi Linear/K2.5 have
            # 256/384 experts each, the same "prefill floods the union" risk
            # GLM was fixed for here (measured 2026-07-18: an unbounded fetch
            # on a 15-token prompt requested ~2.8GB in one shot and was
            # correctly refused by the governor). Qwen3.6 likewise has 256
            # experts per layer and can route a near-complete union during a
            # multi-position prefill, so the same lifetime bound applies.
            self.rc.expert_fetch_batch = 1
        tokenizer_json = self._model_dir / "tokenizer.json"
        if tokenizer_json.exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        else:
            # F92/F93: Kimi checkpoints ship a tiktoken vocab + a custom slow
            # tokenizer class instead of a fast tokenizer.json.
            from .tiktoken_convert import build_kimi_fast_tokenizer, has_tiktoken_tokenizer

            if not has_tiktoken_tokenizer(self._model_dir):
                raise FileNotFoundError(
                    f"no tokenizer.json in {self._model_dir} and it does not "
                    "look like a tiktoken-based checkpoint (need tiktoken.model "
                    "+ tokenization_kimi.py) -- unsupported tokenizer format")
            self.tokenizer = build_kimi_fast_tokenizer(self._model_dir)
        self._suffix_cache = None
        if self.rc.suffix_decoding:
            from .suffix_decoding import (
                SuffixDecodingCache, model_tokenizer_fingerprint)

            self._suffix_cache = SuffixDecodingCache(
                identity=model_tokenizer_fingerprint(self._model_dir),
                max_depth=self.rc.suffix_decoding_max_depth,
                max_spec_tokens=self.rc.suffix_decoding_k,
                factor=self.rc.suffix_decoding_factor,
                min_probability=self.rc.suffix_decoding_min_probability,
                max_cached_requests=(
                    self.rc.suffix_decoding_max_cached_requests),
                max_cached_tokens=self.rc.suffix_decoding_max_cached_tokens,
                max_nodes=self.rc.suffix_decoding_max_nodes,
                max_bytes=self.rc.suffix_decoding_max_bytes,
                max_local_tokens=self.rc.suffix_decoding_max_local_tokens,
            )
        # 2026-07-14: config.json/generation_config.json's eos_token_id doesn't
        # always list every real turn-boundary token a chat-tuned checkpoint
        # actually learned to emit -- found live serving a Qwen2.5-1.5B
        # snapshot whose eos_token_id only listed <|endoftext|>, not the
        # <|im_end|> its own real chat template renders and the model
        # actually stops at when correctly prompted. Without it, generation
        # free-ran past the real turn boundary into a hallucinated next turn.
        # A string-based `stop` sequence can't substitute: this tokenizer's
        # decode() strips special tokens by default (confirmed empirically:
        # decode([<|im_end|>-id]) == ""), so the literal marker text never
        # appears in decoded output for a stop-sequence scan to find.
        for marker in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            marker_id = self.tokenizer.token_to_id(marker)
            if marker_id is not None and marker_id not in self.cfg.eos_token_ids:
                self.cfg.eos_token_ids = self.cfg.eos_token_ids + (marker_id,)
        transform = None
        quant_policy = None
        if self.rc.quant_bits:
            from .quant import QuantPolicy

            quant_policy = QuantPolicy(
                bits=self.rc.quant_bits,
                group_size=self.rc.quant_group_size,
                mode=self.rc.quant_mode,
                quantize_attention=self.rc.quant_attention,
                quantize_mlp=self.rc.quant_mlp,
                quantize_router=self.rc.quant_router,
                quantize_lm_head=self.rc.quant_lm_head,
                min_dim=self.rc.quant_min_dim,
            )
            transform = quant_policy.transform
        warm = None
        if self.rc.warm_mb:
            from .warm_tier import WarmTier

            warm = WarmTier(self.rc.warm_mb * 1_000_000)
        self.cache = WeightCache(self.store, self.rc.max_weight_cache_mb * 1_000_000, transform, warm,
                                  max_fetch_batch=self.rc.expert_fetch_batch)
        self.timer = telemetry.Timer()
        # F42: per-expert page byte estimate for pre-allocation reservations.
        # moe_intermediate_size when the config has it, else the dense size
        # (over-estimate = conservative); MXFP4 stores ~0.53 B/weight.
        inter = getattr(self.cfg, "moe_intermediate_size", None) or self.cfg.intermediate_size
        if self.store.on_disk_quantized:
            resident_bytes_per_weight = self.store.quantized_bytes_per_weight
        elif self.rc.quant_bits:
            resident_bytes_per_weight = self.rc.quant_bits / 8 + (
                8 / self.rc.quant_group_size
                if self.rc.quant_mode == "affine"
                else 1 / self.rc.quant_group_size
            )
        else:
            resident_bytes_per_weight = (
                0.6 if self.cfg.model_type == "gpt_oss" else 2)
        dense_expert_page_bytes = int(
            3 * self.cfg.hidden_size * inter * 2)
        self._expert_page_bytes = int(
            3 * self.cfg.hidden_size * inter * resident_bytes_per_weight)
        self._expert_storage_page_bytes = int(
            3 * self.cfg.hidden_size * inter
            * self.store.expert_storage_bytes_per_weight)
        self._expert_storage_page_bytes = (
            self.store.estimate_expert_storage_page_bytes(
                self.cfg.moe_expert_prefix,
                self._expert_storage_page_bytes,
            )
        )
        # Admission must cover peak load representation, not only the object
        # that survives in WeightCache. A standard pre-quantized checkpoint
        # loads its compact QTensor directly. Runtime quantize-on-load first
        # materializes every BF16 source tensor and retains it while building
        # the compact result, so count both. K2.5's released compressed-tensors
        # INT4 path dequantizes to a dense BF16 expert and therefore naturally
        # lands on the dense resident estimate here.
        self._expert_fetch_page_bytes = (
            self._expert_page_bytes
            if self.store.on_disk_quantized or not self.rc.quant_bits
            else dense_expert_page_bytes + self._expert_page_bytes
        )
        self._layer_transient = 0  # F42: measured compute-scratch high-water mark
        self._token_transient = 0  # F42: whole-token transient (greedy sync point)
        # 2026-07-13: F42's own per-layer/per-token mx.reset_peak_memory() calls
        # (below) mean a caller bracketing a whole generate() with reset_peak_memory
        # + get_peak_memory() gets a near-meaningless number — it only reflects
        # whatever the LAST reset window happened to peak at, not the true
        # across-the-whole-call maximum. Confirmed live: a local-context probe
        # (docs/benchmark_results.md, "Local large-context probe") got 4.04GB from
        # exactly that bracketing pattern while the governor's continuous polling
        # of the SAME mx.get_active_memory()/get_peak_memory() calls caught
        # 9.1-11.5GB during the same run. This tracker piggybacks on the peak
        # reads F42 ALREADY does (zero extra mx calls) and keeps a running max
        # that nothing else ever resets, so a caller can trust it end-to-end.
        self._true_peak_metal_bytes = 0
        # F68: a second, independently-resettable running max, fed by the same
        # _note_true_peak() reads — lets a caller (the chunking loop) measure
        # "true peak reached during just THIS chunk" without disturbing the
        # whole-call tracker above, by resetting this one before each chunk.
        self._chunk_peak_metal_bytes = 0
        self._tap_hidden: dict[int, mx.array] = {}  # F62: optional hidden-state taps
        # F43: a declared context bound <= index_topk means the DSA indexer can
        # never deselect anything — elide its weights and state entirely.
        self._dsa_elided = bool(
            self.cfg.model_type == "glm_moe_dsa" and self.cfg.index_topk
            and self.rc.context_bound and self.rc.context_bound <= self.cfg.index_topk
        )

        # ---- pin persistent tensors ----
        # Embeddings and final norm are touched every token; norm is bytes-sized so
        # it is pinned unconditionally alongside them.
        self._embed_rows = None
        if (self.rc.embed_rows and not self.cfg.tie_word_embeddings
                and not self.store.is_quantized("model.embed_tokens.weight")):
            from .embed_rows import EmbedRows

            self._embed_rows = EmbedRows(self._model_dir, self.store, self.cfg.hidden_size)

        self._streamed_lm_head = None
        if (self.rc.stream_lm_head and not self.cfg.tie_word_embeddings
                and not self.store.is_quantized("lm_head.weight")
                and self.store.has("lm_head.weight")
                and not self.store.vpack2 and not self.store.packed):
            from .lm_head_stream import StreamedLMHead

            self._streamed_lm_head = StreamedLMHead(
                self.store.dir, self.store.weight_map,
                real_name=self.store._real_name.get("lm_head.weight", "lm_head.weight"))

        pin_names = ["model.norm.weight"]
        if self.rc.pin_embeddings and self._embed_rows is None:
            pin_names.append("model.embed_tokens.weight")
        if ((self.rc.pin_lm_head or self.rc.rerank_lm_head)
                and self._streamed_lm_head is None
                and not self.cfg.tie_word_embeddings and self.store.has("lm_head.weight")):
            pin_names.append("lm_head.weight")
        persistent = self.cache.pin("persistent", pin_names)

        self._embed_w = persistent.get("model.embed_tokens.weight")
        self._norm_w = persistent["model.norm.weight"]
        self._lm_head_w = persistent.get("lm_head.weight")
        self._reranked_lm_head_bytes = 0
        if self.rc.rerank_lm_head:
            from .quant import QTensor, make_reranked_q_head

            if self.cfg.tie_word_embeddings:
                raise ValueError(
                    "rerank_lm_head currently requires an untied exact LM head")
            if self._streamed_lm_head is not None or self._lm_head_w is None:
                raise ValueError(
                    "rerank_lm_head requires a resident exact LM head")
            if isinstance(self._lm_head_w, QTensor):
                raise ValueError(
                    "rerank_lm_head requires an unquantized exact LM head")
            self._lm_head_w = make_reranked_q_head(
                self._lm_head_w,
                candidates=self.rc.rerank_lm_head_candidates,
                group_size=self.rc.rerank_lm_head_group_size,
                bits=self.rc.rerank_lm_head_bits,
                mode=self.rc.rerank_lm_head_mode,
            )
            self._reranked_lm_head_bytes = self._lm_head_w.approx.nbytes
        self._tied_lm_head_w = None
        if self.cfg.tie_word_embeddings:
            from .quant import QTensor

            if isinstance(self._embed_w, QTensor):
                # A pre-quantized MLX checkpoint can use the same packed rows
                # for selective embedding dequantization and output matmul.
                self._tied_lm_head_w = self._embed_w
            elif self.rc.quantize_tied_lm_head and quant_policy is not None:
                # Quantize a second view under the lm_head name; the original
                # BF16 matrix remains available for cheap indexed lookup.
                embed_weight = (self._embed_w if self._embed_w is not None
                                else self._embed_weight())
                self._tied_lm_head_w = quant_policy.transform(
                    "lm_head.weight", embed_weight)
                self._eval_weight(self._tied_lm_head_w)

        n = self.cfg.num_hidden_layers
        pinned_layers = set(range(self.rc.pin_first_layers)) | set(
            range(n - self.rc.pin_last_layers, n)
        )
        for i in sorted(pinned_layers):
            # _layer_names: for MoE models this pins attention/norms/router only —
            # experts page separately (pinning all experts would defeat the point)
            self.cache.pin(self._layer_key(i), self._layer_names(i))

        self.expert_usage: dict[tuple[int, int], int] = {}
        self.expert_hits = 0
        self.expert_misses = 0
        self.expert_trace: list[tuple[int, tuple[int, ...]]] = []  # (layer, routed ids) per fetch, in sweep order
        self._expert_compute_batches = 0
        self._max_experts_per_compute_batch = 0
        self._adaptive_expert_batch_clamps = 0
        self._min_adaptive_expert_batch = 0
        self._resident_fast_decode_sweeps = 0
        self._resident_fast_prefill_sweeps = 0
        self._disable_resident_fast_for_request = False
        self._resident_fast_layers = None
        self._resident_fast_evictions = -1
        self._resident_moe_layers = None
        self._resident_moe_bytes = 0
        self._resident_attention_bytes = 0
        self._resident_moe_sweeps = 0
        self._rope_freqs = None
        self._mscale = 1.0
        self.rope_profile = "released"
        self.rope_cache_identity = "released"
        self.effective_max_position_embeddings = int(self.cfg.max_position_embeddings)
        qwen_yarn_factor = float(self.rc.qwen_yarn_factor or 0.0)
        if not math.isfinite(qwen_yarn_factor):
            raise ValueError("qwen_yarn_factor must be finite")
        if qwen_yarn_factor < 0 or (0 < qwen_yarn_factor < 1):
            raise ValueError("qwen_yarn_factor must be 0 or at least 1")
        if qwen_yarn_factor > 1 and self.cfg.model_type != "qwen2":
            raise ValueError("qwen_yarn_factor is supported only for Qwen2 checkpoints")
        if self.cfg.model_type == "qwen2":
            scaling = self.cfg.rope_scaling or {}
            from .rope import supported_qwen_rope_type

            rope_type = supported_qwen_rope_type(scaling)
            checkpoint_yarn = rope_type == "yarn" and qwen_yarn_factor <= 1
            if qwen_yarn_factor > 1 or checkpoint_yarn:
                factor = (qwen_yarn_factor if qwen_yarn_factor > 1
                          else float(scaling["factor"]))
                original = int(scaling.get(
                    "original_max_position_embeddings",
                    self.cfg.max_position_embeddings,
                ))
                beta_fast = float(scaling.get("beta_fast", 32.0))
                beta_slow = float(scaling.get("beta_slow", 1.0))
                mscale = float(scaling.get("mscale", 1.0))
                mscale_all_dim = float(scaling.get("mscale_all_dim", 0.0))
                from .rope import yarn_parameters

                freqs, self._mscale = yarn_parameters(
                    self.cfg.head_dim, self.cfg.rope_theta, factor, original,
                    beta_fast=beta_fast, beta_slow=beta_slow,
                    mscale=mscale, mscale_all_dim=mscale_all_dim,
                )
                self._rope_freqs = mx.array(freqs, dtype=mx.float32)
                mx.eval(self._rope_freqs)
                if qwen_yarn_factor > 1:
                    self.effective_max_position_embeddings = int(original * factor)
                    self.rope_profile = f"experimental-qwen-yarn-{factor:g}x"
                else:
                    self.effective_max_position_embeddings = max(
                        int(self.cfg.max_position_embeddings), int(original * factor))
                    self.rope_profile = f"checkpoint-qwen-yarn-{factor:g}x"
                self.rope_cache_identity = (
                    "qwen-yarn-v1:"
                    f"factor={factor.hex()}:original={original}:"
                    f"beta_fast={beta_fast.hex()}:beta_slow={beta_slow.hex()}:"
                    f"mscale={mscale.hex()}:mscale_all_dim={mscale_all_dim.hex()}"
                )
        if self.cfg.model_type == "gpt_oss":
            self._rope_freqs, self._mscale = _gptoss_rope_state(
                self.cfg, packed=self.store.packed)
            mx.eval(self._rope_freqs)
            # Quarantined: runtime/gptoss.py still differs from OpenAI's
            # inverse-frequency/truncate:false reference. Make that visible in
            # telemetry and invalidate any older cache identity rather than
            # presenting the profile as conformance-validated.
            self.rope_profile = "checkpoint-gptoss-yarn-unvalidated"
            self.rope_cache_identity = "checkpoint-gptoss-yarn-unvalidated-v2"
        if self.cfg.num_experts and not self.store.packed:
            # 2026-07-19 (benchmark-sweep follow-up): some checkpoints ship
            # experts as ONE fused tensor per projection (e.g. Qwen3-VL-235B's
            # real weight_map has "...mlp.experts.gate_up_proj" / "...down_proj",
            # not per-expert "...experts.{e}.gate_proj.weight") -- the engine's
            # expert-fetch path only understands the per-expert-indexed layout
            # and previously failed deep inside a request with a raw, confusing
            # KeyError instead of a clear diagnostic at load time.
            probe_layer = self.cfg.first_k_dense_replace
            if not self.store.names_with_prefix(
                    f"model.layers.{probe_layer}.{self.cfg.moe_expert_prefix}.0."):
                raise RuntimeError(
                    f"{self._model_dir.name}: MoE experts are not in the "
                    f"per-expert-indexed layout this engine expects under "
                    f"'{self.cfg.moe_expert_prefix}.<id>.*' (checked layer "
                    f"{probe_layer}) -- this checkpoint likely ships fused "
                    "per-projection expert tensors instead. Run "
                    "formats.packed.pack_model first, matching gpt-oss's "
                    "checkpoints (this project's own EXPERT unfuse/pack step, "
                    "not a fla-core/compressed-tensors concept)."
                )
        if self.rc.resident_moe_decode:
            self._build_resident_moe_layers()
        self.predictor = None
        if self.cfg.num_experts:
            from .predictor import MarkovExpertPredictor

            self.predictor = MarkovExpertPredictor(
                self.cfg.num_hidden_layers, self.cfg.num_experts,
                path=self._model_dir / "expert_transitions.json",
            )

        # Prefetcher sized against a typical page so budget checks are meaningful.
        if self.cfg.num_experts:
            # Use the actual per-expert width/format estimate. GLM's dense
            # intermediate_size is 12,288 but routed experts are 2,048; the old
            # hint was 6x too large (~453 MB vs ~75.5 MB) and skipped prefetches
            # that comfortably fit the cache budget.
            layer_bytes = self._expert_page_bytes
        else:
            layer_bytes = self._estimate_layer_bytes()
        if self.rc.quant_bits and not self.cfg.num_experts:
            layer_bytes = int(layer_bytes * (self.rc.quant_bits / 16) * 1.15)  # + scales/biases
        workers = (self.rc.prefetch_workers or
                   (2 if self.store.packed else 1))
        self.prefetcher = (
            Prefetcher(self.cache, page_size_hint=layer_bytes, workers=workers)
            if self.rc.prefetch_depth and self._resident_moe_layers is None
            else None
        )

        # F19: deliberate warm-start — preload the historically hottest expert
        # pages (heat derived from the persisted transition counts) onto the
        # prefetch workers. Budget-aware via the prefetcher's would_fit guard;
        # the governor can pause it under pressure.
        if self.rc.warm_start and self.predictor is not None and self.prefetcher is not None:
            from collections import defaultdict

            heat: dict[tuple[int, int], int] = defaultdict(int)
            for (l, e, f), c in self.predictor.counts.items():
                heat[(l, e)] += c
                heat[(l + 1, f)] += c
            top = sorted(heat.items(), key=lambda kv: -kv[1])[: self.rc.warm_start]
            for (l, e), _ in top:
                if l < self.cfg.num_hidden_layers:
                    self.prefetcher.schedule(
                        f"layer.{l}.expert.{e}",
                        self.store.names_with_prefix(
                            f"model.layers.{l}.{self.cfg.moe_expert_prefix}.{e}."),
                    )

        # F16: memory-pressure governor — sheds prefetch, MLX scratch, then cache
        # budget when the SYSTEM (not just our budget) runs short. Default on.
        self.governor = None
        if self.rc.governor:
            from .pressure import MemoryGovernor

            self.governor = MemoryGovernor(
                self.cache,
                self.prefetcher,
                floor_bytes=min(
                    self.cache.max_bytes,
                    self.rc.min_weight_cache_mb * 1_000_000,
                ),
            )

        # Hot-prompt-kv disk persistence (2026-07-15, generalized to a
        # parent-hashed segment DAG later the same day -- see
        # runtime/hot_kv_persist.py's module docstring): reload whatever
        # survived the last restart BEFORE the first request arrives, so a
        # conversation can resume warm instead of paying a full cold prefill
        # again.
        self._hot_kv_persist = None
        self._completed_generations = 0
        # A runtime-quantized dense Qwen checkpoint transforms its BF16 weights
        # lazily on the first full layer sweep.  Restoring a large exact KV at
        # construction can skip prefill, pushing that multi-GB transform into
        # the first decode token while the restored KV is also resident.  On a
        # 16 GB unified-memory host that ordering is unsafe even though either
        # state fits by itself.  Keep durable KV disk-lazy until one ordinary
        # generation has bootstrapped the resident packed/quantized weights.
        self._defer_persisted_kv_until_bootstrap = (
            self._should_defer_persisted_kv_until_bootstrap())
        if self.rc.hot_prompt_kv and self.rc.hot_prompt_kv_persist_dir:
            from .hot_kv_persist import HotPromptKVPersistence

            self._hot_kv_persist = HotPromptKVPersistence(
                self.rc.hot_prompt_kv_persist_dir, self._get_kv_fingerprint(),
                self.rc.hot_prompt_kv_chunk_size,
                max_checkpoints=self.rc.hot_prompt_kv_persist_max_checkpoints,
                config=self.cfg,
                require_dsa=(
                    self.cfg.model_type == "glm_moe_dsa"
                    and bool(self.cfg.index_topk)
                    and not self._dsa_elided),
                require_recurrent=(
                    self.cfg.model_type in (
                        "kimi_linear", "qwen3_5_moe", "qwen3_5")),
            )
            if not self._defer_persisted_kv_until_bootstrap:
                for (tokens, kv, logits, prompt_length, prompt_logits,
                     reusable_prefix, approximate, tool_capsules,
                     segment_chain, persisted_namespace
                     ) in self._hot_kv_persist.load_all(
                        self.cfg.num_hidden_layers, self.rc.hot_prompt_kv_slots):
                    self._hot_prompt_slots.append(_HotPromptSlot(
                        tokens=tokens, kv=kv, logits=logits,
                        prompt_length=prompt_length, prompt_logits=prompt_logits,
                        reusable_prefix=reusable_prefix, approximate=approximate,
                        tool_capsules=tool_capsules,
                        segment_chain=segment_chain,
                        cache_namespace=persisted_namespace,
                    ))
            else:
                print(
                    "[hot-kv] durable restore deferred until dense-Qwen "
                    "weight bootstrap completes", flush=True)

    @staticmethod
    def _kv_nbytes(kv) -> int:
        measure = getattr(kv, "allocated_nbytes", None)
        if measure is None:
            measure = getattr(kv, "nbytes", None)
        try:
            return max(0, int(measure())) if measure is not None else 0
        except (AttributeError, TypeError, ValueError):
            return 0

    def _should_defer_persisted_kv_until_bootstrap(self) -> bool:
        """Whether restart KV restore would collide with lazy weight packing."""
        return bool(
            self.cfg.model_type in ("qwen2", "qwen3")
            and not self.cfg.vision_config
            and not self.cfg.num_experts
            and self.rc.quant_bits
            and self.rc.resident_fast_decode
            and not self.store.on_disk_quantized
        )

    def _persisted_kv_restore_allowed(self) -> bool:
        return bool(
            not self._defer_persisted_kv_until_bootstrap
            or self._completed_generations > 0
        )

    def prompt_cache_memory_snapshot(self) -> dict:
        """Return live/evictable prompt-KV bytes for server preflight.

        ``mx.get_active_memory()`` already includes these arrays.  Reporting
        them separately lets admission project the state *after* unmatched
        hot branches are released, instead of adding a new full KV on top of
        an unrelated retained one and discovering the collision mid-stream.
        """
        seen = set()
        retained = 0
        for slot in self._hot_prompt_slots:
            identity = id(slot.kv)
            if identity in seen:
                continue
            seen.add(identity)
            retained += self._kv_nbytes(slot.kv)
        orphan = 0
        last_kv = getattr(self, "last_kv", None)
        if last_kv is not None and id(last_kv) not in seen:
            orphan = self._kv_nbytes(last_kv)
        return {
            "active_metal_bytes": int(mx.get_active_memory()),
            "retained_prompt_kv_bytes": retained,
            "orphan_prompt_kv_bytes": orphan,
            "evictable_prompt_kv_bytes": retained + orphan,
            "hot_prompt_slots": len(self._hot_prompt_slots),
            "metal_ceiling_bytes": (
                int(self.governor.current_ceiling())
                if self.governor is not None else 0),
        }

    def _project_dense_text_kv_bytes(self, positions: int) -> int:
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            full_layers = sum(
                layer_type == "full_attention"
                for layer_type in self.cfg.layer_types)
            attention = (
                positions * full_layers * 2
                * int(self.cfg.num_key_value_heads)
                * int(self.cfg.head_dim) * 2)
            linear_layers = max(
                0, int(self.cfg.num_hidden_layers) - full_layers)
            recurrent = (
                linear_layers * int(self.cfg.linear_num_value_heads)
                * int(self.cfg.linear_key_head_dim)
                * int(self.cfg.linear_value_head_dim) * 4)
            conv_width = (
                2 * int(self.cfg.linear_num_key_heads)
                * int(self.cfg.linear_key_head_dim)
                + int(self.cfg.linear_num_value_heads)
                * int(self.cfg.linear_value_head_dim))
            conv = (
                linear_layers
                * max(0, int(self.cfg.linear_conv_kernel_dim) - 1)
                * conv_width * 2)
            return attention + recurrent + conv
        if (self.cfg.model_type not in ("qwen2", "qwen3")
                or self.cfg.vision_config or self.cfg.num_experts):
            return 0
        layers = int(self.cfg.num_hidden_layers or 0)
        kv_heads = int(self.cfg.num_key_value_heads or 0)
        head_dim = int(self.cfg.head_dim or 0)
        if min(layers, kv_heads, head_dim, positions) <= 0:
            return 0
        return positions * layers * 2 * kv_heads * head_dim * 2

    @staticmethod
    def _hot_namespace_priority(namespace: str) -> int:
        # The execution branch contains the selected real schemas and is the
        # expensive state needed throughout a tool loop. The decision branch
        # remains durable on disk but is the first in-memory eviction choice.
        if namespace == "gateway_execution":
            return 2
        if namespace == "gateway_decision":
            return 0
        return 1

    def _append_hot_prompt_slot(self, slot: _HotPromptSlot) -> tuple[int, int]:
        """Insert with phase-aware bounded retention; return evicted count/bytes."""
        self._hot_prompt_slots.append(slot)
        evicted_count = 0
        evicted_bytes = 0
        capacity = max(1, self.rc.hot_prompt_kv_slots)
        while len(self._hot_prompt_slots) > capacity:
            # Lowest phase priority goes first; ties retain ordinary LRU order.
            victim_index = min(
                range(len(self._hot_prompt_slots)),
                key=lambda index: (
                    self._hot_namespace_priority(
                        getattr(self._hot_prompt_slots[index],
                                "cache_namespace", "default")),
                    index,
                ),
            )
            victim = self._hot_prompt_slots.pop(victim_index)
            evicted_count += 1
            evicted_bytes += self._kv_nbytes(victim.kv)
            if victim.kv is not slot.kv:
                self._release_kv(victim.kv)
        return evicted_count, evicted_bytes

    def _evict_hot_slots_for_admission(
            self, required_total_kv_bytes: int, keep_kv,
            cache_namespace: str, *, transient_bytes: int = 0) -> dict:
        """Free persisted/unmatched branches until the next KV fits safely.

        Hot slots are already checkpointed before entering the LRU whenever
        persistence is enabled, so this is a RAM eviction, not a cache loss.
        Without persistence it is still preferable to discard an unrelated
        prefix than to enter macOS compression or fail the same allocation on
        every automatic retry.
        """
        current_bytes = self._kv_nbytes(keep_kv) if keep_kv is not None else 0
        incoming = max(0, int(required_total_kv_bytes) - current_bytes)
        transient = max(0, int(transient_bytes))
        stats = {
            "evicted_slots": 0,
            "evicted_bytes": 0,
            "evicted_persisted_slots": 0,
            "projected_incoming_bytes": incoming,
            "projected_transient_bytes": transient,
            "system_available_bytes": int(psutil.virtual_memory().available),
            "system_available_floor_bytes": int(
                getattr(getattr(self, "rc", None),
                        "hot_prompt_kv_min_available_mb", 0) * 1_000_000),
            "governor_reservations": 0,
        }
        if self.governor is None or incoming + transient <= 0:
            return stats

        margin = 400_000_000
        system_floor = stats["system_available_floor_bytes"]

        def pressure_sample():
            active = int(mx.get_active_memory())
            available = int(psutil.virtual_memory().available)
            ceiling = int(self.governor.current_ceiling())
            unsafe = (
                active + incoming + transient + margin > ceiling
                or (system_floor > 0
                    and available - incoming - transient < system_floor)
            )
            return active, available, ceiling, unsafe

        _active, available, ceiling, unsafe = pressure_sample()
        while unsafe:
            candidates = [
                index for index, slot in enumerate(self._hot_prompt_slots)
                if slot.kv is not keep_kv
            ]
            if not candidates:
                break
            # Prefer another phase, then the transient decision phase, then LRU.
            victim_index = min(
                candidates,
                key=lambda index: (
                    int(getattr(self._hot_prompt_slots[index],
                                "cache_namespace", "default")
                        == cache_namespace),
                    self._hot_namespace_priority(
                        getattr(self._hot_prompt_slots[index],
                                "cache_namespace", "default")),
                    index,
                ),
            )
            victim = self._hot_prompt_slots.pop(victim_index)
            victim_bytes = self._kv_nbytes(victim.kv)
            stats["evicted_slots"] += 1
            stats["evicted_bytes"] += victim_bytes
            stats["evicted_persisted_slots"] += int(bool(
                getattr(victim, "segment_chain", ())))
            self._release_kv(victim.kv)
            del victim
            mx.clear_cache()
            _active, available, ceiling, unsafe = pressure_sample()

        # The old path stopped once no retained KV remained, even if the next
        # allocation would still push system-available memory below an optional
        # operator reserve. Ask the governor to reclaim weight-cache pages as
        # the second tier. Its ceiling independently preserves `critical`;
        # choosing the remainder as margin additionally enforces an explicitly
        # configured `available - incoming >= system_floor` policy.
        reserve = getattr(self.governor, "reserve", None)
        reservations_before = int(getattr(
            self.governor, "reservations", 0) or 0)
        reservation_bytes = incoming + transient
        if callable(reserve) and reservation_bytes > 0:
            # `_resident_fast_layers` is a convenience tuple of the same dense
            # arrays owned by WeightCache. Cache-budget shrink can evict their
            # entries, but this second strong reference kept every tensor live
            # until the *next* sweep noticed the eviction counter—too late for
            # this pre-allocation reservation. Drop the view before asking the
            # governor to reclaim pages; the following sweep rebuilds it from
            # whatever cache budget remains.
            if getattr(self, "_resident_fast_layers", None) is not None:
                self._resident_fast_layers = None
                self._resident_fast_evictions = -1
                mx.clear_cache()
            critical = int(getattr(self.governor, "critical", 0) or 0)
            reserve_margin = max(margin, system_floor - critical)
            reserve(reservation_bytes, margin=reserve_margin)
        stats["governor_reservations"] = max(0, int(getattr(
            self.governor, "reservations", 0) or 0) - reservations_before)
        stats["system_available_bytes"] = int(
            psutil.virtual_memory().available)
        return stats

    def _note_true_peak(self):
        p = mx.get_peak_memory()
        if p > self._true_peak_metal_bytes:
            self._true_peak_metal_bytes = p
        if p > self._chunk_peak_metal_bytes:
            self._chunk_peak_metal_bytes = p

    # ---- weights access -------------------------------------------------

    def _layer_key(self, i: int) -> str:
        return f"layer.{i}"

    def _layer_names(self, i: int) -> list[str]:
        """Names for the always-needed part of a layer. For MoE layers this is
        attention + norms + router — experts page separately, after routing."""
        names = self.store.layer_param_names(i)
        if self.cfg.num_experts:
            expert_marker = f".{self.cfg.moe_expert_prefix}."
            names = [n for n in names if expert_marker not in n]
        if self._dsa_elided:
            # F43: with S bounded <= index_topk the indexer selects every position
            # by construction — its weights can never affect output. Skip the bytes.
            names = [n for n in names if ".self_attn.indexer." not in n]
        return names

    def _layer_fetch_bytes_estimate(self, layer: int) -> int:
        """Conservative materialized trunk-page estimate for pre-fetch eviction.

        K2.5's checkpoint stores its trunk/router/shared-expert tensors as BF16;
        only routed experts use compressed-tensors INT4 and they have their own
        lifetime-bounded fetch path. Keeping this architecture-specific avoids
        inventing unsafe generic estimates for packed/fused formats whose load
        representation differs from their logical shapes.
        """
        if self.cfg.model_type != "kimi_k25":
            return 0
        c = self.cfg
        h = c.hidden_size
        heads = c.num_attention_heads
        q_width = heads * (c.qk_nope_head_dim + c.qk_rope_head_dim)
        kv_width = heads * (c.qk_nope_head_dim + c.v_head_dim)
        params = (
            h * c.q_lora_rank
            + c.q_lora_rank * q_width
            + h * (c.kv_lora_rank + c.qk_rope_head_dim)
            + c.kv_lora_rank * kv_width
            + heads * c.v_head_dim * h
            + c.q_lora_rank + c.kv_lora_rank
            + 2 * h
        )
        is_dense = (
            c.mlp_layer_types[layer] == "dense"
            if layer < len(c.mlp_layer_types)
            else layer < c.first_k_dense_replace
        )
        if is_dense:
            params += 3 * h * c.intermediate_size
        else:
            params += c.num_experts * h + c.num_experts
            params += 3 * h * c.moe_intermediate_size * max(1, c.n_shared_experts)
        # BF16 plus 5% for small architecture tensors/metadata omitted above.
        return math.ceil(params * 2 * 1.05)

    def _checkpoint_payload_bytes(self) -> int:
        """Conservative physical payload estimate for resident qualification."""
        index_path = self._model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            import json

            index = json.loads(index_path.read_text())
            total = index.get("metadata", {}).get("total_size")
            if isinstance(total, int) and total > 0:
                return total
            shards = set(index.get("weight_map", {}).values())
            if shards:
                return sum((self._model_dir / shard).stat().st_size for shard in shards)
        return sum(path.stat().st_size for path in self._model_dir.glob("*.safetensors"))

    @staticmethod
    def _eval_weight(value) -> None:
        from .quant import QTensor

        if isinstance(value, QTensor):
            arrays = [value.wq, value.scales]
            if value.biases is not None:
                arrays.append(value.biases)
            mx.eval(arrays)
        else:
            mx.eval(value)

    def _build_resident_moe_layers(self) -> None:
        """Fuse a small prequantized OLMoE checkpoint into gathered experts.

        Out-of-core MoEs keep independent expert pages.  When the complete
        quantized artifact safely fits, retaining that Python/page schedule is
        pure overhead: stack each projection once and leave routing lazy on the
        Metal graph, matching MLX-LM's SwitchGLU execution shape.
        """
        if self.cfg.model_type != "olmoe" or not self.store.on_disk_quantized:
            return
        payload = self._checkpoint_payload_bytes()
        safe = int(self.cache.max_bytes * 0.85)
        if payload <= 0 or payload > safe:
            return

        attention_policy = None
        if self.rc.resident_attention_mode:
            from .quant import QuantPolicy

            attention_policy = QuantPolicy(
                bits=self.rc.resident_attention_bits,
                group_size=self.rc.resident_attention_group_size,
                mode=self.rc.resident_attention_mode,
                quantize_attention=True,
                quantize_mlp=False,
                quantize_router=False,
                quantize_lm_head=False,
                min_dim=0,
            )

        resident_layers = []
        resident_bytes = 0
        for layer in range(self.cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            trunk_names = self._layer_names(layer)
            expert_names = [
                name
                for expert in range(self.cfg.num_experts)
                for name in self.store.names_with_prefix(
                    f"{prefix}.mlp.experts.{expert}.")
            ]
            values, seconds, nbytes = self.store.fetch(trunk_names + expert_names)
            self.cache.stats.disk_s += seconds
            self.cache.stats.bytes_read += nbytes
            trunk = {}
            for name in trunk_names:
                value = values[name]
                if (attention_policy is not None
                        and ".self_attn." in name):
                    transformed = attention_policy.transform(name, value)
                    if transformed is not value:
                        value = transformed
                        self._resident_attention_bytes += value.nbytes
                trunk[name] = value
            fused = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                projection_weights = [
                    values[f"{prefix}.mlp.experts.{expert}.{projection}.weight"]
                    for expert in range(self.cfg.num_experts)
                ]
                fused[projection] = layer_runner.stack_expert_weights(
                    projection_weights)
                self._eval_weight(fused[projection])
            resident_bytes += sum(weight.nbytes for weight in trunk.values())
            resident_bytes += sum(weight.nbytes for weight in fused.values())
            resident_layers.append((trunk, fused))
            del values

        self._resident_moe_layers = tuple(resident_layers)
        self._resident_moe_bytes = resident_bytes
        self._note_true_peak()
        print(
            f"[engine] resident fused OLMoE: {resident_bytes / 1e9:.2f} GB, "
            f"source payload {payload / 1e9:.2f} GB"
            + (f", {self.rc.resident_attention_mode} attention "
               f"{self._resident_attention_bytes / 1e9:.2f} GB"
               if self.rc.resident_attention_mode else ""),
            flush=True,
        )

    def begin_provisional(self):
        """F55: buffer ROUTING statistics (usage/heat + predictor) during a
        speculative verify sweep; commit only the accepted prefix afterwards.
        Cache hit/miss counting is NOT deferred — those fetches physically
        happened, so LFU frequency remains correct either way."""
        self._provisional = []

    def commit_provisional(self, accepted_positions: int):
        """Replay buffered routing observations, keeping only experts routed
        by a COMMITTED window position (< accepted_positions)."""
        buf, self._provisional = self._provisional, None
        for layer, positions in buf:
            kept = [e for e, poss in positions.items()
                    if any(p < accepted_positions for p in poss)]
            for e in kept:
                self.expert_usage[(layer, e)] = self.expert_usage.get((layer, e), 0) + 1
            if self.predictor is not None and kept:
                self.predictor.observe(layer, sorted(kept))

    def _record_expert_route(self, layer: int, expert_ids: list[int],
                             positions: dict[int, list[int]] | None = None) -> None:
        """Record one routed UNION exactly once, independent of compute batches."""
        provisional = getattr(self, "_provisional", None)
        if provisional is not None and positions is not None:
            provisional.append((layer, positions))
        for e in expert_ids:
            if provisional is None:
                self.expert_usage[(layer, e)] = self.expert_usage.get((layer, e), 0) + 1
        self.expert_trace.append((layer, tuple(expert_ids)))
        # Phase 8: learn routing transitions and prefetch next layer's likely experts
        # (F55: during a provisional sweep, observation is deferred to commit)
        if self.predictor is not None:
            if provisional is None:
                self.predictor.observe(layer, expert_ids)
            if self.prefetcher and self.rc.expert_predictive_prefetch:
                for e in self.predictor.predict(layer, expert_ids, top_m=self.cfg.num_experts_per_tok):
                    self.prefetcher.schedule(
                        f"layer.{layer + 1}.expert.{e}",
                        self.store.names_with_prefix(
                            f"model.layers.{layer + 1}.{self.cfg.moe_expert_prefix}.{e}."),
                        only_if_idle=self.rc.expert_prefetch_idle_only,
                    )

    def export_expert_trace(self, path: str | Path) -> Path:
        """Write routed unions for offline layout/prefetch simulation.

        This exports decisions the authoritative router already made; it does
        not evaluate activations, fetch weights, or alter generation.  Sweep
        boundaries are reconstructed from the strictly increasing layer order.
        """
        from .expert_plan import write_trace

        return write_trace(
            path,
            self.expert_trace,
            model=str(self._model_dir),
            num_experts=self.cfg.num_experts,
            expert_page_bytes=self._expert_storage_page_bytes,
        )

    def _fetch_experts(self, layer: int, expert_ids: list[int]) -> dict[int, dict]:
        """Fetch one lifetime-bounded expert batch; routing was recorded already."""
        items = []
        n_missing = 0
        for e in expert_ids:
            key = f"layer.{layer}.expert.{e}"
            if self.cache.contains(key):
                self.expert_hits += 1
            else:
                self.expert_misses += 1
                n_missing += 1
            items.append((
                key,
                self.store.names_with_prefix(
                    f"model.layers.{layer}.{self.cfg.moe_expert_prefix}.{e}."),
            ))
        if self.governor is not None and n_missing:
            # Reserve only the pages that can coexist in THIS compute batch.
            # Reserving the full routed union recreates the 16-22 GB false demand
            # even when fetch and compute lifetimes are correctly bounded.
            self.governor.reserve(
                n_missing * self._expert_fetch_page_bytes + self._layer_transient)

        t0 = time.perf_counter()
        pages = self.cache.get_many(items)
        self.timer.add("expert_wait", time.perf_counter() - t0)
        return {e: pages[f"layer.{layer}.expert.{e}"] for e in expert_ids}

    def _get_experts(self, layer: int, expert_ids: list[int],
                     positions: dict[int, list[int]] | None = None) -> dict[int, dict]:
        """Compatibility path: record and return the complete routed union."""
        self._record_expert_route(layer, expert_ids, positions)
        return self._fetch_experts(layer, expert_ids)

    def _iter_expert_batches(self, layer: int, expert_ids: list[int],
                             positions: dict[int, list[int]] | None = None):
        """Yield bounded expert pages for immediate compute and release.

        This is F74-v2's actual lifetime boundary. ``WeightCache.get_many`` may
        split disk fetches, but returning a dict for the whole union leaves every
        evicted tensor strongly referenced. The GLM runner consumes one yielded
        mapping, ``mx.eval`` materializes its accumulated output, then advances.
        """
        self._record_expert_route(layer, expert_ids, positions)
        position_union = {
            position for expert_positions in (positions or {}).values()
            for position in expert_positions
        }
        single_position = bool(positions) and len(position_union) == 1
        configured_batch_size = (
            self.rc.decode_expert_fetch_batch
            if single_position and self.rc.decode_expert_fetch_batch > 0
            else self.rc.expert_fetch_batch
        ) or len(expert_ids) or 1
        start = 0
        governor = getattr(self, "governor", None)
        while start < len(expert_ids):
            batch_size = min(configured_batch_size, len(expert_ids) - start)
            if governor is not None and batch_size > 1:
                admitted = governor.admissible_units(
                    unit_bytes=self._expert_fetch_page_bytes,
                    fixed_bytes=self._layer_transient,
                    max_units=batch_size,
                )
                if admitted < batch_size:
                    self._adaptive_expert_batch_clamps += 1
                batch_size = admitted
                self._min_adaptive_expert_batch = (
                    batch_size if self._min_adaptive_expert_batch == 0
                    else min(self._min_adaptive_expert_batch, batch_size)
                )
            batch_ids = expert_ids[start:start + batch_size]
            self._expert_compute_batches += 1
            self._max_experts_per_compute_batch = max(
                self._max_experts_per_compute_batch, len(batch_ids)
            )
            yield batch_ids, self._fetch_experts(layer, batch_ids)
            start += batch_size

    def _router_lookahead(self, x: mx.array, nxt: int) -> None:
        """F45 (MoE-SpeQ class; lossless — prefetch is only a cache hint):
        predict layer `nxt`'s routed experts by running its ACTUAL router on
        the current hidden state (routing is largely stable across one block)
        and prefetch that union. Token-conditioned, unlike the Markov
        transition predictor. Never blocks on disk: skips unless the next
        layer's page is already resident, and the default idle-only gate admits
        no backlog behind existing prefetch work."""
        key = self._layer_key(nxt)
        if not self.cache.contains(key):
            return
        w = self.cache.get(key, self._layer_names(nxt))
        p = f"model.layers.{nxt}"
        k = self.cfg.num_experts_per_tok
        ln = w.get(f"{p}.post_attention_layernorm.weight")
        h = mx.fast.rms_norm(x, ln, self.cfg.rms_norm_eps) if ln is not None else x
        router_w = w.get(f"{p}.mlp.router.weight")
        gate_w = w.get(f"{p}.mlp.gate.weight")
        if router_w is not None:  # gpt-oss: linear router + bias, top-k on logits
            logits = h @ router_w.T
            bias = w.get(f"{p}.mlp.router.bias")
            if bias is not None:
                logits = logits + bias
            idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
        elif gate_w is not None:
            if self.cfg.model_type in ("glm_moe_dsa", "kimi_k25"):
                scores = h.astype(mx.float32) @ gate_w.astype(mx.float32).T
            else:
                scores = (h @ gate_w.T).astype(mx.float32)
            bias = w.get(f"{p}.mlp.gate.e_score_correction_bias")
            if bias is not None:  # GLM noaux_tc: SELECTION uses sigmoid + bias
                scores = mx.sigmoid(scores) + bias
            idx = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        else:  # dense layer (e.g. GLM first_k_dense_replace)
            return
        mx.eval(idx)
        for e in sorted({int(i) for i in idx.reshape(-1).tolist()}):
            self.prefetcher.schedule(
                f"layer.{nxt}.expert.{e}",
                self.store.names_with_prefix(
                    f"model.layers.{nxt}.{self.cfg.moe_expert_prefix}.{e}."),
                only_if_idle=self.rc.expert_prefetch_idle_only,
            )

    def _estimate_layer_bytes(self) -> int:
        c = self.cfg
        per_layer_params = (
            c.hidden_size * c.head_dim * (c.num_attention_heads + 2 * c.num_key_value_heads)
            + c.head_dim * c.num_attention_heads * c.hidden_size
            + 3 * c.hidden_size * c.intermediate_size
            + 2 * c.hidden_size
        )
        return per_layer_params * 2  # bf16

    def _embed_weight(self) -> mx.array:
        if self._embed_w is not None:
            return self._embed_w
        return self.cache.get("embeddings", ["model.embed_tokens.weight"])["model.embed_tokens.weight"]

    def _embed(self, tokens: list[int]) -> mx.array:
        if self._embed_rows is not None:
            return self._embed_rows.lookup(tokens)
        return layer_runner.embed(mx.array(tokens), self._embed_weight())

    def _lm_head_weight(self):
        if self.cfg.tie_word_embeddings:
            return (self._tied_lm_head_w if self._tied_lm_head_w is not None
                    else self._embed_weight())
        if self._streamed_lm_head is not None:
            return self._streamed_lm_head
        if self._lm_head_w is not None:
            return self._lm_head_w
        return self.cache.get("lm_head", ["lm_head.weight"])["lm_head.weight"]

    def _final_logits(self, hidden: mx.array, head=None) -> mx.array:
        head = self._lm_head_weight() if head is None else head
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            from .qwen35 import final_logits

            return final_logits(
                hidden, self._norm_w, head, self.cfg.rms_norm_eps)
        return layer_runner.final_logits(
            hidden, self._norm_w, head, self.cfg.rms_norm_eps)

    def _all_logits(self, hidden: mx.array) -> mx.array:
        head = self._lm_head_weight()
        if self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
            from .qwen35 import all_logits

            return all_logits(
                hidden, self._norm_w, head, self.cfg.rms_norm_eps)
        return layer_runner.all_logits(
            hidden, self._norm_w, head, self.cfg.rms_norm_eps)

    # ---- inference --------------------------------------------------------

    def _sweep(self, x: mx.array, kv: KVCache, offset: int,
               final_mlp_last_only: bool = False, tap_layers=None) -> mx.array:
        # F62 (DSpark) prep: optional hidden-state taps, purely additive —
        # capturing `x` after a given layer must never change `x` itself or
        # any subsequent computation. `tap_layers=None` (the default, used by
        # every existing caller) skips capturing but still clears any stale
        # entries from a PRIOR tapped call, so _tap_hidden never holds data
        # from a call other than the most recent one. See
        # tests/test_f62_hidden_taps.py for the tap-on/off identity proof.
        self._tap_hidden = {}
        n = self.cfg.num_hidden_layers
        moe = bool(self.cfg.num_experts)
        if self._resident_moe_layers is not None and tap_layers is None:
            self._resident_moe_sweeps += 1
            for i, (weights, fused_experts) in enumerate(self._resident_moe_layers):
                last_only = (
                    final_mlp_last_only and i == n - 1 and x.shape[1] > 1)
                x = layer_runner.run_fused_moe_block(
                    x,
                    weights,
                    fused_experts,
                    f"model.layers.{i}",
                    self.cfg,
                    kv,
                    i,
                    offset,
                    mlp_last_only=last_only,
                    rope_freqs=self._rope_freqs,
                    rope_mscale=self._mscale,
                    fused_swiglu=self.rc.fused_swiglu,
                    mlx_router_semantics=True,
                )
            return x
        fast_layers = self._resident_fast_layers
        if (fast_layers is not None
                and self._resident_fast_evictions != self.cache.stats.evictions):
            # A governor/cache-budget shrink can invalidate full residency.
            # Drop our strong references immediately so eviction really frees
            # the pages, then re-qualify through the ordinary cache below.
            self._resident_fast_layers = None
            fast_layers = None
        fast_decode_eligible = (
            self.rc.resident_fast_decode and x.shape[1] == 1
            and not self._disable_resident_fast_for_request)
        fast_prefill_eligible = (
            self.rc.resident_fast_prefill_limit > 0
            and x.shape[1] > 1
            and offset + x.shape[1] <= self.rc.resident_fast_prefill_limit)
        fast_eligible = (
            not moe and tap_layers is None
            and (fast_decode_eligible or fast_prefill_eligible))
        if (fast_eligible and fast_layers is None
                and all(self.cache.contains(self._layer_key(i)) for i in range(n))):
            # Cache the already-resident page mappings. Re-taking 28 cache locks
            # for every Qwen decode token is pure Python bookkeeping; the
            # eviction generation above makes this shortcut self-invalidating.
            fast_layers = tuple(
                self.cache.get(self._layer_key(i), self._layer_names(i))
                for i in range(n)
            )
            self._resident_fast_layers = fast_layers
            self._resident_fast_evictions = self.cache.stats.evictions
        if fast_eligible and fast_layers is not None:
            # Once every dense layer is resident, per-layer mx.eval() calls are
            # pure synchronization overhead. Build one lazy graph through the
            # complete stack; greedy() (or forward_tokens' logits eval) remains
            # the required boundary. Prefill is separately bounded by total
            # position because long graphs need the ordinary layer-by-layer
            # governor/transient accounting below.
            if x.shape[1] == 1:
                self._resident_fast_decode_sweeps += 1
            else:
                self._resident_fast_prefill_sweeps += 1
            for i, w in enumerate(fast_layers):
                last_only = (
                    final_mlp_last_only and i == n - 1 and x.shape[1] > 1)
                x = layer_runner.run_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    mlp_last_only=last_only,
                    rope_freqs=self._rope_freqs, rope_mscale=self._mscale,
                    fused_swiglu=self.rc.fused_swiglu,
                )
            return x
        for i in range(n):
            # F36: on the last layer of a prefill whose only consumer is the last
            # position's logits, MLP outputs for earlier positions are dead —
            # attention still runs full-width so the KV cache stays complete.
            last_only = final_mlp_last_only and i == n - 1 and x.shape[1] > 1
            if self.prefetcher:
                for j in range(i + 1, min(i + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(self._layer_key(j), self._layer_names(j))

            t0 = time.perf_counter()
            layer_key = self._layer_key(i)
            layer_names = self._layer_names(i)
            if not self.cache.contains(layer_key):
                incoming_page = self._layer_fetch_bytes_estimate(i)
                if incoming_page:
                    self.cache.prepare_for(incoming_page)
                    if self.governor is not None:
                        self.governor.reserve(incoming_page)
            w = self.cache.get(layer_key, layer_names)
            self.timer.add("weights_wait", time.perf_counter() - t0)

            # 2026-07-13: F42's proactive reserve() was only ever called from
            # _get_experts (MoE expert fetch) and the per-token decode boundary
            # — NEITHER fires during a DENSE model's per-layer prefill compute
            # (layer_runner.run_block has no expert fetch at all). Live-measured
            # consequence: a cold 32K-token dense prefill sweep's true peak rose
            # monotonically, unchecked, from 10.12GB to 13.32GB across 28 layers
            # (docs/benchmark_results.md, "Diagnosis, same day"), protected only
            # by the governor's REACTIVE 2s poll — which a back-to-back,
            # no-repeats-yet layer sweep can outpace. Reserve using the SAME
            # learned _layer_transient this loop already tracks (previously only
            # read by MoE's _get_experts), so every layer type gets the same
            # proactive protection, not just MoE ones.
            if self.governor is not None and self._layer_transient:
                self.governor.reserve(self._layer_transient)

            t0 = time.perf_counter()
            # F42: learn the layer-compute scratch high-water mark; _get_experts
            # declares it to the governor before the next big allocation
            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            if self.cfg.model_type == "gpt_oss":
                from .gptoss import run_gptoss_block

                x = run_gptoss_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    self._get_experts, self._rope_freqs, self._mscale,
                    mlp_last_only=last_only,
                )
            elif self.cfg.model_type in ("glm_moe_dsa", "kimi_k25"):
                # F93: Kimi K2.5's language model is architecturally identical
                # to GLM's MLA+noaux_tc-MoE block (real q_lora MLA, real RoPE
                # -- no NoPE, no DSA, standard .mlp.experts.<id>.gate_proj/
                # up_proj/down_proj naming, confirmed against the real
                # checkpoint) -- run_glm_block applies unmodified. index_topk
                # is 0 for this checkpoint so the DSA-only code paths inside
                # it are dead code here, not actually exercised.
                from .glm import run_glm_block

                x = run_glm_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset, self._get_experts,
                    mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                )
            elif self.cfg.model_type == "kimi_linear":
                from .kimi_linear import run_kimi_linear_block

                x = run_kimi_linear_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset, self._get_experts,
                    mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                )
            elif self.cfg.model_type in ("qwen3_5_moe", "qwen3_5"):
                from .qwen35 import run_qwen35_block

                x = run_qwen35_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                    self._get_experts, mlp_last_only=last_only,
                    iter_expert_batches=self._iter_expert_batches,
                )
            elif moe:
                x = layer_runner.run_moe_block(
                    x, w, f"model.layers.{i}", self.cfg, kv, i, offset, self._get_experts,
                    mlp_last_only=last_only, rope_freqs=self._rope_freqs,
                    rope_mscale=self._mscale,
                )
            else:
                x = layer_runner.run_block(x, w, f"model.layers.{i}", self.cfg, kv, i, offset,
                                           mlp_last_only=last_only,
                                           rope_freqs=self._rope_freqs,
                                           rope_mscale=self._mscale,
                                           fused_swiglu=self.rc.fused_swiglu)
            mx.eval(x)
            self._layer_transient = max(
                self._layer_transient,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            self.timer.add("layer_compute", time.perf_counter() - t0)
            if tap_layers is not None and i in tap_layers:
                self._tap_hidden[i] = x  # read-only capture; x itself is untouched
            if (self.rc.router_lookahead and moe and self.prefetcher
                    and i + 1 < n and x.shape[1] == 1):
                # F45 — decode only: prefill's multi-position unions flooded the
                # cache and halved hit rates (measured; see benchmark_results)
                self._router_lookahead(x, i + 1)
            del w
        return x

    def forward_tokens(self, tokens: list[int], kv, tap_layers=None) -> mx.array:
        """Feed tokens through the streamed model against an existing KV cache.
        Returns logits (len(tokens), vocab) — one distribution per fed position.
        Building block for speculative verification. `tap_layers`: optional
        iterable of layer indices to capture hidden states from (F62 DSpark
        prep) — populates self._tap_hidden, has NO effect on the returned
        logits/tokens (see tests/test_f62_hidden_taps.py)."""
        x = self._embed(list(tokens))
        x = self._sweep(x, kv, offset=kv.offset, tap_layers=tap_layers)
        self._h_window = x  # trunk states for ALL fed positions (F32: rollback needs mid-window states)
        self._h_last = x[:, -1:, :]  # trunk state for MTP drafting (pre final-norm)
        logits = self._all_logits(x)
        mx.eval(logits)
        return logits

    def forward_tokens_serial_positions(
            self, tokens: list[int], kv, tap_layers=None) -> mx.array:
        """Exact dense verification with one weight sweep for many positions.

        Batched ``(L, hidden)`` GEMMs can choose different reduction kernels
        from ordinary one-token greedy decode and were observed to move Qwen-7B
        tokens during speculative verification. Process positions serially at
        every layer instead, but keep the loop layer-major so a streamed target
        fetches each layer only once for the complete verify window.
        """
        if self.cfg.num_experts or self.cfg.model_type in (
                "glm_moe_dsa", "gpt_oss"):
            raise ValueError(
                "serial-position verification currently supports dense models only")
        if not tokens:
            raise ValueError("serial-position verification needs at least one token")
        if len(tokens) == 1:
            return self.forward_tokens(tokens, kv, tap_layers=tap_layers)

        offset = kv.offset
        self._tap_hidden = {}
        tapset = set(tap_layers) if tap_layers is not None else None
        embedded = self._embed(list(tokens))
        positions = [embedded[:, i:i + 1, :] for i in range(len(tokens))]
        n = self.cfg.num_hidden_layers
        for layer in range(n):
            if self.prefetcher:
                for nxt in range(
                        layer + 1,
                        min(layer + 1 + self.rc.prefetch_depth, n)):
                    self.prefetcher.schedule(
                        self._layer_key(nxt), self._layer_names(nxt))
            t0 = time.perf_counter()
            weights = self.cache.get(
                self._layer_key(layer), self._layer_names(layer))
            self.timer.add("weights_wait", time.perf_counter() - t0)
            if self.governor is not None and self._layer_transient:
                self.governor.reserve(self._layer_transient)

            active_before = mx.get_active_memory()
            mx.reset_peak_memory()
            next_positions = []
            for position, hidden in enumerate(positions):
                hidden = layer_runner.run_block(
                    hidden, weights, f"model.layers.{layer}", self.cfg,
                    kv, layer, offset + position,
                    rope_freqs=self._rope_freqs, rope_mscale=self._mscale,
                    fused_swiglu=self.rc.fused_swiglu,
                )
                next_positions.append(hidden)
            # Keep every block call at the ordinary one-token shape, but use a
            # single layer barrier for the position outputs. The lazy KV chain
            # still orders position N before N+1.
            mx.eval(*next_positions)
            positions = next_positions
            if tapset is not None and layer in tapset:
                # Preserve the same post-layer residual stream exposed by
                # _sweep(tap_layers=...).  Concatenating only after the
                # one-token-shaped layer calls have completed cannot change
                # verifier arithmetic, while DSpark receives one ordinary
                # (1, positions, hidden) context tensor per requested layer.
                self._tap_hidden[layer] = mx.concatenate(positions, axis=1)
            self._layer_transient = max(
                self._layer_transient,
                _resident_adjusted_transient(
                    active_before, mx.get_active_memory(),
                    mx.get_peak_memory()))
            self._note_true_peak()
            del weights

        head = self._lm_head_weight()
        logits = []
        for hidden in positions:
            value = self._final_logits(hidden, head=head)
            mx.eval(value)
            logits.append(value)
        result = mx.stack(logits)
        mx.eval(result)
        self._h_window = mx.concatenate(positions, axis=1)
        self._h_last = positions[-1]
        return result

    def _lazy_resident_decode_step(self, token: mx.array, kv):
        """Build one dense decode step without synchronizing its token.

        The caller can submit the result with ``mx.async_eval`` and construct
        the following step from the lazy token before waiting for the current
        one. This overlaps CPU graph construction with Metal execution, like
        MLX-LM's generation loop, while retaining this runtime's own weights and
        KV implementation. Only the fully-resident dense fast path calls this.
        """
        x = layer_runner.embed(token.reshape(-1), self._embed_weight())
        x = self._sweep(x, kv, offset=kv.offset)
        logits = self._final_logits(x)
        return mx.argmax(logits), logits

    def draft_tokens_resident(self, first_token: int, count: int, kv) -> list[int] | None:
        """Build a short fully-resident draft chain with one synchronization.

        Returns ``None`` when the engine does not satisfy the same residency
        contract as pipelined decode. Draft proposals need not be arithmetic-
        identical to synchronized draft calls—the exact target verifies every
        committed token—but preserving one-token graph shapes generally keeps
        acceptance while removing ``count`` Python/Metal boundaries.
        """
        if count <= 0:
            return []
        if (not self.rc.resident_fast_decode
                or self.cfg.num_experts
                or self._embed_rows is not None
                or not isinstance(kv, KVCache)
                or not all(self.cache.contains(self._layer_key(i))
                           for i in range(self.cfg.num_hidden_layers))):
            return None
        current = mx.array(first_token)
        drafted = []
        for _ in range(count):
            current, _logits = self._lazy_resident_decode_step(current, kv)
            drafted.append(current)
        mx.eval(*drafted)
        return [int(token) for token in drafted]

    def _get_kv_fingerprint(self) -> str:
        """Identity of everything that can change what a cached KV MEANS
        (see kv_store.model_fingerprint's own docstring). Shared by F37's
        disk prompt-KV store and the hot-prompt-kv persistence backing
        (runtime/hot_kv_persist.py) -- both need the SAME identity so a
        model/runtime change invalidates both the same way. Cached: cheap to
        call repeatedly."""
        if not hasattr(self, "_kv_fp"):
            from .kv_store import model_fingerprint

            quant = _quantization_cache_identity(self.rc, self.store)
            arithmetic = (
                f"abs{int(self.rc.mla_absorbed_decode)}"
                f"dead{int(self.rc.final_dead_token_elim)}"
                f"head{int(self.rc.stream_lm_head)}"
                f"tiedhead{int(self.rc.quantize_tied_lm_head)}"
                f"resident{int(self.rc.resident_fast_decode)}"
                f"residentprefill{self.rc.resident_fast_prefill_limit}"
                f"residentmoe{int(self.rc.resident_moe_decode)}"
                f"fswiglu{int(self.rc.fused_swiglu)}"
                f"chunk{self.rc.prefill_chunk_size}"
                f"lastsep{int(self.rc.prefill_last_token_separate)}"
                f"ckpt{self.rc.prefill_checkpoint_every}"
                f"expertbatch{self.rc.expert_fetch_batch}"
                f"decodeexpertbatch{self.rc.decode_expert_fetch_batch}"
                f"steppedkv{self.rc.stepped_kv_threshold}"
                f"toolpic{int(self.rc.tool_pic)}"
                f"sharedpic{int(self.rc.tool_pic_shared_pages)}"
                f"toolpicrepair{self.rc.tool_pic_repair_tokens}"
                f"integrity{self.store.integrity_identity}"
                f"rope{self.rope_cache_identity}"
            )
            self._kv_fp = model_fingerprint(
                self._model_dir, self.new_kv().compressed_mla,
                dsa_elided=self._dsa_elided, quant=quant, arithmetic=arithmetic)
        return self._kv_fp

    def new_kv(self, *, stepped: bool = False) -> KVCache:
        """CANONICAL state factory (2026-07-12 audit): every consumer —
        generate(), speculation, probes — gets the SAME state configuration,
        so measurements always exercise the production path."""
        if self.rc.tool_pic_shared_pages:
            from .kv_cache import PositionFreeKVCache, PositionFreePagePool

            if self._position_free_pool is None:
                self._position_free_pool = PositionFreePagePool(
                    self.cfg.num_hidden_layers,
                    self.cfg.num_key_value_heads,
                    self.cfg.head_dim,
                )
            return PositionFreeKVCache(self._position_free_pool)
        if stepped:
            from .kv_cache import SteppedKVCache

            kv = SteppedKVCache(self.cfg.num_hidden_layers)
        else:
            kv = KVCache(self.cfg.num_hidden_layers)
        if self.rc.mla_compressed_kv and self.cfg.model_type == "glm_moe_dsa":
            kv.compressed_mla = True
            kv.mla_absorbed = self.rc.mla_absorbed_decode
            if not self._dsa_elided:  # F43 bounded mode provably never selects
                from .glm_dsa import DSAState

                kv.dsa = DSAState(self.cfg)
        if self.cfg.model_type in ("kimi_linear", "qwen3_5_moe", "qwen3_5"):
            # KDA's recurrent state is fixed-size and not token-indexed. Exact
            # endpoint/extension retention and durable restore carry this
            # companion cache alongside attention KV; arbitrary prefix trims
            # remain forbidden by the candidate-selection gate in generate().
            from .kda_state import KDAStateCache

            kv.kda_cache = KDAStateCache(self.cfg.num_hidden_layers)
        return kv

    def generate(self, prompt: str, max_tokens: int = 64, on_token=None, stop=None,
                 on_progress=None, sampling: SamplingParams | None = None,
                 constraint=None) -> dict:
        """stop: optional list of strings — generation halts as soon as the
        DECODED output contains any of them, and that string is excluded
        from the returned text (matching the OpenAI API's `stop` semantics).
        Checked against the growing decoded suffix each token, so a stop
        string can span multiple tokens; the token that completes a match
        is never passed to `on_token` (streaming clients never see past the
        stop point)."""
        request_t0 = time.perf_counter()
        sampling = sampling or SamplingParams()
        sampling.seed_rng()
        stop = stop or []
        # Text and vision own different KV implementations. A text request on a
        # vision-capable engine invalidates the retained multimodal prefix before
        # allocating its own state; image embeddings remain separately bounded.
        self._vision_prompt_cache = None
        self._provisional = None  # F55 safety: a crashed spec round must not leave buffering on
        self._resident_fast_decode_sweeps = 0
        self._resident_fast_prefill_sweeps = 0
        self._disable_resident_fast_for_request = False
        self._resident_moe_sweeps = 0
        self._true_peak_metal_bytes = mx.get_active_memory()  # see _note_true_peak
        if self.governor is not None:
            self.governor.reset_request_peak(self._true_peak_metal_bytes)
        self._expert_compute_batches = 0
        self._max_experts_per_compute_batch = 0
        self._adaptive_expert_batch_clamps = 0
        self._min_adaptive_expert_batch = 0
        request_cache_before = _cache_io_snapshot(self)
        # F69 proof-carrying execution telemetry: validation harnesses can assert
        # that the feature under test actually ran instead of inferring it from a
        # config flag (a short prompt with chunk_size=4096 is a no-op).
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        cache_namespace = str(
            getattr(prompt, "cache_namespace", "default") or "default")
        path_stats = {
            "prompt_cache_exact_hit": 0,
            "prompt_cache_prefix_tokens": 0,
            "prompt_cache_source": "cold",
            "hot_prompt_lcp_tokens": 0,
            "hot_prompt_reusable_prefix_tokens": 0,
            "prompt_cache_lookup_s": 0.0,
            "hot_prompt_lookup_s": 0.0,
            "disk_prompt_lookup_s": 0.0,
            "prompt_tokenize_s": 0.0,
            "prompt_snapshot_write_s": 0.0,
            "postgen_snapshot_write_s": 0.0,
            "hot_prompt_kv_persist_write_s": 0.0,
            "hot_prompt_kv_gc_s": 0.0,
            "hot_prompt_kv_gc_removed": 0,
            "hot_prompt_kv_disk_hit": 0,
            "prompt_cache_namespace": cache_namespace,
            "hot_prompt_admission_evicted_slots": 0,
            "hot_prompt_admission_evicted_bytes": 0,
            "hot_prompt_admission_evicted_persisted_slots": 0,
            "hot_prompt_admission_projected_incoming_bytes": 0,
            "hot_prompt_admission_projected_transient_bytes": 0,
            "hot_prompt_admission_runtime_retries": 0,
            "hot_prompt_admission_system_available_bytes": 0,
            "hot_prompt_admission_system_floor_bytes": 0,
            "hot_prompt_admission_governor_reservations": 0,
            "hot_prompt_capacity_evicted_slots": 0,
            "hot_prompt_capacity_evicted_bytes": 0,
            "tool_pic": 0,
            "tool_pic_selected_tokens": 0,
            "tool_pic_reused_tokens": 0,
            "tool_pic_repaired_tokens": 0,
            "tool_pic_prefill_s": 0.0,
            "tool_pic_memory_admitted": 0,
            "tool_pic_projected_bytes": 0,
            "tool_pic_rotated_view_projected_bytes": 0,
            "tool_pic_system_available_bytes": 0,
            "tool_pic_system_floor_bytes": 0,
            "tool_pic_system_memory_admitted": 0,
            "prompt_state_approximate": 0,
            "suffix_decoding_enabled": int(self.rc.suffix_decoding),
            "suffix_decoding_used": 0,
            "suffix_decoding_fallback_reason": (
                "disabled" if not self.rc.suffix_decoding else "pending"),
            "suffix_decoding_proposed": 0,
            "suffix_decoding_accepted": 0,
            "suffix_decoding_target_sweeps": 0,
            "suffix_decoding_cpu_s": 0.0,
            "suffix_decoding_cache_update_cpu_s": 0.0,
            "suffix_decoding_lookup_match_tokens": 0,
            "suffix_decoding_local_rounds": 0,
            "suffix_decoding_global_rounds": 0,
            "suffix_decoding_prompt_approximate": 0,
            "suffix_decoding_single_tenant_required": int(
                self.rc.suffix_decoding),
            "prompt_cache_write_tokens": 0,
            "prompt_cache_min_tokens": self.rc.prompt_kv_min_tokens,
            "rope_profile": self.rope_profile,
            "effective_context_limit": self.effective_max_position_embeddings,
            "sampling_profile": sampling.profile,
            "sampling_temperature": float(sampling.temperature),
            "sampling_top_p": float(sampling.top_p),
            "sampling_top_k": int(sampling.top_k),
            "sampling_seed": sampling.seed,
            "constraint_profile": getattr(constraint, "profile", "none"),
            "prefill_chunks": 0,
            "prefill_checkpoints_saved": 0,
            "paged_kv_chunk_cache_clears": 0,
            "adaptive_kv_spill": 0,
            "adaptive_kv_spill_reason": "",
            "resident_fast_memory_fallback": 0,
            "prompt_snapshots_skipped_oversize": 0,
            "adaptive_chunk_failed": 0,
            "reranked_lm_head": int(self.rc.rerank_lm_head),
            "reranked_lm_head_candidates": (
                self.rc.rerank_lm_head_candidates
                if self.rc.rerank_lm_head else 0
            ),
            "reranked_lm_head_approx_bytes": self._reranked_lm_head_bytes,
            "expert_top_k_by_layer": list(self.cfg.expert_top_k_by_layer),
            "weight_integrity_mode": (
                self.store.integrity_mode
            ),
        }
        tokenize_t0 = time.perf_counter()
        prepared_ids = getattr(prompt, "token_ids", None)
        tokens = (list(prepared_ids) if prepared_ids is not None
                  else self.tokenizer.encode(prompt).ids)
        path_stats["prompt_tokenize_s"] = time.perf_counter() - tokenize_t0
        if (self.effective_max_position_embeddings
                and len(tokens) + max_tokens > self.effective_max_position_embeddings):
            raise ValueError(
                f"prompt({len(tokens)})+max_tokens({max_tokens}) exceeds active "
                f"context limit={self.effective_max_position_embeddings} "
                f"({self.rope_profile})")
        if self.rc.context_bound and len(tokens) + max_tokens > self.rc.context_bound:
            # F43: the bound is a correctness contract (indexer weights were never
            # loaded) — refuse rather than silently switch modes mid-run.
            raise ValueError(
                f"context_bound={self.rc.context_bound} but prompt({len(tokens)})"
                f"+max_tokens({max_tokens}) exceeds it")
        adaptive_spill_mb = max(0, int(
            getattr(self.rc, "adaptive_kv_spill_mb", 0) or 0))
        force_adaptive_paged = bool(
            adaptive_spill_mb and getattr(prompt, "force_paged_kv", False))
        use_stepped_kv = bool(
            self.rc.stepped_kv_threshold
            and len(tokens) + max_tokens > self.rc.stepped_kv_threshold
            and not self.rc.max_kv_mb
            and not force_adaptive_paged
            and not self.rc.tool_pic_shared_pages
            and not (self.rc.mla_compressed_kv
                     and self.cfg.model_type == "glm_moe_dsa")
        )
        path_stats["kv_layout"] = (
            "position_free_shared" if self.rc.tool_pic_shared_pages else
            "paged_adaptive" if force_adaptive_paged else
            "stepped" if use_stepped_kv else
            ("paged" if self.rc.max_kv_mb else "concatenated")
        )
        # F37-hot: transfer ownership of the previous request's in-memory state
        # before allocating anything new.  Clearing BOTH engine references first
        # is important on a 16-GB machine: a divergent prompt must never retain
        # the old full KV while constructing a second full KV.  The local
        # `hot_kv` below is the sole owner until it is either trimmed and reused
        # or released before the cold/disk path allocates new state.
        kv = None
        kv_store = None
        matched = 0
        exact_logits = None
        precomputed_prompt_logits = None  # lossy PIC fills a complete prompt KV
        prompt_state_approximate = False
        reusable_watermark = 0
        persist_parent_chain: tuple[str, ...] = ()  # disk-segment parent for
        # this turn's save, if hot-kv persistence is enabled (see below)
        persist_parent_covered = 0  # exact token count `persist_parent_chain`
        # covers -- always equals `best_matched` when a match wins (true for
        # all three cases: endpoint/branch/repeat), 0 when cold. MUST be
        # passed to save() explicitly rather than re-derived, since a
        # "repeat" parent chain's last segment is not chunk-sized.
        # Recurrent state cannot be trimmed to an arbitrary common prefix.
        # It can, however, be transferred exactly at a complete retained
        # endpoint and extended with a suffix. Candidate selection below
        # limits hybrid models to those two no-trim cases.
        recurrent_exact_only = self.cfg.model_type in (
            "kimi_linear", "qwen3_5_moe", "qwen3_5")
        hot_eligible = (self.rc.hot_prompt_kv and not self.rc.max_kv_mb
                        and not force_adaptive_paged)
        resident_prompt_kv_bytes = self._project_dense_text_kv_bytes(len(tokens))
        configured_paged_mb = int(self.rc.max_kv_mb or 0)
        initial_paged_mb = (
            configured_paged_mb
            or (adaptive_spill_mb if force_adaptive_paged else 0))
        required_total_kv_bytes = (
            min(resident_prompt_kv_bytes, initial_paged_mb * 1_000_000)
            if initial_paged_mb else resident_prompt_kv_bytes)
        admission_done = False

        def record_hot_admission(admission):
            path_stats["hot_prompt_admission_evicted_slots"] += int(
                admission["evicted_slots"])
            path_stats["hot_prompt_admission_evicted_bytes"] += int(
                admission["evicted_bytes"])
            path_stats["hot_prompt_admission_evicted_persisted_slots"] += int(
                admission["evicted_persisted_slots"])
            path_stats["hot_prompt_admission_projected_incoming_bytes"] = int(
                admission["projected_incoming_bytes"])
            path_stats["hot_prompt_admission_projected_transient_bytes"] = int(
                admission.get("projected_transient_bytes", 0))
            path_stats["hot_prompt_admission_system_available_bytes"] = int(
                admission.get("system_available_bytes", 0))
            path_stats["hot_prompt_admission_system_floor_bytes"] = int(
                admission.get("system_available_floor_bytes", 0))
            path_stats["hot_prompt_admission_governor_reservations"] += int(
                admission.get("governor_reservations", 0))

        def admit_hot_kv_growth(keep_kv):
            nonlocal admission_done, force_adaptive_paged
            try:
                admission = self._evict_hot_slots_for_admission(
                    required_total_kv_bytes, keep_kv, cache_namespace,
                    transient_bytes=self._layer_transient)
            except MemoryError as resident_error:
                if not adaptive_spill_mb or force_adaptive_paged:
                    raise
                # Resident prompt KV could not coexist with the learned token
                # transient even after durable inactive branches and reclaimable
                # weights were shed. Run this phase cold with bounded paged KV;
                # prior durable checkpoints remain untouched for a later warm
                # request with more headroom.
                force_adaptive_paged = True
                path_stats["adaptive_kv_spill"] = 1
                path_stats["adaptive_kv_spill_reason"] = "resident_admission"
                path_stats["kv_layout"] = "paged_adaptive"
                if keep_kv is not None:
                    # The caller still owns the resident match. Return the
                    # fallback decision first so it can drop that last strong
                    # reference before we admit the paged replacement.
                    admission_done = False
                    return False
                paged_required = min(
                    resident_prompt_kv_bytes, adaptive_spill_mb * 1_000_000)
                admission = self._evict_hot_slots_for_admission(
                    paged_required, None, cache_namespace,
                    transient_bytes=self._layer_transient)
                print(
                    f"[kv] resident admission fell back to "
                    f"{adaptive_spill_mb}MB paged KV: {resident_error}",
                    flush=True,
                )
            admission_done = True
            record_hot_admission(admission)
            return not force_adaptive_paged

        def reserve_decode_step(active_kv):
            # A whole-token lazy graph exists only for fully resident dense/MoE
            # execution. Ordinary streamed MoE synchronizes and reserves each
            # trunk/expert page independently; reserving its historical
            # *whole-sweep* peak again double-counts sequential cache turnover,
            # repeatedly evicts useful pages, and still cannot protect any one
            # allocation more precisely than the per-page reservations do.
            resident_graph = (
                self._resident_moe_layers is not None
                or (self.rc.resident_fast_decode
                    and not self.cfg.num_experts
                    and all(self.cache.contains(self._layer_key(layer))
                            for layer in range(self.cfg.num_hidden_layers))))
            if (not resident_graph or self._disable_resident_fast_for_request
                    or self.governor is None or not self._token_transient):
                return
            try:
                self.governor.reserve(self._token_transient)
            except MemoryError as resident_error:
                # The learned token transient can become unsafe after prefill
                # even though prompt KV itself fit. Prefer evicting a different,
                # already-durable phase from RAM, then retry with the live
                # governor, instead of failing and leaving the harness to rerun
                # with a cold weight cache.
                try:
                    admission = self._evict_hot_slots_for_admission(
                        self._kv_nbytes(active_kv), active_kv, cache_namespace,
                        transient_bytes=self._token_transient)
                except MemoryError:
                    self._disable_resident_fast_for_request = True
                    self._resident_fast_layers = None
                    self._resident_fast_evictions = -1
                    mx.clear_cache()
                    path_stats["resident_fast_memory_fallback"] = 1
                    print(
                        f"[decode] resident token reservation fell back to "
                        f"streamed layers: {resident_error}",
                        flush=True,
                    )
                    return
                record_hot_admission(admission)
                path_stats["hot_prompt_admission_runtime_retries"] += 1

        if hot_eligible:
            hot_t0 = time.perf_counter()
            # `last_kv` normally aliases the winning slot's KV; clear it even if
            # a diagnostic caller replaced it, so no stale request state survives.
            previous_last_kv = self.last_kv
            if (previous_last_kv is not None
                    and all(slot.kv is not previous_last_kv
                            for slot in self._hot_prompt_slots)):
                self._release_kv(previous_last_kv)
            self.last_kv = None
            self._h_window = None
            self._h_last = None

            # Scan every retained slot (not just one) for the best reuse
            # candidate. Non-winning slots are left completely untouched --
            # this is the actual point of an LRU over a single slot: a request
            # that doesn't match slot A must not evict it, so a LATER request
            # matching slot A still can (e.g. the main conversation thread's
            # slot surviving an interleaved title-generation call's slot).
            best_idx = None
            best_matched = 0
            best_exact_logits = None
            best_reusable_watermark = 0
            best_lcp = 0
            best_needs_trim_to: int | None = None  # None = don't trim, else trim(N)
            best_case = None  # repeat | endpoint | extension | branch -- which arm won,
            # needed only to derive the correct disk-persistence parent chain below

            for idx, slot in enumerate(self._hot_prompt_slots):
                if (getattr(slot, "cache_namespace", "default")
                        != cache_namespace):
                    continue
                if not (isinstance(slot.kv, KVCache) and slot.kv.offset == len(slot.tokens)):
                    continue
                lcp = 0
                for old, new in zip(slot.tokens, tokens):
                    if old != new:
                        break
                    lcp += 1

                if recurrent_exact_only:
                    # A complete endpoint carries exactly the recurrent fold
                    # represented by slot.tokens. Extending that endpoint is
                    # exact; repeating/branching would require rewinding the
                    # fold and is therefore deliberately ineligible.
                    if (len(tokens) == len(slot.tokens)
                            and lcp == len(tokens)
                            and slot.logits is not None):
                        candidate_matched = len(tokens)
                        candidate_exact_logits = slot.logits
                        candidate_watermark = 0
                        candidate_trim_to = None
                        candidate_case = "endpoint"
                    elif (len(tokens) > len(slot.tokens)
                          and lcp == len(slot.tokens)):
                        candidate_matched = len(slot.tokens)
                        candidate_exact_logits = None
                        candidate_watermark = 0
                        candidate_trim_to = None
                        candidate_case = "extension"
                    else:
                        continue
                    if candidate_matched > best_matched:
                        best_idx = idx
                        best_matched = candidate_matched
                        best_exact_logits = candidate_exact_logits
                        best_reusable_watermark = candidate_watermark
                        best_lcp = lcp
                        best_needs_trim_to = candidate_trim_to
                        best_case = candidate_case
                    continue

                if (len(tokens) == slot.prompt_length and lcp >= len(tokens)
                        and slot.prompt_logits is not None):
                    # Normal repeat: the previous request decoded several
                    # tokens past its own prompt endpoint. Trim back to that
                    # endpoint and use its separately retained logits; a
                    # repeat should not pay a suffix sweep merely because the
                    # earlier response generated >1 token.
                    candidate_matched = len(tokens)
                    candidate_exact_logits = slot.prompt_logits
                    candidate_watermark = min(slot.reusable_prefix, candidate_matched)
                    candidate_trim_to = len(tokens) if slot.kv.offset > len(tokens) else None
                    candidate_case = "repeat"
                elif (len(tokens) == len(slot.tokens) and lcp == len(tokens)
                        and slot.logits is not None):
                    # Exact endpoint: retained logits are the distribution
                    # after this complete sequence, so no token needs refeeding.
                    candidate_matched = len(tokens)
                    candidate_exact_logits = slot.logits
                    candidate_watermark = min(slot.reusable_prefix, candidate_matched)
                    candidate_trim_to = None
                    candidate_case = "endpoint"
                elif len(tokens) > len(slot.tokens) and lcp == len(slot.tokens):
                    # Normal next turn/tool loop: the complete retained
                    # post-generation sequence is an exact prefix of the new
                    # prompt. Its KV already ends at this endpoint, so reuse it
                    # whole and prefill only the appended turn. No endpoint
                    # logits are needed because the suffix is non-empty.
                    # Keep the old aligned branch watermark: full-endpoint
                    # reuse is exact, but it does not make decode-shaped tokens
                    # safe arbitrary branch boundaries.
                    candidate_matched = len(slot.tokens)
                    candidate_exact_logits = None
                    candidate_watermark = min(
                        slot.reusable_prefix, candidate_matched)
                    candidate_trim_to = None
                    candidate_case = "extension"
                else:
                    # A branch (including a request that is a strict prefix of
                    # the old sequence) has no logits for its LCP endpoint. Keep
                    # at least one target token for the ordinary prefill tail to
                    # produce those logits, then floor to a fixed boundary. The
                    # stable offset also avoids accumulating a new compiled
                    # shape for every slightly different conversation prefix.
                    # Endpoint logits belong only to the untrimmed sequence, so
                    # this candidate never has exact logits.
                    boundary = self.rc.hot_prompt_kv_chunk_size
                    reusable = min(lcp, max(0, len(tokens) - 1))
                    reusable = (reusable // boundary) * boundary
                    reusable = min(reusable, slot.reusable_prefix)
                    candidate_matched = reusable
                    candidate_exact_logits = None
                    candidate_watermark = reusable
                    candidate_trim_to = reusable if reusable else None
                    candidate_case = "branch"

                if candidate_matched > best_matched:
                    best_idx = idx
                    best_matched = candidate_matched
                    best_exact_logits = candidate_exact_logits
                    best_reusable_watermark = candidate_watermark
                    best_lcp = lcp
                    best_needs_trim_to = candidate_trim_to
                    best_case = candidate_case

            # An exact repeat/endpoint/extension always wins. After an edited
            # catalog, however, an EPIC-style selective sweep can avoid most of
            # the suffix work by relocating unchanged tool KV and recomputing
            # each tool boundary. This is explicitly lossy and enabled only by
            # the fast profile; lossless configurations never enter this block.
            if self.rc.tool_pic and getattr(prompt, "tool_capsules", ()):
                from .tool_capsules import (
                    ToolCapsuleSpan, build_pic_plan,
                    prefill_with_tool_capsules)

                current_capsules = tuple(
                    ToolCapsuleSpan(*value)
                    for value in prompt.tool_capsules)
                baseline_positions = len(tokens) - best_matched
                pic_candidates = []
                for idx, slot in enumerate(self._hot_prompt_slots):
                    if (getattr(slot, "cache_namespace", "default")
                            != cache_namespace):
                        continue
                    if slot.approximate or not slot.tool_capsules:
                        continue
                    lcp = 0
                    for old, new in zip(slot.tokens, tokens):
                        if old != new:
                            break
                        lcp += 1
                    # Let zero-prefill/full-endpoint exact paths below consume
                    # the slot. PIC is only for a genuine edited branch.
                    if ((len(tokens) == slot.prompt_length and lcp >= len(tokens))
                            or (len(tokens) == len(slot.tokens) and lcp == len(tokens))
                            or (len(tokens) > len(slot.tokens)
                                and lcp == len(slot.tokens))):
                        continue
                    boundary = self.rc.hot_prompt_kv_chunk_size
                    safe_prefix = min(lcp, max(0, len(tokens) - 1))
                    safe_prefix = (safe_prefix // boundary) * boundary
                    safe_prefix = min(safe_prefix, slot.reusable_prefix)
                    try:
                        plan = build_pic_plan(
                            tokens, current_capsules, slot.tokens,
                            tuple(ToolCapsuleSpan(*value)
                                  for value in slot.tool_capsules),
                            exact_prefix_tokens=safe_prefix,
                            repair_tokens=self.rc.tool_pic_repair_tokens)
                    except ValueError:
                        continue
                    if plan is None:
                        continue
                    savings = baseline_positions - plan.selected_tokens
                    if (savings < self.rc.tool_pic_min_savings
                            or plan.selected_tokens >= baseline_positions * 0.99):
                        continue
                    pic_candidates.append((
                        plan.selected_tokens, -plan.capsule_tokens_reused,
                        -idx, idx, lcp, plan, slot))

                if pic_candidates:
                    (_selected, _neg_reused, _recency, pic_idx, pic_lcp,
                     plan, slot) = min(pic_candidates)
                    # PIC temporarily owns both the source and destination KV.
                    # Admit that duplication against the governor's live sampled
                    # ceiling rather than a fixed machine-size constant.
                    source_positions = max(1, slot.kv.offset)
                    if getattr(slot.kv, "position_free", False):
                        destination_kv_bytes = int(
                            plan.selected_tokens * slot.kv.pool.bytes_per_page())
                        rotated_view_bytes = int(
                            len(tokens) * slot.kv.pool.bytes_per_page()
                            if (plan.selected_tokens
                                > slot.kv.custom_attention_query_limit
                                or len(tokens) >= slot.kv.rotated_view_min_keys)
                            else 0)
                    else:
                        destination_kv_bytes = int(
                            slot.kv.nbytes() * len(tokens) / source_positions)
                        rotated_view_bytes = 0
                    # The selective attention mask is (selected, full prompt),
                    # and MLP/QKV temporaries scale with selected positions.
                    # Count FP32 mask construction conservatively; reserve()'s
                    # own margin remains additional allocator/system slack.
                    attention_bytes = (
                        plan.selected_tokens * len(tokens) * 4)
                    incoming = (
                        destination_kv_bytes + rotated_view_bytes + attention_bytes
                        + int(self._layer_transient))
                    path_stats["tool_pic_projected_bytes"] = incoming
                    path_stats["tool_pic_rotated_view_projected_bytes"] = (
                        rotated_view_bytes)
                    admitted = True
                    (system_admitted, system_available,
                     system_floor) = _system_allocation_preserves_floor(
                        incoming, self.rc.hot_prompt_kv_min_available_mb)
                    path_stats["tool_pic_system_available_bytes"] = (
                        system_available)
                    path_stats["tool_pic_system_floor_bytes"] = system_floor
                    path_stats["tool_pic_system_memory_admitted"] = int(
                        system_admitted)
                    if self.governor is not None:
                        admitted = (
                            mx.get_active_memory() + incoming + int(0.4e9)
                            <= self.governor.current_ceiling()
                            and system_admitted)
                    else:
                        admitted = system_admitted
                    if admitted:
                        try:
                            if self.governor is not None:
                                self.governor.reserve(incoming)
                            pic_t0 = time.perf_counter()
                            pic_kv, pic_logits = prefill_with_tool_capsules(
                                self, tokens, slot.kv, plan)
                            pic_elapsed = time.perf_counter() - pic_t0
                        except (MemoryError, ValueError) as error:
                            print(
                                f"[tool-pic] fallback to exact prefix: "
                                f"{type(error).__name__}: {error}", flush=True)
                        else:
                            source_slot = self._hot_prompt_slots.pop(pic_idx)
                            kv = pic_kv
                            precomputed_prompt_logits = pic_logits
                            prompt_state_approximate = True
                            matched = plan.exact_prefix_tokens
                            # The assembled generation is a new root. Reusing an
                            # old journal parent would claim exact ancestry for
                            # selectively repaired/relocated state.
                            persist_parent_chain = ()
                            persist_parent_covered = 0
                            reusable_watermark = 0
                            path_stats["prompt_cache_source"] = "tool_pic"
                            path_stats["hot_prompt_lcp_tokens"] = pic_lcp
                            path_stats["tool_pic"] = 1
                            path_stats["tool_pic_selected_tokens"] = (
                                plan.selected_tokens)
                            path_stats["tool_pic_reused_tokens"] = (
                                plan.capsule_tokens_reused)
                            path_stats["tool_pic_repaired_tokens"] = (
                                plan.capsule_tokens_repaired)
                            path_stats["tool_pic_prefill_s"] = pic_elapsed
                            path_stats["tool_pic_memory_admitted"] = 1
                            # Shared destinations retained every reused physical
                            # page before this point. Drop the consumed source's
                            # references now; private dense caches have no release
                            # hook and preserve their historical behavior.
                            if source_slot.kv is not kv:
                                self._release_kv(source_slot.kv)

            if kv is not None:
                pass
            elif best_idx is not None and best_matched > 0:
                slot = self._hot_prompt_slots.pop(best_idx)  # consume: remove from the LRU
                prompt_state_approximate = slot.approximate
                if self._hot_kv_persist is not None:
                    # Deliberately do NOT delete this slot's own checkpoint
                    # here just because a NEW continuation consumes it in
                    # memory. In-memory LRU eviction (hot_prompt_kv_slots)
                    # and disk checkpoint retention (gc()'s own recency-
                    # based cap) are separate concerns: leaving this
                    # checkpoint on disk is what lets a LATER, DIFFERENT
                    # continuation from this same point (a fork -- e.g. a
                    # "regenerate" or an edited earlier message) still find
                    # it directly, instead of only the branch that happened
                    # to consume it in memory first. gc() ages it out by
                    # recency like anything else if nothing references it
                    # again.
                    # Derive the correct parent chain per case:
                    #  - "endpoint"/"extension": reuse the FULL old chain.
                    #  - "branch": old chain truncated to best_matched, which is
                    #    always a hot_prompt_kv_chunk_size multiple by construction
                    #    (the flooring above), so it only ever lands on a full-chunk
                    #    segment boundary.
                    #  - "repeat": best_matched == slot.prompt_length exactly. Since
                    #    save() now writes a SEPARATE prompt-tail segment ending
                    #    exactly at prompt_length (before any generation segment),
                    #    the chain up through prompt_length is always addressable:
                    #    the full-chunk count PLUS one more segment iff the prompt
                    #    had a non-chunk-aligned remainder past reusable_prefix.
                    #    This is what lets N independent continuations of the SAME
                    #    prompt (agentic/cron tasks sharing a preamble) each fork
                    #    their own generation segment off this shared parent,
                    #    rather than "repeat" rebuilding from root every time.
                    if best_case in ("endpoint", "extension"):
                        persist_parent_chain = slot.segment_chain
                    elif best_case == "branch":
                        n = best_matched // self.rc.hot_prompt_kv_chunk_size
                        persist_parent_chain = slot.segment_chain[:n]
                    else:  # "repeat"
                        n = slot.reusable_prefix // self.rc.hot_prompt_kv_chunk_size
                        if slot.prompt_length > slot.reusable_prefix:
                            n += 1  # the prompt-tail segment is also a shared parent
                        persist_parent_chain = slot.segment_chain[:n]
                    persist_parent_covered = best_matched
                if best_needs_trim_to is not None:
                    slot.kv.trim(best_needs_trim_to)
                kv = slot.kv
                matched = best_matched
                exact_logits = best_exact_logits
                reusable_watermark = best_reusable_watermark
                path_stats["hot_prompt_lcp_tokens"] = best_lcp
                path_stats["prompt_cache_source"] = "memory"
            elif (self._hot_kv_persist is not None
                  and self._persisted_kv_restore_allowed()):
                # Total in-memory miss. Before falling all the way back to
                # a cold prefill, check whether the disk segment DAG has
                # something useful -- e.g. more concurrent agentic/cron
                # tasks sharing one preamble than fit in
                # hot_prompt_kv_slots, where an EARLIER task's shared
                # prefix is still sitting on disk even though it was
                # evicted from (or never entered) the in-memory LRU. This
                # deliberately does not compete with an in-memory hit above
                # -- it only fills the gap when memory has nothing at all.
                # Loading the winning disk chain allocates real Metal arrays;
                # release persisted, unmatched resident branches first when
                # the live ceiling cannot hold both states simultaneously.
                if admit_hot_kv_growth(None):
                    disk_match = self._hot_kv_persist.find_best_match(
                        tokens, self.rc.hot_prompt_kv_chunk_size,
                        cache_namespace=cache_namespace)
                    if disk_match is not None:
                        loaded = self._hot_kv_persist.load_matched_chain(
                            disk_match, self.cfg.num_hidden_layers)
                        if loaded is not None:
                            loaded_tokens, loaded_kv, loaded_exact_logits = loaded
                            kv = loaded_kv
                            matched = disk_match["matched"]
                            exact_logits = loaded_exact_logits
                            reusable_watermark = disk_match["watermark"]
                            persist_parent_chain = tuple(
                                disk_match["chain"][: disk_match["n_segments"]])
                            persist_parent_covered = disk_match["matched"]
                            prompt_state_approximate = bool(
                                disk_match.get("approximate", False))
                            path_stats["hot_prompt_lcp_tokens"] = disk_match["lcp"]
                            path_stats["prompt_cache_source"] = "hot_disk"
                            path_stats["hot_prompt_kv_disk_hit"] = 1
            elif self._hot_kv_persist is not None:
                # See _should_defer_persisted_kv_until_bootstrap().  The
                # checkpoint remains untouched and becomes eligible on the
                # next request; this first cold prefill is the safe weight
                # bootstrap sweep.
                path_stats["hot_prompt_kv_bootstrap_deferred"] = 1

            hot_elapsed = max(
                0.0, time.perf_counter() - hot_t0
                - path_stats["tool_pic_prefill_s"])
            path_stats["hot_prompt_lookup_s"] = hot_elapsed
            path_stats["prompt_cache_lookup_s"] += hot_elapsed

        if required_total_kv_bytes and not admission_done:
            # Covers an in-memory match (free unrelated branches before suffix
            # growth) and a cold miss when durable persistence is disabled.
            resident_admitted = admit_hot_kv_growth(kv)
            if not resident_admitted and kv is not None:
                self._release_kv(kv)
                kv = None
                mx.clear_cache()
                matched = 0
                exact_logits = None
                reusable_watermark = 0
                persist_parent_chain = ()
                persist_parent_covered = 0
                path_stats["prompt_cache_source"] = "cold"
                paged_required = min(
                    resident_prompt_kv_bytes, adaptive_spill_mb * 1_000_000)
                paged_admission = self._evict_hot_slots_for_admission(
                    paged_required, None, cache_namespace,
                    transient_bytes=self._layer_transient)
                record_hot_admission(paged_admission)
                admission_done = True

        if kv is None:
            paged_kv_mb = (
                self.rc.max_kv_mb
                or (adaptive_spill_mb if force_adaptive_paged else 0))
            if paged_kv_mb:
                from .kv_paged import PagedKVCache

                kv = PagedKVCache(
                    self.cfg.num_hidden_layers,
                    max_bytes=paged_kv_mb * 1_000_000,
                    spill_dir=self.rc.kv_spill_dir,
                    page_positions=self.rc.kv_page_positions,
                    compress_spill=self.rc.kv_spill_compress,
                )
            else:
                kv = self.new_kv(stepped=use_stepped_kv)
        self.last_kv = kv
        if getattr(kv, "position_free", False):
            # The full request length is known before the first layer runs. Grow
            # the shared physical arrays once here instead of copying every
            # layer's pool at each 256-token prefill boundary.
            kv.reserve_growth(max(
                0, len(tokens) + max_tokens - kv.offset))

        # Prompt KV persistence (F37 v1): model-fingerprinted, exact hits skip
        # the sweep entirely (stored logits), compressed-MLA + DSA state restored.
        # The in-memory cache wins when it supplied state; disk is only consulted
        # after a hot miss, avoiding duplicate old/new KV payloads.
        prompt_kv_eligible = bool(
            self.rc.prompt_kv_dir
            and len(tokens) >= self.rc.prompt_kv_min_tokens
        )
        path_stats["prompt_cache_eligible"] = int(prompt_kv_eligible)
        if prompt_kv_eligible and isinstance(kv, KVCache):
            from .kv_store import PromptKVStore

            dsa_state = getattr(kv, "dsa", None)
            if self._prompt_kv_store is None:
                self._prompt_kv_store = PromptKVStore(
                    self.rc.prompt_kv_dir, self._get_kv_fingerprint(),
                    max_bytes=(self.rc.prompt_kv_max_mb or 10**9 * 999) * 1_000_000,
                    chunk_size=self.rc.prompt_kv_journal_chunk_size,
                    config=self.cfg,
                    require_dsa=dsa_state is not None)
            kv_store = self._prompt_kv_store
            if path_stats["prompt_cache_source"] not in (
                    "memory", "hot_disk", "tool_pic"):
                # A hot_disk match (the segment-DAG fallback above) already
                # supplied real state -- consulting F37's own, unrelated
                # disk store here too would risk silently clobbering it
                # with a WORSE match (or an unrelated one), the same reason
                # an in-memory "memory" hit is excluded.
                disk_t0 = time.perf_counter()
                stored_kv, matched, exact_logits = kv_store.load_longest_prefix(
                    tokens, self.cfg.num_hidden_layers, dsa=dsa_state)
                if stored_kv is not None:
                    if dsa_state is not None:
                        stored_kv.dsa = dsa_state
                    kv = stored_kv
                    self.last_kv = kv
                    path_stats["prompt_cache_source"] = "disk"
                disk_elapsed = time.perf_counter() - disk_t0
                path_stats["disk_prompt_lookup_s"] = disk_elapsed
                path_stats["prompt_cache_lookup_s"] += disk_elapsed
        if use_stepped_kv and isinstance(kv, KVCache):
            from .kv_cache import SteppedKVCache

            kv = SteppedKVCache.from_cache(kv)
            self.last_kv = kv
        path_stats["prompt_cache_prefix_tokens"] = matched
        path_stats["hot_prompt_reusable_prefix_tokens"] = reusable_watermark
        if matched and on_progress is not None:
            on_progress({"phase": "prefill", "completed_tokens": matched,
                         "total_tokens": len(tokens),
                         "cache_source": path_stats["prompt_cache_source"]})
        if path_stats["prompt_cache_source"] in ("memory", "hot_disk"):
            # DSA tensors are prefix state and are intentionally retained, but
            # proof telemetry is per request.  Do not let a hot hit report the
            # previous request's sparse/shared observations as if they reran.
            hot_dsa = getattr(kv, "dsa", None)
            if hot_dsa is not None:
                for key in hot_dsa.stats:
                    hot_dsa.stats[key] = 0

        t0 = time.perf_counter()
        if exact_logits is not None:
            logits = exact_logits  # exact hit: zero sweeps
            path_stats["prompt_cache_exact_hit"] = 1
        elif precomputed_prompt_logits is not None:
            # The selective PIC sweep already produced the complete prompt KV
            # and endpoint distribution during hot-cache planning.
            logits = precomputed_prompt_logits
        else:
            pos = matched
            ckpt = self.rc.prefill_checkpoint_every
            # Memory chunking and persistent checkpoints are deliberately
            # separate. F37 v6 journals only new positions at a checkpoint; a
            # checkpoint-only config still uses its cadence as the compute chunk.
            chunk = self.rc.prefill_chunk_size or (ckpt if kv_store is not None else 0)
            if force_adaptive_paged:
                adaptive_paged_chunk = int(
                    self.rc.adaptive_kv_spill_prefill_chunk_size)
                chunk = min(chunk or adaptive_paged_chunk, adaptive_paged_chunk)
            # F68: learn a safe chunk size online instead of trusting a fixed
            # constant measured on a different model. Intended as a scheduling
            # decision, but chunk shapes can alter kernel/reduction selection, so
            # F33 and greedy-token gates remain required.
            adaptive = None
            adaptive_dynamic_ceiling = False
            if chunk and self.rc.adaptive_chunk_size:
                from .adaptive_chunk import AdaptiveChunkController

                adaptive_dynamic_ceiling = self.rc.adaptive_chunk_safe_bytes == 0
                adaptive_safe_bytes = (
                    self.governor.current_ceiling()
                    if adaptive_dynamic_ceiling
                    else self.rc.adaptive_chunk_safe_bytes
                )
                adaptive = AdaptiveChunkController(
                    safe_bytes=adaptive_safe_bytes, initial_chunk=chunk)
                path_stats["adaptive_chunk_events"] = adaptive.events
                path_stats["adaptive_chunk_dynamic_ceiling"] = int(
                    adaptive_dynamic_ceiling)
                path_stats["adaptive_chunk_safe_bytes_min"] = adaptive_safe_bytes
                path_stats["adaptive_chunk_safe_bytes_max"] = adaptive_safe_bytes
            if chunk:
                prefill_limit = (
                    len(tokens) - 1
                    if self.rc.prefill_last_token_separate and len(tokens) > 1
                    else len(tokens)
                )
                while pos < prefill_limit:
                    chunk_start = pos
                    gov = self.governor
                    if adaptive is not None and adaptive_dynamic_ceiling:
                        adaptive.update_safe_bytes(gov.current_ceiling())
                        path_stats["adaptive_chunk_safe_bytes_min"] = (
                            adaptive.min_safe_bytes)
                        path_stats["adaptive_chunk_safe_bytes_max"] = (
                            adaptive.max_safe_bytes)
                    cur_chunk = adaptive.next_chunk_size() if adaptive is not None else chunk
                    end = min(pos + cur_chunk, prefill_limit)
                    # Land exactly on every requested checkpoint boundary even
                    # when chunk and checkpoint intervals are not multiples.
                    if ckpt and kv_store is not None:
                        next_ckpt = ((pos // ckpt) + 1) * ckpt
                        if next_ckpt < len(tokens):
                            end = min(end, next_ckpt)
                    # Leave the final tail to the ordinary path below so it
                    # produces the hidden/logits needed for greedy decode.
                    if (not self.rc.prefill_last_token_separate
                            and end >= len(tokens)):
                        break
                    active_before = mx.get_active_memory()
                    kv_before = kv.nbytes()
                    shrinks_before = gov.shrinks if gov is not None else 0
                    reservations_before = gov.reservations if gov is not None else 0
                    if adaptive is not None:
                        self._chunk_peak_metal_bytes = active_before
                    xc = self._embed(list(tokens[pos:end]))
                    xc = self._sweep(xc, kv, offset=pos,
                                     final_mlp_last_only=self.rc.final_dead_token_elim)
                    path_stats["prefill_chunks"] += 1
                    if adaptive is not None:
                        gov_event = gov is not None and (
                            gov.shrinks > shrinks_before or gov.reservations > reservations_before)
                        adaptive.observe(
                            chunk_size=end - pos, peak=self._chunk_peak_metal_bytes,
                            active_before=active_before, kv_before=kv_before,
                            governor_event=gov_event)
                        if adaptive.failed:
                            # Fail closed: keep asking the frozen controller for
                            # its already-halved size. The old behavior set it to
                            # None and silently restored the original unsafe fixed
                            # chunk, exactly undoing three emergency reductions.
                            path_stats["adaptive_chunk_failed"] = 1
                            if adaptive.unsafe_at_minimum:
                                raise RuntimeError(
                                    "adaptive prefill cannot stay under the memory "
                                    "budget even at chunk size 1"
                                )
                    if ckpt and kv_store is not None and end % ckpt == 0:
                        ck_logits = self._final_logits(xc)
                        mx.eval(ck_logits)
                        write_t0 = time.perf_counter()
                        saved = kv_store.save(tokens[:end], kv, ck_logits,
                                              dsa=getattr(kv, "dsa", None))
                        path_stats["prompt_snapshot_write_s"] += (
                            time.perf_counter() - write_t0)
                        path_stats["prefill_checkpoints_saved"] += int(saved)
                        if saved:
                            path_stats["prompt_cache_write_tokens"] = max(
                                path_stats["prompt_cache_write_tokens"], end)
                        path_stats["prompt_snapshots_skipped_oversize"] += int(not saved)
                    pos = end
                    if (hot_eligible
                            and chunk_start == reusable_watermark
                            and end - chunk_start == self.rc.hot_prompt_kv_chunk_size):
                        reusable_watermark = end
                        path_stats["hot_prompt_reusable_prefix_tokens"] = reusable_watermark
                    if self.rc.max_kv_mb or force_adaptive_paged:
                        # Page reload + concatenation temporaries are dead once
                        # the full layer sweep is materialized. With many small
                        # progressive chunks, leaving `xc` and MLX's buffer cache
                        # alive until the next iteration accumulated enough
                        # reclaimable memory to push system-available below 4 GB.
                        del xc
                        mx.clear_cache()
                        path_stats["paged_kv_chunk_cache_clears"] += 1
                    if on_progress is not None:
                        try:
                            on_progress({
                                "phase": "prefill",
                                "completed_tokens": pos,
                                "total_tokens": len(tokens),
                                "cache_source": path_stats["prompt_cache_source"],
                            })
                        except Exception:
                            # A streaming client can disconnect after observing a
                            # progress boundary but before cold prefill finishes.
                            # Preserve the complete exact chunks already built so
                            # a retry resumes from this boundary instead of paying
                            # for them again.  Durable hot-KV has its own atomic
                            # segment protocol and is intentionally excluded from
                            # this in-memory-only recovery path.
                            self._retain_interrupted_prefill(
                                tokens, kv, reusable_watermark,
                                getattr(prompt, "tool_capsules", ()),
                                cache_namespace)
                            raise
            x = self._embed(list(tokens[pos:]))
            # F36 applies here because generate() consumes only the last position;
            # forward_tokens (speculative verify) must NOT use it — it needs
            # logits and trunk states at every fed position.
            x = self._sweep(x, kv, offset=pos,
                            final_mlp_last_only=self.rc.final_dead_token_elim)
            if (hot_eligible
                    and pos == reusable_watermark
                    and len(tokens) - pos == self.rc.hot_prompt_kv_chunk_size):
                reusable_watermark = len(tokens)
                path_stats["hot_prompt_reusable_prefix_tokens"] = reusable_watermark
            self._h_window = x
            self._h_last = x[:, -1:, :]
            logits = self._final_logits(x)
        sampled_logits = constraint.mask_logits(logits) if constraint is not None else logits
        next_tok = sample(sampled_logits, sampling)
        if constraint is not None:
            constraint.accept_token(next_tok)
        grammar_completed = bool(
            constraint is not None and constraint.completed)
        prompt_endpoint_logits = logits
        prefill_cache_after = _cache_io_snapshot(self)
        prefill_s = (time.perf_counter() - t0
                     + path_stats["tool_pic_prefill_s"])
        if (kv_store is not None and exact_logits is None
                and precomputed_prompt_logits is None and matched < len(tokens)):
            write_t0 = time.perf_counter()
            saved = kv_store.save(tokens, kv, logits, dsa=getattr(kv, "dsa", None))
            path_stats["prompt_snapshot_write_s"] += time.perf_counter() - write_t0
            path_stats["prompt_snapshots_skipped_oversize"] += int(not saved)
            if saved:
                path_stats["prompt_cache_write_tokens"] = max(
                    path_stats["prompt_cache_write_tokens"], len(tokens))

        generated = [next_tok]
        stop_text = None
        matched_stop_sequence = None
        stream_decoder = None
        if on_token:
            from .incremental_decode import IncrementalDetokenizer

            stream_decoder = IncrementalDetokenizer(self.tokenizer, stop)

        def _stop_match(text: str):
            matches = [(text.find(value), index, value)
                       for index, value in enumerate(stop)
                       if value and text.find(value) != -1]
            return min(matches) if matches else None

        if stop:
            decoded = self.tokenizer.decode(generated)
            match = _stop_match(decoded)
            if match is not None:
                cut, _order, matched_stop_sequence = match
                stop_text = decoded[:cut]
        first_token_s = time.perf_counter() - request_t0
        if stop_text is None and stream_decoder is not None:
            delta = stream_decoder.push(generated)
            if delta:
                on_token(delta)

        if (getattr(kv, "position_free", False)
                and kv.offset < kv.rotated_view_min_keys):
            # Wide prefill temporarily retained a direct SDPA view so later
            # chunks did not repeatedly gather the growing page table. For a
            # short decode the fused page kernel is faster, so shed that view at
            # the phase boundary; long contexts keep it for MLX SDPA.
            kv.drop_rotated_view()
        tok_times = []
        pipelined_decode_steps = 0
        remaining_decode = max_tokens - 1
        suffix_state = None
        suffix_cache = self._suffix_cache
        if suffix_cache is not None:
            from .suffix_decoding import fallback_reason

            suffix_reason = fallback_reason(
                self, kv, sampling, constraint,
                terminal=(
                    stop_text is not None
                    or remaining_decode <= 0
                    or next_tok in self.cfg.eos_token_ids
                ),
            )
            suffix_history_eligible = fallback_reason(
                self, kv, sampling, constraint, terminal=False) is None
        else:
            suffix_reason = "disabled"
            suffix_history_eligible = False
        path_stats["prompt_state_approximate"] = int(
            prompt_state_approximate)
        path_stats["suffix_decoding_prompt_approximate"] = int(
            prompt_state_approximate)
        path_stats["suffix_decoding_fallback_reason"] = (
            suffix_reason or "")
        resident_decode_memory_safe = True
        if (self.rc.resident_fast_decode and self.governor is not None
                and self._token_transient):
            resident_decode_memory_safe = (
                mx.get_active_memory() + self._token_transient + int(0.4e9)
                <= self.governor.current_ceiling())
        if not resident_decode_memory_safe:
            # Full-stack lazy decode is a throughput optimization, not a
            # correctness requirement. Under pressure, stream/evaluate one
            # layer at a time; that path performs its own smaller per-layer
            # reservations and avoids failing a response over a stale 1GB
            # resident-token high-water estimate.
            self._disable_resident_fast_for_request = True
            self._resident_fast_layers = None
            self._resident_fast_evictions = -1
            mx.clear_cache()
            path_stats["resident_fast_memory_fallback"] = 1
            print(
                f"[decode] streaming layers under live pressure instead of "
                f"reserving {self._token_transient / 1e9:.2f}GB resident "
                f"token transient",
                flush=True,
            )
        dense_pipeline_ready = (
            self.rc.resident_fast_decode
            and not self._disable_resident_fast_for_request
            and not self.cfg.num_experts
            and all(self.cache.contains(self._layer_key(i))
                    for i in range(self.cfg.num_hidden_layers))
        )
        moe_pipeline_ready = (
            self.rc.resident_moe_decode
            and self._resident_moe_layers is not None
        )
        can_pipeline = (
            sampling.is_greedy
            and constraint is None
            and stop_text is None
            and remaining_decode > 0
            and next_tok not in self.cfg.eos_token_ids
            and (dense_pipeline_ready or moe_pipeline_ready)
            and self._embed_rows is None
            and isinstance(kv, KVCache)
        )
        if suffix_reason is None:
            from .suffix_decoding import run_shared_prefill_suffix_decode

            suffix_state = suffix_cache.begin_request(tokens)
            suffix_state.append_committed(generated)
            suffix_result = run_shared_prefill_suffix_decode(
                self,
                suffix_cache,
                suffix_state,
                prompt_tokens=tokens,
                generated=generated,
                kv=kv,
                logits=logits,
                max_tokens=max_tokens,
                stop=stop,
                stream_decoder=stream_decoder,
                on_token=on_token,
                stop_match=_stop_match,
            )
            logits = suffix_result.logits
            tok_times = suffix_result.token_times
            stop_text = suffix_result.stop_text
            matched_stop_sequence = suffix_result.stop_sequence
            suffix_stats = suffix_result.stats
            path_stats["suffix_decoding_used"] = 1
            path_stats["suffix_decoding_proposed"] = suffix_stats.proposed
            path_stats["suffix_decoding_accepted"] = suffix_stats.accepted
            path_stats["suffix_decoding_target_sweeps"] = suffix_stats.sweeps
            path_stats["suffix_decoding_cpu_s"] = suffix_stats.cpu_s
            path_stats["suffix_decoding_lookup_match_tokens"] = (
                suffix_stats.lookup_match_tokens)
            path_stats["suffix_decoding_local_rounds"] = (
                suffix_stats.local_rounds)
            path_stats["suffix_decoding_global_rounds"] = (
                suffix_stats.global_rounds)
        elif can_pipeline:
            # Submit token N+1 before waiting for token N. The lazy token itself
            # is a valid gather index for the following graph, so CPU graph
            # construction overlaps Metal execution instead of leaving a bubble
            # at every greedy boundary. This is the largest measured dense Q4
            # side-quest win after removing per-layer synchronization.
            decode_t0 = time.perf_counter()
            boundary = mx.get_active_memory()
            reserve_decode_step(kv)
            mx.reset_peak_memory()
            current_token, current_logits = self._lazy_resident_decode_step(
                mx.array(next_tok), kv)
            mx.async_eval(current_token, current_logits)

            for index in range(remaining_decode):
                schedule_future = index + 1 < remaining_decode
                if schedule_future:
                    future_token, future_logits = self._lazy_resident_decode_step(
                        current_token, kv)
                    mx.async_eval(future_token, future_logits)

                next_tok = int(current_token)
                logits = current_logits
                generated.append(next_tok)
                pipelined_decode_steps += 1

                if stop:
                    decoded = self.tokenizer.decode(generated)
                    match = _stop_match(decoded)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                if stop_text is None and stream_decoder is not None:
                    delta = stream_decoder.push(generated)
                    if delta:
                        on_token(delta)

                terminated = (
                    stop_text is not None or next_tok in self.cfg.eos_token_ids)
                if terminated:
                    if schedule_future:
                        # The look-ahead already fed the terminating token. The
                        # runtime's retained-KV contract excludes the final,
                        # unconsumed output token, so materialize then roll back
                        # that one speculative position before persistence/reuse.
                        mx.eval(future_token, future_logits)
                        kv.trim(len(tokens) + len(generated) - 1)
                    break
                if not schedule_future:
                    break
                current_token, current_logits = future_token, future_logits

            mx.eval(logits)
            self._token_transient = max(
                self._token_transient,
                _resident_adjusted_transient(
                    boundary, mx.get_active_memory(), mx.get_peak_memory()))
            self._note_true_peak()
            decode_elapsed = time.perf_counter() - decode_t0
            tok_times = [decode_elapsed / pipelined_decode_steps] * pipelined_decode_steps
        elif stop_text is None:
            for _ in range(max_tokens - 1):
                if grammar_completed or next_tok in self.cfg.eos_token_ids:
                    break
                t0 = time.perf_counter()
                # F42: the real overshoot is the WHOLE-TOKEN transient (measured at
                # the greedy() sync point, not inside any one layer) — learn it and
                # reserve before the next token so the ceiling is never crossed.
                boundary = mx.get_active_memory()
                reserve_decode_step(kv)
                mx.reset_peak_memory()
                x = self._embed([next_tok])
                x = self._sweep(x, kv, offset=kv.offset)
                logits = self._final_logits(x)
                sampled_logits = (
                    constraint.mask_logits(logits)
                    if constraint is not None else logits)
                next_tok = sample(sampled_logits, sampling)
                if constraint is not None:
                    constraint.accept_token(next_tok)
                    grammar_completed = bool(constraint.completed)
                self._token_transient = max(
                    self._token_transient,
                    _resident_adjusted_transient(
                        boundary, mx.get_active_memory(),
                        mx.get_peak_memory()))
                self._note_true_peak()
                tok_times.append(time.perf_counter() - t0)
                generated.append(next_tok)
                if stop:
                    decoded = self.tokenizer.decode(generated)
                    match = _stop_match(decoded)
                    if match is not None:
                        cut, _order, matched_stop_sequence = match
                        stop_text = decoded[:cut]
                        break  # never streamed: on_token withheld for the matching token
                if stream_decoder is not None:
                    delta = stream_decoder.push(generated)
                    if delta:
                        on_token(delta)

        final_text = stop_text if stop_text is not None else self.tokenizer.decode(generated)
        termination_reason = (
            "stop_sequence" if stop_text is not None else
            "grammar" if grammar_completed else
            "eos" if generated[-1] in self.cfg.eos_token_ids else
            "length"
        )
        if stream_decoder is not None:
            delta = stream_decoder.finish(generated, final_text=final_text)
            if delta:
                on_token(delta)

        if suffix_history_eligible:
            suffix_update_t0 = time.process_time()
            suffix_cache.add_output(generated)
            path_stats["suffix_decoding_cache_update_cpu_s"] = (
                time.process_time() - suffix_update_t0)
        if suffix_cache is not None:
            path_stats.update(suffix_cache.telemetry(suffix_state))

        if kv_store is not None and len(generated) > 1:
            # Multi-turn prefix reuse: also persist the POST-GENERATION state.
            # The next chat request's prompt = this prompt + this response +
            # a new turn, so it now prefix-matches through the response
            # instead of only through the previous prompt. The KV holds
            # prompt + generated[:-1] (the final token is never fed) and
            # `logits` is exactly the distribution that predicted the final
            # token — correct exact-hit semantics. Oversized snapshots are
            # rejected before writing; ordinary entries are LRU-budgeted.
            write_t0 = time.perf_counter()
            postgen_tokens = tokens + generated[:-1]
            saved = kv_store.save(postgen_tokens, kv, logits,
                                  dsa=getattr(kv, "dsa", None))
            path_stats["postgen_snapshot_write_s"] += time.perf_counter() - write_t0
            path_stats["prompt_snapshots_skipped_oversize"] += int(not saved)
            if saved:
                path_stats["prompt_cache_write_tokens"] = max(
                    path_stats["prompt_cache_write_tokens"], len(postgen_tokens))

        # F69: proof-carrying telemetry -- expose whether DSA's sparse/shared
        # paths actually ran, not just whether they were configured to. A
        # caller asserting e.g. `dsa_sparse_selects>0` catches a silently
        # no-op long-context run (short prompt, wrong bound, etc.) the same
        # way this exact gap was found in this session's own real-GLM script.
        dsa_state = getattr(kv, "dsa", None)
        if dsa_state is not None:
            path_stats["dsa_observations"] = dsa_state.stats["observations"]
            path_stats["dsa_sparse_selects"] = dsa_state.stats["sparse_selects"]
            path_stats["dsa_shared_reuses"] = dsa_state.stats["shared_reuses"]
        path_stats["expert_compute_batches"] = self._expert_compute_batches
        path_stats["max_experts_per_compute_batch"] = self._max_experts_per_compute_batch
        path_stats["adaptive_expert_batch_clamps"] = self._adaptive_expert_batch_clamps
        path_stats["min_adaptive_expert_batch"] = self._min_adaptive_expert_batch
        path_stats["expert_resident_page_bytes_estimate"] = self._expert_page_bytes
        path_stats["expert_storage_page_bytes_estimate"] = (
            self._expert_storage_page_bytes)
        path_stats["expert_fetch_page_bytes_estimate"] = self._expert_fetch_page_bytes
        path_stats["resident_fast_decode_sweeps"] = self._resident_fast_decode_sweeps
        path_stats["resident_fast_prefill_sweeps"] = self._resident_fast_prefill_sweeps
        path_stats["resident_moe_sweeps"] = self._resident_moe_sweeps
        path_stats["resident_moe_bytes"] = self._resident_moe_bytes
        path_stats["resident_attention_mode"] = self.rc.resident_attention_mode
        path_stats["resident_attention_bytes"] = self._resident_attention_bytes
        path_stats["resident_pipelined_decode_steps"] = pipelined_decode_steps
        path_stats["fused_swiglu"] = int(self.rc.fused_swiglu)
        position_free_pool = self._position_free_pool
        path_stats["position_free_pool_live_pages"] = (
            position_free_pool.live_pages if position_free_pool is not None else 0)
        path_stats["position_free_pool_live_bytes"] = (
            position_free_pool.live_nbytes()
            if position_free_pool is not None else 0)
        path_stats["position_free_pool_allocated_bytes"] = (
            position_free_pool.allocated_nbytes()
            if position_free_pool is not None else 0)
        path_stats["position_free_rotated_view_bytes"] = int(
            kv.rotated_view_nbytes()
            if getattr(kv, "position_free", False) else 0)
        paged_stats = getattr(kv, "stats", None)
        path_stats["paged_kv_spills"] = int(
            getattr(paged_stats, "spills", 0) or 0)
        path_stats["paged_kv_reloads"] = int(
            getattr(paged_stats, "reloads", 0) or 0)
        path_stats["paged_kv_spill_seconds"] = float(
            getattr(paged_stats, "spill_s", 0.0) or 0.0)
        path_stats["paged_kv_reload_seconds"] = float(
            getattr(paged_stats, "reload_s", 0.0) or 0.0)
        if getattr(kv, "position_free", False):
            # The view exists only to make this request's long decode use MLX's
            # fast pre-rotated SDPA. The retained hot slot owns shared physical
            # pages only, so subsequent edited branches do not duplicate KV.
            kv.drop_rotated_view()

        if (hot_eligible and isinstance(kv, KVCache)
                and len(tokens) >= self.rc.hot_prompt_kv_min_tokens):
            # At this point the KV contains exactly the prompt plus every
            # generated token that was fed back (`generated[:-1]`).  `logits`
            # is the distribution that predicted the un-fed final token, so the
            # tuple is a valid exact endpoint for the next request as well as a
            # branchable prefix.  Retain the SAME object -- never a cloned KV.
            mx.eval(logits)
            mx.eval(prompt_endpoint_logits)
            recurrent_state = getattr(kv, "kda_cache", None)
            if recurrent_state is not None:
                recurrent_state.synchronize()
            full_tokens = tuple(tokens + generated[:-1])
            segment_chain: tuple[str, ...] = ()
            if self._hot_kv_persist is not None:
                persist_t0 = time.perf_counter()
                segment_chain = self._hot_kv_persist.save(
                    parent_chain=persist_parent_chain,
                    parent_covered=persist_parent_covered,
                    tokens=full_tokens,
                    kv=kv,
                    logits=logits,
                    prompt_logits=prompt_endpoint_logits,
                    prompt_length=len(tokens),
                    reusable_prefix=reusable_watermark,
                    approximate=prompt_state_approximate,
                    tool_capsules=tuple(getattr(prompt, "tool_capsules", ())),
                    cache_namespace=cache_namespace,
                )
                path_stats["hot_prompt_kv_persist_write_s"] = (
                    time.perf_counter() - persist_t0)
            new_slot = _HotPromptSlot(
                tokens=full_tokens,
                kv=kv,
                logits=logits,
                prompt_length=len(tokens),
                prompt_logits=prompt_endpoint_logits,
                reusable_prefix=reusable_watermark,
                approximate=prompt_state_approximate,
                tool_capsules=tuple(getattr(prompt, "tool_capsules", ())),
                segment_chain=segment_chain,
                cache_namespace=cache_namespace,
            )
            capacity_count, capacity_bytes = self._append_hot_prompt_slot(new_slot)
            path_stats["hot_prompt_capacity_evicted_slots"] = capacity_count
            path_stats["hot_prompt_capacity_evicted_bytes"] = capacity_bytes
            # Capacity eviction frees only the in-memory copy. Its disk
            # checkpoint is governed by the separate durable recency budget.
            if self._hot_kv_persist is not None:
                gc_t0 = time.perf_counter()
                path_stats["hot_prompt_kv_gc_removed"] = self._hot_kv_persist.gc()
                path_stats["hot_prompt_kv_gc_s"] = time.perf_counter() - gc_t0

        if self.governor is not None:
            self._true_peak_metal_bytes = max(
                self._true_peak_metal_bytes,
                self.governor.request_peak(),
                mx.get_active_memory(),
            )
        request_cache_after = _cache_io_snapshot(self)
        _record_cache_io_delta(
            self, request_cache_before, path_stats, after=request_cache_after)
        _record_cache_io_delta(
            self, request_cache_before, path_stats, prefix="prefill_",
            after=prefill_cache_after)
        _record_cache_io_delta(
            self, prefill_cache_after, path_stats, prefix="decode_",
            after=request_cache_after)
        result = {
            "text": final_text,
            "tokens": generated,
            "prefill_s": prefill_s,
            "decode_s": sum(tok_times),
            "first_token_s": first_token_s,
            "total_s": time.perf_counter() - request_t0,
            "tok_per_s": len(tok_times) / sum(tok_times) if tok_times else 0.0,
            "kv_bytes": kv.nbytes(),
            "kv_positions": kv.offset,
            "stopped": stop_text is not None,
            "stop_sequence": matched_stop_sequence,
            "termination_reason": termination_reason,
            "true_peak_metal_bytes": self._true_peak_metal_bytes,
            "path_stats": path_stats,
            # Cache/HTTP telemetry needs the encoded prompt count. `tokens` was
            # already computed above, so exposing it costs no second tokenize.
            "prompt_tokens": len(tokens),
        }
        if ((self.rc.release_paged_kv_after_generate and self.rc.max_kv_mb)
                or force_adaptive_paged):
            self.last_kv = None
            self._release_kv(kv)
            mx.clear_cache()
        self._completed_generations += 1
        return result

    def report(self) -> str:
        lines = [
            self.cache.stats.summary(),
        ]
        if getattr(self, "last_kv", None) is not None and hasattr(self.last_kv, "stats"):
            lines.append(self.last_kv.stats.summary()
                         + f" | kv resident {self.last_kv.nbytes() / 1e6:.1f}MB")
        lines += [
            f"cache resident: {self.cache.total_bytes / 1e6:.0f}MB "
            f"(budget {self.cache.max_bytes / 1e6:.0f}MB), keys={self.cache.resident_keys}",
            telemetry.fmt_mem(),
            self.timer.summary(),
        ]
        if self.prefetcher:
            lines.insert(1, self.prefetcher.summary())
        if self.cache.warm is not None:
            lines.insert(1, self.cache.warm.summary())
        if self.governor is not None and (self.governor.shrinks or self.governor.restores):
            lines.insert(1, self.governor.summary())
        if self.cfg.num_experts and (self.expert_hits + self.expert_misses):
            total = self.expert_hits + self.expert_misses
            uniq = len(self.expert_usage)
            possible = self.cfg.num_hidden_layers * self.cfg.num_experts
            top = sorted(self.expert_usage.items(), key=lambda kv: -kv[1])[:5]
            lines.insert(1, (
                f"experts: {total} activations, cache hit {self.expert_hits / total * 100:.0f}%, "
                f"{uniq}/{possible} unique experts touched, "
                f"hottest {[f'L{l}E{e}x{c}' for (l, e), c in top]}"
            ))
            if self.predictor is not None:
                lines.insert(2, self.predictor.summary())
        return "\n".join(lines)

    def release_request_state(self):
        """Release the sole retained request state before another owner runs."""
        from .request_state import release_generation_state

        release_generation_state(self)
        self._vision_prompt_cache = None
        vision_embeddings = getattr(self, "_vision_embedding_cache", None)
        if vision_embeddings is not None:
            vision_embeddings.clear()

    def discard_failed_request_state(self):
        """Drop only state owned by the request that just failed.

        A long-prefill MemoryError used to leave ``last_kv`` strongly referenced.
        The harness then retried immediately against an allocator still holding
        the failed request's multi-GiB KV, turning one safe refusal into a rapid
        refusal loop.  Preserve unrelated hot slots, but remove/release the slot
        that aliases the failed request (if any) and its diagnostic ``last_kv``.
        """
        self._disable_resident_fast_for_request = False
        failed = self.last_kv
        if failed is None:
            return
        self._hot_prompt_slots = [
            slot for slot in self._hot_prompt_slots if slot.kv is not failed
        ]
        self._release_kv(failed)
        self.last_kv = None
        self._h_window = None
        self._h_last = None
        self._provisional = None

    def _retain_interrupted_prefill(
            self, tokens, kv, reusable_prefix: int, tool_capsules=(),
            cache_namespace: str = "default") -> bool:
        """Retain a complete chunk boundary after an SSE/client interruption.

        The slot deliberately has no endpoint logits: on retry it is only an
        exact prefix of the full prompt, so the ordinary extension path resumes
        at ``kv.offset`` and produces the final endpoint logits normally.
        """
        covered = int(getattr(kv, "offset", 0) or 0)
        if (not self.rc.hot_prompt_kv or self.rc.max_kv_mb
                or self._hot_kv_persist is not None
                or not isinstance(kv, KVCache)
                or covered <= 0 or covered > len(tokens)
                or covered < self.rc.hot_prompt_kv_min_tokens
                or reusable_prefix != covered):
            return False
        retained_capsules = tuple(
            capsule for capsule in tool_capsules
            if len(capsule) >= 3 and int(capsule[2]) <= covered
        )
        self._hot_prompt_slots = [
            slot for slot in self._hot_prompt_slots if slot.kv is not kv
        ]
        self._append_hot_prompt_slot(_HotPromptSlot(
            tokens=tuple(tokens[:covered]),
            kv=kv,
            logits=None,
            prompt_length=covered,
            prompt_logits=None,
            reusable_prefix=covered,
            approximate=False,
            tool_capsules=retained_capsules,
            segment_chain=(),
            cache_namespace=str(cache_namespace or "default"),
        ))
        self.last_kv = kv
        self._h_window = None
        self._h_last = None
        return True

    @staticmethod
    def _release_kv(kv):
        release = getattr(kv, "release", None)
        if release is not None:
            release()

    def close(self):
        self.release_request_state()
        if self._position_free_pool is not None:
            self._position_free_pool.close()
            self._position_free_pool = None
        self._prompt_kv_store = None
        if self.governor is not None:
            self.governor.close()
        if self.prefetcher:
            self.prefetcher.close()
        if self.predictor is not None:
            self.predictor.save()
        if self._streamed_lm_head is not None:
            self._streamed_lm_head.close()
        if self._embed_rows is not None:
            self._embed_rows.close()
